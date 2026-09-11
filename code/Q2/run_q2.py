"""Q2 phase-1 / full-year runner: forecast, LP, causal SOC backtest."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    ADAPTIVE_DIR,
    ATTACHMENT1_XLSX,
    ATTACHMENT2_XLSX,
    DT_HOURS,
    E0_FEB1_KWH,
    E_MAX_KWH,
    E_MIN_KWH,
    ETA_CHARGE,
    ETA_DISCHARGE,
    FROZEN_C_Q82_FULL_YEAR_COST,
    FULL_DIR,
    MODEL_NAMES,
    MPC_STRIDE,
    MPC_STRIDE_FALLBACK,
    OFFICIAL_END,
    OFFICIAL_START,
    OOS_START,
    OPT_DIR,
    P_MAX_KWH,
    PHASE1_DIR,
    PHASE1_END,
    PHASE1_MPC_BUDGET_S,
    PV_SOURCE_V2,
    Q_LADDER,
    REPO_ROOT,
    RESULT2_TEMPLATE_XLSX,
    RESULT2_XLSX,
    RESULT_DIR,
    SIM_START,
    TERMINAL_LAMBDA,
    TERMINAL_TARGET_KWH,
    TUNE_END,
    TUNE_INNER_END,
    TUNE_SELECT_START,
    XGB_PARAMS,
)
from adaptive import (  # noqa: E402
    FEATURE_NAMES,
    dumps,
    flatten_for_table,
    official_summary,
    run_adaptive,
    run_fixed,
    summarise,
)
from bank import check_length, ladder_banks, load_point_bank, point_residual_history  # noqa: E402
from forecast import apply_conservative_bias, precompute_panel, predict_day  # noqa: E402
from leakage import run_leakage_suite  # noqa: E402
from load_data import audit_data, load_prices, load_year_actuals, write_clean  # noqa: E402
from model_lp import solve_day_lp, validate_plan  # noqa: E402
from export_results import export_result2, export_specified_day_tables  # noqa: E402
from simulate import simulate_day, simulate_day_mpc, validate_actual  # noqa: E402


def load_aligned_year(prices: pd.DataFrame) -> dict:
    return load_year_actuals(
        jan1_load=float(prices["typical_load_kw"].iloc[0]),
        jan1_pv=float(prices["typical_pv_kw"].iloc[0]),
    )


def copy_raw_inputs() -> None:
    src_att = REPO_ROOT / "problem_files" / "附件"
    ATTACHMENT1_XLSX.parent.mkdir(parents=True, exist_ok=True)
    if ATTACHMENT1_XLSX.exists() and ATTACHMENT2_XLSX.exists() and RESULT2_TEMPLATE_XLSX.exists():
        return
    shutil.copy2(src_att / "附件1.xlsx", ATTACHMENT1_XLSX)
    shutil.copy2(src_att / "附件2.xlsx", ATTACHMENT2_XLSX)
    shutil.copy2(src_att / "附件5" / "result2.xlsx", RESULT2_TEMPLATE_XLSX)


def mae_rmse(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    err = pred - actual
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def collect_forecasts(
    model_name: str,
    prices: pd.DataFrame,
    year: dict,
    start_idx: int,
    end_idx: int,
    load_panel: pd.DataFrame,
    pv_panel: pd.DataFrame,
    cache: dict | None = None,
    pv_source: str = "model",
) -> tuple[list[dict], float]:
    dates = year["dates"]
    fallback_load = prices["typical_load_kw"].to_numpy(dtype=float)
    fallback_pv = prices["typical_pv_kw"].to_numpy(dtype=float)
    cache = {} if cache is None else cache
    out = []
    t0 = time.perf_counter()
    for day in range(start_idx, end_idx + 1):
        pred = predict_day(
            year["load_kw"],
            year["pv_kw"],
            day,
            dates,
            fallback_load,
            fallback_pv,
            model_name,
            cache,
            load_panel=load_panel,
            pv_panel=pv_panel,
            pv_source=pv_source,
        )
        load_mae, load_rmse = mae_rmse(pred["load_kw"], year["load_kw"][day])
        pv_mae, pv_rmse = mae_rmse(pred["pv_kw"], year["pv_kw"][day])
        out.append(
            {
                "day": day,
                "date": str(pd.Timestamp(dates.iloc[day]).date()),
                "load_kw": pred["load_kw"],
                "pv_kw": pred["pv_kw"],
                "xgb_used": bool(pred["xgb_used"]),
                "load_mae_kw": load_mae,
                "load_rmse_kw": load_rmse,
                "pv_mae_kw": pv_mae,
                "pv_rmse_kw": pv_rmse,
                "neg_pred": bool((pred["load_kw"] < -1e-12).any() or (pred["pv_kw"] < -1e-12).any()),
                "pv_source": pred.get("pv_source", pv_source),
            }
        )
    return out, time.perf_counter() - t0


def apply_bias_forecasts(
    forecasts: list[dict],
    year: dict,
    q_load: float | None,
    q_pv: float | None,
) -> list[dict]:
    load_resid: list[np.ndarray] = []
    pv_resid: list[np.ndarray] = []
    out = []
    for pred in forecasts:
        day = int(pred["day"])
        load_plan, pv_plan = apply_conservative_bias(
            pred["load_kw"],
            pred["pv_kw"],
            load_resid,
            pv_resid,
            q_load,
            q_pv,
            year["pv_kw"][:day],
        )
        load_resid.append(year["load_kw"][day] - pred["load_kw"])
        pv_resid.append(year["pv_kw"][day] - pred["pv_kw"])
        row = dict(pred)
        row["load_point_kw"] = pred["load_kw"]
        row["pv_point_kw"] = pred["pv_kw"]
        row["load_kw"] = load_plan
        row["pv_kw"] = pv_plan
        row["q_load"] = q_load
        row["q_pv"] = q_pv
        row["load_mae_kw"], row["load_rmse_kw"] = mae_rmse(load_plan, year["load_kw"][day])
        row["pv_mae_kw"], row["pv_rmse_kw"] = mae_rmse(pv_plan, year["pv_kw"][day])
        out.append(row)
    return out


def run_model(
    model_name: str,
    prices: pd.DataFrame,
    year: dict,
    start_idx: int,
    end_idx: int,
    load_panel: pd.DataFrame,
    pv_panel: pd.DataFrame,
    terminal_mode: str = "none",
    forecasts: list[dict] | None = None,
    forecast_s: float = 0.0,
    soc_mu: float = 0.0,
    dispatch: str = "greedy",
    mpc_stride: int = 1,
    policy_name: str | None = None,
    collect_traces: bool = False,
) -> dict:
    if forecasts is None:
        forecasts, forecast_s = collect_forecasts(
            model_name, prices, year, start_idx, end_idx, load_panel, pv_panel
        )
    price = prices["price"].to_numpy(dtype=float)
    sim_start = pd.Timestamp(SIM_START)
    soc_actual = E0_FEB1_KWH
    rows = []
    traces: list[dict] = []
    t0 = time.perf_counter()
    n_simultaneous = 0
    all_errors: list[str] = []
    neg_pred = 0
    official_start = pd.Timestamp(OFFICIAL_START)

    for pred in forecasts:
        stamp = pd.Timestamp(pred["date"])
        if stamp < sim_start:
            continue
        if stamp == sim_start:
            soc_actual = E0_FEB1_KWH
        day = int(pred["day"])
        if (pred["load_kw"] < -1e-12).any() or (pred["pv_kw"] < -1e-12).any() or pred["neg_pred"]:
            neg_pred += 1
        load_hat = pred["load_kw"] * DT_HOURS
        pv_hat = pred["pv_kw"] * DT_HOURS
        plan = solve_day_lp(price, load_hat, pv_hat, soc_actual, terminal_mode=terminal_mode, soc_mu=soc_mu)
        all_errors.extend(validate_plan(plan, price, load_hat, pv_hat))
        n_simultaneous += int(plan["n_simultaneous"])
        if dispatch == "mpc":
            actual = simulate_day_mpc(
                price,
                year["load_kwh"][day],
                year["pv_kwh"][day],
                load_hat,
                pv_hat,
                plan["purchase_kwh"],
                soc_actual,
                soc_mu=soc_mu,
                stride=mpc_stride,
            )
        else:
            actual = simulate_day(
                price,
                year["load_kwh"][day],
                year["pv_kwh"][day],
                plan["purchase_kwh"],
                soc_actual,
            )
        all_errors.extend(validate_actual(actual, plan["purchase_kwh"], year["load_kwh"][day], year["pv_kwh"][day]))
        stamp = pd.Timestamp(pred["date"])
        rows.append(
            {
                "date": pred["date"],
                "official": bool(stamp >= official_start),
                "terminal_mode": terminal_mode,
                "soc_mu": float(soc_mu),
                "dispatch": dispatch,
                "mpc_stride": int(mpc_stride) if dispatch == "mpc" else None,
                "policy": policy_name or model_name,
                "soc0_actual": float(soc_actual),
                "soc24_actual": float(actual["soc24_kwh"]),
                "soc24_plan": float(plan["soc_end_kwh"][-1]),
                "purchase_kwh": float(plan["purchase_kwh"].sum()),
                "plan_cost": float(actual["plan_cost"]),
                "emergency_kwh": float(actual["emergency_kwh"].sum()),
                "emergency_slots": int(actual["n_emergency_slots"]),
                "emergency_cost": float(actual["emergency_cost"]),
                "total_cost": float(actual["plan_cost"] + actual["emergency_cost"]),
                "curtail_kwh": float(actual["curtail_kwh"].sum()),
                "load_mae_kw": pred["load_mae_kw"],
                "load_rmse_kw": pred["load_rmse_kw"],
                "pv_mae_kw": pred["pv_mae_kw"],
                "pv_rmse_kw": pred["pv_rmse_kw"],
                "xgb_used": pred["xgb_used"],
                "n_simultaneous_plan": int(plan["n_simultaneous"]),
                "max_unserved_kwh": float(actual["max_unserved_kwh"]),
            }
        )
        soc_actual = float(actual["soc24_kwh"])
        if collect_traces:
            traces.append(
                {
                    "date": pred["date"],
                    "purchase_kwh": np.asarray(plan["purchase_kwh"], dtype=float).copy(),
                    "charge_kwh": np.asarray(actual["charge_kwh"], dtype=float).copy(),
                    "discharge_kwh": np.asarray(actual["discharge_kwh"], dtype=float).copy(),
                    "emergency_kwh": np.asarray(actual["emergency_kwh"], dtype=float).copy(),
                    "soc0_kwh": float(rows[-1]["soc0_actual"]),
                    "soc24_kwh": float(actual["soc24_kwh"]),
                    "plan_cost": float(actual["plan_cost"]),
                    "emergency_cost": float(actual["emergency_cost"]),
                }
            )

    elapsed = time.perf_counter() - t0 + forecast_s
    daily = pd.DataFrame(rows)
    official = daily[daily["official"]].copy()
    summary = {
        "model": model_name,
        "policy": policy_name or model_name,
        "terminal_mode": terminal_mode,
        "soc_mu": float(soc_mu),
        "dispatch": dispatch,
        "mpc_stride": int(mpc_stride) if dispatch == "mpc" else None,
        "n_days_run": int(len(daily)),
        "n_official_days": int(len(official)),
        "elapsed_s": elapsed,
        "forecast_s": forecast_s,
        "load_mae_kw": float(official["load_mae_kw"].mean()) if len(official) else None,
        "load_rmse_kw": float(official["load_rmse_kw"].mean()) if len(official) else None,
        "pv_mae_kw": float(official["pv_mae_kw"].mean()) if len(official) else None,
        "pv_rmse_kw": float(official["pv_rmse_kw"].mean()) if len(official) else None,
        "purchase_kwh": float(official["purchase_kwh"].sum()) if len(official) else 0.0,
        "plan_cost": float(official["plan_cost"].sum()) if len(official) else 0.0,
        "emergency_kwh": float(official["emergency_kwh"].sum()) if len(official) else 0.0,
        "emergency_slots": int(official["emergency_slots"].sum()) if len(official) else 0,
        "emergency_days": int((official["emergency_kwh"] > 1e-6).sum()) if len(official) else 0,
        "emergency_cost": float(official["emergency_cost"].sum()) if len(official) else 0.0,
        "total_cost": float(official["total_cost"].sum()) if len(official) else 0.0,
        "curtail_kwh": float(official["curtail_kwh"].sum()) if len(official) else 0.0,
        "n_simultaneous_plan": n_simultaneous,
        "n_negative_predictions": neg_pred,
        "n_validation_errors": len(all_errors),
        "validation_errors_head": all_errors[:20],
        "feb1_soc0": float(daily.loc[daily["date"] == "2025-02-01", "soc0_actual"].iloc[0])
        if len(daily) and "2025-02-01" in set(daily["date"])
        else None,
        "last_soc24_actual": float(daily["soc24_actual"].iloc[-1]) if len(daily) else None,
        "last_soc24_plan": float(daily["soc24_plan"].iloc[-1]) if len(daily) else None,
        "mean_soc24_actual": float(official["soc24_actual"].mean()) if len(official) else None,
        "mean_soc24_plan": float(official["soc24_plan"].mean()) if len(official) else None,
        "soc_min_actual": float(daily["soc24_actual"].min()) if len(daily) else None,
        "soc_max_actual": float(daily["soc24_actual"].max()) if len(daily) else None,
        "soc_init": "feb1_6000",
        "terminal_lambda": TERMINAL_LAMBDA if terminal_mode == "track6000" else 0.0,
        "terminal_target_kwh": TERMINAL_TARGET_KWH if terminal_mode == "track6000" else None,
    }
    if summary["feb1_soc0"] is not None and abs(summary["feb1_soc0"] - E0_FEB1_KWH) > 1e-6:
        raise RuntimeError(f"Feb 1 00:00 SOC must be {E0_FEB1_KWH}, got {summary['feb1_soc0']}")
    out = {"daily": daily, "summary": summary}
    if collect_traces:
        out["traces"] = traces
    return out


def write_verification_report(audit: dict, leakage: dict, summaries: list[dict], out_path: Path) -> None:
    lines = [
        "# Q2 建模核查报告（第一阶段）",
        "",
        "本报告对应 `code/Q2/run_q2.py --phase1`。正式全年数字以后续 `results/Q2/` 为准。",
        "",
        f"1. 时间粒度：附件1有 {audit['n_price_slots']} 个点，附件2有 {audit['n_days']} 天×144 点，步长均为 10 分钟；"
        f"日期 {audit['date_start']} 至 {audit['date_end']}，重复日期 {audit['duplicate_dates']}，"
        f"缺失负荷 {audit['missing_load']}、光伏 {audit['missing_pv']}。",
        f"2. kW→kWh：统一乘以 10/60={audit['dt_hours']}；第 0 日换算残差 {audit['kwh_check_load_day0']:.3e} kWh。",
        f"3. 未来信息泄漏检查：{'通过' if leakage['passed'] else '未通过'}。"
        f"错误 {leakage['n_errors']} 条。未做随机划分，特征与训练日均要求 < D。",
        "4. 2月1日初始SOC：固定为 6000 kWh。1 月历史只用于预报训练和残差分位，不把 1 月回测 SOC 滚到 2 月 1 日。",
        "5. 每日首尾 SOC 相等：问题2**没有**强制 SOC[d,0]=SOC[d,24]=6000。"
        "仅 2月1日 0:00 为 6000 kWh；此后 SOC 跨日传递实际值，并夹在 1200–10800 kWh。",
        f"6. 效率定义：建模假设充、放各 0.9，交流侧 "
        f"SOC += {ETA_CHARGE}*charge - discharge/{ETA_DISCHARGE}；"
        "不是题面另行给出的单程/往返拆分。",
        "7. 同时充放电：日前 LP 未加 0-1 变量；若 n_simultaneous_plan>0 则汇报，不自动改 MILP。回测规则互斥。",
        f"8. SOC 越界：计划与回测均约束/裁剪在 [{E_MIN_KWH}, {E_MAX_KWH}] kWh。见各模型 soc_min/max。",
        f"9. 功率越界：交流侧 charge,discharge ≤ {P_MAX_KWH:.6f} kWh/10min（5000 kW）。",
        "10. 负预测：预报后截断为 0；n_negative_predictions 统计截断前若仍出现则记次。",
        "11. 弃光：日前与回测均允许 w≥0。回测弃光 = 计划购电+实际光伏+放电+紧急 − 负荷 − 充电 的非负余量，"
        "在因果规则里等于充不进电池的多余能量。",
        "12. 紧急购电只在回测阶段产生；日前 LP 不用紧急变量覆盖预测负荷。",
        "13. 实际 SOC 与计划 SOC：下一天初值只用回测末值 soc24_actual，不用计划末值 soc24_plan。",
        "14. 供电缺口：回测校验 max_unserved_kwh，超过容差即失败。",
        "15. 运行时间见下表 elapsed_s。",
        "",
        "## 阶段指标（正式日子子集）",
        "",
        "| 模型 | Load MAE | PV MAE | 计划电量 | 计划费 | 紧急电量 | 紧急时段 | 紧急费 | 总费用 | 用时s | 同时充放 | 2/1 SOC0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {model} | {load_mae_kw:.3f} | {pv_mae_kw:.3f} | {purchase_kwh:.1f} | {plan_cost:.2f} | "
            "{emergency_kwh:.3f} | {emergency_slots} | {emergency_cost:.2f} | {total_cost:.2f} | "
            "{elapsed_s:.1f} | {n_simultaneous_plan} | {feb1_soc0} |".format(**{**s, "feb1_soc0": s["feb1_soc0"]})
        )
    lines.extend(
        [
            "",
            "## 泄漏检查摘要",
            "",
            json.dumps(leakage, ensure_ascii=False, indent=2),
            "",
            "## XGBoost 第一版超参数（Load/PV 相同）",
            "",
            json.dumps(XGB_PARAMS, ensure_ascii=False, indent=2),
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


PHASE1_TUNE_CONFIGS = [
    {"name": "V1", "pv_source": "model", "q_load": None, "q_pv": None, "soc_mu": 0.0, "dispatch": "greedy"},
    {"name": "A_mu0.25", "pv_source": "model", "q_load": None, "q_pv": None, "soc_mu": 0.25, "dispatch": "greedy"},
    {"name": "A_mu0.4", "pv_source": "model", "q_load": None, "q_pv": None, "soc_mu": 0.4, "dispatch": "greedy"},
    {"name": "B_mpc", "pv_source": "model", "q_load": None, "q_pv": None, "soc_mu": 0.0, "dispatch": "mpc"},
    {"name": "C_pv7d_q82", "pv_source": "baseline_7d", "q_load": 0.8, "q_pv": 0.2, "soc_mu": 0.0, "dispatch": "greedy"},
    {"name": "V2_mu0_q82", "pv_source": "baseline_7d", "q_load": 0.8, "q_pv": 0.2, "soc_mu": 0.0, "dispatch": "mpc"},
    {"name": "V2_mu0.25_q82", "pv_source": "baseline_7d", "q_load": 0.8, "q_pv": 0.2, "soc_mu": 0.25, "dispatch": "mpc"},
    {"name": "V2_mu0.4_q82", "pv_source": "baseline_7d", "q_load": 0.8, "q_pv": 0.2, "soc_mu": 0.4, "dispatch": "mpc"},
]

EXTRA_QUANTILE_CONFIG = {
    "name": "V2_mu0.25_q73",
    "pv_source": "baseline_7d",
    "q_load": 0.7,
    "q_pv": 0.3,
    "soc_mu": 0.25,
    "dispatch": "mpc",
}


def _load_v1_anchor() -> dict:
    path = FULL_DIR / "summary_xgb_expanding.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "total_cost": 15325094.14260998,
        "emergency_cost": 3198303.4307186403,
        "plan_cost": 12126790.711891338,
        "emergency_kwh": 525856.1753043083,
        "mean_soc24_plan": 1200.0,
        "feb1_soc0": 2010.3089489259246,
    }


def choose_mpc_stride(prices: pd.DataFrame, year: dict, forecasts: list[dict]) -> int:
    """Time one MPC day; fall back to hourly stride if 45-day budget would exceed ~3 min."""
    price = prices["price"].to_numpy(dtype=float)
    pred = next(p for p in forecasts if int(p["day"]) >= 7)
    day = int(pred["day"])
    t0 = time.perf_counter()
    plan = solve_day_lp(
        price,
        pred["load_kw"] * DT_HOURS,
        pred["pv_kw"] * DT_HOURS,
        E0_FEB1_KWH,
        soc_mu=0.25,
    )
    simulate_day_mpc(
        price,
        year["load_kwh"][day],
        year["pv_kwh"][day],
        pred["load_kw"] * DT_HOURS,
        pred["pv_kw"] * DT_HOURS,
        plan["purchase_kwh"],
        E0_FEB1_KWH,
        stride=MPC_STRIDE,
    )
    one_day = time.perf_counter() - t0
    projected = one_day * 45.0
    stride = MPC_STRIDE
    if projected > PHASE1_MPC_BUDGET_S:
        stride = MPC_STRIDE_FALLBACK
    print(
        f"mpc probe day {pred['date']}: {one_day:.3f}s/day, "
        f"phase1 project {projected:.1f}s, stride={stride}",
        flush=True,
    )
    return stride


def run_policy(
    cfg: dict,
    prices: pd.DataFrame,
    year: dict,
    start_idx: int,
    end_idx: int,
    load_panel: pd.DataFrame,
    pv_panel: pd.DataFrame,
    forecast_bank: dict,
    mpc_stride: int,
    collect_traces: bool = False,
) -> dict:
    source = cfg["pv_source"]
    forecasts = forecast_bank[source]
    forecast_s = forecast_bank[f"{source}_s"]
    biased = apply_bias_forecasts(forecasts, year, cfg["q_load"], cfg["q_pv"])
    result = run_model(
        "xgb_expanding",
        prices,
        year,
        start_idx,
        end_idx,
        load_panel,
        pv_panel,
        terminal_mode="none",
        forecasts=biased,
        forecast_s=forecast_s,
        soc_mu=float(cfg["soc_mu"]),
        dispatch=cfg["dispatch"],
        mpc_stride=mpc_stride,
        policy_name=cfg["name"],
        collect_traces=collect_traces,
    )
    result["summary"]["q_load"] = cfg["q_load"]
    result["summary"]["q_pv"] = cfg["q_pv"]
    result["summary"]["pv_source"] = source
    return result


def write_opt_comparison(summaries: list[dict], selected: dict, path: Path) -> None:
    anchor = _load_v1_anchor()
    rows = []
    for s in summaries:
        rows.append(
            {
                "policy": s.get("policy"),
                "window": "phase1_official" if s.get("n_official_days", 0) <= 20 else "full_official",
                "n_official_days": s.get("n_official_days"),
                "total_cost": s.get("total_cost"),
                "plan_cost": s.get("plan_cost"),
                "emergency_cost": s.get("emergency_cost"),
                "emergency_kwh": s.get("emergency_kwh"),
                "purchase_kwh": s.get("purchase_kwh"),
                "mean_soc24_actual": s.get("mean_soc24_actual"),
                "mean_soc24_plan": s.get("mean_soc24_plan"),
                "feb1_soc0": s.get("feb1_soc0"),
                "load_mae_kw": s.get("load_mae_kw"),
                "pv_mae_kw": s.get("pv_mae_kw"),
                "soc_mu": s.get("soc_mu"),
                "dispatch": s.get("dispatch"),
                "q_load": s.get("q_load"),
                "q_pv": s.get("q_pv"),
                "pv_source": s.get("pv_source"),
                "mpc_stride": s.get("mpc_stride"),
                "elapsed_s": s.get("elapsed_s"),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    (path.parent / "v1_anchor.json").write_text(json.dumps(anchor, ensure_ascii=False, indent=2), encoding="utf-8")
    if selected:
        (path.parent / "selected_config.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def select_official_config(summaries: list[dict], configs: list[dict]) -> dict:
    v1 = next(s for s in summaries if s["policy"] == "V1")
    non_v1 = [s for s in summaries if s["policy"] != "V1"]
    improved = [
        s
        for s in non_v1
        if s["total_cost"] < v1["total_cost"] and s["emergency_cost"] < v1["emergency_cost"]
    ]
    pool = improved if improved else non_v1
    best = min(pool, key=lambda s: (s["total_cost"], s["emergency_cost"]))
    cfg = next(c for c in configs if c["name"] == best["policy"])
    selected = dict(cfg)
    selected["phase1_total_cost"] = best["total_cost"]
    selected["phase1_emergency_cost"] = best["emergency_cost"]
    selected["phase1_vs_v1_total"] = best["total_cost"] - v1["total_cost"]
    selected["mpc_stride"] = best.get("mpc_stride")
    selected["v1_phase1_total_cost"] = v1["total_cost"]
    selected["improved_on_phase1"] = bool(improved)
    return selected


def run_oracle_phase1(prices: pd.DataFrame, year: dict, start_idx: int, end_idx: int) -> dict:
    price = prices["price"].to_numpy(dtype=float)
    official_start = pd.Timestamp(OFFICIAL_START)
    sim_start = pd.Timestamp(SIM_START)
    soc = E0_FEB1_KWH
    rows = []
    t0 = time.perf_counter()
    for day in range(start_idx, end_idx + 1):
        stamp = pd.Timestamp(year["dates"].iloc[day])
        if stamp < sim_start:
            continue
        if stamp == sim_start:
            soc = E0_FEB1_KWH
        load = year["load_kwh"][day]
        pv = year["pv_kwh"][day]
        plan = solve_day_lp(price, load, pv, soc, soc_mu=0.0)
        actual = simulate_day_mpc(price, load, pv, load, pv, plan["purchase_kwh"], soc, soc_mu=0.0, stride=1)
        rows.append(
            {
                "date": str(stamp.date()),
                "official": bool(stamp >= official_start),
                "total_cost": float(actual["plan_cost"] + actual["emergency_cost"]),
                "plan_cost": float(actual["plan_cost"]),
                "emergency_cost": float(actual["emergency_cost"]),
                "emergency_kwh": float(actual["emergency_kwh"].sum()),
                "soc24_actual": float(actual["soc24_kwh"]),
            }
        )
        soc = float(actual["soc24_kwh"])
    daily = pd.DataFrame(rows)
    official = daily[daily["official"]]
    return {
        "policy": "oracle_actuals",
        "n_official_days": int(len(official)),
        "total_cost": float(official["total_cost"].sum()),
        "plan_cost": float(official["plan_cost"].sum()),
        "emergency_cost": float(official["emergency_cost"].sum()),
        "emergency_kwh": float(official["emergency_kwh"].sum()),
        "mean_soc24_actual": float(official["soc24_actual"].mean()),
        "elapsed_s": time.perf_counter() - t0,
        "note": "perfect-information lower bound; not an official score",
    }


def prepare_forecast_bank(
    prices: pd.DataFrame,
    year: dict,
    start_idx: int,
    end_idx: int,
    load_panel: pd.DataFrame,
    pv_panel: pd.DataFrame,
) -> dict:
    cache: dict = {}
    xgb, t_xgb = collect_forecasts(
        "xgb_expanding", prices, year, start_idx, end_idx, load_panel, pv_panel, cache=cache, pv_source="model"
    )
    hyb, t_hyb = collect_forecasts(
        "xgb_expanding",
        prices,
        year,
        start_idx,
        end_idx,
        load_panel,
        pv_panel,
        cache=cache,
        pv_source="baseline_7d",
    )
    print(f"forecast bank xgb={t_xgb:.1f}s hybrid={t_hyb:.1f}s", flush=True)
    return {"model": xgb, "model_s": t_xgb, "baseline_7d": hyb, "baseline_7d_s": t_hyb}


ADAPTIVE_QMIN_GRID = (0.60, 0.65, 0.70, 0.75, 0.80)
ADAPTIVE_K_GRID = (0.10, 0.20, 0.30)


def _adaptive_inputs(refresh: bool = False):
    copy_raw_inputs()
    prices = load_prices()
    year = load_aligned_year(prices)
    audit_data(prices, year)
    dates = year["dates"]
    end_idx = int(np.where(pd.to_datetime(dates) == pd.Timestamp(OFFICIAL_END))[0][0])
    point = load_point_bank(prices, year, 0, end_idx, refresh=refresh)
    check_length(point)
    banks = ladder_banks(point, year, Q_LADDER)
    residuals = point_residual_history(point, year)
    return prices, year, point, banks, residuals


def run_adaptive_tune(args) -> None:
    """Grid-search the adaptive rule on OFFICIAL_START..TUNE_END only."""
    prices, year, _point, banks, residuals = _adaptive_inputs(refresh=args.refresh_bank)
    ADAPTIVE_DIR.mkdir(parents=True, exist_ok=True)
    windows = [
        ("tune", OFFICIAL_START, TUNE_END),
        ("inner", OFFICIAL_START, TUNE_INNER_END),
        ("select", TUNE_SELECT_START, TUNE_END),
    ]
    rows = []
    summaries = []
    daily_parts = []

    for q in Q_LADDER:
        print(f"tune fixed q={q}", flush=True)
        result = run_fixed(prices, year, banks, q, TUNE_END)
        if result["errors"]:
            raise RuntimeError(f"fixed q={q} failed: {result['errors'][:5]}")
        summary = summarise(result, windows)
        summaries.append(summary)
        rows.append(flatten_for_table(summary))
        daily_parts.append(result["daily"].assign(label=summary["label"]))

    for feature in FEATURE_NAMES:
        for q_min in ADAPTIVE_QMIN_GRID:
            for k in ADAPTIVE_K_GRID:
                label = f"{feature}_qmin{q_min:.2f}_k{k:.2f}"
                print("tune", label, flush=True)
                result = run_adaptive(
                    prices, year, banks, residuals, TUNE_END, feature=feature, q_min=q_min, k=k, label=label
                )
                if result["errors"]:
                    raise RuntimeError(f"{label} failed: {result['errors'][:5]}")
                summary = summarise(result, windows)
                summaries.append(summary)
                rows.append(flatten_for_table(summary))
                daily_parts.append(result["daily"].assign(label=label))

    frame = pd.DataFrame(rows).sort_values("select_total_cost").reset_index(drop=True)
    frame.to_csv(ADAPTIVE_DIR / "tune_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_parts, ignore_index=True).to_csv(
        ADAPTIVE_DIR / "tune_daily_all.csv", index=False, encoding="utf-8-sig"
    )

    baseline = next(s for s in summaries if s["label"] == "fixed_q80")
    base = {name: baseline["windows"][name]["total_cost"] for name in ("tune", "inner", "select")}
    adaptive_only = [s for s in summaries if s["rule"].get("feature") is not None]
    # Nested protocol: a candidate must first help on the inner window, then the winner is
    # the best of those on the select window. Neither step looks at 07-01..12-31.
    eligible = [s for s in adaptive_only if s["windows"]["inner"]["total_cost"] < base["inner"]]
    pool = eligible or adaptive_only
    best = min(pool, key=lambda s: s["windows"]["select"]["total_cost"])
    selected = dict(best["rule"])
    selected["selection_protocol"] = {
        "step1": f"beat fixed 0.8 on inner window {OFFICIAL_START}..{TUNE_INNER_END}",
        "step2": f"lowest total cost on select window {TUNE_SELECT_START}..{TUNE_END}",
        "n_candidates": len(adaptive_only),
        "n_eligible_after_step1": len(eligible),
        "fell_back_to_full_pool": not eligible,
    }
    selected["tune_window"] = {"start": OFFICIAL_START, "end": TUNE_END}
    selected["fixed_q80_costs"] = base
    for name in ("tune", "inner", "select"):
        selected[f"{name}_total_cost"] = best["windows"][name]["total_cost"]
        selected[f"{name}_gain_vs_fixed_q80"] = base[name] - best["windows"][name]["total_cost"]
    selected["beats_fixed_q80_on_tune"] = bool(best["windows"]["tune"]["total_cost"] < base["tune"])
    selected["q_counts_tune"] = best["windows"]["tune"].get("q_counts")
    selected["grid"] = {"features": list(FEATURE_NAMES), "q_min": list(ADAPTIVE_QMIN_GRID), "k": list(ADAPTIVE_K_GRID)}
    selected["note"] = (
        "Tuned on the tune window only. Out-of-sample and full-year numbers come from "
        "--adaptive-full. Does not overwrite results/Q2/opt/ or result2.xlsx."
    )
    (ADAPTIVE_DIR / "selected_rule.json").write_text(dumps(selected), encoding="utf-8")
    (ADAPTIVE_DIR / "tune_summaries.json").write_text(dumps(summaries), encoding="utf-8")
    print(frame.head(12).to_string(index=False), flush=True)
    print("selected", dumps(selected), flush=True)
    print("wrote", ADAPTIVE_DIR, flush=True)


def run_adaptive_full(args) -> None:
    """Freeze the tuned rule, then replay 02-01..12-31 with SOC rolling throughout."""
    rule_path = ADAPTIVE_DIR / "selected_rule.json"
    if not rule_path.exists():
        raise SystemExit("missing results/Q2/adaptive/selected_rule.json; run --adaptive-tune first")
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    prices, year, _point, banks, residuals = _adaptive_inputs(refresh=args.refresh_bank)
    ADAPTIVE_DIR.mkdir(parents=True, exist_ok=True)
    windows = [
        ("tune", OFFICIAL_START, TUNE_END),
        ("oos", OOS_START, OFFICIAL_END),
        ("full", OFFICIAL_START, OFFICIAL_END),
    ]

    print("full year: adaptive", rule["label"], flush=True)
    adaptive = run_adaptive(
        prices,
        year,
        banks,
        residuals,
        OFFICIAL_END,
        feature=rule["feature"],
        q_min=float(rule["q_min"]),
        k=float(rule["k"]),
        label=rule["label"],
        collect_traces=True,
    )
    if adaptive["errors"]:
        raise RuntimeError(f"adaptive full year failed: {adaptive['errors'][:5]}")
    adaptive_summary = summarise(adaptive, windows)
    adaptive["daily"].to_csv(ADAPTIVE_DIR / "daily_adaptive.csv", index=False, encoding="utf-8-sig")

    print("full year: fixed q=0.8 baseline", flush=True)
    base = run_fixed(prices, year, banks, 0.8, OFFICIAL_END)
    if base["errors"]:
        raise RuntimeError(f"fixed q=0.8 full year failed: {base['errors'][:5]}")
    base_summary = summarise(base, windows)
    base["daily"].to_csv(ADAPTIVE_DIR / "daily_fixed_q80.csv", index=False, encoding="utf-8-sig")

    full_adaptive = adaptive_summary["windows"]["full"]
    full_base = base_summary["windows"]["full"]
    oos_adaptive = adaptive_summary["windows"]["oos"]
    oos_base = base_summary["windows"]["oos"]

    gate = {
        "beats_frozen_on_full_year": bool(full_adaptive["total_cost"] < FROZEN_C_Q82_FULL_YEAR_COST),
        "beats_fixed_q80_out_of_sample": bool(oos_adaptive["total_cost"] < oos_base["total_cost"]),
        "clean_validation": bool(
            adaptive_summary["n_validation_errors"] == 0
            and full_adaptive["max_unserved_kwh"] <= 1e-3
            and full_adaptive["n_simultaneous_plan"] == 0
        ),
    }
    gate["passed"] = bool(all(gate.values()))
    gate["allow_export_result2"] = gate["passed"]

    report = {
        "rule": rule,
        "frozen_c82_full_year_cost": FROZEN_C_Q82_FULL_YEAR_COST,
        "adaptive": adaptive_summary,
        "fixed_q80": base_summary,
        "deltas": {
            "full_vs_frozen": full_adaptive["total_cost"] - FROZEN_C_Q82_FULL_YEAR_COST,
            "full_vs_fixed_q80": full_adaptive["total_cost"] - full_base["total_cost"],
            "oos_vs_fixed_q80": oos_adaptive["total_cost"] - oos_base["total_cost"],
            "tune_vs_fixed_q80": adaptive_summary["windows"]["tune"]["total_cost"]
            - base_summary["windows"]["tune"]["total_cost"],
        },
        "gate": gate,
    }
    # The gate only clears the candidate for adoption. The official result2.xlsx and
    # results/Q2/opt/ stay untouched until a teammate confirms the swap, so the staged
    # workbook is written beside the adaptive evidence instead.
    if gate["passed"]:
        staged = export_result2(prices, adaptive["traces"], ADAPTIVE_DIR / "result2_adaptive.xlsx")
        report["staged_result2"] = str(staged)
        print("staged candidate workbook:", staged, flush=True)
    (ADAPTIVE_DIR / "full_year_summary.json").write_text(dumps(report), encoding="utf-8")
    (ADAPTIVE_DIR / "oos_summary.json").write_text(
        dumps({"adaptive": oos_adaptive, "fixed_q80": oos_base, "delta": report["deltas"]["oos_vs_fixed_q80"]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [flatten_for_table(adaptive_summary), flatten_for_table(base_summary)]
    ).to_csv(ADAPTIVE_DIR / "full_comparison.csv", index=False, encoding="utf-8-sig")
    print(dumps(report["deltas"]), flush=True)
    print("gate", dumps(gate), flush=True)
    print("wrote", ADAPTIVE_DIR, flush=True)


def run_adopt_adaptive(args) -> None:
    """Promote the gated adaptive rule to the official policy and re-export result2.

    The superseded fixed-0.8 summary and config are archived beside the new ones rather
    than deleted, so the comparison in the paper keeps a verifiable source.
    """
    rule_path = ADAPTIVE_DIR / "selected_rule.json"
    report_path = ADAPTIVE_DIR / "full_year_summary.json"
    if not rule_path.exists() or not report_path.exists():
        raise SystemExit("run --adaptive-tune and --adaptive-full first")
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report["gate"]["passed"]:
        raise SystemExit("gate did not pass; refusing to adopt")

    prices, year, _point, banks, residuals = _adaptive_inputs(refresh=args.refresh_bank)
    policy = "D_pv7d_adaptive"
    print("full-year replay for adoption:", rule["label"], flush=True)
    result = run_adaptive(
        prices,
        year,
        banks,
        residuals,
        OFFICIAL_END,
        feature=rule["feature"],
        q_min=float(rule["q_min"]),
        k=float(rule["k"]),
        label=policy,
        collect_traces=True,
    )
    summary = official_summary(result, policy, PV_SOURCE_V2)
    if summary["n_validation_errors"]:
        raise RuntimeError(f"adoption replay failed: {summary['validation_errors_head']}")
    want = float(report["adaptive"]["windows"]["full"]["total_cost"])
    if abs(summary["total_cost"] - want) > 1e-6:
        raise RuntimeError(f"adoption replay {summary['total_cost']} != gated {want}")

    OPT_DIR.mkdir(parents=True, exist_ok=True)
    archive = {
        OPT_DIR / "full_year_summary.json": OPT_DIR / "full_year_summary_C_pv7d_q82.json",
        OPT_DIR / "selected_config.json": OPT_DIR / "selected_config_C_pv7d_q82.json",
    }
    for live, kept in archive.items():
        if live.exists() and not kept.exists():
            kept.write_text(live.read_text(encoding="utf-8"), encoding="utf-8")
            print("archived", live.name, "->", kept.name, flush=True)

    selected = {
        "name": policy,
        "pv_source": PV_SOURCE_V2,
        "q_load": None,
        "q_pv": None,
        "soc_mu": 0.0,
        "dispatch": "greedy",
        "mpc_stride": None,
        "margin": {
            "mode": "adaptive",
            "feature": rule["feature"],
            "q_min": float(rule["q_min"]),
            "k": float(rule["k"]),
            "ladder": [float(q) for q in rule["ladder"]],
            "q_warmup": float(rule["q_warmup"]),
            "warmup_days": int(rule["warmup_days"]),
            "min_history": int(rule["min_history"]),
            "formula": "q_L(D) = snap(clip(q_min + k * z(D), min ladder, max ladder)), z = causal percentile",
        },
        "supersedes": {
            "name": "C_pv7d_q82",
            "total_cost": FROZEN_C_Q82_FULL_YEAR_COST,
            "archive": "results/Q2/opt/full_year_summary_C_pv7d_q82.json",
        },
        "selection": rule.get("selection_protocol"),
        "evidence": "results/Q2/adaptive/summary.md",
        "note": (
            "Official Q2 policy. Margin is chosen per day from resid_vol7; everything else "
            "matches C_pv7d_q82, which is kept as the fixed-margin comparison."
        ),
    }
    (OPT_DIR / "selected_config.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OPT_DIR / "full_year_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result["daily"].to_csv(OPT_DIR / "daily_D_pv7d_adaptive.csv", index=False, encoding="utf-8-sig")

    dest = export_result2(prices, result["traces"], RESULT2_XLSX)
    table_paths = export_specified_day_tables(prices, result["traces"])
    print("wrote", dest, flush=True)
    for path in table_paths:
        print("wrote", path, flush=True)
    print("NEW FROZEN total_cost =", repr(summary["total_cost"]), flush=True)
    print("superseded C_pv7d_q82 =", repr(FROZEN_C_Q82_FULL_YEAR_COST), flush=True)
    print("delta =", summary["total_cost"] - FROZEN_C_Q82_FULL_YEAR_COST, flush=True)


def run_opt(args) -> None:
    copy_raw_inputs()
    prices = load_prices()
    year = load_aligned_year(prices)
    audit_data(prices, year)
    dates = year["dates"]
    end_date = pd.Timestamp(OFFICIAL_END if args.full else (args.end_date or PHASE1_END))
    end_idx = int(np.where(pd.to_datetime(dates) == end_date)[0][0])
    start_idx = 0
    print("precomputing causal feature panels...", flush=True)
    load_panel = precompute_panel(year["load_kw"], dates)
    pv_panel = precompute_panel(year["pv_kw"], dates)
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    bank = prepare_forecast_bank(prices, year, start_idx, end_idx, load_panel, pv_panel)
    mpc_stride = choose_mpc_stride(prices, year, bank["baseline_7d"])

    if args.phase1_tune or not args.full:
        configs = list(PHASE1_TUNE_CONFIGS)
        summaries = []
        for cfg in configs:
            print("opt phase1", cfg["name"], flush=True)
            result = run_policy(cfg, prices, year, start_idx, end_idx, load_panel, pv_panel, bank, mpc_stride)
            if result["summary"]["n_validation_errors"]:
                raise RuntimeError(f"{cfg['name']} failed: {result['summary']['validation_errors_head']}")
            result["daily"].to_csv(OPT_DIR / f"phase1_daily_{cfg['name']}.csv", index=False, encoding="utf-8-sig")
            (OPT_DIR / f"phase1_summary_{cfg['name']}.json").write_text(
                json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summaries.append(result["summary"])
            print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
        v1 = next(s for s in summaries if s["policy"] == "V1")
        v2_like = [s for s in summaries if s["policy"].startswith("V2") or s["policy"].startswith("C_")]
        if v2_like and not (min(v2_like, key=lambda s: s["emergency_cost"])["emergency_cost"] < v1["emergency_cost"]):
            print("V2 q=0.8/0.2 emergency did not drop; adding q=0.7/0.3", flush=True)
            cfg = EXTRA_QUANTILE_CONFIG
            configs.append(cfg)
            result = run_policy(cfg, prices, year, start_idx, end_idx, load_panel, pv_panel, bank, mpc_stride)
            result["daily"].to_csv(OPT_DIR / f"phase1_daily_{cfg['name']}.csv", index=False, encoding="utf-8-sig")
            (OPT_DIR / f"phase1_summary_{cfg['name']}.json").write_text(
                json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summaries.append(result["summary"])
        oracle = run_oracle_phase1(prices, year, start_idx, end_idx)
        (OPT_DIR / "phase1_oracle.json").write_text(json.dumps(oracle, ensure_ascii=False, indent=2), encoding="utf-8")
        selected = select_official_config(summaries, configs)
        selected["mpc_stride"] = mpc_stride
        pd.DataFrame(summaries + [oracle]).to_csv(OPT_DIR / "phase1_ablation.csv", index=False, encoding="utf-8-sig")
        write_opt_comparison(summaries, selected, OPT_DIR / "phase1_comparison.csv")
        print("selected", json.dumps(selected, ensure_ascii=False), flush=True)
        if not args.full:
            return
    else:
        selected_path = OPT_DIR / "selected_config.json"
        if not selected_path.exists():
            raise SystemExit("missing results/Q2/opt/selected_config.json; run --phase1-tune first")
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        mpc_stride = int(selected.get("mpc_stride") or mpc_stride)

    print("opt full year", selected["name"], flush=True)
    result = run_policy(selected, prices, year, start_idx, end_idx, load_panel, pv_panel, bank, mpc_stride)
    if result["summary"]["n_validation_errors"]:
        raise RuntimeError(f"full V2 failed: {result['summary']['validation_errors_head']}")
    result["daily"].to_csv(OPT_DIR / "daily_v2.csv", index=False, encoding="utf-8-sig")
    (OPT_DIR / "full_year_summary.json").write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    v1_cfg = next(c for c in PHASE1_TUNE_CONFIGS if c["name"] == "V1")
    print("opt full year same-stack V1", flush=True)
    v1_result = run_policy(v1_cfg, prices, year, start_idx, end_idx, load_panel, pv_panel, bank, mpc_stride)
    v1_result["daily"].to_csv(OPT_DIR / "daily_v1_samestack.csv", index=False, encoding="utf-8-sig")
    (OPT_DIR / "full_year_summary_v1_samestack.json").write_text(
        json.dumps(v1_result["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    anchor = _load_v1_anchor()
    comparison = pd.DataFrame(
        [
            {
                "policy": "V1_repo_anchor",
                "total_cost": anchor["total_cost"],
                "plan_cost": anchor["plan_cost"],
                "emergency_cost": anchor["emergency_cost"],
                "emergency_kwh": anchor["emergency_kwh"],
                "mean_soc24_plan": anchor.get("mean_soc24_plan"),
                "feb1_soc0": anchor.get("feb1_soc0"),
                "source": "results/Q2/full_year/summary_xgb_expanding.json",
            },
            {
                "policy": "V1_samestack",
                "total_cost": v1_result["summary"]["total_cost"],
                "plan_cost": v1_result["summary"]["plan_cost"],
                "emergency_cost": v1_result["summary"]["emergency_cost"],
                "emergency_kwh": v1_result["summary"]["emergency_kwh"],
                "mean_soc24_actual": v1_result["summary"]["mean_soc24_actual"],
                "mean_soc24_plan": v1_result["summary"]["mean_soc24_plan"],
                "feb1_soc0": v1_result["summary"]["feb1_soc0"],
                "purchase_kwh": v1_result["summary"]["purchase_kwh"],
                "load_mae_kw": v1_result["summary"]["load_mae_kw"],
                "pv_mae_kw": v1_result["summary"]["pv_mae_kw"],
                "source": "results/Q2/opt/full_year_summary_v1_samestack.json",
            },
            {
                "policy": selected["name"],
                "total_cost": result["summary"]["total_cost"],
                "plan_cost": result["summary"]["plan_cost"],
                "emergency_cost": result["summary"]["emergency_cost"],
                "emergency_kwh": result["summary"]["emergency_kwh"],
                "mean_soc24_actual": result["summary"]["mean_soc24_actual"],
                "mean_soc24_plan": result["summary"]["mean_soc24_plan"],
                "feb1_soc0": result["summary"]["feb1_soc0"],
                "purchase_kwh": result["summary"]["purchase_kwh"],
                "load_mae_kw": result["summary"]["load_mae_kw"],
                "pv_mae_kw": result["summary"]["pv_mae_kw"],
                "soc_mu": result["summary"]["soc_mu"],
                "dispatch": result["summary"]["dispatch"],
                "mpc_stride": result["summary"]["mpc_stride"],
                "q_load": result["summary"].get("q_load"),
                "q_pv": result["summary"].get("q_pv"),
                "source": "results/Q2/opt/full_year_summary.json",
            },
        ]
    )
    comparison.to_csv(OPT_DIR / "comparison_v1_v2.csv", index=False, encoding="utf-8-sig")
    print("wrote", OPT_DIR / "comparison_v1_v2.csv", flush=True)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


def _assert_matches_frozen(summary: dict) -> None:
    frozen_path = OPT_DIR / "full_year_summary.json"
    if not frozen_path.exists():
        raise SystemExit("missing results/Q2/opt/full_year_summary.json; run --opt --full first")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    keys = ("total_cost", "plan_cost", "emergency_cost", "emergency_kwh", "purchase_kwh", "feb1_soc0")
    mismatches = []
    for key in keys:
        got = float(summary[key])
        want = float(frozen[key])
        scale = max(1.0, abs(want))
        if abs(got - want) > max(1e-6, 1e-10 * scale):
            mismatches.append(f"{key}: frozen {want} vs export {got}")
    if mismatches:
        raise RuntimeError("export replay does not match frozen V2 summary: " + "; ".join(mismatches))


def _export_result2_adaptive(prices: pd.DataFrame, year: dict, selected: dict, margin: dict) -> None:
    """Official export for the adopted adaptive-margin policy."""
    dates = year["dates"]
    end_idx = int(np.where(pd.to_datetime(dates) == pd.Timestamp(OFFICIAL_END))[0][0])
    point = load_point_bank(prices, year, 0, end_idx)
    check_length(point)
    ladder = tuple(margin.get("ladder") or Q_LADDER)
    banks = ladder_banks(point, year, ladder)
    residuals = point_residual_history(point, year)
    print("replay official adaptive policy", selected["name"], "with slot traces", flush=True)
    result = run_adaptive(
        prices,
        year,
        banks,
        residuals,
        OFFICIAL_END,
        feature=margin["feature"],
        q_min=float(margin["q_min"]),
        k=float(margin["k"]),
        ladder=ladder,
        label=selected["name"],
        collect_traces=True,
    )
    summary = official_summary(result, selected["name"], selected["pv_source"])
    if summary["n_validation_errors"]:
        raise RuntimeError(f"result2 replay failed: {summary['validation_errors_head']}")
    _assert_matches_frozen(summary)
    dest = export_result2(prices, result["traces"], RESULT2_XLSX)
    table_paths = export_specified_day_tables(prices, result["traces"])
    print("wrote", dest, flush=True)
    for path in table_paths:
        print("wrote", path, flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def run_export_result2() -> None:
    copy_raw_inputs()
    prices = load_prices()
    year = load_aligned_year(prices)
    audit_data(prices, year)
    selected_path = OPT_DIR / "selected_config.json"
    if not selected_path.exists():
        raise SystemExit("missing results/Q2/opt/selected_config.json; run --opt --phase1-tune first")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    margin = selected.get("margin") or {"mode": "fixed"}
    if margin.get("mode") == "adaptive":
        _export_result2_adaptive(prices, year, selected, margin)
        return
    dates = year["dates"]
    end_idx = int(np.where(pd.to_datetime(dates) == pd.Timestamp(OFFICIAL_END))[0][0])
    start_idx = 0
    print("precomputing causal feature panels...", flush=True)
    load_panel = precompute_panel(year["load_kw"], dates)
    pv_panel = precompute_panel(year["pv_kw"], dates)
    print("forecast bank for result2 export", selected["name"], flush=True)
    source = selected["pv_source"]
    forecasts, forecast_s = collect_forecasts(
        "xgb_expanding",
        prices,
        year,
        start_idx,
        end_idx,
        load_panel,
        pv_panel,
        pv_source=source,
    )
    bank = {source: forecasts, f"{source}_s": forecast_s}
    mpc_stride = int(selected.get("mpc_stride") or MPC_STRIDE)
    print("replay frozen V2 with slot traces", flush=True)
    result = run_policy(
        selected,
        prices,
        year,
        start_idx,
        end_idx,
        load_panel,
        pv_panel,
        bank,
        mpc_stride,
        collect_traces=True,
    )
    if result["summary"]["n_validation_errors"]:
        raise RuntimeError(f"result2 replay failed: {result['summary']['validation_errors_head']}")
    _assert_matches_frozen(result["summary"])
    dest = export_result2(prices, result["traces"], RESULT2_XLSX)
    table_paths = export_specified_day_tables(prices, result["traces"])
    print("wrote", dest, flush=True)
    for path in table_paths:
        print("wrote", path, flush=True)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--opt", action="store_true")
    parser.add_argument("--phase1-tune", action="store_true")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--skip-leakage", action="store_true")
    parser.add_argument("--export-result2", action="store_true")
    parser.add_argument("--adaptive-tune", action="store_true")
    parser.add_argument("--adaptive-full", action="store_true")
    parser.add_argument("--adopt-adaptive", action="store_true")
    parser.add_argument("--refresh-bank", action="store_true")
    args = parser.parse_args()
    if args.adopt_adaptive:
        run_adopt_adaptive(args)
        return
    if args.export_result2:
        run_export_result2()
        return
    if args.adaptive_tune:
        run_adaptive_tune(args)
        return
    if args.adaptive_full:
        run_adaptive_full(args)
        return
    if args.opt:
        if not args.full and not args.phase1_tune:
            args.phase1_tune = True
        run_opt(args)
        return
    if not args.full and not args.phase1:
        args.phase1 = True
    copy_raw_inputs()
    prices = load_prices()
    year = load_aligned_year(prices)
    audit = audit_data(prices, year)
    write_clean(prices, year, audit)

    dates = year["dates"]
    end_date = pd.Timestamp(OFFICIAL_END if args.full else (args.end_date or PHASE1_END))
    end_idx = int(np.where(pd.to_datetime(dates) == end_date)[0][0])
    start_idx = 0

    print("precomputing causal feature panels...", flush=True)
    load_panel = precompute_panel(year["load_kw"], dates)
    pv_panel = precompute_panel(year["pv_kw"], dates)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    leakage = {"passed": True, "n_errors": 0, "errors": [], "note": "skipped"}
    if not args.skip_leakage:
        probe = [1, min(31, end_idx), min(end_idx, 44)]
        probe = sorted(set(i for i in probe if i >= 1))
        print("running leakage suite on days", probe, flush=True)
        leakage = run_leakage_suite(
            year["load_kw"],
            year["pv_kw"],
            dates,
            prices["typical_load_kw"].to_numpy(dtype=float),
            prices["typical_pv_kw"].to_numpy(dtype=float),
            probe,
            MODEL_NAMES,
        )
        print("leakage passed:", leakage["passed"], "errors", leakage["n_errors"], flush=True)
        (PHASE1_DIR / "leakage.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
        if not leakage["passed"]:
            write_verification_report(audit, leakage, [], PHASE1_DIR / "verification_report.md")
            raise SystemExit("LEAKAGE CHECK FAILED; models were not run")

    out_root = FULL_DIR if args.full else PHASE1_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in MODEL_NAMES:
        print("forecasting", name, "through", end_date.date(), flush=True)
        forecasts, forecast_s = collect_forecasts(
            name, prices, year, start_idx, end_idx, load_panel, pv_panel
        )
        print(f"  forecast {forecast_s:.1f}s", flush=True)
        print("  backtest none", flush=True)
        result = run_model(
            name,
            prices,
            year,
            start_idx,
            end_idx,
            load_panel,
            pv_panel,
            terminal_mode="none",
            forecasts=forecasts,
            forecast_s=forecast_s,
        )
        result["daily"].to_csv(out_root / f"daily_{name}.csv", index=False, encoding="utf-8-sig")
        (out_root / f"summary_{name}.json").write_text(
            json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries.append(result["summary"])
        print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
        if result["summary"]["n_validation_errors"]:
            raise RuntimeError(f"{name} failed validation: {result['summary']['validation_errors_head']}")

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(out_root / "model_comparison.csv", index=False, encoding="utf-8-sig")
    if args.full:
        print("wrote", out_root / "model_comparison.csv", flush=True)
    else:
        write_verification_report(audit, leakage, summaries, PHASE1_DIR / "verification_report.md")
        print("wrote", PHASE1_DIR / "verification_report.md", flush=True)


if __name__ == "__main__":
    main()
