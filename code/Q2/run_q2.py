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
    ATTACHMENT1_XLSX,
    ATTACHMENT2_XLSX,
    DT_HOURS,
    E0_JAN1_KWH,
    E_MAX_KWH,
    E_MIN_KWH,
    ETA_CHARGE,
    ETA_DISCHARGE,
    FULL_DIR,
    MODEL_NAMES,
    OFFICIAL_END,
    OFFICIAL_START,
    P_MAX_KWH,
    PHASE1_DIR,
    PHASE1_END,
    REPO_ROOT,
    RESULT2_TEMPLATE_XLSX,
    RESULT_DIR,
    TERMINAL_LAMBDA,
    TERMINAL_TARGET_KWH,
    XGB_PARAMS,
)
from export_results import export_result2  # noqa: E402
from forecast import precompute_panel, predict_day  # noqa: E402
from leakage import run_leakage_suite  # noqa: E402
from load_data import audit_data, load_prices, load_year_actuals, write_clean  # noqa: E402
from model_lp import solve_day_lp, validate_plan  # noqa: E402
from simulate import simulate_day, validate_actual  # noqa: E402


def copy_raw_inputs() -> None:
    src_att = REPO_ROOT / "problem_files" / "附件"
    ATTACHMENT1_XLSX.parent.mkdir(parents=True, exist_ok=True)
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
) -> tuple[list[dict], float]:
    dates = year["dates"]
    fallback_load = prices["typical_load_kw"].to_numpy(dtype=float)
    fallback_pv = prices["typical_pv_kw"].to_numpy(dtype=float)
    cache: dict = {}
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
            }
        )
    return out, time.perf_counter() - t0


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
) -> dict:
    if forecasts is None:
        forecasts, forecast_s = collect_forecasts(
            model_name, prices, year, start_idx, end_idx, load_panel, pv_panel
        )
    price = prices["price"].to_numpy(dtype=float)
    soc_actual = E0_JAN1_KWH
    rows = []
    official_series: list[dict] = []
    t0 = time.perf_counter()
    n_simultaneous = 0
    all_errors: list[str] = []
    neg_pred = 0
    official_start = pd.Timestamp(OFFICIAL_START)

    for pred in forecasts:
        day = int(pred["day"])
        if (pred["load_kw"] < -1e-12).any() or (pred["pv_kw"] < -1e-12).any() or pred["neg_pred"]:
            neg_pred += 1
        load_hat = pred["load_kw"] * DT_HOURS
        pv_hat = pred["pv_kw"] * DT_HOURS
        plan = solve_day_lp(price, load_hat, pv_hat, soc_actual, terminal_mode=terminal_mode)
        all_errors.extend(validate_plan(plan, price, load_hat, pv_hat))
        n_simultaneous += int(plan["n_simultaneous"])
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
        if stamp >= official_start:
            official_series.append(
                {
                    "date": pred["date"],
                    "purchase_kwh": np.asarray(plan["purchase_kwh"], dtype=float),
                    "charge_kwh": np.asarray(actual["charge_kwh"], dtype=float),
                    "discharge_kwh": np.asarray(actual["discharge_kwh"], dtype=float),
                    "emergency_kwh": np.asarray(actual["emergency_kwh"], dtype=float),
                    "soc0_kwh": float(soc_actual),
                    "soc24_kwh": float(actual["soc24_kwh"]),
                    "plan_cost": float(actual["plan_cost"]),
                }
            )
        soc_actual = float(actual["soc24_kwh"])

    elapsed = time.perf_counter() - t0 + forecast_s
    daily = pd.DataFrame(rows)
    official = daily[daily["official"]].copy()
    summary = {
        "model": model_name,
        "terminal_mode": terminal_mode,
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
        if "2025-02-01" in set(daily["date"])
        else None,
        "last_soc24_actual": float(daily["soc24_actual"].iloc[-1]),
        "last_soc24_plan": float(daily["soc24_plan"].iloc[-1]),
        "mean_soc24_actual": float(official["soc24_actual"].mean()) if len(official) else None,
        "mean_soc24_plan": float(official["soc24_plan"].mean()) if len(official) else None,
        "soc_min_actual": float(daily["soc24_actual"].min()),
        "soc_max_actual": float(daily["soc24_actual"].max()),
        "terminal_lambda": TERMINAL_LAMBDA if terminal_mode == "track6000" else 0.0,
        "terminal_target_kwh": TERMINAL_TARGET_KWH if terminal_mode == "track6000" else None,
    }
    return {"daily": daily, "summary": summary, "official_series": official_series}


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
        "4. 2月1日初始SOC：由 2025-01-01 0:00 的 6000 kWh 起，按各模型自己的计划购电 + 实际因果回测滚动到 1月31日 24:00；"
        "不是题面直接给出的 2月1日初值，也不是每天重置 6000。",
        "5. 每日首尾 SOC 相等：问题2第一版**没有**强制 SOC[d,0]=SOC[d,24]=6000。"
        "仅 1月1日 0:00 为 6000 kWh；此后 SOC 跨日传递实际值，并夹在 1200–10800 kWh。"
        "这是对问题1日闭环的改口，题面只要求运行窗，不要求每天回到 6000。",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--skip-leakage", action="store_true")
    args = parser.parse_args()
    if not args.full and not args.phase1:
        args.phase1 = True
    copy_raw_inputs()
    prices = load_prices()
    year = load_year_actuals(
        jan1_load=float(prices["typical_load_kw"].iloc[0]),
        jan1_pv=float(prices["typical_pv_kw"].iloc[0]),
    )
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
    payloads = {}
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
        payloads[name] = result
        print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
        if result["summary"]["n_validation_errors"]:
            raise RuntimeError(f"{name} failed validation: {result['summary']['validation_errors_head']}")

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(out_root / "model_comparison.csv", index=False, encoding="utf-8-sig")
    if args.full:
        winner = min(summaries, key=lambda s: s["total_cost"])["model"]
        export_result2(payloads[winner]["official_series"], year["slot_end_min"], RESULT_DIR / "result2.xlsx")
        print("wrote", out_root / "model_comparison.csv", "winner", winner, flush=True)
    else:
        write_verification_report(audit, leakage, summaries, PHASE1_DIR / "verification_report.md")
        print("wrote", PHASE1_DIR / "verification_report.md", flush=True)


if __name__ == "__main__":
    main()
