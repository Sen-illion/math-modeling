"""Single-factor graft diagnostics on the start-aligned Q2 stack.

Mirrors the external fusion experiment's discipline: declare candidates first, mix
weekly-dow load with XGB at the point layer, compare split vs net-load quantiles,
score on 02-01..06-30 / 07-01..12-31. Does not overwrite result2.xlsx or opt/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from adaptive import (  # noqa: E402
    dumps,
    flatten_for_table,
    run_adaptive,
    run_fixed,
    summarise,
)
from bank import (  # noqa: E402
    bias_net_bank,
    check_length,
    ladder_banks,
    load_point_bank,
    mix_point,
    point_residual_history,
)
from config import (  # noqa: E402
    ADAPTIVE_DIR,
    ETA_DISCHARGE,
    FROZEN_OFFICIAL_FULL_YEAR_COST,
    GRAFT_DIR,
    OFFICIAL_END,
    OFFICIAL_START,
    OOS_START,
    Q_LADDER,
    TUNE_END,
    TUNE_INNER_END,
    TUNE_SELECT_START,
)
from run_q2 import _adaptive_inputs  # noqa: E402

WINDOWS = [
    ("tune", OFFICIAL_START, TUNE_END),
    ("inner", OFFICIAL_START, TUNE_INNER_END),
    ("select", TUNE_SELECT_START, TUNE_END),
    ("oos", OOS_START, OFFICIAL_END),
    ("full", OFFICIAL_START, OFFICIAL_END),
]

DECLARED = [
    {"id": "G0", "lambda": 1.0, "risk": "split", "adaptive": True, "note": "official D_pv7d_adaptive replay"},
    {"id": "G1", "lambda": 1.0, "risk": "split", "adaptive": False, "note": "fixed q=0.8, pure XGB"},
    {"id": "G2", "lambda": 0.5, "risk": "split", "adaptive": False, "note": "fixed q=0.8, 50% weekly + 50% XGB"},
    {"id": "G3a", "lambda": 0.3, "risk": "split", "adaptive": False, "note": "fixed q=0.8, 30% XGB"},
    {"id": "G3b", "lambda": 0.7, "risk": "split", "adaptive": False, "note": "fixed q=0.8, 70% XGB"},
    {"id": "G4", "lambda": 1.0, "risk": "net", "adaptive": False, "note": "pure XGB, net-load alpha=0.8"},
    {"id": "G5", "lambda": 0.5, "risk": "net", "adaptive": False, "note": "50% mix, net-load alpha=0.8"},
]


def _rule() -> dict:
    path = ADAPTIVE_DIR / "selected_rule.json"
    if not path.exists():
        raise SystemExit("missing results/Q2/adaptive/selected_rule.json")
    return json.loads(path.read_text(encoding="utf-8"))


def point_mae(point: list[dict], year: dict, start: str, end: str) -> dict:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    load_err = []
    weekly_err = []
    pv_err = []
    for pred in point:
        stamp = pd.Timestamp(pred["date"])
        if stamp < lo or stamp > hi:
            continue
        day = int(pred["day"])
        load_err.append(np.asarray(pred["load_kw"], dtype=float) - year["load_kw"][day])
        weekly_err.append(np.asarray(pred["load_weekly_kw"], dtype=float) - year["load_kw"][day])
        pv_err.append(np.asarray(pred["pv_kw"], dtype=float) - year["pv_kw"][day])
    load_err = np.vstack(load_err)
    weekly_err = np.vstack(weekly_err)
    pv_err = np.vstack(pv_err)
    return {
        "n_days": int(len(load_err)),
        "load_mae_kw": float(np.mean(np.abs(load_err))),
        "load_rmse_kw": float(np.sqrt(np.mean(load_err**2))),
        "weekly_mae_kw": float(np.mean(np.abs(weekly_err))),
        "pv_mae_kw": float(np.mean(np.abs(pv_err))),
    }


def inventory_adjusted(daily: pd.DataFrame, start: str, end: str, min_price: float) -> float:
    stamps = pd.to_datetime(daily["date"])
    win = daily[(stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))]
    cash = float(win["total_cost"].sum())
    if win.empty:
        return cash
    delta = float(win["soc0_actual"].iloc[0] - win["soc24_actual"].iloc[-1])
    return cash + float(min_price) / ETA_DISCHARGE * delta


def banks_for(point: list[dict], year: dict, spec: dict) -> tuple[dict[float, list[dict]], dict[int, dict]]:
    mixed = mix_point(point, spec["lambda"])
    residuals = point_residual_history(mixed, year)
    if spec["risk"] == "net":
        biased = bias_net_bank(mixed, year, 0.8)
        return {0.8: biased}, residuals
    return ladder_banks(mixed, year, Q_LADDER), residuals


def run_spec(prices, year, point, spec: dict, rule: dict, collect_traces: bool = False) -> dict:
    banks, residuals = banks_for(point, year, spec)
    if spec["adaptive"]:
        result = run_adaptive(
            prices,
            year,
            banks,
            residuals,
            OFFICIAL_END,
            feature=rule["feature"],
            q_min=float(rule["q_min"]),
            k=float(rule["k"]),
            label=spec["id"],
            collect_traces=collect_traces,
        )
    else:
        result = run_fixed(prices, year, banks, 0.8, OFFICIAL_END, collect_traces=collect_traces)
        result["rule"]["label"] = spec["id"]
    if result["errors"]:
        raise RuntimeError(f"{spec['id']} validation: {result['errors'][:5]}")
    summary = summarise(result, WINDOWS)
    mae = point_mae(mix_point(point, spec["lambda"]), year, OFFICIAL_START, OFFICIAL_END)
    summary["point_mae"] = mae
    summary["spec"] = spec
    min_price = float(prices["price"].min())
    for name, start, end in WINDOWS:
        summary["windows"][name]["inventory_adjusted"] = inventory_adjusted(
            result["daily"], start, end, min_price
        )
    result["summary"] = summary
    return result


def phase1_row(summary: dict) -> dict:
    spec = summary["spec"]
    row = {
        "id": spec["id"],
        "lambda": spec["lambda"],
        "risk": spec["risk"],
        "adaptive": spec["adaptive"],
        "note": spec["note"],
        "load_mae_kw": summary["point_mae"]["load_mae_kw"],
        "weekly_mae_kw": summary["point_mae"]["weekly_mae_kw"],
        "n_validation_errors": summary["n_validation_errors"],
    }
    for name in ("tune", "inner", "select", "oos", "full"):
        win = summary["windows"][name]
        row[f"{name}_total_cost"] = win["total_cost"]
        row[f"{name}_plan_cost"] = win["plan_cost"]
        row[f"{name}_emergency_cost"] = win["emergency_cost"]
        row[f"{name}_emergency_kwh"] = win["emergency_kwh"]
        row[f"{name}_emergency_days"] = win["emergency_days"]
        row[f"{name}_curtail_kwh"] = win["curtail_kwh"]
        row[f"{name}_purchase_kwh"] = win["purchase_kwh"]
        row[f"{name}_inventory_adjusted"] = win["inventory_adjusted"]
    row["full_vs_official"] = summary["windows"]["full"]["total_cost"] - FROZEN_OFFICIAL_FULL_YEAR_COST
    return row


def phase1(prices, year, point, rule: dict) -> tuple[pd.DataFrame, dict]:
    GRAFT_DIR.mkdir(parents=True, exist_ok=True)
    declared = {
        "candidates": DECLARED,
        "windows": WINDOWS,
        "note": (
            "Declared before the year replay. Mix is weekly-dow x XGB at the point layer. "
            "Net risk uses one residual on L-P. Does not overwrite result2.xlsx."
        ),
        "official_cost": FROZEN_OFFICIAL_FULL_YEAR_COST,
    }
    (GRAFT_DIR / "declared.json").write_text(dumps(declared), encoding="utf-8")

    rows = []
    summaries = {}
    for spec in DECLARED:
        print("graft phase1", spec["id"], spec["note"], flush=True)
        result = run_spec(prices, year, point, spec, rule)
        summary = result["summary"]
        summaries[spec["id"]] = summary
        rows.append(phase1_row(summary))
        result["daily"].to_csv(GRAFT_DIR / f"daily_{spec['id']}.csv", index=False, encoding="utf-8-sig")
        print(
            f"  full={summary['windows']['full']['total_cost']:.2f} "
            f"oos={summary['windows']['oos']['total_cost']:.2f} "
            f"mae={summary['point_mae']['load_mae_kw']:.2f}",
            flush=True,
        )

    g0 = summaries["G0"]["windows"]["full"]["total_cost"]
    if abs(g0 - FROZEN_OFFICIAL_FULL_YEAR_COST) > 1e-6:
        raise RuntimeError(f"G0 {g0} != official {FROZEN_OFFICIAL_FULL_YEAR_COST}")

    frame = pd.DataFrame(rows)
    frame.to_csv(GRAFT_DIR / "phase1_comparison.csv", index=False, encoding="utf-8-sig")
    (GRAFT_DIR / "phase1_summaries.json").write_text(dumps(summaries), encoding="utf-8")
    return frame, summaries


def oos_winners(summaries: dict) -> list[str]:
    g1 = summaries["G1"]["windows"]["oos"]["total_cost"]
    ids = []
    for key in ("G2", "G3a", "G3b", "G4", "G5"):
        if summaries[key]["windows"]["oos"]["total_cost"] < g1:
            ids.append(key)
    return ids


def phase2_pool(summaries: dict, winners: list[str]) -> list[dict]:
    pool = [
        {"id": "P_official", "lambda": 1.0, "risk": "split", "adaptive": True, "note": "D_pv7d_adaptive"},
        {"id": "P_G1", "lambda": 1.0, "risk": "split", "adaptive": False, "note": "fixed q=0.8 pure XGB"},
    ]
    mix_winners = [w for w in winners if w in ("G2", "G3a", "G3b")]
    net_winners = [w for w in winners if w in ("G4", "G5")]
    by_oos = sorted(mix_winners, key=lambda k: summaries[k]["windows"]["oos"]["total_cost"])
    if by_oos:
        best = summaries[by_oos[0]]["spec"]
        pool.append(
            {
                "id": "P_best_mix_fixed",
                "lambda": best["lambda"],
                "risk": "split",
                "adaptive": False,
                "note": f"best mix OOS {by_oos[0]}",
            }
        )
        pool.append(
            {
                "id": "P_best_mix_adaptive",
                "lambda": best["lambda"],
                "risk": "split",
                "adaptive": True,
                "note": f"adaptive q on {by_oos[0]} lambda, no re-tune",
            }
        )
    if net_winners:
        best_net = min(net_winners, key=lambda k: summaries[k]["windows"]["oos"]["total_cost"])
        spec = summaries[best_net]["spec"]
        pool.append(
            {
                "id": "P_best_net_fixed",
                "lambda": spec["lambda"],
                "risk": "net",
                "adaptive": False,
                "note": f"best net OOS {best_net}",
            }
        )
    # Deduplicate by (lambda, risk, adaptive)
    seen = set()
    uniq = []
    for spec in pool:
        key = (spec["lambda"], spec["risk"], spec["adaptive"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(spec)
    return uniq[:6]


def phase2(prices, year, point, rule: dict, summaries: dict, winners: list[str]) -> dict:
    pool = phase2_pool(summaries, winners)
    (GRAFT_DIR / "phase2_pool.json").write_text(dumps({"winners": winners, "pool": pool}), encoding="utf-8")
    rows = []
    picked = []
    for spec in pool:
        print("graft phase2", spec["id"], spec["note"], flush=True)
        result = run_spec(prices, year, point, spec, rule)
        summary = result["summary"]
        picked.append(summary)
        rows.append(phase1_row(summary))

    frame = pd.DataFrame(rows)
    frame.to_csv(GRAFT_DIR / "phase2_comparison.csv", index=False, encoding="utf-8-sig")

    official = next(s for s in picked if s["spec"]["id"] == "P_official")
    official_inner = official["windows"]["inner"]["total_cost"]
    adaptive_or_fixed = [s for s in picked if s["spec"]["id"] != "P_official"]
    eligible = [
        s
        for s in adaptive_or_fixed
        if s["windows"]["inner"]["inventory_adjusted"] < official["windows"]["inner"]["inventory_adjusted"]
    ]
    pool_sel = eligible or adaptive_or_fixed
    best = min(pool_sel, key=lambda s: s["windows"]["select"]["inventory_adjusted"])
    oos_best = best["windows"]["oos"]["total_cost"]
    oos_off = official["windows"]["oos"]["total_cost"]
    full_best = best["windows"]["full"]["total_cost"]
    gate = {
        "beats_official_full_year": bool(full_best < FROZEN_OFFICIAL_FULL_YEAR_COST),
        "beats_official_oos": bool(oos_best < oos_off),
        "clean_validation": bool(best["n_validation_errors"] == 0),
        "n_eligible_after_inner": len(eligible),
        "fell_back_to_full_pool": not eligible,
        "selected": best["spec"],
        "selected_full": full_best,
        "selected_oos": oos_best,
        "official_full": official["windows"]["full"]["total_cost"],
        "official_oos": oos_off,
    }
    gate["passed"] = bool(
        gate["beats_official_full_year"] and gate["beats_official_oos"] and gate["clean_validation"]
    )
    (GRAFT_DIR / "phase2_gate.json").write_text(dumps(gate), encoding="utf-8")
    print("phase2 gate", dumps(gate), flush=True)
    return gate


def write_summary(frame: pd.DataFrame, summaries: dict, winners: list[str], gate: dict | None) -> None:
    g0 = summaries["G0"]["windows"]["full"]["total_cost"]
    g1 = summaries["G1"]
    lines = [
        "# Q2 反向融合诊断",
        "",
        "对照：正式 `D_pv7d_adaptive` **13697499.17** 元。时刻口径、贪心、光伏 7 日均值、锁死 $G_t$ 均未改。",
        "本目录只做诊断，**未覆盖** `results/Q2/result2.xlsx` 与 `opt/`。",
        "",
        "复现：`python code/Q2/graft_q2.py`",
        "",
        "## 第 0 步：周周期负荷",
        "",
        "因果同星期同刻均值（不足 2 个同星期日则退回 7 日均值），与 XGB 点预测一起写入 `forecast_bank.npz`（格式 2）。",
        f"正式窗周周期负荷 MAE = **{g1['point_mae']['weekly_mae_kw']:.2f}** kW，纯 XGB MAE = **{g1['point_mae']['load_mae_kw']:.2f}** kW。",
        "",
        "## 第 1 步：单因子",
        "",
        f"G0 全年费用 {g0:.10f}，与冻结值差 {g0 - FROZEN_OFFICIAL_FULL_YEAR_COST}（须为 0）。",
        "",
        "| id | λ | 风险 | 自适应 | 负荷MAE | 调参窗 | 样本外 | 全年 | 全年相对正式 | 样本外紧急天数 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in frame.iterrows():
        lines.append(
            f"| {r['id']} | {r['lambda']:.1f} | {r['risk']} | {bool(r['adaptive'])} | "
            f"{r['load_mae_kw']:.2f} | {r['tune_total_cost']:.2f} | {r['oos_total_cost']:.2f} | "
            f"{r['full_total_cost']:.2f} | {r['full_vs_official']:.2f} | {int(r['oos_emergency_days'])} |"
        )

    g1_oos = g1["windows"]["oos"]["total_cost"]
    g1_full = g1["windows"]["full"]["total_cost"]
    g1_mae = g1["point_mae"]["load_mae_kw"]
    mix_rows = frame[frame["id"].isin(["G1", "G2", "G3a", "G3b"])]
    mae_cost_reverse = False
    if len(mix_rows) >= 2:
        cheapest = mix_rows.loc[mix_rows["full_total_cost"].idxmin()]
        lowest_mae = mix_rows.loc[mix_rows["load_mae_kw"].idxmin()]
        mae_cost_reverse = cheapest["id"] != lowest_mae["id"]

    net_vs_split = summaries["G4"]["windows"]["full"]["total_cost"] - g1_full
    mix_vs_xgb_oos = summaries["G2"]["windows"]["oos"]["total_cost"] - g1_oos

    lines += [
        "",
        "## 结论",
        "",
        f"- **同星期混合在起点对齐数据上：** G2 样本外相对 G1 为 **{mix_vs_xgb_oos:.2f}** 元。"
        + (
            "混合在样本外更便宜，对方的零件在我们口径上仍然成立。"
            if mix_vs_xgb_oos < 0
            else "混合在样本外并不更便宜，对方 50% 融合的优势在我们口径上没有复现。"
        ),
        f"- **MAE 与费用是否反向：** {'是' if mae_cost_reverse else '否'}。"
        f" 混合档里全年最便宜的是 `{mix_rows.loc[mix_rows['full_total_cost'].idxmin()]['id']}`，"
        f" MAE 最低的是 `{mix_rows.loc[mix_rows['load_mae_kw'].idxmin()]['id']}`。",
        f"- **净负荷分位相对分量 0.8/0.2：** G4 全年相对 G1 为 **{net_vs_split:.2f}** 元。"
        + ("净负荷分位更好。" if net_vs_split < 0 else "分量分位仍然更好或持平。"),
        f"- 第 1 步样本外相对 G1 的赢家：{winners if winners else '无'}。",
    ]
    if gate is None:
        lines.append("- 第 2 步未开：没有混合/净分位档在样本外胜过 G1，不做月度切换，也未改冻结数字。")
    else:
        lines.append(
            f"- 第 2 步选中 `{gate['selected']['id']}`，全年 {gate['selected_full']:.2f}，"
            f" 样本外 {gate['selected_oos']:.2f}，门禁 {'通过' if gate['passed'] else '未通过'}。"
            " 即使通过也只暂存在本目录，**未覆盖正式 result2.xlsx**。"
        )
    lines += [
        "",
        "**冻结数字仍是 13697499.17 元。**",
        "",
    ]
    (GRAFT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    GRAFT_DIR.mkdir(parents=True, exist_ok=True)
    rule = _rule()
    prices, year, point, _banks, _residuals = _adaptive_inputs()
    check_length(point)
    if "load_weekly_kw" not in point[0]:
        raise SystemExit("forecast bank missing load_weekly_kw; BANK_FORMAT should be 2")

    frame, summaries = phase1(prices, year, point, rule)
    winners = oos_winners(summaries)
    print("phase1 OOS winners vs G1:", winners, flush=True)
    gate = None
    if winners:
        gate = phase2(prices, year, point, rule, summaries, winners)
    else:
        (GRAFT_DIR / "phase2_skipped.json").write_text(
            dumps({"reason": "no mix/net config beat G1 out of sample", "winners": []}),
            encoding="utf-8",
        )
    write_summary(frame, summaries, winners, gate)
    print("wrote", GRAFT_DIR / "summary.md", flush=True)


if __name__ == "__main__":
    main()
