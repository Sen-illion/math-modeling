"""Align Q4-3 LA with the current Q3 quantile buffer.

hat0 stays locked all day. Settlement uses attachment 4. Default mode writes
diagnostics only. Pass --promote LA_h48_q65 to replace result4-3.xlsx after
the nested windows have already been inspected. Never writes result2.xlsx,
result3.xlsx, or result4-2.xlsx.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

Q4_DIR = Path(__file__).resolve().parent
CODE_DIR = Q4_DIR.parent
Q3_DIR = CODE_DIR / "Q3"
REPO_ROOT = CODE_DIR.parent
if str(Q3_DIR) not in sys.path:
    sys.path.insert(0, str(Q3_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from quantile import QuantileBank  # noqa: E402
from run_q3 import load_bundle  # noqa: E402

from Q4.config import (  # noqa: E402
    ALIGN_QUANTILE_DIAG_DIR,
    OFFICIAL_END,
    OFFICIAL_START,
    OOS_START,
    RESULT4_2_XLSX,
    RESULT4_3_XLSX,
    RESULT_DIR,
    RETUNE_DIAG_DIR,
    TUNE_END,
    TUNE_INNER_END,
    TUNE_SELECT_START,
)
from Q4.export_results import (  # noqa: E402
    export_paper_q43,
    export_result4_3,
    merge_metrics,
    write_json,
    write_run_manifest,
)
from Q4.price_bank import load_price_bank  # noqa: E402
from Q4.run_q4_3 import _window_slice, run_policy_q43  # noqa: E402

Q2_XLSX = REPO_ROOT / "results" / "Q2" / "result2.xlsx"
Q3_XLSX = REPO_ROOT / "results" / "Q3" / "result3.xlsx"
EPS = 1e-6

CANDIDATES = (
    {
        "label": "LA_h48_q65",
        "lookahead_hours": 48,
        "q_lock": 0.65,
        "q_open": 0.50,
        "q_evening": 0.50,
    },
    {
        "label": "LA_h24_q65",
        "lookahead_hours": 24,
        "q_lock": 0.65,
        "q_open": 0.50,
        "q_evening": 0.50,
    },
)

WINDOWS = (
    ("inner", OFFICIAL_START, TUNE_INNER_END),
    ("select", TUNE_SELECT_START, TUNE_END),
    ("tune", OFFICIAL_START, TUNE_END),
    ("oos", OOS_START, OFFICIAL_END),
    ("full", OFFICIAL_START, OFFICIAL_END),
)


def _stamp(path: Path) -> tuple[int, float] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime


def _assert_untouched(path: Path, before: tuple[int, float] | None, label: str) -> None:
    after = _stamp(path)
    if before != after:
        raise RuntimeError(f"{label} changed unexpectedly: {path}")


def _cheaper(candidate: float, baseline: float) -> bool:
    return float(candidate) < float(baseline) - EPS


def _load_current_official() -> dict:
    selected_path = RETUNE_DIAG_DIR / "selected.json"
    metrics_path = RESULT_DIR / "metrics.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    current = dict(selected["q4_3"]["current"])
    la = metrics["q4_3"]["LA"]
    current["full_cost"] = float(la["total_cost"])
    current["full_emergency_kwh"] = float(la["emergency_kwh"])
    current["full_plan_cost"] = float(la["plan_only_cost"])
    current["emergency_cost"] = float(la["emergency_cost"])
    current["lookahead_hours"] = 24
    current["q_lock"] = None
    current["q_open"] = None
    current["q_evening"] = None
    current["buffer"] = "beta_sigma"
    tune_em = float(current.get("tune_emergency_kwh", 0.0))
    current["oos_emergency_kwh"] = float(current["full_emergency_kwh"]) - tune_em
    current["oos_cost"] = float(current["full_cost"]) - float(current["tune_cost"])
    return current


def _row_from_payload(label: str, cand: dict, payload: dict) -> dict:
    daily = payload["daily"]
    summary = payload["summary"]
    row = {
        "label": label,
        "policy": "LA",
        "lookahead_hours": int(cand["lookahead_hours"]),
        "q_lock": float(cand["q_lock"]),
        "q_open": float(cand["q_open"]),
        "q_evening": float(cand["q_evening"]),
        "buffer": "net_load_quantile",
        "elapsed_s": float(summary["elapsed_s"]),
        "full_plan_cost": float(summary["plan_only_cost"]),
        "full_emergency_cost": float(summary["emergency_cost"]),
    }
    for name, start, end in WINDOWS:
        win = _window_slice(daily, start, end)
        row[f"{name}_cost"] = float(win["total_cost"].sum()) if len(win) else 0.0
        row[f"{name}_emergency_kwh"] = float(win["emergency_kwh"].sum()) if len(win) else 0.0
        row[f"{name}_n_days"] = int(len(win))
    return row


def _delta_block(row: dict, current: dict) -> dict:
    out = {}
    for key in ("inner_cost", "select_cost", "tune_cost", "oos_cost", "full_cost", "full_emergency_kwh"):
        cand = float(row[key])
        base = float(current[key])
        out[key] = cand
        out[f"{key}_delta"] = cand - base
        out[f"{key}_cheaper"] = _cheaper(cand, base)
    out["beats_tune_and_oos"] = bool(out["tune_cost_cheaper"] and out["oos_cost_cheaper"])
    return out


def _fmt(value: float) -> str:
    return f"{float(value):,.2f}"


def _write_summary_md(out_dir: Path, current: dict, rows: list[dict], comparison: dict) -> None:
    promoted = bool(comparison.get("promote"))
    title = "# Q4-3 对齐问题 3 分位数"
    status = (
        f"已晋升 `{comparison.get('promoted_label')}` 为 `result4-3.xlsx`。未改 result2 / result3 / result4-2。"
        if promoted
        else "hat0 全天锁定，结算用附件 4。6/12/18 只更新光伏/负荷。未改 result2 / result3 / result4-2 / result4-3。"
    )
    lines = [
        title,
        "",
        status,
        "",
        "48 h 前瞻把**当天 hat0** 按 144 槽取模铺到次日（`_horizon_price`）。附件 1 上每天同价无害；附件 4 上次日真价不同，因此 24 h 与 48 h 都跑。",
        "",
        "对照窗：inner 2025-02-01–04-30，select 05-01–06-30，tune 02-01–06-30，OOS 07-01–12-31。",
        "",
        f"对照基线：`{current['label']}`，24 h + $\\beta_{{\\mathrm{{lock}}}}=1.0$ / $\\beta_{{\\mathrm{{open}}}}=0.2$，无分位数。",
        "",
        "| 规则 | 前瞻 | 缓冲 | tune | OOS | 全年 | 全年紧急 kWh | tune 更省 | OOS 更省 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        (
            f"| 基线 {current['label']} | 24 | $\\beta=1.0/0.2$ | "
            f"{_fmt(current['tune_cost'])} | {_fmt(current['oos_cost'])} | "
            f"{_fmt(current['full_cost'])} | {_fmt(current['full_emergency_kwh'])} | — | — |"
        ),
    ]
    for row in rows:
        block = comparison["candidates"][row["label"]]
        lines.append(
            f"| {row['label']} | {int(row['lookahead_hours'])} | "
            f"$q_L=0.65,\\ q_O=0.50,\\ q_E=0.50$ | "
            f"{_fmt(row['tune_cost'])} | {_fmt(row['oos_cost'])} | "
            f"{_fmt(row['full_cost'])} | {_fmt(row['full_emergency_kwh'])} | "
            f"{'是' if block['tune_cost_cheaper'] else '否'} | "
            f"{'是' if block['oos_cost_cheaper'] else '否'} |"
        )
    lines.extend(
        [
            "",
            (
                f"正式表已换成 `{comparison.get('promoted_label')}`。"
                if promoted
                else "本轮**不替换** `result4-3.xlsx`。是否晋升由人工确认。"
            ),
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_candidate(cand: dict, bundle: dict, bank: dict, quantile_bank) -> dict:
    return run_policy_q43(
        "LA",
        OFFICIAL_END,
        plan_source="hat0",
        bundle=bundle,
        bank=bank,
        lookahead_hours=int(cand["lookahead_hours"]),
        q_lock=float(cand["q_lock"]),
        q_open=float(cand["q_open"]),
        q_evening=float(cand["q_evening"]),
        quantile_bank=quantile_bank,
    )


def promote(label: str) -> None:
    cand = next((c for c in CANDIDATES if c["label"] == label), None)
    if cand is None:
        raise SystemExit(f"unknown promote label {label}")
    before = {
        "result4-2": _stamp(RESULT4_2_XLSX),
        "result2": _stamp(Q2_XLSX),
        "result3": _stamp(Q3_XLSX),
    }
    out_dir = ALIGN_QUANTILE_DIAG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if RESULT4_3_XLSX.exists():
        shutil.copy2(RESULT4_3_XLSX, out_dir / "result4-3_before.xlsx")

    expected_path = out_dir / "comparison.json"
    expected_cost = None
    if expected_path.exists():
        expected_cost = float(json.loads(expected_path.read_text(encoding="utf-8"))["candidates"][label]["full_cost"])

    bundle = load_bundle(write=False)
    bank = load_price_bank()
    print("building QuantileBank", flush=True)
    quantile_bank = QuantileBank.build(bundle)
    print("promoting", label, flush=True)
    payload = _run_candidate(cand, bundle, bank, quantile_bank)
    new_cost = float(payload["summary"]["total_cost"])
    if expected_cost is not None and abs(new_cost - expected_cost) > 1e-4:
        raise RuntimeError(f"{label} cost {new_cost} != diagnostic {expected_cost}")

    dest = export_result4_3(payload["official"], payload["end_min"])
    tables = export_paper_q43(payload["official"], payload["end_min"])
    payload["daily"].to_csv(RESULT_DIR / "q43_LA_daily.csv", index=False, encoding="utf-8-sig")
    write_json(RESULT_DIR / "q43_LA_summary.json", payload["summary"])
    q43_metrics = {}
    metrics_path = RESULT_DIR / "metrics.json"
    if metrics_path.exists():
        q43_metrics = json.loads(metrics_path.read_text(encoding="utf-8")).get("q4_3", {})
    q43_metrics["LA"] = payload["summary"]
    q43_metrics["official_policy"] = "LA"
    merge_metrics(q43_metrics=q43_metrics)
    q43_file = RESULT_DIR / "q43_metrics.json"
    existing_q43 = json.loads(q43_file.read_text(encoding="utf-8")) if q43_file.exists() else {}
    existing_q43["LA"] = payload["summary"]
    existing_q43["official_policy"] = "LA"
    write_json(q43_file, existing_q43)
    write_run_manifest(
        {
            "q4_3": {
                "official_policy": "LA",
                "label": label,
                "lookahead_hours": int(cand["lookahead_hours"]),
                "q_lock": float(cand["q_lock"]),
                "q_open": float(cand["q_open"]),
                "q_evening": float(cand["q_evening"]),
                "metrics": q43_metrics,
                "result_xlsx": str(dest),
                "specified_tables": [str(p) for p in tables],
                "aligned_from": str(out_dir),
            }
        }
    )
    if expected_path.exists():
        comparison = json.loads(expected_path.read_text(encoding="utf-8"))
        comparison["promote"] = True
        comparison["promoted_label"] = label
        comparison["note"] = f"promoted {label} to result4-3.xlsx; result4-2/result2/result3 untouched"
        write_json(expected_path, comparison)
        rows = [comparison["candidates"][c["label"]] for c in CANDIDATES if c["label"] in comparison["candidates"]]
        _write_summary_md(out_dir, comparison["current"], rows, comparison)
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "promote": True,
            "promoted_label": label,
            "wrote_result4_3": True,
            "result_xlsx": str(dest),
            "total_cost": new_cost,
        },
    )

    _assert_untouched(RESULT4_2_XLSX, before["result4-2"], "result4-2.xlsx")
    _assert_untouched(Q2_XLSX, before["result2"], "result2.xlsx")
    _assert_untouched(Q3_XLSX, before["result3"], "result3.xlsx")
    print("promoted", dest, new_cost, flush=True)


def run_compare() -> None:
    before = {
        "result4-2": _stamp(RESULT4_2_XLSX),
        "result4-3": _stamp(RESULT4_3_XLSX),
        "result2": _stamp(Q2_XLSX),
        "result3": _stamp(Q3_XLSX),
    }
    out_dir = ALIGN_QUANTILE_DIAG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    current = _load_current_official()
    print("current official", current["label"], current["full_cost"], flush=True)

    bundle = load_bundle(write=False)
    bank = load_price_bank()
    print("building QuantileBank", flush=True)
    quantile_bank = QuantileBank.build(bundle)

    rows = []
    summaries = {}
    for cand in CANDIDATES:
        label = cand["label"]
        print("running", label, flush=True)
        payload = _run_candidate(cand, bundle, bank, quantile_bank)
        daily = payload["daily"]
        daily.assign(label=label).to_csv(out_dir / f"q43_{label}_daily.csv", index=False, encoding="utf-8-sig")
        write_json(out_dir / f"q43_{label}_summary.json", payload["summary"])
        row = _row_from_payload(label, cand, payload)
        rows.append(row)
        summaries[label] = payload["summary"]
        print(label, row["full_cost"], "emergency", row["full_emergency_kwh"], flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "q43_grid.csv", index=False, encoding="utf-8-sig")

    comparison = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "promote": False,
        "note": "diagnostic only; result4-3.xlsx was not rewritten",
        "plan_price": "hat0_locked_all_day",
        "settle_price": "attachment4_actual",
        "horizon_price": "wrap_today_hat0_onto_tomorrow",
        "current": current,
        "windows": {name: [start, end] for name, start, end in WINDOWS},
        "candidates": {row["label"]: {**row, **_delta_block(row, current)} for row in rows},
    }
    write_json(out_dir / "comparison.json", comparison)
    _write_summary_md(out_dir, current, rows, comparison)
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "promote": False,
            "wrote_result4_3": False,
            "candidates": [c["label"] for c in CANDIDATES],
            "out_dir": str(out_dir),
            "summaries": summaries,
        },
    )

    _assert_untouched(RESULT4_2_XLSX, before["result4-2"], "result4-2.xlsx")
    _assert_untouched(RESULT4_3_XLSX, before["result4-3"], "result4-3.xlsx")
    _assert_untouched(Q2_XLSX, before["result2"], "result2.xlsx")
    _assert_untouched(Q3_XLSX, before["result3"], "result3.xlsx")
    print("wrote", out_dir / "comparison.json", flush=True)
    print(json.dumps({"promote": False, "candidates": list(comparison["candidates"])}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", default=None, help="promote a candidate label to result4-3.xlsx")
    args = parser.parse_args()
    if args.promote:
        promote(args.promote)
        return
    run_compare()


if __name__ == "__main__":
    main()
