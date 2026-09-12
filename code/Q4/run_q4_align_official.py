"""Align Q4-2/Q4-3 to the current Q2/Q3 official methods. Diagnostics only.

Q4-2 candidate: E_pv7d_netrho (alpha=0.8, S*=2400, rho=0.625) with hat0 planning
and attachment-4 settlement. Q4-3 candidate: LA/N0/M0L with 48 h look-ahead and
January-calibrated q_lock=0.60. Does not write result2/result3/result4-x.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

Q4_DIR = Path(__file__).resolve().parent
CODE_DIR = Q4_DIR.parent
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from Q4.config import (  # noqa: E402
    ALIGN_OFFICIAL_DIAG_DIR,
    OFFICIAL_END,
    OFFICIAL_START,
    OOS_START,
    RESULT4_2_XLSX,
    RESULT4_3_XLSX,
    RESULT_DIR,
    TUNE_END,
)
from Q4.export_results import write_json  # noqa: E402

Q2_XLSX = REPO_ROOT / "results" / "Q2" / "result2.xlsx"
Q3_XLSX = REPO_ROOT / "results" / "Q3" / "result3.xlsx"
EPS = 1e-6
WINDOWS = (
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


def _run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(Q4_DIR / script), *extra]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _window_slice(daily: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    stamps = pd.to_datetime(daily["date"])
    mask = (stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))
    return daily.loc[mask]


def _cost_block(daily: pd.DataFrame) -> dict:
    out = {}
    for name, start, end in WINDOWS:
        win = _window_slice(daily, start, end)
        out[f"{name}_cost"] = float(win["total_cost"].sum()) if len(win) else 0.0
        em_col = "emergency_kwh" if "emergency_kwh" in win.columns else None
        out[f"{name}_emergency_kwh"] = float(win[em_col].sum()) if em_col and len(win) else 0.0
        out[f"{name}_n_days"] = int(len(win))
    return out


def _fmt(value: float) -> str:
    return f"{float(value):,.2f}"


def _delta(cand: float, base: float) -> dict:
    return {
        "value": float(cand),
        "delta": float(cand) - float(base),
        "cheaper": float(cand) < float(base) - EPS,
    }


def main() -> None:
    before = {
        "result4-2": _stamp(RESULT4_2_XLSX),
        "result4-3": _stamp(RESULT4_3_XLSX),
        "result2": _stamp(Q2_XLSX),
        "result3": _stamp(Q3_XLSX),
    }
    out_dir = ALIGN_OFFICIAL_DIAG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    _run(
        "run_q4_2.py",
        [
            "--full",
            "--plan-source",
            "hat0",
            "--margin",
            "netrho",
            "--out-dir",
            str(out_dir),
        ],
    )
    _run(
        "run_q4_3.py",
        [
            "--full",
            "--plan-source",
            "hat0",
            "--policies",
            "N0,LA,M0L",
            "--lookahead-hours",
            "48",
            "--q-lock",
            "0.60",
            "--q-open",
            "0.50",
            "--q-evening",
            "0.50",
            "--out-dir",
            str(out_dir),
        ],
    )

    metrics = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
    q42_current_daily = pd.read_csv(RESULT_DIR / "q42_daily.csv")
    q43_current_daily = pd.read_csv(RESULT_DIR / "q43_LA_daily.csv")
    q42_new_daily = pd.read_csv(out_dir / "q42_netrho_daily.csv")
    q43_new = {
        name: pd.read_csv(out_dir / f"q43_{name}_daily.csv") for name in ("N0", "LA", "M0L")
    }
    q42_new_summary = json.loads((out_dir / "q42_netrho_summary.json").read_text(encoding="utf-8"))
    q43_new_metrics = json.loads((out_dir / "q43_metrics.json").read_text(encoding="utf-8"))

    q42_cur = _cost_block(q42_current_daily)
    q42_cur["full_cost"] = float(metrics["q4_2"]["total_cost"])
    q42_cur["full_emergency_kwh"] = float(metrics["q4_2"]["emergency_kwh"])
    q42_new = _cost_block(q42_new_daily)
    q43_cur = _cost_block(q43_current_daily)
    q43_cur["full_cost"] = float(metrics["q4_3"]["LA"]["total_cost"])
    q43_cur["full_emergency_kwh"] = float(metrics["q4_3"]["LA"]["emergency_kwh"])
    q43_la = _cost_block(q43_new["LA"])

    comparison = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "promote": False,
        "note": "diagnostic only; result4-2.xlsx and result4-3.xlsx were not rewritten",
        "plan_price": "hat0_locked",
        "settle_price": "attachment4_actual",
        "copied_from": {
            "q2": "E_pv7d_netrho alpha=0.8 S*=2400 rho=0.625",
            "q3": "LA 48h q_lock=0.60 q_open=0.50 q_evening=0.50",
        },
        "q4_2": {
            "current": {
                "policy": "D_pv7d_adaptive",
                **q42_cur,
            },
            "candidate": {
                "policy": "E_pv7d_netrho",
                **q42_new,
                "summary": q42_new_summary,
            },
            "tune": _delta(q42_new["tune_cost"], q42_cur["tune_cost"]),
            "oos": _delta(q42_new["oos_cost"], q42_cur["oos_cost"]),
            "full": _delta(q42_new["full_cost"], q42_cur["full_cost"]),
        },
        "q4_3": {
            "current": {
                "policy": "LA",
                "q_lock": 0.65,
                "lookahead_hours": 48,
                **q43_cur,
            },
            "candidate_LA": {
                "policy": "LA",
                "q_lock": 0.60,
                "lookahead_hours": 48,
                **q43_la,
                "summary": q43_new_metrics.get("LA"),
            },
            "controls": {
                name: {**_cost_block(frame), "summary": q43_new_metrics.get(name)}
                for name, frame in q43_new.items()
                if name != "LA"
            },
            "tune": _delta(q43_la["tune_cost"], q43_cur["tune_cost"]),
            "oos": _delta(q43_la["oos_cost"], q43_cur["oos_cost"]),
            "full": _delta(q43_la["full_cost"], q43_cur["full_cost"]),
        },
    }
    write_json(out_dir / "comparison.json", comparison)

    lines = [
        "# Q4 对齐当前问题 2 / 问题 3 正式方法（诊断，未换正式表）",
        "",
        "hat0 决策、附件 4 结算。参数原样搬运附件 1 世界冻结值，未在 2–12 月重选。未改 result2 / result3 / result4-2 / result4-3。",
        "",
        "## Q4-2",
        "",
        "| 规则 | tune | OOS | 全年 | 紧急 kWh |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| 当前 `D_pv7d_adaptive` | {_fmt(q42_cur['tune_cost'])} | {_fmt(q42_cur['oos_cost'])} | "
            f"{_fmt(q42_cur['full_cost'])} | {_fmt(q42_cur['full_emergency_kwh'])} |"
        ),
        (
            f"| 候选 `E_pv7d_netrho` | {_fmt(q42_new['tune_cost'])} | {_fmt(q42_new['oos_cost'])} | "
            f"{_fmt(q42_new['full_cost'])} | {_fmt(q42_new['full_emergency_kwh'])} |"
        ),
        "",
        f"全年差额 {q42_new['full_cost'] - q42_cur['full_cost']:+,.2f} 元。",
        "",
        "## Q4-3",
        "",
        "| 规则 | tune | OOS | 全年 | 紧急 kWh |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| 当前 LA $q_L=0.65$ | {_fmt(q43_cur['tune_cost'])} | {_fmt(q43_cur['oos_cost'])} | "
            f"{_fmt(q43_cur['full_cost'])} | {_fmt(q43_cur['full_emergency_kwh'])} |"
        ),
        (
            f"| 候选 LA $q_L=0.60$ | {_fmt(q43_la['tune_cost'])} | {_fmt(q43_la['oos_cost'])} | "
            f"{_fmt(q43_la['full_cost'])} | {_fmt(q43_la['full_emergency_kwh'])} |"
        ),
    ]
    for name in ("N0", "M0L"):
        block = comparison["q4_3"]["controls"][name]
        lines.append(
            f"| 对照 {name} $q_L=0.60$ | {_fmt(block['tune_cost'])} | {_fmt(block['oos_cost'])} | "
            f"{_fmt(block['full_cost'])} | {_fmt(block['full_emergency_kwh'])} |"
        )
    lines.extend(
        [
            "",
            f"LA 全年差额 {q43_la['full_cost'] - q43_cur['full_cost']:+,.2f} 元。",
            "",
            "本轮**不替换**正式表。是否晋升由人工确认。",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "promote": False,
            "wrote_result4_2": False,
            "wrote_result4_3": False,
            "out_dir": str(out_dir),
        },
    )

    _assert_untouched(RESULT4_2_XLSX, before["result4-2"], "result4-2.xlsx")
    _assert_untouched(RESULT4_3_XLSX, before["result4-3"], "result4-3.xlsx")
    _assert_untouched(Q2_XLSX, before["result2"], "result2.xlsx")
    _assert_untouched(Q3_XLSX, before["result3"], "result3.xlsx")
    print("wrote", out_dir / "comparison.json", flush=True)
    print(
        json.dumps(
            {
                "promote": False,
                "q4_2_full": q42_new["full_cost"],
                "q4_3_LA_full": q43_la["full_cost"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
