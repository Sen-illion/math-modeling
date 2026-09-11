"""Retune Q4-2 q_min/k and Q4-3 LA beta on attachment-4 settlement.

hat0 stays frozen. Nested windows match problem 2. Promote a scheme only if both
the tune window and the 7-12 month out-of-sample window beat the current official
rule. Does not write result2.xlsx / result3.xlsx, and does not touch result4-x
unless that scheme's gate passes.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

Q4_DIR = Path(__file__).resolve().parent
CODE_DIR = Q4_DIR.parent
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from Q4.config import (  # noqa: E402
    CURRENT_Q42_K,
    CURRENT_Q42_QMIN,
    CURRENT_Q43_BETA_LOCK,
    CURRENT_Q43_BETA_OPEN,
    OFFICIAL_END,
    OFFICIAL_START,
    OOS_START,
    Q2_ADAPTIVE_POLICY,
    Q3_POLICY,
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
    export_result4_2,
    export_result4_3,
    export_specified_q42,
    merge_metrics,
    write_json,
    write_run_manifest,
)

Q2_XLSX = REPO_ROOT / "results" / "Q2" / "result2.xlsx"
Q3_XLSX = REPO_ROOT / "results" / "Q3" / "result3.xlsx"
EPS = 1e-6


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


def _copy_if_exists(src: Path, dest: Path) -> None:
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _row_to_dict(row: pd.Series) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, (np.floating, float)):
            out[str(key)] = float(value)
        elif isinstance(value, (np.integer, int)):
            out[str(key)] = int(value)
        elif isinstance(value, (np.bool_, bool)):
            out[str(key)] = bool(value)
        else:
            out[str(key)] = value
    return out


def _pick_current(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    hit = frame.loc[mask]
    if len(hit) != 1:
        raise RuntimeError(f"expected exactly one current row for {label}, got {len(hit)}")
    return hit.iloc[0]


def nested_pick(frame: pd.DataFrame, current: pd.Series) -> dict:
    eligible = frame.loc[frame["inner_cost"] < float(current["inner_cost"]) - EPS]
    if eligible.empty:
        selected = current
        kept = True
    else:
        selected = eligible.loc[eligible["select_cost"].idxmin()]
        kept = False
    return {
        "n_candidates": int(len(frame)),
        "n_eligible_after_inner": int(len(eligible)),
        "kept_current": kept,
        "eligible_labels": eligible["label"].tolist() if len(eligible) else [],
        "current": _row_to_dict(current),
        "selected": _row_to_dict(selected),
    }


def _promote_gate(selected: dict, current: dict) -> tuple[bool, str]:
    if selected.get("label") == current.get("label"):
        return False, "nested pick kept the current official rule"
    tune_ok = float(selected["tune_cost"]) < float(current["tune_cost"]) - EPS
    oos_ok = float(selected["oos_cost"]) < float(current["oos_cost"]) - EPS
    if tune_ok and oos_ok:
        return True, "tune and OOS both cheaper than the current official rule"
    parts = []
    if not tune_ok:
        parts.append("tune window not cheaper")
    if not oos_ok:
        parts.append("OOS window not cheaper")
    return False, "; ".join(parts)


def _write_official_summary(metrics: dict) -> None:
    q42 = metrics["q4_2"]
    q43 = metrics["q4_3"]
    lines = [
        "# Q4 正式调度（冻结 $\\hat p_0$）",
        "",
        "决策电价：当天 0:00 三层 expanding $\\hat p_0=b+\\Delta_{\\mathrm{week}}+\\Delta_{\\mathrm{recent}}$。结算电价：附件 4 真值。Q4-3 的 6/12/18 不改电价。正式窗 2025-02-01–12-31，334 天。门禁：价格泄漏 0，SOC 盒约束通过。",
        "",
        "缓冲档在附件 4 世界上按问题 2 的嵌套窗重选；未过门禁的方案保留对齐后的正式规则。",
        "",
        "| 方案 | 策略 | 总费用（元） | 紧急电量（kWh） |",
        "| --- | --- | ---: | ---: |",
        f"| Q4-2 正式 | {q42.get('policy')} | {float(q42['total_cost']):,.2f} | {float(q42['emergency_kwh']):,.2f} |",
        f"| Q4-3 正式 | {q43.get('official_policy', Q3_POLICY)} | {float(q43[q43.get('official_policy', Q3_POLICY)]['total_cost']):,.2f} | {float(q43[q43.get('official_policy', Q3_POLICY)]['emergency_kwh']):,.2f} |",
    ]
    if "M1" in q43:
        lines.append(
            f"| Q4-3 对照 | M1 | {float(q43['M1']['total_cost']):,.2f} | {float(q43['M1']['emergency_kwh']):,.2f} |"
        )
    if "N0" in q43:
        lines.append(
            f"| Q4-3 对照 | N0 | {float(q43['N0']['total_cost']):,.2f} | {float(q43['N0']['emergency_kwh']):,.2f} |"
        )
    lines.extend(
        [
            "",
            "产物：`result4-2.xlsx`、`result4-3.xlsx`。重选过程见 `diagnostics/retune_q_beta/`。未改 `results/Q2/result2.xlsx` / `results/Q3/result3.xlsx`。",
            "",
        ]
    )
    (RESULT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("all", "q42", "q43", "promote"), default="all")
    parser.add_argument("--skip-sweep", action="store_true", help="reuse existing grid CSVs")
    args = parser.parse_args()

    RETUNE_DIAG_DIR.mkdir(parents=True, exist_ok=True)
    before_q2 = _stamp(Q2_XLSX)
    before_q3 = _stamp(Q3_XLSX)
    before_r42 = _stamp(RESULT4_2_XLSX)
    before_r43 = _stamp(RESULT4_3_XLSX)

    _copy_if_exists(RESULT4_2_XLSX, RETUNE_DIAG_DIR / "result4-2_before.xlsx")
    _copy_if_exists(RESULT4_3_XLSX, RETUNE_DIAG_DIR / "result4-3_before.xlsx")
    _copy_if_exists(RESULT_DIR / "metrics.json", RETUNE_DIAG_DIR / "metrics_before.json")

    metrics = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
    do_q42 = args.only in {"all", "q42"}
    do_q43 = args.only in {"all", "q43"}
    do_promote = args.only in {"all", "promote"}

    if do_q42 and not args.skip_sweep:
        _run("run_q4_2.py", ["--sweep-buffers", "--out-dir", str(RETUNE_DIAG_DIR)])
    if do_q43 and not args.skip_sweep:
        _run("run_q4_3.py", ["--sweep-buffers", "--out-dir", str(RETUNE_DIAG_DIR)])

    selected_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "inner": [OFFICIAL_START, TUNE_INNER_END],
            "select": [TUNE_SELECT_START, TUNE_END],
            "tune": [OFFICIAL_START, TUNE_END],
            "oos": [OOS_START, OFFICIAL_END],
            "note": "inner must beat the current official rule; select takes the lowest among eligible; promote only if tune and OOS both beat current. No full-year argmin. No fallback if nobody clears inner.",
        },
        "plan_price": "hat0_expanding_layers",
        "settle_price": "attachment4_actual",
    }

    promote_q42 = False
    promote_q43 = False
    q42_pick = None
    q43_pick = None

    q42_csv = RETUNE_DIAG_DIR / "q42_grid.csv"
    if q42_csv.exists() and (do_q42 or do_promote):
        q42 = pd.read_csv(q42_csv)
        current = _pick_current(
            q42,
            np.isclose(q42["q_min"], CURRENT_Q42_QMIN) & np.isclose(q42["k"], CURRENT_Q42_K),
            "Q4-2 current official",
        )
        q42_pick = nested_pick(q42, current)
        promote_q42, reason = _promote_gate(q42_pick["selected"], q42_pick["current"])
        q42_pick["promote"] = promote_q42
        q42_pick["reason"] = reason
        selected_payload["q4_2"] = q42_pick
        print(
            "Q4-2 nested selected",
            q42_pick["selected"]["label"],
            "promote" if promote_q42 else "keep",
            reason,
            flush=True,
        )

    q43_csv = RETUNE_DIAG_DIR / "q43_grid.csv"
    if q43_csv.exists() and (do_q43 or do_promote):
        q43 = pd.read_csv(q43_csv)
        current = _pick_current(
            q43,
            np.isclose(q43["beta_lock"], CURRENT_Q43_BETA_LOCK)
            & np.isclose(q43["beta_open"], CURRENT_Q43_BETA_OPEN),
            "Q4-3 current official",
        )
        q43_pick = nested_pick(q43, current)
        official_full = float(metrics["q4_3"]["LA"]["total_cost"])
        current_tune = float(q43_pick["current"]["tune_cost"])
        q43_pick["current"]["full_cost"] = official_full
        q43_pick["current"]["oos_cost"] = official_full - current_tune
        same_beta = (
            abs(float(q43_pick["selected"]["beta_lock"]) - CURRENT_Q43_BETA_LOCK) < 1e-12
            and abs(float(q43_pick["selected"]["beta_open"]) - CURRENT_Q43_BETA_OPEN) < 1e-12
        )
        if same_beta:
            q43_pick["selected"]["full_cost"] = official_full
            q43_pick["selected"]["oos_cost"] = official_full - float(q43_pick["selected"]["tune_cost"])
            q43_pick["full_year_rerun"] = False
        else:
            print("Q4-3 nested winner differs; running full year", flush=True)
            payload_path = RETUNE_DIAG_DIR / "q43_winner_payload.pkl"
            _run(
                "run_q4_3.py",
                [
                    "--full",
                    "--plan-source",
                    "hat0",
                    "--policies",
                    Q3_POLICY,
                    "--beta-lock",
                    str(q43_pick["selected"]["beta_lock"]),
                    "--beta-open",
                    str(q43_pick["selected"]["beta_open"]),
                    "--out-dir",
                    str(RETUNE_DIAG_DIR),
                    "--dump-payload",
                    str(payload_path),
                ],
            )
            winner_summary = json.loads(
                (RETUNE_DIAG_DIR / "q43_LA_summary.json").read_text(encoding="utf-8")
            )
            full_cost = float(winner_summary["total_cost"])
            q43_pick["selected"]["full_cost"] = full_cost
            q43_pick["selected"]["oos_cost"] = full_cost - float(q43_pick["selected"]["tune_cost"])
            q43_pick["selected"]["full_emergency_kwh"] = float(winner_summary["emergency_kwh"])
            q43_pick["full_year_summary"] = winner_summary
            q43_pick["full_year_rerun"] = True
        promote_q43, reason = _promote_gate(q43_pick["selected"], q43_pick["current"])
        q43_pick["promote"] = promote_q43
        q43_pick["reason"] = reason
        selected_payload["q4_3"] = q43_pick
        print(
            "Q4-3 nested selected",
            q43_pick["selected"]["label"],
            "promote" if promote_q43 else "keep",
            reason,
            flush=True,
        )

    promoted = {}
    if do_promote and promote_q42 and q42_pick is not None:
        payload_path = RETUNE_DIAG_DIR / "q42_winner_payload.pkl"
        q_min = float(q42_pick["selected"]["q_min"])
        k = float(q42_pick["selected"]["k"])
        _run(
            "run_q4_2.py",
            [
                "--full",
                "--plan-source",
                "hat0",
                "--margin",
                "adaptive",
                "--q-min",
                str(q_min),
                "--k",
                str(k),
                "--out-dir",
                str(RETUNE_DIAG_DIR),
                "--dump-payload",
                str(payload_path),
            ],
        )
        new_q42 = json.loads((RETUNE_DIAG_DIR / "q42_adaptive_summary.json").read_text(encoding="utf-8"))
        payload = pickle.loads(payload_path.read_bytes())
        dest = export_result4_2(payload["prices"], payload["traces"])
        tables = export_specified_q42(payload["prices"], payload["traces"])
        (RESULT_DIR / "q42_summary.json").write_text(
            json.dumps(new_q42, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        shutil.copy2(RETUNE_DIAG_DIR / "q42_adaptive_daily.csv", RESULT_DIR / "q42_daily.csv")
        merge_metrics(q42_summary=new_q42)
        write_run_manifest(
            {
                "q4_2": {
                    "policy": Q2_ADAPTIVE_POLICY["name"],
                    "summary": new_q42,
                    "result_xlsx": str(dest),
                    "specified_tables": [str(p) for p in tables],
                    "retune": q42_pick,
                }
            }
        )
        promoted["q4_2"] = str(dest)
        print("promoted Q4-2", dest, flush=True)

    if do_promote and promote_q43 and q43_pick is not None:
        payload_path = RETUNE_DIAG_DIR / "q43_winner_payload.pkl"
        if not payload_path.exists():
            _run(
                "run_q4_3.py",
                [
                    "--full",
                    "--plan-source",
                    "hat0",
                    "--policies",
                    Q3_POLICY,
                    "--beta-lock",
                    str(q43_pick["selected"]["beta_lock"]),
                    "--beta-open",
                    str(q43_pick["selected"]["beta_open"]),
                    "--out-dir",
                    str(RETUNE_DIAG_DIR),
                    "--dump-payload",
                    str(payload_path),
                ],
            )
        new_la = json.loads((RETUNE_DIAG_DIR / "q43_LA_summary.json").read_text(encoding="utf-8"))
        payload = pickle.loads(payload_path.read_bytes())
        dest = export_result4_3(payload["official"], payload["end_min"])
        tables = export_paper_q43(payload["official"], payload["end_min"])
        q43_metrics = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8")).get("q4_3", {})
        q43_metrics["LA"] = new_la
        q43_metrics["official_policy"] = Q3_POLICY
        (RESULT_DIR / "q43_LA_summary.json").write_text(
            json.dumps(new_la, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if (RETUNE_DIAG_DIR / "q43_LA_daily.csv").exists():
            shutil.copy2(RETUNE_DIAG_DIR / "q43_LA_daily.csv", RESULT_DIR / "q43_LA_daily.csv")
        merge_metrics(q43_metrics=q43_metrics)
        write_run_manifest(
            {
                "q4_3": {
                    "official_policy": Q3_POLICY,
                    "metrics": q43_metrics,
                    "result_xlsx": str(dest),
                    "specified_tables": [str(p) for p in tables],
                    "retune": q43_pick,
                }
            }
        )
        promoted["q4_3"] = str(dest)
        print("promoted Q4-3", dest, flush=True)

    if not promote_q42:
        _assert_untouched(RESULT4_2_XLSX, before_r42, "result4-2.xlsx")
    if not promote_q43:
        _assert_untouched(RESULT4_3_XLSX, before_r43, "result4-3.xlsx")
    _assert_untouched(Q2_XLSX, before_q2, "result2.xlsx")
    _assert_untouched(Q3_XLSX, before_q3, "result3.xlsx")

    write_json(RETUNE_DIAG_DIR / "selected.json", selected_payload)
    metrics_now = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
    if promoted:
        _write_official_summary(metrics_now)

    lines = [
        "# Q4 附件 4 世界缓冲档重选",
        "",
        "hat0 冻结，结算用附件 4。嵌套协议与问题 2 相同：inner 2025-02-01–04-30 须优于当前正式规则；select 2025-05-01–06-30 在过关者中取最低；无人过关则不换档。晋升要求 tune（2/1–6/30）与 OOS（7/1–12/31）都更省。未扫回放 γ，未改 result2/result3。",
        "",
    ]
    if q42_pick is not None:
        cur = q42_pick["current"]
        sel = q42_pick["selected"]
        lines.extend(
            [
                "## Q4-2 `resid_vol7` $q_{\\min}\\times k$",
                "",
                f"当前正式：`{cur['label']}`。inner 过关 {q42_pick['n_eligible_after_inner']} / {q42_pick['n_candidates']}。嵌套选定 `{sel['label']}`。",
                "",
                "| 规则 | inner | select | tune | OOS | 全年 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                f"| 当前 {cur['label']} | {cur['inner_cost']:,.2f} | {cur['select_cost']:,.2f} | {cur['tune_cost']:,.2f} | {cur['oos_cost']:,.2f} | {cur['full_cost']:,.2f} |",
                f"| 选定 {sel['label']} | {sel['inner_cost']:,.2f} | {sel['select_cost']:,.2f} | {sel['tune_cost']:,.2f} | {sel['oos_cost']:,.2f} | {sel['full_cost']:,.2f} |",
                "",
                f"晋升：{'是' if promote_q42 else '否'}（{q42_pick['reason']}）。",
                "",
            ]
        )
    if q43_pick is not None:
        cur = q43_pick["current"]
        sel = q43_pick["selected"]
        lines.extend(
            [
                "## Q4-3 LA $\\beta_{\\mathrm{lock}}\\times\\beta_{\\mathrm{open}}$",
                "",
                f"当前正式：`{cur['label']}`。inner 过关 {q43_pick['n_eligible_after_inner']} / {q43_pick['n_candidates']}。嵌套选定 `{sel['label']}`。",
                "",
                "| 规则 | inner | select | tune | OOS | 全年 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                f"| 当前 {cur['label']} | {cur['inner_cost']:,.2f} | {cur['select_cost']:,.2f} | {cur['tune_cost']:,.2f} | {cur['oos_cost']:,.2f} | {cur['full_cost']:,.2f} |",
                f"| 选定 {sel['label']} | {sel['inner_cost']:,.2f} | {sel['select_cost']:,.2f} | {sel['tune_cost']:,.2f} | {sel.get('oos_cost', float('nan')):,.2f} | {sel.get('full_cost', float('nan')):,.2f} |",
                "",
                f"晋升：{'是' if promote_q43 else '否'}（{q43_pick['reason']}）。",
                "",
            ]
        )
    lines.append("网格表：`q42_grid.csv`、`q43_grid.csv`。")
    (RETUNE_DIAG_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        RETUNE_DIAG_DIR / "run_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "promoted": promoted,
            "q4_2_promote": promote_q42,
            "q4_3_promote": promote_q43,
        },
    )
    print(json.dumps({"promoted": promoted, "q4_2": promote_q42, "q4_3": promote_q43}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
