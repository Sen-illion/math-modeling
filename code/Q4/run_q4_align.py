"""Align Q4 dispatch with current official Q2/Q3 policies. Promote only if cheaper."""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

Q4_DIR = Path(__file__).resolve().parent
CODE_DIR = Q4_DIR.parent
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from Q4.config import (  # noqa: E402
    ALIGN_DIAG_DIR,
    Q2_ADAPTIVE_POLICY,
    Q3_POLICY,
    RESULT4_2_XLSX,
    RESULT4_3_XLSX,
    RESULT_DIR,
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


def main() -> None:
    ALIGN_DIAG_DIR.mkdir(parents=True, exist_ok=True)
    before_q2 = _stamp(Q2_XLSX)
    before_q3 = _stamp(Q3_XLSX)
    before_r42 = _stamp(RESULT4_2_XLSX)
    before_r43 = _stamp(RESULT4_3_XLSX)

    metrics = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
    old_q42 = metrics["q4_2"]
    old_m1 = metrics["q4_3"]["M1"]
    old_n0 = metrics["q4_3"].get("N0")
    _copy_if_exists(RESULT_DIR / "q42_summary.json", ALIGN_DIAG_DIR / "baseline_q42_C_pv7d_q82.json")
    _copy_if_exists(RESULT_DIR / "q43_M1_summary.json", ALIGN_DIAG_DIR / "baseline_q43_M1.json")

    q42_payload = ALIGN_DIAG_DIR / "q42_adaptive_payload.pkl"
    q43_payload = ALIGN_DIAG_DIR / "q43_LA_payload.pkl"
    _run(
        "run_q4_2.py",
        [
            "--full",
            "--plan-source",
            "hat0",
            "--margin",
            "adaptive",
            "--out-dir",
            str(ALIGN_DIAG_DIR),
            "--dump-payload",
            str(q42_payload),
        ],
    )
    _run(
        "run_q4_3.py",
        [
            "--full",
            "--plan-source",
            "hat0",
            "--policies",
            Q3_POLICY,
            "--out-dir",
            str(ALIGN_DIAG_DIR),
            "--dump-payload",
            str(q43_payload),
        ],
    )

    new_q42 = json.loads((ALIGN_DIAG_DIR / "q42_adaptive_summary.json").read_text(encoding="utf-8"))
    new_la = json.loads((ALIGN_DIAG_DIR / "q43_LA_summary.json").read_text(encoding="utf-8"))
    q42_new_cost = float(new_q42["total_cost"])
    q42_old_cost = float(old_q42["total_cost"])
    la_cost = float(new_la["total_cost"])
    m1_cost = float(old_m1["total_cost"])
    promote_q42 = q42_new_cost < q42_old_cost - 1e-6
    promote_q43 = la_cost < m1_cost - 1e-6

    comparison = {
        "q4_2": {
            "baseline_policy": old_q42.get("policy"),
            "baseline_total_cost": q42_old_cost,
            "candidate_policy": new_q42.get("policy"),
            "candidate_total_cost": q42_new_cost,
            "delta": q42_new_cost - q42_old_cost,
            "baseline_emergency_kwh": old_q42.get("emergency_kwh"),
            "candidate_emergency_kwh": new_q42.get("emergency_kwh"),
            "promoted": promote_q42,
        },
        "q4_3": {
            "baseline_policy": "M1",
            "baseline_total_cost": m1_cost,
            "candidate_policy": Q3_POLICY,
            "candidate_total_cost": la_cost,
            "delta": la_cost - m1_cost,
            "baseline_emergency_kwh": old_m1.get("emergency_kwh"),
            "candidate_emergency_kwh": new_la.get("emergency_kwh"),
            "promoted": promote_q43,
        },
    }
    write_json(ALIGN_DIAG_DIR / "comparison.json", comparison)

    promoted = {}
    if promote_q42:
        _copy_if_exists(RESULT4_2_XLSX, ALIGN_DIAG_DIR / "result4-2_C_pv7d_q82.xlsx")
        payload = pickle.loads(q42_payload.read_bytes())
        dest = export_result4_2(payload["prices"], payload["traces"])
        tables = export_specified_q42(payload["prices"], payload["traces"])
        (RESULT_DIR / "q42_summary.json").write_text(
            json.dumps(new_q42, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        shutil.copy2(ALIGN_DIAG_DIR / "q42_adaptive_daily.csv", RESULT_DIR / "q42_daily.csv")
        merge_metrics(q42_summary=new_q42)
        write_run_manifest(
            {
                "q4_2": {
                    "policy": Q2_ADAPTIVE_POLICY["name"],
                    "summary": new_q42,
                    "result_xlsx": str(dest),
                    "specified_tables": [str(p) for p in tables],
                    "replaced": "C_pv7d_q82",
                }
            }
        )
        promoted["q4_2"] = str(dest)
        print("promoted Q4-2", dest, flush=True)
    else:
        print("kept Q4-2 C_pv7d_q82; adaptive was not cheaper", flush=True)

    if promote_q43:
        _copy_if_exists(RESULT4_3_XLSX, ALIGN_DIAG_DIR / "result4-3_M1.xlsx")
        payload = pickle.loads(q43_payload.read_bytes())
        dest = export_result4_3(payload["official"], payload["end_min"])
        tables = export_paper_q43(payload["official"], payload["end_min"])
        q43_metrics = dict(metrics.get("q4_3", {}))
        q43_metrics["LA"] = new_la
        q43_metrics["M1"] = old_m1
        if old_n0 is not None:
            q43_metrics["N0"] = old_n0
        q43_metrics["official_policy"] = Q3_POLICY
        (RESULT_DIR / "q43_LA_summary.json").write_text(
            json.dumps(new_la, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (RESULT_DIR / "q43_metrics.json").write_text(
            json.dumps(q43_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        merge_metrics(q43_metrics=q43_metrics)
        write_run_manifest(
            {
                "q4_3": {
                    "official_policy": Q3_POLICY,
                    "metrics": q43_metrics,
                    "result_xlsx": str(dest),
                    "specified_tables": [str(p) for p in tables],
                    "replaced": "M1",
                }
            }
        )
        promoted["q4_3"] = str(dest)
        print("promoted Q4-3", dest, flush=True)
    else:
        print("kept Q4-3 M1; LA was not cheaper", flush=True)

    if not promote_q42:
        _assert_untouched(RESULT4_2_XLSX, before_r42, "result4-2.xlsx")
    if not promote_q43:
        _assert_untouched(RESULT4_3_XLSX, before_r43, "result4-3.xlsx")
    _assert_untouched(Q2_XLSX, before_q2, "result2.xlsx")
    _assert_untouched(Q3_XLSX, before_q3, "result3.xlsx")

    lines = [
        "# Q4 对齐当前 Q2/Q3 正式策略",
        "",
        "hat0 决策、附件 4 结算不变。Q4-2 候选为问题 2 冻结的 `D_pv7d_adaptive`；Q4-3 候选为问题 3 的 `LA`。未重选 rho / q_min / k / beta。",
        "",
        "| 方案 | 现正式 | 现费用（元） | 候选 | 候选费用（元） | 差额 | 紧急电量 旧→新 | 是否晋升 |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
        "| Q4-2 | {bpol} | {bold:,.2f} | {cpol} | {cnew:,.2f} | {d42:+,.2f} | {eold:.1f} → {enew:.1f} | {p42} |".format(
            bpol=comparison["q4_2"]["baseline_policy"],
            bold=q42_old_cost,
            cpol=comparison["q4_2"]["candidate_policy"],
            cnew=q42_new_cost,
            d42=comparison["q4_2"]["delta"],
            eold=float(comparison["q4_2"]["baseline_emergency_kwh"] or 0),
            enew=float(comparison["q4_2"]["candidate_emergency_kwh"] or 0),
            p42="是" if promote_q42 else "否",
        ),
        "| Q4-3 | M1 | {bold:,.2f} | LA | {cnew:,.2f} | {d43:+,.2f} | {eold:.1f} → {enew:.1f} | {p43} |".format(
            bold=m1_cost,
            cnew=la_cost,
            d43=comparison["q4_3"]["delta"],
            eold=float(comparison["q4_3"]["baseline_emergency_kwh"] or 0),
            enew=float(comparison["q4_3"]["candidate_emergency_kwh"] or 0),
            p43="是" if promote_q43 else "否",
        ),
        "",
        "规则参数仍是附件 1 世界上选出的，只是放到附件 4 世界上评估。",
    ]
    (ALIGN_DIAG_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        ALIGN_DIAG_DIR / "run_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "comparison": comparison,
            "promoted": promoted,
        },
    )
    print(json.dumps({"promoted": promoted, "comparison": comparison}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
