"""Diagnostic: oracle attachment-4 prices in the LP vs frozen hat0 decisions.

Runs Q4-2 and Q4-3 M1 as separate processes so Q2/Q3 config do not clash.
Does not export result4-2.xlsx / result4-3.xlsx. Settlement stays on attachment 4.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

Q4_DIR = Path(__file__).resolve().parent
CODE_DIR = Q4_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from Q4.config import (  # noqa: E402
    ORACLE_DIAG_DIR,
    RESULT4_2_XLSX,
    RESULT4_3_XLSX,
    RESULT_DIR,
)
from Q4.export_results import write_json  # noqa: E402
from Q4.price_bank import load_price_bank  # noqa: E402

Q2_FROZEN_TOTAL = 13765167.58173598
Q3_M1_FROZEN_TOTAL = 13519092.792762008
REPO_ROOT = CODE_DIR.parent


def _rel_pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old


def _row(scheme: str, hat0: float, oracle: float, frozen: float | None) -> dict:
    gap = hat0 - oracle
    row = {
        "scheme": scheme,
        "hat0_total_cost": hat0,
        "oracle_total_cost": oracle,
        "forecast_gap": gap,
        "forecast_gap_pct": _rel_pct(hat0, oracle),
    }
    if frozen is not None:
        row["frozen_typical_tariff_cost"] = frozen
        row["env_gap_oracle_minus_frozen"] = oracle - frozen
        row["env_gap_pct"] = _rel_pct(oracle, frozen)
        row["total_gap_hat0_minus_frozen"] = hat0 - frozen
        row["total_gap_pct"] = _rel_pct(hat0, frozen)
    return row


def _forbid_official_xlsx() -> dict[str, tuple[int, float]]:
    stamps = {}
    for path in (RESULT4_2_XLSX, RESULT4_3_XLSX):
        if not path.exists():
            raise RuntimeError(f"official {path.name} missing; diagnostic will not recreate it")
        stat = path.stat()
        stamps[str(path)] = (stat.st_size, stat.st_mtime)
    return stamps


def _assert_official_untouched(before: dict[str, tuple[int, float]]) -> None:
    for path, (size, mtime) in before.items():
        stat = Path(path).stat()
        if stat.st_size != size or stat.st_mtime != mtime:
            raise RuntimeError(f"diagnostic touched official file {path}")


def _run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(Q4_DIR / script), "--full", "--plan-source", "oracle", *extra]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    ORACLE_DIAG_DIR.mkdir(parents=True, exist_ok=True)
    before = _forbid_official_xlsx()
    official_metrics = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
    hat0_q42 = float(official_metrics["q4_2"]["total_cost"])
    hat0_m1 = float(official_metrics["q4_3"]["M1"]["total_cost"])

    bank = load_price_bank()
    dates = pd.to_datetime(bank["dates"])
    official = (dates >= "2025-02-01") & (dates <= "2025-12-31")
    actual = np.asarray(bank["actual"], dtype=float)[official]
    hat0 = np.asarray(bank["hat0"], dtype=float)[official]
    cold = np.asarray(bank["cold_start"], dtype=float)
    price_level = {
        "att4_mean_official": float(actual.mean()),
        "hat0_mean_official": float(hat0.mean()),
        "att1_typical_mean": float(cold.mean()),
        "att4_minus_att1_mean": float(actual.mean() - cold.mean()),
        "hat0_mae_official": float(np.mean(np.abs(hat0 - actual))),
    }

    _run("run_q4_2.py", [])
    _run("run_q4_3.py", ["--policies", "M1"])

    q42 = json.loads((ORACLE_DIAG_DIR / "q42_oracle_summary.json").read_text(encoding="utf-8"))
    m1 = json.loads((ORACLE_DIAG_DIR / "q43_M1_oracle_summary.json").read_text(encoding="utf-8"))
    oracle_q42 = float(q42["total_cost"])
    oracle_m1 = float(m1["total_cost"])
    rows = [
        _row("Q4-2 C_pv7d_q82", hat0_q42, oracle_q42, Q2_FROZEN_TOTAL),
        _row("Q4-3 M1", hat0_m1, oracle_m1, Q3_M1_FROZEN_TOTAL),
    ]
    pd.DataFrame(rows).to_csv(ORACLE_DIAG_DIR / "comparison.csv", index=False, encoding="utf-8-sig")

    env_share = []
    for row in rows:
        total = row["total_gap_hat0_minus_frozen"]
        env_share.append(
            {
                "scheme": row["scheme"],
                "environment_share_of_total_gap": None if abs(total) < 1e-9 else row["env_gap_oracle_minus_frozen"] / total,
                "forecast_share_of_total_gap": None if abs(total) < 1e-9 else row["forecast_gap"] / total,
            }
        )

    mean_oracle = 0.5 * (oracle_q42 + oracle_m1)
    mean_hat0 = 0.5 * (hat0_q42 + hat0_m1)
    if mean_oracle >= 13.6e6 and abs(mean_hat0 - mean_oracle) < 0.4e6:
        verdict = (
            "environment: oracle remains around 14 million yuan, so the increase vs Q2/Q3 "
            "is mainly the attachment-4 tariff level/shape, not 0:00 forecast error."
        )
        verdict_zh = "费用上涨主要来自附件4真实电价环境，而不是 0:00 预测误差。"
    elif mean_oracle <= 13.4e6 and mean_hat0 >= 14.0e6:
        verdict = (
            "forecast: oracle returns near 13 million yuan while hat0 stays above 14 million, "
            "so 0:00 price forecast error causes a large dispatch loss."
        )
        verdict_zh = "费用上涨主要来自 0:00 电价预测误差导致的调度损失。"
    else:
        verdict = "mixed: both the attachment-4 environment and 0:00 forecast error contribute."
        verdict_zh = "附件4电价环境与 0:00 预测误差都有贡献，需看分项占比。"

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "note": "LP may use today's true attachment-4 price; not an official Q4 method.",
        "settle_price": "attachment4_actual",
        "hat0_from": str(RESULT_DIR / "metrics.json"),
        "price_level": price_level,
        "comparison": rows,
        "gap_shares": env_share,
        "verdict": verdict,
        "verdict_zh": verdict_zh,
        "q42_oracle": q42,
        "q43_m1_oracle": m1,
        "official_xlsx_guard": "untouched",
    }
    write_json(ORACLE_DIAG_DIR / "run_manifest.json", payload)

    lines = [
        "# Q4 完美电价（oracle）诊断",
        "",
        "只改 LP 决策电价：hat0 → 当天附件 4 真值。负荷/光伏预报、SOC、回放、结算、策略不变。结算仍用附件 4。本实验允许看当天真电价，不能作为正式方案。未覆盖 result4-2.xlsx / result4-3.xlsx。",
        "",
        "| 方案 | hat0 决策（元） | oracle 决策（元） | 预测误差代价 | 相对 oracle | 原 Q2/Q3 典型电价（元） | oracle−原问题 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {scheme} | {hat0:,.2f} | {oracle:,.2f} | {gap:,.2f} | {gpct:+.2f}% | {frozen:,.2f} | {env:,.2f} |".format(
                scheme=row["scheme"],
                hat0=row["hat0_total_cost"],
                oracle=row["oracle_total_cost"],
                gap=row["forecast_gap"],
                gpct=row["forecast_gap_pct"],
                frozen=row["frozen_typical_tariff_cost"],
                env=row["env_gap_oracle_minus_frozen"],
            )
        )
    lines.extend(
        [
            "",
            f"- 正式窗附件 4 均价 {price_level['att4_mean_official']:.4f} 元/kWh，附件 1 典型日均价 {price_level['att1_typical_mean']:.4f}，差 {price_level['att4_minus_att1_mean']:.4f}。",
            f"- hat0 正式窗 MAE {price_level['hat0_mae_official']:.4f}。",
            "",
            f"**结论：{verdict_zh}**",
            "",
            verdict,
        ]
    )
    (ORACLE_DIAG_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _assert_official_untouched(before)
    print(json.dumps({"verdict_zh": verdict_zh, "comparison": rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
