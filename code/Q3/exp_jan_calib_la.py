"""January-only selection of q_lock on the frozen LA algorithm.

Point forecasts, 48 h look-ahead, three-segment quantiles, two-sided
adjustments and greedy playback stay as in the official LA. Only the
lock quantile is chosen on 2025-01-15..01-31 (cash + inventory score).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

Q3_DIR = Path(__file__).resolve().parent
CODE_DIR = Q3_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(Q3_DIR) not in sys.path:
    sys.path.insert(0, str(Q3_DIR))

from config import ETA_DISCHARGE, EXP_DIR, LOOKAHEAD_HOURS, Q_EVENING, Q_OPEN
from pv_forecast import causal_sigma
from quantile import QuantileBank
from rolling import POLICIES, run_span
from run_q3 import load_bundle
from validate import leakage_errors, official_errors, summarize

FROZEN_LA_COST = 13263910.989105027
FROZEN_LA_EM_KWH = 39678.65388158653
JAN_START = 14
JAN_END = 31
OFFICIAL_START = 31
OFFICIAL_END = 365
Q_LOCK_GRID = (0.50, 0.55, 0.60, 0.63, 0.65, 0.70, 0.75)


def _la(q_lock: float):
    base = POLICIES["LA"]
    name = f"LA_ql{q_lock:g}"
    return replace(
        base,
        name=name,
        look_ahead=True,
        lookahead_hours=LOOKAHEAD_HOURS,
        q_lock=q_lock,
        q_open=Q_OPEN,
        q_evening=Q_EVENING,
    )


def _nu(price144: np.ndarray) -> float:
    return float(np.min(price144)) / ETA_DISCHARGE


def _inventory_score(summary: dict, soc_end: float, nu: float) -> float:
    return float(summary["total_cost"]) + nu * (6000.0 - soc_end)


def main() -> int:
    out_dir = EXP_DIR / "jan_calib_la"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(p0_mode="measured", write=False)
    bank = QuantileBank.build(bundle)
    price144 = bundle["prices"]["price"].to_numpy(dtype=float)
    load_kwh = bundle["year"]["load_kwh"]
    pv_kwh = bundle["year"]["pv_kwh"]
    dates = bundle["year"]["dates"]
    typical = bundle["prices"]["typical_load_kw"].to_numpy(dtype=float)
    interp = bundle["interp"]
    aligned = bundle["aligned"]

    def sigma_fn(day: int):
        return causal_sigma(interp, aligned, day)

    nu = _nu(price144)
    jan_rows = []
    t0 = time.perf_counter()
    for q_lock in Q_LOCK_GRID:
        policy = _la(q_lock)
        POLICIES[policy.name] = policy
        payload = run_span(
            JAN_START,
            JAN_END,
            JAN_START,
            price144,
            load_kwh,
            pv_kwh,
            dates,
            typical,
            interp,
            sigma_fn,
            policy,
            quantile_bank=bank,
        )
        leak = leakage_errors(
            payload["records"],
            load_kwh,
            dates,
            typical,
            quantile_bank=bank,
            pv_kwh=pv_kwh,
        )
        if leak:
            raise RuntimeError(f"{policy.name} January leakage: {leak[:8]}")
        summary = summarize(payload["official"])
        soc_end = float(payload["official"][-1]["soc24_kwh"])
        score = _inventory_score(summary, soc_end, nu)
        jan_rows.append(
            {
                "q_lock": q_lock,
                "q_open": Q_OPEN,
                "q_evening": Q_EVENING,
                "jan_cash": summary["total_cost"],
                "jan_soc_end": soc_end,
                "nu": nu,
                "jan_score": score,
                "jan_emergency_kwh": summary["emergency_kwh"],
            }
        )
        print(
            f"q_lock={q_lock:g}: cash={summary['total_cost']:.2f} "
            f"score={score:.2f} em={summary['emergency_kwh']:.1f}",
            flush=True,
        )

    jan = pd.DataFrame(jan_rows).sort_values("jan_score").reset_index(drop=True)
    q_star = float(jan.iloc[0]["q_lock"])
    win_policy = _la(q_star)
    POLICIES[win_policy.name] = win_policy
    payload = run_span(
        OFFICIAL_START,
        OFFICIAL_END,
        0,
        price144,
        load_kwh,
        pv_kwh,
        dates,
        typical,
        interp,
        sigma_fn,
        win_policy,
        quantile_bank=bank,
    )
    leak = leakage_errors(
        payload["records"],
        load_kwh,
        dates,
        typical,
        quantile_bank=bank,
        pv_kwh=pv_kwh,
    )
    gate = official_errors(
        payload["official"],
        dates,
        payload["soc_track"],
        load_kwh,
        pv_kwh,
        price144,
        expected_n=OFFICIAL_END - OFFICIAL_START,
    )
    if leak:
        raise RuntimeError(f"full-year leakage: {leak[:8]}")
    if gate:
        raise RuntimeError(f"full-year gate: {gate[:8]}")
    full = summarize(payload["official"])
    full["elapsed_s"] = time.perf_counter() - t0
    full["leakage_errors"] = leak
    full["gate_errors"] = gate
    jan.to_csv(out_dir / "january_grid.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"{win_policy.name}_summary.json").write_text(
        json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verdict = {
        "rule": "choose q_lock by Jan 15-31 cash + nu*(6000-soc_end); Feb-Dec confirmation only",
        "nu": nu,
        "selected_q_lock": q_star,
        "q_open": Q_OPEN,
        "q_evening": Q_EVENING,
        "lookahead_hours": LOOKAHEAD_HOURS,
        "jan_score": float(jan.iloc[0]["jan_score"]),
        "full_cost": full["total_cost"],
        "delta_vs_frozen_q065": full["total_cost"] - FROZEN_LA_COST,
        "full_emergency_kwh": full["emergency_kwh"],
        "delta_em_kwh": full["emergency_kwh"] - FROZEN_LA_EM_KWH,
        "matches_freeze": abs(full["total_cost"] - FROZEN_LA_COST) < 1e-6,
    }
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# January-calibrated q_lock on frozen LA",
        "",
        jan.to_string(index=False),
        "",
        f"Winner q_lock={q_star}",
        f"Feb-Dec cost: {full['total_cost']:.2f} (delta vs freeze 0.65: {verdict['delta_vs_frozen_q065']:.2f})",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(jan.to_string(index=False))
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
