"""Sweep 48 h look-ahead and three-segment net-load quantiles. Experiment only."""

from __future__ import annotations

import argparse
import json
import sys
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

from config import EXP_DIR, LOOKAHEAD_HOURS, PV_P0_MODE, Q_EVENING_GRID, Q_LOCK_GRID, Q_OPEN_DEFAULT
from rolling import POLICIES
from run_q3 import load_bundle, run_phase


def _tag(lookahead: int, q_lock, q_open, q_evening, beta: str) -> str:
    qpart = "beta" if q_lock is None else f"ql{q_lock:g}_qo{q_open:g}_qe{q_evening:g}"
    return f"h{lookahead}_{qpart}_{beta}"


def _extras(official: list[dict]) -> dict:
    dp = sum(r["bill"]["delta_plus_kwh"] for r in official)
    dm = sum(r["bill"]["delta_minus_kwh"] for r in official)
    soc24 = np.array([r["actual"]["soc24_kwh"] for r in official], dtype=float)
    em = np.vstack([r["actual"]["emergency_kwh"] for r in official])
    em_total = float(em.sum())
    em_20 = float(em[:, 120:126].sum())
    return {
        "delta_plus_kwh": float(dp),
        "delta_minus_kwh": float(dm),
        "soc24_median": float(np.median(soc24)),
        "soc24_at_min_days": int(np.sum(soc24 <= 1250.0)),
        "emergency_share_20h": (em_20 / em_total) if em_total > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("phase1", "official"), default="official")
    parser.add_argument("--base", default="LA")
    parser.add_argument("--out-dir", default="quantile_sweep")
    parser.add_argument("--pv-p0", default=PV_P0_MODE)
    parser.add_argument(
        "--stage",
        choices=("baselines", "qlock", "qevening", "edges", "all"),
        default="all",
        help="baselines=48h/no-beta; qlock=sweep q_lock; qevening=best lock + q_evening 0.7; edges=0.50 and 0.90",
    )
    parser.add_argument(
        "--q-lock",
        default=None,
        help="comma-separated q_lock values; default is the grid minus the 0.50/0.90 edges",
    )
    args = parser.parse_args()
    base = POLICIES[args.base]
    specs = []

    def add(lookahead, q_lock, q_open, q_evening, use_buffer, beta_lock, beta_open):
        name = f"{args.base}_{_tag(lookahead, q_lock, q_open, q_evening, 'buf' if use_buffer else 'nobuf')}"
        POLICIES[name] = replace(
            base,
            name=name,
            look_ahead=True,
            lookahead_hours=int(lookahead),
            q_lock=q_lock,
            q_open=q_open,
            q_evening=q_evening,
            use_buffer=use_buffer,
            beta_lock=beta_lock,
            beta_open=beta_open,
        )
        specs.append(name)

    stage = args.stage
    if stage in ("baselines", "all"):
        add(LOOKAHEAD_HOURS, None, None, None, True, base.beta_lock, base.beta_open)
        add(48, None, None, None, True, base.beta_lock, base.beta_open)
        add(48, None, None, None, False, 0.0, 0.0)
    q_locks = (
        [float(x) for x in args.q_lock.split(",")]
        if args.q_lock
        else [q for q in Q_LOCK_GRID if q not in (0.50, 0.90)]
    )
    if stage in ("qlock", "all"):
        for q in q_locks:
            add(48, q, Q_OPEN_DEFAULT, 0.50, False, 0.0, 0.0)
    if stage in ("qevening", "all"):
        # Placeholder locks; after qlock we re-run this stage with --q-lock set to the winner.
        for q in q_locks:
            add(48, q, Q_OPEN_DEFAULT, 0.70, False, 0.0, 0.0)
    if stage in ("edges", "all"):
        for q in (0.50, 0.90):
            add(48, q, Q_OPEN_DEFAULT, 0.50, False, 0.0, 0.0)

    out_dir = EXP_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(p0_mode=args.pv_p0, write=False)
    result = run_phase(bundle, args.phase, policy_names=specs, export=False, out_dir=out_dir)

    rows = []
    for name in specs:
        summary = result["metrics"][name]
        policy = POLICIES[name]
        extra = _extras(result["payloads"][name]["official"])
        rows.append(
            {
                "policy": name,
                "lookahead_hours": policy.lookahead_hours,
                "q_lock": policy.q_lock,
                "q_open": policy.q_open,
                "q_evening": policy.q_evening,
                "use_buffer": policy.use_buffer,
                "beta_lock": policy.beta_lock,
                "beta_open": policy.beta_open,
                "total_cost": summary["total_cost"],
                "plan_only_cost": summary["plan_only_cost"],
                "emergency_cost": summary["emergency_cost"],
                "emergency_kwh": summary["emergency_kwh"],
                **extra,
            }
        )
    frame = pd.DataFrame(rows).sort_values("total_cost").reset_index(drop=True)
    csv_path = out_dir / f"{args.phase}_quantile_sweep.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    (out_dir / f"{args.phase}_quantile_best.json").write_text(
        json.dumps(frame.iloc[0].to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
