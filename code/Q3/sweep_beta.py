"""Sweep the conservative PV buffer betas. Experiment only: never writes frozen results."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

Q3_DIR = Path(__file__).resolve().parent
CODE_DIR = Q3_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(Q3_DIR) not in sys.path:
    sys.path.insert(0, str(Q3_DIR))

from config import (
    BETA_LOCK_GRID,
    BETA_OPEN_GRID,
    EXP_DIR,
    LOCK_SLOTS,
    PV_P0_MODE,
    PV_P0_MODES,
)
from rolling import POLICIES
from run_q3 import load_bundle, run_phase


def _tag(beta_lock: float, beta_open: float, lock_slots: int) -> str:
    return f"bl{beta_lock:g}_bo{beta_open:g}_ls{lock_slots}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("phase1", "official"), default="phase1")
    parser.add_argument("--base", default="M1", help="policy whose buffer is swept")
    parser.add_argument("--out-dir", default="beta_sweep")
    parser.add_argument("--pv-p0", choices=PV_P0_MODES, default=PV_P0_MODE)
    parser.add_argument("--lock-slots", type=int, default=LOCK_SLOTS)
    parser.add_argument(
        "--pairs",
        default=None,
        help='explicit "lock:open" pairs instead of the full grid, e.g. "1.5:0,1:0"',
    )
    args = parser.parse_args()

    if args.pairs:
        combos = []
        for item in args.pairs.split(","):
            lock_s, open_s = item.strip().split(":")
            combos.append((float(lock_s), float(open_s)))
    else:
        combos = [(bl, bo) for bl in BETA_LOCK_GRID for bo in BETA_OPEN_GRID]

    base = POLICIES[args.base]
    names = []
    for beta_lock, beta_open in combos:
        name = f"{args.base}_{_tag(beta_lock, beta_open, args.lock_slots)}"
        POLICIES[name] = replace(
            base,
            name=name,
            beta_lock=float(beta_lock),
            beta_open=float(beta_open),
            lock_slots=int(args.lock_slots),
        )
        names.append(name)

    out_dir = EXP_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(p0_mode=args.pv_p0, write=False)
    result = run_phase(bundle, args.phase, policy_names=names, export=False, out_dir=out_dir)

    rows = []
    for name in names:
        summary = result["metrics"][name]
        policy = POLICIES[name]
        rows.append(
            {
                "policy": name,
                "beta_lock": policy.beta_lock,
                "beta_open": policy.beta_open,
                "lock_slots": policy.lock_slots,
                "pv_p0_mode": args.pv_p0,
                "total_cost": summary["total_cost"],
                "plan_only_cost": summary["plan_only_cost"],
                "emergency_cost": summary["emergency_cost"],
                "emergency_kwh": summary["emergency_kwh"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("total_cost").reset_index(drop=True)
    frozen = frame.loc[
        (frame["beta_lock"] == base.beta_lock) & (frame["beta_open"] == base.beta_open),
        "total_cost",
    ]
    if len(frozen):
        frame["vs_frozen_beta"] = frame["total_cost"] - float(frozen.iloc[0])
    csv_path = out_dir / f"{args.phase}_beta_sweep.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    (out_dir / f"{args.phase}_beta_best.json").write_text(
        json.dumps(frame.iloc[0].to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
