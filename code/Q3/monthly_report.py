"""Month-by-month cost and emergency energy for one policy, to test whether a
full-year win is spread across the year or carried by a few months."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EXP_DIR, PV_P0_MODE, PV_P0_MODES
from run_q3 import load_bundle, run_phase


def monthly_rows(official: list[dict]) -> list[dict]:
    cost: dict[str, float] = defaultdict(float)
    em_kwh: dict[str, float] = defaultdict(float)
    days: dict[str, int] = defaultdict(int)
    for rec in official:
        month = rec["date"][:7]
        cost[month] += float(rec["bill"]["total_cost"])
        em_kwh[month] += float(rec["actual"]["emergency_kwh"].sum())
        days[month] += 1
    return [
        {
            "month": m,
            "n_days": days[m],
            "total_cost": cost[m],
            "emergency_kwh": em_kwh[m],
        }
        for m in sorted(cost)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--pv-p0", choices=PV_P0_MODES, default=PV_P0_MODE)
    parser.add_argument("--out-dir", required=True, help="name under results/Q3/exp")
    args = parser.parse_args()

    out_dir = EXP_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(p0_mode=args.pv_p0, write=False)
    result = run_phase(
        bundle,
        "official",
        export=False,
        policy_names=[args.policy],
        out_dir=out_dir,
    )
    rows = monthly_rows(result["payloads"][args.policy]["official"])
    tag = f"{args.policy}_{args.pv_p0}"
    pd.DataFrame(rows).to_csv(out_dir / f"monthly_{tag}.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"policy": args.policy, "pv_p0": args.pv_p0, "months": rows},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
