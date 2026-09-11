"""Out-of-sample robustness sweep for every rule that beat fixed 0.8 in-sample.

The tune-window winner failing out-of-sample could be bad luck in the argmax. This
script replays *all* in-sample winners over the full year and reports both windows, so
the conclusion is about the whole family of rules rather than one draw. Selecting a rule
on the out-of-sample column would be snooping; this table is evidence only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from config import ADAPTIVE_DIR, OFFICIAL_END, OFFICIAL_START, OOS_START, TUNE_END  # noqa: E402
from adaptive import flatten_for_table, run_adaptive, run_fixed, summarise  # noqa: E402
from run_q2 import _adaptive_inputs  # noqa: E402

WINDOWS = [
    ("tune", OFFICIAL_START, TUNE_END),
    ("oos", OOS_START, OFFICIAL_END),
    ("full", OFFICIAL_START, OFFICIAL_END),
]


def main() -> None:
    tune_table = pd.read_csv(ADAPTIVE_DIR / "tune_comparison.csv")
    base_tune = float(tune_table.loc[tune_table.label == "fixed_q80", "tune_total_cost"].iloc[0])
    winners = tune_table[
        (tune_table["tune_total_cost"] < base_tune) & tune_table["feature"].notna()
    ].sort_values("tune_total_cost")
    if winners.empty:
        raise SystemExit("no in-sample winner to re-check")

    prices, year, _point, banks, residuals = _adaptive_inputs()
    rows = []

    print("full year: fixed q=0.8 reference", flush=True)
    base = run_fixed(prices, year, banks, 0.8, OFFICIAL_END)
    base_summary = summarise(base, WINDOWS)
    rows.append(flatten_for_table(base_summary))
    ref = base_summary["windows"]

    for _, row in winners.iterrows():
        label = str(row["label"])
        print("full year:", label, flush=True)
        result = run_adaptive(
            prices,
            year,
            banks,
            residuals,
            OFFICIAL_END,
            feature=str(row["feature"]),
            q_min=float(row["q_min"]),
            k=float(row["k"]),
            label=label,
        )
        if result["errors"]:
            raise RuntimeError(f"{label}: {result['errors'][:5]}")
        summary = summarise(result, WINDOWS)
        flat = flatten_for_table(summary)
        flat["tune_gain_vs_q80"] = ref["tune"]["total_cost"] - summary["windows"]["tune"]["total_cost"]
        flat["oos_gain_vs_q80"] = ref["oos"]["total_cost"] - summary["windows"]["oos"]["total_cost"]
        flat["full_gain_vs_q80"] = ref["full"]["total_cost"] - summary["windows"]["full"]["total_cost"]
        rows.append(flat)

    frame = pd.DataFrame(rows)
    frame.to_csv(ADAPTIVE_DIR / "oos_sweep.csv", index=False, encoding="utf-8-sig")
    survivors = frame[(frame.get("oos_gain_vs_q80", 0) > 0) & (frame.get("full_gain_vs_q80", 0) > 0)]
    verdict = {
        "n_in_sample_winners": int(len(winners)),
        "n_surviving_out_of_sample": int(len(survivors)),
        "surviving_labels": [str(x) for x in survivors["label"].tolist()],
        "fixed_q80": {name: ref[name]["total_cost"] for name in ("tune", "oos", "full")},
        "conclusion": (
            "no in-sample winner also beats fixed 0.8 out of sample"
            if survivors.empty
            else "at least one rule generalises; re-check before adopting"
        ),
    }
    (ADAPTIVE_DIR / "oos_sweep_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cols = [
        "label",
        "tune_total_cost",
        "tune_gain_vs_q80",
        "oos_total_cost",
        "oos_gain_vs_q80",
        "full_total_cost",
        "full_gain_vs_q80",
    ]
    print(frame[[c for c in cols if c in frame.columns]].to_string(index=False), flush=True)
    print(json.dumps(verdict, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
