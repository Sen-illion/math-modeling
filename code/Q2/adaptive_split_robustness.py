"""How much does the nested protocol's answer depend on where the inner split falls?

The nested inner/select protocol was written after the out-of-sample sweep had already
been seen, so its success is not a clean out-of-sample result on its own. What can still
be checked honestly is whether the protocol is stable: if every reasonable position of
the inner split picks the same rule, then the specific split date carried no information
and the contamination is confined to the choice of protocol family.

Scoring is free here: tune_daily_all.csv already holds per-day costs for every candidate
over 02-01..06-30, and each sub-window score is a slice of that same rolled trajectory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from config import ADAPTIVE_DIR, OFFICIAL_START, TUNE_END  # noqa: E402

BASELINE = "fixed_q80"


def window_cost(daily: pd.DataFrame, start, end) -> pd.Series:
    mask = (daily["date"] >= pd.Timestamp(start)) & (daily["date"] <= pd.Timestamp(end))
    return daily[mask].groupby("label")["total_cost"].sum()


def main() -> None:
    daily = pd.read_csv(ADAPTIVE_DIR / "tune_daily_all.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    adaptive_labels = sorted(set(daily.loc[daily["q_source"] != "fixed", "label"]))

    rows = []
    for inner_end in pd.date_range("2025-03-15", "2025-05-31", freq="7D"):
        select_start = inner_end + pd.Timedelta(days=1)
        inner = window_cost(daily, OFFICIAL_START, inner_end)
        select = window_cost(daily, select_start, TUNE_END)
        eligible = [lab for lab in adaptive_labels if inner[lab] < inner[BASELINE]]
        pool = eligible or adaptive_labels
        pick = min(pool, key=lambda lab: select[lab])
        rows.append(
            {
                "inner_end": inner_end.date().isoformat(),
                "select_start": select_start.date().isoformat(),
                "inner_days": int(((daily["date"] >= pd.Timestamp(OFFICIAL_START)) & (daily["date"] <= inner_end)).sum() / daily["label"].nunique()),
                "n_eligible": len(eligible),
                "fell_back": not eligible,
                "pick": pick,
                "pick_select_gain_vs_q80": float(select[BASELINE] - select[pick]),
                "pick_inner_gain_vs_q80": float(inner[BASELINE] - inner[pick]),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(ADAPTIVE_DIR / "split_robustness.csv", index=False, encoding="utf-8-sig")

    sweep_path = ADAPTIVE_DIR / "oos_sweep.csv"
    known_oos = {}
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        if "full_gain_vs_q80" in sweep.columns:
            known_oos = {
                str(r.label): {"oos_gain": float(r.oos_gain_vs_q80), "full_gain": float(r.full_gain_vs_q80)}
                for r in sweep.dropna(subset=["full_gain_vs_q80"]).itertuples()
            }

    counts = frame["pick"].value_counts()
    verdict = {
        "n_splits_tested": int(len(frame)),
        "distinct_picks": int(frame["pick"].nunique()),
        "pick_counts": {str(k): int(v) for k, v in counts.items()},
        "modal_pick": str(counts.index[0]),
        "modal_share": float(counts.iloc[0] / len(frame)),
        "known_out_of_sample": {lab: known_oos.get(lab) for lab in counts.index.astype(str)},
        "picks_with_unknown_oos": [lab for lab in counts.index.astype(str) if lab not in known_oos],
        "all_picks_share_one_feature": frame["pick"].str.split("_qmin").str[0].nunique() == 1,
    }
    (ADAPTIVE_DIR / "split_robustness_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
