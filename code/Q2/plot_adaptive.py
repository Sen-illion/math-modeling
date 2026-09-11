"""Self-reference feedback check for the adaptive quantile.

The em7 state feature is built from the policy's own realised emergency energy, so the
rule can in principle chase its own tail: a thin margin causes emergency, which thickens
the margin, which removes the emergency, which thins the margin again. This script plots
q_L(D) over time and reports three oscillation statistics so that behaviour is visible
rather than assumed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import ADAPTIVE_DIR, OFFICIAL_END, OFFICIAL_START, OOS_START, TUNE_END  # noqa: E402

FIG_DIR = ADAPTIVE_DIR / "figures"
ASSET_FIG_DIR = ROOT.parents[1] / "paper" / "assets" / "figures"


def run_lengths(values: np.ndarray) -> list[int]:
    if len(values) == 0:
        return []
    out = [1]
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            out[-1] += 1
        else:
            out.append(1)
    return out


def oscillation_stats(frame: pd.DataFrame, name: str) -> dict:
    q = frame["q_load"].to_numpy(dtype=float)
    if len(q) < 3:
        return {"window": name, "n_days": int(len(q))}
    switches = int(np.sum(q[1:] != q[:-1]))
    runs = run_lengths(q)
    # A two-cycle flip is q[i-1] == q[i+1] != q[i]: the rule undoes itself the next day.
    flips = int(np.sum((q[:-2] == q[2:]) & (q[1:-1] != q[:-2])))
    centred = q - q.mean()
    denom = float(np.sum(centred**2))
    acf1 = float(np.sum(centred[1:] * centred[:-1]) / denom) if denom > 0 else float("nan")
    return {
        "window": name,
        "n_days": int(len(q)),
        "n_switches": switches,
        "switch_rate": switches / (len(q) - 1),
        "n_two_day_flips": flips,
        "two_day_flip_rate": flips / (len(q) - 2),
        "mean_run_length": float(np.mean(runs)),
        "max_run_length": int(np.max(runs)),
        "lag1_autocorr": acf1,
        "mean_q": float(q.mean()),
        "std_q": float(q.std()),
        "q_counts": {str(k): int(v) for k, v in pd.Series(q).value_counts().sort_index().items()},
    }


def main() -> None:
    daily_path = ADAPTIVE_DIR / "daily_adaptive.csv"
    if not daily_path.exists():
        raise SystemExit(f"missing {daily_path}; run run_q2.py --adaptive-full first")
    daily = pd.read_csv(daily_path)
    daily["date"] = pd.to_datetime(daily["date"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_FIG_DIR.mkdir(parents=True, exist_ok=True)

    tune = daily[daily["date"] <= pd.Timestamp(TUNE_END)]
    oos = daily[daily["date"] >= pd.Timestamp(OOS_START)]
    rule_days = daily[daily["q_source"] == "rule"]
    stats = {
        "source": str(daily_path),
        "windows": [
            oscillation_stats(tune, "tune"),
            oscillation_stats(oos, "oos"),
            oscillation_stats(daily, "full"),
            oscillation_stats(rule_days, "rule_days_only"),
        ],
        "warmup_days": int((daily["q_source"] == "warmup").sum()),
        "verdict_note": (
            "two_day_flip_rate near or above 0.5 with mean_run_length near 1 would indicate "
            "self-referential oscillation; switch the state feature to resid_vol7 in that case."
        ),
    }
    full = stats["windows"][2]
    stats["oscillating"] = bool(full.get("mean_run_length", 99) < 1.5 and full.get("two_day_flip_rate", 0) > 0.5)
    (ADAPTIVE_DIR / "feedback_check.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True, height_ratios=[2.0, 1.4, 1.4])
    split = pd.Timestamp(OOS_START)

    ax = axes[0]
    ax.step(daily["date"], daily["q_load"], where="post", color="#1f4e79", lw=1.4)
    warm = daily[daily["q_source"] == "warmup"]
    if len(warm):
        ax.scatter(warm["date"], warm["q_load"], s=16, color="#c00000", zorder=3, label="warm start (fixed 0.8)")
    ax.axhline(0.8, color="#888888", ls="--", lw=1.0, label="frozen fixed 0.8")
    ax.axvline(split, color="#2e7d32", ls=":", lw=1.4)
    ax.set_ylabel(r"$q_L(D)$")
    ax.set_title("Adaptive day-level load quantile: tune window (left of dotted line) vs out-of-sample")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(daily["date"], daily["feature_percentile"], color="#7030a0", lw=1.0)
    ax.axvline(split, color="#2e7d32", ls=":", lw=1.4)
    ax.set_ylabel(r"$z(D)$ percentile")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.bar(daily["date"], daily["emergency_kwh"], width=1.0, color="#c55a11")
    ax.axvline(split, color="#2e7d32", ls=":", lw=1.4)
    ax.set_ylabel("emergency kWh")
    ax.set_xlabel("date (2025)")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        for base in (FIG_DIR, ASSET_FIG_DIR):
            fig.savefig(base / f"q2_adaptive_quantile_timeseries.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(stats["windows"], ensure_ascii=False, indent=2))
    print("oscillating:", stats["oscillating"])
    print("wrote", FIG_DIR, "and", ASSET_FIG_DIR)


if __name__ == "__main__":
    main()
