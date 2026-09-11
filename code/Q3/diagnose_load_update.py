"""Read-only: how much of the intraday load information does the update rule use?

The frozen method compresses everything known about today into one scalar ratio
(`apply_morning_ratio`). This script measures the ceiling of that family and
compares it with a causal per-slot regression on the 0:00 residual.
Nothing here writes into the frozen result directories.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DT_HOURS, EXP_DIR
from load_data import load_prices, load_year_actuals
from load_xgb import ensure_xgb_load

MIN_HISTORY_DAYS = 20
OFFICIAL_FIRST_DAY = 31


def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred - truth
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
    }


def main() -> int:
    prices = load_prices()
    typical = prices["typical_load_kw"].to_numpy(dtype=float)
    year = load_year_actuals(
        jan1_load=float(typical[0]), jan1_pv=float(prices["typical_pv_kw"].iloc[0])
    )
    actual = year["load_kwh"] / DT_HOURS
    fc0 = ensure_xgb_load(actual, year["dates"], typical)
    residual = actual - fc0
    n_days = actual.shape[0]
    d0 = OFFICIAL_FIRST_DAY

    rows = []
    for hour in (6, 12, 18):
        s = hour * 6
        truth = actual[d0:, s:]
        base = fc0[d0:, s:]
        elapsed_mean = residual[:, :s].mean(axis=1)

        # Current rule: one multiplicative scalar from the elapsed means.
        el_f = fc0[d0:, :s].mean(axis=1)
        el_a = actual[d0:, :s].mean(axis=1)
        scalar = base * np.where(el_f > 1e-6, el_a / el_f, 1.0)[:, None]

        # Ceiling of the whole scalar family: best single ratio chosen in hindsight.
        best_ratio = ((truth * base).sum(axis=1) / np.maximum((base * base).sum(axis=1), 1e-9))
        oracle_scalar = base * best_ratio[:, None]

        # The elapsed ratio carries real signal but the frozen rule applies it at full
        # strength. Shrink it toward 1 by a single causally estimated factor.
        ratio_all = np.where(fc0[:, :s].mean(axis=1) > 1e-6,
                             actual[:, :s].mean(axis=1) / fc0[:, :s].mean(axis=1), 1.0)
        best_all = ((actual[:, s:] * fc0[:, s:]).sum(axis=1)
                    / np.maximum((fc0[:, s:] * fc0[:, s:]).sum(axis=1), 1e-9))
        shrunk = np.array(base, dtype=float, copy=True)
        shrink_used = np.full(n_days - d0, np.nan)
        for i, day in enumerate(range(d0, n_days)):
            if day < MIN_HISTORY_DAYS:
                continue
            u = ratio_all[:day] - 1.0
            v = best_all[:day] - 1.0
            k = float((u * v).sum() / max((u * u).sum(), 1e-12))
            shrink_used[i] = k
            shrunk[i] = base[i] * (1.0 + k * (ratio_all[day] - 1.0))

        # Causal per-slot regression of the remaining residual on the elapsed residual.
        reg = np.array(base, dtype=float, copy=True)
        for i, day in enumerate(range(d0, n_days)):
            if day < MIN_HISTORY_DAYS:
                continue
            x = elapsed_mean[:day]
            y = residual[:day, s:]
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()
            y_mean = y.mean(axis=0)
            slope = ((x - x_mean)[:, None] * (y - y_mean)).sum(axis=0) / max(x_var, 1e-9)
            reg[i] = base[i] + (y_mean - slope * x_mean) + slope * elapsed_mean[day]

        for name, pred in (
            ("raw XGB, no update", base),
            ("frozen: scalar ratio", scalar),
            ("shrunk ratio, causal factor", shrunk),
            ("causal per-slot regression", reg),
            ("ceiling of scalar family (hindsight)", oracle_scalar),
        ):
            rows.append({"issue_hour": hour, "variant": name, **_metrics(pred, truth)})
        print(f"issue {hour}:00 causal shrink factor: "
              f"median={np.nanmedian(shrink_used):.3f} last={shrink_used[-1]:.3f}")

    table = pd.DataFrame(rows)
    out_dir = EXP_DIR / "load_update_diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "intraday_update_accuracy.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))
    print(f"\nmean load {actual[d0:].mean():.1f} kW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
