"""Build causal 0:00 price forecasts and actual settlement prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .load_data import load_att1_curve, load_year_prices
from .price_structured import forecast_hat0


def load_price_bank() -> dict:
    cold = load_att1_curve()
    year = load_year_prices(cold)
    dates = pd.to_datetime(year["dates"])
    dow = dates.dt.weekday.to_numpy()
    pack = forecast_hat0(year["price"], dow, cold)
    if pack["hat0"].shape != year["price"].shape:
        raise ValueError("hat0 shape mismatch")
    # Official days must not plan on a frozen attachment-1 curve.
    feb1 = int(np.where(dates == pd.Timestamp("2025-02-01"))[0][0])
    if np.allclose(pack["hat0"][feb1], cold):
        raise RuntimeError("Feb 1 hat0 collapsed to attachment-1 cold start")
    if np.allclose(pack["b"][feb1], year["price"].mean(axis=0)):
        raise RuntimeError("Feb 1 b used the full-year mean")
    return {
        "dates": dates,
        "actual": year["price"],
        "hat0": pack["hat0"],
        "b": pack["b"],
        "cold_start": cold,
        "source": year["source"],
        "alignment": year["alignment"],
        "n_base_used": pack["n_base_used"],
    }
