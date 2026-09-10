"""Causal load forecasts: last-week same weekday, intra-day ratio update."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import N_INTERVALS


def week_similar_index(day: int, dates: pd.Series | pd.DatetimeIndex, n_days: int) -> int:
    if day < 0:
        raise ValueError("day must be non-negative")
    target_day = day
    target_dow = None
    stamps = pd.to_datetime(dates)
    if day < n_days:
        target_dow = int(pd.Timestamp(stamps.iloc[day]).dayofweek)
    else:
        # virtual next-year day: same weekday as the last in-sample day shifted by (day - last)
        last = pd.Timestamp(stamps.iloc[n_days - 1])
        virtual = last + pd.Timedelta(days=int(day - (n_days - 1)))
        target_dow = int(virtual.dayofweek)
    src = day - 7
    if 0 <= src < n_days:
        return src
    for d in range(min(day, n_days) - 1, -1, -1):
        if int(pd.Timestamp(stamps.iloc[d]).dayofweek) == target_dow:
            return d
    if min(day, n_days) > 0:
        return min(day, n_days) - 1
    return 0


def forecast_day_kw(
    load_kw: np.ndarray,
    day: int,
    dates: pd.Series | pd.DatetimeIndex,
    typical_kw: np.ndarray,
) -> np.ndarray:
    from load_xgb import predict_load_day

    n_days = load_kw.shape[0]
    if day <= 0 and n_days > 0:
        return np.asarray(typical_kw, dtype=float).copy()
    if day >= n_days:
        src = week_similar_index(day, dates, n_days)
        return load_kw[src].copy()
    return predict_load_day(load_kw, dates, typical_kw, day)


def apply_morning_ratio(forecast_kw: np.ndarray, actual_today_kw: np.ndarray, start_slot: int) -> np.ndarray:
    out = np.asarray(forecast_kw, dtype=float).copy()
    if start_slot <= 0:
        return out
    hist_f = float(np.mean(out[:start_slot]))
    hist_a = float(np.mean(actual_today_kw[:start_slot]))
    ratio = 1.0 if hist_f <= 1e-6 else hist_a / hist_f
    out[start_slot:] *= ratio
    return out


def horizon_load_kw(
    load_kw: np.ndarray,
    dates,
    typical_kw: np.ndarray,
    day: int,
    start_slot: int,
    n_horizon: int,
    actual_today_kw: np.ndarray | None,
) -> np.ndarray:
    today = forecast_day_kw(load_kw, day, dates, typical_kw)
    if actual_today_kw is not None and start_slot > 0:
        today = apply_morning_ratio(today, actual_today_kw, start_slot)
    tomorrow = forecast_day_kw(load_kw, day + 1, dates, typical_kw)
    wrapped = np.concatenate([today[start_slot:], tomorrow[:start_slot]])
    if len(wrapped) < n_horizon:
        raise ValueError("load horizon shorter than requested")
    return wrapped[:n_horizon]
