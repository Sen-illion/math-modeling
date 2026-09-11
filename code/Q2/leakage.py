"""Automatic future-leakage checks for Q2 forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURE_COLS, N_INTERVALS
from forecast import feature_matrix, predict_day, training_days


def _max_history_index_used(day: int) -> int:
    return day - 1


def check_feature_causality(mat: np.ndarray, dates: pd.Series, day: int) -> list[str]:
    errors = []
    if day <= 0:
        return errors
    feats = feature_matrix(mat, day, dates)
    if list(feats.columns) != FEATURE_COLS:
        errors.append("feature columns diverged from the frozen list")
    probe = mat.copy()
    probe[day:] = 1.0e9
    feats_probe = feature_matrix(probe, day, dates)
    if not np.allclose(feats.to_numpy(), feats_probe.to_numpy(), equal_nan=True):
        errors.append(f"day {day} features changed after editing day {day} and later actuals")
    if day + 1 < len(mat):
        future = mat.copy()
        future[day + 1] = 1.0e9
        if not np.allclose(feats.to_numpy(), feature_matrix(future, day, dates).to_numpy(), equal_nan=True):
            errors.append(f"day {day} features used day {day + 1} actuals")
    return errors


def check_training_window(day: int, window: str) -> list[str]:
    days = list(training_days(day, window))
    errors = []
    if any(d >= day for d in days):
        errors.append(f"{window}: training day >= forecast day {day}")
    if days and min(days) < 0:
        errors.append(f"{window}: negative training index")
    return errors


def check_prediction_immune_to_same_day(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    dates: pd.Series,
    fallback_load: np.ndarray,
    fallback_pv: np.ndarray,
    model_name: str,
    day: int,
) -> list[str]:
    errors = []
    cache = {}
    base = predict_day(load_kw, pv_kw, day, dates, fallback_load, fallback_pv, model_name, cache)
    tainted_load = load_kw.copy()
    tainted_pv = pv_kw.copy()
    tainted_load[day] = load_kw[day] + 5000.0
    tainted_pv[day] = pv_kw[day] + 5000.0
    cache2 = {}
    alt = predict_day(tainted_load, tainted_pv, day, dates, fallback_load, fallback_pv, model_name, cache2)
    if not np.allclose(base["load_kw"], alt["load_kw"], atol=1e-8):
        errors.append(f"{model_name} day {day} load forecast used same-day load")
    if not np.allclose(base["pv_kw"], alt["pv_kw"], atol=1e-8):
        errors.append(f"{model_name} day {day} PV forecast used same-day PV")
    if day + 1 < len(load_kw):
        future_load = load_kw.copy()
        future_load[day + 1] += 8000.0
        cache3 = {}
        alt_f = predict_day(future_load, pv_kw, day, dates, fallback_load, fallback_pv, model_name, cache3)
        if not np.allclose(base["load_kw"], alt_f["load_kw"], atol=1e-8):
            errors.append(f"{model_name} day {day} load forecast used a future day")
    return errors


def run_leakage_suite(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    dates: pd.Series,
    fallback_load: np.ndarray,
    fallback_pv: np.ndarray,
    probe_days: list[int],
    model_names: tuple[str, ...],
) -> dict:
    errors: list[str] = []
    for day in probe_days:
        errors.extend(check_feature_causality(load_kw, dates, day))
        errors.extend(check_feature_causality(pv_kw, dates, day))
        for window in ("expanding", "rolling30", "rolling60"):
            errors.extend(check_training_window(day, window))
        for name in model_names:
            errors.extend(
                check_prediction_immune_to_same_day(
                    load_kw, pv_kw, dates, fallback_load, fallback_pv, name, day
                )
            )
        if _max_history_index_used(day) >= day:
            errors.append(f"history index for day {day} is not strictly before D")
    return {
        "passed": len(errors) == 0,
        "n_errors": len(errors),
        "errors": errors,
        "probe_days": probe_days,
        "n_slots": N_INTERVALS,
        "note": "No random train/test split is used; models are refit on dates < D only.",
    }
