"""Causal same-slot baseline and XGBoost forecasts. No same-day actuals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from config import FEATURE_COLS, N_INTERVALS, XGB_PARAMS


def _available_window(mat: np.ndarray, day: int, width: int) -> np.ndarray:
    start = max(0, day - width)
    if start >= day:
        return np.empty((0, mat.shape[1]))
    return mat[start:day]


def baseline_7d(mat: np.ndarray, day: int, fallback: np.ndarray) -> np.ndarray:
    window = _available_window(mat, day, 7)
    if len(window) == 0:
        return fallback.copy()
    return window.mean(axis=0)


def _window_mean(mat: np.ndarray, day: int, width: int) -> np.ndarray:
    window = _available_window(mat, day, width)
    if len(window) == 0:
        return np.full(mat.shape[1], np.nan)
    return window.mean(axis=0)


def _window_std(mat: np.ndarray, day: int, width: int) -> np.ndarray:
    window = _available_window(mat, day, width)
    if len(window) == 0:
        return np.full(mat.shape[1], np.nan)
    if len(window) == 1:
        return np.zeros(mat.shape[1])
    return window.std(axis=0, ddof=0)


def _lag(mat: np.ndarray, day: int, lag: int) -> np.ndarray:
    src = day - lag
    if src < 0:
        return np.full(mat.shape[1], np.nan)
    return mat[src]


def feature_row(mat: np.ndarray, day: int, slot: int, dates: pd.Series) -> dict:
    stamp = pd.Timestamp(dates.iloc[day])
    return {
        "slot": slot,
        "dow": int(stamp.dayofweek),
        "month": int(stamp.month),
        "doy": int(stamp.dayofyear),
        "lag_1d": float(_lag(mat, day, 1)[slot]),
        "lag_2d": float(_lag(mat, day, 2)[slot]),
        "lag_7d": float(_lag(mat, day, 7)[slot]),
        "mean_3d": float(_window_mean(mat, day, 3)[slot]),
        "mean_7d": float(_window_mean(mat, day, 7)[slot]),
        "std_7d": float(_window_std(mat, day, 7)[slot]),
        "mean_30d": float(_window_mean(mat, day, 30)[slot]),
    }


def feature_matrix(mat: np.ndarray, day: int, dates: pd.Series) -> pd.DataFrame:
    rows = [feature_row(mat, day, slot, dates) for slot in range(N_INTERVALS)]
    return pd.DataFrame(rows, columns=FEATURE_COLS)


def precompute_panel(mat: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    rows = []
    n_days = len(dates)
    for day in range(n_days):
        feats = feature_matrix(mat, day, dates)
        feats["day"] = day
        feats["y"] = mat[day]
        rows.append(feats)
    return pd.concat(rows, ignore_index=True)


def training_days(day: int, window: str) -> range:
    if window == "expanding":
        start = 0
    elif window == "rolling30":
        start = max(0, day - 30)
    elif window == "rolling60":
        start = max(0, day - 60)
    else:
        raise ValueError(window)
    return range(start, day)


def _fit_from_panel(panel: pd.DataFrame, day: int, window: str) -> xgb.XGBRegressor | None:
    start = 0 if window == "expanding" else max(0, day - (30 if window == "rolling30" else 60))
    train = panel[(panel["day"] >= start) & (panel["day"] < day)]
    if len(train) < N_INTERVALS:
        return None
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train[FEATURE_COLS], train["y"].to_numpy())
    return model


def apply_pv_night_zero(pred: np.ndarray, history: np.ndarray) -> np.ndarray:
    out = pred.copy()
    if len(history) == 0:
        return out
    out[history.max(axis=0) <= 1e-8] = 0.0
    return out


def clip_nonneg(pred: np.ndarray) -> np.ndarray:
    return np.maximum(pred, 0.0)


def predict_day(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    day: int,
    dates: pd.Series,
    fallback_load: np.ndarray,
    fallback_pv: np.ndarray,
    model_name: str,
    cache: dict,
    load_panel: pd.DataFrame | None = None,
    pv_panel: pd.DataFrame | None = None,
) -> dict:
    load_base = baseline_7d(load_kw, day, fallback_load)
    pv_base = baseline_7d(pv_kw, day, fallback_pv)
    history_pv = pv_kw[:day]

    if model_name == "baseline_7d":
        return {
            "load_kw": clip_nonneg(load_base),
            "pv_kw": clip_nonneg(apply_pv_night_zero(pv_base, history_pv)),
            "xgb_used": False,
        }

    window = {
        "xgb_expanding": "expanding",
        "xgb_rolling30": "rolling30",
        "xgb_rolling60": "rolling60",
    }[model_name]
    key = (model_name, day)
    if key not in cache:
        if load_panel is None:
            load_panel = precompute_panel(load_kw[: day + 1], dates.iloc[: day + 1])
        if pv_panel is None:
            pv_panel = precompute_panel(pv_kw[: day + 1], dates.iloc[: day + 1])
        cache[key] = {
            "load": _fit_from_panel(load_panel, day, window),
            "pv": _fit_from_panel(pv_panel, day, window),
        }

    load_model = cache[key]["load"]
    pv_model = cache[key]["pv"]
    if load_panel is None:
        x_load = feature_matrix(load_kw, day, dates)
        x_pv = feature_matrix(pv_kw, day, dates)
    else:
        x_load = load_panel[load_panel["day"] == day][FEATURE_COLS]
        x_pv = pv_panel[pv_panel["day"] == day][FEATURE_COLS]

    xgb_used = True
    if load_model is None:
        load_hat = load_base
        xgb_used = False
    else:
        load_hat = np.asarray(load_model.predict(x_load), dtype=float)
    if pv_model is None:
        pv_hat = pv_base
        xgb_used = False
    else:
        pv_hat = np.asarray(pv_model.predict(x_pv), dtype=float)

    return {
        "load_kw": clip_nonneg(load_hat),
        "pv_kw": clip_nonneg(apply_pv_night_zero(pv_hat, history_pv)),
        "xgb_used": xgb_used,
    }
