"""Causal Expanding XGBoost load forecasts. Same features as code/Q2/forecast.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from config import CLEAN_DIR, FEATURE_COLS, N_INTERVALS, RESULT_DIR, XGB_PARAMS

CACHE_NPY = CLEAN_DIR / "load_xgb_expanding_kw.npy"
DIAG_NPY = RESULT_DIR / "diagnostics" / "load_xgb_expanding_kw.npy"

_CACHE: np.ndarray | None = None


def _window(mat: np.ndarray, day: int, width: int) -> np.ndarray:
    start = max(0, day - width)
    return mat[start:day]


def _lag(mat: np.ndarray, day: int, lag: int) -> np.ndarray:
    src = day - lag
    if src < 0:
        return np.full(mat.shape[1], np.nan)
    return mat[src]


def feature_matrix(mat: np.ndarray, day: int, dates: pd.Series) -> pd.DataFrame:
    stamp = pd.Timestamp(dates.iloc[day])
    rows = []
    for slot in range(N_INTERVALS):
        w3 = _window(mat, day, 3)
        w7 = _window(mat, day, 7)
        w30 = _window(mat, day, 30)
        rows.append(
            {
                "slot": slot,
                "dow": int(stamp.dayofweek),
                "month": int(stamp.month),
                "doy": int(stamp.dayofyear),
                "lag_1d": float(_lag(mat, day, 1)[slot]),
                "lag_2d": float(_lag(mat, day, 2)[slot]),
                "lag_7d": float(_lag(mat, day, 7)[slot]),
                "mean_3d": float(w3.mean(axis=0)[slot] if len(w3) else np.nan),
                "mean_7d": float(w7.mean(axis=0)[slot] if len(w7) else np.nan),
                "std_7d": float(w7.std(axis=0, ddof=0)[slot] if len(w7) > 1 else np.nan),
                "mean_30d": float(w30.mean(axis=0)[slot] if len(w30) else np.nan),
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLS)


def _week_similar(load_kw: np.ndarray, day: int, dates: pd.Series, typical: np.ndarray) -> np.ndarray:
    from load_forecast import week_similar_index

    n_days = load_kw.shape[0]
    if day <= 0:
        return np.asarray(typical, dtype=float).copy()
    src = week_similar_index(day, dates, n_days)
    return load_kw[src].copy()


def causal_xgb_load(load_kw: np.ndarray, dates: pd.Series, typical: np.ndarray) -> np.ndarray:
    n_days = load_kw.shape[0]
    rows = [feature_matrix(load_kw, day, dates).assign(day=day, y=load_kw[day]) for day in range(n_days)]
    panel = pd.concat(rows, ignore_index=True)
    out = np.zeros_like(load_kw, dtype=float)
    out[0] = np.asarray(typical, dtype=float)
    for day in range(1, n_days):
        train = panel[panel["day"] < day]
        if len(train) < N_INTERVALS:
            out[day] = _week_similar(load_kw, day, dates, typical)
            continue
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(train[FEATURE_COLS], train["y"].to_numpy())
        pred = panel[panel["day"] == day]
        out[day] = np.maximum(np.asarray(model.predict(pred[FEATURE_COLS]), dtype=float), 0.0)
    return out


def ensure_xgb_load(load_kw: np.ndarray, dates: pd.Series, typical: np.ndarray) -> np.ndarray:
    global _CACHE
    if _CACHE is not None and _CACHE.shape == load_kw.shape:
        return _CACHE
    for path in (CACHE_NPY, DIAG_NPY):
        if path.exists():
            arr = np.load(path)
            if arr.shape == load_kw.shape:
                if path != CACHE_NPY:
                    CACHE_NPY.parent.mkdir(parents=True, exist_ok=True)
                    np.save(CACHE_NPY, arr)
                _CACHE = arr
                return arr
    arr = causal_xgb_load(load_kw, dates, typical)
    CACHE_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_NPY, arr)
    _CACHE = arr
    return arr


def predict_load_day(load_kw: np.ndarray, dates: pd.Series, typical: np.ndarray, day: int) -> np.ndarray:
    n_days = load_kw.shape[0]
    if day <= 0:
        return np.asarray(typical, dtype=float).copy()
    if day >= n_days:
        return _week_similar(load_kw, day, dates, typical)
    arr = ensure_xgb_load(load_kw, dates, typical)
    return arr[day].copy()
