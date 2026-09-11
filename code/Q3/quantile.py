"""Causal net-load residual quantiles, aligned to each PV-issue clock.

Residuals are actual minus point forecast (same sign as Q2). Load is shifted by
quantile q, PV by quantile 1-q. Offsets use only days strictly before the
decision day; the 48 h tail uses days whose actual next day has also closed.
Default policies leave q_lock=None and never import this into the frozen path
except via rolling's guarded branch.
"""

from __future__ import annotations

import numpy as np

from config import DT_HOURS, ISSUE_HOURS, LOCK_SLOTS, N_INTERVALS, SIGMA_MIN_SAMPLES
from load_forecast import horizon_load_kw, tomorrow_forecast_kw
from pv_forecast import aligned_actual_pv, tomorrow_pv_kw


def q_vector(
    n_horizon: int,
    n_today: int,
    q_lock: float,
    q_open: float,
    q_evening: float,
    lock_slots: int = LOCK_SLOTS,
) -> np.ndarray:
    q = np.empty(n_horizon, dtype=float)
    lock_n = min(int(lock_slots), n_today, n_horizon)
    q[:lock_n] = q_lock
    open_end = min(n_today, n_horizon)
    q[lock_n:open_end] = q_open
    if n_horizon > n_today:
        q[n_today:] = q_evening
    return q


def slot_quantile(stacked: np.ndarray, q, min_samples: int = SIGMA_MIN_SAMPLES) -> np.ndarray:
    stacked = np.asarray(stacked, dtype=float)
    if stacked.size == 0:
        n_slots = len(q) if not np.isscalar(q) else 0
        return np.zeros(n_slots)
    n_slots = stacked.shape[1]
    q_arr = np.full(n_slots, float(q)) if np.isscalar(q) else np.asarray(q, dtype=float)
    out = np.zeros(n_slots)
    for s in range(n_slots):
        col = stacked[:, s]
        col = col[~np.isnan(col)]
        if col.size < min_samples:
            out[s] = 0.0
        else:
            out[s] = float(np.quantile(col, float(q_arr[s])))
    return out


def apply_net_quantile(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    load_off: np.ndarray,
    pv_off: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.maximum(np.asarray(load_kw, dtype=float) + load_off, 0.0),
        np.maximum(np.asarray(pv_kw, dtype=float) + pv_off, 0.0),
    )


def _aligned_actual_load(actual_kw: np.ndarray) -> np.ndarray:
    n_days = actual_kw.shape[0]
    aligned = np.full((n_days, len(ISSUE_HOURS), N_INTERVALS), np.nan)
    for i, hour in enumerate(ISSUE_HOURS):
        start = hour * 6
        for d in range(n_days):
            for step in range(N_INTERVALS):
                slot = start + step
                if slot < N_INTERVALS:
                    aligned[d, i, step] = actual_kw[d, slot]
                elif d + 1 < n_days:
                    aligned[d, i, step] = actual_kw[d + 1, slot - N_INTERVALS]
    return aligned


def _load_point_24(load_kw: np.ndarray, dates, typical: np.ndarray) -> np.ndarray:
    n_days = load_kw.shape[0]
    out = np.zeros((n_days, len(ISSUE_HOURS), N_INTERVALS))
    for d in range(n_days):
        actual_today = load_kw[d]
        for i, hour in enumerate(ISSUE_HOURS):
            start = hour * 6
            out[d, i] = horizon_load_kw(
                load_kw,
                dates,
                typical,
                d,
                start,
                N_INTERVALS,
                actual_today if hour > 0 else None,
            )
    return out


def _tail_from_start(values: np.ndarray, start_slot: int, n_slots: int = N_INTERVALS) -> np.ndarray:
    extra = np.full(n_slots, np.nan)
    piece = np.asarray(values[start_slot:], dtype=float)
    extra[: piece.size] = piece
    return extra


class QuantileBank:
    """Per-issue residual cubes. offsets(day, ...) uses only closed history."""

    def __init__(
        self,
        load_resid24: np.ndarray,
        pv_resid24: np.ndarray,
        load_resid_extra: np.ndarray,
        pv_resid_extra: np.ndarray,
    ):
        self.load_resid24 = load_resid24
        self.pv_resid24 = pv_resid24
        self.load_resid_extra = load_resid_extra
        self.pv_resid_extra = pv_resid_extra

    @classmethod
    def build(cls, bundle: dict) -> "QuantileBank":
        load_kw = np.asarray(bundle["year"]["load_kwh"], dtype=float) / DT_HOURS
        dates = bundle["year"]["dates"]
        typical = bundle["prices"]["typical_load_kw"].to_numpy(dtype=float)
        interp = np.asarray(bundle["interp"], dtype=float)
        pv_actual = np.asarray(bundle["year"]["pv_kw"], dtype=float)
        n_days = load_kw.shape[0]

        load_hat = _load_point_24(load_kw, dates, typical)
        load_act = _aligned_actual_load(load_kw)
        pv_act = aligned_actual_pv(pv_actual, interp)
        load_resid24 = load_act - load_hat
        pv_resid24 = pv_act - interp

        load_resid_extra = np.full((n_days, len(ISSUE_HOURS), N_INTERVALS), np.nan)
        pv_resid_extra = np.full_like(load_resid_extra, np.nan)
        for d in range(n_days):
            tom_load = tomorrow_forecast_kw(load_kw, dates, typical, d)
            for i, hour in enumerate(ISSUE_HOURS):
                start = hour * 6
                hat = _tail_from_start(tom_load, start)
                hat_pv = _tail_from_start(tomorrow_pv_kw(pv_actual, d), start)
                if d + 1 < n_days:
                    act_load = _tail_from_start(load_kw[d + 1], start)
                    act_pv = _tail_from_start(pv_actual[d + 1], start)
                    load_resid_extra[d, i] = act_load - hat
                    pv_resid_extra[d, i] = act_pv - hat_pv
                else:
                    load_resid_extra[d, i] = np.nan
        return cls(load_resid24, pv_resid24, load_resid_extra, pv_resid_extra)

    def offsets(
        self,
        day: int,
        iss: int,
        n_horizon: int,
        q_vec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q_vec = np.asarray(q_vec, dtype=float)
        if q_vec.size != n_horizon:
            raise ValueError("q_vec length must equal n_horizon")
        load_off = np.zeros(n_horizon)
        pv_off = np.zeros(n_horizon)
        n24 = min(n_horizon, N_INTERVALS)
        if day > 0:
            load_off[:n24] = slot_quantile(self.load_resid24[:day, iss, :n24], q_vec[:n24])
            pv_off[:n24] = slot_quantile(self.pv_resid24[:day, iss, :n24], 1.0 - q_vec[:n24])
        extra_n = n_horizon - n24
        if extra_n > 0 and day > 1:
            load_off[n24:] = slot_quantile(
                self.load_resid_extra[: day - 1, iss, :extra_n], q_vec[n24:]
            )
            pv_off[n24:] = slot_quantile(
                self.pv_resid_extra[: day - 1, iss, :extra_n], 1.0 - q_vec[n24:]
            )
        return load_off, pv_off
