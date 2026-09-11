"""Hourly PV forecasts interpolated to 10-minute slots with causal error sigma."""

from __future__ import annotations

import numpy as np

from config import (
    DT_HOURS,
    ISSUE_HOURS,
    ISSUE_P0_ZERO,
    N_INTERVALS,
    SIGMA_MIN_SAMPLES,
)


def interpolate_issue(fc24: np.ndarray, issue_h: int) -> np.ndarray:
    p0 = 0.0 if issue_h in ISSUE_P0_ZERO else float(fc24[0])
    knots_x = np.arange(0.0, 25.0)
    knots_y = np.concatenate([[p0], np.asarray(fc24, dtype=float)])
    offsets = np.arange(1, N_INTERVALS + 1) / 6.0
    return np.maximum(np.interp(offsets, knots_x, knots_y), 0.0)


def interpolate_all(hourly_kw: np.ndarray) -> np.ndarray:
    n_days, n_issues, n_h = hourly_kw.shape
    if n_issues != len(ISSUE_HOURS) or n_h != 24:
        raise ValueError("hourly forecast shape must be (days, 4, 24)")
    out = np.zeros((n_days, n_issues, N_INTERVALS))
    for d in range(n_days):
        for i, hour in enumerate(ISSUE_HOURS):
            out[d, i] = interpolate_issue(hourly_kw[d, i], hour)
    return out


def aligned_actual_pv(actual_kw: np.ndarray, interp_kw: np.ndarray) -> np.ndarray:
    n_days = actual_kw.shape[0]
    aligned = np.full_like(interp_kw, np.nan)
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


def causal_sigma(interp_kw: np.ndarray, aligned_actual: np.ndarray, day: int) -> np.ndarray:
    if day <= 0:
        return np.zeros((len(ISSUE_HOURS), N_INTERVALS))
    err = interp_kw[:day] - aligned_actual[:day]
    n_obs = np.sum(~np.isnan(err), axis=0)
    with np.errstate(all="ignore"):
        sigma = np.nanstd(err, axis=0, ddof=0)
    sigma = np.where(np.isnan(sigma), 0.0, sigma)
    return np.where(n_obs >= SIGMA_MIN_SAMPLES, sigma, 0.0)


def conservative_pv_kwh(
    interp_kw_issue: np.ndarray,
    sigma_issue: np.ndarray,
    beta_lock: float,
    beta_open: float,
    lock_slots: int,
) -> np.ndarray:
    beta = np.full_like(interp_kw_issue, beta_open)
    beta[: min(lock_slots, len(interp_kw_issue))] = beta_lock
    pv_kw = np.maximum(0.0, interp_kw_issue - beta * sigma_issue)
    return pv_kw * DT_HOURS
