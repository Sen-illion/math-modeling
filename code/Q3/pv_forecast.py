"""Hourly PV forecasts interpolated to 10-minute slots with causal error sigma."""

from __future__ import annotations

import numpy as np

from config import (
    DT_HOURS,
    ISSUE_HOURS,
    ISSUE_P0_ZERO,
    N_INTERVALS,
    PV_P0_MODE,
    PV_P0_MODES,
    SIGMA_MIN_SAMPLES,
)


def interpolate_issue(fc24: np.ndarray, issue_h: int, p0_kw: float | None = None) -> np.ndarray:
    if p0_kw is None:
        p0 = 0.0 if issue_h in ISSUE_P0_ZERO else float(fc24[0])
    else:
        p0 = max(0.0, float(p0_kw))
    knots_x = np.arange(0.0, 25.0)
    knots_y = np.concatenate([[p0], np.asarray(fc24, dtype=float)])
    offsets = np.arange(N_INTERVALS) / 6.0
    return np.maximum(np.interp(offsets, knots_x, knots_y), 0.0)


def measured_p0_kw(actual_kw: np.ndarray, day: int, issue_h: int) -> float:
    """Last fully completed 10-minute mean before the issue clock."""
    slot = issue_h * 6 - 1
    if slot >= 0:
        return float(actual_kw[day, slot])
    if day > 0:
        return float(actual_kw[day - 1, N_INTERVALS - 1])
    return 0.0


def interpolate_all(
    hourly_kw: np.ndarray,
    actual_kw: np.ndarray | None = None,
    p0_mode: str = PV_P0_MODE,
) -> np.ndarray:
    n_days, n_issues, n_h = hourly_kw.shape
    if n_issues != len(ISSUE_HOURS) or n_h != 24:
        raise ValueError("hourly forecast shape must be (days, 4, 24)")
    if p0_mode not in PV_P0_MODES:
        raise ValueError(f"p0_mode must be one of {PV_P0_MODES}")
    if p0_mode == "measured" and actual_kw is None:
        raise ValueError("measured p0 needs the actual PV series")
    out = np.zeros((n_days, n_issues, N_INTERVALS))
    for d in range(n_days):
        for i, hour in enumerate(ISSUE_HOURS):
            p0 = measured_p0_kw(actual_kw, d, hour) if p0_mode == "measured" else None
            out[d, i] = interpolate_issue(hourly_kw[d, i], hour, p0)
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


def horizon_pv_kw(interp_kw: np.ndarray, day: int, iss: int, n_horizon: int) -> np.ndarray:
    """Issue-relative PV, then the rest of tomorrow from the next day's 0:00 issue.

    The first 144 slots are already issued at `iss`. Anything longer is calendar
    tomorrow from the same clock hour to 24:00, taken from day+1's 0:00 forecast.
    """
    first = np.asarray(interp_kw[day, iss], dtype=float)
    if n_horizon <= first.size:
        return first[:n_horizon].copy()
    extra_n = n_horizon - first.size
    start = ISSUE_HOURS[iss] * 6
    n_days = interp_kw.shape[0]
    if day + 1 < n_days:
        extra = np.asarray(interp_kw[day + 1, 0, start : start + extra_n], dtype=float)
        if extra.size < extra_n:
            extra = np.pad(extra, (0, extra_n - extra.size))
    else:
        extra = np.zeros(extra_n)
    return np.concatenate([first, extra])


def horizon_sigma(sigma: np.ndarray, iss: int, start_slot: int, n_horizon: int) -> np.ndarray:
    """Extend the 144-slot issue sigma with the 0:00-issue pattern for tomorrow."""
    first = np.asarray(sigma[iss], dtype=float)
    if n_horizon <= first.size:
        return first[:n_horizon].copy()
    extra_n = n_horizon - first.size
    extra = np.asarray(sigma[0, start_slot : start_slot + extra_n], dtype=float)
    if extra.size < extra_n:
        extra = np.pad(extra, (0, extra_n - extra.size))
    return np.concatenate([first, extra])
