"""Causal intra-day execution with locked purchase. Same rule as code/Q2/simulate.py."""

from __future__ import annotations

import numpy as np

from config import (
    ABS_TOL_KWH,
    E_MAX_KWH,
    E_MIN_KWH,
    ETA_CHARGE,
    ETA_DISCHARGE,
    N_INTERVALS,
    P_MAX_KWH,
    REL_TOL,
)


def _clip_soc(soc: float) -> float:
    return float(min(E_MAX_KWH, max(E_MIN_KWH, soc)))


def simulate_range(
    load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    purchase_kwh: np.ndarray,
    soc0: float,
    t0: int,
    t1: int,
) -> dict:
    """Dispatch slots [t0, t1) using only current-slot actuals. Emergency is not used to charge."""
    n = t1 - t0
    charge = np.zeros(n)
    discharge = np.zeros(n)
    emergency = np.zeros(n)
    curtail = np.zeros(n)
    soc = np.zeros(n)
    soc_prev = float(soc0)
    if soc_prev < E_MIN_KWH - ABS_TOL_KWH or soc_prev > E_MAX_KWH + ABS_TOL_KWH:
        raise ValueError(f"actual SOC0 {soc_prev} outside [{E_MIN_KWH}, {E_MAX_KWH}]")

    for i, t in enumerate(range(t0, t1)):
        residual = load_kwh[t] - pv_kwh[t] - purchase_kwh[t]
        if residual <= 0:
            surplus = -residual
            charge_cap = min(P_MAX_KWH, max(0.0, (E_MAX_KWH - soc_prev) / ETA_CHARGE))
            charge[i] = min(surplus, charge_cap)
            curtail[i] = surplus - charge[i]
        else:
            discharge_cap = min(P_MAX_KWH, max(0.0, ETA_DISCHARGE * (soc_prev - E_MIN_KWH)))
            discharge[i] = min(residual, discharge_cap)
            emergency[i] = residual - discharge[i]
        soc_prev = _clip_soc(soc_prev + ETA_CHARGE * charge[i] - discharge[i] / ETA_DISCHARGE)
        soc[i] = soc_prev

    return {
        "charge_kwh": charge,
        "discharge_kwh": discharge,
        "emergency_kwh": emergency,
        "curtail_kwh": curtail,
        "soc_end_kwh": soc,
        "soc_last_kwh": float(soc_prev),
    }


def simulate_day(
    price: np.ndarray,
    load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    purchase_kwh: np.ndarray,
    soc0: float,
) -> dict:
    n = N_INTERVALS
    piece = simulate_range(load_kwh, pv_kwh, purchase_kwh, soc0, 0, n)
    shortage = (
        load_kwh
        - pv_kwh
        - purchase_kwh
        - piece["discharge_kwh"]
        + piece["charge_kwh"]
        + piece["curtail_kwh"]
        - piece["emergency_kwh"]
    )
    return {
        "charge_kwh": piece["charge_kwh"],
        "discharge_kwh": piece["discharge_kwh"],
        "emergency_kwh": piece["emergency_kwh"],
        "curtail_kwh": piece["curtail_kwh"],
        "soc_end_kwh": piece["soc_end_kwh"],
        "soc0_kwh": float(soc0),
        "soc24_kwh": float(piece["soc_last_kwh"]),
        "plan_cost": float(np.dot(price, purchase_kwh)),
        "emergency_cost": float(np.dot(5.0 * price, piece["emergency_kwh"])),
        "n_emergency_slots": int(np.sum(piece["emergency_kwh"] > ABS_TOL_KWH)),
        "max_unserved_kwh": float(np.max(np.maximum(shortage, 0.0))),
    }


def validate_actual(actual: dict, purchase: np.ndarray, load_kwh: np.ndarray, pv_kwh: np.ndarray) -> list[str]:
    errors = []
    tol = lambda scale: max(ABS_TOL_KWH, REL_TOL * abs(scale))
    soc_prev = actual["soc0_kwh"]
    for t in range(N_INTERVALS):
        residual = (
            purchase[t]
            + pv_kwh[t]
            + actual["discharge_kwh"][t]
            + actual["emergency_kwh"][t]
            - load_kwh[t]
            - actual["charge_kwh"][t]
            - actual["curtail_kwh"][t]
        )
        if abs(residual) > tol(max(load_kwh[t], 1.0)):
            errors.append(f"actual t={t} energy residual {residual}")
        soc_expected = soc_prev + ETA_CHARGE * actual["charge_kwh"][t] - actual["discharge_kwh"][t] / ETA_DISCHARGE
        if abs(soc_expected - actual["soc_end_kwh"][t]) > tol(max(actual["soc_end_kwh"][t], 1.0)):
            errors.append(f"actual t={t} SOC mismatch")
        if actual["soc_end_kwh"][t] < E_MIN_KWH - ABS_TOL_KWH or actual["soc_end_kwh"][t] > E_MAX_KWH + ABS_TOL_KWH:
            errors.append(f"actual t={t} SOC out of bounds {actual['soc_end_kwh'][t]}")
        if actual["charge_kwh"][t] > P_MAX_KWH + ABS_TOL_KWH or actual["discharge_kwh"][t] > P_MAX_KWH + ABS_TOL_KWH:
            errors.append(f"actual t={t} power out of bounds")
        if actual["charge_kwh"][t] > ABS_TOL_KWH and actual["discharge_kwh"][t] > ABS_TOL_KWH:
            errors.append(f"actual t={t} simultaneous charge/discharge")
        if actual["emergency_kwh"][t] < -ABS_TOL_KWH:
            errors.append(f"actual t={t} negative emergency")
        soc_prev = actual["soc_end_kwh"][t]
    if actual["max_unserved_kwh"] > ABS_TOL_KWH:
        errors.append(f"unserved energy {actual['max_unserved_kwh']}")
    return errors
