"""Constraint and accounting checks for a Q1 dispatch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ABS_TOL_KWH,
    E0_KWH,
    E_MAX_KWH,
    E_MIN_KWH,
    ETA,
    P_MAX_KWH,
    REL_TOL,
)


def _tol(scale: float) -> float:
    return max(ABS_TOL_KWH, REL_TOL * abs(scale))


def validate_dispatch(frame: pd.DataFrame, dispatch: dict, require_cost_below_b0: float | None = None) -> list[str]:
    errors: list[str] = []
    purchase = np.asarray(dispatch["purchase_kwh"], dtype=float)
    charge = np.asarray(dispatch["charge_kwh"], dtype=float)
    discharge = np.asarray(dispatch["discharge_kwh"], dtype=float)
    curtail = np.asarray(dispatch["curtail_kwh"], dtype=float)
    soc = np.asarray(dispatch["soc_end_kwh"], dtype=float)
    load = frame["load_kwh"].to_numpy()
    pv = frame["pv_kwh"].to_numpy()
    price = frame["price"].to_numpy()

    if len(purchase) != len(frame):
        return [f"dispatch length {len(purchase)} != {len(frame)}"]

    soc_prev = E0_KWH
    for t in range(len(frame)):
        residual = purchase[t] + pv[t] + ETA * discharge[t] - load[t] - charge[t] / ETA - curtail[t]
        if abs(residual) > _tol(max(load[t], 1.0)):
            errors.append(f"t={t+1} energy balance residual {residual:.6f} kWh")
        soc_expected = soc_prev + charge[t] - discharge[t]
        if abs(soc_expected - soc[t]) > _tol(max(soc[t], 1.0)):
            errors.append(f"t={t+1} SOC update mismatch {soc_expected:.6f} vs {soc[t]:.6f}")
        if soc[t] < E_MIN_KWH - ABS_TOL_KWH or soc[t] > E_MAX_KWH + ABS_TOL_KWH:
            errors.append(f"t={t+1} SOC {soc[t]:.4f} outside [{E_MIN_KWH}, {E_MAX_KWH}]")
        if charge[t] < -ABS_TOL_KWH or discharge[t] < -ABS_TOL_KWH or purchase[t] < -ABS_TOL_KWH:
            errors.append(f"t={t+1} negative purchase/charge/discharge")
        if charge[t] > P_MAX_KWH + ABS_TOL_KWH or discharge[t] > P_MAX_KWH + ABS_TOL_KWH:
            errors.append(f"t={t+1} charge/discharge exceeds {P_MAX_KWH:.4f} kWh")
        if charge[t] * discharge[t] > 1e-4:
            errors.append(f"t={t+1} simultaneous charge and discharge")
        soc_prev = soc[t]

    if abs(soc[0] - (E0_KWH + charge[0] - discharge[0])) > _tol(E0_KWH):
        errors.append("initial SOC step is inconsistent")
    if abs(soc[-1] - E0_KWH) > _tol(E0_KWH):
        errors.append(f"final SOC {soc[-1]:.6f} != {E0_KWH}")

    reported_cost = float(np.dot(price, purchase))
    if "cost" in dispatch and abs(dispatch["cost"] - reported_cost) > _tol(max(reported_cost, 1.0)):
        errors.append("reported cost does not match sum(price * purchase)")

    if require_cost_below_b0 is not None and reported_cost > require_cost_below_b0 + _tol(require_cost_below_b0):
        errors.append(f"M0 cost {reported_cost:.4f} is not below B0 cost {require_cost_below_b0:.4f}")

    return errors


def summarize(frame: pd.DataFrame, dispatch: dict, name: str) -> dict:
    purchase = np.asarray(dispatch["purchase_kwh"], dtype=float)
    return {
        "name": name,
        "purchase_kwh": float(purchase.sum()),
        "cost": float(np.dot(frame["price"].to_numpy(), purchase)),
        "charge_kwh": float(np.asarray(dispatch["charge_kwh"]).sum()),
        "discharge_kwh": float(np.asarray(dispatch["discharge_kwh"]).sum()),
        "curtail_kwh": float(np.asarray(dispatch["curtail_kwh"]).sum()),
        "soc0_kwh": E0_KWH,
        "soc24_kwh": float(dispatch["soc_end_kwh"][-1]),
    }
