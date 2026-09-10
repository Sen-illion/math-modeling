"""Day-ahead purchase LP on predicted load/PV. AC-side charge/discharge."""

from __future__ import annotations

import numpy as np
import pulp

from config import (
    ABS_TOL_KWH,
    E_MAX_KWH,
    E_MIN_KWH,
    ETA_CHARGE,
    ETA_DISCHARGE,
    N_INTERVALS,
    P_MAX_KWH,
    REL_TOL,
    SIMULTANEOUS_TOL,
    SOLVER_TIME_LIMIT_S,
    TERMINAL_LAMBDA,
    TERMINAL_TARGET_KWH,
)


def solve_day_lp(
    price: np.ndarray,
    load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    soc0: float,
    terminal_mode: str = "none",
) -> dict:
    n = N_INTERVALS
    if len(price) != n or len(load_kwh) != n or len(pv_kwh) != n:
        raise ValueError("LP input length must be 144")
    if soc0 < E_MIN_KWH - ABS_TOL_KWH or soc0 > E_MAX_KWH + ABS_TOL_KWH:
        raise ValueError(f"planned SOC0 {soc0} outside [{E_MIN_KWH}, {E_MAX_KWH}]")

    prob = pulp.LpProblem("q2_day_ahead", pulp.LpMinimize)
    g = [pulp.LpVariable(f"G_{t}", lowBound=0) for t in range(n)]
    c = [pulp.LpVariable(f"c_{t}", lowBound=0, upBound=P_MAX_KWH) for t in range(n)]
    d = [pulp.LpVariable(f"d_{t}", lowBound=0, upBound=P_MAX_KWH) for t in range(n)]
    w = [pulp.LpVariable(f"w_{t}", lowBound=0) for t in range(n)]
    e = [pulp.LpVariable(f"E_{t}", lowBound=E_MIN_KWH, upBound=E_MAX_KWH) for t in range(n)]

    eps = 1e-7
    obj = pulp.lpSum(price[t] * g[t] + eps * (c[t] + d[t]) for t in range(n))
    if terminal_mode == "track6000":
        surplus = pulp.LpVariable("term_pos", lowBound=0)
        shortage = pulp.LpVariable("term_neg", lowBound=0)
        obj += TERMINAL_LAMBDA * (surplus + shortage)
    elif terminal_mode != "none":
        raise ValueError(f"unknown terminal_mode {terminal_mode}")
    prob += obj

    for t in range(n):
        prev = soc0 if t == 0 else e[t - 1]
        prob += g[t] + pv_kwh[t] + d[t] == load_kwh[t] + c[t] + w[t]
        prob += e[t] == prev + ETA_CHARGE * c[t] - d[t] / ETA_DISCHARGE
    if terminal_mode == "track6000":
        prob += e[n - 1] - TERMINAL_TARGET_KWH == surplus - shortage

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=SOLVER_TIME_LIMIT_S))
    status_name = pulp.LpStatus[status]
    if status_name != "Optimal":
        raise RuntimeError(f"Q2 LP not Optimal: {status_name}")

    purchase = np.array([float(pulp.value(g[t])) for t in range(n)])
    charge = np.array([float(pulp.value(c[t])) for t in range(n)])
    discharge = np.array([float(pulp.value(d[t])) for t in range(n)])
    curtail = np.array([float(pulp.value(w[t])) for t in range(n)])
    soc = np.array([float(pulp.value(e[t])) for t in range(n)])
    simultaneous = int(np.sum((charge > SIMULTANEOUS_TOL) & (discharge > SIMULTANEOUS_TOL)))
    return {
        "status": status_name,
        "purchase_kwh": purchase,
        "charge_kwh": charge,
        "discharge_kwh": discharge,
        "curtail_kwh": curtail,
        "soc_end_kwh": soc,
        "soc0_kwh": float(soc0),
        "plan_cost": float(np.dot(price, purchase)),
        "n_simultaneous": simultaneous,
        "terminal_mode": terminal_mode,
        "terminal_abs_dev": abs(float(soc[-1]) - TERMINAL_TARGET_KWH),
    }


def _tol(scale: float) -> float:
    return max(ABS_TOL_KWH, REL_TOL * abs(scale))


def validate_plan(plan: dict, price: np.ndarray, load_kwh: np.ndarray, pv_kwh: np.ndarray) -> list[str]:
    errors = []
    soc_prev = plan["soc0_kwh"]
    for t in range(N_INTERVALS):
        residual = (
            plan["purchase_kwh"][t]
            + pv_kwh[t]
            + plan["discharge_kwh"][t]
            - load_kwh[t]
            - plan["charge_kwh"][t]
            - plan["curtail_kwh"][t]
        )
        if abs(residual) > _tol(max(load_kwh[t], 1.0)):
            errors.append(f"plan t={t} energy residual {residual}")
        soc_expected = soc_prev + ETA_CHARGE * plan["charge_kwh"][t] - plan["discharge_kwh"][t] / ETA_DISCHARGE
        if abs(soc_expected - plan["soc_end_kwh"][t]) > _tol(max(plan["soc_end_kwh"][t], 1.0)):
            errors.append(f"plan t={t} SOC mismatch")
        if plan["soc_end_kwh"][t] < E_MIN_KWH - ABS_TOL_KWH or plan["soc_end_kwh"][t] > E_MAX_KWH + ABS_TOL_KWH:
            errors.append(f"plan t={t} SOC out of bounds")
        if plan["charge_kwh"][t] > P_MAX_KWH + ABS_TOL_KWH or plan["discharge_kwh"][t] > P_MAX_KWH + ABS_TOL_KWH:
            errors.append(f"plan t={t} power out of bounds")
        soc_prev = plan["soc_end_kwh"][t]
    return errors
