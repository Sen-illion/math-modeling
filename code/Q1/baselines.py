"""Baseline purchase policies for Q1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import E0_KWH, E_MAX_KWH, E_MIN_KWH, ETA, P_MAX_KWH


def _empty_result(n: int) -> dict[str, np.ndarray]:
    return {
        "purchase_kwh": np.zeros(n),
        "charge_kwh": np.zeros(n),
        "discharge_kwh": np.zeros(n),
        "curtail_kwh": np.zeros(n),
        "soc_end_kwh": np.zeros(n),
    }


def baseline_no_storage(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(frame)
    result = _empty_result(n)
    net = frame["load_kwh"].to_numpy() - frame["pv_kwh"].to_numpy()
    result["purchase_kwh"] = np.maximum(net, 0.0)
    result["curtail_kwh"] = np.maximum(-net, 0.0)
    result["soc_end_kwh"] = np.full(n, E0_KWH)
    return result


def baseline_tou_greedy(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Charge in cheap hours, discharge in expensive hours, then restore SOC if possible."""
    n = len(frame)
    price = frame["price"].to_numpy()
    load = frame["load_kwh"].to_numpy()
    pv = frame["pv_kwh"].to_numpy()
    q_low = float(np.quantile(price, 0.33))
    q_high = float(np.quantile(price, 0.67))

    result = _empty_result(n)
    soc = E0_KWH
    for t in range(n):
        surplus = max(pv[t] - load[t], 0.0)
        deficit = max(load[t] - pv[t], 0.0)
        charge_cap = min(P_MAX_KWH, E_MAX_KWH - soc)
        discharge_cap = min(P_MAX_KWH, soc - E_MIN_KWH)

        charge = 0.0
        discharge = 0.0
        purchase = 0.0
        curtail = 0.0

        if price[t] <= q_low:
            charge = charge_cap
            ac_charge = charge / ETA
            if surplus >= ac_charge:
                curtail = surplus - ac_charge
                purchase = deficit
            else:
                purchase = deficit + (ac_charge - surplus)
        elif price[t] >= q_high:
            discharge = min(discharge_cap, deficit / ETA if ETA > 0 else 0.0)
            remain = deficit - ETA * discharge
            purchase = max(remain, 0.0)
            if surplus > 0 and discharge == 0:
                charge = min(charge_cap, surplus * ETA)
                curtail = surplus - charge / ETA
            else:
                curtail = surplus
        else:
            purchase = deficit
            charge = min(charge_cap, surplus * ETA)
            curtail = surplus - charge / ETA

        soc = soc + charge - discharge
        result["purchase_kwh"][t] = purchase
        result["charge_kwh"][t] = charge
        result["discharge_kwh"][t] = discharge
        result["curtail_kwh"][t] = curtail
        result["soc_end_kwh"][t] = soc

    _restore_cyclic_soc(frame, result)
    return result


def _restore_cyclic_soc(frame: pd.DataFrame, result: dict[str, np.ndarray]) -> None:
    soc_end = result["soc_end_kwh"]
    gap = soc_end[-1] - E0_KWH
    if abs(gap) <= 1e-6:
        return

    price = frame["price"].to_numpy()
    order = np.argsort(price) if gap < 0 else np.argsort(-price)
    need = abs(gap)
    soc = np.concatenate(([E0_KWH], soc_end))

    for t in order:
        if need <= 1e-9:
            break
        if gap < 0:
            room = min(P_MAX_KWH - result["charge_kwh"][t], E_MAX_KWH - soc[t + 1], need)
            room = max(room, 0.0)
            if room <= 0 or result["discharge_kwh"][t] > 1e-9:
                continue
            result["charge_kwh"][t] += room
            result["purchase_kwh"][t] += room / ETA
            need -= room
            soc[t + 1 :] += room
        else:
            room = min(P_MAX_KWH - result["discharge_kwh"][t], soc[t] - E_MIN_KWH, need)
            room = max(room, 0.0)
            replace = min(result["purchase_kwh"][t], ETA * room)
            extra_discharge = replace / ETA if ETA > 0 else 0.0
            extra_discharge = min(extra_discharge, room)
            if extra_discharge <= 0 or result["charge_kwh"][t] > 1e-9:
                continue
            result["discharge_kwh"][t] += extra_discharge
            result["purchase_kwh"][t] -= ETA * extra_discharge
            need -= extra_discharge
            soc[t + 1 :] -= extra_discharge

    result["soc_end_kwh"] = soc[1:]
