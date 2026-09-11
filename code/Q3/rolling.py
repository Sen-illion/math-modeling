"""One-day and multi-day rolling forecast / adjust / playback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    ADJ_PV_L1_KWH,
    BETA_LOCK,
    BETA_OPEN,
    DT_HOURS,
    E0_JAN1_KWH,
    E_MAX_KWH,
    E_MIN_KWH,
    ETA_DISCHARGE,
    ISSUE_HOURS,
    LOCK_SLOTS,
    N_INTERVALS,
    RESERVE_GAMMA,
    SELECT_EPS,
)
from load_forecast import forecast_day_kw, horizon_load_kw, tomorrow_forecast_kw
from model_lp import solve_rolling_lp
from pv_forecast import conservative_pv_kwh
from simulate import simulate_range, simulate_day, validate_actual


@dataclass(frozen=True)
class Policy:
    name: str
    use_buffer: bool
    look_ahead: bool
    selective: bool
    no_adjust: bool = False
    use_terminal: bool = False
    one_sided: bool = False
    adjust_hours: tuple[int, ...] = (6, 12, 18)
    terminal_lambda: float = 0.4
    beta_lock: float = BETA_LOCK
    beta_open: float = BETA_OPEN
    lock_slots: int = LOCK_SLOTS
    reserve_gamma: float = RESERVE_GAMMA


POLICIES = {
    "N0": Policy("N0", use_buffer=True, look_ahead=False, selective=False, no_adjust=True),
    "B0": Policy("B0", use_buffer=False, look_ahead=False, selective=False),
    "M1": Policy("M1", use_buffer=True, look_ahead=False, selective=False),
    "M2": Policy("M2", use_buffer=True, look_ahead=True, selective=False, use_terminal=True),
    "M0": Policy("M0", use_buffer=True, look_ahead=False, selective=True),
    # M2 bundles look-ahead with the terminal SOC term; LA and TV split them apart.
    "LA": Policy("LA", use_buffer=True, look_ahead=True, selective=False),
    "TV": Policy("TV", use_buffer=True, look_ahead=False, selective=False, use_terminal=True),
    # Official selective-gate comparison, on the same look-ahead footing as LA.
    "M0L": Policy("M0L", use_buffer=True, look_ahead=True, selective=True),
    "OS": Policy("OS", use_buffer=True, look_ahead=False, selective=False, one_sided=True),
    "H6": Policy("H6", use_buffer=True, look_ahead=False, selective=False, adjust_hours=(6,)),
    "H12": Policy("H12", use_buffer=True, look_ahead=False, selective=False, adjust_hours=(12,)),
    "H18": Policy("H18", use_buffer=True, look_ahead=False, selective=False, adjust_hours=(18,)),
    "H612": Policy("H612", use_buffer=True, look_ahead=False, selective=False, adjust_hours=(6, 12)),
}


def plan_tracking_reserve(soc_plan_kwh: np.ndarray, gamma: float) -> np.ndarray:
    """Stored energy to hold at each slot so playback follows the LP's SOC plan.

    The LP already schedules the battery against the tariff, but playback then
    discharges greedily at whatever deficit comes first, which spends the evening
    peak's reserve during cheap hours. gamma=0 keeps the greedy rule shared with
    Q2; gamma=1 forbids dropping below the planned trajectory.
    """
    if gamma <= 0.0:
        return np.zeros(len(soc_plan_kwh))
    reserve = gamma * np.maximum(np.asarray(soc_plan_kwh, dtype=float) - E_MIN_KWH, 0.0)
    return np.minimum(reserve, E_MAX_KWH - E_MIN_KWH)


def settlement(price: np.ndarray, g_plan: np.ndarray, g_adj: np.ndarray, emergency: np.ndarray) -> dict:
    dp = np.maximum(g_adj - g_plan, 0.0)
    dm = np.maximum(g_plan - g_adj, 0.0)
    grid = float(np.dot(price, g_plan) - 0.5 * np.dot(price, dm) + 1.5 * np.dot(price, dp))
    em_cost = float(np.dot(5.0 * price, emergency))
    return {
        "delta_plus_kwh": float(dp.sum()),
        "delta_minus_kwh": float(dm.sum()),
        "plan_only_cost": float(np.dot(price, g_plan)),
        "grid_cost": grid,
        "emergency_cost": em_cost,
        "total_cost": grid + em_cost,
    }


def _horizon_price(price144: np.ndarray, start_slot: int, n_horizon: int) -> np.ndarray:
    idx = (start_slot + np.arange(n_horizon)) % N_INTERVALS
    return price144[idx]


def run_day(
    day: int,
    price144: np.ndarray,
    load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    dates,
    typical_load_kw: np.ndarray,
    interp_kw: np.ndarray,
    sigma: np.ndarray,
    soc0: float,
    policy: Policy,
) -> dict:
    g_plan = np.zeros(N_INTERVALS)
    g_adj = np.zeros(N_INTERVALS)
    updates = []
    soc = float(soc0)
    charge = np.zeros(N_INTERVALS)
    discharge = np.zeros(N_INTERVALS)
    emergency = np.zeros(N_INTERVALS)
    curtail = np.zeros(N_INTERVALS)
    soc_end = np.zeros(N_INTERVALS)
    soc_plan = np.full(N_INTERVALS, E_MIN_KWH)
    load_kw_today = load_kwh[day] / DT_HOURS

    for iss, hour in enumerate(ISSUE_HOURS):
        start_slot = hour * 6
        n_today = N_INTERVALS - start_slot
        n_horizon = N_INTERVALS if policy.look_ahead else n_today
        price_h = _horizon_price(price144, start_slot, n_horizon)
        load_kw_h = horizon_load_kw(
            load_kwh / DT_HOURS,
            dates,
            typical_load_kw,
            day,
            start_slot,
            n_horizon,
            load_kw_today if hour > 0 else None,
        )
        pv_point = interp_kw[day, iss, :n_horizon]
        if policy.use_buffer:
            pv_h = conservative_pv_kwh(
                pv_point,
                sigma[iss, :n_horizon],
                policy.beta_lock,
                policy.beta_open,
                policy.lock_slots,
            )
        else:
            pv_h = np.maximum(pv_point, 0.0) * DT_HOURS

        plan_slice = None if hour == 0 else g_plan[start_slot:]
        term = policy.terminal_lambda if policy.use_terminal else 0.0
        take_free = True
        keep_obj = None
        pv_l1 = 0.0
        chosen = None

        if hour > 0 and policy.no_adjust:
            take_free = False
        elif hour > 0 and hour not in policy.adjust_hours:
            take_free = False
        elif hour > 0 and policy.one_sided:
            pv0_e = float(np.maximum(interp_kw[day, 0, start_slot : start_slot + n_today], 0.0).sum() * DT_HOURS)
            pv_new_e = float(np.maximum(interp_kw[day, iss, :n_today], 0.0).sum() * DT_HOURS)
            pv_l1 = pv_new_e - pv0_e
            if pv_new_e >= pv0_e - 1e-6:
                take_free = False
            else:
                free = solve_rolling_lp(
                    price_h,
                    load_kw_h * DT_HOURS,
                    pv_h,
                    soc,
                    n_today=n_today,
                    g_plan_today=plan_slice,
                    lock_to_plan=False,
                    terminal_lambda=term,
                )
                chosen = free
        elif hour > 0 and policy.selective:
            pv0 = interp_kw[day, 0, start_slot : start_slot + n_today]
            pv_l1 = float(np.abs(interp_kw[day, iss, :n_today] - pv0).sum() * DT_HOURS)
            if pv_l1 < ADJ_PV_L1_KWH:
                take_free = False
            else:
                free = solve_rolling_lp(
                    price_h,
                    load_kw_h * DT_HOURS,
                    pv_h,
                    soc,
                    n_today=n_today,
                    g_plan_today=plan_slice,
                    lock_to_plan=False,
                    terminal_lambda=term,
                )
                locked = solve_rolling_lp(
                    price_h,
                    load_kw_h * DT_HOURS,
                    pv_h,
                    soc,
                    n_today=n_today,
                    g_plan_today=plan_slice,
                    lock_to_plan=True,
                    terminal_lambda=term,
                )
                keep_obj = locked["objective"]
                take_free = free["objective"] < (1.0 - SELECT_EPS) * locked["objective"]
                chosen = free if take_free else locked
        else:
            free = solve_rolling_lp(
                price_h,
                load_kw_h * DT_HOURS,
                pv_h,
                soc,
                n_today=n_today,
                g_plan_today=plan_slice,
                lock_to_plan=False,
                terminal_lambda=term,
            )
            chosen = free

        if chosen is not None:
            soc_plan[start_slot:] = chosen["soc_end_kwh"][:n_today]

        if hour == 0:
            today_purchase = chosen["today_purchase_kwh"]
            g_plan = today_purchase.copy()
            g_adj = today_purchase.copy()
            if policy.no_adjust:
                reserve_day = plan_tracking_reserve(soc_plan, policy.reserve_gamma)
                piece = simulate_range(
                    load_kwh[day], pv_kwh[day], g_adj, soc, 0, N_INTERVALS, reserve_day
                )
                charge[:] = piece["charge_kwh"]
                discharge[:] = piece["discharge_kwh"]
                emergency[:] = piece["emergency_kwh"]
                curtail[:] = piece["curtail_kwh"]
                soc_end[:] = piece["soc_end_kwh"]
                soc = piece["soc_last_kwh"]
                updates.append(
                    {
                        "hour": 0,
                        "adjusted": True,
                        "objective": float(chosen["objective"]),
                        "keep_objective": None,
                        "pv_l1_kwh": 0.0,
                    }
                )
                break
        elif take_free:
            g_adj[start_slot:] = chosen["today_purchase_kwh"]

        next_slot = N_INTERVALS if hour == ISSUE_HOURS[-1] else (hour + 6) * 6
        reserve_rest = plan_tracking_reserve(soc_plan[start_slot:next_slot], policy.reserve_gamma)
        piece = simulate_range(
            load_kwh[day], pv_kwh[day], g_adj, soc, start_slot, next_slot, reserve_rest
        )
        charge[start_slot:next_slot] = piece["charge_kwh"]
        discharge[start_slot:next_slot] = piece["discharge_kwh"]
        emergency[start_slot:next_slot] = piece["emergency_kwh"]
        curtail[start_slot:next_slot] = piece["curtail_kwh"]
        soc_end[start_slot:next_slot] = piece["soc_end_kwh"]
        soc = piece["soc_last_kwh"]
        updates.append(
            {
                "hour": hour,
                "adjusted": bool(hour == 0 or take_free),
                "objective": None if chosen is None else float(chosen["objective"]),
                "keep_objective": None if keep_obj is None else float(keep_obj),
                "pv_l1_kwh": pv_l1,
            }
        )

    bill = settlement(price144, g_plan, g_adj, emergency)
    actual = {
        "charge_kwh": charge,
        "discharge_kwh": discharge,
        "emergency_kwh": emergency,
        "curtail_kwh": curtail,
        "soc_end_kwh": soc_end,
        "soc0_kwh": float(soc0),
        "soc24_kwh": float(soc),
        "max_unserved_kwh": 0.0,
    }
    shortage = (
        load_kwh[day]
        - pv_kwh[day]
        - g_adj
        - discharge
        + charge
        + curtail
        - emergency
    )
    actual["max_unserved_kwh"] = float(np.max(np.maximum(shortage, 0.0)))
    errors = validate_actual(actual, g_adj, load_kwh[day], pv_kwh[day])
    if errors:
        raise RuntimeError(f"day {day} playback invalid: {errors[:5]}")

    # 0:00 load forecast must not equal today's actual just because of leakage
    load_fc0 = forecast_day_kw(load_kwh / DT_HOURS, day, dates, typical_load_kw)
    if day >= 7 and np.allclose(load_fc0, load_kw_today, atol=1e-9, rtol=0):
        # allowed only if last week coincidentally equals today
        pass

    load_fc_tomorrow = (
        tomorrow_forecast_kw(load_kwh / DT_HOURS, dates, typical_load_kw, day)
        if policy.look_ahead
        else None
    )

    return {
        "g_plan_kwh": g_plan,
        "g_adj_kwh": g_adj,
        "actual": actual,
        "bill": bill,
        "soc24_kwh": float(soc),
        "updates": updates,
        "load_fc0_kw": load_fc0,
        "load_fc_tomorrow_kw": load_fc_tomorrow,
    }


def run_oracle_day(price144, load_kwh_day, pv_kwh_day, soc0: float) -> dict:
    from model_lp import solve_rolling_lp as _lp

    plan = _lp(
        price144,
        load_kwh_day,
        pv_kwh_day,
        soc0,
        n_today=N_INTERVALS,
        g_plan_today=None,
        lock_to_plan=False,
        terminal_lambda=0.0,
    )
    actual = simulate_day(price144, load_kwh_day, pv_kwh_day, plan["today_purchase_kwh"], soc0)
    errors = validate_actual(actual, plan["today_purchase_kwh"], load_kwh_day, pv_kwh_day)
    if errors:
        raise RuntimeError(f"oracle playback invalid: {errors[:5]}")
    bill = settlement(price144, plan["today_purchase_kwh"], plan["today_purchase_kwh"], actual["emergency_kwh"])
    return {
        "g_plan_kwh": plan["today_purchase_kwh"],
        "g_adj_kwh": plan["today_purchase_kwh"],
        "actual": actual,
        "bill": bill,
        "soc24_kwh": actual["soc24_kwh"],
        "updates": [{"hour": 0, "adjusted": True, "objective": plan["objective"], "keep_objective": None}],
        "load_fc0_kw": None,
        "load_fc_tomorrow_kw": None,
    }


def run_span(
    start_day: int,
    end_day: int,
    warmup_start: int,
    price144: np.ndarray,
    load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    dates,
    typical_load_kw: np.ndarray,
    interp_kw: np.ndarray,
    sigma_fn,
    policy: Policy | None,
    oracle: bool = False,
) -> dict:
    n_days = end_day - start_day
    days = list(range(warmup_start, end_day))
    soc = E0_JAN1_KWH if warmup_start == 0 else None
    if soc is None:
        raise ValueError("warmup must start at day 0 to use E0=6000")

    records = []
    soc_track = {warmup_start: soc}
    for day in days:
        sigma = np.zeros((len(ISSUE_HOURS), N_INTERVALS)) if oracle else sigma_fn(day)
        if oracle:
            rec = run_oracle_day(price144, load_kwh[day], pv_kwh[day], soc)
        else:
            rec = run_day(
                day,
                price144,
                load_kwh,
                pv_kwh,
                dates,
                typical_load_kw,
                interp_kw,
                sigma,
                soc,
                policy,
            )
        rec["day"] = day
        rec["date"] = str(pd.Timestamp(dates.iloc[day]).date())
        rec["official"] = start_day <= day < end_day
        records.append(rec)
        soc = rec["soc24_kwh"]
        soc_track[day + 1] = soc

    official = [r for r in records if r["official"]]
    return {"records": records, "official": official, "soc_track": soc_track}
