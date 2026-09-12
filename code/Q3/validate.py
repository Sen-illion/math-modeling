"""Gate checks for Q3 official results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ABS_TOL_KWH,
    DT_HOURS,
    E0_JAN1_KWH,
    E_MAX_KWH,
    E_MIN_KWH,
    ISSUE_HOURS,
    LOCK_SLOTS,
    N_INTERVALS,
    OFFICIAL_END,
    OFFICIAL_START,
    REL_TOL,
)
from load_forecast import forecast_day_kw, tomorrow_forecast_kw
from pv_forecast import tomorrow_pv_kw
from quantile import q_lock_at, q_vector
from simulate import validate_actual


def _tol(scale: float) -> float:
    return max(ABS_TOL_KWH, REL_TOL * abs(scale))


def recompute_day_cost(price: np.ndarray, g_plan: np.ndarray, g_adj: np.ndarray, emergency: np.ndarray) -> dict:
    """Rebuild settlement from kWh arrays and price. Do not use stored bill fields."""
    price = np.asarray(price, dtype=float)
    g_plan = np.asarray(g_plan, dtype=float)
    g_adj = np.asarray(g_adj, dtype=float)
    emergency = np.asarray(emergency, dtype=float)
    dp = np.maximum(g_adj - g_plan, 0.0)
    dm = np.maximum(g_plan - g_adj, 0.0)
    grid = float(np.dot(price, g_plan) - 0.5 * np.dot(price, dm) + 1.5 * np.dot(price, dp))
    em_cost = float(np.dot(5.0 * price, emergency))
    return {
        "plan_only_cost": float(np.dot(price, g_plan)),
        "grid_cost": grid,
        "emergency_cost": em_cost,
        "total_cost": grid + em_cost,
        "delta_plus_kwh": float(dp.sum()),
        "delta_minus_kwh": float(dm.sum()),
    }


def leakage_errors(
    records: list[dict],
    load_kwh: np.ndarray,
    dates,
    typical_kw: np.ndarray,
    quantile_bank=None,
    pv_kwh: np.ndarray | None = None,
) -> list[str]:
    errors = []
    load_kw = load_kwh / DT_HOURS
    n_days = load_kw.shape[0]
    for rec in records:
        day = rec["day"]
        if rec.get("load_fc0_kw") is None:
            continue
        expected = forecast_day_kw(load_kw, day, dates, typical_kw)
        if not np.allclose(rec["load_fc0_kw"], expected, atol=1e-8, rtol=0):
            errors.append(f"day {day} load forecast mismatch vs causal 0:00 model")
        if day >= 7:
            if np.allclose(rec["load_fc0_kw"], load_kw[day], atol=1e-12, rtol=0) and not np.allclose(
                load_kw[day - 7], load_kw[day], atol=1e-12, rtol=0
            ):
                errors.append(f"day {day} 0:00 load forecast leaked today's actual")
        fc_tomorrow = rec.get("load_fc_tomorrow_kw")
        if fc_tomorrow is not None:
            expected_next = tomorrow_forecast_kw(load_kw, dates, typical_kw, day)
            if not np.allclose(fc_tomorrow, expected_next, atol=1e-8, rtol=0):
                errors.append(f"day {day} look-ahead load forecast is not the causal next-day model")
            if day + 1 < n_days and np.allclose(fc_tomorrow, load_kw[day + 1], atol=1e-12, rtol=0):
                src = day - 6
                coincidence = src >= 0 and np.allclose(load_kw[src], load_kw[day + 1], atol=1e-12, rtol=0)
                if not coincidence:
                    errors.append(f"day {day} look-ahead forecast leaked tomorrow's actual")
        pv_tomorrow = rec.get("pv_fc_tomorrow_kw")
        if pv_tomorrow is not None and pv_kwh is not None:
            pv_kw = pv_kwh / DT_HOURS
            expected_pv = tomorrow_pv_kw(pv_kw, day)
            if not np.allclose(pv_tomorrow, expected_pv, atol=1e-8, rtol=0):
                errors.append(f"day {day} look-ahead PV is not the causal week-similar series")
            if day + 1 < n_days and np.allclose(pv_tomorrow, pv_kw[day + 1], atol=1e-12, rtol=0):
                src = day - 6
                coincidence = src >= 0 and np.allclose(pv_kw[src], pv_kw[day + 1], atol=1e-12, rtol=0)
                if not coincidence:
                    errors.append(f"day {day} look-ahead PV leaked tomorrow's actual")
        if rec["updates"][0]["hour"] != 0:
            errors.append(f"day {day} missing 0:00 update")
        if not np.allclose(rec["g_plan_kwh"][:36], rec["g_adj_kwh"][:36], atol=ABS_TOL_KWH, rtol=0):
            errors.append(f"day {day} 0:00-6:00 contract changed after 0:00")
        if len(rec["updates"]) == 1 and not np.allclose(
            rec["g_plan_kwh"], rec["g_adj_kwh"], atol=ABS_TOL_KWH, rtol=0
        ):
            errors.append(f"day {day} no-adjust run changed remaining contract")
        if rec.get("q_lock") is not None:
            if quantile_bank is None:
                errors.append(f"day {day} used quantiles but no residual bank was provided")
            elif day >= 7:
                for upd in rec["updates"]:
                    stored = upd.get("load_q_offset_kw")
                    if stored is None:
                        continue
                    hour = int(upd["hour"])
                    iss = ISSUE_HOURS.index(hour)
                    n_horizon = int(upd["n_horizon"])
                    n_today = N_INTERVALS - hour * 6
                    q_vec = q_vector(
                        n_horizon,
                        n_today,
                        q_lock_at(rec["q_lock"], hour),
                        float(rec.get("q_open") if rec.get("q_open") is not None else 0.5),
                        float(rec.get("q_evening") if rec.get("q_evening") is not None else 0.5),
                        LOCK_SLOTS,
                    )
                    expected, _pv = quantile_bank.offsets(
                        day,
                        iss,
                        n_horizon,
                        q_vec,
                        window=rec.get("resid_window"),
                    )
                    if not np.allclose(stored, expected, atol=1e-8, rtol=0):
                        errors.append(f"day {day} h={hour} quantile offset is not the causal bank")
    if n_days < 0:
        errors.append("empty year")
    return errors


def official_errors(official: list[dict], dates, soc_track: dict, load_kwh, pv_kwh, price144, expected_n: int | None = 334) -> list[str]:
    errors = []
    start = pd.Timestamp(OFFICIAL_START)
    end = pd.Timestamp(OFFICIAL_END)
    if expected_n is not None and len(official) != expected_n:
        errors.append(f"official days {len(official)} != {expected_n}")
    for rec in official:
        stamp = pd.Timestamp(rec["date"])
        if stamp < start or stamp > end:
            errors.append(f"{rec['date']} outside official window")
        if (rec["g_plan_kwh"] < -ABS_TOL_KWH).any() or (rec["g_adj_kwh"] < -ABS_TOL_KWH).any():
            errors.append(f"{rec['date']} negative purchase")
        day = rec["day"]
        errors.extend(
            [
                f"{rec['date']} {e}"
                for e in validate_actual(rec["actual"], rec["g_adj_kwh"], load_kwh[day], pv_kwh[day])
            ]
        )
        rebuilt = recompute_day_cost(price144, rec["g_plan_kwh"], rec["g_adj_kwh"], rec["actual"]["emergency_kwh"])
        bill = rec["bill"]
        if abs(rebuilt["total_cost"] - bill["total_cost"]) > _tol(max(bill["total_cost"], 1.0)):
            errors.append(f"{rec['date']} independent cost mismatch")
        if rec["day"] == 0:
            if abs(rec["actual"]["soc0_kwh"] - E0_JAN1_KWH) > ABS_TOL_KWH:
                errors.append("Jan 1 SOC0 != 6000")
        if rec["actual"]["soc0_kwh"] < E_MIN_KWH - ABS_TOL_KWH or rec["actual"]["soc0_kwh"] > E_MAX_KWH + ABS_TOL_KWH:
            errors.append(f"{rec['date']} SOC0 out of bounds")
    for prev, cur in zip(official, official[1:]):
        if abs(prev["soc24_kwh"] - cur["actual"]["soc0_kwh"]) > ABS_TOL_KWH:
            errors.append(f"{cur['date']} SOC0 != previous 24:00")
    return errors


def summarize(official: list[dict]) -> dict:
    total = sum(r["bill"]["total_cost"] for r in official)
    emergency = sum(r["bill"]["emergency_cost"] for r in official)
    plan = sum(r["bill"]["plan_only_cost"] for r in official)
    em_kwh = float(sum(r["actual"]["emergency_kwh"].sum() for r in official))
    adj_kwh = float(sum(r["g_adj_kwh"].sum() for r in official))
    n_adjust_days = int(
        sum(any(u["hour"] > 0 and u["adjusted"] for u in r["updates"]) for r in official)
    )
    dp = float(sum(r["bill"]["delta_plus_kwh"] for r in official))
    dm = float(sum(r["bill"]["delta_minus_kwh"] for r in official))
    soc24 = np.array([r["actual"]["soc24_kwh"] for r in official], dtype=float)
    em = np.vstack([r["actual"]["emergency_kwh"] for r in official]) if official else np.zeros((0, 144))
    em_20 = float(em[:, 120:126].sum()) if official else 0.0
    return {
        "n_days": len(official),
        "total_cost": total,
        "plan_only_cost": plan,
        "emergency_cost": emergency,
        "emergency_kwh": em_kwh,
        "adj_purchase_kwh": adj_kwh,
        "n_days_with_intraday_adjust": n_adjust_days,
        "delta_plus_kwh": dp,
        "delta_minus_kwh": dm,
        "soc24_median": float(np.median(soc24)) if len(soc24) else None,
        "soc24_at_min_days": int(np.sum(soc24 <= 1250.0)) if len(soc24) else 0,
        "emergency_share_20h": (em_20 / em_kwh) if em_kwh > 0 else 0.0,
    }
