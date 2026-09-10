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
    OFFICIAL_END,
    OFFICIAL_START,
    REL_TOL,
)
from load_forecast import forecast_day_kw
from simulate import validate_actual


def _tol(scale: float) -> float:
    return max(ABS_TOL_KWH, REL_TOL * abs(scale))


def leakage_errors(records: list[dict], load_kwh: np.ndarray, dates, typical_kw: np.ndarray) -> list[str]:
    errors = []
    load_kw = load_kwh / DT_HOURS
    n_days = load_kw.shape[0]
    for rec in records:
        day = rec["day"]
        if rec.get("load_fc0_kw") is None:
            continue
        expected = forecast_day_kw(load_kw, day, dates, typical_kw)
        if not np.allclose(rec["load_fc0_kw"], expected, atol=1e-8, rtol=0):
            errors.append(f"day {day} load forecast mismatch vs causal week-lag")
        if day >= 7:
            src = load_kw[day - 7]
            if not np.allclose(rec["load_fc0_kw"], src, atol=1e-8, rtol=0):
                errors.append(f"day {day} 0:00 load forecast is not last-week same weekday")
            if np.allclose(rec["load_fc0_kw"], load_kw[day], atol=1e-12, rtol=0) and not np.allclose(
                src, load_kw[day], atol=1e-12, rtol=0
            ):
                errors.append(f"day {day} 0:00 load forecast leaked today's actual")
        if rec["updates"][0]["hour"] != 0:
            errors.append(f"day {day} missing 0:00 update")
        if not np.allclose(rec["g_plan_kwh"][:36], rec["g_adj_kwh"][:36], atol=ABS_TOL_KWH, rtol=0):
            errors.append(f"day {day} 0:00-6:00 contract changed after 0:00")
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
        bill = rec["bill"]
        recon = rec["bill"]["grid_cost"] + rec["bill"]["emergency_cost"]
        if abs(recon - bill["total_cost"]) > _tol(max(bill["total_cost"], 1.0)):
            errors.append(f"{rec['date']} cost mismatch")
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
    return {
        "n_days": len(official),
        "total_cost": total,
        "plan_only_cost": plan,
        "emergency_cost": emergency,
        "emergency_kwh": em_kwh,
        "adj_purchase_kwh": adj_kwh,
        "n_days_with_intraday_adjust": n_adjust_days,
    }
