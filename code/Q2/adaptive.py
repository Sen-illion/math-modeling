"""Day-level adaptive conservative margin for the Q2 day-ahead LP.

Everything downstream of the margin is unchanged: XGB-Expanding load, 7-day same-slot
PV, day-ahead LP on the biased forecast, locked purchase, greedy causal playback.
The only new decision is which rung of the quantile ladder to use on day D, and it is
taken at 00:00 from state that depends on days < D only.

    q_L(D) = snap_to_ladder( clip( q_min + k * z(D), min(ladder), max(ladder) ) )

z(D) is the causal empirical percentile of a state feature among its own past values,
so it is scale free and needs no separate normalisation constants.

State features, all causal:
  em7         previous 7 executed days' realised emergency energy (policy's own history)
  resid_vol7  previous 7 days' std of the net-load point-forecast residual
  peak_ratio  today's forecast net-load peak over the 5000 kW interface cap
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from config import (
    ADAPTIVE_WARMUP_DAYS,
    DT_HOURS,
    E0_FEB1_KWH,
    OFFICIAL_START,
    P_MAX_KWH,
    Q_LADDER,
    Q_WARMUP,
)
from model_lp import solve_day_lp, validate_plan
from simulate import simulate_day, validate_actual

FEATURE_NAMES = ("em7", "resid_vol7", "peak_ratio")
MIN_HISTORY = 21


def snap_to_ladder(q: float, ladder=Q_LADDER) -> float:
    rungs = np.asarray(ladder, dtype=float)
    lo, hi = float(rungs.min()), float(rungs.max())
    q = float(min(hi, max(lo, q)))
    return float(rungs[int(np.argmin(np.abs(rungs - q)))])


def causal_percentile(history: list[float], value: float) -> float | None:
    """Share of past observations at or below `value`. None until history is usable."""
    if len(history) < MIN_HISTORY:
        return None
    past = np.asarray(history, dtype=float)
    return float(np.mean(past <= value))


def net_residual_std(residuals: list[dict]) -> float:
    """Std of the net-load point-forecast residual over the kept days, in kW."""
    if not residuals:
        return 0.0
    net = np.vstack([r["load"] - r["pv"] for r in residuals])
    return float(np.std(net))


def forecast_peak_ratio(load_kw: np.ndarray, pv_kw: np.ndarray) -> float:
    """Forecast net-load peak as a share of the interface cap, in energy terms."""
    net_kwh = (np.asarray(load_kw, dtype=float) - np.asarray(pv_kw, dtype=float)) * DT_HOURS
    return float(np.max(net_kwh) / P_MAX_KWH)


def run_adaptive(
    prices: pd.DataFrame,
    year: dict,
    banks: dict[float, list[dict]],
    residuals: dict[int, dict],
    end_date: str,
    feature: str = "em7",
    q_min: float = 0.8,
    k: float = 0.0,
    ladder=Q_LADDER,
    collect_traces: bool = False,
    label: str | None = None,
) -> dict:
    """Sequential causal replay. SOC always starts from 6000 kWh on OFFICIAL_START."""
    if feature not in FEATURE_NAMES:
        raise ValueError(f"unknown feature {feature}; expected one of {FEATURE_NAMES}")
    price = prices["price"].to_numpy(dtype=float)
    start = pd.Timestamp(OFFICIAL_START)
    stop = pd.Timestamp(end_date)
    rungs = sorted(float(q) for q in ladder)
    reference = banks[rungs[0]]

    soc = E0_FEB1_KWH
    feature_history: list[float] = []
    emergency_history: list[float] = []
    rows: list[dict] = []
    traces: list[dict] = []
    errors: list[str] = []
    load_err: list[np.ndarray] = []
    pv_err: list[np.ndarray] = []
    t0 = time.perf_counter()

    for pred in reference:
        stamp = pd.Timestamp(pred["date"])
        if stamp < start or stamp > stop:
            continue
        day = int(pred["day"])
        load_act = year["load_kwh"][day]
        pv_act = year["pv_kwh"][day]

        if feature == "em7":
            usable = len(emergency_history) >= ADAPTIVE_WARMUP_DAYS
            value = float(np.sum(emergency_history[-ADAPTIVE_WARMUP_DAYS:])) if usable else float("nan")
        elif feature == "resid_vol7":
            past = [residuals[d] for d in range(max(0, day - ADAPTIVE_WARMUP_DAYS), day) if d in residuals]
            usable = len(past) >= ADAPTIVE_WARMUP_DAYS
            value = net_residual_std(past) if usable else float("nan")
        else:
            usable = True
            value = forecast_peak_ratio(pred["load_kw"], pred["pv_kw"])

        z = causal_percentile(feature_history, value) if usable else None
        if z is None:
            q_load = float(Q_WARMUP)
            source = "warmup"
        else:
            q_load = snap_to_ladder(q_min + k * z, ladder)
            source = "rule"
        if usable:
            feature_history.append(value)

        plan_pred = banks[q_load][day]
        plan = solve_day_lp(
            price,
            plan_pred["load_kw"] * DT_HOURS,
            plan_pred["pv_kw"] * DT_HOURS,
            soc,
        )
        errors.extend(
            validate_plan(plan, price, plan_pred["load_kw"] * DT_HOURS, plan_pred["pv_kw"] * DT_HOURS)
        )
        actual = simulate_day(price, load_act, pv_act, plan["purchase_kwh"], soc)
        errors.extend(validate_actual(actual, plan["purchase_kwh"], load_act, pv_act))

        load_err.append(np.asarray(plan_pred["load_kw"], dtype=float) - year["load_kw"][day])
        pv_err.append(np.asarray(plan_pred["pv_kw"], dtype=float) - year["pv_kw"][day])
        emergency_kwh = float(actual["emergency_kwh"].sum())
        rows.append(
            {
                "date": pred["date"],
                "q_load": q_load,
                "q_pv": round(1.0 - q_load, 6),
                "q_source": source,
                "feature": feature,
                "feature_value": value if usable else np.nan,
                "feature_percentile": z if z is not None else np.nan,
                "soc0_actual": float(soc),
                "soc24_actual": float(actual["soc24_kwh"]),
                "soc24_plan": float(plan["soc_end_kwh"][-1]),
                "soc_min_actual": float(np.min(actual["soc_end_kwh"])),
                "soc_max_actual": float(np.max(actual["soc_end_kwh"])),
                "purchase_kwh": float(plan["purchase_kwh"].sum()),
                "plan_cost": float(actual["plan_cost"]),
                "emergency_kwh": emergency_kwh,
                "emergency_cost": float(actual["emergency_cost"]),
                "total_cost": float(actual["plan_cost"] + actual["emergency_cost"]),
                "curtail_kwh": float(actual["curtail_kwh"].sum()),
                "emergency_slots": int(actual["n_emergency_slots"]),
                "n_simultaneous_plan": int(plan["n_simultaneous"]),
                "max_unserved_kwh": float(actual["max_unserved_kwh"]),
                "n_negative_predictions": int(
                    np.sum(np.asarray(plan_pred["load_kw"]) < 0) + np.sum(np.asarray(plan_pred["pv_kw"]) < 0)
                ),
            }
        )
        if collect_traces:
            traces.append(
                {
                    "date": pred["date"],
                    "purchase_kwh": np.asarray(plan["purchase_kwh"], dtype=float).copy(),
                    "charge_kwh": np.asarray(actual["charge_kwh"], dtype=float).copy(),
                    "discharge_kwh": np.asarray(actual["discharge_kwh"], dtype=float).copy(),
                    "emergency_kwh": np.asarray(actual["emergency_kwh"], dtype=float).copy(),
                    "soc0_kwh": float(soc),
                    "soc24_kwh": float(actual["soc24_kwh"]),
                    "plan_cost": float(actual["plan_cost"]),
                    "emergency_cost": float(actual["emergency_cost"]),
                }
            )
        emergency_history.append(emergency_kwh)
        soc = float(actual["soc24_kwh"])

    daily = pd.DataFrame(rows)
    return {
        "daily": daily,
        "traces": traces,
        "errors": errors,
        "load_err_kw": np.vstack(load_err) if load_err else np.zeros((0, 1)),
        "pv_err_kw": np.vstack(pv_err) if pv_err else np.zeros((0, 1)),
        "rule": {
            "label": label or f"{feature}_qmin{q_min}_k{k}",
            "feature": feature,
            "q_min": float(q_min),
            "k": float(k),
            "ladder": [float(q) for q in rungs],
            "q_warmup": float(Q_WARMUP),
            "warmup_days": int(ADAPTIVE_WARMUP_DAYS),
            "min_history": int(MIN_HISTORY),
        },
        "elapsed_s": time.perf_counter() - t0,
    }


def run_fixed(
    prices: pd.DataFrame,
    year: dict,
    banks: dict[float, list[dict]],
    q_load: float,
    end_date: str,
    collect_traces: bool = False,
) -> dict:
    """Fixed-quantile baseline through the same code path (k = 0, no warmup switch)."""
    price = prices["price"].to_numpy(dtype=float)
    start = pd.Timestamp(OFFICIAL_START)
    stop = pd.Timestamp(end_date)
    soc = E0_FEB1_KWH
    rows: list[dict] = []
    traces: list[dict] = []
    errors: list[str] = []
    t0 = time.perf_counter()
    for pred in banks[float(q_load)]:
        stamp = pd.Timestamp(pred["date"])
        if stamp < start or stamp > stop:
            continue
        day = int(pred["day"])
        load_act = year["load_kwh"][day]
        pv_act = year["pv_kwh"][day]
        plan = solve_day_lp(price, pred["load_kw"] * DT_HOURS, pred["pv_kw"] * DT_HOURS, soc)
        actual = simulate_day(price, load_act, pv_act, plan["purchase_kwh"], soc)
        errors.extend(validate_actual(actual, plan["purchase_kwh"], load_act, pv_act))
        rows.append(
            {
                "date": pred["date"],
                "q_load": float(q_load),
                "q_pv": round(1.0 - float(q_load), 6),
                "q_source": "fixed",
                "soc0_actual": float(soc),
                "soc24_actual": float(actual["soc24_kwh"]),
                "purchase_kwh": float(plan["purchase_kwh"].sum()),
                "plan_cost": float(actual["plan_cost"]),
                "emergency_kwh": float(actual["emergency_kwh"].sum()),
                "emergency_cost": float(actual["emergency_cost"]),
                "total_cost": float(actual["plan_cost"] + actual["emergency_cost"]),
                "curtail_kwh": float(actual["curtail_kwh"].sum()),
                "n_simultaneous_plan": int(plan["n_simultaneous"]),
                "max_unserved_kwh": float(actual["max_unserved_kwh"]),
            }
        )
        if collect_traces:
            traces.append(
                {
                    "date": pred["date"],
                    "purchase_kwh": np.asarray(plan["purchase_kwh"], dtype=float).copy(),
                    "charge_kwh": np.asarray(actual["charge_kwh"], dtype=float).copy(),
                    "discharge_kwh": np.asarray(actual["discharge_kwh"], dtype=float).copy(),
                    "emergency_kwh": np.asarray(actual["emergency_kwh"], dtype=float).copy(),
                    "soc0_kwh": float(soc),
                    "soc24_kwh": float(actual["soc24_kwh"]),
                    "plan_cost": float(actual["plan_cost"]),
                    "emergency_cost": float(actual["emergency_cost"]),
                }
            )
        soc = float(actual["soc24_kwh"])
    return {
        "daily": pd.DataFrame(rows),
        "traces": traces,
        "errors": errors,
        "rule": {"label": f"fixed_q{int(round(float(q_load) * 100))}", "feature": None, "q_min": float(q_load), "k": 0.0},
        "elapsed_s": time.perf_counter() - t0,
    }


def window_metrics(daily: pd.DataFrame, start: str, end: str, name: str) -> dict:
    stamps = pd.to_datetime(daily["date"])
    mask = (stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))
    win = daily[mask]
    out = {
        "window": name,
        "start": start,
        "end": end,
        "n_days": int(len(win)),
        "total_cost": float(win["total_cost"].sum()),
        "plan_cost": float(win["plan_cost"].sum()),
        "emergency_cost": float(win["emergency_cost"].sum()),
        "emergency_kwh": float(win["emergency_kwh"].sum()),
        "emergency_days": int((win["emergency_kwh"] > 1e-6).sum()),
        "purchase_kwh": float(win["purchase_kwh"].sum()),
        "curtail_kwh": float(win["curtail_kwh"].sum()),
        "mean_soc24_actual": float(win["soc24_actual"].mean()) if len(win) else None,
        "max_unserved_kwh": float(win["max_unserved_kwh"].max()) if len(win) else 0.0,
        "n_simultaneous_plan": int(win["n_simultaneous_plan"].sum()),
    }
    if "q_load" in win.columns and len(win):
        out["mean_q_load"] = float(win["q_load"].mean())
        out["q_counts"] = {str(q): int(c) for q, c in win["q_load"].value_counts().sort_index().items()}
    return out


def summarise(result: dict, windows: list[tuple[str, str, str]]) -> dict:
    daily = result["daily"]
    summary = {
        "label": result["rule"]["label"],
        "rule": result["rule"],
        "n_validation_errors": len(result["errors"]),
        "validation_errors_head": result["errors"][:20],
        "elapsed_s": result["elapsed_s"],
        "windows": {name: window_metrics(daily, start, end, name) for name, start, end in windows},
    }
    return summary


def official_summary(result: dict, policy: str, pv_source: str) -> dict:
    """Same key shape as run_model's summary so the frozen-number guard can compare them."""
    daily = result["daily"]
    load_err = result["load_err_kw"]
    pv_err = result["pv_err_kw"]
    rule = result["rule"]
    return {
        "model": "xgb_expanding",
        "policy": policy,
        "terminal_mode": "none",
        "soc_mu": 0.0,
        "dispatch": "greedy",
        "mpc_stride": None,
        "n_days_run": int(len(daily)),
        "n_official_days": int(len(daily)),
        "elapsed_s": result["elapsed_s"],
        "load_mae_kw": float(np.mean(np.abs(load_err))),
        "load_rmse_kw": float(np.sqrt(np.mean(load_err**2))),
        "pv_mae_kw": float(np.mean(np.abs(pv_err))),
        "pv_rmse_kw": float(np.sqrt(np.mean(pv_err**2))),
        "purchase_kwh": float(daily["purchase_kwh"].sum()),
        "plan_cost": float(daily["plan_cost"].sum()),
        "emergency_kwh": float(daily["emergency_kwh"].sum()),
        "emergency_slots": int(daily["emergency_slots"].sum()),
        "emergency_days": int((daily["emergency_kwh"] > 1e-6).sum()),
        "emergency_cost": float(daily["emergency_cost"].sum()),
        "total_cost": float(daily["total_cost"].sum()),
        "curtail_kwh": float(daily["curtail_kwh"].sum()),
        "n_simultaneous_plan": int(daily["n_simultaneous_plan"].sum()),
        "n_negative_predictions": int(daily["n_negative_predictions"].sum()),
        "n_validation_errors": len(result["errors"]),
        "validation_errors_head": result["errors"][:20],
        "feb1_soc0": float(daily["soc0_actual"].iloc[0]),
        "last_soc24_actual": float(daily["soc24_actual"].iloc[-1]),
        "last_soc24_plan": float(daily["soc24_plan"].iloc[-1]),
        "mean_soc24_actual": float(daily["soc24_actual"].mean()),
        "mean_soc24_plan": float(daily["soc24_plan"].mean()),
        "soc_min_actual": float(daily["soc_min_actual"].min()),
        "soc_max_actual": float(daily["soc_max_actual"].max()),
        "soc_init": "feb1_6000",
        "terminal_lambda": 0.0,
        "terminal_target_kwh": None,
        "margin_mode": "adaptive",
        "margin_rule": rule,
        "q_load_mean": float(daily["q_load"].mean()),
        "q_load_counts": {str(q): int(c) for q, c in daily["q_load"].value_counts().sort_index().items()},
        "q_warmup_days": int((daily["q_source"] == "warmup").sum()),
        "pv_source": pv_source,
    }


def flatten_for_table(summary: dict) -> dict:
    row = {"label": summary["label"], "feature": summary["rule"].get("feature"), "q_min": summary["rule"].get("q_min"), "k": summary["rule"].get("k")}
    for name, win in summary["windows"].items():
        row[f"{name}_total_cost"] = win["total_cost"]
        row[f"{name}_plan_cost"] = win["plan_cost"]
        row[f"{name}_emergency_cost"] = win["emergency_cost"]
        row[f"{name}_emergency_kwh"] = win["emergency_kwh"]
        row[f"{name}_n_days"] = win["n_days"]
        if "mean_q_load" in win:
            row[f"{name}_mean_q"] = win["mean_q_load"]
    row["n_validation_errors"] = summary["n_validation_errors"]
    return row


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=float)
