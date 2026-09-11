"""Read-only Q2 diagnostics.

Part A  headroom and waste
  - perfect-information lower bound
  - slot-level waste in the frozen plan (bought-and-curtailed, emergency causes)
  - knobs that were never rechecked after the start-align switch
  - Feb-1 SOC from a January roll instead of a free 6000 reset

Part B  how much is realistically on the table
  - fixed quantile grid on one shared forecast bank
  - a clairvoyant daily quantile selector (SOC still rolls, so the trajectory is
    legal; only the choice is unfair). This bounds the fixed-quantile policy class.

Writes only to results/Q2/diagnostics/. Never touches results/Q2/opt/ or result2.xlsx.

    python code/Q2/diagnose_q2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    DT_HOURS,
    E0_FEB1_KWH,
    E_MAX_KWH,
    E_MIN_KWH,
    ETA_CHARGE,
    OFFICIAL_END,
    OFFICIAL_START,
    P_MAX_KWH,
    RESULT_DIR,
)
from forecast import apply_pv_night_zero, clip_nonneg, precompute_panel  # noqa: E402
from load_data import audit_data, load_prices, load_year_actuals  # noqa: E402
from model_lp import solve_day_lp  # noqa: E402
from run_q2 import collect_forecasts, copy_raw_inputs  # noqa: E402
from simulate import simulate_day, simulate_day_mpc, validate_actual  # noqa: E402

DIAG_DIR = RESULT_DIR / "diagnostics"
FROZEN_TOTAL = 13765167.58173598
Q_GRID = [0.6, 0.7, 0.8, 0.85, 0.9]


def _slot_quantile_vec(residuals: list[np.ndarray], q) -> np.ndarray:
    stacked = np.vstack(residuals)
    if np.isscalar(q):
        return np.quantile(stacked, float(q), axis=0)
    q = np.asarray(q, dtype=float)
    return np.array([np.quantile(stacked[:, s], q[s]) for s in range(stacked.shape[1])])


def bias_forecasts(forecasts: list[dict], year: dict, q_load, q_pv) -> list[dict]:
    """Causal residual-quantile bias. q may be a scalar or a 144-vector."""
    load_resid: list[np.ndarray] = []
    pv_resid: list[np.ndarray] = []
    out = []
    for pred in forecasts:
        day = int(pred["day"])
        load_plan = pred["load_kw"].copy()
        pv_plan = pred["pv_kw"].copy()
        if q_load is not None and load_resid:
            load_plan = load_plan + _slot_quantile_vec(load_resid, q_load)
        if q_pv is not None and pv_resid:
            pv_plan = pv_plan + _slot_quantile_vec(pv_resid, q_pv)
        load_resid.append(year["load_kw"][day] - pred["load_kw"])
        pv_resid.append(year["pv_kw"][day] - pred["pv_kw"])
        row = dict(pred)
        row["load_kw"] = clip_nonneg(load_plan)
        row["pv_kw"] = clip_nonneg(apply_pv_night_zero(pv_plan, year["pv_kw"][:day]))
        out.append(row)
    return out


def replay(
    name: str,
    prices: pd.DataFrame,
    year: dict,
    biased: list[dict],
    sim_start: pd.Timestamp,
    soc_mu: float = 0.0,
    dispatch: str = "greedy",
    stride: int = 1,
    oracle: bool = False,
    collect_traces: bool = False,
) -> dict:
    price = prices["price"].to_numpy(dtype=float)
    official_start = pd.Timestamp(OFFICIAL_START)
    soc = E0_FEB1_KWH
    rows: list[dict] = []
    traces: list[dict] = []
    errors: list[str] = []
    t0 = time.perf_counter()
    for pred in biased:
        stamp = pd.Timestamp(pred["date"])
        if stamp < sim_start:
            continue
        if stamp == sim_start:
            soc = E0_FEB1_KWH
        day = int(pred["day"])
        load_act = year["load_kwh"][day]
        pv_act = year["pv_kwh"][day]
        if oracle:
            load_hat, pv_hat = load_act, pv_act
        else:
            load_hat = pred["load_kw"] * DT_HOURS
            pv_hat = pred["pv_kw"] * DT_HOURS
        plan = solve_day_lp(price, load_hat, pv_hat, soc, soc_mu=soc_mu)
        if dispatch == "mpc":
            actual = simulate_day_mpc(
                price, load_act, pv_act, load_hat, pv_hat, plan["purchase_kwh"], soc, soc_mu=soc_mu, stride=stride
            )
        else:
            actual = simulate_day(price, load_act, pv_act, plan["purchase_kwh"], soc)
        errors.extend(validate_actual(actual, plan["purchase_kwh"], load_act, pv_act))
        rows.append(
            {
                "date": pred["date"],
                "official": bool(stamp >= official_start),
                "soc0_actual": float(soc),
                "soc24_actual": float(actual["soc24_kwh"]),
                "purchase_kwh": float(plan["purchase_kwh"].sum()),
                "plan_cost": float(actual["plan_cost"]),
                "emergency_kwh": float(actual["emergency_kwh"].sum()),
                "emergency_cost": float(actual["emergency_cost"]),
                "total_cost": float(actual["plan_cost"] + actual["emergency_cost"]),
                "curtail_kwh": float(actual["curtail_kwh"].sum()),
            }
        )
        if collect_traces and stamp >= official_start:
            traces.append(
                {
                    "date": pred["date"],
                    "purchase_kwh": np.asarray(plan["purchase_kwh"], dtype=float).copy(),
                    "curtail_kwh": np.asarray(actual["curtail_kwh"], dtype=float).copy(),
                    "emergency_kwh": np.asarray(actual["emergency_kwh"], dtype=float).copy(),
                    "soc_end_kwh": np.asarray(actual["soc_end_kwh"], dtype=float).copy(),
                    "soc0_kwh": float(soc),
                }
            )
        soc = float(actual["soc24_kwh"])
    daily = pd.DataFrame(rows)
    off = daily[daily["official"]]
    summary = {
        "name": name,
        "sim_start": str(sim_start.date()),
        "dispatch": dispatch,
        "mpc_stride": stride if dispatch == "mpc" else None,
        "soc_mu": soc_mu,
        "oracle": oracle,
        "n_official_days": int(len(off)),
        "total_cost": float(off["total_cost"].sum()),
        "plan_cost": float(off["plan_cost"].sum()),
        "emergency_cost": float(off["emergency_cost"].sum()),
        "emergency_kwh": float(off["emergency_kwh"].sum()),
        "emergency_days": int((off["emergency_kwh"] > 1e-6).sum()),
        "purchase_kwh": float(off["purchase_kwh"].sum()),
        "curtail_kwh": float(off["curtail_kwh"].sum()),
        "feb1_soc0": float(daily.loc[daily["date"] == "2025-02-01", "soc0_actual"].iloc[0]),
        "mean_soc24_actual": float(off["soc24_actual"].mean()),
        "vs_frozen": float(off["total_cost"].sum() - FROZEN_TOTAL),
        "n_validation_errors": len(errors),
        "elapsed_s": time.perf_counter() - t0,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return {"summary": summary, "daily": daily, "traces": traces}


def slot_diagnostics(prices: pd.DataFrame, traces: list[dict]) -> dict:
    """Ex-post waste in the frozen plan, at 10-minute resolution."""
    price = prices["price"].to_numpy(dtype=float)
    paid_curtail_kwh = paid_curtail_cost = 0.0
    curtail_total = curtail_soc_full = curtail_power_cap = 0.0
    em_kwh = em_cost = em_soc_at_min = 0.0
    em_slots = both_days = 0
    by_slot = np.zeros((144, 3))
    rows = []
    for tr in traces:
        g, cur, em = tr["purchase_kwh"], tr["curtail_kwh"], tr["emergency_kwh"]
        soc_prev = np.concatenate([[tr["soc0_kwh"]], tr["soc_end_kwh"][:-1]])
        overlap = np.minimum(g, cur)
        paid_curtail_kwh += float(overlap.sum())
        paid_curtail_cost += float(np.dot(price, overlap))
        curtail_total += float(cur.sum())
        headroom = np.maximum(0.0, (E_MAX_KWH - soc_prev) / ETA_CHARGE)
        hit = cur > 1e-9
        curtail_soc_full += float(cur[hit & (headroom <= P_MAX_KWH + 1e-6)].sum())
        curtail_power_cap += float(cur[hit & (headroom > P_MAX_KWH + 1e-6)].sum())
        em_kwh += float(em.sum())
        em_cost += float(np.dot(5.0 * price, em))
        em_hit = em > 1e-9
        em_slots += int(em_hit.sum())
        em_soc_at_min += float(em[em_hit & (soc_prev <= E_MIN_KWH + 1e-6)].sum())
        if em.sum() > 1e-6 and cur.sum() > 1e-6:
            both_days += 1
        by_slot[:, 0] += overlap
        by_slot[:, 1] += em
        by_slot[:, 2] += g
        rows.append(
            {
                "date": tr["date"],
                "paid_curtail_kwh": float(overlap.sum()),
                "paid_curtail_cost": float(np.dot(price, overlap)),
                "curtail_kwh": float(cur.sum()),
                "emergency_kwh": float(em.sum()),
            }
        )
    slot_frame = pd.DataFrame(
        {
            "slot": np.arange(144),
            "start_min": prices["start_min"].to_numpy(dtype=int),
            "price": price,
            "paid_curtail_kwh": by_slot[:, 0],
            "emergency_kwh": by_slot[:, 1],
            "purchase_kwh": by_slot[:, 2],
        }
    )
    return {
        "paid_and_curtailed_kwh": paid_curtail_kwh,
        "paid_and_curtailed_cost": paid_curtail_cost,
        "paid_and_curtailed_share_of_total": paid_curtail_cost / FROZEN_TOTAL,
        "curtail_total_kwh": curtail_total,
        "curtail_when_soc_full_kwh": curtail_soc_full,
        "curtail_when_power_capped_kwh": curtail_power_cap,
        "emergency_kwh": em_kwh,
        "emergency_cost": em_cost,
        "emergency_slots": em_slots,
        "emergency_kwh_with_empty_battery": em_soc_at_min,
        "emergency_kwh_with_energy_left": em_kwh - em_soc_at_min,
        "days_with_emergency_and_curtail": both_days,
        "_daily": pd.DataFrame(rows),
        "_slot": slot_frame,
    }


def sequential_hindsight(
    name: str,
    prices: pd.DataFrame,
    year: dict,
    banks: dict[float, list[dict]],
    water_value: float,
) -> dict:
    """Walk day by day. Try every quantile from the actual current SOC, keep the best."""
    price = prices["price"].to_numpy(dtype=float)
    official_start = pd.Timestamp(OFFICIAL_START)
    order = sorted(banks)
    days = [p for p in banks[order[0]] if pd.Timestamp(p["date"]) >= official_start]
    soc = E0_FEB1_KWH
    rows = []
    t0 = time.perf_counter()
    for i, ref in enumerate(days):
        day = int(ref["day"])
        load_act = year["load_kwh"][day]
        pv_act = year["pv_kwh"][day]
        best = None
        for q in order:
            pred = banks[q][day]
            plan = solve_day_lp(price, pred["load_kw"] * DT_HOURS, pred["pv_kw"] * DT_HOURS, soc)
            actual = simulate_day(price, load_act, pv_act, plan["purchase_kwh"], soc)
            cost = actual["plan_cost"] + actual["emergency_cost"]
            score = cost - water_value * float(actual["soc24_kwh"])
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "q": q,
                    "cost": cost,
                    "plan_cost": actual["plan_cost"],
                    "emergency_cost": actual["emergency_cost"],
                    "emergency_kwh": float(actual["emergency_kwh"].sum()),
                    "curtail_kwh": float(actual["curtail_kwh"].sum()),
                    "purchase_kwh": float(plan["purchase_kwh"].sum()),
                    "soc24": float(actual["soc24_kwh"]),
                }
        rows.append(
            {
                "date": ref["date"],
                "q_load": best["q"],
                "soc0_actual": soc,
                "soc24_actual": best["soc24"],
                "total_cost": best["cost"],
                "plan_cost": best["plan_cost"],
                "emergency_cost": best["emergency_cost"],
                "emergency_kwh": best["emergency_kwh"],
                "curtail_kwh": best["curtail_kwh"],
                "purchase_kwh": best["purchase_kwh"],
            }
        )
        soc = best["soc24"]
        if (i + 1) % 100 == 0:
            print(f"  {name}: {i + 1}/{len(days)} days", flush=True)
    daily = pd.DataFrame(rows)
    summary = {
        "name": name,
        "water_value_yuan_per_kwh": water_value,
        "n_official_days": int(len(daily)),
        "total_cost": float(daily["total_cost"].sum()),
        "plan_cost": float(daily["plan_cost"].sum()),
        "emergency_cost": float(daily["emergency_cost"].sum()),
        "emergency_kwh": float(daily["emergency_kwh"].sum()),
        "emergency_days": int((daily["emergency_kwh"] > 1e-6).sum()),
        "purchase_kwh": float(daily["purchase_kwh"].sum()),
        "curtail_kwh": float(daily["curtail_kwh"].sum()),
        "mean_soc24_actual": float(daily["soc24_actual"].mean()),
        "vs_frozen": float(daily["total_cost"].sum() - FROZEN_TOTAL),
        "q_counts": {str(q): int((daily["q_load"] == q).sum()) for q in order},
        "elapsed_s": time.perf_counter() - t0,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    daily.to_csv(DIAG_DIR / f"bound_daily_{name}.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    copy_raw_inputs()
    prices = load_prices()
    year = load_year_actuals(
        jan1_load=float(prices["typical_load_kw"].iloc[0]),
        jan1_pv=float(prices["typical_pv_kw"].iloc[0]),
    )
    audit_data(prices, year)
    dates = year["dates"]
    end_idx = int(np.where(pd.to_datetime(dates) == pd.Timestamp(OFFICIAL_END))[0][0])
    price = prices["price"].to_numpy(dtype=float)

    print("panels...", flush=True)
    load_panel = precompute_panel(year["load_kw"], dates)
    pv_panel = precompute_panel(year["pv_kw"], dates)
    print("forecast bank (xgb expanding load + 7d PV)...", flush=True)
    cache: dict = {}
    point, t_fc = collect_forecasts(
        "xgb_expanding", prices, year, 0, end_idx, load_panel, pv_panel, cache=cache, pv_source="baseline_7d"
    )
    print(f"bank done in {t_fc:.1f}s", flush=True)

    feb1 = pd.Timestamp(OFFICIAL_START)
    jan1 = pd.Timestamp(dates.iloc[0])
    banks = {q: bias_forecasts(point, year, q, round(1.0 - q, 4)) for q in Q_GRID}
    q82 = banks[0.8]
    summaries: list[dict] = []

    print("=== A. frozen policy + slot traces ===", flush=True)
    base = replay("C_q82_frozen", prices, year, q82, feb1, collect_traces=True)
    summaries.append(base["summary"])
    base["daily"].to_csv(DIAG_DIR / "daily_C_q82_frozen.csv", index=False, encoding="utf-8-sig")

    diag = slot_diagnostics(prices, base["traces"])
    diag["_daily"].to_csv(DIAG_DIR / "daily_waste_C_q82.csv", index=False, encoding="utf-8-sig")
    diag["_slot"].to_csv(DIAG_DIR / "slot_profile_C_q82.csv", index=False, encoding="utf-8-sig")
    waste = {k: v for k, v in diag.items() if not k.startswith("_")}
    (DIAG_DIR / "waste_C_q82.json").write_text(json.dumps(waste, ensure_ascii=False, indent=2), encoding="utf-8")
    print("waste", json.dumps(waste, ensure_ascii=False), flush=True)

    print("=== A. perfect information ===", flush=True)
    oracle = replay("oracle_perfect_info", prices, year, point, feb1, oracle=True)["summary"]
    summaries.append(oracle)

    print("=== A. Feb-1 SOC from a January roll ===", flush=True)
    summaries.append(replay("C_q82_jan_rolled_soc", prices, year, q82, jan1)["summary"])

    print("=== A. end-of-day energy credit ===", flush=True)
    for mu in (0.25, 0.4):
        summaries.append(replay(f"C_q82_mu{mu}", prices, year, q82, feb1, soc_mu=mu)["summary"])

    print("=== A. price-dependent quantile ===", flush=True)
    med = float(np.median(price))
    cheap = price <= med
    for lo, hi in ((0.7, 0.9), (0.6, 0.9), (0.7, 0.95)):
        biased = bias_forecasts(point, year, np.where(cheap, hi, lo), np.where(cheap, 1.0 - hi, 1.0 - lo))
        summaries.append(replay(f"C_price_q{int(lo * 100)}_{int(hi * 100)}", prices, year, biased, feb1)["summary"])

    print("=== A. intra-day re-dispatch ===", flush=True)
    summaries.append(replay("C_q82_mpc_stride6", prices, year, q82, feb1, dispatch="mpc", stride=6)["summary"])

    frame = pd.DataFrame(summaries).sort_values("total_cost")
    frame.to_csv(DIAG_DIR / "variant_comparison.csv", index=False, encoding="utf-8-sig")
    (DIAG_DIR / "variant_comparison.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False), flush=True)

    print("=== B. fixed quantile grid ===", flush=True)
    fixed = []
    for q in Q_GRID:
        if q == 0.8:
            fixed.append(base["summary"])
            continue
        res = replay(f"fixed_q{int(q * 100)}", prices, year, banks[q], feb1)
        fixed.append(res["summary"])
        res["daily"].to_csv(DIAG_DIR / f"bound_daily_fixed_q{int(q * 100)}.csv", index=False, encoding="utf-8-sig")

    print("=== B. clairvoyant daily quantile selector ===", flush=True)
    sel_myopic = sequential_hindsight("hindsight_myopic", prices, year, banks, water_value=0.0)
    water = float(np.mean(price)) * 0.9
    sel_water = sequential_hindsight("hindsight_watervalue", prices, year, banks, water_value=water)

    best_fixed = min(fixed, key=lambda s: s["total_cost"])
    best_sel = min([sel_myopic, sel_water], key=lambda s: s["total_cost"])
    oracle_total = oracle["total_cost"]
    report = {
        "frozen_official_total": FROZEN_TOTAL,
        "oracle_perfect_info_total": oracle_total,
        "gap_frozen_to_oracle": FROZEN_TOTAL - oracle_total,
        "best_fixed_quantile": best_fixed["name"],
        "best_fixed_total": best_fixed["total_cost"],
        "best_daily_hindsight": best_sel["name"],
        "best_daily_hindsight_total": best_sel["total_cost"],
        "hindsight_gain_vs_frozen": FROZEN_TOTAL - best_sel["total_cost"],
        "hindsight_share_of_oracle_gap": (FROZEN_TOTAL - best_sel["total_cost"]) / (FROZEN_TOTAL - oracle_total),
        "water_value_yuan_per_kwh": water,
        "q_grid": Q_GRID,
    }
    rows = fixed + [sel_myopic, sel_water]
    pd.DataFrame(rows).sort_values("total_cost").to_csv(
        DIAG_DIR / "bound_comparison.csv", index=False, encoding="utf-8-sig"
    )
    (DIAG_DIR / "bound_report.json").write_text(
        json.dumps({"report": report, "runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("wrote", DIAG_DIR, flush=True)


if __name__ == "__main__":
    main()
