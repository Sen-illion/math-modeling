"""Q4-2: Q2 dispatch with hat0 planning prices and attachment-4 settlement."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

Q4_DIR = Path(__file__).resolve().parent
CODE_DIR = Q4_DIR.parent
Q2_DIR = CODE_DIR / "Q2"
sys.path.insert(0, str(Q2_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from adaptive import causal_percentile, net_residual_std, snap_to_ladder  # noqa: E402
from config import (  # noqa: E402
    ADAPTIVE_WARMUP_DAYS,
    DT_HOURS,
    E0_FEB1_KWH,
    OFFICIAL_END,
    OFFICIAL_START,
    PHASE1_END,
    Q_LADDER,
    Q_WARMUP,
)
from forecast import precompute_panel  # noqa: E402
from load_data import audit_data, load_prices, load_year_actuals  # noqa: E402
from model_lp import solve_day_lp, validate_plan  # noqa: E402
from run_q2 import (  # noqa: E402
    ADAPTIVE_K_GRID,
    ADAPTIVE_QMIN_GRID,
    apply_bias_forecasts,
    collect_forecasts,
    copy_raw_inputs,
)
from simulate import simulate_day, validate_actual  # noqa: E402

from Q4.config import (  # noqa: E402
    OOS_START,
    ORACLE_DIAG_DIR,
    PHASE1_DIR,
    Q2_ADAPTIVE_POLICY,
    Q2_POLICY,
    Q2_SELECTED_RULE,
    RESULT_DIR,
    RETUNE_DIAG_DIR,
    TUNE_END,
    TUNE_INNER_END,
    TUNE_SELECT_START,
)
from Q4.export_results import export_result4_2, export_specified_q42, merge_metrics, write_run_manifest
from Q4.price_bank import load_price_bank


def _day_index(dates, stamp: str) -> int:
    target = pd.Timestamp(stamp).normalize()
    idx = pd.to_datetime(dates).dt.normalize()
    hits = np.where(idx == target)[0]
    if len(hits) != 1:
        raise ValueError(f"cannot locate {stamp}")
    return int(hits[0])


def _load_adaptive_rule() -> dict:
    if not Q2_SELECTED_RULE.exists():
        raise FileNotFoundError(f"missing {Q2_SELECTED_RULE}")
    rule = json.loads(Q2_SELECTED_RULE.read_text(encoding="utf-8"))
    if rule.get("feature") != "resid_vol7":
        raise RuntimeError(f"expected resid_vol7 rule, got {rule.get('feature')}")
    return rule


def _plan_price(plan_source: str, hat0_day: np.ndarray, settle: np.ndarray, stamp: str, leak: list[str]) -> np.ndarray:
    if plan_source == "oracle":
        return settle
    if plan_source == "hat0":
        if np.allclose(hat0_day, settle, atol=1e-12):
            leak.append(f"{stamp} hat0 equals today's actual price")
        return hat0_day
    raise ValueError(f"unknown plan_source {plan_source}")


def _n_official_days(end_stamp: str) -> int:
    return int((pd.Timestamp(end_stamp).normalize() - pd.Timestamp(OFFICIAL_START).normalize()).days) + 1


def prepare_q42(end_stamp: str, margin: str = "adaptive") -> dict:
    copy_raw_inputs()
    prices = load_prices()
    year = load_year_actuals(
        jan1_load=float(prices["typical_load_kw"].iloc[0]),
        jan1_pv=float(prices["typical_pv_kw"].iloc[0]),
    )
    audit_data(prices, year)
    bank = load_price_bank()
    if not year["dates"].reset_index(drop=True).equals(bank["dates"].reset_index(drop=True)):
        raise RuntimeError("Q2 year dates do not match attachment 4")
    dates = year["dates"]
    end_idx = _day_index(dates, end_stamp)
    load_panel = precompute_panel(year["load_kw"], dates)
    pv_panel = precompute_panel(year["pv_kw"], dates)
    policy = Q2_ADAPTIVE_POLICY if margin == "adaptive" else Q2_POLICY
    point, forecast_s = collect_forecasts(
        "xgb_expanding",
        prices,
        year,
        0,
        end_idx,
        load_panel,
        pv_panel,
        pv_source=policy["pv_source"],
    )
    rule = _load_adaptive_rule() if margin == "adaptive" else None
    ladder = tuple(float(q) for q in (rule.get("ladder", Q_LADDER) if rule else Q_LADDER))
    banks = None
    residuals = None
    by_day = None
    if margin == "adaptive":
        banks = {
            float(q): {
                int(row["day"]): row
                for row in apply_bias_forecasts(point, year, float(q), round(1.0 - float(q), 6))
            }
            for q in ladder
        }
        residuals = {
            int(pred["day"]): {
                "load": year["load_kw"][int(pred["day"])] - np.asarray(pred["load_kw"], dtype=float),
                "pv": year["pv_kw"][int(pred["day"])] - np.asarray(pred["pv_kw"], dtype=float),
            }
            for pred in point
        }
    elif margin == "fixed":
        biased = apply_bias_forecasts(point, year, Q2_POLICY["q_load"], Q2_POLICY["q_pv"])
        by_day = {int(row["day"]): row for row in biased}
    else:
        raise ValueError(f"unknown margin {margin}")
    return {
        "prices": prices,
        "year": year,
        "bank": bank,
        "point": point,
        "banks": banks,
        "residuals": residuals,
        "by_day": by_day,
        "ladder": ladder,
        "rule": rule,
        "policy": policy,
        "forecast_s": forecast_s,
        "margin": margin,
    }


def run_q4_2(
    end_stamp: str,
    collect_traces: bool,
    plan_source: str = "hat0",
    margin: str = "fixed",
    q_min: float | None = None,
    k: float | None = None,
    prep: dict | None = None,
) -> dict:
    if prep is None:
        prep = prepare_q42(end_stamp, margin=margin)
    elif prep["margin"] != margin:
        raise ValueError("prep margin does not match")
    prices = prep["prices"]
    year = prep["year"]
    bank = prep["bank"]
    point = prep["point"]
    banks = prep["banks"]
    residuals = prep["residuals"]
    by_day = prep["by_day"]
    ladder = prep["ladder"]
    rule = dict(prep["rule"] or {})
    policy = prep["policy"]
    forecast_s = prep["forecast_s"]
    if margin == "adaptive":
        q_min = float(rule["q_min"] if q_min is None else q_min)
        k = float(rule["k"] if k is None else k)
        rule["q_min"] = q_min
        rule["k"] = k
        rule["label"] = f"resid_vol7_qmin{q_min:.2f}_k{k:.2f}"
        warmup = float(rule.get("q_warmup", Q_WARMUP))
        warmup_days = int(rule.get("warmup_days", ADAPTIVE_WARMUP_DAYS))
    elif margin == "fixed":
        q_min = float(Q2_POLICY["q_load"])
        k = 0.0
        warmup = q_min
        warmup_days = ADAPTIVE_WARMUP_DAYS
    else:
        raise ValueError(f"unknown margin {margin}")

    hat0 = bank["hat0"]
    actual_p = bank["actual"]
    sim_start = pd.Timestamp(OFFICIAL_START)
    end_ts = pd.Timestamp(end_stamp)
    soc = E0_FEB1_KWH
    rows = []
    traces = []
    errors = []
    leak = []
    feature_history: list[float] = []
    t0 = time.perf_counter()

    for point_pred in point:
        stamp = pd.Timestamp(point_pred["date"])
        if stamp < sim_start:
            continue
        if stamp > end_ts:
            break
        day = int(point_pred["day"])
        if margin == "adaptive":
            past = [
                residuals[d]
                for d in range(max(0, day - warmup_days), day)
                if d in residuals
            ]
            usable = len(past) >= warmup_days
            value = net_residual_std(past) if usable else float("nan")
            z = causal_percentile(feature_history, value) if usable else None
            if z is None:
                q_load = warmup
                source = "warmup"
            else:
                q_load = float(snap_to_ladder(q_min + k * z, ladder))
                source = "rule"
            if usable:
                feature_history.append(value)
            bank_key = min(banks, key=lambda q: abs(q - float(q_load)))
            pred = banks[bank_key][day]
            if int(pred["day"]) != day:
                raise RuntimeError(f"bank index mismatch at day {day}")
        else:
            pred = by_day[day]
            q_load = float(Q2_POLICY["q_load"])
            source = "fixed"
            value = float("nan")
            z = None

        settle_price = actual_p[day]
        plan_price = _plan_price(plan_source, hat0[day], settle_price, pred["date"], leak)
        load_hat = pred["load_kw"] * DT_HOURS
        pv_hat = pred["pv_kw"] * DT_HOURS
        plan = solve_day_lp(
            plan_price,
            load_hat,
            pv_hat,
            soc,
            terminal_mode="none",
            soc_mu=float(policy["soc_mu"]),
        )
        errors.extend(validate_plan(plan, plan_price, load_hat, pv_hat))
        actual = simulate_day(
            settle_price,
            year["load_kwh"][day],
            year["pv_kwh"][day],
            plan["purchase_kwh"],
            soc,
        )
        errors.extend(validate_actual(actual, plan["purchase_kwh"], year["load_kwh"][day], year["pv_kwh"][day]))
        rebuilt_plan = float(np.dot(settle_price, plan["purchase_kwh"]))
        rebuilt_em = float(np.dot(5.0 * settle_price, actual["emergency_kwh"]))
        if abs(rebuilt_plan - actual["plan_cost"]) > 1e-6:
            errors.append(f"{pred['date']} plan cost mismatch")
        rows.append(
            {
                "date": pred["date"],
                "official": True,
                "q_load": float(q_load),
                "q_pv": round(1.0 - float(q_load), 6),
                "q_source": source,
                "feature_value": value,
                "feature_percentile": z if z is not None else np.nan,
                "soc0_actual": float(soc),
                "soc24_actual": float(actual["soc24_kwh"]),
                "soc24_plan": float(plan["soc_end_kwh"][-1]),
                "purchase_kwh": float(plan["purchase_kwh"].sum()),
                "plan_cost": rebuilt_plan,
                "emergency_kwh": float(actual["emergency_kwh"].sum()),
                "emergency_slots": int(actual["n_emergency_slots"]),
                "emergency_cost": rebuilt_em,
                "total_cost": rebuilt_plan + rebuilt_em,
                "curtail_kwh": float(actual["curtail_kwh"].sum()),
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
                    "plan_cost": rebuilt_plan,
                    "emergency_cost": rebuilt_em,
                }
            )
        soc = float(actual["soc24_kwh"])

    daily = pd.DataFrame(rows)
    summary = {
        "policy": policy["name"],
        "margin": margin,
        "plan_source": plan_source,
        "plan_price": "attachment4_oracle_diagnostic" if plan_source == "oracle" else "hat0_expanding_layers",
        "diagnostic_only": plan_source == "oracle",
        "settle_price": "attachment4_actual",
        "n_official_days": int(len(daily)),
        "total_cost": float(daily["total_cost"].sum()) if len(daily) else 0.0,
        "plan_cost": float(daily["plan_cost"].sum()) if len(daily) else 0.0,
        "emergency_cost": float(daily["emergency_cost"].sum()) if len(daily) else 0.0,
        "emergency_kwh": float(daily["emergency_kwh"].sum()) if len(daily) else 0.0,
        "purchase_kwh": float(daily["purchase_kwh"].sum()) if len(daily) else 0.0,
        "feb1_soc0": float(daily["soc0_actual"].iloc[0]) if len(daily) else None,
        "elapsed_s": time.perf_counter() - t0 + forecast_s,
        "forecast_s": forecast_s,
        "n_validation_errors": len(errors),
        "validation_errors_head": errors[:20],
        "price_leakage": leak[:20],
        "n_price_leakage": len(leak),
        "end_date": end_stamp,
        "price_source": bank["source"],
        "adaptive_rule": rule,
        "q_min": q_min,
        "k": k,
        "q_counts": {str(k): int(v) for k, v in daily["q_load"].value_counts().sort_index().items()} if len(daily) else {},
    }
    expected = _n_official_days(end_stamp)
    if summary["n_official_days"] != expected:
        raise RuntimeError(f"Q4-2 expected {expected} official days, got {summary['n_official_days']}")
    if summary["feb1_soc0"] is not None and abs(summary["feb1_soc0"] - E0_FEB1_KWH) > 1e-6:
        raise RuntimeError(f"Feb 1 SOC0 must be {E0_FEB1_KWH}")
    if errors:
        raise RuntimeError(f"Q4-2 validation failed: {errors[:5]}")
    return {"daily": daily, "summary": summary, "traces": traces, "prices": prices, "bank": bank}


def _window_slice(daily: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    stamps = pd.to_datetime(daily["date"])
    mask = (stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))
    return daily.loc[mask]


def sweep_q42_buffers(out_dir: Path | None = None) -> pd.DataFrame:
    """Replay resid_vol7 q_min x k on cached point/banks. Nested pick happens in sweep_q4_buffers."""
    out_dir = out_dir or RETUNE_DIAG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    prep = prepare_q42(OFFICIAL_END, margin="adaptive")
    windows = (
        ("inner", OFFICIAL_START, TUNE_INNER_END),
        ("select", TUNE_SELECT_START, TUNE_END),
        ("tune", OFFICIAL_START, TUNE_END),
        ("oos", OOS_START, OFFICIAL_END),
        ("full", OFFICIAL_START, OFFICIAL_END),
    )
    rows = []
    daily_parts = []
    for q_min in ADAPTIVE_QMIN_GRID:
        for k in ADAPTIVE_K_GRID:
            label = f"resid_vol7_qmin{q_min:.2f}_k{k:.2f}"
            print("q4-2 sweep", label, flush=True)
            result = run_q4_2(
                OFFICIAL_END,
                collect_traces=False,
                plan_source="hat0",
                margin="adaptive",
                q_min=float(q_min),
                k=float(k),
                prep=prep,
            )
            daily = result["daily"]
            daily_parts.append(daily.assign(q_min=float(q_min), k=float(k), label=label))
            summary = result["summary"]
            row = {
                "label": label,
                "feature": "resid_vol7",
                "q_min": float(q_min),
                "k": float(k),
                "full_cost": float(summary["total_cost"]),
                "full_emergency_kwh": float(summary["emergency_kwh"]),
                "full_plan_cost": float(summary["plan_cost"]),
                "elapsed_s": float(summary["elapsed_s"]),
                "lp_s": float(summary["elapsed_s"] - summary["forecast_s"]),
                "q_counts": json.dumps(summary["q_counts"], ensure_ascii=False),
            }
            for name, start, end in windows:
                win = _window_slice(daily, start, end)
                row[f"{name}_cost"] = float(win["total_cost"].sum()) if len(win) else 0.0
                row[f"{name}_emergency_kwh"] = float(win["emergency_kwh"].sum()) if len(win) else 0.0
                row[f"{name}_n_days"] = int(len(win))
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["select_cost", "q_min", "k"]).reset_index(drop=True)
    frame.to_csv(out_dir / "q42_grid.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_parts, ignore_index=True).to_csv(
        out_dir / "q42_daily_all.csv", index=False, encoding="utf-8-sig"
    )
    print("wrote", out_dir / "q42_grid.csv", flush=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--plan-source", choices=("hat0", "oracle"), default="hat0")
    parser.add_argument("--margin", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dump-payload", type=Path, default=None)
    parser.add_argument("--q-min", type=float, default=None)
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--sweep-buffers", action="store_true")
    args = parser.parse_args()
    if args.sweep_buffers:
        sweep_q42_buffers(args.out_dir)
        return
    if args.plan_source == "oracle" and args.export:
        raise SystemExit("oracle is diagnostic-only and must not export result4-2.xlsx")
    if not args.full:
        args.phase1 = True
    end_stamp = args.end_date or (OFFICIAL_END if args.full else PHASE1_END)
    collect = bool(args.dump_payload) or bool(args.full and args.export) or (
        args.full and args.plan_source == "hat0" and args.out_dir is None
    )
    result = run_q4_2(
        end_stamp,
        collect_traces=collect,
        plan_source=args.plan_source,
        margin=args.margin,
        q_min=args.q_min,
        k=args.k,
    )
    if args.out_dir is not None:
        out_dir = args.out_dir
        stem = "q42_adaptive" if args.margin == "adaptive" else "q42"
    elif args.plan_source == "oracle":
        out_dir = ORACLE_DIAG_DIR
        stem = "q42_oracle"
    else:
        out_dir = RESULT_DIR if args.full else PHASE1_DIR
        stem = "q42"
    out_dir.mkdir(parents=True, exist_ok=True)
    result["daily"].to_csv(out_dir / f"{stem}_daily.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"{stem}_summary.json").write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dump_payload is not None:
        args.dump_payload.parent.mkdir(parents=True, exist_ok=True)
        args.dump_payload.write_bytes(pickle.dumps({"prices": result["prices"], "traces": result["traces"], "summary": result["summary"]}))
    print(json.dumps(result["summary"], ensure_ascii=False, default=str), flush=True)
    if args.full and args.export:
        dest = export_result4_2(result["prices"], result["traces"])
        tables = export_specified_q42(result["prices"], result["traces"])
        merge_metrics(q42_summary=result["summary"])
        write_run_manifest(
            {
                "q4_2": {
                    "policy": result["summary"]["policy"],
                    "summary": result["summary"],
                    "result_xlsx": str(dest),
                    "specified_tables": [str(p) for p in tables],
                }
            }
        )
        print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
