"""Q4-3: frozen M1/N0 with hat0 locked all day; settlement uses attachment 4."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

Q4_DIR = Path(__file__).resolve().parent
CODE_DIR = Q4_DIR.parent
Q3_DIR = CODE_DIR / "Q3"
sys.path.insert(0, str(Q3_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from pv_forecast import causal_sigma
from quantile import QuantileBank
from rolling import POLICIES, run_day, settlement
from run_q3 import _day_index, load_bundle
from simulate import validate_actual as q3_validate_actual
from validate import leakage_errors, recompute_day_cost, summarize
from config import (
    ABS_TOL_KWH,
    BETA_LOCK_GRID,
    BETA_OPEN_GRID,
    E0_JAN1_KWH,
    E_MAX_KWH,
    E_MIN_KWH,
    PV_P0_MODE,
    REL_TOL,
)

from Q4.config import (
    OFFICIAL_END,
    OFFICIAL_START,
    ORACLE_DIAG_DIR,
    PHASE1_DIR,
    PHASE1_END,
    Q3_POLICY,
    RESULT_DIR,
    RETUNE_DIAG_DIR,
    TUNE_END,
    TUNE_INNER_END,
    TUNE_SELECT_START,
)
from Q4.export_results import export_paper_q43, export_result4_3, merge_metrics, write_run_manifest
from Q4.price_bank import load_price_bank


def _rebill(records: list[dict], actual: np.ndarray) -> None:
    for rec in records:
        rec["bill"] = settlement(
            actual[rec["day"]],
            rec["g_plan_kwh"],
            rec["g_adj_kwh"],
            rec["actual"]["emergency_kwh"],
        )


def _price_leakage(records: list[dict], hat0: np.ndarray, actual: np.ndarray) -> list[str]:
    errors = []
    for rec in records:
        if not rec.get("official"):
            continue
        day = rec["day"]
        if np.allclose(hat0[day], actual[day], atol=1e-12):
            errors.append(f"{rec['date']} hat0 equals today's actual price")
    return errors


def _n_official_days(end_stamp: str) -> int:
    return int((pd.Timestamp(end_stamp).normalize() - pd.Timestamp(OFFICIAL_START).normalize()).days) + 1


def _official_daily(official: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": rec["date"],
                "total_cost": float(rec["bill"]["total_cost"]),
                "plan_cost": float(rec["bill"]["plan_only_cost"]),
                "emergency_cost": float(rec["bill"]["emergency_cost"]),
                "emergency_kwh": float(np.asarray(rec["actual"]["emergency_kwh"]).sum()),
            }
            for rec in official
        ]
    )


def _window_slice(daily: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    stamps = pd.to_datetime(daily["date"])
    mask = (stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))
    return daily.loc[mask]


_UNSET = object()


def _policy_q(value) -> float | None:
    return None if value is None else float(value)


def run_policy_q43(
    name: str,
    end_stamp: str,
    plan_source: str = "hat0",
    bundle: dict | None = None,
    bank: dict | None = None,
    policy=None,
    beta_lock: float | None = None,
    beta_open: float | None = None,
    lookahead_hours: int | None = None,
    q_lock=_UNSET,
    q_open=_UNSET,
    q_evening=_UNSET,
    quantile_bank=None,
) -> dict:
    if bundle is None:
        bundle = load_bundle(write=False)
    if bank is None:
        bank = load_price_bank()
    dates = bundle["year"]["dates"]
    if not pd.to_datetime(dates).reset_index(drop=True).equals(bank["dates"].reset_index(drop=True)):
        raise RuntimeError("Q3 dates do not match attachment 4")
    start_day = _day_index(dates, OFFICIAL_START)
    end_day = _day_index(dates, end_stamp) + 1
    load_kwh = bundle["year"]["load_kwh"]
    pv_kwh = bundle["year"]["pv_kwh"]
    typical = bundle["prices"]["typical_load_kw"].to_numpy(dtype=float)
    interp = bundle["interp"]
    aligned = bundle["aligned"]
    hat0 = bank["hat0"]
    actual = bank["actual"]
    if policy is None:
        policy = POLICIES[name]
    updates = {}
    if beta_lock is not None:
        updates["beta_lock"] = float(beta_lock)
    if beta_open is not None:
        updates["beta_open"] = float(beta_open)
    if lookahead_hours is not None:
        updates["lookahead_hours"] = int(lookahead_hours)
    if q_lock is not _UNSET:
        updates["q_lock"] = _policy_q(q_lock)
    if q_open is not _UNSET:
        updates["q_open"] = _policy_q(q_open)
    if q_evening is not _UNSET:
        updates["q_evening"] = _policy_q(q_evening)
    if updates:
        policy = replace(policy, **updates)
    if policy.q_lock is not None and quantile_bank is None:
        quantile_bank = QuantileBank.build(bundle)

    def sigma_fn(day: int):
        return causal_sigma(interp, aligned, day)

    records = []
    soc = float(E0_JAN1_KWH)
    soc_track = {0: soc}
    t0 = time.perf_counter()
    for day in range(0, end_day):
        plan_price = actual[day] if plan_source == "oracle" else hat0[day]
        if plan_source not in {"hat0", "oracle"}:
            raise ValueError(f"unknown plan_source {plan_source}")
        rec = run_day(
            day,
            plan_price,
            load_kwh,
            pv_kwh,
            dates,
            typical,
            interp,
            sigma_fn(day),
            soc,
            policy,
            quantile_bank=quantile_bank,
        )
        rec["day"] = day
        rec["date"] = str(pd.Timestamp(dates.iloc[day]).date())
        rec["official"] = start_day <= day < end_day
        records.append(rec)
        soc = rec["soc24_kwh"]
        soc_track[day + 1] = soc
    _rebill(records, actual)
    official = [r for r in records if r["official"]]
    leak = leakage_errors(
        records,
        load_kwh,
        dates,
        typical,
        quantile_bank=quantile_bank,
        pv_kwh=pv_kwh,
    )
    if plan_source == "hat0":
        leak.extend(_price_leakage(official, hat0, actual))
    gate = []

    def _tol(scale: float) -> float:
        return max(ABS_TOL_KWH, REL_TOL * abs(scale))

    for rec in official:
        gate.extend(
            f"{rec['date']} {e}"
            for e in q3_validate_actual(rec["actual"], rec["g_adj_kwh"], load_kwh[rec["day"]], pv_kwh[rec["day"]])
        )
        rebuilt = recompute_day_cost(
            actual[rec["day"]], rec["g_plan_kwh"], rec["g_adj_kwh"], rec["actual"]["emergency_kwh"]
        )
        if abs(rebuilt["total_cost"] - rec["bill"]["total_cost"]) > _tol(max(rec["bill"]["total_cost"], 1.0)):
            gate.append(f"{rec['date']} settlement mismatch vs attachment-4 rebuild")
        soc0 = rec["actual"]["soc0_kwh"]
        if soc0 < E_MIN_KWH - ABS_TOL_KWH or soc0 > E_MAX_KWH + ABS_TOL_KWH:
            gate.append(f"{rec['date']} SOC0 out of bounds")
    for prev, cur in zip(official, official[1:]):
        if abs(prev["soc24_kwh"] - cur["actual"]["soc0_kwh"]) > ABS_TOL_KWH:
            gate.append(f"{cur['date']} SOC0 != previous 24:00")
    summary = summarize(official)
    summary.update(
        {
            "elapsed_s": time.perf_counter() - t0,
            "leakage_errors": leak,
            "gate_errors": gate,
            "plan_source": plan_source,
            "plan_price": "attachment4_oracle_locked_all_day" if plan_source == "oracle" else "hat0_locked_all_day",
            "diagnostic_only": plan_source == "oracle",
            "settle_price": "attachment4_actual",
            "end_date": end_stamp,
            "price_source": bank["source"],
            "policy": name,
            "pv_p0_mode": PV_P0_MODE,
            "beta_lock": float(policy.beta_lock),
            "beta_open": float(policy.beta_open),
            "lookahead_hours": int(policy.lookahead_hours),
            "q_lock": _policy_q(policy.q_lock),
            "q_open": _policy_q(policy.q_open),
            "q_evening": _policy_q(policy.q_evening),
        }
    )
    expected = _n_official_days(end_stamp)
    if summary["n_days"] != expected:
        raise RuntimeError(f"Q4-3 {name} expected {expected} official days, got {summary['n_days']}")
    if name == "N0" and summary["n_days_with_intraday_adjust"] != 0:
        raise RuntimeError("N0 must keep the 0:00 contract all day")
    if leak or gate:
        raise RuntimeError(f"Q4-3 {name} failed leak={leak[:5]} gate={gate[:5]}")
    return {
        "records": records,
        "official": official,
        "daily": _official_daily(official),
        "summary": summary,
        "bundle": bundle,
        "bank": bank,
        "end_min": bundle["year"]["slot_end_min"],
        "policy": policy,
    }


def sweep_q43_buffers(out_dir: Path | None = None) -> pd.DataFrame:
    """LA beta grid through TUNE_END, reusing one Q3 bundle. Nested pick happens in sweep_q4_buffers."""
    out_dir = out_dir or RETUNE_DIAG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(write=False)
    bank = load_price_bank()
    windows = (
        ("inner", OFFICIAL_START, TUNE_INNER_END),
        ("select", TUNE_SELECT_START, TUNE_END),
        ("tune", OFFICIAL_START, TUNE_END),
    )
    rows = []
    daily_parts = []
    for beta_lock in BETA_LOCK_GRID:
        for beta_open in BETA_OPEN_GRID:
            label = f"LA_bl{beta_lock:g}_bo{beta_open:g}"
            print("q4-3 sweep", label, flush=True)
            payload = run_policy_q43(
                "LA",
                TUNE_END,
                plan_source="hat0",
                bundle=bundle,
                bank=bank,
                beta_lock=float(beta_lock),
                beta_open=float(beta_open),
                lookahead_hours=24,
                q_lock=None,
            )
            daily = payload["daily"]
            daily_parts.append(
                daily.assign(beta_lock=float(beta_lock), beta_open=float(beta_open), label=label)
            )
            summary = payload["summary"]
            row = {
                "label": label,
                "policy": "LA",
                "beta_lock": float(beta_lock),
                "beta_open": float(beta_open),
                "tune_emergency_kwh": float(summary["emergency_kwh"]),
                "elapsed_s": float(summary["elapsed_s"]),
            }
            for name, start, end in windows:
                win = _window_slice(daily, start, end)
                row[f"{name}_cost"] = float(win["total_cost"].sum()) if len(win) else 0.0
                row[f"{name}_emergency_kwh"] = float(win["emergency_kwh"].sum()) if len(win) else 0.0
                row[f"{name}_n_days"] = int(len(win))
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["select_cost", "beta_lock", "beta_open"]).reset_index(drop=True)
    frame.to_csv(out_dir / "q43_grid.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_parts, ignore_index=True).to_csv(
        out_dir / "q43_tune_daily_all.csv", index=False, encoding="utf-8-sig"
    )
    print("wrote", out_dir / "q43_grid.csv", flush=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--policies", default=None)
    parser.add_argument("--plan-source", choices=("hat0", "oracle"), default="hat0")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dump-payload", type=Path, default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--beta-lock", type=float, default=None)
    parser.add_argument("--beta-open", type=float, default=None)
    parser.add_argument("--lookahead-hours", type=int, default=None)
    parser.add_argument("--q-lock", type=float, default=None)
    parser.add_argument("--q-open", type=float, default=None)
    parser.add_argument("--q-evening", type=float, default=None)
    parser.add_argument("--disable-quantile", action="store_true")
    parser.add_argument("--sweep-buffers", action="store_true")
    args = parser.parse_args()
    if args.sweep_buffers:
        sweep_q43_buffers(args.out_dir)
        return
    if args.plan_source == "oracle" and args.export:
        raise SystemExit("oracle is diagnostic-only and must not export result4-3.xlsx")
    if not args.full:
        args.phase1 = True
    end_stamp = args.end_date or (OFFICIAL_END if args.full else PHASE1_END)
    policy_names = [n.strip() for n in (args.policies or Q3_POLICY).split(",") if n.strip()]
    if args.out_dir is not None:
        out_dir = args.out_dir
        stem_prefix = "q43_"
        stem_suffix = ""
    elif args.plan_source == "oracle":
        out_dir = ORACLE_DIAG_DIR
        stem_prefix = "q43_"
        stem_suffix = "_oracle"
    else:
        out_dir = RESULT_DIR if args.full else PHASE1_DIR
        stem_prefix = "q43_"
        stem_suffix = ""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    winner = None
    winner_name = None
    for name in policy_names:
        print("running", name, args.plan_source, "through", end_stamp, flush=True)
        q_kwargs = {}
        if args.disable_quantile:
            q_kwargs["q_lock"] = None
        else:
            if args.q_lock is not None:
                q_kwargs["q_lock"] = args.q_lock
            if args.q_open is not None:
                q_kwargs["q_open"] = args.q_open
            if args.q_evening is not None:
                q_kwargs["q_evening"] = args.q_evening
        payload = run_policy_q43(
            name,
            end_stamp,
            plan_source=args.plan_source,
            beta_lock=args.beta_lock,
            beta_open=args.beta_open,
            lookahead_hours=args.lookahead_hours,
            **q_kwargs,
        )
        metrics[name] = payload["summary"]
        (out_dir / f"{stem_prefix}{name}{stem_suffix}_summary.json").write_text(
            json.dumps(payload["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        payload["daily"].to_csv(
            out_dir / f"{stem_prefix}{name}{stem_suffix}_daily.csv", index=False, encoding="utf-8-sig"
        )
        print(name, payload["summary"]["total_cost"], flush=True)
        if name == Q3_POLICY:
            winner = payload
            winner_name = name
    metrics["official_policy"] = Q3_POLICY
    metrics_name = "q43_oracle_metrics.json" if args.plan_source == "oracle" else "q43_metrics.json"
    (out_dir / metrics_name).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dump_payload is not None:
        if winner is None:
            raise SystemExit(f"{Q3_POLICY} must be in --policies to dump a payload")
        args.dump_payload.parent.mkdir(parents=True, exist_ok=True)
        args.dump_payload.write_bytes(
            pickle.dumps(
                {
                    "official": winner["official"],
                    "end_min": winner["end_min"],
                    "summary": winner["summary"],
                    "metrics": metrics,
                }
            )
        )
    if args.full and args.export:
        if winner is None:
            raise SystemExit(f"{Q3_POLICY} must be in --policies to export result4-3")
        dest = export_result4_3(winner["official"], winner["end_min"])
        tables = export_paper_q43(winner["official"], winner["end_min"])
        merge_metrics(q43_metrics=metrics)
        write_run_manifest(
            {
                "q4_3": {
                    "official_policy": winner_name,
                    "metrics": metrics,
                    "result_xlsx": str(dest),
                    "specified_tables": [str(p) for p in tables],
                }
            }
        )
        print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
