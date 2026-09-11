"""Q3 entry: clean data, rolling LP, validate, export."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

Q3_DIR = Path(__file__).resolve().parent
CODE_DIR = Q3_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(Q3_DIR) not in sys.path:
    sys.path.insert(0, str(Q3_DIR))

from config import (
    ATTACHMENT1_XLSX,
    ATTACHMENT2_XLSX,
    ATTACHMENT3_XLSX,
    CLEAN_DIR,
    EXP_DIR,
    LOG_DIR,
    OFFICIAL_END,
    OFFICIAL_START,
    PHASE1_DIR,
    PHASE1_END,
    PREVIOUS_OFFICIAL_M1_COST,
    PV_P0_MODE,
    PV_P0_MODES,
    RESERVE_GAMMA,
    RESULT3_TEMPLATE_XLSX,
    RESULT_DIR,
)
from export_results import export_paper_tables, export_result3
from load_data import audit_data, load_prices, load_pv_hourly_forecasts, load_year_actuals, write_clean
from load_xgb import ensure_xgb_load
from pv_forecast import aligned_actual_pv, causal_sigma, interpolate_all
from rolling import POLICIES, run_span
from validate import leakage_errors, official_errors, summarize


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _day_index(dates, stamp: str) -> int:
    target = pd.Timestamp(stamp).normalize()
    idx = pd.to_datetime(dates).dt.normalize()
    hits = np.where(idx == target)[0]
    if len(hits) != 1:
        raise ValueError(f"cannot locate {stamp}")
    return int(hits[0])


def load_bundle(p0_mode: str = PV_P0_MODE, write: bool = True) -> dict:
    prices = load_prices()
    year = load_year_actuals(
        jan1_load=float(prices["typical_load_kw"].iloc[0]),
        jan1_pv=float(prices["typical_pv_kw"].iloc[0]),
    )
    forecasts = load_pv_hourly_forecasts()
    y_dates = [pd.Timestamp(d).date() for d in year["dates"]]
    f_dates = [pd.Timestamp(d).date() for d in forecasts["dates"]]
    if y_dates != f_dates:
        raise ValueError("attachment 2/3 dates do not align")
    audit = audit_data(prices, year, forecasts)
    audit["pv_p0_mode"] = p0_mode
    interp = interpolate_all(forecasts["hourly_kw"], year["pv_kw"], p0_mode)
    aligned = aligned_actual_pv(year["pv_kw"], interp)
    if write:
        # Frozen clean artifacts describe the frozen p0 rule only.
        write_clean(prices, year, forecasts, audit)
        np.save(CLEAN_DIR / "pv_interp_10min.npy", interp)
    typical = prices["typical_load_kw"].to_numpy(dtype=float)
    ensure_xgb_load(np.asarray(year["load_kw"], dtype=float), year["dates"], typical)
    return {
        "prices": prices,
        "year": year,
        "forecasts": forecasts,
        "audit": audit,
        "interp": interp,
        "aligned": aligned,
    }


def _policy_names(phase: str, override: str | None) -> list[str]:
    if override:
        names = [n.strip() for n in override.split(",") if n.strip()]
        if not names:
            raise ValueError("empty --policies")
        return names
    if phase == "phase1":
        return ["N0", "B0", "M1", "TV", "LA", "M2", "M0L", "oracle"]
    if phase == "official":
        return ["N0", "LA", "M0L"]
    raise ValueError("phase must be phase1 or official")


def run_phase(
    bundle: dict,
    phase: str,
    policy_names: list[str] | None = None,
    export: bool = True,
    out_dir: Path | None = None,
) -> dict:
    dates = bundle["year"]["dates"]
    start_day = _day_index(dates, OFFICIAL_START)
    if phase == "phase1":
        end_day = _day_index(dates, PHASE1_END) + 1
        default_dir = PHASE1_DIR
    elif phase == "official":
        end_day = _day_index(dates, OFFICIAL_END) + 1
        default_dir = RESULT_DIR
    else:
        raise ValueError("phase must be phase1 or official")
    out_dir = out_dir or default_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    price144 = bundle["prices"]["price"].to_numpy(dtype=float)
    load_kwh = bundle["year"]["load_kwh"]
    pv_kwh = bundle["year"]["pv_kwh"]
    typical = bundle["prices"]["typical_load_kw"].to_numpy(dtype=float)
    interp = bundle["interp"]
    aligned = bundle["aligned"]

    def sigma_fn(day: int):
        return causal_sigma(interp, aligned, day)

    metrics = {}
    payloads = {}
    names = policy_names or _policy_names(phase, None)
    for name in names:
        t0 = time.perf_counter()
        if name == "oracle":
            payload = run_span(
                start_day,
                end_day,
                0,
                price144,
                load_kwh,
                pv_kwh,
                dates,
                typical,
                interp,
                sigma_fn,
                None,
                oracle=True,
            )
        else:
            payload = run_span(
                start_day,
                end_day,
                0,
                price144,
                load_kwh,
                pv_kwh,
                dates,
                typical,
                interp,
                sigma_fn,
                POLICIES[name],
                oracle=False,
            )
        elapsed = time.perf_counter() - t0
        leak = leakage_errors(payload["records"], load_kwh, dates, typical)
        gate = official_errors(
            payload["official"],
            dates,
            payload["soc_track"],
            load_kwh,
            pv_kwh,
            price144,
            expected_n=end_day - start_day,
        )
        summary = summarize(payload["official"])
        summary["elapsed_s"] = elapsed
        summary["leakage_errors"] = leak
        summary["gate_errors"] = gate
        metrics[name] = summary
        payloads[name] = payload
        (out_dir / f"{name}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if name in ("M0", "M0L", "N0", "M1", "LA"):
            jan1 = payload["records"][0]
            if abs(jan1["actual"]["soc0_kwh"] - 6000.0) > 1e-3:
                raise RuntimeError("Jan 1 0:00 SOC != 6000")
            if start_day > 0:
                if abs(payload["records"][start_day - 1]["soc24_kwh"] - payload["official"][0]["actual"]["soc0_kwh"]) > 1e-3:
                    raise RuntimeError("Feb 1 SOC0 != Jan 31 SOC24")
        if name == "N0" and summary["n_days_with_intraday_adjust"] != 0:
            raise RuntimeError("N0 must keep the 0:00 contract all day")
        if leak:
            raise RuntimeError(f"{name} leakage: {leak[:8]}")
        if gate:
            raise RuntimeError(f"{name} gate: {gate[:8]}")
        if name == "oracle":
            rivals = [c["total_cost"] for n, c in metrics.items() if n != "oracle"]
            if rivals and metrics["oracle"]["total_cost"] - 1e-6 > min(rivals):
                metrics["oracle"]["note"] = "oracle cost above best policy"

    if phase == "official" and any(n != "oracle" for n in names):
        candidates = [n for n in names if n != "oracle"]
        winner = min(candidates, key=lambda n: metrics[n]["total_cost"])
        metrics["official_winner"] = winner
        jan31 = start_day - 1
        if jan31 >= 0:
            soc_feb1 = payloads[winner]["soc_track"][start_day]
            last_warm = payloads[winner]["records"][jan31]
            if abs(last_warm["soc24_kwh"] - soc_feb1) > 1e-6:
                raise RuntimeError("Feb 1 SOC0 != Jan 31 SOC24")
        if export:
            winner_cost = metrics[winner]["total_cost"]
            metrics["previous_endpoint_m1_cost"] = PREVIOUS_OFFICIAL_M1_COST
            metrics["freeze_official"] = True
            metrics["winner_cost"] = winner_cost
            win = payloads[winner]
            export_result3(win["official"], bundle["year"]["slot_end_min"], RESULT_DIR / "result3.xlsx")
            export_paper_tables(win["official"], bundle["year"]["slot_end_min"], price144)

    return {"metrics": metrics, "payloads": payloads, "start_day": start_day, "end_day": end_day}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("phase1", "official"), default="phase1")
    parser.add_argument("--policies", default=None, help="comma-separated policy names")
    parser.add_argument("--no-export", action="store_true", help="do not overwrite result3.xlsx")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="write under results/Q3/exp/<name> instead of the frozen directories",
    )
    parser.add_argument("--pv-p0", choices=PV_P0_MODES, default=PV_P0_MODE)
    parser.add_argument(
        "--reserve-gamma",
        type=float,
        default=RESERVE_GAMMA,
        help="hold SOC back for later dearer deficits; 0 keeps the greedy Q2 rule",
    )
    args = parser.parse_args()
    if args.out_dir and not args.no_export:
        raise SystemExit("--out-dir is for experiments; pass --no-export as well")
    if args.pv_p0 != PV_P0_MODE and not args.out_dir:
        raise SystemExit("--pv-p0 changes the frozen inputs; route it to --out-dir")
    if args.reserve_gamma != RESERVE_GAMMA and not args.out_dir:
        raise SystemExit("--reserve-gamma changes the playback rule; route it to --out-dir")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    exp_dir = (EXP_DIR / args.out_dir) if args.out_dir else None
    bundle = load_bundle(p0_mode=args.pv_p0, write=exp_dir is None)
    names = _policy_names(args.phase, args.policies)
    if args.reserve_gamma != RESERVE_GAMMA:
        for name in names:
            if name in POLICIES:
                POLICIES[name] = replace(POLICIES[name], reserve_gamma=args.reserve_gamma)
    result = run_phase(
        bundle,
        args.phase,
        policy_names=names,
        export=not args.no_export,
        out_dir=exp_dir,
    )
    elapsed = time.perf_counter() - started
    out_dir = exp_dir or (PHASE1_DIR if args.phase == "phase1" else RESULT_DIR)
    metrics_out = dict(result["metrics"])
    manifest_path = out_dir / "run_manifest.json"
    if args.policies and manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged = dict(old.get("metrics", {}))
        incoming = {k: v for k, v in result["metrics"].items() if k != "official_winner"}
        merged.update(incoming)
        if args.phase == "official" and not args.no_export:
            merged["official_winner"] = result["metrics"].get("official_winner")
        metrics_out = merged
    manifest = {
        "phase": args.phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed,
        "start_day": result["start_day"],
        "end_day": result["end_day"],
        "inputs": {
            "attachment1": _sha256(ATTACHMENT1_XLSX),
            "attachment2": _sha256(ATTACHMENT2_XLSX),
            "attachment3": _sha256(ATTACHMENT3_XLSX),
            "template": _sha256(RESULT3_TEMPLATE_XLSX),
        },
        "audit": bundle["audit"],
        "metrics": metrics_out,
        "ran_policies": names,
        "pv_p0_mode": args.pv_p0,
        "reserve_gamma": args.reserve_gamma,
    }
    freeze = result["metrics"].get("freeze_official")
    if args.no_export:
        (out_dir / "ablation_metrics.json").write_text(
            json.dumps(result["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if exp_dir is not None:
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    elif args.phase == "official" and freeze is False:
        (RESULT_DIR / "candidate_run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (RESULT_DIR / "candidate_metrics.json").write_text(
            json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.phase == "official":
            (RESULT_DIR / "metrics.json").write_text(
                json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
