"""Cached causal point forecasts and their quantile-biased variants.

The point forecast for day D (XGB-Expanding load + 7-day same-slot PV) depends only
on days < D, so it is identical for every policy. Refitting it costs ~160 s, so it is
cached on disk under a fingerprint of the raw attachments and the model settings.

The residual quantile used to bias day D is also built from days < D only, so a whole
ladder of quantiles can be precomputed once and indexed by (q, day) during a replay.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ATTACHMENT1_XLSX,
    ATTACHMENT2_XLSX,
    CLEAN_DIR,
    FEATURE_COLS,
    FORECAST_BANK_NPZ,
    N_INTERVALS,
    XGB_PARAMS,
)
from forecast import apply_pv_night_zero, clip_nonneg, weekly_dow

BANK_MODEL = "xgb_expanding"
BANK_PV_SOURCE = "baseline_7d"
BANK_FORMAT = 2


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bank_fingerprint(start_idx: int, end_idx: int) -> str:
    payload = {
        "format": BANK_FORMAT,
        "model": BANK_MODEL,
        "pv_source": BANK_PV_SOURCE,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "xgb_params": XGB_PARAMS,
        "feature_cols": FEATURE_COLS,
        "attachment1": _file_digest(ATTACHMENT1_XLSX),
        "attachment2": _file_digest(ATTACHMENT2_XLSX),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _to_arrays(point: list[dict]) -> dict:
    return {
        "days": np.array([int(p["day"]) for p in point], dtype=np.int32),
        "dates": np.array([str(p["date"]) for p in point]),
        "load_kw": np.vstack([np.asarray(p["load_kw"], dtype=float) for p in point]),
        "load_weekly_kw": np.vstack([np.asarray(p["load_weekly_kw"], dtype=float) for p in point]),
        "pv_kw": np.vstack([np.asarray(p["pv_kw"], dtype=float) for p in point]),
        "xgb_used": np.array([bool(p["xgb_used"]) for p in point]),
    }


def _from_arrays(data) -> list[dict]:
    days = data["days"]
    dates = data["dates"]
    load_kw = data["load_kw"]
    weekly = data["load_weekly_kw"]
    pv_kw = data["pv_kw"]
    xgb_used = data["xgb_used"]
    return [
        {
            "day": int(days[i]),
            "date": str(dates[i]),
            "load_kw": np.asarray(load_kw[i], dtype=float),
            "load_weekly_kw": np.asarray(weekly[i], dtype=float),
            "pv_kw": np.asarray(pv_kw[i], dtype=float),
            "xgb_used": bool(xgb_used[i]),
            "pv_source": BANK_PV_SOURCE,
        }
        for i in range(len(days))
    ]


def load_point_bank(
    prices: pd.DataFrame,
    year: dict,
    start_idx: int,
    end_idx: int,
    load_panel: pd.DataFrame | None = None,
    pv_panel: pd.DataFrame | None = None,
    refresh: bool = False,
) -> list[dict]:
    """Point forecasts for days start_idx..end_idx, from cache when the fingerprint matches."""
    from forecast import precompute_panel
    from run_q2 import collect_forecasts

    want = bank_fingerprint(start_idx, end_idx)
    if not refresh and FORECAST_BANK_NPZ.exists():
        with np.load(FORECAST_BANK_NPZ, allow_pickle=False) as data:
            if str(data["fingerprint"]) == want:
                point = _from_arrays(data)
                print(f"forecast bank: cache hit ({len(point)} days) {FORECAST_BANK_NPZ}", flush=True)
                return point
        print("forecast bank: fingerprint changed, refitting", flush=True)

    if load_panel is None:
        print("forecast bank: precomputing causal feature panels...", flush=True)
        load_panel = precompute_panel(year["load_kw"], year["dates"])
    if pv_panel is None:
        pv_panel = precompute_panel(year["pv_kw"], year["dates"])
    t0 = time.perf_counter()
    point, _ = collect_forecasts(
        BANK_MODEL,
        prices,
        year,
        start_idx,
        end_idx,
        load_panel,
        pv_panel,
        cache={},
        pv_source=BANK_PV_SOURCE,
    )
    print(f"forecast bank: fitted in {time.perf_counter() - t0:.1f}s", flush=True)
    fallback = prices["typical_load_kw"].to_numpy(dtype=float) if "typical_load_kw" in prices.columns else year["load_kw"][0]
    dates = year["dates"]
    for pred in point:
        day = int(pred["day"])
        pred["load_weekly_kw"] = weekly_dow(year["load_kw"], day, dates, fallback)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FORECAST_BANK_NPZ, fingerprint=want, **_to_arrays(point))
    print("forecast bank: cached to", FORECAST_BANK_NPZ, flush=True)
    return point


def slot_quantile(residuals: list[np.ndarray], q) -> np.ndarray:
    """Per-slot residual quantile. q may be a scalar or a 144-vector."""
    stacked = np.vstack(residuals)
    if np.isscalar(q):
        return np.quantile(stacked, float(q), axis=0)
    q = np.asarray(q, dtype=float)
    return np.array([np.quantile(stacked[:, s], q[s]) for s in range(stacked.shape[1])])


def season_id(stamp) -> int:
    """Meteorological season: 0=DJF, 1=MAM, 2=JJA, 3=SON."""
    month = int(pd.Timestamp(stamp).month)
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def _pool_arrays(
    history: list[tuple[int, np.ndarray]],
    day: int,
    dates: pd.Series,
    pool: dict | None,
) -> list[np.ndarray]:
    """Causal residual subset for day D. Default is expanding (all days < D)."""
    if not history:
        return []
    spec = pool or {"mode": "expanding"}
    mode = spec.get("mode", "expanding")
    if mode == "expanding":
        picked = history
    elif mode == "rolling":
        picked = history[-int(spec["window"]) :]
    elif mode == "season":
        want = season_id(dates.iloc[day])
        picked = [(d, arr) for d, arr in history if season_id(dates.iloc[d]) == want]
    elif mode == "month":
        month = int(pd.Timestamp(dates.iloc[day]).month)
        picked = [(d, arr) for d, arr in history if int(pd.Timestamp(dates.iloc[d]).month) == month]
        if len(picked) < int(spec.get("min_days", 14)):
            picked = history
    else:
        raise ValueError(f"unknown residual pool mode {mode}")
    return [arr for _, arr in picked]


def bias_bank(point: list[dict], year: dict, q_load, q_pv, pool: dict | None = None) -> list[dict]:
    """Shift point forecasts by causal residual quantiles. Residuals use days < D only."""
    dates = year["dates"]
    load_hist: list[tuple[int, np.ndarray]] = []
    pv_hist: list[tuple[int, np.ndarray]] = []
    out = []
    for pred in point:
        day = int(pred["day"])
        load_plan = np.asarray(pred["load_kw"], dtype=float).copy()
        pv_plan = np.asarray(pred["pv_kw"], dtype=float).copy()
        load_pool = _pool_arrays(load_hist, day, dates, pool)
        pv_pool = _pool_arrays(pv_hist, day, dates, pool)
        if q_load is not None and load_pool:
            load_plan = load_plan + slot_quantile(load_pool, q_load)
        if q_pv is not None and pv_pool:
            pv_plan = pv_plan + slot_quantile(pv_pool, q_pv)
        load_hist.append((day, year["load_kw"][day] - pred["load_kw"]))
        pv_hist.append((day, year["pv_kw"][day] - pred["pv_kw"]))
        row = dict(pred)
        row["load_kw"] = clip_nonneg(load_plan)
        row["pv_kw"] = clip_nonneg(apply_pv_night_zero(pv_plan, year["pv_kw"][:day]))
        row["resid_pool_n"] = int(len(load_pool))
        out.append(row)
    return out


def mix_point(point: list[dict], lam: float) -> list[dict]:
    """Blend weekly and XGB load at the point-forecast layer. lam=1 is pure XGB."""
    lam = float(lam)
    out = []
    for pred in point:
        row = dict(pred)
        xgb = np.asarray(pred["load_kw"], dtype=float)
        weekly = np.asarray(pred["load_weekly_kw"], dtype=float)
        row["load_kw"] = clip_nonneg((1.0 - lam) * weekly + lam * xgb)
        row["load_mix"] = lam
        out.append(row)
    return out


def bias_net_bank(point: list[dict], year: dict, alpha: float, window: int | None = None) -> list[dict]:
    """Conservative net-load quantile: one residual on (L-P), not split load/PV.

    window=None uses all completed days (expanding). A positive window keeps only the
    last `window` completed residuals, matching the other paper's W=7 net quantile.
    """
    net_resid: list[np.ndarray] = []
    out = []
    for pred in point:
        day = int(pred["day"])
        load_hat = np.asarray(pred["load_kw"], dtype=float)
        pv_hat = np.asarray(pred["pv_kw"], dtype=float)
        net_hat = load_hat - pv_hat
        net_plan = net_hat.copy()
        if net_resid:
            pool = net_resid[-int(window) :] if window else net_resid
            net_plan = net_hat + slot_quantile(pool, alpha)
        pv_plan = clip_nonneg(apply_pv_night_zero(pv_hat.copy(), year["pv_kw"][:day]))
        load_plan = clip_nonneg(pv_plan + net_plan)
        net_resid.append((year["load_kw"][day] - year["pv_kw"][day]) - net_hat)
        row = dict(pred)
        row["load_kw"] = load_plan
        row["pv_kw"] = pv_plan
        row["risk"] = "net"
        row["alpha"] = float(alpha)
        row["resid_window"] = None if window is None else int(window)
        out.append(row)
    return out


def ladder_banks(point: list[dict], year: dict, ladder, pool: dict | None = None) -> dict[float, list[dict]]:
    """One biased bank per rung, keyed by q_load. q_pv mirrors it as 1 - q_load."""
    return {
        float(q): bias_bank(point, year, float(q), round(1.0 - float(q), 6), pool=pool) for q in ladder
    }


def point_residual_history(point: list[dict], year: dict) -> dict[int, dict]:
    """Per-day point-forecast residuals in kW, for building causal state features."""
    out = {}
    for pred in point:
        day = int(pred["day"])
        out[day] = {
            "load": year["load_kw"][day] - np.asarray(pred["load_kw"], dtype=float),
            "pv": year["pv_kw"][day] - np.asarray(pred["pv_kw"], dtype=float),
        }
    return out


def check_length(bank: list[dict]) -> None:
    for pred in bank:
        if len(pred["load_kw"]) != N_INTERVALS or len(pred["pv_kw"]) != N_INTERVALS:
            raise ValueError(f"{pred['date']}: forecast is not {N_INTERVALS} slots")
