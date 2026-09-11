"""Load attachments 1–2 and calendar-align 10-minute start timestamps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ATTACHMENT1_XLSX,
    ATTACHMENT2_XLSX,
    CLEAN_DIR,
    DT_HOURS,
    N_INTERVALS,
)

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from common.time_slots import (  # noqa: E402
    CALENDAR_END_MIN,
    CALENDAR_START_MIN,
    EXPECTED_RAW_START_MIN,
    minutes_to_label,
    parse_start_minutes,
    rotate_typical_day,
    stitch_year,
)


def _pick_column(columns, keywords: tuple[str, ...]) -> str:
    for col in columns:
        if any(key in str(col) for key in keywords):
            return col
    raise KeyError(f"cannot find column matching {keywords} in {list(columns)}")


def load_prices(path: Path | None = None) -> pd.DataFrame:
    raw = pd.read_excel(path or ATTACHMENT1_XLSX)
    time_col = _pick_column(raw.columns, ("时间", "时刻"))
    price_col = _pick_column(raw.columns, ("电价",))
    load_col = _pick_column(raw.columns, ("负载", "负荷"))
    pv_col = _pick_column(raw.columns, ("光伏",))
    starts = [parse_start_minutes(v) for v in raw[time_col]]
    if starts != EXPECTED_RAW_START_MIN:
        raise ValueError("attachment 1 timestamps are not start labels 00:10 ... 0:00+1")
    price = rotate_typical_day(pd.to_numeric(raw[price_col], errors="coerce").to_numpy())
    typical_load = rotate_typical_day(pd.to_numeric(raw[load_col], errors="coerce").to_numpy())
    typical_pv = rotate_typical_day(pd.to_numeric(raw[pv_col], errors="coerce").to_numpy())
    frame = pd.DataFrame(
        {
            "t": range(1, N_INTERVALS + 1),
            "start_min": CALENDAR_START_MIN,
            "end_min": CALENDAR_END_MIN,
            "price": price,
            "typical_load_kw": typical_load,
            "typical_pv_kw": typical_pv,
        }
    )
    if frame[["price", "typical_load_kw", "typical_pv_kw"]].isna().any().any():
        raise ValueError("attachment 1 has missing numeric values")
    if (frame["price"] <= 0).any() or (frame["typical_load_kw"] <= 0).any() or (frame["typical_pv_kw"] < 0).any():
        raise ValueError("attachment 1 has invalid price/load/PV")
    frame["t_start"] = frame["start_min"].map(minutes_to_label)
    frame["t_end"] = frame["end_min"].map(lambda m: "24:00" if int(m) >= 24 * 60 else minutes_to_label(m))
    return frame


def _wide_sheet(path: Path, sheet: str) -> tuple[pd.DatetimeIndex, np.ndarray, list]:
    raw = pd.read_excel(path, sheet_name=sheet)
    date_col = raw.columns[0]
    dates = pd.to_datetime(raw[date_col])
    slot_cols = list(raw.columns[1:])
    if len(slot_cols) != N_INTERVALS:
        raise ValueError(f"{sheet}: expected {N_INTERVALS} time columns, got {len(slot_cols)}")
    values = raw[slot_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return dates, values, slot_cols


def load_year_actuals(path: Path | None = None, jan1_load: float | None = None, jan1_pv: float | None = None) -> dict:
    source = path or ATTACHMENT2_XLSX
    load_dates, load_kw, load_cols = _wide_sheet(source, "小区负载")
    pv_dates, pv_kw, pv_cols = _wide_sheet(source, "光伏发电实际功率")
    if not load_dates.equals(pv_dates):
        raise ValueError("attachment 2 load and PV dates do not match")
    if [str(c) for c in load_cols] != [str(c) for c in pv_cols]:
        raise ValueError("attachment 2 load and PV time columns do not match")
    if load_dates.duplicated().any():
        raise ValueError("attachment 2 has duplicate dates")
    if pd.isna(load_kw).any() or pd.isna(pv_kw).any():
        raise ValueError("attachment 2 has missing values")
    if (load_kw < 0).any() or (pv_kw < 0).any():
        raise ValueError("attachment 2 has negative load or PV")

    starts = [parse_start_minutes(c) for c in load_cols]
    if starts != EXPECTED_RAW_START_MIN:
        raise ValueError("attachment 2 time columns are not start labels 00:10 ... 0:00+1")

    if jan1_load is None or jan1_pv is None:
        raise ValueError("jan1 midnight fill from typical day is required")
    load_kw = stitch_year(load_kw, jan1_load)
    pv_kw = stitch_year(pv_kw, jan1_pv)
    slot_end_min = np.array(CALENDAR_END_MIN, dtype=int)

    diffs = load_dates.diff().dropna()
    if not (diffs == pd.Timedelta(days=1)).all():
        raise ValueError("attachment 2 dates are not consecutive calendar days")

    return {
        "dates": load_dates.reset_index(drop=True),
        "load_kw": load_kw,
        "pv_kw": pv_kw,
        "load_kwh": load_kw * DT_HOURS,
        "pv_kwh": pv_kw * DT_HOURS,
        "slot_end_min": slot_end_min,
        "slot_cols": load_cols,
    }


def audit_data(prices: pd.DataFrame, year: dict) -> dict:
    dates = year["dates"]
    load_kw = year["load_kw"]
    pv_kw = year["pv_kw"]
    return {
        "n_price_slots": int(len(prices)),
        "price_dt_minutes": 10,
        "attachment2_dt_minutes": 10,
        "n_days": int(len(dates)),
        "date_start": str(dates.iloc[0].date()),
        "date_end": str(dates.iloc[-1].date()),
        "duplicate_dates": int(dates.duplicated().sum()),
        "missing_load": int(np.isnan(load_kw).sum()),
        "missing_pv": int(np.isnan(pv_kw).sum()),
        "duplicate_slots": int(len(year["slot_end_min"]) != len(set(year["slot_end_min"]))),
        "load_kw_min": float(np.min(load_kw)),
        "load_kw_max": float(np.max(load_kw)),
        "pv_kw_min": float(np.min(pv_kw)),
        "pv_kw_max": float(np.max(pv_kw)),
        "dt_hours": DT_HOURS,
        "kwh_check_load_day0": float(year["load_kwh"][0].sum() - load_kw[0].sum() * DT_HOURS),
        "alignment": "start_clock_stitched_to_calendar",
    }


def write_clean(prices: pd.DataFrame, year: dict, audit: dict) -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(CLEAN_DIR / "prices.csv", index=False, encoding="utf-8-sig")
    dates = pd.to_datetime(year["dates"]).dt.strftime("%Y-%m-%d")
    slot_labels = [f"{int(m):04d}" for m in year["slot_end_min"]]
    pd.DataFrame(year["load_kw"], index=dates, columns=slot_labels).to_csv(
        CLEAN_DIR / "load_kw.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(year["pv_kw"], index=dates, columns=slot_labels).to_csv(
        CLEAN_DIR / "pv_kw.csv", encoding="utf-8-sig"
    )
    (CLEAN_DIR / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
