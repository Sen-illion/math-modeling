"""Load attachments 1–3 and align 10-minute endpoint timestamps."""

from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ATTACHMENT1_XLSX,
    ATTACHMENT2_XLSX,
    ATTACHMENT3_XLSX,
    CLEAN_DIR,
    DT_HOURS,
    ISSUE_HOURS,
    N_INTERVALS,
)


def parse_end_minutes(value) -> int:
    if isinstance(value, time):
        minutes = value.hour * 60 + value.minute
        return 24 * 60 if minutes == 0 else minutes
    if isinstance(value, datetime):
        minutes = value.hour * 60 + value.minute
        return 24 * 60 if minutes == 0 else minutes
    text = str(value).strip().replace("：", ":")
    compact = text.replace(" ", "")
    if "+1" in compact or compact in {"24:00", "24:00:00"}:
        return 24 * 60
    parts = compact.split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    minutes = hour * 60 + minute
    return 24 * 60 if minutes == 0 else minutes


def _minutes_to_label(minutes: int) -> str:
    if minutes >= 24 * 60:
        return "24:00"
    hour, minute = divmod(int(minutes), 60)
    return f"{hour:02d}:{minute:02d}"


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
    frame = pd.DataFrame(
        {
            "t": range(1, len(raw) + 1),
            "time_raw": raw[time_col],
            "price": pd.to_numeric(raw[price_col], errors="coerce"),
            "typical_load_kw": pd.to_numeric(raw[load_col], errors="coerce"),
            "typical_pv_kw": pd.to_numeric(raw[pv_col], errors="coerce"),
        }
    )
    if len(frame) != N_INTERVALS:
        raise ValueError(f"attachment 1 expected {N_INTERVALS} rows, got {len(frame)}")
    if frame[["price", "typical_load_kw", "typical_pv_kw"]].isna().any().any():
        raise ValueError("attachment 1 has missing numeric values")
    if (frame["price"] <= 0).any() or (frame["typical_load_kw"] <= 0).any() or (frame["typical_pv_kw"] < 0).any():
        raise ValueError("attachment 1 has invalid price/load/PV")
    frame["end_min"] = frame["time_raw"].map(parse_end_minutes)
    frame["start_min"] = frame["end_min"] - 10
    if frame["end_min"].tolist() != list(range(10, 24 * 60 + 1, 10)):
        raise ValueError("attachment 1 timestamps are not a continuous 10-minute cover")
    frame["t_start"] = frame["start_min"].map(_minutes_to_label)
    frame["t_end"] = frame["end_min"].map(_minutes_to_label)
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


def load_year_actuals(path: Path | None = None) -> dict:
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

    slot_end_min = np.array([parse_end_minutes(c) for c in load_cols], dtype=int)
    expected = np.array(list(range(10, 24 * 60 + 1, 10)), dtype=int)
    if not np.array_equal(slot_end_min, expected):
        raise ValueError("attachment 2 time columns are not a continuous 10-minute cover")

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


def load_pv_hourly_forecasts(path: Path | None = None) -> dict:
    raw = pd.read_excel(path or ATTACHMENT3_XLSX)
    date_col = raw.columns[0]
    issue_col = raw.columns[1]
    fc_cols = list(raw.columns[2:26])
    if len(fc_cols) != 24:
        raise ValueError(f"attachment 3 expected 24 forecast columns, got {len(fc_cols)}")
    dates = pd.to_datetime(raw[date_col]).ffill()
    issue_map = {"0:00": 0, "6:00": 6, "12:00": 12, "18:00": 18}
    issue_raw = raw[issue_col].astype(str).str.replace(" ", "", regex=False)
    issue_h = issue_raw.map(issue_map)
    if issue_h.isna().any():
        raise ValueError("attachment 3 has unrecognized issue times")
    values = raw[fc_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any() or (values < -1e-9).any():
        raise ValueError("attachment 3 has missing or negative forecasts")

    unique_dates = pd.DatetimeIndex(pd.to_datetime(dates.unique()))
    n_days = len(unique_dates)
    hourly = np.full((n_days, len(ISSUE_HOURS), 24), np.nan)
    date_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(unique_dates)}
    hour_to_idx = {h: i for i, h in enumerate(ISSUE_HOURS)}
    for row_i, (d, h) in enumerate(zip(dates, issue_h)):
        di = date_to_idx[pd.Timestamp(d).normalize()]
        hi = hour_to_idx[int(h)]
        hourly[di, hi, :] = values[row_i]
    if np.isnan(hourly).any():
        raise ValueError("attachment 3 is missing some date/issue combinations")
    return {"dates": unique_dates, "hourly_kw": hourly}


def audit_data(prices: pd.DataFrame, year: dict, forecasts: dict) -> dict:
    dates = year["dates"]
    return {
        "n_price_slots": int(len(prices)),
        "n_days": int(len(dates)),
        "date_start": str(pd.Timestamp(dates.iloc[0]).date()),
        "date_end": str(dates.iloc[-1].date()),
        "forecast_days": int(forecasts["hourly_kw"].shape[0]),
        "forecast_issues": list(ISSUE_HOURS),
        "missing_load": int(np.isnan(year["load_kw"]).sum()),
        "missing_pv": int(np.isnan(year["pv_kw"]).sum()),
        "load_kw_min": float(np.min(year["load_kw"])),
        "load_kw_max": float(np.max(year["load_kw"])),
        "pv_kw_min": float(np.min(year["pv_kw"])),
        "pv_kw_max": float(np.max(year["pv_kw"])),
        "fc_kw_min": float(np.min(forecasts["hourly_kw"])),
        "fc_kw_max": float(np.max(forecasts["hourly_kw"])),
        "dt_hours": DT_HOURS,
    }


def write_clean(prices: pd.DataFrame, year: dict, forecasts: dict, audit: dict) -> None:
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
    np.save(CLEAN_DIR / "pv_hourly_forecast.npy", forecasts["hourly_kw"])
    (CLEAN_DIR / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
