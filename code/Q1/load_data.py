"""Read attachment 1 and build end-aligned 10-minute intervals."""

from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from config import CLEAN_DIR, DT_HOURS, N_INTERVALS, ATTACHMENT1_XLSX


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


def _pick_column(columns, keywords: tuple[str, ...]) -> str:
    for col in columns:
        name = str(col)
        if any(key in name for key in keywords):
            return col
    raise KeyError(f"cannot find column matching {keywords} in {list(columns)}")


def load_intervals(path: Path | None = None) -> pd.DataFrame:
    source = path or ATTACHMENT1_XLSX
    raw = pd.read_excel(source)
    time_col = _pick_column(raw.columns, ("时间", "时刻"))
    price_col = _pick_column(raw.columns, ("电价",))
    load_col = _pick_column(raw.columns, ("负载", "负荷"))
    pv_col = _pick_column(raw.columns, ("光伏",))

    frame = pd.DataFrame(
        {
            "t": range(1, len(raw) + 1),
            "time_raw": raw[time_col],
            "price": pd.to_numeric(raw[price_col], errors="coerce"),
            "load_kw": pd.to_numeric(raw[load_col], errors="coerce"),
            "pv_kw": pd.to_numeric(raw[pv_col], errors="coerce"),
        }
    )
    if len(frame) != N_INTERVALS:
        raise ValueError(f"expected {N_INTERVALS} intervals, got {len(frame)}")
    if frame[["price", "load_kw", "pv_kw"]].isna().any().any():
        raise ValueError("attachment 1 has missing numeric values")
    if (frame["load_kw"] <= 0).any() or (frame["price"] <= 0).any() or (frame["pv_kw"] < 0).any():
        raise ValueError("attachment 1 has invalid price, load, or PV values")

    frame["end_min"] = frame["time_raw"].map(parse_end_minutes)
    frame["start_min"] = frame["end_min"] - 10
    if frame["end_min"].tolist() != list(range(10, 24 * 60 + 1, 10)):
        raise ValueError("timestamps are not a continuous 10-minute cover of 00:10-24:00")

    frame["load_kwh"] = frame["load_kw"] * DT_HOURS
    frame["pv_kwh"] = frame["pv_kw"] * DT_HOURS
    frame["t_start"] = frame["start_min"].map(_minutes_to_label)
    frame["t_end"] = frame["end_min"].map(_minutes_to_label)
    return frame


def _minutes_to_label(minutes: int) -> str:
    if minutes >= 24 * 60:
        return "24:00"
    hour, minute = divmod(int(minutes), 60)
    return f"{hour:02d}:{minute:02d}"


def write_clean_tables(frame: pd.DataFrame, source_path: Path) -> dict:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    export = frame[
        ["t", "t_start", "t_end", "end_min", "price", "load_kw", "pv_kw", "load_kwh", "pv_kwh"]
    ].copy()
    csv_path = CLEAN_DIR / "intervals.csv"
    export.to_csv(csv_path, index=False, encoding="utf-8-sig")

    profile = {
        "n_intervals": int(len(frame)),
        "dt_hours": DT_HOURS,
        "source": str(source_path),
        "load_energy_kwh": float(frame["load_kwh"].sum()),
        "pv_energy_kwh": float(frame["pv_kwh"].sum()),
        "net_load_energy_kwh": float((frame["load_kwh"] - frame["pv_kwh"]).sum()),
        "price_min": float(frame["price"].min()),
        "price_max": float(frame["price"].max()),
        "pv_positive_intervals": int((frame["pv_kw"] > 0).sum()),
    }
    profile_path = CLEAN_DIR / "profile.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile
