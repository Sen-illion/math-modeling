"""Read attachment 1 and align 10-minute intervals to calendar 00:00-24:00."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from config import CLEAN_DIR, DT_HOURS, N_INTERVALS, ATTACHMENT1_XLSX

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
)


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

    starts = [parse_start_minutes(v) for v in raw[time_col]]
    if starts != EXPECTED_RAW_START_MIN:
        raise ValueError("timestamps are not attachment start labels 00:10 ... 0:00+1")

    price = rotate_typical_day(pd.to_numeric(raw[price_col], errors="coerce").to_numpy())
    load_kw = rotate_typical_day(pd.to_numeric(raw[load_col], errors="coerce").to_numpy())
    pv_kw = rotate_typical_day(pd.to_numeric(raw[pv_col], errors="coerce").to_numpy())

    frame = pd.DataFrame(
        {
            "t": range(1, N_INTERVALS + 1),
            "start_min": CALENDAR_START_MIN,
            "end_min": CALENDAR_END_MIN,
            "price": price,
            "load_kw": load_kw,
            "pv_kw": pv_kw,
        }
    )
    if frame[["price", "load_kw", "pv_kw"]].isna().any().any():
        raise ValueError("attachment 1 has missing numeric values")
    if (frame["load_kw"] <= 0).any() or (frame["price"] <= 0).any() or (frame["pv_kw"] < 0).any():
        raise ValueError("attachment 1 has invalid price, load, or PV values")

    frame["load_kwh"] = frame["load_kw"] * DT_HOURS
    frame["pv_kwh"] = frame["pv_kw"] * DT_HOURS
    frame["t_start"] = frame["start_min"].map(minutes_to_label)
    frame["t_end"] = frame["end_min"].map(lambda m: "24:00" if int(m) >= 24 * 60 else minutes_to_label(m))
    return frame


def write_clean_tables(frame: pd.DataFrame, source_path: Path) -> dict:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    export = frame[
        ["t", "t_start", "t_end", "start_min", "end_min", "price", "load_kw", "pv_kw", "load_kwh", "pv_kwh"]
    ].copy()
    csv_path = CLEAN_DIR / "intervals.csv"
    export.to_csv(csv_path, index=False, encoding="utf-8-sig")

    profile = {
        "n_intervals": int(len(frame)),
        "dt_hours": DT_HOURS,
        "source": str(source_path),
        "alignment": "start_clock_rotated_to_calendar",
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
