"""Load attachment 1 cold-start curve and calendar-aligned attachment 4 prices."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

Q4_DIR = Path(__file__).resolve().parent
CODE_DIR = Q4_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.time_slots import (  # noqa: E402
    EXPECTED_RAW_START_MIN,
    parse_start_minutes,
    rotate_typical_day,
    stitch_year,
)

N_INTERVALS = 144
REPO_ROOT = Q4_DIR.parents[1]
ATTACHMENT1_XLSX = REPO_ROOT / "data_raw" / "Q3" / "attachment1.xlsx"
ATTACHMENT4_CANDIDATES = (
    REPO_ROOT / "data_raw" / "Q4" / "attachment4.xlsx",
    REPO_ROOT / "problem_files" / "附件" / "附件4.xlsx",
)


def _pick_column(columns, keywords: tuple[str, ...]) -> str:
    for col in columns:
        if any(key in str(col) for key in keywords):
            return col
    raise KeyError(f"cannot find column matching {keywords} in {list(columns)}")


def find_attachment4() -> Path:
    for path in ATTACHMENT4_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("attachment 4 not found")


def load_att1_curve(path: Path | None = None) -> np.ndarray:
    raw = pd.read_excel(path or ATTACHMENT1_XLSX)
    time_col = _pick_column(raw.columns, ("时间", "时刻"))
    price_col = _pick_column(raw.columns, ("电价",))
    starts = [parse_start_minutes(v) for v in raw[time_col]]
    if starts != EXPECTED_RAW_START_MIN:
        raise ValueError("attachment 1 timestamps are not start labels 00:10 ... 0:00+1")
    price = rotate_typical_day(pd.to_numeric(raw[price_col], errors="coerce").to_numpy(dtype=float))
    if price.shape != (N_INTERVALS,) or np.isnan(price).any() or (price <= 0).any():
        raise ValueError("attachment 1 price curve is invalid")
    return price


def load_year_prices(cold_start: np.ndarray, path: Path | None = None) -> dict:
    source = path or find_attachment4()
    raw = pd.read_excel(source)
    dates = pd.to_datetime(raw.iloc[:, 0])
    slot_cols = list(raw.columns[1:])
    if len(slot_cols) != N_INTERVALS:
        raise ValueError(f"attachment 4 expected {N_INTERVALS} time columns")
    starts = [parse_start_minutes(c) for c in slot_cols]
    if starts != EXPECTED_RAW_START_MIN:
        raise ValueError("attachment 4 time columns are not start labels")
    values = raw[slot_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if values.shape != (365, N_INTERVALS) or np.isnan(values).any():
        raise ValueError("attachment 4 is not a complete 365 x 144 price year")
    diffs = dates.diff().dropna()
    if not (diffs == pd.Timedelta(days=1)).all():
        raise ValueError("attachment 4 dates are not consecutive")
    cal = stitch_year(values, float(cold_start[0]))
    return {
        "dates": dates.reset_index(drop=True),
        "price": cal,
        "source": str(source),
        "alignment": "stitch_year_calendar_00_24",
    }
