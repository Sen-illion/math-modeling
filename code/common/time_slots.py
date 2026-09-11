"""Shared 10-minute slot alignment: table clocks are interval starts."""

from __future__ import annotations

from datetime import datetime, time

import numpy as np

N_INTERVALS = 144
EXPECTED_RAW_START_MIN = list(range(10, 24 * 60, 10)) + [0]
CALENDAR_START_MIN = list(range(0, 24 * 60, 10))
CALENDAR_END_MIN = list(range(10, 24 * 60 + 1, 10))


def parse_start_minutes(value) -> int:
    if isinstance(value, time):
        return value.hour * 60 + value.minute
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute
    text = str(value).strip().replace("：", ":")
    compact = text.replace(" ", "")
    next_day = "+1" in compact
    compact = compact.replace("+1", "")
    if compact in {"24:00", "24:00:00"}:
        return 0
    parts = compact.split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    minutes = hour * 60 + minute
    if next_day:
        return minutes % (24 * 60)
    return minutes


def minutes_to_label(minutes: int) -> str:
    minutes = int(minutes) % (24 * 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}"


def rotate_typical_day(arr) -> np.ndarray:
    values = np.asarray(arr)
    if values.shape[-1] != N_INTERVALS:
        raise ValueError(f"expected last axis {N_INTERVALS}, got {values.shape}")
    return np.concatenate([values[..., -1:], values[..., :-1]], axis=-1)


def stitch_year(raw: np.ndarray, jan1_slot0: float | np.ndarray) -> np.ndarray:
    if raw.ndim != 2 or raw.shape[1] != N_INTERVALS:
        raise ValueError(f"expected (days, {N_INTERVALS}), got {raw.shape}")
    cal = np.empty_like(raw)
    fill = np.asarray(jan1_slot0, dtype=raw.dtype)
    cal[0, 0] = fill
    cal[0, 1:] = raw[0, :-1]
    if raw.shape[0] > 1:
        cal[1:, 0] = raw[:-1, -1]
        cal[1:, 1:] = raw[1:, :-1]
    return cal


def calendar_end_minutes() -> np.ndarray:
    return np.array(CALENDAR_END_MIN, dtype=int)


def calendar_start_minutes() -> np.ndarray:
    return np.array(CALENDAR_START_MIN, dtype=int)


def _clock_to_minutes(part: str) -> tuple[int, bool]:
    text = str(part).strip().replace("：", ":").replace(" ", "")
    next_day = "+1" in text
    text = text.replace("+1", "")
    hour_s, minute_s = (text.split(":") + ["0"])[:2]
    minutes = int(hour_s) * 60 + int(minute_s)
    return minutes, next_day


def template_header_slot(header: str) -> tuple[int, int]:
    """Map a template interval to (day_offset, calendar_slot). Slot 0 is 00:00-00:10."""
    text = str(header).strip().replace("：", ":")
    left, right = text.split("-", 1)
    start_m, start_next = _clock_to_minutes(left)
    end_m, end_next = _clock_to_minutes(right)
    if end_next and end_m == 0:
        end_m = 24 * 60
    if start_next or (end_next and start_m == 0 and end_m == 10):
        return 1, 0
    if end_m % 10 != 0 or end_m < 10:
        raise ValueError(f"cannot parse template header {header}")
    slot = end_m // 10 - 1
    if slot < 0 or slot >= N_INTERVALS:
        raise ValueError(f"slot out of range for {header}")
    return 0, slot
