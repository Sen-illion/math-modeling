"""Fill official result2.xlsx from V2 day-ahead purchase and causal playback traces."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from config import (
    ABS_TOL_KWH,
    ASSET_TABLE_DIR,
    E0_FEB1_KWH,
    FOUR_HOUR_BLOCKS,
    N_INTERVALS,
    OFFICIAL_END,
    OFFICIAL_START,
    OPT_DIR,
    PAPER_TABLE1_END_MINUTES,
    RESULT2_TEMPLATE_XLSX,
    RESULT2_XLSX,
    RESULT_DIR,
    SPECIFIED_DATES,
)


def _clock(minutes: int) -> str:
    if minutes >= 24 * 60:
        return "24:00"
    hour, minute = divmod(int(minutes), 60)
    return f"{hour:02d}:{minute:02d}"


def _unpadded_clock(minutes: int) -> str:
    if minutes >= 24 * 60:
        return "24:00"
    hour, minute = divmod(int(minutes), 60)
    return f"{hour}:{minute:02d}"


def merge_emergency_periods(
    emergency_kwh: np.ndarray,
    start_min: np.ndarray,
    end_min: np.ndarray,
    abs_tol: float = ABS_TOL_KWH,
) -> list[tuple[str, float]]:
    """Merge consecutive 10-minute emergency slots into contiguous clock intervals."""
    emergency = np.asarray(emergency_kwh, dtype=float)
    periods: list[tuple[str, float]] = []
    i = 0
    n = len(emergency)
    while i < n:
        if emergency[i] <= abs_tol:
            i += 1
            continue
        j = i
        total = 0.0
        while j < n and emergency[j] > abs_tol:
            total += float(emergency[j])
            j += 1
        label = f"{_unpadded_clock(int(start_min[i]))}-{_unpadded_clock(int(end_min[j - 1]))}"
        periods.append((label, total))
        i = j
    return periods


def _block_sums(end_min: np.ndarray, values: np.ndarray) -> dict[str, float]:
    totals = {}
    for start, end, label in FOUR_HOUR_BLOCKS:
        mask = (end_min > start) & (end_min <= end)
        totals[label] = float(np.asarray(values, dtype=float)[mask].sum())
    return totals


def _official_traces(traces: list[dict]) -> list[dict]:
    start = pd.Timestamp(OFFICIAL_START)
    end = pd.Timestamp(OFFICIAL_END)
    official = []
    for row in traces:
        stamp = pd.Timestamp(row["date"])
        if start <= stamp <= end:
            official.append(row)
    return official


def export_result2(prices: pd.DataFrame, traces: list[dict], dest: Path | None = None) -> Path:
    dest = dest or RESULT2_XLSX
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(RESULT2_TEMPLATE_XLSX, dest)

    official = _official_traces(traces)
    if len(official) != 334:
        raise ValueError(f"expected 334 official days, got {len(official)}")

    end_min = prices["end_min"].to_numpy(dtype=int)
    start_min = prices["start_min"].to_numpy(dtype=int)
    if len(end_min) != N_INTERVALS:
        raise ValueError("attachment 1 does not have 144 slots")

    wb = load_workbook(dest)
    purchase_sheet = wb["计划购电量"]
    storage_sheet = wb["充放电量"]
    emergency_sheet = wb["紧急购电量"]

    n_time_cols = N_INTERVALS
    if purchase_sheet.max_column < n_time_cols + 3:
        raise ValueError("result2 purchase sheet is missing time or daily-total columns")
    if purchase_sheet.max_row - 1 != len(official):
        raise ValueError(
            f"template purchase rows {purchase_sheet.max_row - 1} != official days {len(official)}"
        )

    for i, (t_start, t_end) in enumerate(zip(prices["t_start"], prices["t_end"]), start=2):
        purchase_sheet.cell(1, i).value = f"{t_start}-{t_end}"
    purchase_sheet.cell(1, n_time_cols + 2).value = "全天购电量"
    purchase_sheet.cell(1, n_time_cols + 3).value = "全天购电费"

    for row_idx, row in enumerate(official, start=2):
        stamp = pd.Timestamp(row["date"]).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
        template_date = purchase_sheet.cell(row_idx, 1).value
        if pd.Timestamp(template_date).date() != stamp.date():
            raise ValueError(f"purchase sheet date mismatch at row {row_idx}: {template_date} vs {stamp.date()}")
        purchase = np.asarray(row["purchase_kwh"], dtype=float)
        if len(purchase) != n_time_cols:
            raise ValueError(f"{row['date']} purchase length {len(purchase)}")
        for col, value in enumerate(purchase, start=2):
            purchase_sheet.cell(row_idx, col).value = float(value)
        purchase_sheet.cell(row_idx, n_time_cols + 2).value = float(purchase.sum())
        purchase_sheet.cell(row_idx, n_time_cols + 3).value = float(row["plan_cost"])

    if storage_sheet.max_row > 1:
        storage_sheet.delete_rows(2, storage_sheet.max_row - 1)
    storage_row = 2
    prev_soc24 = None
    for day_i, row in enumerate(official):
        stamp = pd.Timestamp(row["date"]).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
        soc0 = float(row["soc0_kwh"])
        soc24 = float(row["soc24_kwh"])
        if day_i == 0 and abs(soc0 - E0_FEB1_KWH) > 1e-6:
            raise ValueError(f"Feb 1 00:00 SOC must be {E0_FEB1_KWH}, got {soc0}")
        if prev_soc24 is not None and abs(soc0 - prev_soc24) > 1e-6:
            raise ValueError(f"{row['date']} SOC0 {soc0} != previous SOC24 {prev_soc24}")
        prev_soc24 = soc24
        charge_blocks = _block_sums(end_min, row["charge_kwh"])
        discharge_blocks = _block_sums(end_min, row["discharge_kwh"])
        for block_i, (_, _, label) in enumerate(FOUR_HOUR_BLOCKS):
            storage_sheet.cell(storage_row, 1).value = stamp if block_i == 0 else None
            storage_sheet.cell(storage_row, 2).value = label
            storage_sheet.cell(storage_row, 3).value = charge_blocks[label]
            storage_sheet.cell(storage_row, 4).value = discharge_blocks[label]
            if block_i == 0:
                storage_sheet.cell(storage_row, 5).value = "0:00"
                storage_sheet.cell(storage_row, 6).value = soc0
            elif block_i == 1:
                storage_sheet.cell(storage_row, 5).value = "24:00"
                storage_sheet.cell(storage_row, 6).value = soc24
            else:
                storage_sheet.cell(storage_row, 5).value = None
                storage_sheet.cell(storage_row, 6).value = None
            storage_row += 1

    if emergency_sheet.max_row > 1:
        emergency_sheet.delete_rows(2, emergency_sheet.max_row - 1)
    emergency_row = 2
    n_emergency_periods = 0
    emergency_total = 0.0
    for row in official:
        periods = merge_emergency_periods(row["emergency_kwh"], start_min, end_min)
        if not periods:
            continue
        stamp = pd.Timestamp(row["date"]).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
        for period_i, (label, amount) in enumerate(periods):
            emergency_sheet.cell(emergency_row, 1).value = stamp if period_i == 0 else None
            emergency_sheet.cell(emergency_row, 2).value = label
            emergency_sheet.cell(emergency_row, 3).value = float(amount)
            emergency_row += 1
            n_emergency_periods += 1
            emergency_total += amount

    wb.save(dest)

    written = load_workbook(dest)
    purchase_written = written["计划购电量"]
    for row_idx, row in enumerate(official, start=2):
        slot_sum = sum(
            float(purchase_written.cell(row_idx, col).value or 0.0) for col in range(2, n_time_cols + 2)
        )
        daily_total = float(purchase_written.cell(row_idx, n_time_cols + 2).value)
        if abs(slot_sum - daily_total) > 1e-6:
            raise ValueError(f"{row['date']} purchase slot sum {slot_sum} != daily total {daily_total}")
        if abs(slot_sum - float(np.asarray(row["purchase_kwh"]).sum())) > 1e-6:
            raise ValueError(f"{row['date']} written purchase does not match trace")
    storage_written = written["充放电量"]
    if storage_written.max_row != 1 + 6 * len(official):
        raise ValueError(f"storage sheet rows {storage_written.max_row}, expected {1 + 6 * len(official)}")
    emergency_written = written["紧急购电量"]
    written_emergency = 0.0
    for r in range(2, emergency_written.max_row + 1):
        value = emergency_written.cell(r, 3).value
        if value not in (None, "⁝"):
            written_emergency += float(value)
    if abs(written_emergency - emergency_total) > 1e-6:
        raise ValueError(f"emergency sheet sum {written_emergency} != {emergency_total}")

    checks = {
        "n_official_days": len(official),
        "n_emergency_periods": n_emergency_periods,
        "emergency_kwh": emergency_total,
        "purchase_kwh": float(sum(np.asarray(r["purchase_kwh"]).sum() for r in official)),
        "plan_cost": float(sum(float(r["plan_cost"]) for r in official)),
        "feb1_soc0": float(official[0]["soc0_kwh"]),
        "last_soc24": float(official[-1]["soc24_kwh"]),
        "path": str(dest),
    }
    (RESULT_DIR / "result2_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dest


def export_specified_day_tables(prices: pd.DataFrame, traces: list[dict]) -> list[Path]:
    """Paper Table 1 / 2 / 3 numbers for the four dates named in the problem statement."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    by_date = {str(pd.Timestamp(row["date"]).date()): row for row in _official_traces(traces)}
    end_min = prices["end_min"].to_numpy(dtype=int)
    start_min = prices["start_min"].to_numpy(dtype=int)

    table1_rows = []
    table2_rows = []
    table3_rows = []
    for date in SPECIFIED_DATES:
        if date not in by_date:
            raise ValueError(f"missing specified date {date} in V2 traces")
        row = by_date[date]
        purchase = np.asarray(row["purchase_kwh"], dtype=float)
        for end in PAPER_TABLE1_END_MINUTES:
            idx = int(np.where(end_min == end)[0][0])
            table1_rows.append(
                {
                    "date": date,
                    "period": f"{_clock(end - 10)}-{_clock(end)}",
                    "purchase_kwh": float(purchase[idx]),
                }
            )
        table1_rows.append({"date": date, "period": "daily_purchase_kwh", "purchase_kwh": float(purchase.sum())})
        table1_rows.append({"date": date, "period": "daily_plan_cost_yuan", "purchase_kwh": float(row["plan_cost"])})
        table1_rows.append(
            {
                "date": date,
                "period": "daily_emergency_cost_yuan",
                "purchase_kwh": float(row["emergency_cost"]),
            }
        )
        table1_rows.append(
            {
                "date": date,
                "period": "daily_total_cost_yuan",
                "purchase_kwh": float(row["plan_cost"] + row["emergency_cost"]),
            }
        )

        charge_blocks = _block_sums(end_min, row["charge_kwh"])
        discharge_blocks = _block_sums(end_min, row["discharge_kwh"])
        for _, _, label in FOUR_HOUR_BLOCKS:
            table2_rows.append(
                {
                    "date": date,
                    "period": label,
                    "charge_kwh": charge_blocks[label],
                    "discharge_kwh": discharge_blocks[label],
                }
            )
        table2_rows.append(
            {"date": date, "period": "soc_0:00_kwh", "charge_kwh": float(row["soc0_kwh"]), "discharge_kwh": np.nan}
        )
        table2_rows.append(
            {"date": date, "period": "soc_24:00_kwh", "charge_kwh": float(row["soc24_kwh"]), "discharge_kwh": np.nan}
        )

        periods = merge_emergency_periods(row["emergency_kwh"], start_min, end_min)
        if not periods:
            table3_rows.append({"date": date, "period": "none", "emergency_kwh": 0.0})
        else:
            for label, amount in periods:
                table3_rows.append({"date": date, "period": label, "emergency_kwh": float(amount)})

    out_dir = RESULT_DIR / "specified_days"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    mapping = {
        "Q2_specified_table1_purchase.csv": pd.DataFrame(table1_rows),
        "Q2_specified_table2_storage.csv": pd.DataFrame(table2_rows),
        "Q2_specified_table3_emergency.csv": pd.DataFrame(table3_rows),
    }
    for name, frame in mapping.items():
        path = out_dir / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        frame.to_csv(ASSET_TABLE_DIR / name, index=False, encoding="utf-8-sig")
        paths.append(path)
    return paths
