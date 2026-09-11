"""Read result3.xlsx back and re-verify it against the tariff, independently of the run.

This is the stage-E gate: everything here is recomputed from the workbook cells and
attachment 1, so it cannot inherit a bug from the rolling-horizon code that wrote it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ABS_TOL_KWH,
    E_MAX_KWH,
    E_MIN_KWH,
    ETA_CHARGE,
    ETA_DISCHARGE,
    N_INTERVALS,
    OFFICIAL_END,
    OFFICIAL_START,
    RESULT_DIR,
)
from export_results import template_header_slot
from load_data import load_prices


def _sheet_matrix(ws, n_days: int) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return (dates, 144-column matrix in template order, row totals, row costs)."""
    headers = [ws.cell(1, t + 2).value for t in range(N_INTERVALS)]
    for h in headers:
        template_header_slot(str(h))
    dates, values, totals, costs = [], [], [], []
    for i in range(n_days):
        row = i + 2
        dates.append(str(ws.cell(row, 1).value))
        values.append([float(ws.cell(row, t + 2).value) for t in range(N_INTERVALS)])
        totals.append(float(ws.cell(row, N_INTERVALS + 2).value))
        costs.append(float(ws.cell(row, N_INTERVALS + 3).value))
    return dates, np.asarray(values), np.asarray(totals), np.asarray(costs)


def check(path: Path, metrics_path: Path) -> list[str]:
    errors: list[str] = []
    prices = load_prices()
    price144 = prices["price"].to_numpy(dtype=float)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    winner = metrics["official_winner"]
    expected_cost = float(metrics[winner]["total_cost"])
    expected_em_kwh = float(metrics[winner]["emergency_kwh"])
    n_days = int(metrics[winner]["n_days"])

    wb = load_workbook(path, data_only=True)
    for name in ("计划购电量", "调整购电量", "充放电量", "紧急购电量"):
        if name not in wb.sheetnames:
            errors.append(f"missing sheet {name}")
    if errors:
        return errors

    plan_ws, adj_ws = wb["计划购电量"], wb["调整购电量"]
    plan_headers = [plan_ws.cell(1, t + 2).value for t in range(N_INTERVALS)]
    adj_headers = [adj_ws.cell(1, t + 2).value for t in range(N_INTERVALS)]
    if plan_headers != adj_headers:
        errors.append("plan/adjust headers differ")
    p_dates, p_vals, p_tot, p_cost = _sheet_matrix(plan_ws, n_days)
    a_dates, a_vals, a_tot, a_cost = _sheet_matrix(adj_ws, n_days)

    if p_dates != a_dates:
        errors.append("plan/adjust date columns differ")
    if p_dates[0] != OFFICIAL_START or p_dates[-1] != OFFICIAL_END:
        errors.append(f"window is {p_dates[0]}..{p_dates[-1]}, expected {OFFICIAL_START}..{OFFICIAL_END}")
    if len(set(p_dates)) != n_days:
        errors.append("duplicate dates")
    if plan_ws.max_row - 1 > n_days and any(
        plan_ws.cell(n_days + 2, c).value not in (None, "") for c in range(1, N_INTERVALS + 4)
    ):
        errors.append("rows beyond the official window are not blank")

    if (p_vals < -ABS_TOL_KWH).any() or (a_vals < -ABS_TOL_KWH).any():
        errors.append("negative purchase in the template")
    if not np.allclose(p_vals.sum(axis=1), p_tot, atol=1e-6, rtol=0):
        errors.append("plan row total != sum of 144 columns")
    if not np.allclose(a_vals.sum(axis=1), a_tot, atol=1e-6, rtol=0):
        errors.append("adjust row total != sum of 144 columns")

    # Before 6:00 the contract cannot have been revised yet: slots 0..35 must match.
    header_slots = [template_header_slot(str(h)) for h in plan_headers]
    pre6 = [i for i, (off, slot) in enumerate(header_slots) if off == 0 and slot < 36]
    if not np.allclose(p_vals[:, pre6], a_vals[:, pre6], atol=ABS_TOL_KWH, rtol=0):
        errors.append("adjust differs from plan before 6:00")

    # Independent settlement, rebuilt from the workbook alone.
    em_ws = wb["紧急购电量"]
    em_by_date: dict[str, float] = {d: 0.0 for d in p_dates}
    cur = None
    for row in range(2, em_ws.max_row + 1):
        date_cell = em_ws.cell(row, 1).value
        if date_cell not in (None, ""):
            cur = str(date_cell)
        qty = em_ws.cell(row, 3).value
        if cur is None or qty in (None, ""):
            continue
        if cur not in em_by_date:
            errors.append(f"emergency sheet has unknown date {cur}")
            continue
        em_by_date[cur] += float(qty)
    em_total = sum(em_by_date.values())
    if abs(em_total - expected_em_kwh) > 1e-3:
        errors.append(f"emergency total {em_total} != metrics {expected_em_kwh}")

    # A template row runs 0:10 today .. 0:10 tomorrow, so a calendar day's slot 0 sits in
    # the *previous* row's tail column. Day 0's slot 0 predates the window and is absent.
    slot_of = {i: slot for i, (off, slot) in enumerate(header_slots) if off == 0}
    cols = sorted(slot_of, key=lambda c: slot_of[c])
    if [slot_of[c] for c in cols] != list(range(1, N_INTERVALS)):
        errors.append("same-day template columns are not calendar slots 1..143")
        return errors
    tail = [i for i, (off, _) in enumerate(header_slots) if off == 1]
    if len(tail) != 1:
        errors.append(f"expected exactly one next-day template column, found {len(tail)}")
        return errors
    tail = tail[0]

    def calendar(vals: np.ndarray, day: int) -> np.ndarray:
        return np.concatenate([[vals[day - 1, tail]], vals[day, cols]])

    plan_cost = np.zeros(n_days)
    total_cost = np.zeros(n_days)
    for i in range(1, n_days):
        g_plan, g_adj = calendar(p_vals, i), calendar(a_vals, i)
        dp = np.maximum(g_adj - g_plan, 0.0)
        dm = np.maximum(g_plan - g_adj, 0.0)
        plan_cost[i] = float(price144 @ g_plan)
        total_cost[i] = plan_cost[i] - 0.5 * float(price144 @ dm) + 1.5 * float(price144 @ dp)
    checked = slice(1, n_days)
    if not np.allclose(plan_cost[checked], p_cost[checked], atol=1e-4, rtol=0):
        bad = 1 + int(np.argmax(np.abs(plan_cost[checked] - p_cost[checked])))
        errors.append(f"plan-only cost mismatch, worst {p_dates[bad]}: {plan_cost[bad]} vs {p_cost[bad]}")

    total_from_sheet = float(a_cost.sum())
    if abs(total_from_sheet - expected_cost) > 1e-3:
        errors.append(f"adjust sheet cost total {total_from_sheet} != metrics {expected_cost}")
    # Per day the sheet total is the deviation settlement plus 5*pi*emergency. The emergency
    # sheet only carries merged blocks, so check the residual is non-negative and vanishes
    # exactly on days that booked no emergency energy.
    diff = a_cost[checked] - total_cost[checked]
    if (diff < -1e-4).any():
        bad = 1 + int(np.argmin(diff))
        errors.append(f"day total below the deviation settlement, worst {p_dates[bad]}: {diff[bad - 1]}")
    zero_em = [i - 1 for i, d in enumerate(p_dates) if i >= 1 and em_by_date[d] <= ABS_TOL_KWH]
    if zero_em and np.abs(diff[zero_em]).max() > 1e-4:
        errors.append("non-zero emergency cost on a day with no emergency rows")

    # Battery sheet: SOC bounds, 4-hour blocks, and cross-day continuity.
    cd_ws = wb["充放电量"]
    soc0, soc24, dates_cd = {}, {}, []
    charge, discharge = {}, {}
    cur = None
    for row in range(2, cd_ws.max_row + 1):
        date_cell = cd_ws.cell(row, 1).value
        if date_cell not in (None, ""):
            cur = str(date_cell)
            dates_cd.append(cur)
        if cur is None:
            continue
        charge[cur] = charge.get(cur, 0.0) + float(cd_ws.cell(row, 3).value or 0.0)
        discharge[cur] = discharge.get(cur, 0.0) + float(cd_ws.cell(row, 4).value or 0.0)
        stamp, val = cd_ws.cell(row, 5).value, cd_ws.cell(row, 6).value
        if stamp == "0:00":
            soc0[cur] = float(val)
        elif stamp == "24:00":
            soc24[cur] = float(val)
    if dates_cd != p_dates:
        errors.append("battery sheet dates differ from the purchase sheets")
    for d in p_dates:
        if d not in soc0 or d not in soc24:
            errors.append(f"{d} missing SOC endpoints")
            continue
        for label, v in (("soc0", soc0[d]), ("soc24", soc24[d])):
            if v < E_MIN_KWH - 1e-3 or v > E_MAX_KWH + 1e-3:
                errors.append(f"{d} {label}={v} outside [{E_MIN_KWH}, {E_MAX_KWH}]")
        got = soc0[d] + ETA_CHARGE * charge[d] - discharge[d] / ETA_DISCHARGE
        if abs(got - soc24[d]) > 1e-2:
            errors.append(f"{d} SOC balance off: {got} vs {soc24[d]}")
    for a, b in zip(p_dates, p_dates[1:]):
        if abs(soc24[a] - soc0[b]) > 1e-3:
            errors.append(f"SOC discontinuous {a}->{b}: {soc24[a]} vs {soc0[b]}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", default=str(RESULT_DIR / "result3.xlsx"))
    parser.add_argument("--metrics", default=str(RESULT_DIR / "metrics.json"))
    args = parser.parse_args()
    errors = check(Path(args.xlsx), Path(args.metrics))
    if errors:
        for e in errors:
            print(f"FAIL {e}")
        return 1
    print("stage E OK: headers, window, non-negativity, row totals, pre-6:00 lock,")
    print("  per-day contract cost, per-day deviation settlement, emergency total,")
    print("  SOC bounds / balance / cross-day continuity.")
    print(f"  {OFFICIAL_START} slot 0 is not in the template (it belongs to the 01-31 row).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
