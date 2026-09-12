"""Graft the other paper's Q2 controls onto our XGB + 7-day PV stack.

Keeps: start-aligned clock, XGB-Expanding load, 7-day same-slot PV, locked G,
5x emergency, Feb 1 SOC=6000. Does not overwrite result2.xlsx.

Adds, one factor at a time:
  - net-load residual quantile instead of split load/PV
  - hard day-ahead terminal SOC S*
  - rho-protected tracking of the planned SOC (rho=0 is greedy)
  - residual window W=7 versus expanding
  - January 15-31 16-grid freeze of (alpha, rho, S*), their protocol

    python code/Q2/rho_graft_q2.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from adaptive import dumps, summarise  # noqa: E402
from bank import bias_net_bank, ladder_banks  # noqa: E402
from config import (  # noqa: E402
    DT_HOURS,
    E0_FEB1_KWH,
    E0_JAN1_KWH,
    ETA_DISCHARGE,
    FROZEN_C_Q82_FULL_YEAR_COST,
    FROZEN_OFFICIAL_FULL_YEAR_COST,
    OFFICIAL_END,
    OFFICIAL_START,
    OOS_START,
    RHO_GRAFT_DIR,
    TUNE_END,
)
from model_lp import solve_day_lp, validate_plan  # noqa: E402
from run_q2 import _adaptive_inputs  # noqa: E402
from simulate import simulate_day, validate_actual  # noqa: E402

CALIB_START = "2025-01-15"
CALIB_END = "2025-01-31"
ALPHAS = (0.8, 0.825)
RHOS = (0.5, 0.625)
TERMINALS = (2400.0, 3600.0, 4800.0, 6000.0)
WINDOWS = [
    ("tune", OFFICIAL_START, TUNE_END),
    ("oos", OOS_START, OFFICIAL_END),
    ("full", OFFICIAL_START, OFFICIAL_END),
]
ANNUAL = [
    {
        "id": "R0",
        "risk": "split",
        "alpha": 0.8,
        "rho": 0.0,
        "terminal": None,
        "window": None,
        "note": "current fixed split 0.8, greedy, free terminal",
    },
    {
        "id": "R1",
        "risk": "net",
        "alpha": 0.8,
        "rho": 0.0,
        "terminal": None,
        "window": None,
        "note": "net alpha=0.8, greedy, free terminal (G4)",
    },
    {
        "id": "R2",
        "risk": "net",
        "alpha": 0.8,
        "rho": 0.0,
        "terminal": 2400.0,
        "window": None,
        "note": "net 0.8 + hard S*=2400, greedy",
    },
    {
        "id": "R3",
        "risk": "net",
        "alpha": 0.8,
        "rho": 0.625,
        "terminal": None,
        "window": None,
        "note": "net 0.8 + rho=0.625, free terminal",
    },
    {
        "id": "R4",
        "risk": "net",
        "alpha": 0.8,
        "rho": 0.625,
        "terminal": 2400.0,
        "window": None,
        "note": "net 0.8 + rho=0.625 + S*=2400",
    },
    {
        "id": "R5",
        "risk": "net",
        "alpha": 0.825,
        "rho": 0.625,
        "terminal": 2400.0,
        "window": None,
        "note": "their frozen (0.825, 0.625, 2400), expanding residual",
    },
    {
        "id": "R6",
        "risk": "net",
        "alpha": 0.825,
        "rho": 0.625,
        "terminal": 2400.0,
        "window": 7,
        "note": "their frozen controls + W=7 residual",
    },
    {
        "id": "R7",
        "risk": "net",
        "alpha": 0.8,
        "rho": 0.0,
        "terminal": None,
        "window": 7,
        "note": "net 0.8 greedy, W=7 residual only",
    },
]


def inventory_adjusted(daily: pd.DataFrame, start: str, end: str, unit: float) -> float:
    stamps = pd.to_datetime(daily["date"])
    win = daily[(stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))]
    cash = float(win["total_cost"].sum()) if len(win) else 0.0
    if win.empty:
        return cash
    delta = float(win["soc0_actual"].iloc[0] - win["soc24_actual"].iloc[-1])
    return cash + unit * delta


def run_controls(
    prices: pd.DataFrame,
    year: dict,
    bank: list[dict],
    start_date: str,
    end_date: str,
    soc0: float,
    rho: float,
    terminal: float | None,
    label: str,
    collect_traces: bool = False,
) -> dict:
    price = prices["price"].to_numpy(dtype=float)
    start = pd.Timestamp(start_date)
    stop = pd.Timestamp(end_date)
    soc = float(soc0)
    rows: list[dict] = []
    traces: list[dict] = []
    errors: list[str] = []
    t0 = time.perf_counter()
    for pred in bank:
        stamp = pd.Timestamp(pred["date"])
        if stamp < start or stamp > stop:
            continue
        day = int(pred["day"])
        load_act = year["load_kwh"][day]
        pv_act = year["pv_kwh"][day]
        plan = solve_day_lp(
            price,
            pred["load_kw"] * DT_HOURS,
            pred["pv_kw"] * DT_HOURS,
            soc,
            terminal_soc=terminal,
        )
        errors.extend(
            validate_plan(plan, price, pred["load_kw"] * DT_HOURS, pred["pv_kw"] * DT_HOURS)
        )
        actual = simulate_day(
            price,
            load_act,
            pv_act,
            plan["purchase_kwh"],
            soc,
            planned_soc=plan["soc_end_kwh"],
            rho=float(rho),
        )
        errors.extend(validate_actual(actual, plan["purchase_kwh"], load_act, pv_act))
        rows.append(
            {
                "date": pred["date"],
                "alpha": float(pred.get("alpha", np.nan)),
                "rho": float(rho),
                "terminal": np.nan if terminal is None else float(terminal),
                "soc0_actual": float(soc),
                "soc24_actual": float(actual["soc24_kwh"]),
                "soc24_plan": float(plan["soc_end_kwh"][-1]),
                "soc_min_actual": float(np.min(actual["soc_end_kwh"])),
                "soc_max_actual": float(np.max(actual["soc_end_kwh"])),
                "purchase_kwh": float(plan["purchase_kwh"].sum()),
                "plan_cost": float(actual["plan_cost"]),
                "emergency_kwh": float(actual["emergency_kwh"].sum()),
                "emergency_cost": float(actual["emergency_cost"]),
                "total_cost": float(actual["plan_cost"] + actual["emergency_cost"]),
                "curtail_kwh": float(actual["curtail_kwh"].sum()),
                "emergency_slots": int(actual["n_emergency_slots"]),
                "n_simultaneous_plan": int(plan["n_simultaneous"]),
                "max_unserved_kwh": float(actual["max_unserved_kwh"]),
                "dispatch": actual["dispatch"],
            }
        )
        if collect_traces:
            traces.append(
                {
                    "date": pred["date"],
                    "purchase_kwh": np.asarray(plan["purchase_kwh"], dtype=float).copy(),
                    "charge_kwh": np.asarray(actual["charge_kwh"], dtype=float).copy(),
                    "discharge_kwh": np.asarray(actual["discharge_kwh"], dtype=float).copy(),
                    "emergency_kwh": np.asarray(actual["emergency_kwh"], dtype=float).copy(),
                    "soc0_kwh": float(soc),
                    "soc24_kwh": float(actual["soc24_kwh"]),
                    "plan_cost": float(actual["plan_cost"]),
                    "emergency_cost": float(actual["emergency_cost"]),
                }
            )
        soc = float(actual["soc24_kwh"])
    daily = pd.DataFrame(rows)
    return {
        "daily": daily,
        "traces": traces,
        "errors": errors,
        "elapsed_s": time.perf_counter() - t0,
        "rule": {
            "label": label,
            "rho": float(rho),
            "terminal": None if terminal is None else float(terminal),
        },
    }


def bank_for(point, year, spec: dict, split_banks: dict[float, list[dict]], cache: dict) -> list[dict]:
    if spec["risk"] == "split":
        return split_banks[float(spec["alpha"])]
    key = (float(spec["alpha"]), spec["window"])
    if key not in cache:
        cache[key] = bias_net_bank(point, year, float(spec["alpha"]), window=spec["window"])
    return cache[key]


def spec_row(spec: dict, result: dict, unit: float) -> dict:
    summary = summarise(result, WINDOWS)
    full = summary["windows"]["full"]
    row = {
        "id": spec["id"],
        "risk": spec["risk"],
        "alpha": spec["alpha"],
        "rho": spec["rho"],
        "terminal": spec["terminal"],
        "window": spec["window"],
        "note": spec["note"],
        "n_validation_errors": summary["n_validation_errors"],
        "elapsed_s": summary["elapsed_s"],
        "last_soc24": float(result["daily"]["soc24_actual"].iloc[-1]) if len(result["daily"]) else None,
        "full_vs_q82": full["total_cost"] - FROZEN_C_Q82_FULL_YEAR_COST,
        "full_vs_official": full["total_cost"] - FROZEN_OFFICIAL_FULL_YEAR_COST,
    }
    for name in ("tune", "oos", "full"):
        win = summary["windows"][name]
        row[f"{name}_total_cost"] = win["total_cost"]
        row[f"{name}_plan_cost"] = win["plan_cost"]
        row[f"{name}_emergency_cost"] = win["emergency_cost"]
        row[f"{name}_emergency_kwh"] = win["emergency_kwh"]
        row[f"{name}_emergency_days"] = win["emergency_days"]
        row[f"{name}_purchase_kwh"] = win["purchase_kwh"]
        row[f"{name}_curtail_kwh"] = win["curtail_kwh"]
        row[f"{name}_inventory_adjusted"] = inventory_adjusted(
            result["daily"], win["start"], win["end"], unit
        )
    return row


def _md_table(df: pd.DataFrame, columns: list[str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            val = row[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cells.append("—")
            elif col in fmt:
                cells.append(fmt[col].format(val))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(path: Path, annual: pd.DataFrame, jan_rows: pd.DataFrame, picks: list[dict]) -> None:
    lines = [
        "# Q2 ρ / 日末库存 反向嫁接",
        "",
        "预测仍是 XGB-Expanding 负荷 + 7 日同刻光伏，时刻仍是起点对齐。",
        "本目录只做诊断，**未覆盖** `results/Q2/result2.xlsx`。",
        "",
        f"对照：固定分量 0.8 **{FROZEN_C_Q82_FULL_YEAR_COST:.2f}** 元；",
        f"现冻结自适应 **{FROZEN_OFFICIAL_FULL_YEAR_COST:.2f}** 元。",
        "对方写作包在终点口径上全年 13,673,391.84（固定协议）/ 13,582,556.77（月度模块），",
        "不能与下表直接加减。",
        "",
        "复现：`python code/Q2/rho_graft_q2.py`",
        "",
        "## 1 月 15–31 日 16 组网格（对方协议：从 6000 起反事实回放，含库存折算）",
        "",
    ]
    if len(jan_rows):
        show = jan_rows.sort_values("inventory_adjusted")
        lines.append(
            _md_table(
                show,
                ["window", "alpha", "rho", "terminal", "cash", "inventory_adjusted", "emergency_kwh", "soc24"],
                {
                    "alpha": "{:.3f}",
                    "rho": "{:.3f}",
                    "terminal": "{:.0f}",
                    "cash": "{:.2f}",
                    "inventory_adjusted": "{:.2f}",
                    "emergency_kwh": "{:.2f}",
                    "soc24": "{:.2f}",
                },
            )
        )
        lines.append("")
    for pick in picks:
        lines.append(
            f"- `{pick['id']}` 选中 α={pick['alpha']}, ρ={pick['rho']}, "
            f"S*={pick['terminal']}, W={pick['window']}；"
            f"1 月库存折算 {pick['jan_score']:.2f} 元。"
        )
    lines += [
        "",
        "## 全年消融（2/1–12/31，现金费用）",
        "",
        _md_table(
            annual,
            [
                "id",
                "risk",
                "alpha",
                "rho",
                "terminal",
                "window",
                "full_total_cost",
                "full_vs_q82",
                "full_vs_official",
                "full_emergency_kwh",
                "full_emergency_days",
                "oos_total_cost",
                "last_soc24",
                "n_validation_errors",
            ],
            {
                "alpha": "{:.3f}",
                "rho": "{:.3f}",
                "terminal": "{:.0f}",
                "full_total_cost": "{:.2f}",
                "full_vs_q82": "{:+.2f}",
                "full_vs_official": "{:+.2f}",
                "full_emergency_kwh": "{:.2f}",
                "full_emergency_days": "{:.0f}",
                "oos_total_cost": "{:.2f}",
                "last_soc24": "{:.2f}",
                "n_validation_errors": "{:.0f}",
            },
        ),
        "",
        "## 读法",
        "",
        "- R0 应对齐固定 0.8 的 13,765,167.58。R1 应对齐此前 G4 净分位 13,714,552.77。",
        "- ρ=0 且无日末约束时，调度与现贪心相同；ρ>0 时缺口只放到计划 SOC 的保护下界，其余买紧急电。",
        "- 1 月选出的配置若全年更贵，说明对方的 1 月网格在我们预测器上不能外推。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT = RHO_GRAFT_DIR
    OUT.mkdir(parents=True, exist_ok=True)
    declared = {
        "keep": [
            "start-aligned clock",
            "XGB-Expanding load",
            "7-day same-slot PV",
            "locked G",
            "5x emergency",
            "Feb 1 SOC=6000",
        ],
        "graft": ["net-load quantile", "hard terminal SOC", "rho SOC tracking", "W=7 residual window"],
        "january_grid": {
            "alpha": list(ALPHAS),
            "rho": list(RHOS),
            "terminal": list(TERMINALS),
            "score": "Jan 15-31 cash plus (SOC0-SOC24)*min(pi)/eta, start SOC=6000, no Jan 1-14 warmup",
        },
        "annual": ANNUAL,
        "note": "Declared before replay. Does not overwrite result2.xlsx.",
        "official_cost": FROZEN_OFFICIAL_FULL_YEAR_COST,
        "q82_cost": FROZEN_C_Q82_FULL_YEAR_COST,
    }
    (OUT / "declared.json").write_text(dumps(declared), encoding="utf-8")

    prices, year, point, split_banks, _residuals = _adaptive_inputs()
    unit = float(prices["price"].min()) / ETA_DISCHARGE
    net_cache: dict = {}
    split_banks = split_banks if 0.8 in split_banks else ladder_banks(point, year, (0.8,))

    jan_records = []
    for window in (None, 7):
        for alpha, rho, terminal in itertools.product(ALPHAS, RHOS, TERMINALS):
            spec = {
                "id": f"jan_a{alpha}_r{rho}_s{int(terminal)}_w{window}",
                "risk": "net",
                "alpha": alpha,
                "rho": rho,
                "terminal": terminal,
                "window": window,
            }
            print("january", spec["id"], flush=True)
            bank = bank_for(point, year, spec, split_banks, net_cache)
            result = run_controls(
                prices, year, bank, CALIB_START, CALIB_END, E0_JAN1_KWH, rho, terminal, spec["id"]
            )
            if result["errors"]:
                raise RuntimeError(f"{spec['id']} jan: {result['errors'][:5]}")
            daily = result["daily"]
            cash = float(daily["total_cost"].sum())
            adj = inventory_adjusted(daily, CALIB_START, CALIB_END, unit)
            jan_records.append(
                {
                    "window": "expanding" if window is None else "roll7",
                    "alpha": alpha,
                    "rho": rho,
                    "terminal": terminal,
                    "cash": cash,
                    "inventory_adjusted": adj,
                    "emergency_kwh": float(daily["emergency_kwh"].sum()),
                    "emergency_days": int((daily["emergency_kwh"] > 1e-6).sum()),
                    "soc24": float(daily["soc24_actual"].iloc[-1]),
                    "elapsed_s": result["elapsed_s"],
                }
            )
    jan_df = pd.DataFrame(jan_records)
    jan_df.to_csv(OUT / "january_grid.csv", index=False, encoding="utf-8-sig")

    picks = []
    annual_specs = list(ANNUAL)
    for window, tag in ((None, "R_jan_exp"), (7, "R_jan_w7")):
        subset = jan_df[jan_df["window"] == ("expanding" if window is None else "roll7")]
        best = subset.sort_values("inventory_adjusted").iloc[0]
        spec = {
            "id": tag,
            "risk": "net",
            "alpha": float(best["alpha"]),
            "rho": float(best["rho"]),
            "terminal": float(best["terminal"]),
            "window": window,
            "note": f"January 16-grid pick, W={'expanding' if window is None else 7}",
        }
        picks.append(
            {
                "id": tag,
                "alpha": spec["alpha"],
                "rho": spec["rho"],
                "terminal": spec["terminal"],
                "window": window,
                "jan_score": float(best["inventory_adjusted"]),
            }
        )
        duplicate = any(
            s["risk"] == "net"
            and abs(s["alpha"] - spec["alpha"]) < 1e-12
            and abs(s["rho"] - spec["rho"]) < 1e-12
            and s["terminal"] == spec["terminal"]
            and s["window"] == spec["window"]
            for s in annual_specs
        )
        if not duplicate:
            annual_specs.append(spec)

    (OUT / "january_picks.json").write_text(dumps(picks), encoding="utf-8")

    annual_rows = []
    for spec in annual_specs:
        print("annual", spec["id"], spec["note"], flush=True)
        bank = bank_for(point, year, spec, split_banks, net_cache)
        result = run_controls(
            prices,
            year,
            bank,
            OFFICIAL_START,
            OFFICIAL_END,
            E0_FEB1_KWH,
            float(spec["rho"]),
            spec["terminal"],
            spec["id"],
        )
        if result["errors"]:
            raise RuntimeError(f"{spec['id']}: {result['errors'][:5]}")
        result["daily"].to_csv(OUT / f"daily_{spec['id']}.csv", index=False, encoding="utf-8-sig")
        annual_rows.append(spec_row(spec, result, unit))
        print(
            f"  {spec['id']} full={annual_rows[-1]['full_total_cost']:.2f} "
            f"vs0.8={annual_rows[-1]['full_vs_q82']:+.2f} "
            f"vsD={annual_rows[-1]['full_vs_official']:+.2f}",
            flush=True,
        )

    annual_df = pd.DataFrame(annual_rows)
    annual_df.to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
    (OUT / "annual.json").write_text(dumps(annual_rows), encoding="utf-8")
    write_summary(OUT / "summary.md", annual_df, jan_df, picks)

    r0 = annual_df.loc[annual_df["id"] == "R0", "full_total_cost"].iloc[0]
    print("R0 vs frozen 0.8:", r0 - FROZEN_C_Q82_FULL_YEAR_COST, flush=True)
    print("wrote", OUT / "summary.md", flush=True)


if __name__ == "__main__":
    main()
