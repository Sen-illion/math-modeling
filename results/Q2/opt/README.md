# Q2 V2 optimization results

Official policy selected on phase-1 (2025-02-01 to 2025-02-14): **C_pv7d_q82**.

- Load: XGB-Expanding point forecast, then causal 80% residual quantile.
- PV: 7-day same-slot mean, then causal 20% residual quantile.
- Day-ahead LP: no leftover-SOC credit (`soc_mu=0`).
- Playback: greedy (MPC was worse than greedy once the forecast was conservative).
- SOC init: **2025-02-01 00:00 actual SOC = 6000 kWh**. January is forecast history only.
- Files: `phase1_ablation.csv`, `selected_config.json`, `daily_v2.csv`, `full_year_summary.json`, `comparison_v1_v2.csv`.
- Official workbook: `results/Q2/result2.xlsx` from `python code/Q2/run_q2.py --export-result2` (334 days, matches frozen C totals).

Official window 2025-02-01 to 2025-12-31 (334 days):

- C_pv7d_q82 total cost: 13631166.17 yuan (`feb1_soc0=6000`)
- Same-stack V1 total cost: 14830254.38 yuan (`feb1_soc0=6000`)
- Repo V1 anchor (`results/Q2/full_year/`, old January SOC warmup): 15308596.59 yuan
