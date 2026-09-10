# Q2 V2 optimization results

Official policy selected on phase-1 (2025-02-01 to 2025-02-14): **C_pv7d_q82**.

- Load: XGB-Expanding point forecast, then causal 80% residual quantile.
- PV: 7-day same-slot mean, then causal 20% residual quantile.
- Day-ahead LP: no leftover-SOC credit (`soc_mu=0`).
- Playback: greedy (MPC was worse than greedy once the forecast was conservative).
- Files: `phase1_ablation.csv`, `selected_config.json`, `daily_v2.csv`, `full_year_summary.json`, `comparison_v1_v2.csv`.

Official window 2025-02-01 to 2025-12-31 (334 days):

- C_pv7d_q82 total cost: 13632825.10 yuan
- Same-stack V1 total cost: 14832448.48 yuan
- Repo V1 anchor (`results/Q2/full_year/`): 15308596.59 yuan
