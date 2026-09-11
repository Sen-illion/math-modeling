# Q3 滚动购电

入口：

```text
python code/Q3/run_q3.py --phase phase1
python code/Q3/run_q3.py --phase official
python code/Q3/check_result3.py
```

`phase1` 跑 1 月预热 + 2 月 1–14 日的 N0/B0/M1/TV/LA/M2/M0L/先知对照。
`official` 跑 N0/LA/M0L 的全年正式窗口，取全年总费用最低者导出 `results/Q3/result3.xlsx`。
`check_result3.py` 是导出端门禁：从 `result3.xlsx` 的单元格和附件 1 电价独立重算，不复用 rolling 代码。

实验改参数一律配 `--out-dir`（落在 `results/Q3/exp/`）和 `--no-export`，否则 `--pv-p0`、
`--reserve-gamma`、`--lookahead-hours`、`--q-lock` 会被拒绝，避免污染冻结结果。
`code/Q3/sweep_beta.py` 扫光伏余量 β；`code/Q3/sweep_quantile.py` 扫 48 h 视野与三段净负荷分位。

口径见 `planning/Q3_execution_plan.md`。储能与回放与 `code/Q2/simulate.py` 相同。
