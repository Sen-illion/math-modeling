# Q3 滚动购电

入口：

```text
python code/Q3/run_q3.py --phase phase1
python code/Q3/run_q3.py --phase official
```

`phase1` 跑 1 月预热 + 2 月 1–14 日的 B0/M1/M2/M0/先知对照。
`official` 只跑 M0 全年正式窗口并导出 `results/Q3/result3.xlsx`。

口径见 `planning/Q3_execution_plan.md`。储能与回放与 `code/Q2/simulate.py` 相同。
