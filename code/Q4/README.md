# Q4

问题四使用当天 0:00 的三层扩展窗电价预测
\(\hat p_0=b+\Delta_{\mathrm{week}}+\Delta_{\mathrm{recent}}\) 进行决策，使用附件 4
真实电价结算。Q4-3 的 6:00、12:00 和 18:00 更新只改变负荷与光伏信息，不改变当天
0:00 已锁定的预测电价。

当前正式方案：

- Q4-2：`E_pv7d_netrho`，净负荷分位 \(\alpha=0.8\)，回放系数 \(\rho=0.625\)，计划日末 SOC 目标 \(S^*=2400\) kWh；全年 14,413,479.02 元。
- Q4-3：`LA`，48 h 展望，已锁定段 \(q_L=0.60\)，开放段与次日段 \(q=0.50\)，光伏插值使用最近已完成时段实测值；全年 14,001,199.26 元。
- 两类正式回放均从 2025-02-01 的 6000 kWh SOC 开始，覆盖 334 天。

关键文件：

- `price_structured.py`：因果电价预测及逐日残差更新。
- `run_q4_2.py`：Q4-2 固定分位、自适应分位和净负载回放策略。
- `run_q4_3.py`：Q4-3 日内滚动策略及附件 4 结算。
- `run_q4_align_official.py`：与当前 Q2/Q3 正式方法对齐的诊断。
- `sweep_q4_buffers.py`：缓冲参数诊断，不作为当前正式结果的默认入口。

正式数字以 `results/Q4/metrics.json`、`run_manifest.json`、`q42_summary.json` 和
`q43_metrics.json` 为准。任何诊断不得覆盖 `results/Q2/result2.xlsx`、
`results/Q3/result3.xlsx`、`results/Q4/result4-2.xlsx` 或 `result4-3.xlsx`。
