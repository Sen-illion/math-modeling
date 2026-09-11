# Q4

用冻结的 0:00 三层 expanding 电价 \(\hat p_0\) 重算问题 2 / 问题 3。决策用 \(\hat p_0\)，结算用附件 4 真值。6/12/18 只更新光伏/负荷，不改电价。

正式方案（附件 4 世界上已对齐并经过缓冲档嵌套门禁）：

- Q4-2：`D_pv7d_adaptive`（`resid_vol7`，\(q_{\min}=0.65\)，\(k=0.20\)），**14,411,957.36** 元。
- Q4-3：`LA`（\(\beta=1.0/0.2\)），日内电价锁当天 \(\hat p_0\)，**14,217,728.67** 元。

方法记录见 `planning/Q4_notes.md`。数字以 `results/Q4/metrics.json` 为准。

- `price_structured.py`：\(\hat p_0=b+\Delta_{\mathrm{week}}+\Delta_{\mathrm{recent}}\)。
- `run_q4_2.py`：固定裕度 `C_pv7d_q82` 或 `--margin adaptive`。
- `run_q4_3.py`：默认 `LA`；可用 `--beta-lock` / `--beta-open` 覆盖缓冲，不改 Q3 默认值。
- `run_q4_align.py`：对齐当前 Q2/Q3 正式策略，仅当更省时替换 `result4-x`。
- `sweep_q4_buffers.py`：在附件 4 世界上重选 \(q_{\min},k\) 与 LA \(\beta\)；tune 与 OOS 都更省才晋升。

不要覆盖 `results/Q2/result2.xlsx` 或 `results/Q3/result3.xlsx`。

```
python code/Q4/run_q4_align.py
python code/Q4/sweep_q4_buffers.py
python code/Q4/diagnose_oracle_price.py
```
