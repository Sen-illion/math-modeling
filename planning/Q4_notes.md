# 问题四记录

公共假设与符号见 [`modeling_notes.md`](modeling_notes.md)。  
论文数字以 `results/Q4/` 已核验文件为准：`metrics.json`、`result4-2.xlsx`、`result4-3.xlsx`。

**当前正式方案（附件 4 世界）**

- 决策电价：当天 0:00 因果三层 expanding \(\hat p_0=b+\Delta_{\mathrm{week}}+\Delta_{\mathrm{recent}}\)，\(\rho=1.0\)（等权 7 日残差均值）。1 月 1 日的 \(b\) 只用附件 1 冷启动。
- 结算电价：附件 4 真值 \(\pi_t\)（少买 \(0.5\pi\)、多买 \(1.5\pi\)、紧急 \(5\pi\)）。
- Q4-2：`D_pv7d_adaptive`，特征 `resid_vol7`，\(q_{\min}=0.65\)，\(k=0.20\)。全年 **14,411,957.36** 元，紧急 79,132.69 kWh。\(q_L\)：0.7 / 0.8 / 0.85 用 129 / 145 / 60 天。
- Q4-3：`LA`，48 h 前瞻，\(q_L=0.65,\ q_O=0.50,\ q_E=0.50\)（替换 \(\beta\sigma\)），光伏插值起点 `measured`。6/12/18 只更新光伏/负荷，**电价锁当天 \(\hat p_0\)**。全年 **13,987,150.03** 元，紧急 46,371.78 kWh。
- 对照：Q4-2 固定裕度 `C_pv7d_q82` 14,480,739.96 元；Q4-3 `M1` 14,233,917.43 元、`N0` 15,786,744.39 元。
- 不覆盖 `results/Q2/result2.xlsx` / `results/Q3/result3.xlsx`。

正式窗 2025-02-01–12-31，334 天，2 月 1 日 SOC0 = 6000 kWh。价格泄漏 0，SOC 盒约束通过。

---

## 相对问题 2 / 问题 3 的口径

调度结构与当前 Q2 / Q3 正式策略对齐，只把**决策电价**换成 \(\hat p_0\)、**结算电价**换成附件 4。不在附件 4 世界上重选 \(\rho\)，不加日内电价修正 \(\lambda\) / \(\Delta_{\mathrm{today}}\)。

Oracle 诊断（决策若用当天真价、结算仍用附件 4）见 `results/Q4/diagnostics/oracle_price/`：Q4-2 hat0 相对 oracle 约 +0.57%。相对附件 1 世界的费用跳升主要来自附件 4 环境，不是 \(\hat p_0\) 误差。

---

## 缓冲档重选（未改正式表）

协议与问题 2 自适应相同，禁止全年 argmin：

- inner：2025-02-01–04-30，须优于**当前正式规则**在该窗的费用
- select：2025-05-01–06-30，在过关者中取最低；无人过关则不换档
- 晋升：tune（2/1–6/30）与 OOS（7/1–12/31）都更省才替换对应 `result4-x`

网格：Q4-2 只动 `resid_vol7` 的 \(q_{\min}\in\{0.60,0.65,0.70,0.75,0.80\}\times k\in\{0.10,0.20,0.30\}\)；Q4-3 只动 LA 的 \(\beta_{\mathrm{lock}}\in\{0,0.5,1.0,1.5,2.0\}\times\beta_{\mathrm{open}}\in\{0,0.2,0.5,1.0\}\)。未扫回放 \(\gamma\)。

结果（`results/Q4/diagnostics/retune_q_beta/`）：

- Q4-2 inner 过关 2/15，嵌套选定 \(0.60/0.20\)。tune 更省，OOS 更贵，全年 14,517,356.21 元，**不晋升**。
- Q4-3 inner 过关 1/20，嵌套选定 \(1.0/0.0\)。tune 与 OOS 都更贵，全年 14,226,321.88 元，**不晋升**。

---

## 对齐问题 3 分位数（已晋升 48 h）

hat0 仍全天锁定。把当前 Q3 正式缓冲接到 Q4-3：\(q_L=0.65,\ q_O=0.50,\ q_E=0.50\) 替换 \(\beta\sigma\)。48 h 前瞻会把当天 \(\hat p_0\) wrap 到次日，所以 24 h 与 48 h 都跑。诊断目录：`results/Q4/diagnostics/align_q3_quantile/`。

| 规则 | 前瞻 | tune | OOS | 全年 | 紧急 kWh | tune | OOS |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 原正式 `LA` \(\beta=1.0/0.2\) | 24 | 5,663,507.54 | 8,554,221.13 | 14,217,728.67 | 92,795.33 | — | — |
| `LA_h48_q65`（现正式） | 48 | 5,569,995.18 | 8,417,154.85 | 13,987,150.03 | 46,371.78 | 更省 | 更省 |
| `LA_h24_q65` | 24 | 5,589,777.53 | 8,435,929.45 | 14,025,706.98 | 49,313.18 | 更省 | 更省 |

两档 tune 与 OOS 都优于原 24 h+\(\beta\)。人工确认后晋升 **48 h + \(q=0.65\)**：全年约省 23.1 万，紧急约减半。未改 `result4-2.xlsx`。

---

## 复现

```
python code/Q4/run_q4_align.py
python code/Q4/sweep_q4_buffers.py
python code/Q4/run_q4_align_quantile.py
python code/Q4/run_q4_align_quantile.py --promote LA_h48_q65
```

对齐脚本仅当候选全年更省时才写 `result4-2.xlsx` / `result4-3.xlsx`。重选脚本仅当 tune 与 OOS 都更省时才替换对应正式表。分位数对齐默认只写诊断；`--promote` 才替换 `result4-3.xlsx`。
