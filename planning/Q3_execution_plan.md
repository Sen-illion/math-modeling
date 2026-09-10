# Q3 滚动购电：建模—代码—执行计划

状态：口径已按阶段 1 证据修订（去掉展望、补 N0、收紧选择门）
适用问题：2026 国赛 C 题第三问
工作分支：`task/q3-rolling-lp`（不直接提交 `main`）
依赖：当前 Q2 交流侧储能核与因果回放（物理等价于 Q1 接口 5000 kW）

协作记录见 [`modeling_notes.md`](modeling_notes.md)。改口径先改本文件再改代码。

## 0. 已冻结口径

1. **任务**：每天 0:00 用附件 3 的 0:00 光伏预报和负荷周相似预报排计划购电；6:00 / 12:00 / 18:00 可用新光伏预报调整剩余合同。正式窗口 2025-02-01 至 2025-12-31。
2. **信息集**：0:00 不得使用当天任何实测。6/12/18 点可以使用当天已经发生的负荷、光伏、SOC，以及该时刻新发布的 24 小时光伏预报。
3. **电价**：附件 1，每天同一套。不用附件 4。
4. **光伏**：只用附件 3。`预报k小时` 对齐发布时间 \(T+k\) 整点功率，再线性插到 10 分钟。不用 Q2 的 XGB 或昨天光伏。
5. **负荷**：0:00 用上周同一星期几的 144 点；不足 7 天时用已有同星期日，再退化为昨天。6/12/18 点用「今日已发生均值 / 上周同时段均值」比例缩放剩余负荷。
6. **储能**：与 Q2 相同。交流侧 \(c_t,d_t\le 5000\Delta t\)，\(E_t=E_{t-1}+0.9 c_t-d_t/0.9\)。2025-01-01 0:00 为 6000 kWh。不强制 \(E_0=E_{24}\)。SOC 跨日传递回放末值。
7. **结算**：
   \[
   C_{\mathrm{grid}}=\sum_t\bigl(\pi_t G^{\mathrm{plan}}_t-0.5\pi_t\Delta^-_t+1.5\pi_t\Delta^+_t\bigr),\quad
   \Delta^+_t=\max(G^{\mathrm{adj}}_t-G^{\mathrm{plan}}_t,0),\ 
   \Delta^-_t=\max(G^{\mathrm{plan}}_t-G^{\mathrm{adj}}_t,0)
   \]
   回放缺口紧急购电 \(5\pi_t G^{\mathrm{em}}_t\)。总费用 = 上式 + 紧急费。
8. **回放**：与 Q2 相同。锁死当天最终合同 \(G^{\mathrm{adj}}\)，单时段因果充放，紧急电不充电。
9. **正式候选**（阶段 1 已证明 24 h 展望会抬高总费用，正式模型不再用展望、不再软跟踪 \(E_{24}\)）：
   - **N0**：0:00 用余量排计划，日内**不改合同**（\(G^{\mathrm{adj}}=G^{\mathrm{plan}}\)）。这是回答「是否需要其他时刻预报」的对照。
   - **M1**：余量 + 6/12/18 每次都调剩余合同。
   - **M0**：余量 + 选择性更新。仅当剩余时段光伏预报相对 0:00 的 L1 电量 \(\ge 3000\) kWh，且当天剩余 LP 目标 \(C^{\mathrm{adj}}<(1-\varepsilon)C^{\mathrm{keep}}\) 才改，\(\varepsilon=0.03\)。比较只用当天剩余时段，不含展望。
   - \(\beta_{\mathrm{lock}}=1.0\)，\(\beta_{\mathrm{open}}=0.2\)；\(\sigma\) 只用过去日。
   - **M2**（只作阶段 1 消融）：余量 + 展望 + 每次都调，不作正式结果。
   - `result3.xlsx` 冻结为 N0/M1/M0 中全年总费用最低者。
10. **填表**：`result3.xlsx` 按列序与附件 2 的 144 列对齐。计划表「全天购电费」= \(\sum\pi G^{\mathrm{plan}}\)；调整表「全天购电费」= 偏差结算 + 紧急费。充放电与紧急购电按 334 天展开。

## 1. 目录与产物

```text
data_raw/Q3/                 附件1、附件2、附件3、result3 模板
data_clean/Q3/               prices.csv, load_kw.csv, pv_kw.csv, pv_forecast.parquet, audit.json
code/Q3/
  config.py
  load_data.py
  pv_forecast.py
  load_forecast.py
  model_lp.py
  rolling.py
  simulate.py
  baselines.py
  export_results.py
  validate.py
  run_q3.py
results/Q3/
  phase1/
  run_manifest.json
  metrics.json
  result3.xlsx
  table1_specified_days.csv
  table2_specified_days.csv
  table3_emergency.csv
```

数字只以 `results/Q3/` 为准。效率公式不得与 Q2 分叉。

## 2. 一次滚动 LP

决策时刻 \(\tau\in\{0,6,12,18\}\) 点。正式模型只优化**当天剩余**时段；M2 消融才把预报窗延到次日。

光伏：

\[
\hat P^{\mathrm{pv},-}_{t}=\max\bigl(0,\;\hat P^{\mathrm{pv}}_{t}-\beta_t\sigma_{h(t),\ell(t)}\bigr)
\]

\(t\) 在下一次更新之前用 \(\beta_{\mathrm{lock}}\)，否则用 \(\beta_{\mathrm{open}}\)。

交流侧平衡（与 Q2 相同，增加虚拟紧急购电）：

\[
G_t+\hat P^{\mathrm{pv},-}_t\Delta t+d_t+\hat G^{\mathrm{em}}_t
=\hat P^{\mathrm{load}}_t\Delta t+c_t+w_t
\]

\[
E_t=E_{t-1}+0.9 c_t-d_t/0.9,\quad
1200\le E_t\le 10800,\quad
0\le c_t,d_t\le 5000\Delta t
\]

\(\tau=0:00\) 时 \(G=G^{\mathrm{plan}}\)。之后 \(G=G^{\mathrm{plan}}+\Delta^+-\Delta^-\)。

## 3. 对照

阶段 1（1/1–2/14，评价 2/1–2/14）必须跑：

- N0：余量、0:00 计划锁死全天
- B0：点预报、每次都调、无余量
- M1：余量、每次都调
- M2：余量 + 24 h 展望（消融）
- M0：余量 + 选择门
- 先知：当天实测日前 LP

全年跑 N0/M1/M0，`result3.xlsx` 取总费用最低者。

## 4. 执行顺序

### 阶段 A｜方法入库

写本文件并更新 `modeling_notes.md` 第 4 节。提交 `method(Q3): freeze rolling LP with buffer and selective update`。

### 阶段 B｜数据

复制附件到 `data_raw/Q3/`。清洗为 365×144 负荷/光伏、144 点电价、4×365×24 光伏预报。门禁：无缺失、无负值、日期连续、预报四时刻齐全。

### 阶段 C｜代码与阶段 1

`python code/Q3/run_q3.py --phase phase1`

1 月预热 + 2 月前 14 天正式。检查：无未来函数泄漏、合同只改剩余时段、费用可加总、SOC 跨日连续。

### 阶段 D｜全年

`python code/Q3/run_q3.py --phase official`

### 阶段 E｜门禁

容差相对 \(10^{-4}\) 或绝对 \(10^{-3}\) kWh：

1. 正式结果恰好 334 天
2. 计划/调整各 144 个非负购电，列序与附件 2 一致
3. 回放电量平衡、SOC 窗、功率上限
4. 1 月 1 日 0:00 储电量 = 6000，相邻日 24:00 与次日 0:00 相等
5. 调整合同在 0:00–6:00 若 6:00 才首次可调，则与计划一致（除非 0:00 之后没有更早更新）
6. 当日调整表购电费 = 偏差结算 + 紧急费
7. 先知总费用不高于 M0 总费用

未过门禁不得写入 `q3.tex`。

## 5. 提交切分

修订提交（相对第一版）：

1. `method(Q3): drop look-ahead and add no-adjust baseline`
2. `code(Q3): revise rolling policies N0/M1/M0`
3. `results(Q3): refresh result3 from cheapest official policy`

## 6. 完成定义

- [x] 口径已写入本文件和 `modeling_notes.md`
- [x] `python code/Q3/run_q3.py --phase official` 可复现
- [x] `results/Q3/result3.xlsx` 通过阶段 E
- [x] `metrics.json` 含 N0/B0/M1/M2/M0/先知的阶段 1 费用，以及 N0/M1/M0 全年总费用和正式赢家
- [ ] 方法、代码、结果已按切分提交（未推送不得称完成）
