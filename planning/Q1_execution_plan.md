# Q1 确定性购电：建模—代码—执行计划

协作记录：共同假设见 [`modeling_notes.md`](modeling_notes.md)，本问细节见 [`Q1_notes.md`](Q1_notes.md)。

状态：已按微网接口 5000 kW 口径修订
适用问题：2026 国赛 C 题第一问
工作分支：`task/integrate-q123`（不直接提交 `main`）

## 0. 已冻结口径

开始写代码前不再讨论下列条目；若要改，先改本文件并单独提交方法。

1. **任务**：已知全天电价、负载、光伏预测，最小化计划购电费；供电不低于负载；\(E_{0}=E_{24}\)。
2. **时间**：144 个 10 分钟时段覆盖当天 0:00–24:00。附件时刻为**时段终点**（`00:10` 表示 `00:00–00:10`，`0:00+1` 表示 `23:50–24:00`）。
3. **初值**：基线 \(E_{0}=E_{24}=6000\) kWh。对照实验再放开 \(E_{0}=E_{24}\in[1200,10800]\)。
4. **不可售电**：购电 \(G_{t}\ge 0\)；光伏过剩允许弃光 \(w_{t}\ge 0\)。
5. **充放电计量**（电池侧）：\(c_{t}\) 为进入电池的电量，\(d_{t}\) 为电池放出的电量。交流侧充电取电 \(c_{t}/\eta\)，放电供电 \(\eta d_{t}\)，\(\eta=0.9\)。
6. **功率上限**：附录「最大充放电功率 5000 kW」指储能与微网接口功率。因此 \(c_{t}/\eta\le 5000\Delta t\)，\(\eta d_{t}\le 5000\Delta t\)。电池侧充电最多 \(4500\) kW，放电取电最多约 \(5555.56\) kW。
7. **基线求解器**：线性规划，PuLP + CBC。第一问不加 0-1 变量（\(\eta<1\) 时同时充放非最优）。
8. **填表**：`result1.xlsx` 按附件 1 的 144 行写入，并把「时间段」改写为物理区间（`00:00-00:10` … `23:50-24:00`）；论文表 1 的 `10:00–10:10` 取终点为 `10:10` 的时段。

## 1. 目录与产物

```text
data_raw/Q1/                 附件1 与 result1 模板的只读副本
data_clean/Q1/               intervals.csv, profile.json
code/Q1/
  config.py                  储能与时间参数
  load_data.py               读附件、对齐时段
  model_lp.py                LP 建模与求解
  baselines.py               无储能 / 分时贪婪对照
  export_results.py          填 result1 与论文表
  validate.py                约束与口径检查
  run_q1.py                  唯一入口
results/Q1/
  run_manifest.json
  metrics.json
  table1.csv / table2.csv
  result1.xlsx
  logs/solve.log
  figures/                   工作图
paper/assets/figures/        核验后的论文图
paper/assets/tables/         核验后的论文表
```

数字只以 `results/Q1/` 为准；论文素材从这里拷贝，不手改。本轮不把未核验数字写入 `paper/sections/q1.tex`。

## 2. 数学模型

对 \(t=1,\ldots,144\)，\(\Delta t=1/6\) h：

**决策** \(G_{t},c_{t},d_{t},w_{t},E_{t}\ge 0\)

**目标**
\[\min \sum_{t}\pi_{t} G_{t}\]

**电量平衡**
\[
G_{t}+P^{\mathrm{pv}}_{t}\Delta t+\eta d_{t}
=P^{\mathrm{load}}_{t}\Delta t+\frac{c_{t}}{\eta}+w_{t}
\]

**SOC**
\[
E_{t}=E_{t-1}+c_{t}-d_{t},\quad
1200\le E_{t}\le 10800,\quad
E_{0}=E_{144}=6000
\]

**功率（微网接口）**
\[
0\le \frac{c_{t}}{\eta}\le 5000\Delta t,\quad
0\le \eta d_{t}\le 5000\Delta t,\quad G_{t},w_{t}\ge 0
\]

对照模型：

| 名称 | 规则 | 用途 |
|---|---|---|
| B0 无储能 | \(G_{t}=\max(P^{\mathrm{load}}_{t}-P^{\mathrm{pv}}_{t},0)\Delta t\) | 购电费上界 |
| B1 分时贪婪 | 低价尽量充、高价尽量放，再补净负荷 | 启发式对照 |
| M0 正式 LP | 上式 | 正式结果 |

## 3. 数据与导出口径

- 附件 1 字段：时间、电价（元/kWh）、小区负载（kW）、光伏发电预测功率（kW）。
- `*_kwh = *_kw / 6`。
- 四小时充放电聚合：终点落在 `(0,4]`、`(4,8]`、…、`(20,24]` 的时段求和。
- `result1.xlsx` 的「时间段」按物理区间填写（时段终点对齐），避免官方模板相对真实时段错后 10 分钟。

## 4. 校验门禁

相对容差 \(10^{-4}\) 或绝对容差 \(10^{-3}\) kWh：

1. 每时段电量平衡残差；
2. SOC 落在 \([1200,10800]\)；
3. \(E_{0}=E_{24}=6000\)；
4. 微网接口充放电功率不超过 5000 kW（即 \(c_t/\eta\le 5000\Delta t\)，\(\eta d_t\le 5000\Delta t\)）；
5. 供电不低于负载；
6. \(G_{t}\ge 0\)，无售电；
7. 购电费 = \(\sum \pi_{t} G_{t}\)；
8. 同时充放 \(c_{t}d_{t}\) 应接近 0；
9. M0 购电费必须低于 B0。

非 Optimal 或门禁失败时，不写正式 `result1.xlsx`，不把数字写入论文。

## 5. 完成定义

- [x] `python code/Q1/run_q1.py` 可复现跑通
- [x] `results/Q1/result1.xlsx` 通过校验
- [x] `metrics.json` 含全天购电量、购电费、B0/B1/M0 对照
- [x] 论文表 1、表 2 的 CSV 与 xlsx 数字一致
- [ ] 方法、代码、结果已按切分提交并推送
