# January-calibrated q_lock on frozen LA

Algorithm unchanged: 48 h LA, q_open=q_evening=0.50, expanding residual, two-sided adjust, greedy playback.
Only q_lock is chosen on 2025-01-15..01-31 by cash + ν(6000−SOC_end).

## January ranking

| q_lock | jan cash / 元 | jan score / 元 | emergency kWh |
|---:|---:|---:|---:|
| **0.60** | 870,900.68 | **869,919.88** | 3,433.65 |
| 0.63 | 871,055.16 | 870,059.40 | 2,760.25 |
| 0.65 | 871,993.29 | 870,986.36 | 2,310.19 |
| 0.55 | 872,521.73 | 871,583.66 | 4,701.13 |
| 0.70 | 876,219.18 | 875,164.86 | 1,094.51 |
| 0.50 | 877,864.13 | 876,970.04 | 6,439.14 |
| 0.75 | 887,440.74 | 886,346.10 | 561.29 |

Winner: **0.60**. Freeze 0.65 is third on January.

## Feb–Dec confirmation (334 days)

| | cost / 元 | emergency kWh |
|---|---:|---:|
| January winner q_L=0.60 | 13,274,667.54 | 58,355.20 |
| In-sample freeze q_L=0.65 | 13,263,910.99 | 39,678.65 |
| delta | **+10,757** | +18,677 |

Bit-matches `results/Q3/exp/causal_h48_q060`. Official freeze unchanged.
