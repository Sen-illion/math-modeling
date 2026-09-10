# Q1 确定性购电

在仓库根目录运行：

```bash
pip install -r code/Q1/requirements.txt
python code/Q1/run_q1.py
```

入口会清洗附件 1、求解 LP、对照无储能/分时贪婪、校验约束，并写出 `results/Q1/result1.xlsx`。

5000 kW 为储能与微网接口的充放电功率上限：充电时微网侧取电不超过 5000 kW，电池最多增加 \(5000\times 0.9=4500\) kW。
