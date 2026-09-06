# 全国大学生数学建模竞赛项目

这是三人协作的数学建模项目仓库。

## 目录说明

- `problem_files/`：赛题和附件；
- `data_raw/`：未经修改的原始数据；
- `data_clean/`：清洗后的数据；
- `code/`：`Q1` 到 `Qn` 的问题代码及公共代码，问题数量按实际赛题决定；
- `results/`：每个问题的实验结果、指标和图表；
- `paper/`：Overleaf LaTeX 论文源码；
- `paper/assets/`：论文协作素材入口；
- `prompts/`：数学建模 skill 使用的提示词；
- `references/templates/`：两个 Word 模板，仅作结构和格式参考；
- `planning/`：建模计划、决策和工作流状态；
- `support_materials/`：代码、数据说明和 AI 使用详情等支撑材料。

## 论文模板

进入 `paper/` 后，在 Overleaf 中使用 `main.tex` 编译电子版，使用 `XeLaTeX`。三个模板使用说明位于：

- `paper/README_01_快速开始.md`；
- `paper/README_02_写作与文件组织.md`；
- `paper/README_03_提交检查.md`。

## 协作规则

开始工作前先同步最新代码；每个 Qx 尽量只修改自己的目录；公共文件修改前通知队友。任何方法或模型修改必须提交并推送 GitHub 后才能交给下游使用。论文图表、参考资料和阅读笔记统一放入 `paper/assets/`。详细规则见根目录 `AGENTS.md`。
