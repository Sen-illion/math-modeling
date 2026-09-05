# 全国大学生数学建模竞赛 LaTeX 模板（2026 提交版）

本目录基于 `cumcmthesis` 模板整理，适用于在 Overleaf 中编写全国大学生数学建模竞赛论文。

## 入口文件

- `main.tex`：电子版论文入口，使用 `withoutpreface`，不输出承诺书和编号专用页。
- `main_print.tex`：纸质版论文入口，保留承诺书和编号专用页。
- `common_setup.tex`：公共宏包、题目信息和队伍信息。
- `paper_body.tex`：摘要、正文入口、AI 工具声明、参考文献和附录。
- `sections/`：各章节内容文件。
- `cumcmthesis.cls`：原模板类文件，除非必要不要修改。

## 编译

Overleaf 的 Compiler 选择 **XeLaTeX**。电子版从 `main.tex` 编译并下载 PDF；纸质版打印时从 `main_print.tex` 编译。正文不设目录，正文不超过 30 页；附录应列出支撑材料和完整可运行代码。使用 AI 时，应在参考文献前保留真实的 AI 工具使用声明，并准备 AI 使用详情 PDF。

详细说明见 `README_01_快速开始.md`、`README_02_写作与文件组织.md` 和 `README_03_提交检查.md`。
