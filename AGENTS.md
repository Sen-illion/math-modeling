# 全国大学生数学建模竞赛项目协作规范

## Codex 会话冷启动协议

- Codex 第一次在本项目工作，或新的队员/电脑首次打开项目时，必须先读取本文件和 `skills/skill_manifest.json`，然后运行 `powershell -ExecutionPolicy Bypass -File scripts/cold_start_check.ps1`。
- 冷启动检查包括：Git 仓库与远程状态、项目关键目录、数学建模 skill 是否存在、已发现版本及必需技能目录是否完整。
- 检查通过后，在本机生成 `.local/codex_environment.json`。`.local/` 不提交 GitHub，只代表当前电脑的验证状态。
- 如果 `.local/codex_environment.json` 中记录的 manifest 指纹与当前 `skills/skill_manifest.json` 一致，后续普通会话不重复安装；但 Codex 仍需在开始实质工作前拉取仓库最新状态。
- 如果 skill 缺失、版本不符、必需组件不完整或 manifest 已更新，Codex 必须主动报告需要安装/更新的仓库和版本，并在正式建模前请求用户执行或确认安装。
- 冷启动脚本只检查和报告，不自动覆盖或安装 skill；不得在检查失败时假装环境可用并继续正式建模。

## 仓库与分工

- `main` 分支为稳定版本，不直接提交。
- 任何方法、模型、假设、目标函数、约束、参数、数据口径或评价指标的修改，都必须先 `git pull --rebase`，完成修改和基本验证后立即提交并推送到 GitHub；未推送成功不得宣称“已完成”或通知下游继续工作。
- 方法类提交使用清晰的提交信息，例如 `method(Q1): revise objective and constraints`；推送后在协作群中同步提交 SHA、修改路径和影响的问题。
- Codex 开始生成代码、运行实验或写论文前，必须先 `git pull --rebase`，检查当前分支和最新提交，并以仓库中的最新方法文件为准；不得依据聊天中未提交的旧方案继续工作。
- 若在任务分支上修改方法，负责成员必须使用同一分支及最新提交 SHA；需要让全体成员或主论文分支使用时，应先通过 Pull Request 合并到 `main`，不能只推送分支后继续假定 `main` 已更新。
- 若 `git push` 失败、发生冲突或本地分支落后，必须明确报告“尚未同步”，先解决同步问题，不得覆盖队友改动。
- 遇到任何 Git、Pull Request、rebase、Overleaf Git 或文件同步冲突时，Codex 必须立即暂停，不得自行合并、选择一方、删除冲突标记或执行强制覆盖。必须向用户报告：冲突文件和位置、双方提交/版本、双方具体修改内容、可能影响的问题/结果/论文数字，以及建议的保留或整合方案；只有用户明确确认后，才能编辑冲突文件并继续同步。
- 冲突未获用户确认前，不得运行依赖冲突文件的正式实验、更新论文最终数字、合并到 `main` 或宣称同步完成；允许进行只读的 `git status`、`git diff` 和提交历史检查。
- 禁止为解决冲突使用 `git push --force`、`git reset --hard`、`git checkout -- <file>` 或其他会丢弃队友修改的命令，除非用户明确指定要丢弃的文件和版本。
- 不按子问题预先建立固定分支；问题通过 `code/Q1/`、`results/Q1/`、`paper/sections/q1.tex` 等目录区分，题目有几问就按实际创建到 `Qn`。分支按任务或功能建立，例如方法更新、代码修复或论文修订。
- 每位成员负责自己认领的 `code/Qx/`、`results/Qx/` 和 `paper/sections/qx.tex`；新增问题时沿用同一命名规则。
- `AGENTS.md`、`paper/main.tex`、`paper/common_setup.tex`、公共符号表和最终结论属于共享文件，修改前应通知队友并经过复核。
- 每次提交只完成一个清晰任务，并在提交信息中说明内容。
- 普通文字润色、临时探索和个人草稿可以不立即推送，但一旦影响方法、结果、图表、论文数字或下游写作，必须按上述同步规则提交并推送。

## 数学建模工作流

- 先解析题目，再检查数据，再筛选模型，再生成代码和实验。
- AI 可以处理机械性工作，但模型取舍、假设合理性、结果解释和最终提交由队员确认。
- 不得编造数据、模型结果、评价指标、引用或结论。
- `data_raw/` 中的原始数据不得修改；清洗结果写入 `data_clean/`。
- 每个问题的代码、结果、图表和日志放在对应的 `Qx` 目录中，不得互相覆盖；如果题目出现 Q4 或更多问题，按相同规则新增目录。
- 使用固定随机种子，保存输入、参数、指标和输出路径，确保结果可复现。

## LaTeX 与 Overleaf

- `paper/main.tex` 是电子版论文入口，使用 XeLaTeX。
- `paper/main_print.tex` 仅用于纸质版；电子版不得包含承诺书和编号专用页。
- 正文不设目录，正文不超过 30 页；附录列出支撑材料和完整可运行代码。
- 各子问题分别写入 `paper/sections/qx.tex`；题目有新增问题时，按相同规则新增文件并更新主文档引用顺序。
- 图片放入 `paper/figures/`，文件名使用英文、数字和下划线。
- 所有供论文作者使用的素材统一放在 `paper/assets/`：图表放 `paper/assets/figures/`，表格源文件放 `paper/assets/tables/`，参考论文和公开资料放 `paper/assets/references/`，阅读摘要、出处和使用建议放 `paper/assets/literature_notes/`。不要把论文素材散落在个人桌面或 Qx 代码目录。
- `paper/assets/` 是论文素材的协作入口；Codex 写作前必须检查其中是否有新增或更新文件，并优先使用已核验的素材。
- 论文正文引用的最终数字仍以 `results/Qx/` 中的验证结果为准；`paper/assets/` 中的图表和资料必须标注来源、用途和对应问题。
- 论文数字必须来自已验证的结果文件；冻结数字更新时必须说明原因并重新检查受影响内容。
- 各子问题可以在 Overleaf 实时协作，但不要同时修改同一个 `.tex` 文件；Q4、Q5 等新增问题沿用相同规则。
- 默认只修改本地同步副本；执行同步或推送前先检查 `git diff`。

## AI 工具使用

- 使用 AI 时，记录工具名称、版本、使用环节、采纳内容、人工修改和核验情况。
- 论文参考文献前保留真实的 AI 工具使用声明，并在 `support_materials/` 准备 AI 使用详情文件。
- 不得把 Overleaf 密码、Git token、API key 或其他敏感信息提交到仓库。

## 数学建模 skill 同步

- skill 的源仓库和版本记录放在 `skills/skill_manifest.json`；三位成员必须安装同一仓库、同一分支或同一提交版本。
- skill 不会因为 GitHub 同步项目文件而自动安装到每个人的 Codex 环境；每位成员首次使用前都要按 `skills/README.md` 完成本机安装。
- 不要只在个人电脑中修改 skill。若确需修改，先在 skill 源仓库形成提交，再更新 `skills/skill_manifest.json`，并通知所有成员重新安装或更新。
- Codex 读取项目规则时，以本仓库的 `AGENTS.md` 和 `skills/skill_manifest.json` 为准；使用 skill 前先检查本机安装版本是否匹配。
- 本机冷启动结果保存在 `.local/codex_environment.json`；该文件只用于避免重复检查，不得作为其他队员已经安装成功的证据。
