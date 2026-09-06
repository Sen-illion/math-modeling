# 数学建模 skill 的三人同步说明

项目文件可以通过 GitHub 同步，但 Codex skill 的安装位置通常在每个人自己的本机环境中，不会因为克隆本项目而自动安装。

## 推荐做法

1. 三个人都从同一个源仓库安装 skill：`https://github.com/zhnnky329/MathModeling-skills`。
2. 使用同一个分支或固定提交版本，版本记录见 `skill_manifest.json`。
3. 每次开始比赛前检查 `skill_manifest.json`；版本不一致时先更新本机 skill，再进行建模或写作。
4. 如果 skill 本身需要改动，应在源仓库中提交并形成新的版本，再更新本项目的 manifest；不要只改某一个人的本地缓存。

## Codex 中的安装提示词

每位成员可以在自己的 Codex 中发送：

```text
请按照当前项目 skills/skill_manifest.json 的仓库和版本要求，安装或更新数学建模 skill；不要覆盖已有的项目文件。
```

安装完成后，重启或开启新的 Codex 会话，再在项目目录中工作。项目内的 `AGENTS.md` 会约束具体工作方式。

## 重要区别

- `AGENTS.md`、题目、数据、代码、结果、论文和素材：通过 GitHub 共享；
- `mathmodeling-skills` 的安装状态：每个人本机独立，但必须按同一 manifest 对齐；
- skill 源码修改：在 skill 源仓库完成，不在本项目里偷偷改缓存。
