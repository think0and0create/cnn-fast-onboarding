# Ch 1 · EVIDENCE — 运行 / 测试证据

> 本文件由作者 commit 前自查 + Tech Reviewer 验证使用。
> 任何 ❌ 项 → 章节未通过 Tech 审查，不得合并。

**关于 tests/ 目录**：作者在本地维护 pytest 测试用于 TDD 验证，**不进入公开仓库**（见根目录 `.gitignore` 的 `articles/*/tests/` 规则 + `_agent_sop/05_DECISIONS.md` ADR-009）。读者 clone 下来的仓库不会有 tests/。本文件的测试数字是作者本地验证结果的记录，不是读者需要重复的步骤。

---

## 环境

```text
Python:     3.12.13 (via uv)
OS:         linux
uv:         latest
章节目录:   articles/01-python-crash-course/
```

依赖锁定（`pyproject.toml` + `uv.lock`）：

```text
pytest==8.3.4
```

（本章刻意只装 pytest——扫盲不需要 numpy / pillow / opencv 等第三方库。详见 `_agent_sop/05_DECISIONS.md` ADR-009。）

## §2.1 数据准备

本章**不需要外部数据**。所有练习使用 Python 字面量或 `tmp_path` fixture（pytest 自带）。

- [x] 无需 `download_data.py`
- [x] 无需 `data/` 目录
- [x] 无需 `splits.json`

## §2.2 训练入口

本章**没有 ML 训练**。三个练习均为独立可运行脚本：

```bash
# 在 articles/01-python-crash-course/ 目录下
.venv/bin/python -m code.ex1_basics
.venv/bin/python -m code.ex2_functions_classes
.venv/bin/python -m code.ex3_file_io_cli <path> [--mode lines|json]
```

- [x] 每个入口脚本可独立运行
- [x] 默认参数跑通 ≤ 1 秒
- [x] 失败时打印清晰错误信息（如缺失文件 → stderr + exit code 2）

## §2.3 测试（核心阻塞）

```text
$ .venv/bin/python -m pytest tests/ -v
============================== 39 passed in 0.04s ==============================
```

测试位置：**本地 `tests/` 目录**（gitignored）。公开仓库无此目录。

完整测试覆盖（按文件分组）：

| 文件 | 测试数 | 覆盖 |
|---|---|---|
| `tests/test_ex1_basics.py` | 13 | `describe_number`（含 parametrize 表驱动）、`summarize_numbers`、`first_n_squares` 边界、`main` 返回值 |
| `tests/test_ex2_functions_classes.py` | 9 | `greet` keyword-only 拒绝位置参数、`variadic_sum` 含 scale、`Rectangle` 不可变 / 相等 / repr |
| `tests/test_ex3_file_io_cli.py` | 12 | `count_lines` 边界（空 / 单行 / 多行）、`read_json` Unicode / 缺失、`main` CLI 各模式 + 错误退出码 |

- [x] 每个 `.py` 有对应 `tests/test_*.py`（**本地**）
- [x] `pytest tests/ -v` 全绿（**本地**）
- [x] 无 `pytest.skip(...)` 静默跳过
- [x] 单元测试 ≤ 5 秒跑完（实测 0.04 秒）
- [x] tests/ 已加入 `.gitignore`，不会进入公开仓库

## §2.4 评估与产物

本章**无 ML 评估指标**（无训练、无 ground truth）。

每个练习的「成功标准」：

- ex1: 输出三行：`"positive odd"`、dict、`[1, 4, 9, 16, 25]`
- ex2: 输出 4 行：两个 greeting、变参和、rect 面积
- ex3: 输出文件行数（或 JSON 内容）；文件缺失 → stderr 报错 + 退出码 2

实际输出已在 §5 验证段记录。

产物：无（本章无模型、无可视化）。

## §2.5 可复现性

- [x] `pyproject.toml` 锁定依赖版本（pytest==8.3.4）
- [x] `uv.lock` 锁定完整解析树
- [x] 镜像源配置（阿里云）：`pyproject.toml` 的 `[[tool.uv.index]]`
- [x] 完整复现命令：

```bash
cd articles/01-python-crash-course
uv sync
.venv/bin/python -m pytest tests/ -v
```

- [x] 给定相同 `uv.lock`，跨平台结果一致

---

## §3 教学审查项（参考，非阻塞）

- [x] §2 动机承接：解释了为什么 Ch 1 不直接上 CNN（满足「数据灵活、深入知识在应用中熟悉」原则）
- [x] §3 项目介绍：每个练习的输入 / 输出 / 教学目标明确
- [x] §6 回顾：明确列出覆盖和未覆盖的内容
- [x] §7 下篇预告：vague（用户曾明确「NumPy 处理方式后面再说」），不剧透具体实现

## §4 风格审查项

- [x] 七段结构齐全
- [x] 字数 ~2500（1500-3000 范围内）
- [x] TDD 节奏：tests/ 与 code/ 同步，每练习先有测试意图
- [x] 无 magic number（除 pytest 自身注解）
- [x] 反模式黑名单检查：无「你应该」「加油」、无贬低其他方法、未跑通的代码

## §5 视觉元素（例外说明）

本章不适用 STYLE §5 强制要求的「训练曲线 / 结果可视化 / 架构图」三项，理由：

- 本章是 Python 扫盲，**无 ML 训练**（无 loss 曲线可画）
- 无预测任务（无可视化目标 vs 真值）
- 无模型（无架构图）

这是 STYLE §5 的合理例外。**若 Ch 2（首次 CNN）仍不补这三项，则 STYLE §5 需要 ADR 修订**。

## 状态

- [x] Tech: 通过（39 测试全绿 + 三个 CLI 入口工作 + 依赖锁定）
- [ ] Pedagogy: 待审
- [ ] Style: 待审
- [ ] Human: 待审
- [ ] Published: 未发布

---

## 修订历史

- 2026-07-25: 初版提交（39 tests, 3 exercises, stdlib-only）