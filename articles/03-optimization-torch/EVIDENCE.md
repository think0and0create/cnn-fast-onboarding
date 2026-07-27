# Ch 3 · EVIDENCE — 运行 / 测试证据

> 本文件由作者 commit 前自查 + Tech Reviewer 验证使用。
> 任何 ❌ 项 → 章节未通过 Tech 审查，不得合并。

**关于 `tests/` 目录**：作者在本地维护 pytest 测试用于 TDD 验证，**不进入公开仓库**（见根目录 `.gitignore` 的 `articles/*/tests/` 规则 + `_agent_sop/05_DECISIONS.md` ADR-009）。读者 clone 下来的仓库不会有 `tests/`。本文件的测试数字是作者本地验证结果的记录，不是读者需要重复的步骤。

---

## 环境

```text
Python:     3.12.13 (via uv)
OS:         linux
章节目录:   articles/03-optimization-torch/
```

依赖锁定（`pyproject.toml` + `uv.lock`）：

```text
torch==2.13.0
pytest==8.3.4
```

运行时实际加载的 PyTorch 版本是：

```text
torch.__version__ = 2.13.0+cu130
```

本章通过阿里云 PyPI 镜像安装依赖：

```text
https://mirrors.aliyun.com/pypi/simple
```

`pyproject.toml` 声明的是 `torch==2.13.0`，但从阿里云镜像实际解析并安装的是 **CUDA-enabled wheel**（运行时版本后缀为 `+cu130`）。这是当前环境的真实状态，不是把 CPU wheel 写成 CUDA wheel。原因是 `download.pytorch.org/whl/cpu` 在中国环境拉取不通；在这个约束下，阿里云镜像实际提供的 CUDA-enabled wheel 是可用安装结果。对本章使用的 autograd 功能而言，CPU wheel 与这个 wheel 的功能完全一样；代价是磁盘占用更大：`torch` 包实际约 **4.5GB（含 CUDA 库）**，比 CPU 方案多约 3GB。后续章节若不需要 CUDA 依赖，再按 ADR-013 的状态说明跟进瘦身，不在本章临时修改安装结果。

本章没有把 `numpy` 加入依赖。PyTorch 在导入时仍会探测 NumPy，因此测试输出中的 NumPy warning 会保留在 §2.3，不把它误报成测试失败。

## §2.1 数据准备

**N/A — 本章无外部数据。** 三个 demo 都使用代码中定义的标量 / 张量和合成小数据，不下载数据集，也不依赖网络、图像文件或数据库。

- [x] 所有 demo 使用合成数据或固定张量 fixture
- [x] 无需 `download_data.py`
- [x] 无需 `data/` 目录
- [x] 无需 `splits.json`
- [x] 不把任何数据文件作为章节产物

## §2.2 训练入口

**N/A — 本章没有 ML 训练任务。** 这里运行的是最优化概念 demo：有参数更新循环和一个很小的 hinge-loss 例子，但没有真实数据集、CNN 模型或训练 pipeline。三个入口都通过 `main()` 返回结构化数值结果。

命令必须在 `articles/03-optimization-torch/` 目录下执行，并显式设置 `PYTHONPATH=.`：

```bash
PYTHONPATH=. .venv/bin/python -c "from code.ex1_first_order import main; print(main())"
PYTHONPATH=. .venv/bin/python -c "from code.ex2_second_order_kkt import main; print(main())"
PYTHONPATH=. .venv/bin/python -c "from code.ex3_optimizers_duality import main; print(main())"
```

`PYTHONPATH=.` 不是可省略的装饰。章节自己的 `code/` 包与 Python 标准库中的 `code` 模块同名；显式把章节目录放进模块搜索路径，是为了让 `from code.exN_...` 解析到本地章节包，而不是标准库 `code`。不设置它时，导入解析可能落到标准库同名模块，三个本地 demo 就不能按上述路径解析。

## §2.3 测试（核心阻塞）

测试目录只在作者本地存在。实际运行命令如下：

```text
$ PYTHONPATH=. .venv/bin/pytest tests/ -q
..............................                                           [100%]
=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:368
  <项目根目录>/articles/03-optimization-torch/.venv/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:368: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /__w/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
    cpu = _conversion_method_template(device=torch.device("cpu"))

tests/test_ex3_optimizers_duality.py:161
  <项目根目录>/articles/03-optimization-torch/tests/test_ex3_optimizers_duality.py:161: PytestUnknownMarkWarning: Unknown pytest.mark.integration - is this a typo? You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.integration

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================= 30 passed, 2 warnings in 2.24s =========================
```

上面的 GREEN 结果是本章的阻塞证据：**30 passed**。两个 warning 的全文和真实原因如下，均不影响测试通过，也按当前任务要求不修复：

1. **PyTorch NumPy 探测 warning**

   ```text
   UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /__w/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
   ```

   本章依赖没有 `numpy`；这是 torch 内部探测 NumPy 失败的 `UserWarning`，本章 demo 只使用 torch 张量，功能不受影响。

2. **pytest 自定义 marker warning**

   ```text
   PytestUnknownMarkWarning: Unknown pytest.mark.integration - is this a typo? You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
   ```

   `tests/test_ex3_optimizers_duality.py` 第 161 行使用了 `@pytest.mark.integration`，当前 `pyproject.toml` 没有登记这个自定义 marker，所以 pytest 发出 warning。对应测试仍然通过；warning 只涉及 marker 注册元信息，不影响测试结果。后续统一调整 SOP / pytest 配置时再处理，本任务不改 warning。

`PYTHONPATH=.` 同样适用于测试命令：pytest 收集测试时也需要把本地 `code/` 放在标准库同名 `code` 之前，否则测试导入的可能不是章节自己的三个模块。这就是测试命令不能简写成裸 `pytest tests/` 的由来。

覆盖关系：

| 实现文件 | 本地测试文件 | 状态 |
|---|---|---|
| `code/ex1_first_order.py` | `tests/test_ex1_first_order.py` | ✅ |
| `code/ex2_second_order_kkt.py` | `tests/test_ex2_second_order_kkt.py` | ✅ |
| `code/ex3_optimizers_duality.py` | `tests/test_ex3_optimizers_duality.py` | ✅ |

## §2.4 评估与产物

本章不训练真实 ML 模型，因此评估对象是三个 demo 返回的数值。每个 demo 的成功标准和实际结果如下。下面的代码块只展示 `print(main())` 返回的结构化 stdout；导入 torch 时可能出现的 NumPy warning 已在 §2.3 完整记录。

### ex1 — 一阶方法与 autograd

**成功标准**：梯度下降后的 `x` 接近二次函数 `x=0`，最终损失是有限且接近零的数；解析梯度与 autograd 路径由测试校验。

```text
$ PYTHONPATH=. .venv/bin/python -c "from code.ex1_first_order import main; print(main())"
{'final_x_sgd': tensor([4.0741e-10], dtype=torch.float64), 'final_loss_sgd': 1.6598062275523998e-19}
```

判定：✅ `final_x_sgd` 为 `4.0741e-10`，`final_loss_sgd` 为 `1.6598062275523998e-19`。

### ex2 — Newton 与 KKT

**成功标准**：Newton 迭代结果保持有限并向零靠近；给定候选最优点的 KKT residual 为零。

```text
$ PYTHONPATH=. .venv/bin/python -c "from code.ex2_second_order_kkt import main; print(main())"
{'newton_final_x': tensor([0.0006], dtype=torch.float64), 'kkt_residual_at_optimum': 0.0}
```

判定：✅ `newton_final_x` 为 `tensor([0.0006])`，`kkt_residual_at_optimum` 精确为 `0.0`。

### ex3 — 优化器、对偶与 SVM

**成功标准**：SGD 与 Adam 都返回有限的最终点和损失；强对偶 gap 为零；小型 hinge-loss 分类器返回有限的权重。成功标准不预设 Adam 在每个固定设置下都优于 SGD。

```text
$ PYTHONPATH=. .venv/bin/python -c "from code.ex3_optimizers_duality import main; print(main())"
{'sgd_final_x': tensor([2.8545e-05], dtype=torch.float64), 'sgd_loss': 8.14814390533794e-10, 'adam_final_x': tensor([7.6971e-05], dtype=torch.float64), 'adam_loss': 5.924475875199164e-09, 'strong_duality_gap': 0.0, 'svm_weights': tensor([1.3197, 1.3639], dtype=torch.float64)}
```

判定：✅ `strong_duality_gap=0.0`，`svm_weights` 为 `tensor([1.3197, 1.3639])`，两个优化器都得到有限结果。必须如实记录一个反直觉事实：在这组固定设置下，**SGD 的最终 loss `8.14814390533794e-10` 反而小于 Adam 的最终 loss `5.924475875199164e-09`**。因此本证据不写「Adam 总是更好」；Adam 的表现仍取决于目标、学习率、步数和其它设置。

产物说明：本章没有模型 checkpoint、外部数据文件或强制视觉产物。可核验产物就是上述三个 `main()` 的结构化数值输出和测试结果；`.pytest_cache/`、`__pycache__/` 等仅属于本地运行缓存，不是教学产物。

## §2.5 可复现性

- [x] `pyproject.toml` 锁定 `torch==2.13.0` 与 `pytest==8.3.4`
- [x] `uv.lock` 锁定解析树
- [x] 阿里云镜像已配置：`https://mirrors.aliyun.com/pypi/simple`
- [x] 记录运行时实际 wheel：`torch==2.13.0+cu130`（CUDA-enabled）
- [x] 已记录 CPU index 在中国环境拉取不通这一安装约束，以及 CUDA wheel 约 4.5GB 的磁盘代价
- [x] 每个导入 / 测试命令都显式使用 `PYTHONPATH=.`

完整命令序列（在干净的本章环境中执行；`tests/` 仅为作者本地目录）：

```bash
cd <项目根目录>/articles/03-optimization-torch
uv sync

PYTHONPATH=. .venv/bin/pytest tests/ -q
PYTHONPATH=. .venv/bin/python -c "from code.ex1_first_order import main; print(main())"
PYTHONPATH=. .venv/bin/python -c "from code.ex2_second_order_kkt import main; print(main())"
PYTHONPATH=. .venv/bin/python -c "from code.ex3_optimizers_duality import main; print(main())"
```

`uv sync` 使用本章 `pyproject.toml` 中的阿里云默认 index 和 `uv.lock`。如果要复核实际运行时版本，可执行：

```bash
PYTHONPATH=. .venv/bin/python -c "import torch; print(torch.__version__)"
# 2.13.0+cu130
```

这些 demo 使用固定的代码内张量 / 合成 fixture，不需要下载外部数据；复现重点是依赖锁定、镜像来源、`PYTHONPATH=.` 和上述命令顺序。CPU wheel 与 CUDA-enabled wheel 在本章 autograd 用法上的功能相同，但本机磁盘占用不会相同，不能把 4.5GB 的实际安装写成 CPU 安装。

## §3 教学审查项（自检，advisory）

以下是作者自检，不替代后续 Pedagogy reviewer：

- [x] 目标措辞保持「见过 / 认得」，没有把扫盲章写成精通承诺
- [x] 先解释优化、梯度、Newton、KKT、对偶和 Adam，再进入三个 demo 的实操
- [x] 每个 demo 都交代了输入、输出和对应的数值成功标准
- [x] 对 Adam 的描述没有过度承诺，明确记录了本次 SGD 最终 loss 更小的结果
- [x] §6 回顾明确说明本章与 Ch 2 的技术差异和未展开边界

状态仍为待审：本节是作者自检记录，不能把 advisory 自检写成 reviewer 已签字。

## §4 风格审查项（自检，advisory）

- [x] `article.md` 七段结构齐全，顺序为 §1–§7
- [x] `article.md` 实测 **193 行 / 9130 字符**（`wc -l -m article.md`）
- [x] 代码与测试按三个 demo 对应，测试目录保持本地私有
- [x] 文章中的 SOP / 反模式 grep 为 **0 命中**；没有把 `_agent_sop/`、ADR 或 reviewer 流程泄漏到教程正文
- [x] 本章的数学符号审查额外记录允许：`λ`、`g(x)`、`α`。它们分别用于 KKT / 不等式约束和 SVM 对偶直觉；这是本章已有教学记号的审查记录，不把其它未使用符号一并放宽
- [x] 文章没有把「自适应」偷换成「最终 loss 必然更低」，并保留 SGD < Adam 的实际数字

SOP grep 记录：在 `article.md` 上执行既有反模式 / SOP 泄漏检查，结果为 **0 命中**。本节为 Style 自检，状态仍待 Style reviewer 审查。

## §5 视觉元素（豁免说明）

本章**不强制视觉元素**，与 Ch 1 的豁免情形类似：

- 本章没有真实 ML 训练过程或 epoch 序列，不产生必须绘制的训练曲线
- 本章没有预测 vs 真值的图像任务，不产生结果可视化
- 本章没有 CNN 模型架构图；三个 demo 是标量 / 小张量优化实验

因此本章不生成训练曲线、结果可视化或架构图，也不虚构 `output/` 图片路径。该豁免只记录本章事实，不修改 `03_STYLE.md` 的全局规则。

## 状态

| 审查项 | 状态 | 证据 / 说明 |
|---|---|---|
| Tech | ✅ | 30 passed；两个 warning 已逐字记录；三个入口输出已记录；torch 与镜像已锁定 |
| Pedagogy | ⏳ | 作者自检完成，待 advisory reviewer |
| Style | ⏳ | 作者自检完成，SOP grep 0 命中，待 advisory reviewer |
| Human | ⏳ | 待人工复核 |
| Published | ❌ | 尚未发布 |

## 修订历史

- 2026-07-26：初版建立（30 tests passed + 2 warnings；三个 optimization demo；`torch==2.13.0`，运行时 `2.13.0+cu130`；阿里云 PyPI 镜像；`article.md` 193 行 / 9130 字符）。

## ADR-013 状态

本章节临时接受 **CUDA-enabled torch wheel**：阿里云镜像实际拉取的是 `2.13.0+cu130`，而 `download.pytorch.org/whl/cpu` 在中国环境拉不动；本章 autograd 功能不受影响，但 `torch` 实际占用约 **4.5GB（含 CUDA 库）**，磁盘代价比 CPU 方案多约 3GB。后续章节如果需要缩小环境体积，会在 **ADR-013** 中跟进记录和处理；本任务只在此保留状态引用，不修改 `article.md`、`code/*.py`、`tests/*.py`、`pyproject.toml` 或 `uv.lock`。

当前 `_agent_sop/05_DECISIONS.md` 尚未落入 ADR-013 正文，因此这里将其明确标为后续跟进项，而不是虚构一个已经完成的 ADR。
