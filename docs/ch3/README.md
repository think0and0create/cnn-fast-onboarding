# Ch 3 · 最优化扫盲：从梯度到 Adam

> 一句话：用三个小 demo 见过梯度、牛顿法、KKT、对偶和 Adam，见到这些名字时不再陌生。

## §1 目标

跑完本章，我们会见过一条从目标函数到参数更新的完整路径：把模型参数看成 `x`，把损失看成 `f(x)`，再用梯度决定下一步往哪里走。看到 `∇f` 时，我们能认出它表示梯度；看到 Hessian、KKT、Lagrangian 和 Adam 时，我们知道它们分别在解决什么问题。

这里的承诺是「见过」和「认得」。我们不会在这一章推完所有优化算法，也不会把每个算法的工程细节都展开。读完后，论文、训练日志或 PyTorch 代码里出现这些词，我们能先对上号，再知道该查哪一类资料。

## §2 为什么先做这一章

Ch 2 先把向量、矩阵、范数、内积和矩阵分解这些工具放到手边。Ch 3 用它们描述「怎样让一个函数变小」。这一步很关键，因为后面的 CNN 训练可以先压缩成一句话：`f(参数)=loss`，求 `∇f`，让 `f` 尽量小。

因此，优化不是一块和深度学习分开的数学背景。训练 CNN 时，权重和偏置就是待调整的参数，损失函数就是目标，反向传播给出梯度，优化器根据梯度更新参数。后面看到 `SGD`、`Adam` 或学习率时，我们至少知道它们位于这条链上的哪一环。

## §3 概念：先讲后练

先把三个 demo 背后的语言和直觉讲清楚，再看代码。这样到了实操部分，代码里的 `requires_grad`、Newton step 和 KKT residual 都有对应的概念，不需要靠猜。

### 3.1 项目概述

本章的项目不是一个外部数据集，而是一组很小、能重复运行的 PyTorch 张量实验。第一组实验使用一维二次函数 `f(x)=x²`，输入是初始点、学习率和步数，输出是更新后的 `x` 和损失。第二组实验在四次函数 `f(x)=x⁴` 上做 Newton 步，同时检查一个带约束的二次规划。第三组实验调用 `torch.optim.SGD` 和 `torch.optim.Adam`，再用同一个小型二分类数据集演示 hinge loss。

这样选的原因是，我们想把注意力放在优化过程本身，而不是数据清洗、模型结构或 GPU 速度上。项目的评价指标也很直接：一阶实验看最终点和最终损失，二阶实验看 Newton 迭代后的点与 KKT residual，对偶实验看 primal 和 dual objective 的差距，SVM 实验看训练后得到的两个权重。所有输入输出都在 `code/` 下的三个脚本中定义，测试会检查这些数值是否有限、是否收敛、是否满足边界条件。

### 3.2 优化问题建模

最简单的无约束优化可以写成「让 `f(x)` 尽量小」：`min f(x)`。`x` 可以是一个数，也可以是一个参数向量。训练模型时，`x` 的每个分量就可以对应一个权重或偏置，`f(x)` 是当前参数产生的损失。

现实问题经常有约束，所以更完整的写法是 `min f(x) subject to constraints`。等式约束可以写成 `h(x)=0`，不等式约束可以写成 `g(x)≤0`。例如，`min x² subject to x≥1` 可以把约束改写为 `g(x)=1-x≤0`。没有约束时，二次函数的最低点是 `x=0`；加上约束后，允许的区域从 `x=1` 开始，因此最优点变成边界 `x=1`。

约束改变的不只是答案，也改变了我们判断「已经到达最优」的方式。无约束问题常看梯度是否接近零；有约束问题还要确认候选点没有跑出可行域，并检查约束和目标之间的作用是否平衡。KKT 条件就是这套检查语言。

### 3.3 凸性

凸集的直觉是，集合里任意两个点连成的线段仍然完全留在集合里。凸函数的图像则像一个没有向内凹陷的碗。对凸函数来说，如果我们找到一个满足条件、且梯度为零的点，它就是全局最优点，而不是某个局部小坑。

这就是 convex 重要的原因。非凸函数可能有很多局部最低点，某个算法停下来时，我们还要问它是不是只找到了附近的答案。凸问题把这个问题简化了：在合适的条件下，局部最优和全局最优重合。`f(x)=x²` 是本章最简单的凸函数，`x=0` 是它唯一的最低点；`min x² subject to x≥1` 仍然是一个很小的凸优化例子，最优点只是被约束推到了 `x=1`。

凸集和凸函数并不保证任意实现都自动稳定，也不保证任意学习率都合适。它们提供的是更好的问题结构，让我们知道「找到的这个点」为什么有机会代表全局答案。

### 3.4 梯度与 PyTorch autograd

梯度 `∇f` 描述函数增长最快的方向。站在一个点上，沿着梯度方向走，函数上升得最快；反过来沿着 `-∇f` 走，就是让函数下降的最直接方向。对一维函数，梯度退化成普通导数；对参数向量，梯度会为每个参数给出一个分量。

PyTorch 的 autograd 会记录张量运算，随后从结果反向计算导数。需要求导的叶子张量设置 `requires_grad=True`，损失算出来后调用 `.backward()`，梯度通常会出现在张量的 `.grad` 中。一个最小心智模型如下：

```python
x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
lr = 0.1
loss = x.square().sum()
loss.backward()

with torch.no_grad():
    x -= lr * x.grad
x.grad.zero_()
```

`requires_grad` 是「记录这条计算路径」的开关，`.backward()` 是「沿路径反向算梯度」的动作。参数更新本身不需要再被记录，否则下一轮会把旧的更新也接进计算图，所以更新通常放在 `torch.no_grad()` 中。实际的优化器还会替我们处理清空梯度和更新参数，第三个 demo 会看到这一点。

### 3.5 一阶方法：梯度下降

梯度下降（gradient descent）只使用一阶信息，也就是梯度。最基本的更新式是 `x_new = x - lr * ∇f(x)`。`lr` 是 learning rate，决定每一步沿负梯度走多远。步子很小，通常更稳但需要更多步；步子过大，可能越过最低点，甚至在最低点两侧来回放大。

以 `f(x)=x²` 为例，`∇f=2x`。从 `x=2` 开始，`lr=0.1` 时，每次更新都会把点向零拉近。这个例子是全量梯度，不涉及小批量采样，但 PyTorch 里的 `SGD` 名称和更新规则是一致的。到了 CNN 训练，梯度往往来自一个 batch，因此工程上通常把同一类更新称作 stochastic gradient descent。

### 3.6 二阶方法：Hessian 与 Newton 步

梯度告诉我们坡面朝哪边，Hessian `∇²f` 则告诉我们坡面弯得有多厉害。把 Hessian 记成 `H`，Newton 方法会用曲率修正步长，形式可以写成 `x_new = x - H⁻¹∇f(x)`。一维时，它就是 `x_new = x - f'(x)/f''(x)`。

曲率信息让 Newton 方法在接近解时常有二次收敛，也就是误差可以比线性收敛更快地缩小。但多维情况下，直接求逆或解 Hessian 线性系统的代价大致是 `O(n³)`，内存也会随着参数数量迅速变重。CNN 的参数通常很多，实际训练不会把完整 Hessian 直接求出来，所以一阶方法和近似二阶方法更常见。

本章用 `f(x)=x⁴` 展示 Newton 步。它很适合说明两件事：曲率能改变更新速度，以及 Hessian 不是永远可逆。因为 `f''(x)=12x²`，当 `x=0` 时二阶导数为零，直接相除会得到无效数值。这个边界情况会在踩坑一节和测试里具体说明。

### 3.7 约束优化与 KKT

约束优化先要分清两类边界。等式约束必须严格满足，例如 `h(x)=0`；不等式约束允许落在一侧，例如 `g(x)≤0`。为了把目标和约束放在同一个表达式里，我们引入 Lagrangian：

`L(x, λ) = f(x) + λ·g(x)`

对于本章的约定 `g(x)≤0`，对应的不等式乘子 `λ` 需要满足 `λ≥0`。等式约束也能加入 Lagrangian，只是等式乘子不受同样的非负限制。这里的 `λ` 可以理解成约束的影子价格：约束越像一个真正限制答案的边界，它的乘子越可能发挥作用。

KKT 条件可以看成一个候选解的四项体检。第一项是 stationarity，要求目标和约束合在一起后，对 `x` 的导数平衡。第二项是 primal feasibility，要求 `h(x)=0` 且 `g(x)≤0`，也就是候选点确实在原问题允许的区域。第三项是 dual feasibility，要求不等式乘子符合符号限制，本章的约定下是 `λ≥0`。第四项是互补松弛（complementary slackness），要求 `λ·g(x)=0`，意思是一个不活跃的约束不应凭空产生作用，真正有作用的乘子则对应着紧约束。

对 `min x² subject to x≥1`，我们写 `g(x)=1-x`，最优点是 `x*=1`，乘子是 `λ*=2`，原问题最优值是 `p*=1`。此时 stationarity、primal feasibility、dual feasibility 和互补松弛同时成立。代码里的 `kkt_optimality_check` 把前三项转成 residual，便于测试一个候选点离条件还差多少。

### 3.8 对偶与拉格朗日

原问题叫 primal problem，最优值记成 `p*`。从 Lagrangian 出发，把 `x` 先消掉、只留下乘子 `λ`，就得到 dual problem 和 dual objective `d(λ)`。在合适的约定下，任意满足对偶可行性的 `λ` 都会给出原问题最优值的一个下界，所以弱对偶写成 `d≤p*`。

如果原问题是满足条件的凸问题，原问题和对偶问题之间可能没有间隙，这就是强对偶。此时最大化 dual objective 得到的值正好等于 `p*`。本章的最小二次规划在 `λ=2` 处达到这个情况，代码返回的 `strong_duality_gap` 就是 primal objective 与 dual objective 的绝对差。

### 3.9 桥接到 Adam

纯梯度下降只看当前这一刻的梯度。Momentum 会累积梯度历史，像给更新方向加了一点惯性，连续朝同一方向的梯度会叠加，来回摆动的方向则有机会被平滑掉。Adam 在此基础上为每个参数维护梯度的一阶和二阶历史，用它们调整每个参数自己的有效步长，所以常被称为带自适应学习率的优化器。

这解释了 Adam 为什么常比纯 GD 更快进入有效下降区，尤其是不同参数尺度差异很大、梯度噪声明显或目标地形狭长时。它不是「任何问题、任何步数都更低」的保证，学习率和其他设置仍然重要。本章的固定二次函数反而提供了一个很好的校准：运行 `ex3.main()` 后，`sgd_loss=8.15e-10`，`adam_loss=5.92e-9`。在这组 `lr=0.1`、`n_steps=50` 的设置下，SGD 的最终损失更低。Adam 的价值不能只用一个小例子的最终数字判断，应该结合收敛速度、调参成本和实际模型的梯度尺度来观察。

## §4 动手：跑三个 demo

概念有了对应位置，现在依次跑三个脚本。每个脚本的 `main()` 都返回结构化结果，测试也直接检查这些结果。代码只展示核心调用，完整实现保留在相应的 `code/` 文件中。

### 4.1 一阶：手写梯度和 autograd

第一组实验先用解析梯度跑更新，再用 autograd 对同一个二次函数求导。核心调用是：

```python
x0 = torch.tensor([2.0], dtype=torch.float64)
final_x = gradient_descent(x0, lr=0.1, n_steps=100)
auto_grad = autograd_quadratic_gradient(x0)
```

`gradient_descent` 直接使用 `2x`，方便把更新式和结果对应起来；`autograd_quadratic_gradient` 则验证 PyTorch 给出的梯度和解析答案一致。我们运行 `ex1.main()` 得到 `final_x_sgd=4.07e-10`，`final_loss_sgd=1.66e-19`。这两个数说明当前点已经非常接近二次函数的最低点，但它们是这个小函数、这个初始点、这个学习率和这组步数的结果，不是所有任务的通用门槛。

### 4.2 二阶：Newton 与 KKT residual

第二组实验把 `x=2.0` 放进 `f(x)=x⁴`，连续应用 Newton 步，并在另一个简单二次规划上检查 KKT：

```python
x = torch.tensor([2.0], dtype=torch.float64)
for _ in range(20):
    x = newton_step_quartic(x)

residual = kkt_residual_simple_qp(
    torch.tensor([1.0], dtype=torch.float64),
    torch.tensor([2.0], dtype=torch.float64),
)
```

这里的两段计算互不混淆。Newton 迭代回答「曲率信息怎样推动无约束函数下降」，KKT residual 回答「这个点是否同时满足约束优化的必要条件」。我们运行 `ex2.main()` 得到 `newton_final_x=6.0e-4`，`kkt_residual_at_optimum=0.0`。前一个结果还没有变成零，是因为四次函数在这个起点下每一步按固定比例缩小；后一个结果为零，表示代码给出的 `x=1.0`、`λ=2.0` 正好满足这个小 QP 的检查条件。

### 4.3 优化器、对偶和 SVM

第三组实验把手写更新交给 `torch.optim`，并在同一个二次目标上比较 SGD 和 Adam：

```python
sgd_final_x, sgd_loss = train_with_optimizer(
    "SGD", lr=0.1, n_steps=50
)
adam_final_x, adam_loss = train_with_optimizer(
    "Adam", lr=0.1, n_steps=50
)
```

这段循环的关键顺序是 `zero_grad()`、计算 loss、`loss.backward()`、`step()`。缺少其中一个环节，训练结果就不再代表我们想比较的更新过程。接着，`duality_gap_minimal_qp` 在 `λ=2.0` 处比较 primal 和 dual objective，`train_svm_hinge_demo` 则在四个二维样本上训练 hinge loss 分类器。

我们运行 `ex3.main()` 得到 `sgd_loss=8.15e-10`、`adam_loss=5.92e-9`、`strong_duality_gap=0.0`、`svm_weights=[1.32, 1.36]`。其中前两个数字是固定训练设置下的最终损失，`strong_duality_gap=0.0` 表示这个最小 QP 的 primal 和 dual 数值相等，`svm_weights` 是小型线性分类器训练后的两个权重。这个 demo 的重点不是把四个样本变成一个生产模型，而是把优化器、对偶视角和 SVM 术语连起来。

## §5 验证

在章节目录下，可以用下面三条命令分别调用三个 `main()`。命令显式打印返回值，因为脚本入口负责计算，结构化结果由调用者决定怎样展示。

```bash
cd <项目根目录>/articles/03-optimization-torch
PYTHONPATH=. .venv/bin/python -c "from code.ex1_first_order import main; print(main())"
PYTHONPATH=. .venv/bin/python -c "from code.ex2_second_order_kkt import main; print(main())"
PYTHONPATH=. .venv/bin/python -c "from code.ex3_optimizers_duality import main; print(main())"
```

再运行完整测试：

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -q
```

测试结果是 `30 passed`。这些测试不只检查最终损失，还检查从零点开始不会被 Newton 步变成 NaN，梯度下降在 `lr=0` 时不会偷偷移动，未知优化器名称会报错，以及弱对偶在一组 `λ` 值上保持 `d≤p*`。因此，验证数字和实现边界是一起被检查的。
## §6 回顾：本章带你扫过的内容

和 Ch 2 相比，Ch 2 主要处理向量和矩阵怎样表示、怎样运算；Ch 3 把这些工具放进一个「让目标函数变小」的过程里。现在我们见过凸集和凸函数，知道凸结构为什么让全局最优更容易讨论；见过梯度、梯度下降和 learning rate，能把一个更新式和参数变化对应起来；也见过 Hessian、Newton 步和它的计算代价。

我们还见过约束优化里的 KKT 四项条件，认得 Lagrangian `L(x, λ) = f(x) + λ·g(x)`，知道弱对偶 `d≤p*` 和凸问题上的强对偶分别在说什么。最后把动量、逐参数自适应学习率和 Adam 接到同一条训练链上，并用 PyTorch autograd 和 `torch.optim` 跑了小例子。

这一章没有展开拟牛顿 BFGS、内点法、强对偶成立时常用的 Slater 条件、conjugate gradient，也没有讨论分布式训练。它们各自都值得单独学习，但不是把 CNN 训练代码读下去所需的第一层词汇。这里先把边界画清楚，遇到这些名字时认得它们属于哪一类即可。

## §7 下篇预告

下一章进入 Ch 4，用 PyTorch 训练一个真实的图像分类模型。本章见过的梯度下降、KKT 约束直觉和 Adam 都会在训练循环里再次出现，只是这次目标函数不再是一个一维小曲线，而是由真实图像和 CNN 参数共同决定的损失。
