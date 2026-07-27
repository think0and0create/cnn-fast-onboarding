# Ch 1 · Python 光速扫盲

> 一句话：1-2 小时过完 Python 最常用的语法与模式，让你后续见到这些语法时**认得出、大致懂**，不要求精通。

## §1 目标

跑完本章，你将能够：

- **认出**后续章节里出现的 Python 语法（看到 `dataclass`、`*args`、`argparse` 时知道它在干什么大致）。
- **一知半解地往下读**后续代码——遇到不认识的细节，知道去查文档或回头翻这一章。
- 知道 `uv venv` / `uv sync` / `uv add` 是什么、能跑通。

**不承诺**：

- ❌ 流畅读懂所有代码——那是「精通」不是「扫盲」。
- ❌ 写出生产级 Python 代码。
- ❌ 掌握装饰器 / 异步 / 描述符 / 元类——后续章节**用到时会回过头来加深理解**。

## §2 为什么先做这一章

本课的后续章节都会用到 Python：类型注解、`@dataclass`、`*args`、`argparse`、`pathlib`、文件 I/O。这些**不是「先学一遍」就能掌握的**——它们的细节会在用到时加深。

所以 Ch 1 只做最低限度扫盲：把这套语法见一遍，让你见到时不陌生。具体细节忘了不要紧，回头翻这一章或查文档即可。

**本章不教什么**（明确范围）：

- 工程概念：packaging、CI、跨语言绑定。
- OOP 高级：继承 / MRO / 描述符 / 元类。
- 并发：`async` / `await`、多线程。
- 第三方库：NumPy / Pillow / OpenCV / PyTorch——这些是后续章节的内容。

如果你已经熟悉 Python，可以快速翻一遍这一章（重点看类型注解和 uv 速通两节），然后直接进下一章。

## §3 概念：Python 基础与项目背景

在动手写代码之前，先把几件基础的事讲清楚。3.1-3.11 是 **Python 语言本身**——数据类型、控制流、函数、类等等；3.12-3.16 是 **项目组织**——文件 I/O、依赖、虚拟环境、uv。

### 3.1 本章的项目

本章要做的是三个**纯标准库**的短练习（§4），把 3.1-3.11 讲的语法全部用一遍：

1. **基础语法**（`ex1_basics.py`）：类型、集合、控制流、列表推导。
2. **函数与类**（`ex2_functions_classes.py`）：函数参数、`@dataclass`、不可变对象。
3. **文件 I/O 与 CLI**（`ex3_file_io_cli.py`）：`pathlib`、JSON、`argparse`、错误处理。

三个练习加起来不到 200 行代码，覆盖本节讲到的语法。

### 3.2 Python 是什么

Python 是一种**解释型**、**动态类型**、**多范式**的编程语言。三个修饰词分别意味着：

**解释型**：你写的 `.py` 文件不需要预先编译成机器码。Python 解释器（`python` 命令）读一行执行一行。这跟 C / Go / Rust 之类的编译型语言不同。

**动态类型**：变量不需要声明类型，`x = 1` 后 `x = "hello"` 完全合法。类型是在运行时才确定的。这跟 Java / C++ 之类的静态类型语言不同。

**多范式**：你可以写面向过程、面向对象、函数式、甚至 async / await 代码，Python 都支持。

实际后果：写 Python 很快——不用写类型声明、不用编译、REPL 试错也快。但相应的，**有些错误要等到运行时才暴露**（比如把字符串当数字用）。这正是 3.9 节「类型注解」要补的洞。

本课用 Python 3.12+。3.9 之前的语法（特别是泛型和 union 写法）跟现在不太一样，看到老代码不要被绊住——遇到查一下就行。

### 3.3 数据类型与变量

Python 的基本类型：

| 类型 | 例子 | 用途 |
|---|---|---|
| `int` | `42`、`-7` | 整数（任意精度，不溢出） |
| `float` | `3.14`、`1e-5` | 双精度浮点 |
| `str` | `"hello"`、`'world'` | 字符串（不可变） |
| `bool` | `True` / `False` | 布尔值（**首字母大写**） |
| `None` | `None` | 「无值」/「空」 |

容器类型：

| 类型 | 例子 | 特点 |
|---|---|---|
| `list` | `[1, 2, 3]` | **有序可变**，最常用 |
| `tuple` | `(1, 2, 3)` | **有序不可变**，可作 dict key |
| `dict` | `{"a": 1, "b": 2}` | 键值映射，**3.7+ 保证有序** |
| `set` | `{1, 2, 3}` | **无序去重**，可作集合运算 |

变量赋值直接写，不用声明类型：

```python
age = 25              # int
name = "Alice"        # str
scores = [90, 85, 92] # list
is_active = True      # bool
nothing = None        # None
```

变量名约定：小写 + 下划线（`snake_case`）。常量全大写（`MAX_SIZE = 100`），但 Python 没有真正的「常量」语法——全大写只是约定。

类型转换：构造函数同时是类型转换器：

```python
int("42")      # 42
str(42)        # "42"
list((1, 2))   # [1, 2]
bool(0)        # False（0、空容器、None 都视为 False）
```

### 3.4 运算符

算术（`+ - * / // % **`）：

```python
7 + 3       # 10
7 - 3       # 4
7 * 3       # 21
7 / 3       # 2.333...（真除法，结果是 float）
7 // 3      # 2（整除，向下取整）
7 % 3       # 1（取余）
2 ** 10     # 1024（幂）
```

比较（`== != < > <= >=`）和逻辑（`and or not`）：

```python
1 < 2 and 3 < 4      # True
not (1 == 2)          # True
"a" in ["a", "b"]    # True（成员检查）
```

`and` / `or` 是**短路求值**的：`A and B` 如果 A 为假就不算 B；`A or B` 如果 A 为真就不算 B。这经常用来简化代码：

```python
name = user_input or "anonymous"  # 如果 user_input 为空，用 "anonymous"
```

### 3.5 控制流：if / for / while

**if / elif / else**（注意是 `elif`，不是 `else if`）：

```python
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"
```

**for 循环**遍历可迭代对象（list、dict、str、range 等）：

```python
for fruit in ["apple", "banana", "cherry"]:
    print(fruit)

# 带索引
for i, fruit in enumerate(["apple", "banana"]):
    print(f"{i}: {fruit}")
# 0: apple
# 1: banana
```

`range(n)` 是 `0` 到 `n-1` 的整数序列；`range(1, n + 1)` 是 `1` 到 `n`（**闭区间**）。常见写法：

```python
for i in range(10):       # 0..9
    ...
for i in range(1, 11):    # 1..10
    ...
for i in range(0, 10, 2): # 0, 2, 4, 6, 8（步长 2）
    ...
```

**while 循环**：条件为真就一直循环。`break` 跳出，`continue` 跳到下一轮：

```python
n = 0
while n < 10:
    n += 1
    if n == 5:
        continue   # 跳过 5
    if n == 8:
        break      # 到 8 就停
```

**列表推导式**（非常常用，把循环 + 收集结果压成一行）：

```python
squares = [i * i for i in range(5)]              # [0, 1, 4, 9, 16]
evens = [x for x in range(10) if x % 2 == 0]     # [0, 2, 4, 6, 8]
```

dict / set 也有推导式：

```python
word_lengths = {w: len(w) for w in ["hi", "hello"]}  # {"hi": 2, "hello": 5}
unique_lengths = {len(w) for w in ["hi", "hello"]}    # {2, 5}
```

### 3.6 函数

**基本形态**：

```python
def add(a: int, b: int) -> int:
    return a + b
```

- `def` 关键字 + 函数名 + 参数列表 + `:` + 缩进的函数体。
- `return` 给出返回值；如果没有 `return`，函数返回 `None`。
- 类型注解（`a: int`、`-> int`）**运行时无效**，但对读者和工具有用（详见 3.10）。

**参数类型**：

```python
def f(pos, key="default", *args, kw_only, **kwargs):
    ...
```

四种参数：

- **位置参数**：`pos` 必须按顺序传。
- **关键字参数**：调用时 `f(pos=1)`。
- **默认参数**：`key="default"` 调用时可省略。
- **`*args`**：收所有多余的位置参数成元组。
- **`**kwargs`**：收所有多余的关键字参数成 dict。
- **keyword-only**：`*` 之后的参数（如 `kw_only`）必须用关键字传。

```python
def greet(name: str, *, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

greet("Alice")                    # "Hello, Alice!"
greet("Bob", greeting="Hi")       # "Hi, Bob!"
greet("Bob", "Hi")                # TypeError（greeting 是 keyword-only）
```

**作用域**：函数内赋值的变量是**局部变量**。函数外是**全局变量**。Python 的作用域规则是「LEGB」（Local / Enclosing / Global / Built-in），细节不展开——记住「函数内不写 `global` 就动不了外面的变量」就够。

### 3.7 类与对象

**基本形态**：

```python
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width          # 实例属性
        self.height = height

    def area(self) -> float:        # 实例方法
        return self.width * self.height
```

- `class` 关键字 + 类名（通常 PascalCase）。
- `__init__(self, ...)` 是构造函数，**创建实例时自动调用**。
- `self` 是当前实例（Java / C++ 里的 `this`），**必须**作为第一个参数。
- 实例属性用 `self.xxx` 在 `__init__` 里创建。

**使用**：

```python
rect = Rectangle(3.0, 4.0)   # 调用 __init__
print(rect.area())             # 12.0
print(rect.width)              # 3.0
```

**`@dataclass` 自动样板**：手写 `__init__`、`__repr__`、`__eq__` 烦人，用 `@dataclass` 装饰器自动生成：

```python
from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float

    def area(self) -> float:
        return self.width * self.height
```

`@dataclass` 等价于手写了一个把所有字段塞进 `__init__`、按字段生成 `__repr__` 和 `__eq__` 的类。后续章节（训练配置、数据集类、模型包装）会大量用。

`frozen=True` 让实例不可变：`rect.width = 5.0` 会 raise `FrozenInstanceError`。不可变对象可以放进 `set`、当 dict key、调试时少一类 bug。

**继承**（基础）：

```python
class Square(Rectangle):
    def __init__(self, side: float):
        super().__init__(side, side)   # 调用父类 __init__
```

`Square` 继承 `Rectangle`，复用父类方法。`super().__init__(...)` 调用父类构造函数。本课后续章节**不会深入继承**（OOP 高级不教），到这里够用。

### 3.8 容器操作与推导式

`list` 的常见操作（**最常用的容器**）：

```python
xs = [1, 2, 3]
xs.append(4)        # [1, 2, 3, 4]（尾部追加）
xs.extend([5, 6])   # [1, 2, 3, 4, 5, 6]（展开追加）
xs.pop()            # 弹出末尾（返回 6）
xs[0]               # 索引
xs[-1]              # 最后一个
xs[1:3]             # 切片 [2, 3]
```

`dict` 的常见操作：

```python
d = {"a": 1, "b": 2}
d["c"] = 3                  # 添加/修改
d.get("missing", 0)         # 安全读取，默认值 0
"a" in d                    # True（检查 key）
d.keys() / d.values() / d.items()   # 视图对象（可迭代）
for k, v in d.items():      # 遍历
    print(f"{k}={v}")
```

`set` 的常见操作（**去重 + 集合运算**）：

```python
a = {1, 2, 3}
b = {2, 3, 4}
a | b            # {1, 2, 3, 4}（并集）
a & b            # {2, 3}（交集）
a - b            # {1}（差集）
```

### 3.9 错误处理：try / except

错误（异常）在 Python 里是**对象**，可以用 `try / except` 捕获：

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"不能除零: {e}")
except Exception as e:
    print(f"其他错误: {e}")
else:
    print("没出错时才执行")
finally:
    print("不管出不出错都执行")
```

常见异常类型（记住这几个就够应付 90% 的情况）：

| 异常 | 触发场景 |
|---|---|
| `KeyError` | `dict["missing_key"]` |
| `IndexError` | `list[100]`（越界） |
| `TypeError` | 类型不匹配（`"a" + 1`） |
| `ValueError` | 类型对但值不对（`int("abc")`） |
| `FileNotFoundError` | `open("nope.txt")` |
| `ZeroDivisionError` | `1 / 0` |

**原则**：

- **不要**用裸 `except:` 或 `except Exception: pass` 静默吞错——bug 就藏在这里。
- **不要**过宽地捕获——捕获你**准备处理**的特定异常。
- **重新 raise**：`raise`（不带参数）把当前异常原样抛出，常用于「我先看了一下但不处理，交给上层」。

### 3.10 类型注解：声明，不是强制

Python 是动态类型，但你可以**写**类型提示——**不影响运行**：

```python
def add(x: int, y: int) -> int:
    return x + y
```

`x: int` 告诉读者和 IDE：「x 应该是整数」。`-> int` 告诉读者：「返回值是整数」。但解释器**完全不看这些注解**——你可以写 `add("hello", "world")` 也不会报错，结果会做字符串拼接。

类型注解的实际价值在于三点：

- **代码可读性**：看到 `def f(data: list[dict]) -> None` 就知道函数期望的数据形状，不用读实现。
- **IDE 智能提示**：VSCode / PyCharm 根据注解自动补全属性、检查拼写错误。
- **静态检查**：mypy / pyright 可以按注解查类型错误，**但不运行代码**。

本课统一用 Python 3.9+ 的现代写法：

```python
nums: list[int] = [1, 2, 3]            # 不要写 typing.List[int]
config: dict[str, int] = {"a": 1}      # 不要写 typing.Dict[str, int]
path: str | Path = "file.txt"          # 不要写 Union[str, Path] 或 Optional[...]
```

老代码会看到 `List[int]`、`Optional[str]`、`Union[A, B]`——那是 3.9 之前的写法。看到不要紧，自己写用新风格。

### 3.11 模块、包与导入

**模块**就是一个 `.py` 文件。**包**是一个含 `__init__.py` 的目录，里面可以有多个模块。

**导入**：

```python
import os                       # 导入整个模块，用 os.path
from os import path             # 从模块导入一个名字
from os.path import join        # 从子模块导入
import numpy as np              # 起别名
```

**模块入口惯例**：

```python
# 在 my_module.py 末尾
if __name__ == "__main__":
    main()                       # 当作脚本直接运行才执行
```

`__name__` 是 Python 的内置变量：文件被**直接运行**时是 `"__main__"`，被**import**时是模块名。`if __name__ == "__main__":` 让同一文件既能当模块被引用、又能当脚本直接跑——本课三个练习都用了这个惯例。

**包结构**：

```
my_pkg/
├── __init__.py                # 空文件或导出符号
├── utils.py                   # my_pkg.utils
└── io.py                      # my_pkg.io
```

`__init__.py` 让目录被识别为 Python 包。空文件就够用，也可以在这里写 `from .utils import *` 之类的导出。

### 3.12 文件 I/O 与路径

**`pathlib.Path` 处理路径**（替代 `os.path`）：

```python
from pathlib import Path

p = Path("data") / "raw" / "image.png"   # 跨平台路径拼接
print(p.exists())                          # 是否存在
print(p.suffix)                            # ".png"
print(p.parent)                            # 上级目录
print(p.read_text(encoding="utf-8"))       # 一行读
p.write_text("hello", encoding="utf-8")    # 一行写
```

**`open` / `with` 读写文件**：

```python
# 读取
with open("data.txt", encoding="utf-8") as f:
    text = f.read()                # 全部读完
    # 或者：
    for line in f:                 # 逐行（推荐，省内存）
        print(line.rstrip())

# 写入
with open("out.txt", "w", encoding="utf-8") as f:
    f.write("hello\n")
```

**`encoding="utf-8"` 必须显式写**——默认是平台编码（Windows 上是 cp936），不写迟早踩坑。**`with` 自动管理文件资源**，用完自动 close，不需要 try/finally。

**JSON**：

```python
import json

data = json.loads('{"a": 1}')          # str → dict
text = json.dumps(data, ensure_ascii=False)   # dict → str
config = json.loads(Path("config.json").read_text(encoding="utf-8"))
```

`ensure_ascii=False` 让中文字符不被转义成 `\uXXXX`，输出可读。

### 3.13 CLI（argparse）

需要给脚本加命令行参数时，用标准库的 `argparse`：

```python
import argparse

def main() -> int:
    parser = argparse.ArgumentParser(prog="my_tool", description="...")
    parser.add_argument("input", type=Path, help="输入文件")
    parser.add_argument("--mode", choices=["a", "b"], default="a")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    print(args.input, args.mode, args.verbose)
    return 0
```

跑 `--help` 自动生成帮助文档。`choices` 限定取值；`action="store_true"` 把 `--verbose` 变成布尔 flag。

**错误输出走 stderr**：

```python
import sys
print("error: 文件不存在", file=sys.stderr)   # 错误信息走 stderr
return 2                                        # 退出码非 0 = 失败
```

错误应该输出到 `stderr` 而不是 `stdout`——Unix 惯例：`>` 重定向 stdout 时，错误仍会显示在屏幕上。退出码 `0` = 成功，`非 0` = 失败，shell 脚本会看这个判断。

### 3.14 依赖、虚拟环境、为什么需要它们

**什么是依赖**：你自己写的代码只能做基础的事。要做更复杂的（比如读写 Excel、训练神经网络），需要用别人写的库。这些库就是**依赖**——你的项目「依赖」它们才能工作。

Python 装依赖用 `pip install xxx`。但立刻就有一个问题：**不同项目可能需要不同版本的同一个库**。比如项目 A 要 `numpy==1.26`（老 API），项目 B 要 `numpy==2.0`（新 API），你没办法同时让两个都满意。如果都装到系统 Python 里，`import numpy` 时到底用哪个版本？后装的覆盖先装的——靠不住了。

**虚拟环境**就是为这个问题而生的：每个项目拥有自己独立的 Python 解释器和库目录，**互不干扰**。`venv` 是标准库自带的虚拟环境工具，`uv venv` 是更现代的等价命令。

虚拟环境的存在感：它就是一个目录（一般是 `.venv`），里面有个独立的 Python 解释器（`bin/python` 或 `Scripts/python.exe`）和独立的 `lib/` 装库目录。你在这个环境里 `pip install`，装的是这个环境里的，不是系统的。要运行这个环境的 Python，要用 `.venv/bin/python`（不能直接 `python`）。

### 3.15 uv：现代 Python 工具链

**uv** 是 Astral 公司出的 Python 工具链，同一团队也做了代码检查工具 `ruff`。它把历史上分散的几个工具合成了一个：

| 旧工具 | uv 替代 | 用途 |
|---|---|---|
| `python -m venv` | `uv venv` | 创建虚拟环境 |
| `pip install xxx` | `uv pip install xxx` 或 `uv add xxx` | 装依赖 |
| `pip-tools`（pip-compile / pip-sync） | `uv lock` / `uv sync` | 锁定依赖版本 |
| `python xxx.py`（用 venv 解释器） | `uv run xxx.py` | 在虚拟环境里跑命令 |

**为什么本课用 uv**，不用 pip + venv：

1. **快**。uv 用 Rust 实现，比 pip + setuptools 快 10-100 倍。装 PyTorch 这种大库时差距尤其明显。
2. **统一**。一个工具干所有事，不用记 `pip` vs `pip3` vs `python -m pip` 这些历史包袱的区别。
3. **锁定文件**。`uv.lock` 精确记录每个传递依赖的版本，跨机器复现 byte-for-byte 一致。本章的 `uv.lock` 入库，读者 `uv sync` 后保证跑出一致结果。
4. **现代**。本课全程用 uv，不会再教老工具——读者学完这一章，2026 年的工具链就够了。

uv 不是 Python 标准库，需要先单独装一次：

```bash
# 推荐（Linux/macOS）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者用 pip
pip install uv
```

### 3.16 uv 项目结构

一个最小 uv 项目长这样：

```
my_project/
├── pyproject.toml    ← 项目元数据 + 依赖声明（人写）
├── uv.lock           ← 锁定版本（uv 自动生成）
├── .venv/            ← 虚拟环境（uv 自动管理，gitignored）
└── src/              ← 你的代码
```

`pyproject.toml` 是核心文件，TOML 格式。最小长这样：

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "requests==2.32.3",
]
```

- `[project]` 段是项目元数据。
- `dependencies` 列出运行时需要的库和版本。
- `requires-python` 声明最低 Python 版本。

**关键命令**：

- `uv add requests` — 把 `requests` 加进 `pyproject.toml` 并写入 `uv.lock`。
- `uv sync` — 读 `uv.lock`，创建 `.venv` 并安装所有依赖。
- `uv run python xxx.py` — 在 `.venv` 环境下运行脚本（不需要手动激活）。

`uv.lock` 是机器生成的 JSON，不要手改。**这个文件要入库**——它是跨机器复现的关键。

**镜像源配置**（中国读者）：在 `pyproject.toml` 加这一段：

```toml
[[tool.uv.index]]
url = "https://mirrors.aliyun.com/pypi/simple"
default = true
```

这样 `uv add` / `uv sync` 走阿里云镜像，速度比官方源快 10 倍。本章已配好。海外读者可以删掉这一段，或用 `UV_INDEX_URL=https://pypi.org/simple uv sync` 临时切回官方源。

## §4 动手：三个练习

概念讲完了。现在做三个短练习，每个演示一类语法。

### 4.1 把环境搭起来

进入本章目录，让 uv 创建虚拟环境并装依赖：

```bash
cd articles/01-python-crash-course
uv sync
```

如果 `uv sync` 不报错，说明环境健康。下面的命令都在这个目录下执行。

### 4.2 练习 1：基础语法（`ex1_basics.py`）

演示类型、集合、控制流、列表推导。每个函数都是独立可运行的：

```python
def describe_number(n: int) -> str:
    if n > 0:
        parity = "odd" if n % 2 else "even"
        return f"positive {parity}"
    if n < 0:
        return "negative"
    return "zero"
```

`if/elif/else` 链一路 return；条件表达式 `... if ... else ...` 在一行里给变量赋值；f-string `f"..."` 是 Python 3.6+ 的字符串格式化主力。

```python
def summarize_numbers(nums: list[int]) -> dict[str, int]:
    return {
        "count": len(nums),
        "sum": sum(nums),
        "distinct_count": len(set(nums)),
    }
```

`list[int]`、`dict[str, int]` 是 PEP 585 内置泛型（3.9+），替代老写法 `List[int]` / `Dict[str, int]`。`set(nums)` 一行去重。

```python
def first_n_squares(n: int) -> list[int]:
    return [i * i for i in range(1, n + 1)]
```

列表推导 `[expr for x in iterable]` 是「循环 + 收集结果」最常见写法。看到任何「for ... append」三行结构几乎都可以改成一行的列表推导。`range(1, n + 1)` 是闭区间 1 到 n；记错这个会出经典 off-by-one bug。

运行：

```bash
$ .venv/bin/python -m code.ex1_basics
positive odd
{'count': 6, 'sum': 14, 'distinct_count': 3}
[1, 4, 9, 16, 25]
```

### 4.3 练习 2：函数与类（`ex2_functions_classes.py`）

演示函数参数细节、`@dataclass` 装饰器、`@property`：

```python
def greet(name: str, *, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"
```

`*` 之后的参数是 **keyword-only**：`greet("Bob", "Hi")` 会失败，必须 `greet("Bob", greeting="Hi")`。这是显式 API 设计的好习惯——一旦你写了 `*`，后面参数就强制用关键字传。

```python
def variadic_sum(*nums: int, scale: float = 1.0) -> float:
    return sum(nums) * scale
```

`*nums` 收所有位置参数成元组。`*nums` 之后还能接 keyword-only 参数 `scale`。调用 `variadic_sum(1, 2, 3)` → `nums = (1, 2, 3)`，返回 6。

```python
@dataclass(frozen=True)
class Rectangle:
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height
```

`@dataclass` 自动生成 `__init__`、`__repr__`、`__eq__`——你只声明字段，重复代码由装饰器写。`frozen=True` 让实例不可变：`rect.width = 5.0` 会 raise `FrozenInstanceError`，而不是静默修改。`@property` 把方法伪装成属性：`rect.area` 不是 `rect.area()`。

本课后续章节会大量用 `@dataclass`——训练配置、数据集类、模型包装都靠它。

运行：

```bash
$ .venv/bin/python -m code.ex2_functions_classes
Hello, Alice!
Hi, Bob!
12.0
rect area: 12.0
```

### 4.4 练习 3：文件 I/O 与 CLI（`ex3_file_io_cli.py`）

演示 `pathlib`、`with`、`json`、`argparse`、错误处理：

```python
from pathlib import Path

def read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))
```

`pathlib.Path` 是标准库推荐的路径处理方式，替代 `os.path`。`Path("a/b/c.txt")` 在 Unix 和 Windows 都工作。`p.read_text(encoding="utf-8")` 一行读文件——**`encoding="utf-8"` 必须显式写**，默认是平台编码（Windows 上是 cp936），不写迟早踩坑。`str | Path` 是 PEP 604 union 语法（3.10+），表示「字符串或 Path 都行」。

```python
def count_lines(path: str | Path) -> int:
    p = Path(path)
    count = 0
    with p.open(encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count
```

`with` 语句自动管理文件资源——文件用完自动 close，不用 try/finally。**任何「获取资源 + 用 + 释放」三步模式都该用它**。逐行迭代文件是最稳的读大文件方式，不会一次把 10 GB 装进内存。

```python
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.mode == "lines":
            print(count_lines(args.path))
        else:
            print(json.dumps(read_json(args.path), ensure_ascii=False))
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
```

`argparse` 是标准库 CLI 框架，本课所有命令行入口（下载数据、训练、评估）都用它。`choices=` 限定选项的取值范围。错误输出走 `stderr`（`file=sys.stderr`），不是 `stdout`——这是 Unix 惯例：`>` 重定向 stdout 时，错误仍会显示在屏幕上。退出码 `0` = 成功，`非 0` = 失败，shell 脚本会看这个判断。

运行：

```bash
$ printf "a\nb\nc\n" > /tmp/f.txt
$ .venv/bin/python -m code.ex3_file_io_cli /tmp/f.txt --mode lines
3
$ .venv/bin/python -m code.ex3_file_io_cli /tmp/missing.json --mode json
error: file not found: /tmp/missing.json
$ echo $?
2
```

## §5 验证

三个练习的完整性检查：

```bash
# 1. 环境健康
uv sync                       # 不报错

# 2. 三个入口脚本都能跑
.venv/bin/python -m code.ex1_basics
.venv/bin/python -m code.ex2_functions_classes
.venv/bin/python -m code.ex3_file_io_cli <path> [--mode lines|json]
```

预期输出已在 §4 各小节贴出。
## §6 回顾

**本章带你扫过**（仅「见过」深度，不要求掌握）：

- Python 是什么（解释型、动态类型、多范式）
- 数据类型（int/float/str/bool/None、list/tuple/dict/set）
- 运算符（算术/比较/逻辑/成员）
- 控制流（if/elif/else、for、while、break/continue、列表推导）
- 函数（def、参数、return、*args、keyword-only）
- 类（class、__init__、self、@dataclass、继承基础）
- 容器操作（list/dict/set 常用方法）
- 错误处理（try/except、常见异常类型）
- 类型注解（运行时无效、现代写法）
- 模块与包（import、__init__.py、__main__）
- 文件 I/O（pathlib、with open、JSON、encoding="utf-8"）
- CLI（argparse、错误输出到 stderr、退出码）
- 依赖与虚拟环境（pip、venv、版本冲突）
- uv（uv venv / uv add / uv sync / uv run）
- uv 项目结构（pyproject.toml、uv.lock、阿里云镜像）

**本章没覆盖**（后续章节用到时会自然带出）：

- 生成器（`yield`）
- 装饰器（`@decorator`）
- 异步（`async` / `await`）
- OOP 高级（MRO / 描述符 / 元类 / 多继承）
- 上下文管理器（`@contextmanager`，自己写）
- `typing.Protocol`（结构化类型）
- `logging` 模块（替代 print 调试）

## §7 下篇预告

下一章会用本章的 Python 工具，去碰一个**图像分类项目**。你会第一次见到：

- 一个常见的小图像数据集
- NumPy 的 `ndarray` 是什么、怎么算矩阵乘法
- 一个能在你笔记本上跑通的最小 CNN 训练循环

不会讲数学证明，不会讲论文细节——只讲「跑起来需要知道什么」。

---

**作业**（自检，不强制）：

1. 在 `ex1_basics.py` 加一个 `fizzbuzz(n)` 函数，返回 1 到 n 的 FizzBuzz 结果（用控制流 + 列表推导）。
2. 在 `ex2_functions_classes.py` 加一个 `Triangle` dataclass（含三边长 + 计算周长方法）。
3. 给 `ex3_file_io_cli.main` 加一个 `--encoding` 参数，默认 `utf-8`。