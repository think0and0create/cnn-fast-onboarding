# Ch 1 · Python 光速扫盲

> 一句话：1-2 小时过完 Python 最常用的语法与模式，让你后续见到这些语法时**认得出、大致懂**，不要求精通。

## §1 目标

跑完本章，你将能够：

- **认出**后续章节里出现的 Python 语法（看到 `dataclass`、`*args`、`argparse` 时知道它在干什么大致）。
- **边读边动手**：每讲完一个语法点就能在终端敲一行验证。
- 知道 `uv` 是什么、能用 `uv run` 跑脚本。
- 写三个小练习：基础语法、函数与类、文件 I/O 与 CLI。

**不承诺**：

- ❌ 流畅读懂所有代码——那是「精通」不是「扫盲」。
- ❌ 写出生产级 Python 代码。
- ❌ 掌握装饰器 / 异步 / 描述符 / 元类——后续章节**用到时会回过头来加深理解**。

## §2 为什么先做这一章

本课的后续章节都会用到 Python：类型注解、`@dataclass`、`*args`、`argparse`、`pathlib`、文件 I/O。这些**不是「先学一遍」就能掌握的**——它们的细节会在用到时加深。

本课采用「**边学边跑**」的方式：每讲完一个语法点，立刻在终端敲一行验证。**等你读完第一个语法点，就已经在命令行里动过 Python 了**——这是本章跟传统教材「先讲完所有理论再动手」最大的区别。

如果你已经熟悉 Python，可以快速翻一遍 §3（uv）和 §5（三个练习），跳过中间的语法描述。

## §3 跑 Python：环境与第一个程序

这一节**先不讲语法**，先让你跑出第一行 Python 代码。

### 3.1 装 uv

**uv** 是 Astral 公司出的 Python 工具链，同一个团队也做了代码检查工具 `ruff`。它把 `pip`、`venv`、`pip-tools` 这些历史工具合并成一个，**快 10-100 倍**。

```bash
# 推荐（Linux/macOS）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者用 pip
pip install uv
```

装完后确认：

```bash
$ uv --version
uv 0.x.x
```

### 3.2 第一个 print

uv 装好之后，**不需要任何额外配置**就能跑 Python：

```bash
$ uv run python -c "print('hello, world')"
hello, world
```

`uv run python` 让 uv 自动管理 Python 解释器和虚拟环境——你不用先创建 venv、不用激活、不用装任何东西，直接跑。

### 3.3 第一个脚本

把代码放进文件里：

```bash
$ mkdir hello && cd hello
$ echo 'print("hello, world")' > hello.py
$ uv run python hello.py
hello, world
```

`uv run` 会自动检测当前目录的 `pyproject.toml`（如果有），用项目配置的 Python 版本和依赖运行。

### 3.4 REPL 是什么

**REPL**（Read-Eval-Print Loop）就是「读一行 → 算一行 → 打印结果 → 循环」：

```bash
$ uv run python
Python 3.12.x
>>> 1 + 2
3
>>> name = "Alice"
>>> f"hello, {name}"
'hello, Alice'
>>> exit()
```

`>>>` 是输入提示。REPL 是 Python 的「计算器」，用来快速试错、查类型、调试小段代码。

**学完这一节，你的学习姿势是这样**：

1. 读完下面 §4 一个语法点
2. 看到「**试一下**」小段，复制里面的命令到终端，**跑一遍**
3. 看到输出符合预期，**这一节就过了**
4. 进下一节

**不需要先建项目、不需要先建 venv**——uv 帮你搞定了。

### 3.5 项目结构（用到时再看）

如果你要写一个**带依赖**的项目（比如要装 `numpy`），需要建 `pyproject.toml`：

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
]
```

```bash
uv sync        # 读 pyproject.toml，创建 .venv，装依赖
uv run python my_script.py   # 在 .venv 里跑
```

**镜像源配置**（中国读者）：在 `pyproject.toml` 加：

```toml
[[tool.uv.index]]
url = "https://mirrors.aliyun.com/pypi/simple"
default = true
```

这样 `uv sync` 走阿里云镜像，比官方源快 10 倍。海外读者可以删掉这一段，或用 `UV_INDEX_URL=https://pypi.org/simple uv sync` 临时切回官方源。

**本章用不上依赖**（三个练习都是纯标准库），所以你不需要建 `pyproject.toml`，直接用 `uv run python` 跑就行。后续章节要装 `numpy` / `torch` 时再回来看这一节。

## §4 语法速览

每节一个**最小语法点**，讲完立刻有一个「**试一下**」小段。**强烈建议你一边读一边敲**——光看是记不住的。

### 4.1 数据类型与变量

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
```

变量名约定：小写 + 下划线（`snake_case`）。常量全大写（`MAX_SIZE = 100`），但 Python 没有真正的「常量」语法——全大写只是约定。

**试一下**：

```bash
$ uv run python -c "x = 42; print(type(x), x)"
<class 'int'> 42
$ uv run python -c "print(list('abc'))"
['a', 'b', 'c']
```

### 4.2 运算符

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

**试一下**：

```bash
$ uv run python -c "print(7 // 3, 7 % 3, 2 ** 10)"
2 1 1024
$ uv run python -c "print(1 < 2 and 3 < 4, 'a' in ['a', 'b'])"
True True
```

### 4.3 控制流：if / for / while

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

**试一下**：

```bash
$ uv run python -c "print([i*i for i in range(5)])"
[0, 1, 4, 9, 16]
$ uv run python -c "print([x for x in range(10) if x % 2 == 0])"
[0, 2, 4, 6, 8]
```

### 4.4 函数

**基本形态**：

```python
def add(a: int, b: int) -> int:
    return a + b
```

- `def` 关键字 + 函数名 + 参数列表 + `:` + 缩进的函数体。
- `return` 给出返回值；如果没有 `return`，函数返回 `None`。
- 类型注解（`a: int`、`-> int`）**运行时无效**，但对读者和工具有用（详见 4.8）。

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

**试一下**：

```bash
$ uv run python -c "def square(x): return x*x; print(square(5))"
25
```

### 4.5 类与对象

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

`@dataclass(frozen=True)` 让实例不可变：`rect.width = 5.0` 会 raise `FrozenInstanceError`。不可变对象可以放进 `set`、当 dict key、调试时少一类 bug。

**继承**（基础）：

```python
class Square(Rectangle):
    def __init__(self, side: float):
        super().__init__(side, side)   # 调用父类 __init__
```

`Square` 继承 `Rectangle`，复用父类方法。`super().__init__(...)` 调用父类构造函数。本课后续章节**不会深入继承**（OOP 高级不教），到这里够用。

**试一下**：

```bash
$ uv run python -c "from dataclasses import dataclass; @dataclass\nclass P:\n    x: int\n    y: int\nprint(P(1, 2))"
P(x=1, y=2)
```

### 4.6 容器操作与推导式

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

**试一下**：

```bash
$ uv run python -c "print({1,2,3} | {2,3,4}, {1,2,3} & {2,3,4}, {1,2,3} - {2,3,4})"
{1, 2, 3, 4} {2, 3} {1}
```

### 4.7 错误处理：try / except

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

**试一下**：

```bash
$ uv run python -c "try:\n  int('abc')\nexcept ValueError as e:\n  print('caught:', e)"
caught: invalid literal for int() with base 10: 'abc'
```

### 4.8 类型注解：声明，不是强制

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

**试一下**：

```bash
$ uv run python -c "def add(x: int, y: int) -> int: return x + y\nprint(add(2, 3), add('a', 'b'))"
2 ab
```

（注意第二行：传字符串不报错，注解只影响阅读，不影响运行。）

### 4.9 模块、包与导入

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

**试一下**：

```bash
$ uv run python -c "import math; print(math.sqrt(16))"
4.0
```

### 4.10 文件 I/O 与路径

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

**试一下**：

```bash
$ uv run python -c "from pathlib import Path; p = Path('/tmp/_test.txt'); p.write_text('hi\n', encoding='utf-8'); print(p.read_text(encoding='utf-8').strip())"
hi
```

### 4.11 CLI（argparse）

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

**试一下**：

```bash
$ uv run python -c "import argparse; p = argparse.ArgumentParser(); p.add_argument('--name', default='world'); print(p.parse_args([]).name)"
world
```

## §5 三个动手练习

语法过完了。现在做三个短练习，每个演示一类语法，加深理解。**这一节需要 `pyproject.toml`**——三个练习在 `articles/01-python-crash-course/code/` 下，是一个完整的 uv 项目。

### 5.1 准备环境

```bash
cd <项目根目录>/articles/01-python-crash-course
uv sync
```

`uv sync` 读 `pyproject.toml`，创建 `.venv`，装好依赖（虽然依赖只有 `pytest`，但 uv 会按规范执行）。下面的命令都在这个目录下执行。

### 5.2 练习 1：基础语法（`ex1_basics.py`）

演示类型、集合、控制流、列表推导。

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

**运行**：

```bash
$ .venv/bin/python -m code.ex1_basics
positive odd
{'count': 6, 'sum': 14, 'distinct_count': 3}
[1, 4, 9, 16, 25]
```

### 5.3 练习 2：函数与类（`ex2_functions_classes.py`）

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

**运行**：

```bash
$ .venv/bin/python -m code.ex2_functions_classes
Hello, Alice!
Hi, Bob!
12.0
rect area: 12.0
```

### 5.4 练习 3：文件 I/O 与 CLI（`ex3_file_io_cli.py`）

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

**运行**：

```bash
$ printf "a\nb\nc\n" > /tmp/f.txt
$ .venv/bin/python -m code.ex3_file_io_cli /tmp/f.txt --mode lines
3
$ .venv/bin/python -m code.ex3_file_io_cli /tmp/missing.json --mode json
error: file not found: /tmp/missing.json
$ echo $?
2
```

## §6 验证

三个练习的完整性检查：

```bash
# 1. 环境健康
uv sync                       # 不报错

# 2. 三个入口脚本都能跑
.venv/bin/python -m code.ex1_basics
.venv/bin/python -m code.ex2_functions_classes
.venv/bin/python -m code.ex3_file_io_cli <path> [--mode lines|json]
```

预期输出已在 §5 各小节贴出。

## §7 回顾

**本章带你扫过**（仅「见过」深度，不要求掌握）：

- **环境**（§3）：uv 装好、`uv run python` 跑命令、REPL 用法、`pyproject.toml` 含义
- **语法**（§4）：类型、运算符、控制流、函数、类、`@dataclass`、容器、错误处理、类型注解、模块、文件 I/O、CLI
- **三个练习**（§5）：基础语法、函数与类、文件 I/O 与 CLI

**本章没覆盖**（后续章节用到时会自然带出）：

- 生成器（`yield`）
- 装饰器（`@decorator`）
- 异步（`async` / `await`）
- OOP 高级（MRO / 描述符 / 元类 / 多继承）
- 上下文管理器（`@contextmanager`，自己写）
- `typing.Protocol`（结构化类型）
- `logging` 模块（替代 print 调试）

## §8 下篇预告

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
