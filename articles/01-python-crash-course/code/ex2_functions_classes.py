"""ex2_functions_classes.py — 函数 / 类 / dataclass 演示。

本章用这个文件演示：
- 函数定义、默认参数、keyword-only 参数
- *args 可变位置参数
- @dataclass 自动生成 __init__ / __repr__ / __eq__
- @property 把方法当属性用
- frozen=True 让实例不可变（hashable）
"""

from __future__ import annotations

from dataclasses import dataclass


def greet(name: str, *, greeting: str = "Hello") -> str:
    """Return a greeting. `greeting` is keyword-only (after the `*`)."""
    return f"{greeting}, {name}!"


def variadic_sum(*nums: int, scale: float = 1.0) -> float:
    """Sum any number of ints, optionally scaled.

    Covers: *args, keyword-only after *args.
    """
    return sum(nums) * scale


@dataclass(frozen=True)
class Rectangle:
    """A 2D rectangle. frozen=True makes instances immutable."""

    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height


def main() -> int:
    print(greet("Alice"))
    print(greet("Bob", greeting="Hi"))
    print(variadic_sum(1, 2, 3, scale=2.0))
    rect = Rectangle(width=3.0, height=4.0)
    print(f"rect area: {rect.area}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())