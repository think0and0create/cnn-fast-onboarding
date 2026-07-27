"""ex1_basics.py — 类型 / 集合 / 控制流 演示。

本章用这个文件演示 Python 最基础的三件事：
- 基本类型与类型注解
- 集合（list / set / dict）的常见操作
- 控制流（if / for / 列表推导）

每个函数都很短，可以单独 `python -m code.ex1_basics` 看输出。
"""

from __future__ import annotations


def describe_number(n: int) -> str:
    """Return a human-readable description of `n`.

    Covers: int, str, if/elif/else, f-string, conditional expression.
    """
    if n > 0:
        parity = "odd" if n % 2 else "even"
        return f"positive {parity}"
    if n < 0:
        return "negative"
    return "zero"


def summarize_numbers(nums: list[int]) -> dict[str, int]:
    """Return summary stats for a list of ints.

    Covers: list, set, dict, len / sum / set builtins.
    """
    return {
        "count": len(nums),
        "sum": sum(nums),
        "distinct_count": len(set(nums)),
    }


def first_n_squares(n: int) -> list[int]:
    """Return the first `n` perfect squares (1, 4, 9, ...).

    Covers: range, list comprehension, edge cases (n == 0).
    """
    return [i * i for i in range(1, n + 1)]


def main() -> int:
    print(describe_number(7))
    print(summarize_numbers([1, 2, 2, 3, 3, 3]))
    print(first_n_squares(5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())