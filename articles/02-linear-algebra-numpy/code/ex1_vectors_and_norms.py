"""Ch 2 demo 1 — 向量与范数(L1/L2、内积、线性组合、正交判定)。

为什么不直接用 numpy:*写一遍把语义锁在测试里*,生产里还是调 numpy。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def vector_norm(v: np.ndarray, ord: int = 2) -> float:
    """向量 p-范数;`ord=1` 绝对值之和,`ord=2` Euclidean 默认。"""
    return float(np.linalg.norm(v, ord=ord))


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """向量内积;形状不一致必须 raise ValueError(不给 nan 蒙混过关)。"""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.dot(a, b))


def linear_combine(coeffs: Sequence[float], vectors: Sequence[np.ndarray]) -> np.ndarray:
    """线性组合 sum(c_i * v_i);空输入返回 shape==(0,) 的空数组。"""
    if not coeffs:
        return np.array([], dtype=float)
    if len(coeffs) != len(vectors):
        raise ValueError(f"length mismatch: {len(coeffs)} vs {len(vectors)}")
    acc = np.zeros_like(np.asarray(vectors[0], dtype=float))
    for c, v in zip(coeffs, vectors):
        acc = acc + float(c) * np.asarray(v, dtype=float)
    return acc


def are_orthogonal(a: np.ndarray, b: np.ndarray, *, tol: float = 1e-8) -> bool:
    """正交判定:|a·b| < tol。"""
    return abs(float(np.dot(a, b))) < tol


def main() -> int:
    """演示入口 — 跑几个向量算例,print 关键结果。"""
    v = np.array([3.0, 4.0])
    u = np.array([4.0, -3.0])
    print(f"L2 norm of {v} = {vector_norm(v):.4f}")
    print(f"dot(v, u) = {dot(v, u)} (理论上 0)")
    print(f"orthogonal(v, u) = {are_orthogonal(v, u)}")
    coeffs = [2.0, 3.0]
    bases = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    print(f"linear_combine({coeffs}, bases) = {linear_combine(coeffs, bases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
