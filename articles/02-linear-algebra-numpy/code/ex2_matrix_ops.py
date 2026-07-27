"""Ch 2 demo 2 — 矩阵运算(matmul / transpose / inverse / determinant)。

数学实现要点:
- 矩阵乘法在生产里直接调 A @ B;这里把\"形状不匹配\"做成显式 raise ValueError。
- 求逆前先判断行列式为零,失败抛 numpy.linalg.LinAlgError。
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """矩阵乘法;A 的列数 != B 的行数(或非 2-D)时 raise ValueError。"""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"expected 2-D matrices, got {A.ndim}-D and {B.ndim}-D")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"inner-dim mismatch: {A.shape} @ {B.shape}")
    return np.asarray(A) @ np.asarray(B)


def transpose(A: np.ndarray) -> np.ndarray:
    """矩阵转置。"""
    return np.asarray(A).T


def inverse(A: np.ndarray) -> np.ndarray:
    """矩阵求逆;奇异矩阵(det == 0)抛 LinAlgError,不静默返回 nan 矩阵。"""
    if np.linalg.det(A) == 0.0:
        raise LinAlgError("matrix is singular; cannot invert")
    return np.linalg.inv(A)


def determinant(A: np.ndarray) -> float:
    """矩阵行列式。"""
    return float(np.linalg.det(A))


def main() -> int:
    """演示入口 — 2x2 矩阵乘法 + 求逆 + 行列式。"""
    A = np.array([[4.0, 7.0], [2.0, 6.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    print(f"matmul(A, I) = {matmul(A, B).tolist()}")
    print(f"det(A) = {determinant(A):.4f}")
    inv = inverse(A)
    print(f"inv(A) = {inv.tolist()}")
    print(f"verify A @ inv(A) = {matmul(A, inv).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
