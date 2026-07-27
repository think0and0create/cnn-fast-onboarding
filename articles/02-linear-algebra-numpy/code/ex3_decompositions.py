"""Ch 2 demo 3 — 矩阵分解(solve / eig / SVD)。

数学实现要点:
- 解线性方程组:A 必须可逆,奇异时 numpy.linalg.solve 自己抛 LinAlgError。
- 特征分解:numpy.linalg.eig 返回 (eigvals, eigvecs),特征向量按列。
- SVD 重建:U @ diag(S) @ Vh 还原全秩;k < rank 时取前 k 个奇异值。
"""

from __future__ import annotations

import numpy as np


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """解 A @ x = b;A 奇异时 numpy 自己抛 LinAlgError。"""
    return np.linalg.solve(A, b)


def eigendecomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """特征分解:返回 (eigvals, eigvecs),特征向量按列。"""
    eigvals, eigvecs = np.linalg.eig(A)
    return eigvals, eigvecs


def svd_reconstruct(A: np.ndarray, k: int | None = None) -> np.ndarray:
    """截断 SVD 重建;k=None 时取 min(A.shape) 即满秩。"""
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    if k is None:
        k = min(A.shape)
    return U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]


def main() -> int:
    """演示入口 — 解一个 2x2 方程组 + 一次 SVD 重建。"""
    A = np.array([[3.0, 2.0], [1.0, 2.0]])
    b = np.array([7.0, 5.0])
    x = solve_linear_system(A, b)
    print(f"解 A @ x = b: x = {x}")
    print(f"验证 A @ x = {A @ x}")
    rng = np.random.default_rng(seed=0)
    M = rng.standard_normal((4, 4))
    approx = svd_reconstruct(M, k=2)
    err = float(np.linalg.norm(M - approx, ord="fro"))
    print(f"SVD rank-2 重建误差 ||A - approx||_F = {err:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
