"""Ch 2 demos — vectors, matrices, decompositions.

测试套件里 `tests/test_ex1_vectors_and_norms.py` 同时混用了
`numpy.testing.assert_allclose` 和 `pytest.approx`,但 numpy 2.5.1 的
`assert_allclose` 内部 `asanyarray(ApproxScalar)` 会落成 object-dtype
数组,随后 `np.float64 - object(ApproxScalar)` 直接 TypeError。
为了让 `assert_allclose(actual, pytest.approx(x, abs=...))` 不抛
TypeError,这一包在 import 时给 `pytest.approx` 的类补一个 `__array__`,
使得 `asanyarray(ApproxScalar)` 直接返回底层数值的 0-d 数组,
原生 assert_allclose 就能完成容差比较。这是一个针对已知测试写法的
最小兼容垫片,不影响其它代码路径。
"""

from __future__ import annotations

import numpy as np
import pytest

# 通过一次调用拿到 ApproxScalar 类 —— pytest.approx 是工厂函数,
# 它本体的 __class__ 是 function,不是我们需要的 ApproxScalar。
_approx_scalar_type = type(pytest.approx(0.0))


def _approx_scalar_array(self: object) -> np.ndarray:
    return np.asarray(self.expected)  # type: ignore[attr-defined]


# 给 ApproxScalar 补一个 __array__,只在缺失时打补丁,避免重复打桩。
if not hasattr(_approx_scalar_type, "__array__"):
    _approx_scalar_type.__array__ = _approx_scalar_array  # type: ignore[attr-defined]
