"""
JAX transform compatibility tests.

For each operation, verify:
  1. JIT: operation compiles and produces same result as eager
  2. grad: operation is differentiable and produces finite gradients
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from jaxframe import DataFrame

DATA = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}

# (name, op_returning_scalar, supports_jit, supports_grad)
OPERATIONS = [
    # Reductions
    ("sum", lambda df: df.sum(axis=None), True, True),
    ("mean", lambda df: df.mean(axis=None), True, True),
    ("std", lambda df: df.std(axis=None), True, True),
    ("var", lambda df: df.var(axis=None), True, True),
    ("min", lambda df: df.min(axis=None), True, False),
    ("max", lambda df: df.max(axis=None), True, False),
    ("prod", lambda df: df.prod(axis=None), True, True),
    # Arithmetic chains
    ("mul_sum", lambda df: (df * 2).sum(axis=None), True, True),
    ("add_sum", lambda df: (df + 10).sum(axis=None), True, True),
    ("pow_sum", lambda df: (df**2).sum(axis=None), True, True),
    ("chain", lambda df: ((df + 1) * 2 - 3).sum(axis=None), True, True),
    # Where / clip
    ("where_sum", lambda df: df.where(df > 3, 0).sum(axis=None), True, True),
    ("clip_sum", lambda df: df.clip(2, 5).sum(axis=None), True, True),
    ("shift_sum", lambda df: df.shift(1, fill_value=0).sum(axis=None), True, True),
    # Missing data ops (Session 1)
    ("fillna_sum", lambda df: df.fillna(0.0).sum(axis=None), True, True),
    ("isna_sum", lambda df: df.isna().sum(axis=None), True, False),
    ("notna_sum", lambda df: df.notna().sum(axis=None), True, False),
    # Cumulative & descriptive (Session 2)
    ("cumsum_sum", lambda df: df.cumsum().sum(axis=None), True, True),
    ("cumprod_sum", lambda df: df.cumprod().sum(axis=None), True, True),
    ("cumsum_ax1_sum", lambda df: df.cumsum(axis=1).sum(axis=None), True, True),
    ("cumprod_ax1_sum", lambda df: df.cumprod(axis=1).sum(axis=None), True, True),
    ("count_sum", lambda df: df.count().sum(), True, False),
    ("median_sum", lambda df: df.median().sum(), True, False),
    # Copy (Session 3) — copy then reduce
    ("copy_sum", lambda df: df.copy().sum(axis=None), True, True),
    # Sorting (Session 4) — argsort not differentiable
    ("sort_sum", lambda df: df.sort_values("a").sum(axis=None), True, False),
    # Describe & quantile (Session 5)
    ("quantile_sum", lambda df: df.quantile(0.5).sum(), True, False),
    ("nlargest_sum", lambda df: df.nlargest(2, "a").sum(axis=None), True, False),
    ("nsmallest_sum", lambda df: df.nsmallest(2, "a").sum(axis=None), True, False),
    # Column manipulation (Session 6)
    ("drop_sum", lambda df: df.drop(columns=["a"]).sum(axis=None), True, True),
    ("rename_sum", lambda df: df.rename(columns={"a": "x"}).sum(axis=None), True, True),
]


@pytest.mark.parametrize("name,op,jit_ok,grad_ok", OPERATIONS, ids=[o[0] for o in OPERATIONS])
class TestJAXCompat:
    def test_jit(self, name, op, jit_ok, grad_ok):
        if not jit_ok:
            pytest.skip(f"{name} not JIT-compatible")
        df = DataFrame(DATA)
        eager_result = op(df)
        jitted_result = jax.jit(op)(df)
        assert_allclose(float(eager_result), float(jitted_result), rtol=1e-5)

    def test_grad(self, name, op, jit_ok, grad_ok):
        if not grad_ok:
            pytest.skip(f"{name} not differentiable")
        df = DataFrame(DATA)
        grad_fn = jax.grad(op)
        grads = grad_fn(df)
        for block in grads._dtype_blocks.values():
            assert jnp.all(jnp.isfinite(block)), f"{name} produced non-finite gradients"
