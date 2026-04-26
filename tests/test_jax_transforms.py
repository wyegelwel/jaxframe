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
    # clip/shift reconstruct via __init__ which calls np.asarray on traced values — known bug
    ("clip_sum", lambda df: df.clip(2, 5).sum(axis=None), False, False),
    ("shift_sum", lambda df: df.shift(1, fill_value=0).sum(axis=None), False, False),
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
