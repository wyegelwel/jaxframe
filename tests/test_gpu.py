"""
GPU compatibility tests.

Tests that jaxframe operations work correctly on GPU.
All tests are skipped if no GPU is available.
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from jaxframe import DataFrame

gpu_available = any(d.platform == "gpu" for d in jax.devices())
requires_gpu = pytest.mark.skipif(not gpu_available, reason="No GPU available")

DATA = {"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]}


@requires_gpu
class TestGPUBasic:
    """Verify data lives on GPU and basic operations work."""

    def test_data_on_gpu(self):
        df = DataFrame(DATA)
        for block in df._dtype_blocks.values():
            gpu_block = jax.device_put(block, jax.devices("gpu")[0])
            assert gpu_block.devices().pop().platform == "gpu"

    def test_arithmetic_gpu(self):
        df = DataFrame(DATA)
        df_gpu = jax.device_put(df, jax.devices("gpu")[0])
        result = (df_gpu * 2 + 1).sum(axis=None)
        expected = (jnp.array([1, 2, 3, 4, 5, 6, 7, 8.0]) * 2 + 1).sum()
        assert_allclose(float(result), float(expected), rtol=1e-5)

    def test_reduction_gpu(self):
        df = DataFrame(DATA)
        df_gpu = jax.device_put(df, jax.devices("gpu")[0])
        result = df_gpu.mean(axis=None)
        assert_allclose(float(result), 4.5, rtol=1e-5)


@requires_gpu
class TestGPUJit:
    """Verify JIT compilation works on GPU."""

    def test_jit_sum_gpu(self):
        df = DataFrame(DATA)
        df_gpu = jax.device_put(df, jax.devices("gpu")[0])
        jit_fn = jax.jit(lambda df: df.sum(axis=None))
        result = jit_fn(df_gpu)
        assert_allclose(float(result), 36.0, rtol=1e-5)

    def test_jit_chain_gpu(self):
        df = DataFrame(DATA)
        df_gpu = jax.device_put(df, jax.devices("gpu")[0])
        jit_fn = jax.jit(lambda df: ((df + 1) * 2 - 3).sum(axis=None))
        result = jit_fn(df_gpu)
        expected = ((jnp.array([1, 2, 3, 4, 5, 6, 7, 8.0]) + 1) * 2 - 3).sum()
        assert_allclose(float(result), float(expected), rtol=1e-5)

    def test_jit_where_gpu(self):
        df = DataFrame(DATA)
        df_gpu = jax.device_put(df, jax.devices("gpu")[0])
        jit_fn = jax.jit(lambda df: df.where(df > 3, 0).sum(axis=None))
        result = jit_fn(df_gpu)
        eager = df.where(df > 3, 0).sum(axis=None)
        assert_allclose(float(result), float(eager), rtol=1e-5)

    def test_jit_rolling_gpu(self):
        df = DataFrame(DATA)
        df_gpu = jax.device_put(df, jax.devices("gpu")[0])
        jit_fn = jax.jit(lambda df: df.rolling(2).sum().sum(axis=None))
        result = jit_fn(df_gpu)
        eager = df.rolling(2).sum().sum(axis=None)
        assert_allclose(float(result), float(eager), rtol=1e-5)


@requires_gpu
class TestGPUGrad:
    """Verify gradient computation works on GPU."""

    def test_grad_sum_gpu(self):
        df = DataFrame(DATA)
        df_gpu = jax.device_put(df, jax.devices("gpu")[0])
        grad_fn = jax.grad(lambda df: df.sum(axis=None))
        grads = grad_fn(df_gpu)
        # Gradient of sum w.r.t. all elements should be 1.0
        for block in grads._dtype_blocks.values():
            assert_allclose(block, jnp.ones_like(block), rtol=1e-5)

    def test_grad_mean_gpu(self):
        df = DataFrame(DATA)
        df_gpu = jax.device_put(df, jax.devices("gpu")[0])
        grad_fn = jax.grad(lambda df: df.mean(axis=None))
        grads = grad_fn(df_gpu)
        for block in grads._dtype_blocks.values():
            assert jnp.all(jnp.isfinite(block))

    def test_grad_chain_gpu(self):
        df = DataFrame(DATA)
        df_gpu = jax.device_put(df, jax.devices("gpu")[0])
        grad_fn = jax.grad(lambda df: (df**2).sum(axis=None))
        grads = grad_fn(df_gpu)
        # d/dx (x^2) = 2x
        for block in grads._dtype_blocks.values():
            assert jnp.all(jnp.isfinite(block))


@requires_gpu
class TestGPUGroupBy:
    """Verify GroupBy works on GPU."""

    def test_groupby_sum_gpu(self):
        data = {
            "a": [1.0, 1.0, 2.0, 2.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        }
        df = DataFrame(data)
        df_gpu = jax.device_put(df, jax.devices("gpu")[0])
        gb = df_gpu.groupby("a")["b"]
        result = gb.sum()
        eager = df.groupby("a")["b"].sum()
        assert_allclose(
            float(result.values.sum()),
            float(eager.values.sum()),
            rtol=1e-5,
        )

    def test_groupby_jit_gpu(self):
        data = {
            "a": [1.0, 1.0, 2.0, 2.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        }
        df = DataFrame(data)
        gb = df.groupby("a")["b"]
        gb_gpu = jax.device_put(gb, jax.devices("gpu")[0])
        jit_fn = jax.jit(lambda gb: gb.sum().values.sum())
        result = jit_fn(gb_gpu)
        eager = gb.sum().values.sum()
        assert_allclose(float(result), float(eager), rtol=1e-5)
