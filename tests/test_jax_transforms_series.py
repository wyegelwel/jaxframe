"""JAX transform compatibility tests for Series operations.

Same contract as test_jax_transforms.py: each op is
(name, op_returning_scalar, supports_jit, supports_grad).
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from jaxframe import Series

DATA = [3.0, 1.0, 4.0, 2.0, 5.0]

SERIES_OPERATIONS = [
    # Reductions
    ("sum", lambda s: s.sum(), True, True),
    ("mean", lambda s: s.mean(), True, True),
    ("std", lambda s: s.std(), True, True),
    ("var", lambda s: s.var(), True, True),
    ("prod", lambda s: s.prod(), True, True),
    ("min", lambda s: s.min(), True, True),  # subgradient
    ("max", lambda s: s.max(), True, True),
    ("median", lambda s: s.median(), True, True),
    ("quantile", lambda s: s.quantile(0.3), True, True),
    ("sem", lambda s: s.sem(), True, True),
    ("skew", lambda s: s.skew(), True, True),
    ("kurt", lambda s: s.kurt(), True, True),
    ("count", lambda s: s.count() * 1.0, True, False),  # integer output
    ("autocorr", lambda s: s.autocorr(), True, True),
    ("dot", lambda s: s.dot(s), True, True),
    ("corr_shift", lambda s: s.corr(s * 2.0), True, True),
    ("cov_shift", lambda s: s.cov(s * 2.0), True, True),
    # Cumulative
    ("cumsum", lambda s: s.cumsum().sum(), True, True),
    ("cumprod", lambda s: s.cumprod().sum(), True, True),
    ("cummax", lambda s: s.cummax().sum(), True, True),
    ("cummin", lambda s: s.cummin().sum(), True, True),
    # Elementwise / fills
    ("abs", lambda s: s.abs().sum(), True, True),
    ("clip", lambda s: s.clip(1.5, 4.5).sum(), True, True),
    ("round", lambda s: s.round().sum(), True, False),  # zero grad step fn
    ("ffill", lambda s: s.ffill().sum(), True, True),
    ("bfill", lambda s: s.bfill().sum(), True, True),
    ("interpolate", lambda s: s.interpolate().sum(), True, True),
    ("fillna", lambda s: s.fillna(0.0).sum(), True, True),
    ("where", lambda s: s.where(s > 2, 0.0).sum(), True, True),
    ("mask", lambda s: s.mask(s > 2, 0.0).sum(), True, True),
    ("case_when", lambda s: s.case_when([(s > 3, 10.0)]).sum(), True, True),
    ("isna", lambda s: s.isna().sum() * 1.0, True, False),
    ("isin", lambda s: s.isin([1.0, 2.0]).sum() * 1.0, True, False),
    # Named arithmetic
    ("add", lambda s: s.add(2).sum(), True, True),
    ("rsub", lambda s: s.rsub(10).sum(), True, True),
    ("mul", lambda s: s.mul(3).sum(), True, True),
    ("div", lambda s: s.div(2).sum(), True, True),
    ("pow", lambda s: s.pow(2).sum(), True, True),
    ("combine_first", lambda s: s.combine_first(s * 0.0).sum(), True, True),
    # Shifts / diffs
    ("shift", lambda s: s.shift(1, fill_value=0.0).sum(), True, True),
    ("diff", lambda s: s.diff().fillna(0.0).sum(), True, True),
    # Sorting / ranking / selection
    ("sort_values", lambda s: s.sort_values().cumsum().sum(), True, True),
    ("sort_values_desc", lambda s: s.sort_values(ascending=False).cumsum().sum(), True, True),
    ("rank", lambda s: s.rank().sum(), True, False),  # step function
    ("nlargest", lambda s: s.nlargest(2).sum(), True, True),
    ("nsmallest", lambda s: s.nsmallest(2).sum(), True, True),
    ("head", lambda s: s.head(3).sum(), True, True),
    ("tail", lambda s: s.tail(3).sum(), True, True),
    ("take", lambda s: s.take([0, 2]).sum(), True, True),
    ("argsort", lambda s: s.argsort().sum() * 1.0, True, False),
    ("searchsorted", lambda s: s.sort_values().searchsorted(2.5) * 1.0, True, False),
    # Function application / windows
    ("apply", lambda s: s.apply(lambda x: x * 2).sum(), True, True),
    ("transform", lambda s: s.transform(lambda x: x - x.mean()).sum(), True, True),
    ("agg_callable", lambda s: s.agg(lambda x: x.sum()), True, True),
    ("pipe", lambda s: s.pipe(lambda x: x * 2).sum(), True, True),
    ("rolling_sum", lambda s: s.rolling(2).sum().fillna(0.0).sum(), True, True),
    ("rolling_mean", lambda s: s.rolling(2).mean().fillna(0.0).sum(), True, True),
    ("expanding_sum", lambda s: s.expanding().sum().sum(), True, True),
    ("ewm_mean", lambda s: s.ewm(alpha=0.5).mean().sum(), True, True),
    ("describe", lambda s: s.describe().values.sum(), True, True),
]


@pytest.mark.parametrize(
    "name,op,jit_ok,grad_ok", SERIES_OPERATIONS, ids=[o[0] for o in SERIES_OPERATIONS]
)
class TestSeriesJAXCompat:
    def test_jit(self, name, op, jit_ok, grad_ok):
        if not jit_ok:
            pytest.skip(f"{name} not JIT-compatible")
        s = Series(DATA, name="x")
        eager_result = op(s)
        jitted_result = jax.jit(op)(s)
        assert_allclose(float(eager_result), float(jitted_result), rtol=1e-5)

    def test_grad(self, name, op, jit_ok, grad_ok):
        if not grad_ok:
            pytest.skip(f"{name} not differentiable")
        s = Series(DATA, name="x")
        grads = jax.grad(op)(s)
        assert jnp.all(jnp.isfinite(grads._data)), f"{name} produced non-finite gradients"


def test_series_grad_sort_values_routes_correctly():
    """Gradient through sort must land on the original positions."""
    s = Series([3.0, 1.0, 2.0])
    g = jax.grad(lambda x: (x.sort_values().values * jnp.array([1.0, 2.0, 3.0])).sum())(s)
    # sorted order is [1(idx1), 2(idx2), 3(idx0)] with weights [1,2,3]
    assert_allclose(jnp.asarray(g._data), jnp.array([3.0, 1.0, 2.0]))


def test_series_grad_nlargest_routes_correctly():
    s = Series([3.0, 1.0, 2.0])
    g = jax.grad(lambda x: x.nlargest(2).sum())(s)
    assert_allclose(jnp.asarray(g._data), jnp.array([1.0, 0.0, 1.0]))


def test_series_grad_ffill_accumulates():
    s = Series([1.0, jnp.nan, 3.0])
    g = jax.grad(lambda x: x.ffill().fillna(0.0).sum())(s)
    assert_allclose(jnp.asarray(g._data), jnp.array([2.0, 0.0, 1.0]))
