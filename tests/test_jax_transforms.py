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
    # DataFrame * DataFrame, DataFrame * array (elementwise JIT compat)
    ("df_mul_df", lambda df: (df * df).sum(axis=None), True, True),
    ("df_add_df", lambda df: (df + df).sum(axis=None), True, True),
    ("df_sub_df", lambda df: (df - df).sum(axis=None), True, True),
    ("df_mul_arr", lambda df: (df * jnp.array([2.0, 3.0])).sum(axis=None), True, True),
    # diff / pct_change
    ("diff_sum", lambda df: df.diff().sum(axis=None), True, True),
    ("pct_change_sum", lambda df: df.pct_change().sum(axis=None), True, False),
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
    # pipe, between, rank
    ("pipe_sum", lambda df: df.pipe(lambda d: d * 2).sum(axis=None), True, True),
    ("rank_sum", lambda df: df.rank().sum(axis=None), True, False),
    # Column manipulation (Session 6)
    ("drop_sum", lambda df: df.drop(columns=["a"]).sum(axis=None), True, True),
    ("rename_sum", lambda df: df.rename(columns={"a": "x"}).sum(axis=None), True, True),
    # Apply (Session 7)
    ("apply_sum", lambda df: df.apply(jnp.sum, axis=0).sum(), True, True),
    # Reverse operators (Session 8)
    ("radd_sum", lambda df: (10 + df).sum(axis=None), True, True),
    ("rsub_sum", lambda df: (100 - df).sum(axis=None), True, True),
    ("rmul_sum", lambda df: (3 * df).sum(axis=None), True, True),
    # Session 15: all, any, round, idxmin, idxmax, isin
    ("all_sum", lambda df: df.all(axis=0).values.sum(), True, False),  # boolean output
    ("any_sum", lambda df: df.any(axis=0).values.sum(), True, False),  # boolean output
    ("round_sum", lambda df: df.round(1).sum(axis=None), True, False),
    ("idxmin_sum", lambda df: df.idxmin(axis=0).values.sum(), True, False),  # discrete (argmin)
    ("idxmax_sum", lambda df: df.idxmax(axis=0).values.sum(), True, False),  # discrete (argmax)
    ("isin_sum", lambda df: df.isin([1.0, 4.0]).sum(axis=None), True, False),
    # Session 16: Rolling windows (fixed-size, JIT-compatible)
    ("roll_sum", lambda df: df.rolling(2).sum().sum(axis=None), True, True),
    ("roll_mean", lambda df: df.rolling(2).mean().sum(axis=None), True, True),
    ("roll_min", lambda df: df.rolling(2).min().sum(axis=None), True, False),
    ("roll_max", lambda df: df.rolling(2).max().sum(axis=None), True, False),
    # Expanding windows (JIT-compatible)
    ("exp_sum", lambda df: df.expanding().sum().sum(axis=None), True, True),
    ("exp_mean", lambda df: df.expanding().mean().sum(axis=None), True, True),
    ("exp_min", lambda df: df.expanding().min().sum(axis=None), True, False),
    ("exp_max", lambda df: df.expanding().max().sum(axis=None), True, False),
    # EWM
    ("ewm_mean", lambda df: df.ewm(span=2).mean().sum(axis=None), True, True),
    # Session 17: skew, kurt, sem (JIT+grad)
    ("skew_sum", lambda df: df.skew().values.sum(), True, True),
    ("sem_sum", lambda df: df.sem().values.sum(), True, True),
]

# GroupBy JAX compat — segment ops are JIT+grad compatible
# Group discovery is eager; aggregation ops use jax.ops.segment_*
GROUPBY_OPS = [
    ("gb_sum", lambda sgb: sgb.sum().values.sum(), True, True),
    ("gb_mean", lambda sgb: sgb.mean().values.sum(), True, True),
    ("gb_min", lambda sgb: sgb.min().values.sum(), True, False),
    ("gb_max", lambda sgb: sgb.max().values.sum(), True, False),
    ("gb_var", lambda sgb: sgb.var().values.sum(), True, True),
    ("gb_std", lambda sgb: sgb.std().values.sum(), True, True),
    ("gb_count", lambda sgb: sgb.count().values.sum(), True, False),
    ("gb_prod", lambda sgb: sgb.prod().values.sum(), True, False),
    ("gb_first", lambda sgb: sgb.first().values.sum(), True, False),
    ("gb_last", lambda sgb: sgb.last().values.sum(), True, False),
    ("gb_transform_sum", lambda sgb: sgb.transform("sum").values.sum(), True, True),
    ("gb_transform_mean", lambda sgb: sgb.transform("mean").values.sum(), True, True),
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


# Kurtosis needs n>=4 (denominator has (n-2)*(n-3))
KURT_DATA = {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 5.0, 6.0, 7.0]}


def _kurt_op(df):
    return df.kurt().values.sum()


class TestKurtJAX:
    def test_jit(self):
        df = DataFrame(KURT_DATA)
        eager = _kurt_op(df)
        jitted = jax.jit(_kurt_op)(df)
        assert_allclose(float(eager), float(jitted), rtol=1e-5)

    def test_grad(self):
        df = DataFrame(KURT_DATA)
        grad_fn = jax.grad(_kurt_op)
        grads = grad_fn(df)
        for block in grads._dtype_blocks.values():
            assert jnp.all(jnp.isfinite(block)), "kurt produced non-finite grads"


# All groups have >=2 elements so var/std gradients are finite
GROUPBY_JAXDATA = {"a": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}


@pytest.mark.parametrize("name,op,jit_ok,grad_ok", GROUPBY_OPS, ids=[o[0] for o in GROUPBY_OPS])
class TestGroupByJAX:
    def test_jit(self, name, op, jit_ok, grad_ok):
        if not jit_ok:
            pytest.skip(f"{name} not JIT-compatible")
        df = DataFrame(GROUPBY_JAXDATA)
        gb = df.groupby("a")["b"]  # SeriesGroupBy — pytree registered
        eager_result = op(gb)
        jitted_result = jax.jit(op)(gb)
        assert_allclose(float(eager_result), float(jitted_result), rtol=1e-5)

    def test_grad(self, name, op, jit_ok, grad_ok):
        if not grad_ok:
            pytest.skip(f"{name} not differentiable")
        df = DataFrame(GROUPBY_JAXDATA)
        gb = df.groupby("a")["b"]
        grad_fn = jax.grad(op)
        grads = grad_fn(gb)
        # grads is a SeriesGroupBy pytree — check the data leaf
        assert jnp.all(jnp.isfinite(grads._data)), f"{name} produced non-finite grads"
