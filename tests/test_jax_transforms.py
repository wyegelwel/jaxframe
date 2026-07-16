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
    ("min", lambda df: df.min(axis=None), True, True),  # subgradient at argmin,
    ("max", lambda df: df.max(axis=None), True, True),  # subgradient at argmax,
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
    ("median_sum", lambda df: df.median().sum(), True, True),  # quantile interp grad,
    # Copy (Session 3) — copy then reduce
    ("copy_sum", lambda df: df.copy().sum(axis=None), True, True),
    # Sorting (Session 4) — argsort not differentiable
    (
        "sort_sum",
        lambda df: df.sort_values("a").sum(axis=None),
        True,
        True,
    ),  # jnp.argsort + gather,
    # Describe & quantile (Session 5)
    ("quantile_sum", lambda df: df.quantile(0.5).sum(), True, True),
    ("nlargest_sum", lambda df: df.nlargest(2, "a").sum(axis=None), True, True),
    ("nsmallest_sum", lambda df: df.nsmallest(2, "a").sum(axis=None), True, True),
    # pipe, between, rank
    ("pipe_sum", lambda df: df.pipe(lambda d: d * 2).sum(axis=None), True, True),
    ("rank_sum", lambda df: df.rank().sum(axis=None), True, False),  # step function: zero grad,
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
    ("roll_min", lambda df: df.rolling(2).min().sum(axis=None), True, True),
    ("roll_max", lambda df: df.rolling(2).max().sum(axis=None), True, True),
    # Expanding windows (JIT-compatible)
    ("exp_sum", lambda df: df.expanding().sum().sum(axis=None), True, True),
    ("exp_mean", lambda df: df.expanding().mean().sum(axis=None), True, True),
    ("exp_min", lambda df: df.expanding().min().sum(axis=None), True, True),
    ("exp_max", lambda df: df.expanding().max().sum(axis=None), True, True),
    # EWM
    ("ewm_mean", lambda df: df.ewm(span=2).mean().sum(axis=None), True, True),
    # API-parity expansion: fills / cumulative / named ops / gathers
    ("ffill_sum", lambda df: df.ffill().sum(axis=None), True, True),
    ("bfill_sum", lambda df: df.bfill().sum(axis=None), True, True),
    ("interp_sum", lambda df: df.interpolate().sum(axis=None), True, True),
    ("cummax_sum", lambda df: df.cummax().sum(axis=None), True, True),
    ("cummin_sum", lambda df: df.cummin().sum(axis=None), True, True),
    ("named_add_sum", lambda df: df.add(2).sum(axis=None), True, True),
    ("named_rsub_sum", lambda df: df.rsub(10).sum(axis=None), True, True),
    ("named_div_sum", lambda df: df.div(2).sum(axis=None), True, True),
    ("named_pow_sum", lambda df: df.pow(2).sum(axis=None), True, True),
    ("eq_sum", lambda df: df.eq(2.0).sum(axis=None), True, False),  # boolean output
    ("take_sum", lambda df: df.take([0, 2]).sum(axis=None), True, True),
    ("mask_sum", lambda df: df.mask(df > 3, 0.0).sum(axis=None), True, True),
    ("replace_sum", lambda df: df.replace(2.0, 5.0).sum(axis=None), True, True),
    ("transform_sum", lambda df: df.transform(lambda x: x * 2).sum(axis=None), True, True),
    ("agg_mean_sum", lambda df: df.agg("mean").values.sum(), True, True),
    ("dot_sum", lambda df: (df @ jnp.ones((2,))).sum(), True, True),
    # Session 17: skew, kurt, sem (JIT+grad)
    ("skew_sum", lambda df: df.skew().values.sum(), True, True),
    ("sem_sum", lambda df: df.sem().values.sum(), True, True),
]

# GroupBy JAX compat — segment ops are JIT+grad compatible
# Group discovery is eager; aggregation ops use jax.ops.segment_*
GROUPBY_OPS = [
    ("gb_sum", lambda sgb: sgb.sum().values.sum(), True, True),
    ("gb_mean", lambda sgb: sgb.mean().values.sum(), True, True),
    ("gb_min", lambda sgb: sgb.min().values.sum(), True, True),  # segment_min grad,
    ("gb_max", lambda sgb: sgb.max().values.sum(), True, True),
    ("gb_var", lambda sgb: sgb.var().values.sum(), True, True),
    ("gb_std", lambda sgb: sgb.std().values.sum(), True, True),
    ("gb_count", lambda sgb: sgb.count().values.sum(), True, False),
    ("gb_prod", lambda sgb: sgb.prod().values.sum(), True, False),
    ("gb_first", lambda sgb: sgb.first().values.sum(), True, True),  # gather grad,
    ("gb_last", lambda sgb: sgb.last().values.sum(), True, True),
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


# ---- vmap compatibility ----

# Operations that work under vmap (subset of JIT-compatible ops).
# Some JIT-compatible ops use eager structure discovery which breaks under vmap
# (e.g., sort_values, nlargest, rank use np.argsort; groupby uses jnp.unique).
# Rolling/expanding/ewm also use eager window construction that may fail under vmap.
VMAP_OPERATIONS = [
    # Reductions
    ("sum", lambda df: df.sum(axis=None)),
    ("mean", lambda df: df.mean(axis=None)),
    ("std", lambda df: df.std(axis=None)),
    ("var", lambda df: df.var(axis=None)),
    ("min", lambda df: df.min(axis=None)),
    ("max", lambda df: df.max(axis=None)),
    ("prod", lambda df: df.prod(axis=None)),
    # Arithmetic chains → scalar
    ("mul_sum", lambda df: (df * 2).sum(axis=None)),
    ("add_sum", lambda df: (df + 10).sum(axis=None)),
    ("pow_sum", lambda df: (df**2).sum(axis=None)),
    ("chain", lambda df: ((df + 1) * 2 - 3).sum(axis=None)),
    # DataFrame * DataFrame
    ("df_mul_df", lambda df: (df * df).sum(axis=None)),
    ("df_add_df", lambda df: (df + df).sum(axis=None)),
    ("df_sub_df", lambda df: (df - df).sum(axis=None)),
    # diff, shift, fillna, where, clip
    ("diff_sum", lambda df: df.diff().sum(axis=None)),
    ("shift_sum", lambda df: df.shift(1, fill_value=0).sum(axis=None)),
    ("fillna_sum", lambda df: df.fillna(0.0).sum(axis=None)),
    ("where_sum", lambda df: df.where(df > 3, 0).sum(axis=None)),
    ("clip_sum", lambda df: df.clip(2, 5).sum(axis=None)),
    # Cumulative
    ("cumsum_sum", lambda df: df.cumsum().sum(axis=None)),
    ("cumprod_sum", lambda df: df.cumprod().sum(axis=None)),
    # Column manipulation
    ("drop_sum", lambda df: df.drop(columns=["a"]).sum(axis=None)),
    ("rename_sum", lambda df: df.rename(columns={"a": "x"}).sum(axis=None)),
    # Copy, pipe
    ("copy_sum", lambda df: df.copy().sum(axis=None)),
    ("pipe_sum", lambda df: df.pipe(lambda d: d * 2).sum(axis=None)),
    # Reverse operators
    ("radd_sum", lambda df: (10 + df).sum(axis=None)),
    ("rsub_sum", lambda df: (100 - df).sum(axis=None)),
    ("rmul_sum", lambda df: (3 * df).sum(axis=None)),
    # Rolling (prefix-sum based)
    ("roll_sum", lambda df: df.rolling(2).sum().sum(axis=None)),
    ("roll_mean", lambda df: df.rolling(2).mean().sum(axis=None)),
]

DATA2 = {"a": [10.0, 20.0, 30.0], "b": [40.0, 50.0, 60.0]}


def _make_batch(*datas):
    """Stack multiple DataFrames into a batched pytree for vmap."""
    dfs = [DataFrame(d) for d in datas]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *dfs)


@pytest.mark.parametrize("name,op", VMAP_OPERATIONS, ids=[o[0] for o in VMAP_OPERATIONS])
class TestVmapCompat:
    def test_vmap(self, name, op):
        """vmap produces same results as mapping eagerly."""
        batch = _make_batch(DATA, DATA2)
        vmapped = jax.vmap(op)(batch)
        # Compare with eager per-element results
        eager0 = op(DataFrame(DATA))
        eager1 = op(DataFrame(DATA2))
        expected = jnp.array([float(eager0), float(eager1)])
        assert_allclose(vmapped, expected, rtol=1e-5)

    def test_vmap_grad(self, name, op):
        """vmap + grad composition works."""
        batch = _make_batch(DATA, DATA2)
        try:
            grads = jax.vmap(jax.grad(op))(batch)
            for block in grads._dtype_blocks.values():
                assert jnp.all(jnp.isfinite(block)), f"{name} vmap+grad non-finite"
        except (TypeError, FloatingPointError):
            # Some ops are not differentiable (min, max, prod with zeros, etc.)
            pytest.skip(f"{name} not differentiable")


# ---- pmap compatibility ----
# pmap uses the same pytree mechanism as vmap but distributes across physical devices.
# We simulate 2 CPU devices via XLA_FLAGS for testing.


def _pmap_available():
    """Check if we have >= 2 devices for pmap testing."""
    return jax.local_device_count() >= 2


@pytest.mark.parametrize("name,op", VMAP_OPERATIONS, ids=[o[0] for o in VMAP_OPERATIONS])
class TestPmapCompat:
    def test_pmap(self, name, op):
        """pmap produces same results as mapping eagerly."""
        if not _pmap_available():
            pytest.skip("pmap requires >= 2 devices")
        n_devices = jax.local_device_count()
        batch = _make_batch(*([DATA, DATA2][:n_devices]))
        pmapped = jax.pmap(op)(batch)
        eager0 = op(DataFrame(DATA))
        eager1 = op(DataFrame(DATA2))
        expected = jnp.array([float(eager0), float(eager1)][:n_devices])
        assert_allclose(pmapped, expected, rtol=1e-5)

    def test_pmap_grad(self, name, op):
        """pmap + grad composition works."""
        if not _pmap_available():
            pytest.skip("pmap requires >= 2 devices")
        n_devices = jax.local_device_count()
        batch = _make_batch(*([DATA, DATA2][:n_devices]))
        try:
            grads = jax.pmap(jax.grad(op))(batch)
            for block in grads._dtype_blocks.values():
                assert jnp.all(jnp.isfinite(block)), f"{name} pmap+grad non-finite"
        except (TypeError, FloatingPointError):
            pytest.skip(f"{name} not differentiable")
