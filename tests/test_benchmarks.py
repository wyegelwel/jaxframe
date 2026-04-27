"""
Smoke tests for benchmark infrastructure.

For full benchmarks: uv run python benchmarks/run.py
For report generation: uv run python benchmarks/report.py
"""

import jax
import numpy as np
import pandas as pd

from jaxframe import DataFrame


def make_data(n_rows=100, n_cols=5, seed=42):
    rng = np.random.default_rng(seed)
    return {f"col_{i}": rng.standard_normal(n_rows) for i in range(n_cols)}


class TestBenchmarkSmoke:
    """Quick sanity checks that benchmark operations work correctly."""

    def test_scalar_reductions_eager(self):
        data = make_data()
        jdf = DataFrame(data)
        for method in ["sum", "mean", "std", "var", "min", "max", "prod"]:
            result = getattr(jdf, method)(axis=None)
            assert np.isfinite(float(result)), f"{method} returned non-finite"

    def test_scalar_reductions_jit(self):
        data = make_data()
        jdf = DataFrame(data)
        fn = jax.jit(lambda df: df.sum(axis=None))
        result = fn(jdf)
        assert np.isfinite(float(result))

    def test_arithmetic_chain_jit(self):
        data = make_data()
        jdf = DataFrame(data)
        fn = jax.jit(lambda df: ((df + 1) * 2 - 3).sum(axis=None))
        result = fn(jdf)
        assert np.isfinite(float(result))

    def test_column_reductions(self):
        data = make_data()
        jdf = DataFrame(data)
        for method in ["sum", "mean", "std"]:
            result = getattr(jdf, method)()
            assert result.values.shape[0] == 5

    def test_cumulative(self):
        data = make_data()
        jdf = DataFrame(data)
        assert jdf.cumsum().shape == (100, 5)
        assert jdf.cumprod().shape == (100, 5)

    def test_shift_diff(self):
        data = make_data()
        jdf = DataFrame(data)
        assert jdf.diff().shape == (100, 5)
        assert jdf.shift(1).shape == (100, 5)

    def test_rolling(self):
        data = make_data()
        jdf = DataFrame(data)
        assert jdf.rolling(10).sum().shape == (100, 5)

    def test_expanding(self):
        data = make_data()
        jdf = DataFrame(data)
        assert jdf.expanding().sum().shape == (100, 5)

    def test_ewm(self):
        data = make_data()
        jdf = DataFrame(data)
        assert jdf.ewm(span=10).mean().shape == (100, 5)

    def test_groupby(self):
        rng = np.random.default_rng(42)
        keys = rng.integers(0, 3, size=100)
        vals = rng.standard_normal(100)
        jdf = DataFrame({"key": keys, "val": vals})
        result = jdf.groupby("key")["val"].sum()
        assert float(result.values.sum()) != 0.0

    def test_data_cleaning(self):
        data = make_data()
        jdf = DataFrame(data)
        assert jdf.fillna(0).shape == (100, 5)
        assert jdf.clip(-1, 1).shape == (100, 5)

    def test_sorting(self):
        data = make_data()
        jdf = DataFrame(data)
        assert jdf.sort_values("col_0").shape == (100, 5)
        assert jdf.rank().shape == (100, 5)

    def test_overhead_raw_vs_jaxframe(self):
        """Ensure jaxframe JIT gives same result as raw jnp."""
        import jax.numpy as jnp

        data = make_data()
        raw = jnp.array(np.column_stack(list(data.values())), dtype=jnp.float32)
        jdf = DataFrame(data)

        raw_fn = jax.jit(lambda x: ((x + 1) * 2 - 3).sum())
        jf_fn = jax.jit(lambda df: ((df + 1) * 2 - 3).sum(axis=None))

        raw_result = float(raw_fn(raw))
        jf_result = float(jf_fn(jdf))
        np.testing.assert_allclose(jf_result, raw_result, rtol=1e-4)

    def test_pandas_agreement(self):
        """Spot-check that jaxframe matches pandas for key ops."""
        data = make_data()
        pdf = pd.DataFrame(data)
        jdf = DataFrame(data)

        np.testing.assert_allclose(
            float(jdf.sum(axis=None)),
            pdf.values.sum(),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(jdf.mean(axis=None)),
            pdf.values.mean(),
            rtol=1e-5,
        )
