"""Regression guards for window-op performance and large-data correctness.

The expanding/EWM kernels were once lax.scan-based: sequential, ~30 seconds at
1M rows on GPU (~200x slower than pandas). The parallel rewrites (cumsum /
associative_scan) must stay orders of magnitude below that. These bounds are
deliberately loose — they catch a return to sequential scanning, not noise.
"""

import time

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from jaxframe import DataFrame

N = 200_000


def _timed(fn, repeats=3):
    fn().block_until_ready()  # warm compile
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn().block_until_ready()
    return (time.perf_counter() - t0) / repeats


def _big_frame():
    rng = np.random.default_rng(0)
    return DataFrame({f"c{i}": rng.standard_normal(N) for i in range(5)})


class TestWindowPerfRegression:
    def test_expanding_sum_not_sequential(self):
        jdf = _big_frame()
        assert _timed(lambda: jdf.expanding().sum()) < 1.0

    def test_expanding_mean_not_sequential(self):
        jdf = _big_frame()
        assert _timed(lambda: jdf.expanding().mean()) < 1.0

    def test_expanding_var_not_sequential(self):
        jdf = _big_frame()
        assert _timed(lambda: jdf.expanding().var()) < 1.0

    def test_ewm_mean_not_sequential(self):
        jdf = _big_frame()
        assert _timed(lambda: jdf.ewm(alpha=0.3).mean()) < 1.0


class TestWindowLargeDataEquivalence:
    """Pandas equivalence at 10k rows with 5% NaNs (parity tests use tiny data)."""

    def _data(self):
        rng = np.random.default_rng(1)
        d = rng.standard_normal(10_000)
        d[rng.random(10_000) < 0.05] = np.nan
        return d

    def _pair(self):
        d = self._data()
        return pd.DataFrame({"x": d}), DataFrame({"x": d})

    def test_expanding_sum(self):
        p, j = self._pair()
        assert_allclose(
            np.asarray(j.expanding().sum().values).ravel(),
            p.expanding().sum().values.ravel(),
            rtol=2e-3,
            atol=1e-4,
            equal_nan=True,
        )

    def test_expanding_mean(self):
        p, j = self._pair()
        assert_allclose(
            np.asarray(j.expanding().mean().values).ravel(),
            p.expanding().mean().values.ravel(),
            rtol=2e-3,
            atol=1e-4,
            equal_nan=True,
        )

    def test_expanding_var(self):
        p, j = self._pair()
        assert_allclose(
            np.asarray(j.expanding().var().values).ravel(),
            p.expanding().var().values.ravel(),
            rtol=5e-3,
            atol=1e-4,
            equal_nan=True,
        )

    def test_ewm_mean(self):
        p, j = self._pair()
        assert_allclose(
            np.asarray(j.ewm(alpha=0.3).mean().values).ravel(),
            p.ewm(alpha=0.3).mean().values.ravel(),
            rtol=2e-3,
            atol=1e-4,
            equal_nan=True,
        )
