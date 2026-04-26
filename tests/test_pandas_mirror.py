"""
Pandas-mirrored test suite.

Each test runs the same lambda on both pd.DataFrame and jaxframe.DataFrame,
then compares results. Adding a new test = adding a tuple to a list.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import (
    NUMERIC_2COL,
    NUMERIC_3COL,
    WITH_NANS,
    WITH_NEGATIVES,
    run_equiv,
)

from jaxframe import Series

# ============================
# Arithmetic
# ============================

ARITHMETIC_CASES = [
    ("add_scalar", NUMERIC_2COL, lambda df: df + 10),
    ("sub_scalar", NUMERIC_2COL, lambda df: df - 5),
    ("mul_scalar", NUMERIC_2COL, lambda df: df * 2),
    ("div_scalar", NUMERIC_2COL, lambda df: df / 3),
    ("floordiv_scalar", NUMERIC_2COL, lambda df: df // 3),
    ("mod_scalar", NUMERIC_2COL, lambda df: df % 7),
    ("pow_scalar", NUMERIC_2COL, lambda df: df**2),
    ("add_df", NUMERIC_2COL, lambda df: df + df),
    ("mul_df", NUMERIC_2COL, lambda df: df * df),
    ("chain_arith", NUMERIC_2COL, lambda df: (df + 10) * 2 - 5),
    ("negatives_add", WITH_NEGATIVES, lambda df: df + 100),
    ("negatives_abs", WITH_NEGATIVES, lambda df: df.abs()),
]


@pytest.mark.parametrize("name,data,op", ARITHMETIC_CASES, ids=[c[0] for c in ARITHMETIC_CASES])
def test_arithmetic(name, data, op):
    run_equiv(data, op)


# ============================
# Comparisons
# ============================

COMPARISON_CASES = [
    ("gt_scalar", NUMERIC_2COL, lambda df: df > 3),
    ("ge_scalar", NUMERIC_2COL, lambda df: df >= 3),
    ("lt_scalar", NUMERIC_2COL, lambda df: df < 30),
    ("le_scalar", NUMERIC_2COL, lambda df: df <= 30),
    ("eq_scalar", NUMERIC_2COL, lambda df: df == 3),
    ("ne_scalar", NUMERIC_2COL, lambda df: df != 3),
]


@pytest.mark.parametrize("name,data,op", COMPARISON_CASES, ids=[c[0] for c in COMPARISON_CASES])
def test_comparison(name, data, op):
    run_equiv(data, op)


# ============================
# Reductions
# ============================

REDUCTION_CASES = [
    ("sum_axis0", NUMERIC_3COL, lambda df: df.sum(axis=0)),
    ("sum_axis1", NUMERIC_3COL, lambda df: df.sum(axis=1)),
    ("mean_axis0", NUMERIC_3COL, lambda df: df.mean(axis=0)),
    ("mean_axis1", NUMERIC_3COL, lambda df: df.mean(axis=1)),
    ("std_axis0", NUMERIC_3COL, lambda df: df.std(axis=0)),
    ("var_axis0", NUMERIC_3COL, lambda df: df.var(axis=0)),
    ("min_axis0", NUMERIC_3COL, lambda df: df.min(axis=0)),
    ("max_axis0", NUMERIC_3COL, lambda df: df.max(axis=0)),
    ("prod_axis0", NUMERIC_3COL, lambda df: df.prod(axis=0)),
]


@pytest.mark.parametrize("name,data,op", REDUCTION_CASES, ids=[c[0] for c in REDUCTION_CASES])
def test_reduction(name, data, op):
    run_equiv(data, op)


# ============================
# Indexing
# ============================

INDEXING_CASES = [
    ("head_3", NUMERIC_2COL, lambda df: df.head(3)),
    ("tail_2", NUMERIC_2COL, lambda df: df.tail(2)),
    ("single_col", NUMERIC_2COL, lambda df: df["a"]),
    ("multi_col", NUMERIC_3COL, lambda df: df[["a", "c"]]),
]


@pytest.mark.parametrize("name,data,op", INDEXING_CASES, ids=[c[0] for c in INDEXING_CASES])
def test_indexing(name, data, op):
    run_equiv(data, op)


# ============================
# Time series
# ============================

TIME_SERIES_CASES = [
    ("shift_1", NUMERIC_2COL, lambda df: df.shift(1)),
    ("shift_neg1", NUMERIC_2COL, lambda df: df.shift(-1)),
    ("diff_1", NUMERIC_2COL, lambda df: df.diff(1)),
    ("pct_change", NUMERIC_2COL, lambda df: df.pct_change(1)),
]


@pytest.mark.parametrize("name,data,op", TIME_SERIES_CASES, ids=[c[0] for c in TIME_SERIES_CASES])
def test_time_series(name, data, op):
    run_equiv(data, op)


# ============================
# Masking
# ============================

MASKING_CASES = [
    ("clip_both", NUMERIC_2COL, lambda df: df.clip(10, 40)),
    ("clip_lower", NUMERIC_2COL, lambda df: df.clip(lower=10)),
    ("clip_upper", NUMERIC_2COL, lambda df: df.clip(upper=40)),
]


@pytest.mark.parametrize("name,data,op", MASKING_CASES, ids=[c[0] for c in MASKING_CASES])
def test_masking(name, data, op):
    run_equiv(data, op)


# ============================
# Statistical
# ============================

STAT_CASES = [
    ("corr", NUMERIC_3COL, lambda df: df.corr()),
    ("cov", NUMERIC_3COL, lambda df: df.cov()),
    ("transpose", NUMERIC_3COL, lambda df: df.T),
]


@pytest.mark.parametrize("name,data,op", STAT_CASES, ids=[c[0] for c in STAT_CASES])
def test_statistical(name, data, op):
    run_equiv(data, op)


# ============================
# Missing data (Session 1)
# ============================

MISSING_DATA_CASES = [
    ("isna", WITH_NANS, lambda df: df.isna()),
    ("isnull", WITH_NANS, lambda df: df.isnull()),
    ("notna", WITH_NANS, lambda df: df.notna()),
    ("notnull", WITH_NANS, lambda df: df.notnull()),
    ("fillna_scalar", WITH_NANS, lambda df: df.fillna(0.0)),
    ("fillna_neg1", WITH_NANS, lambda df: df.fillna(-1.0)),
    ("isna_no_nans", NUMERIC_2COL, lambda df: df.isna()),
    ("fillna_no_nans", NUMERIC_2COL, lambda df: df.fillna(0.0)),
]


@pytest.mark.parametrize("name,data,op", MISSING_DATA_CASES, ids=[c[0] for c in MISSING_DATA_CASES])
def test_missing_data(name, data, op):
    run_equiv(data, op)


# ============================
# Cumulative & descriptive (Session 2)
# ============================

CUMULATIVE_CASES = [
    ("cumsum_axis0", NUMERIC_2COL, lambda df: df.cumsum()),
    ("cumsum_axis1", NUMERIC_3COL, lambda df: df.cumsum(axis=1)),
    ("cumprod_axis0", NUMERIC_2COL, lambda df: df.cumprod()),
    ("cumprod_axis1", NUMERIC_3COL, lambda df: df.cumprod(axis=1)),
]


@pytest.mark.parametrize("name,data,op", CUMULATIVE_CASES, ids=[c[0] for c in CUMULATIVE_CASES])
def test_cumulative(name, data, op):
    run_equiv(data, op)


DESCRIPTIVE_CASES = [
    ("count_axis0", NUMERIC_2COL, lambda df: df.count()),
    ("count_with_nans", WITH_NANS, lambda df: df.count()),
    ("median_axis0", NUMERIC_3COL, lambda df: df.median()),
    ("median_axis1", NUMERIC_3COL, lambda df: df.median(axis=1)),
]


@pytest.mark.parametrize("name,data,op", DESCRIPTIVE_CASES, ids=[c[0] for c in DESCRIPTIVE_CASES])
def test_descriptive(name, data, op):
    run_equiv(data, op)


# ============================
# Series operators (Session 1)
# ============================


class TestSeriesArithmetic:
    """Test that Series arithmetic matches pandas Series."""

    def _compare(self, pd_series, jf_series, rtol=1e-5):
        np.testing.assert_allclose(
            np.asarray(jf_series.values),
            pd_series.values.astype(np.float32),
            rtol=rtol,
        )

    def test_add_scalar(self):
        ps = pd.Series([1.0, 2.0, 3.0])
        js = Series([1.0, 2.0, 3.0])
        self._compare(ps + 10, js + 10)

    def test_sub_scalar(self):
        ps = pd.Series([1.0, 2.0, 3.0])
        js = Series([1.0, 2.0, 3.0])
        self._compare(ps - 1, js - 1)

    def test_mul_scalar(self):
        ps = pd.Series([1.0, 2.0, 3.0])
        js = Series([1.0, 2.0, 3.0])
        self._compare(ps * 3, js * 3)

    def test_truediv_scalar(self):
        ps = pd.Series([10.0, 20.0, 30.0])
        js = Series([10.0, 20.0, 30.0])
        self._compare(ps / 5, js / 5)

    def test_floordiv_scalar(self):
        ps = pd.Series([10.0, 21.0, 35.0])
        js = Series([10.0, 21.0, 35.0])
        self._compare(ps // 7, js // 7)

    def test_mod_scalar(self):
        ps = pd.Series([10.0, 21.0, 35.0])
        js = Series([10.0, 21.0, 35.0])
        self._compare(ps % 7, js % 7)

    def test_pow_scalar(self):
        ps = pd.Series([1.0, 2.0, 3.0])
        js = Series([1.0, 2.0, 3.0])
        self._compare(ps**2, js**2)

    def test_neg(self):
        ps = pd.Series([1.0, -2.0, 3.0])
        js = Series([1.0, -2.0, 3.0])
        self._compare(-ps, -js)

    def test_add_series(self):
        ps = pd.Series([1.0, 2.0, 3.0])
        js = Series([1.0, 2.0, 3.0])
        self._compare(ps + ps, js + js)

    def test_gt(self):
        ps = pd.Series([1.0, 2.0, 3.0])
        js = Series([1.0, 2.0, 3.0])
        pd_result = ps > 2
        jf_result = js > 2
        np.testing.assert_array_equal(np.asarray(jf_result.values), pd_result.values)

    def test_lt(self):
        ps = pd.Series([1.0, 2.0, 3.0])
        js = Series([1.0, 2.0, 3.0])
        pd_result = ps < 2
        jf_result = js < 2
        np.testing.assert_array_equal(np.asarray(jf_result.values), pd_result.values)

    def test_ge(self):
        ps = pd.Series([1.0, 2.0, 3.0])
        js = Series([1.0, 2.0, 3.0])
        pd_result = ps >= 2
        jf_result = js >= 2
        np.testing.assert_array_equal(np.asarray(jf_result.values), pd_result.values)

    def test_le(self):
        ps = pd.Series([1.0, 2.0, 3.0])
        js = Series([1.0, 2.0, 3.0])
        pd_result = ps <= 2
        jf_result = js <= 2
        np.testing.assert_array_equal(np.asarray(jf_result.values), pd_result.values)

    def test_eq(self):
        ps = pd.Series([1.0, 2.0, 3.0])
        js = Series([1.0, 2.0, 3.0])
        pd_result = ps == 2
        jf_result = js == 2
        np.testing.assert_array_equal(np.asarray(jf_result.values), pd_result.values)

    def test_ne(self):
        ps = pd.Series([1.0, 2.0, 3.0])
        js = Series([1.0, 2.0, 3.0])
        pd_result = ps != 2
        jf_result = js != 2
        np.testing.assert_array_equal(np.asarray(jf_result.values), pd_result.values)

    def test_abs(self):
        ps = pd.Series([-1.0, 2.0, -3.0])
        js = Series([-1.0, 2.0, -3.0])
        self._compare(ps.abs(), js.abs())


# ============================
# Pandas interop + copy (Session 3)
# ============================


class TestPandasInterop:
    """Test to_pandas, from_pandas, to_numpy, copy."""

    def test_to_pandas_roundtrip(self):
        """DataFrame -> to_pandas() -> from_pandas() preserves data."""
        from jaxframe import DataFrame

        jdf = DataFrame(NUMERIC_3COL)
        pdf = jdf.to_pandas()
        assert isinstance(pdf, pd.DataFrame)
        assert list(pdf.columns) == ["a", "b", "c"]
        np.testing.assert_allclose(pdf.values, np.array(jdf.values), rtol=1e-5)

    def test_from_pandas(self):
        """pd.DataFrame -> from_pandas() -> same values."""
        from jaxframe import DataFrame

        pdf = pd.DataFrame(NUMERIC_2COL)
        jdf = DataFrame.from_pandas(pdf)
        assert jdf.shape == pdf.shape
        assert list(jdf.columns) == list(pdf.columns)
        np.testing.assert_allclose(np.array(jdf.values), pdf.values.astype(np.float32), rtol=1e-5)

    def test_from_pandas_with_nans(self):
        """NaN values preserved through from_pandas."""
        from jaxframe import DataFrame

        pdf = pd.DataFrame(WITH_NANS)
        jdf = DataFrame.from_pandas(pdf)
        np.testing.assert_array_equal(
            np.isnan(np.array(jdf.values)),
            np.isnan(pdf.values.astype(np.float32)),
        )

    def test_to_numpy(self):
        """to_numpy() returns plain numpy array."""
        from jaxframe import DataFrame

        jdf = DataFrame(NUMERIC_2COL)
        arr = jdf.to_numpy()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, np.array(jdf.values), rtol=1e-5)

    def test_copy_independent(self):
        """copy() creates an independent DataFrame."""
        from jaxframe import DataFrame

        jdf = DataFrame(NUMERIC_2COL)
        jdf2 = jdf.copy()
        assert jdf.shape == jdf2.shape
        np.testing.assert_allclose(np.array(jdf.values), np.array(jdf2.values), rtol=1e-5)
        # Verify they don't share column_order list
        assert jdf._column_order is not jdf2._column_order


# ============================
# Sorting (Session 4)
# ============================

SORT_CASES = [
    ("sort_by_a_asc", NUMERIC_2COL, lambda df: df.sort_values("a")),
    ("sort_by_a_desc", NUMERIC_2COL, lambda df: df.sort_values("a", ascending=False)),
    ("sort_by_b", NUMERIC_3COL, lambda df: df.sort_values("b")),
    (
        "sort_negatives",
        WITH_NEGATIVES,
        lambda df: df.sort_values("a"),
    ),
]


@pytest.mark.parametrize("name,data,op", SORT_CASES, ids=[c[0] for c in SORT_CASES])
def test_sorting(name, data, op):
    run_equiv(data, op)


# ============================
# Describe & quantile (Session 5)
# ============================

QUANTILE_CASES = [
    ("quantile_50", NUMERIC_3COL, lambda df: df.quantile(0.5)),
    ("quantile_25", NUMERIC_3COL, lambda df: df.quantile(0.25)),
    ("quantile_75", NUMERIC_3COL, lambda df: df.quantile(0.75)),
]


@pytest.mark.parametrize("name,data,op", QUANTILE_CASES, ids=[c[0] for c in QUANTILE_CASES])
def test_quantile(name, data, op):
    run_equiv(data, op)


NLARGEST_CASES = [
    ("nlargest_2_a", NUMERIC_2COL, lambda df: df.nlargest(2, "a")),
    ("nsmallest_2_a", NUMERIC_2COL, lambda df: df.nsmallest(2, "a")),
    ("nlargest_3_b", NUMERIC_3COL, lambda df: df.nlargest(3, "b")),
]


@pytest.mark.parametrize("name,data,op", NLARGEST_CASES, ids=[c[0] for c in NLARGEST_CASES])
def test_nlargest(name, data, op):
    run_equiv(data, op)


# ============================
# Column manipulation (Session 6)
# ============================

COLUMN_CASES = [
    ("drop_col", NUMERIC_3COL, lambda df: df.drop(columns=["b"])),
    ("drop_multi", NUMERIC_3COL, lambda df: df.drop(columns=["a", "c"])),
    ("rename_cols", NUMERIC_2COL, lambda df: df.rename(columns={"a": "x", "b": "y"})),
    ("assign_new", NUMERIC_2COL, lambda df: df.assign(c=df["a"] + df["b"])),
]


@pytest.mark.parametrize("name,data,op", COLUMN_CASES, ids=[c[0] for c in COLUMN_CASES])
def test_column_ops(name, data, op):
    run_equiv(data, op)


# ============================
# Apply (Session 7)
# ============================

APPLY_CASES = [
    ("apply_sum_ax0", NUMERIC_3COL, lambda df: df.apply(np.sum, axis=0)),
    ("apply_mean_ax0", NUMERIC_3COL, lambda df: df.apply(np.mean, axis=0)),
    ("apply_sum_ax1", NUMERIC_3COL, lambda df: df.apply(np.sum, axis=1)),
]


@pytest.mark.parametrize("name,data,op", APPLY_CASES, ids=[c[0] for c in APPLY_CASES])
def test_apply(name, data, op):
    run_equiv(data, op)


# ============================
# Reverse operators (Session 8)
# ============================

REVERSE_OP_CASES = [
    ("radd", NUMERIC_2COL, lambda df: 10 + df),
    ("rsub", NUMERIC_2COL, lambda df: 100 - df),
    ("rmul", NUMERIC_2COL, lambda df: 3 * df),
    ("rtruediv", NUMERIC_2COL, lambda df: 100.0 / df),
    ("rfloordiv", NUMERIC_2COL, lambda df: 100 // df),
    ("rmod", NUMERIC_2COL, lambda df: 100 % df),
    ("rpow", NUMERIC_2COL, lambda df: 2**df),
]


@pytest.mark.parametrize("name,data,op", REVERSE_OP_CASES, ids=[c[0] for c in REVERSE_OP_CASES])
def test_reverse_ops(name, data, op):
    run_equiv(data, op)
