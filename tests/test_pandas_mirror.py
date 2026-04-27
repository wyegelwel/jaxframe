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


# ============================
# GroupBy (Sessions 9-11)
# ============================

GROUPBY_DATA = {
    "key": [1.0, 1.0, 2.0, 2.0, 3.0],
    "val": [10.0, 20.0, 30.0, 40.0, 50.0],
}
GROUPBY_MULTI = {
    "key": [1.0, 1.0, 2.0, 2.0],
    "a": [10.0, 20.0, 30.0, 40.0],
    "b": [100.0, 200.0, 300.0, 400.0],
}
GROUPBY_STD = {
    "key": [1.0, 1.0, 2.0, 2.0, 2.0],
    "val": [10.0, 20.0, 30.0, 40.0, 50.0],
}


class TestGroupBy:
    """Test groupby operations match pandas."""

    def _compare_series(self, pd_result, jf_result, rtol=1e-5):
        """Compare pandas Series vs jaxframe Series, sorted by index."""
        pd_sorted = pd_result.sort_index()
        jf_idx = np.asarray(jf_result.index)
        jf_vals = np.asarray(jf_result.values)
        sort_order = np.argsort(jf_idx)
        np.testing.assert_allclose(
            jf_vals[sort_order].astype(np.float32),
            pd_sorted.values.astype(np.float32),
            rtol=rtol,
        )

    def _compare_df(self, pd_result, jf_result, rtol=1e-5):
        """Compare pandas DataFrame vs jaxframe DataFrame, sorted by index."""
        pd_sorted = pd_result.sort_index()
        jf_pandas = jf_result.to_pandas().sort_index()
        np.testing.assert_allclose(
            jf_pandas.values.astype(np.float32),
            pd_sorted.values.astype(np.float32),
            rtol=rtol,
        )

    def test_sum(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_DATA), DataFrame(GROUPBY_DATA)
        self._compare_series(
            pdf.groupby("key")["val"].sum(),
            jdf.groupby("key")["val"].sum(),
        )

    def test_mean(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_DATA), DataFrame(GROUPBY_DATA)
        self._compare_series(
            pdf.groupby("key")["val"].mean(),
            jdf.groupby("key")["val"].mean(),
        )

    def test_count(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_DATA), DataFrame(GROUPBY_DATA)
        self._compare_series(
            pdf.groupby("key")["val"].count(),
            jdf.groupby("key")["val"].count(),
        )

    def test_min(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_DATA), DataFrame(GROUPBY_DATA)
        self._compare_series(
            pdf.groupby("key")["val"].min(),
            jdf.groupby("key")["val"].min(),
        )

    def test_max(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_DATA), DataFrame(GROUPBY_DATA)
        self._compare_series(
            pdf.groupby("key")["val"].max(),
            jdf.groupby("key")["val"].max(),
        )

    def test_std(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_STD), DataFrame(GROUPBY_STD)
        self._compare_series(
            pdf.groupby("key")["val"].std(),
            jdf.groupby("key")["val"].std(),
        )

    def test_var(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_STD), DataFrame(GROUPBY_STD)
        self._compare_series(
            pdf.groupby("key")["val"].var(),
            jdf.groupby("key")["val"].var(),
        )

    def test_df_sum(self):
        """GroupBy on full DataFrame."""
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_MULTI), DataFrame(GROUPBY_MULTI)
        self._compare_df(
            pdf.groupby("key").sum(),
            jdf.groupby("key").sum(),
        )

    def test_df_mean(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_MULTI), DataFrame(GROUPBY_MULTI)
        self._compare_df(
            pdf.groupby("key").mean(),
            jdf.groupby("key").mean(),
        )

    def test_df_min(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_MULTI), DataFrame(GROUPBY_MULTI)
        self._compare_df(
            pdf.groupby("key").min(),
            jdf.groupby("key").min(),
        )

    def test_df_max(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_MULTI), DataFrame(GROUPBY_MULTI)
        self._compare_df(
            pdf.groupby("key").max(),
            jdf.groupby("key").max(),
        )

    def test_df_std(self):
        from jaxframe import DataFrame

        data = {
            "key": [1.0, 1.0, 2.0, 2.0, 2.0],
            "a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "b": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
        pdf, jdf = pd.DataFrame(data), DataFrame(data)
        self._compare_df(
            pdf.groupby("key").std(),
            jdf.groupby("key").std(),
        )

    def test_df_var(self):
        from jaxframe import DataFrame

        data = {
            "key": [1.0, 1.0, 2.0, 2.0, 2.0],
            "a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "b": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
        pdf, jdf = pd.DataFrame(data), DataFrame(data)
        self._compare_df(
            pdf.groupby("key").var(),
            jdf.groupby("key").var(),
        )

    def test_df_prod(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_MULTI), DataFrame(GROUPBY_MULTI)
        self._compare_df(
            pdf.groupby("key").prod(),
            jdf.groupby("key").prod(),
        )

    def test_df_first(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_MULTI), DataFrame(GROUPBY_MULTI)
        self._compare_df(
            pdf.groupby("key").first(),
            jdf.groupby("key").first(),
        )

    def test_df_last(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_MULTI), DataFrame(GROUPBY_MULTI)
        self._compare_df(
            pdf.groupby("key").last(),
            jdf.groupby("key").last(),
        )

    def test_series_prod(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_DATA), DataFrame(GROUPBY_DATA)
        self._compare_series(
            pdf.groupby("key")["val"].prod(),
            jdf.groupby("key")["val"].prod(),
        )

    def test_series_first(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_DATA), DataFrame(GROUPBY_DATA)
        self._compare_series(
            pdf.groupby("key")["val"].first(),
            jdf.groupby("key")["val"].first(),
        )

    def test_series_last(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_DATA), DataFrame(GROUPBY_DATA)
        self._compare_series(
            pdf.groupby("key")["val"].last(),
            jdf.groupby("key")["val"].last(),
        )

    def test_series_agg_single(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_DATA), DataFrame(GROUPBY_DATA)
        self._compare_series(
            pdf.groupby("key")["val"].agg("sum"),
            jdf.groupby("key")["val"].agg("sum"),
        )

    def test_series_agg_multi(self):
        """agg with multiple functions returns DataFrame."""
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_DATA), DataFrame(GROUPBY_DATA)
        pd_result = pdf.groupby("key")["val"].agg(["sum", "mean"])
        jf_result = jdf.groupby("key")["val"].agg(["sum", "mean"])
        pd_sorted = pd_result.sort_index()
        jf_pandas = jf_result.to_pandas().sort_index()
        np.testing.assert_allclose(
            jf_pandas.values.astype(np.float32),
            pd_sorted.values.astype(np.float32),
            rtol=1e-5,
        )

    def test_df_agg_single(self):
        from jaxframe import DataFrame

        pdf, jdf = pd.DataFrame(GROUPBY_MULTI), DataFrame(GROUPBY_MULTI)
        self._compare_df(
            pdf.groupby("key").agg("sum"),
            jdf.groupby("key").agg("sum"),
        )

    def test_transform_sum(self):
        """transform broadcasts group result back to original shape."""
        from jaxframe import DataFrame

        data = {"key": [1.0, 1.0, 2.0, 2.0], "val": [10.0, 20.0, 30.0, 40.0]}
        pdf, jdf = pd.DataFrame(data), DataFrame(data)
        pd_result = pdf.groupby("key")["val"].transform("sum")
        jf_result = jdf.groupby("key")["val"].transform("sum")
        np.testing.assert_allclose(
            np.asarray(jf_result.values).astype(np.float32),
            pd_result.values.astype(np.float32),
            rtol=1e-5,
        )

    def test_transform_mean(self):
        from jaxframe import DataFrame

        data = {"key": [1.0, 1.0, 2.0, 2.0], "val": [10.0, 20.0, 30.0, 40.0]}
        pdf, jdf = pd.DataFrame(data), DataFrame(data)
        pd_result = pdf.groupby("key")["val"].transform("mean")
        jf_result = jdf.groupby("key")["val"].transform("mean")
        np.testing.assert_allclose(
            np.asarray(jf_result.values).astype(np.float32),
            pd_result.values.astype(np.float32),
            rtol=1e-5,
        )


# ============================
# DateTime / .dt accessor (Sessions 12-13)
# ============================


class TestDatetime:
    """Test datetime index and .dt accessor."""

    def test_datetime_index_creation(self):
        """Create DataFrame with datetime index from strings."""
        from jaxframe import DataFrame

        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        data = {"val": [1.0, 2.0, 3.0]}
        pdf = pd.DataFrame(data, index=dates)
        jdf = DataFrame(data, index=dates)
        assert jdf.shape == pdf.shape
        np.testing.assert_allclose(np.array(jdf.values), pdf.values.astype(np.float32), rtol=1e-5)

    def test_dt_year(self):
        from jaxframe import Series

        dates = pd.date_range("2024-01-15", periods=4, freq="ME")
        ps = pd.Series(dates)
        js = Series(dates)
        np.testing.assert_array_equal(np.asarray(js.dt.year), ps.dt.year.values)

    def test_dt_month(self):
        from jaxframe import Series

        dates = pd.date_range("2024-01-15", periods=4, freq="ME")
        ps = pd.Series(dates)
        js = Series(dates)
        np.testing.assert_array_equal(np.asarray(js.dt.month), ps.dt.month.values)

    def test_dt_day(self):
        from jaxframe import Series

        dates = pd.date_range("2024-01-15", periods=4, freq="ME")
        ps = pd.Series(dates)
        js = Series(dates)
        np.testing.assert_array_equal(np.asarray(js.dt.day), ps.dt.day.values)

    def test_dt_hour(self):
        from jaxframe import Series

        dates = pd.date_range("2024-01-01 08:30:00", periods=3, freq="h")
        ps = pd.Series(dates)
        js = Series(dates)
        np.testing.assert_array_equal(np.asarray(js.dt.hour), ps.dt.hour.values)

    def test_dt_minute(self):
        from jaxframe import Series

        dates = pd.date_range("2024-01-01 08:30:00", periods=3, freq="h")
        ps = pd.Series(dates)
        js = Series(dates)
        np.testing.assert_array_equal(np.asarray(js.dt.minute), ps.dt.minute.values)

    def test_dt_second(self):
        from jaxframe import Series

        dates = pd.date_range("2024-01-01 08:30:45", periods=3, freq="h")
        ps = pd.Series(dates)
        js = Series(dates)
        np.testing.assert_array_equal(np.asarray(js.dt.second), ps.dt.second.values)

    def test_dt_dayofweek(self):
        from jaxframe import Series

        dates = pd.date_range("2024-01-01", periods=7, freq="D")
        ps = pd.Series(dates)
        js = Series(dates)
        np.testing.assert_array_equal(np.asarray(js.dt.dayofweek), ps.dt.dayofweek.values)

    def test_dt_date(self):
        """dt.date returns date portion as datetime64[D]."""
        from jaxframe import Series

        dates = pd.date_range("2024-01-01 08:30:00", periods=3, freq="D")
        ps = pd.Series(dates)
        js = Series(dates)
        # Compare as strings since date types may differ
        pd_dates = ps.dt.date.values.astype("datetime64[D]")
        jf_dates = np.asarray(js.dt.date).astype("datetime64[D]")
        np.testing.assert_array_equal(jf_dates, pd_dates)


# ============================
# .str accessor (Session 14)
# ============================

STR_DATA = ["Hello", "  World  ", "PYTHON", "foo bar", "test123"]


class TestStrAccessor:
    """Test .str accessor on Series with string data."""

    def _compare(self, pd_result, jf_result):
        np.testing.assert_array_equal(np.asarray(jf_result), pd_result.values)

    def test_lower(self):
        from jaxframe import Series

        ps = pd.Series(STR_DATA)
        js = Series(STR_DATA)
        self._compare(ps.str.lower(), js.str.lower())

    def test_upper(self):
        from jaxframe import Series

        ps = pd.Series(STR_DATA)
        js = Series(STR_DATA)
        self._compare(ps.str.upper(), js.str.upper())

    def test_strip(self):
        from jaxframe import Series

        ps = pd.Series(STR_DATA)
        js = Series(STR_DATA)
        self._compare(ps.str.strip(), js.str.strip())

    def test_lstrip(self):
        from jaxframe import Series

        ps = pd.Series(STR_DATA)
        js = Series(STR_DATA)
        self._compare(ps.str.lstrip(), js.str.lstrip())

    def test_rstrip(self):
        from jaxframe import Series

        ps = pd.Series(STR_DATA)
        js = Series(STR_DATA)
        self._compare(ps.str.rstrip(), js.str.rstrip())

    def test_len(self):
        from jaxframe import Series

        ps = pd.Series(STR_DATA)
        js = Series(STR_DATA)
        np.testing.assert_array_equal(np.asarray(js.str.len()), ps.str.len().values)

    def test_contains(self):
        from jaxframe import Series

        ps = pd.Series(STR_DATA)
        js = Series(STR_DATA)
        np.testing.assert_array_equal(
            np.asarray(js.str.contains("o")), ps.str.contains("o", regex=False).values
        )

    def test_startswith(self):
        from jaxframe import Series

        ps = pd.Series(STR_DATA)
        js = Series(STR_DATA)
        np.testing.assert_array_equal(
            np.asarray(js.str.startswith("H")), ps.str.startswith("H").values
        )

    def test_endswith(self):
        from jaxframe import Series

        ps = pd.Series(STR_DATA)
        js = Series(STR_DATA)
        np.testing.assert_array_equal(np.asarray(js.str.endswith("3")), ps.str.endswith("3").values)

    def test_replace(self):
        from jaxframe import Series

        ps = pd.Series(STR_DATA)
        js = Series(STR_DATA)
        self._compare(
            ps.str.replace("o", "0", regex=False),
            js.str.replace("o", "0"),
        )

    def test_split(self):
        from jaxframe import Series

        data = ["a-b", "c-d-e", "f"]
        ps = pd.Series(data)
        js = Series(data)
        # Compare element by element since split returns lists
        pd_result = ps.str.split("-")
        jf_result = js.str.split("-")
        for p, j in zip(pd_result.values, jf_result):
            assert p == j


# ============================
# Session 15: all, any, round, idxmin, idxmax, isin
# ============================

WITH_ZEROS = {"a": [0.0, 1.0, 2.0], "b": [1.0, 1.0, 1.0]}
ALL_ZEROS = {"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 1.0]}

ALL_ANY_CASES = [
    ("all_axis0", NUMERIC_2COL, lambda df: df.all(axis=0)),
    ("any_axis0", NUMERIC_2COL, lambda df: df.any(axis=0)),
    ("all_axis0_with_zero", WITH_ZEROS, lambda df: df.all(axis=0)),
    ("any_axis0_with_zero", ALL_ZEROS, lambda df: df.any(axis=0)),
    ("all_axis1", NUMERIC_3COL, lambda df: df.all(axis=1)),
    ("any_axis1", NUMERIC_3COL, lambda df: df.any(axis=1)),
]


@pytest.mark.parametrize("name,data,op", ALL_ANY_CASES, ids=[c[0] for c in ALL_ANY_CASES])
def test_all_any(name, data, op):
    run_equiv(data, op)


ROUND_CASES = [
    ("round_0", {"a": [1.234, 2.567, 3.891], "b": [4.123, 5.678, 6.999]}, lambda df: df.round(0)),
    ("round_1", {"a": [1.234, 2.567, 3.891], "b": [4.123, 5.678, 6.999]}, lambda df: df.round(1)),
    ("round_2", {"a": [1.234, 2.567, 3.891], "b": [4.123, 5.678, 6.999]}, lambda df: df.round(2)),
]


@pytest.mark.parametrize("name,data,op", ROUND_CASES, ids=[c[0] for c in ROUND_CASES])
def test_round(name, data, op):
    run_equiv(data, op)


IDXMIN_IDXMAX_CASES = [
    ("idxmin_axis0", NUMERIC_2COL, lambda df: df.idxmin(axis=0)),
    ("idxmax_axis0", NUMERIC_2COL, lambda df: df.idxmax(axis=0)),
    ("idxmin_negatives", WITH_NEGATIVES, lambda df: df.idxmin(axis=0)),
    ("idxmax_negatives", WITH_NEGATIVES, lambda df: df.idxmax(axis=0)),
]


@pytest.mark.parametrize(
    "name,data,op", IDXMIN_IDXMAX_CASES, ids=[c[0] for c in IDXMIN_IDXMAX_CASES]
)
def test_idxmin_idxmax(name, data, op):
    # idxmin/idxmax returns index labels — compare as integers
    from jaxframe import DataFrame

    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)
    pd_result = op(pdf)
    jf_result = op(jdf)
    np.testing.assert_array_equal(np.asarray(jf_result.values), pd_result.values.astype(np.int64))


ISIN_CASES = [
    ("isin_basic", NUMERIC_2COL, lambda df: df.isin([1.0, 2.0, 10.0])),
    ("isin_empty", NUMERIC_2COL, lambda df: df.isin([])),
    (
        "isin_all",
        NUMERIC_2COL,
        lambda df: df.isin([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
    ),
]


@pytest.mark.parametrize("name,data,op", ISIN_CASES, ids=[c[0] for c in ISIN_CASES])
def test_isin(name, data, op):
    run_equiv(data, op)


# ============================
# Session 16: Rolling window ops
# ============================

ROLLING_DATA = {
    "a": [1.0, 2.0, 3.0, 4.0, 5.0],
    "b": [10.0, 20.0, 30.0, 40.0, 50.0],
}

ROLLING_CASES = [
    ("roll_sum_3", ROLLING_DATA, lambda df: df.rolling(3).sum()),
    ("roll_mean_3", ROLLING_DATA, lambda df: df.rolling(3).mean()),
    ("roll_std_3", ROLLING_DATA, lambda df: df.rolling(3).std()),
    ("roll_var_3", ROLLING_DATA, lambda df: df.rolling(3).var()),
    ("roll_min_3", ROLLING_DATA, lambda df: df.rolling(3).min()),
    ("roll_max_3", ROLLING_DATA, lambda df: df.rolling(3).max()),
    ("roll_sum_2", ROLLING_DATA, lambda df: df.rolling(2).sum()),
    ("roll_mean_2", ROLLING_DATA, lambda df: df.rolling(2).mean()),
]


@pytest.mark.parametrize("name,data,op", ROLLING_CASES, ids=[c[0] for c in ROLLING_CASES])
def test_rolling(name, data, op):
    run_equiv(data, op)


class TestTimeRolling:
    """Time-based rolling windows (not JIT-compatible)."""

    def _make_data(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        return dates, {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]}

    def test_time_rolling_sum(self):
        from jaxframe import DataFrame

        dates, data = self._make_data()
        pdf = pd.DataFrame(data, index=dates)
        jdf = DataFrame(data, index=dates.values)
        pd_result = pdf.rolling("3D").sum()
        jf_result = jdf.rolling("3D").sum()
        np.testing.assert_allclose(
            np.asarray(jf_result.values),
            pd_result.values.astype(np.float32),
            rtol=1e-5,
        )

    def test_time_rolling_mean(self):
        from jaxframe import DataFrame

        dates, data = self._make_data()
        pdf = pd.DataFrame(data, index=dates)
        jdf = DataFrame(data, index=dates.values)
        pd_result = pdf.rolling("2D").mean()
        jf_result = jdf.rolling("2D").mean()
        np.testing.assert_allclose(
            np.asarray(jf_result.values),
            pd_result.values.astype(np.float32),
            rtol=1e-5,
        )

    def test_time_rolling_std(self):
        from jaxframe import DataFrame

        dates, data = self._make_data()
        pdf = pd.DataFrame(data, index=dates)
        jdf = DataFrame(data, index=dates.values)
        pd_result = pdf.rolling("3D").std()
        jf_result = jdf.rolling("3D").std()
        np.testing.assert_allclose(
            np.asarray(jf_result.values),
            pd_result.values.astype(np.float32),
            rtol=1e-4,
        )


# ============================
# Session 17: value_counts, nunique, mode, skew, kurt, sem
# ============================

STATS_DATA = {
    "a": [1.0, 2.0, 3.0, 4.0, 5.0],
    "b": [10.0, 20.0, 30.0, 40.0, 50.0],
}

SKEW_DATA = {
    "a": [1.0, 2.0, 2.0, 3.0, 10.0],
    "b": [1.0, 1.0, 1.0, 2.0, 100.0],
}


def test_nunique():
    from jaxframe import DataFrame

    pdf = pd.DataFrame(STATS_DATA)
    jdf = DataFrame(STATS_DATA)
    pd_result = pdf.nunique()
    jf_result = jdf.nunique()
    np.testing.assert_array_equal(np.asarray(jf_result.values), pd_result.values)


def test_skew():
    from jaxframe import DataFrame

    pdf = pd.DataFrame(SKEW_DATA)
    jdf = DataFrame(SKEW_DATA)
    pd_result = pdf.skew()
    jf_result = jdf.skew()
    np.testing.assert_allclose(
        np.asarray(jf_result.values),
        pd_result.values.astype(np.float32),
        rtol=1e-4,
    )


def test_kurt():
    from jaxframe import DataFrame

    pdf = pd.DataFrame(SKEW_DATA)
    jdf = DataFrame(SKEW_DATA)
    pd_result = pdf.kurt()
    jf_result = jdf.kurt()
    np.testing.assert_allclose(
        np.asarray(jf_result.values),
        pd_result.values.astype(np.float32),
        rtol=1e-4,
    )


def test_sem():
    from jaxframe import DataFrame

    pdf = pd.DataFrame(STATS_DATA)
    jdf = DataFrame(STATS_DATA)
    pd_result = pdf.sem()
    jf_result = jdf.sem()
    np.testing.assert_allclose(
        np.asarray(jf_result.values),
        pd_result.values.astype(np.float32),
        rtol=1e-5,
    )


def test_mode():
    from jaxframe import DataFrame as JDF

    data = {"a": [1.0, 1.0, 2.0, 3.0, 3.0], "b": [10.0, 10.0, 10.0, 20.0, 20.0]}
    pdf = pd.DataFrame(data)
    jdf = JDF(data)
    pd_result = pdf.mode()
    jf_result = jdf.mode()
    # mode may return multiple rows; compare first row
    np.testing.assert_allclose(
        np.asarray(jf_result.values)[0],
        pd_result.values[0].astype(np.float32),
        rtol=1e-5,
    )


def test_value_counts_series():
    from jaxframe import Series

    data = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0]
    ps = pd.Series(data)
    js = Series(data)
    pd_result = ps.value_counts()
    jf_result = js.value_counts()
    # Sort by value for consistent comparison
    pd_sorted = pd_result.sort_index()
    jf_vals = np.asarray(jf_result.values)
    jf_idx = np.asarray(jf_result.index)
    order = np.argsort(jf_idx)
    np.testing.assert_array_equal(jf_vals[order], pd_sorted.values)
    np.testing.assert_allclose(
        jf_idx[order].astype(np.float32),
        pd_sorted.index.values.astype(np.float32),
    )


# ============================
# Session 18: reset_index, set_index, sort_index
# ============================


class TestIndexOps:
    def test_reset_index(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
        jdf = DataFrame(data, index=np.array([10, 20, 30]))
        result = jdf.reset_index()
        # After reset, index should be 0, 1, 2
        np.testing.assert_array_equal(result._index, np.arange(3))
        # Values should be unchanged
        np.testing.assert_allclose(
            np.asarray(result.values),
            np.asarray(jdf.values),
        )

    def test_reset_index_drop_false(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
        pdf = pd.DataFrame(data, index=pd.Index([10, 20, 30], name="idx"))
        jdf = DataFrame(data, index=np.array([10, 20, 30]))
        jdf._index_name = "idx"
        result = jdf.reset_index(drop=False)
        pdf.reset_index()
        # Should have 3 columns: idx, a, b
        assert result.shape[1] == 3

    def test_set_index(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
        jdf = DataFrame(data)
        result = jdf.set_index("a")
        np.testing.assert_allclose(
            np.asarray(result._index),
            np.array([1.0, 2.0, 3.0]),
        )
        # Column "a" should be removed
        assert "a" not in result._column_order

    def test_sort_index(self):
        from jaxframe import DataFrame

        data = {"a": [3.0, 1.0, 2.0], "b": [30.0, 10.0, 20.0]}
        jdf = DataFrame(data, index=np.array([2, 0, 1]))
        result = jdf.sort_index()
        np.testing.assert_array_equal(result._index, np.array([0, 1, 2]))
        # Values should be reordered
        np.testing.assert_allclose(
            np.asarray(result["a"].values),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

    def test_sort_index_descending(self):
        from jaxframe import DataFrame

        data = {"a": [3.0, 1.0, 2.0], "b": [30.0, 10.0, 20.0]}
        jdf = DataFrame(data, index=np.array([2, 0, 1]))
        result = jdf.sort_index(ascending=False)
        np.testing.assert_array_equal(result._index, np.array([2, 1, 0]))


# ============================
# Session 19: astype, select_dtypes
# ============================


class TestTypeOps:
    def test_astype_float32(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
        jdf = DataFrame(data)
        result = jdf.astype("float32")
        for block in result._dtype_blocks.values():
            assert block.dtype == np.float32

    def test_astype_int32(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
        jdf = DataFrame(data)
        result = jdf.astype("int32")
        for block in result._dtype_blocks.values():
            assert block.dtype == np.int32

    def test_select_dtypes_include(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0], "b": [3, 4], "c": ["x", "y"]}
        jdf = DataFrame(data)
        result = jdf.select_dtypes(include=["float64"])
        assert "a" in result._column_order
        assert "b" not in result._column_order
        assert "c" not in result._column_order

    def test_select_dtypes_exclude(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0], "b": [3, 4], "c": ["x", "y"]}
        jdf = DataFrame(data)
        result = jdf.select_dtypes(exclude=["object"])
        assert "c" not in result._column_order
        assert "a" in result._column_order


# ============================
# JAX compat info API
# ============================


def test_jax_info_query():
    from jaxframe import jax_info

    result = jax_info("sum")
    assert result["jit"] is True
    assert result["grad"] is True
    assert result["reason"] is None

    result = jax_info("min")
    assert result["jit"] is True
    assert result["grad"] is False
    assert "Non-smooth" in result["reason"]

    result = jax_info("to_csv")
    assert result["jit"] is False

    assert jax_info("nonexistent_op") is None


def test_jax_info_table(capsys):
    from jaxframe import jax_info

    jax_info()  # prints table
    captured = capsys.readouterr()
    assert "Operation" in captured.out
    assert "sum" in captured.out
    assert "min" in captured.out


# ============================
# Session 20: I/O — to_csv, read_csv
# ============================


# ============================
# EWM (exponentially weighted)
# ============================


class TestEWM:
    def test_ewm_mean_span(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.ewm(span=3).mean()
        expected = pdf.ewm(span=3).mean()
        np.testing.assert_allclose(np.asarray(result.values), expected.values, rtol=1e-3)

    def test_ewm_mean_alpha(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.ewm(alpha=0.5).mean()
        expected = pdf.ewm(alpha=0.5).mean()
        np.testing.assert_allclose(np.asarray(result.values), expected.values, rtol=1e-3)


# ============================
# duplicated / drop_duplicates
# ============================


class TestDuplicates:
    def test_duplicated_first(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 1.0, 3.0, 2.0], "b": [10.0, 20.0, 10.0, 30.0, 20.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.duplicated()
        expected = pdf.duplicated()
        np.testing.assert_array_equal(np.asarray(result.values), expected.values)

    def test_duplicated_last(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 1.0, 3.0, 2.0], "b": [10.0, 20.0, 10.0, 30.0, 20.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.duplicated(keep="last")
        expected = pdf.duplicated(keep="last")
        np.testing.assert_array_equal(np.asarray(result.values), expected.values)

    def test_duplicated_false(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 1.0, 3.0, 2.0], "b": [10.0, 20.0, 10.0, 30.0, 20.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.duplicated(keep=False)
        expected = pdf.duplicated(keep=False)
        np.testing.assert_array_equal(np.asarray(result.values), expected.values)

    def test_drop_duplicates(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 1.0, 3.0], "b": [10.0, 20.0, 10.0, 30.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.drop_duplicates()
        expected = pdf.drop_duplicates()
        np.testing.assert_allclose(np.asarray(result.values), expected.values, rtol=1e-5)
        assert result.shape == expected.shape

    def test_drop_duplicates_subset(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 1.0, 3.0], "b": [10.0, 20.0, 99.0, 30.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.drop_duplicates(subset=["a"])
        expected = pdf.drop_duplicates(subset=["a"])
        np.testing.assert_allclose(np.asarray(result.values), expected.values, rtol=1e-5)


# ============================
# pipe, between, map, replace, rank
# ============================


class TestPipeBetweenMapReplace:
    def test_pipe_df(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
        jdf = DataFrame(data)
        result = jdf.pipe(lambda df: df * 2 + 1)
        expected = jdf * 2 + 1
        np.testing.assert_allclose(np.asarray(result.values), np.asarray(expected.values))

    def test_pipe_series(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0]}
        jdf = DataFrame(data)
        result = jdf["a"].pipe(lambda s: s * 3)
        np.testing.assert_allclose(np.asarray(result.values), [3.0, 6.0, 9.0], rtol=1e-5)

    def test_between_series(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf["a"].between(2.0, 4.0)
        expected = pdf["a"].between(2.0, 4.0)
        np.testing.assert_array_equal(np.asarray(result.values), expected.values)

    def test_between_exclusive(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf["a"].between(2.0, 4.0, inclusive="neither")
        expected = pdf["a"].between(2.0, 4.0, inclusive="neither")
        np.testing.assert_array_equal(np.asarray(result.values), expected.values)

    def test_map_series(self):
        import jax.numpy as jnp

        from jaxframe import DataFrame

        data = {"a": [1.0, 4.0, 9.0]}
        jdf = DataFrame(data)
        result = jdf["a"].map(jnp.sqrt)
        np.testing.assert_allclose(np.asarray(result.values), [1.0, 2.0, 3.0], rtol=1e-5)

    def test_replace_series(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0, 2.0]}
        jdf = DataFrame(data)
        result = jdf["a"].replace(2.0, 99.0)
        np.testing.assert_allclose(np.asarray(result.values), [1.0, 99.0, 3.0, 99.0], rtol=1e-5)


# ============================
# Rank
# ============================


class TestRank:
    def test_rank_ordinal(self):
        from jaxframe import DataFrame

        # No ties, so ordinal == first == average
        data = {"a": [3.0, 1.0, 4.0, 2.0], "b": [10.0, 40.0, 20.0, 30.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.rank()
        expected = pdf.rank(method="first")
        np.testing.assert_allclose(np.asarray(result.values), expected.values, rtol=1e-5)

    def test_rank_descending(self):
        from jaxframe import DataFrame

        data = {"a": [3.0, 1.0, 4.0, 2.0], "b": [10.0, 40.0, 20.0, 30.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.rank(ascending=False)
        expected = pdf.rank(ascending=False, method="first")
        np.testing.assert_allclose(np.asarray(result.values), expected.values, rtol=1e-5)


# ============================
# Expanding window
# ============================


class TestExpanding:
    def test_expanding_sum(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.expanding().sum()
        expected = pdf.expanding().sum()
        np.testing.assert_allclose(np.asarray(result.values), expected.values, rtol=1e-5)

    def test_expanding_mean(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.expanding().mean()
        expected = pdf.expanding().mean()
        np.testing.assert_allclose(np.asarray(result.values), expected.values, rtol=1e-5)

    def test_expanding_min(self):
        from jaxframe import DataFrame

        data = {"a": [3.0, 1.0, 4.0, 2.0], "b": [40.0, 20.0, 30.0, 10.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.expanding().min()
        expected = pdf.expanding().min()
        np.testing.assert_allclose(np.asarray(result.values), expected.values, rtol=1e-5)

    def test_expanding_max(self):
        from jaxframe import DataFrame

        data = {"a": [3.0, 1.0, 4.0, 2.0], "b": [40.0, 20.0, 30.0, 10.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.expanding().max()
        expected = pdf.expanding().max()
        np.testing.assert_allclose(np.asarray(result.values), expected.values, rtol=1e-5)

    def test_expanding_std(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.expanding(min_periods=2).std()
        expected = pdf.expanding(min_periods=2).std()
        # Skip first row (NaN due to min_periods=2)
        np.testing.assert_allclose(np.asarray(result.values)[1:], expected.values[1:], rtol=1e-4)

    def test_expanding_var(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.expanding(min_periods=2).var()
        expected = pdf.expanding(min_periods=2).var()
        np.testing.assert_allclose(np.asarray(result.values)[1:], expected.values[1:], rtol=1e-4)


# ============================
# diff / pct_change
# ============================


class TestDiffPctChange:
    def test_diff_default(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 3.0, 6.0, 10.0], "b": [10.0, 20.0, 30.0, 40.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.diff()
        expected = pdf.diff()
        # First row is NaN in both
        np.testing.assert_allclose(np.asarray(result.values)[1:], expected.values[1:], rtol=1e-5)
        assert np.all(np.isnan(np.asarray(result.values)[0]))

    def test_diff_periods_2(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 3.0, 6.0, 10.0], "b": [10.0, 20.0, 30.0, 40.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.diff(periods=2)
        expected = pdf.diff(periods=2)
        np.testing.assert_allclose(np.asarray(result.values)[2:], expected.values[2:], rtol=1e-5)

    def test_diff_negative_periods(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 3.0, 6.0, 10.0], "b": [10.0, 20.0, 30.0, 40.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.diff(periods=-1)
        expected = pdf.diff(periods=-1)
        # Last row is NaN
        np.testing.assert_allclose(np.asarray(result.values)[:-1], expected.values[:-1], rtol=1e-5)

    def test_pct_change_default(self):
        from jaxframe import DataFrame

        data = {"a": [10.0, 20.0, 30.0], "b": [100.0, 50.0, 200.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.pct_change()
        expected = pdf.pct_change()
        np.testing.assert_allclose(np.asarray(result.values)[1:], expected.values[1:], rtol=1e-5)

    def test_pct_change_periods_2(self):
        from jaxframe import DataFrame

        data = {"a": [10.0, 20.0, 30.0, 40.0], "b": [100.0, 50.0, 200.0, 400.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf.pct_change(periods=2)
        expected = pdf.pct_change(periods=2)
        np.testing.assert_allclose(np.asarray(result.values)[2:], expected.values[2:], rtol=1e-5)

    def test_series_diff(self):
        from jaxframe import DataFrame

        data = {"a": [1.0, 3.0, 6.0, 10.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf["a"].diff()
        expected = pdf["a"].diff()
        np.testing.assert_allclose(np.asarray(result.values)[1:], expected.values[1:], rtol=1e-5)

    def test_series_pct_change(self):
        from jaxframe import DataFrame

        data = {"a": [10.0, 20.0, 30.0]}
        jdf = DataFrame(data)
        pdf = pd.DataFrame(data)
        result = jdf["a"].pct_change()
        expected = pdf["a"].pct_change()
        np.testing.assert_allclose(np.asarray(result.values)[1:], expected.values[1:], rtol=1e-5)


class TestIO:
    def test_to_csv_and_read_csv(self, tmp_path):
        from jaxframe import DataFrame, read_csv

        data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
        jdf = DataFrame(data)
        path = str(tmp_path / "test.csv")
        jdf.to_csv(path)
        result = read_csv(path)
        np.testing.assert_allclose(
            np.asarray(result.values),
            np.asarray(jdf.values),
            rtol=1e-5,
        )

    def test_to_csv_index(self, tmp_path):
        from jaxframe import DataFrame, read_csv

        data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
        jdf = DataFrame(data, index=np.array([10, 20, 30]))
        path = str(tmp_path / "test_idx.csv")
        jdf.to_csv(path, index=True)
        # Just verify it writes without error
        result = read_csv(path, index_col=0)
        assert result.shape == (3, 2)
