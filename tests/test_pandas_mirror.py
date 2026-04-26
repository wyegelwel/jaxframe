"""
Pandas-mirrored test suite.

Each test runs the same lambda on both pd.DataFrame and jaxframe.DataFrame,
then compares results. Adding a new test = adding a tuple to a list.
"""

import pytest
from conftest import (
    NUMERIC_2COL,
    NUMERIC_3COL,
    WITH_NEGATIVES,
    run_equiv,
)

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
