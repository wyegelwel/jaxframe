"""Shared fixtures and comparison helpers for jaxframe tests."""

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from jaxframe import DataFrame

# ============================
# Data fixtures (as dicts)
# ============================

NUMERIC_2COL = {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]}
NUMERIC_3COL = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]}
WITH_NEGATIVES = {"a": [-1.0, 2.0, -3.0, 4.0], "b": [10.0, -20.0, 30.0, -40.0]}
WITH_NANS = {"a": [1.0, np.nan, 3.0, 4.0], "b": [np.nan, 5.0, 6.0, np.nan]}
SINGLE_COL = {"a": [1.0, 2.0, 3.0, 4.0, 5.0]}
SINGLE_ROW = {"a": [5.0], "b": [10.0]}


# ============================
# Comparison helpers
# ============================


def assert_frame_equiv(pd_result, jf_result, rtol=1e-5):
    """Assert a pandas result matches a jaxframe result."""
    if isinstance(pd_result, pd.DataFrame):
        assert pd_result.shape == jf_result.shape
        assert_allclose(
            np.asarray(jf_result.values),
            pd_result.values.astype(np.float32),
            rtol=rtol,
        )
    elif isinstance(pd_result, pd.Series):
        pd_vals = pd_result.values.astype(np.float32)
        jf_vals = np.asarray(jf_result.values if hasattr(jf_result, "values") else jf_result._data)
        assert_allclose(jf_vals, pd_vals, rtol=rtol)
    elif isinstance(pd_result, int | float | np.number):
        assert_allclose(float(jf_result), float(pd_result), rtol=rtol)
    else:
        assert_allclose(np.asarray(jf_result), np.asarray(pd_result), rtol=rtol)


def run_equiv(data, op, rtol=1e-5):
    """Run the same op on pandas and jaxframe, assert equivalence."""
    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)
    pd_result = op(pdf)
    jf_result = op(jdf)
    assert_frame_equiv(pd_result, jf_result, rtol=rtol)
