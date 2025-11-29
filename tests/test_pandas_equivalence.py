"""
Tests comparing JAXFrame against pandas for API equivalence.

This test suite ensures that JAXFrame's pandas-compatible API
produces the same results as pandas (within numerical tolerance).
"""

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp

from jaxframe import DataFrame, concat


class TestCoreAttributes:
    """Test core DataFrame attributes against pandas."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [1, 2, 3, 4, 5],
            'b': [10.5, 20.5, 30.5, 40.5, 50.5],
            'c': [100, 200, 300, 400, 500],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_size(self):
        """Test size attribute."""
        assert self.jf_df.size == self.pd_df.size

    def test_ndim(self):
        """Test ndim attribute."""
        assert self.jf_df.ndim == self.pd_df.ndim

    def test_empty(self):
        """Test empty attribute."""
        assert self.jf_df.empty == self.pd_df.empty

        # Test with empty DataFrame
        jf_empty = DataFrame({})
        pd_empty = pd.DataFrame({})
        assert jf_empty.empty == pd_empty.empty

    def test_shape(self):
        """Test shape property."""
        assert self.jf_df.shape == self.pd_df.shape

    def test_values(self):
        """Test values property."""
        np.testing.assert_allclose(
            self.jf_df.values,
            self.pd_df.values,
            rtol=1e-5
        )


class TestArithmeticOperators:
    """Test arithmetic operators against pandas."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [10.0, 20.0, 30.0],
            'b': [5.0, 10.0, 15.0],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_division(self):
        """Test division operator."""
        jf_result = self.jf_df / 2
        pd_result = self.pd_df / 2
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )

    def test_floor_division(self):
        """Test floor division operator."""
        jf_result = self.jf_df // 3
        pd_result = self.pd_df // 3
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )

    def test_modulo(self):
        """Test modulo operator."""
        jf_result = self.jf_df % 7
        pd_result = self.pd_df % 7
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )

    def test_power(self):
        """Test power operator."""
        jf_result = self.jf_df ** 2
        pd_result = self.pd_df ** 2
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )

    def test_matmul(self):
        """Test matrix multiplication operator."""
        # Test with compatible shapes
        other = np.array([[1.0, 2.0], [3.0, 4.0]])

        jf_result = self.jf_df @ other
        pd_result = self.pd_df @ other

        np.testing.assert_allclose(
            jf_result,
            pd_result,
            rtol=1e-5
        )


class TestComparisonOperators:
    """Test comparison operators against pandas."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_greater_equal(self):
        """Test >= operator."""
        jf_result = self.jf_df >= 3
        pd_result = self.pd_df >= 3
        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_less_than(self):
        """Test < operator."""
        jf_result = self.jf_df < 3
        pd_result = self.pd_df < 3
        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_less_equal(self):
        """Test <= operator."""
        jf_result = self.jf_df <= 3
        pd_result = self.pd_df <= 3
        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_equal(self):
        """Test == operator."""
        jf_result = self.jf_df == 3
        pd_result = self.pd_df == 3
        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_not_equal(self):
        """Test != operator."""
        jf_result = self.jf_df != 3
        pd_result = self.pd_df != 3
        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )


class TestLogicalOperators:
    """Test logical operators against pandas."""

    def setup_method(self):
        """Create test data."""
        # Use comparison results which produce boolean arrays
        data = {
            'a': [1, 2, 3, 4],
            'b': [5, 4, 3, 2],
        }
        self.jf_df_base = DataFrame(data)
        self.pd_df_base = pd.DataFrame(data)

    def test_and_operator(self):
        """Test & operator with boolean results from comparisons."""
        # Create boolean DataFrames from comparisons
        jf_bool1 = self.jf_df_base > 2
        jf_bool2 = self.jf_df_base < 5
        pd_bool1 = self.pd_df_base > 2
        pd_bool2 = self.pd_df_base < 5

        jf_result = jf_bool1 & jf_bool2
        pd_result = pd_bool1 & pd_bool2

        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_or_operator(self):
        """Test | operator with boolean results from comparisons."""
        # Create boolean DataFrames from comparisons
        jf_bool1 = self.jf_df_base > 3
        jf_bool2 = self.jf_df_base < 2
        pd_bool1 = self.pd_df_base > 3
        pd_bool2 = self.pd_df_base < 2

        jf_result = jf_bool1 | jf_bool2
        pd_result = pd_bool1 | pd_bool2

        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_invert_operator(self):
        """Test ~ operator with boolean results from comparisons."""
        # Create boolean DataFrame from comparison
        jf_bool = self.jf_df_base > 2
        pd_bool = self.pd_df_base > 2

        jf_result = ~jf_bool
        pd_result = ~pd_bool

        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )


class TestReductionMethods:
    """Test reduction methods against pandas."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_std(self):
        """Test standard deviation."""
        jf_result = self.jf_df.std()
        pd_result = self.pd_df.std()
        np.testing.assert_allclose(
            jf_result._data,
            pd_result.values,
            rtol=1e-5
        )

    def test_var(self):
        """Test variance."""
        jf_result = self.jf_df.var()
        pd_result = self.pd_df.var()
        np.testing.assert_allclose(
            jf_result._data,
            pd_result.values,
            rtol=1e-5
        )

    def test_min(self):
        """Test minimum."""
        jf_result = self.jf_df.min()
        pd_result = self.pd_df.min()
        np.testing.assert_allclose(
            jf_result._data,
            pd_result.values,
            rtol=1e-5
        )

    def test_max(self):
        """Test maximum."""
        jf_result = self.jf_df.max()
        pd_result = self.pd_df.max()
        np.testing.assert_allclose(
            jf_result._data,
            pd_result.values,
            rtol=1e-5
        )

    def test_prod(self):
        """Test product."""
        jf_result = self.jf_df.prod()
        pd_result = self.pd_df.prod()
        np.testing.assert_allclose(
            jf_result._data,
            pd_result.values,
            rtol=1e-5
        )

    def test_abs(self):
        """Test absolute value."""
        data_with_neg = {
            'a': [-1.0, 2.0, -3.0],
            'b': [10.0, -20.0, 30.0],
        }
        jf_df = DataFrame(data_with_neg)
        pd_df = pd.DataFrame(data_with_neg)

        jf_result = jf_df.abs()
        pd_result = pd_df.abs()

        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )


class TestIndexingMethods:
    """Test indexing methods against pandas."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': [100, 200, 300, 400, 500],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_head(self):
        """Test head method."""
        jf_result = self.jf_df.head(3)
        pd_result = self.pd_df.head(3)
        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_tail(self):
        """Test tail method."""
        jf_result = self.jf_df.tail(2)
        pd_result = self.pd_df.tail(2)
        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_iloc_single_row(self):
        """Test iloc with single row."""
        jf_result = self.jf_df.iloc[2]
        pd_result = self.pd_df.iloc[2]
        np.testing.assert_allclose(
            jf_result._data,
            pd_result.values,
            rtol=1e-5
        )

    def test_iloc_slice(self):
        """Test iloc with slice."""
        jf_result = self.jf_df.iloc[1:4]
        pd_result = self.pd_df.iloc[1:4]
        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_iloc_2d(self):
        """Test iloc with 2D indexing."""
        jf_result = self.jf_df.iloc[1:3, 0:2]
        pd_result = self.pd_df.iloc[1:3, 0:2]
        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_attribute_access(self):
        """Test attribute access to columns."""
        jf_result = self.jf_df.a
        pd_result = self.pd_df.a
        np.testing.assert_array_equal(
            jf_result._data,
            pd_result.values
        )


class TestMaskingMethods:
    """Test masking methods against pandas."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_mask(self):
        """Test mask method."""
        jf_result = self.jf_df.mask(self.jf_df > 3, 0)
        pd_result = self.pd_df.mask(self.pd_df > 3, 0)
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )

    def test_clip(self):
        """Test clip method."""
        jf_result = self.jf_df.clip(2, 40)
        pd_result = self.pd_df.clip(2, 40)
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )

    def test_clip_lower_only(self):
        """Test clip with lower bound only."""
        jf_result = self.jf_df.clip(lower=3)
        pd_result = self.pd_df.clip(lower=3)
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )

    def test_clip_upper_only(self):
        """Test clip with upper bound only."""
        jf_result = self.jf_df.clip(upper=30)
        pd_result = self.pd_df.clip(upper=30)
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )


class TestStatisticalFunctions:
    """Test statistical functions against pandas."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [2.0, 3.0, 4.0, 5.0, 6.0],
            'c': [5.0, 4.0, 3.0, 2.0, 1.0],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_corr(self):
        """Test correlation matrix."""
        jf_result = self.jf_df.corr()
        pd_result = self.pd_df.corr()
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )

    def test_cov(self):
        """Test covariance matrix."""
        jf_result = self.jf_df.cov()
        pd_result = self.pd_df.cov()
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )


class TestShapeOperations:
    """Test shape operations against pandas."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_transpose(self):
        """Test transpose method."""
        jf_result = self.jf_df.transpose()
        pd_result = self.pd_df.transpose()
        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_T_property(self):
        """Test T property."""
        jf_result = self.jf_df.T
        pd_result = self.pd_df.T
        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_concat_axis0(self):
        """Test concatenation along axis 0."""
        data2 = {'a': [7, 8], 'b': [9, 10]}
        jf_df2 = DataFrame(data2)
        pd_df2 = pd.DataFrame(data2)

        jf_result = concat([self.jf_df, jf_df2], axis=0)
        pd_result = pd.concat([self.pd_df, pd_df2], axis=0, ignore_index=True)

        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )

    def test_concat_axis1(self):
        """Test concatenation along axis 1."""
        data2 = {'c': [7, 8, 9], 'd': [10, 11, 12]}
        jf_df2 = DataFrame(data2)
        pd_df2 = pd.DataFrame(data2)

        jf_result = concat([self.jf_df, jf_df2], axis=1)
        pd_result = pd.concat([self.pd_df, pd_df2], axis=1)

        np.testing.assert_array_equal(
            jf_result.values,
            pd_result.values
        )


class TestTimeSeriesMethods:
    """Test time series methods against pandas."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_shift_forward(self):
        """Test shift with positive periods (lag)."""
        jf_result = self.jf_df.shift(1, fill_value=0)
        pd_result = self.pd_df.shift(1, fill_value=0)
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )

    def test_shift_backward(self):
        """Test shift with negative periods (lead)."""
        jf_result = self.jf_df.shift(-1, fill_value=0)
        pd_result = self.pd_df.shift(-1, fill_value=0)
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )

    def test_diff(self):
        """Test diff method."""
        jf_result = self.jf_df.diff(1)
        pd_result = self.pd_df.diff(1).fillna(0)  # pandas uses NaN, we use 0
        np.testing.assert_allclose(
            jf_result.values,
            pd_result.values,
            rtol=1e-5
        )

    def test_pct_change(self):
        """Test percentage change."""
        # Note: pandas uses NaN for first row, we use fill_value
        # So we'll compare from row 1 onwards
        jf_result = self.jf_df.pct_change(1)
        pd_result = self.pd_df.pct_change(1)

        # Compare non-NaN values (skip first row where pandas has NaN)
        np.testing.assert_allclose(
            jf_result.values[1:],
            pd_result.values[1:],
            rtol=1e-5
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
