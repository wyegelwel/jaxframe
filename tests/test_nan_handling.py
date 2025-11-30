"""
Comprehensive tests for NaN handling in JAXFrame to ensure pandas equivalence.

This test suite verifies that JAXFrame handles NaN values identically to pandas
across all operations.
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp
import pytest

from jaxframe import DataFrame


class TestNaNArithmetic:
    """Test arithmetic operations with NaN values."""

    def test_addition_with_nan(self):
        """Test addition with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # DataFrame + scalar
        jf_result = jf_df + 10
        pd_result = pd_df + 10
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

        # DataFrame + DataFrame
        jf_result = jf_df + jf_df
        pd_result = pd_df + pd_df
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_subtraction_with_nan(self):
        """Test subtraction with NaN values matches pandas."""
        data = {'a': [10.0, np.nan, 30.0], 'b': [40.0, 50.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df - 5
        pd_result = pd_df - 5
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_multiplication_with_nan(self):
        """Test multiplication with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df * 2
        pd_result = pd_df * 2
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_division_with_nan(self):
        """Test division with NaN values matches pandas."""
        data = {'a': [10.0, np.nan, 30.0], 'b': [40.0, 50.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df / 2
        pd_result = pd_df / 2
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_power_with_nan(self):
        """Test power operations with NaN values matches pandas."""
        data = {'a': [2.0, np.nan, 4.0], 'b': [1.0, 2.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df ** 2
        pd_result = pd_df ** 2
        np.testing.assert_array_equal(jf_result.values, pd_result.values)


class TestNaNComparisons:
    """Test comparison operations with NaN values."""

    def test_greater_than_with_nan(self):
        """Test > comparison with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df > 2
        pd_result = pd_df > 2
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_less_than_with_nan(self):
        """Test < comparison with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df < 2
        pd_result = pd_df < 2
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_equality_with_nan(self):
        """Test == comparison with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # NaN != NaN in IEEE 754
        jf_result = jf_df == 2
        pd_result = pd_df == 2
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_nan_equality(self):
        """Test that NaN == NaN returns False (IEEE 754 behavior)."""
        data = {'a': [np.nan, np.nan, 3.0]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df == jf_df
        pd_result = pd_df == pd_df
        np.testing.assert_array_equal(jf_result.values, pd_result.values)


class TestNaNReductions:
    """Test reduction operations with NaN values."""

    def test_sum_with_nan(self):
        """Test sum with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.sum()
        pd_result = pd_df.sum()

        # Compare values (NaN propagation)
        for col in ['a', 'b']:
            if np.isnan(pd_result[col]):
                assert np.isnan(jf_result._data[jf_result._index.tolist().index(col)])
            else:
                np.testing.assert_allclose(
                    jf_result._data[jf_result._index.tolist().index(col)],
                    pd_result[col]
                )

    def test_mean_with_nan(self):
        """Test mean with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.mean()
        pd_result = pd_df.mean()

        for col in ['a', 'b']:
            if np.isnan(pd_result[col]):
                assert np.isnan(jf_result._data[jf_result._index.tolist().index(col)])
            else:
                np.testing.assert_allclose(
                    jf_result._data[jf_result._index.tolist().index(col)],
                    pd_result[col]
                )

    def test_min_with_nan(self):
        """Test min with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.min()
        pd_result = pd_df.min()

        for col in ['a', 'b']:
            if np.isnan(pd_result[col]):
                assert np.isnan(jf_result._data[jf_result._index.tolist().index(col)])
            else:
                np.testing.assert_allclose(
                    jf_result._data[jf_result._index.tolist().index(col)],
                    pd_result[col]
                )

    def test_max_with_nan(self):
        """Test max with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.max()
        pd_result = pd_df.max()

        for col in ['a', 'b']:
            if np.isnan(pd_result[col]):
                assert np.isnan(jf_result._data[jf_result._index.tolist().index(col)])
            else:
                np.testing.assert_allclose(
                    jf_result._data[jf_result._index.tolist().index(col)],
                    pd_result[col]
                )

    def test_std_with_nan(self):
        """Test std with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0, 4.0, 5.0], 'b': [4.0, 5.0, 6.0, np.nan, 8.0]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.std()
        pd_result = pd_df.std()

        for col in ['a', 'b']:
            if np.isnan(pd_result[col]):
                assert np.isnan(jf_result._data[jf_result._index.tolist().index(col)])
            else:
                np.testing.assert_allclose(
                    jf_result._data[jf_result._index.tolist().index(col)],
                    pd_result[col],
                    rtol=1e-5
                )


class TestNaNTimeSeriesOperations:
    """Test time series operations with NaN values."""

    def test_shift_creates_nan(self):
        """Test that shift() creates NaN values by default (matching pandas)."""
        data = {'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.shift(1)
        pd_result = pd_df.shift(1)

        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_shift_with_nan_data(self):
        """Test shift with existing NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.shift(1)
        pd_result = pd_df.shift(1)

        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_shift_negative_with_nan(self):
        """Test negative shift with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0, 4.0], 'b': [5.0, 6.0, np.nan, 8.0]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.shift(-1)
        pd_result = pd_df.shift(-1)

        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_diff_creates_nan(self):
        """Test that diff() creates NaN for first row (matching pandas)."""
        data = {'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.diff()
        pd_result = pd_df.diff()

        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_diff_with_nan_data(self):
        """Test diff with existing NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0, 4.0], 'b': [5.0, 6.0, np.nan, 8.0]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.diff()
        pd_result = pd_df.diff()

        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_diff_periods_with_nan(self):
        """Test diff with periods > 1 and NaN values matches pandas."""
        data = {'a': [1.0, 2.0, np.nan, 4.0, 5.0], 'b': [6.0, np.nan, 8.0, 9.0, 10.0]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.diff(2)
        pd_result = pd_df.diff(2)

        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_pct_change_creates_nan(self):
        """Test that pct_change() creates NaN for first row (matching pandas)."""
        data = {'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.pct_change()
        pd_result = pd_df.pct_change()

        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_pct_change_with_nan_data(self):
        """Test pct_change with existing NaN values matches pandas (with fill_method=None)."""
        data = {'a': [1.0, np.nan, 3.0, 4.0], 'b': [5.0, 6.0, np.nan, 8.0]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.pct_change()
        # Use fill_method=None to match JAXFrame behavior (pandas default is deprecated)
        pd_result = pd_df.pct_change(fill_method=None)

        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5, equal_nan=True)


class TestNaNIndexing:
    """Test indexing operations with NaN values."""

    def test_iloc_with_nan(self):
        """Test iloc with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # Select rows
        jf_result = jf_df.iloc[0:2]
        pd_result = pd_df.iloc[0:2]
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

        # Select columns
        jf_result = jf_df[['a']]
        pd_result = pd_df[['a']]
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    # TODO: Re-enable once Series supports comparison operators
    # def test_boolean_indexing_with_nan(self):
    #     """Test boolean indexing with NaN values matches pandas."""
    #     data = {'a': [1.0, np.nan, 3.0, 4.0], 'b': [5.0, 6.0, np.nan, 8.0]}
    #
    #     jf_df = DataFrame(data)
    #     pd_df = pd.DataFrame(data)
    #
    #     # Boolean mask - NaN comparisons return False
    #     mask_jf = jf_df['a'] > 2
    #     mask_pd = pd_df['a'] > 2
    #
    #     # Verify masks are the same
    #     np.testing.assert_array_equal(mask_jf._data, mask_pd.values)


class TestNaNStatisticalOperations:
    """Test statistical operations with NaN values."""

    def test_var_with_nan(self):
        """Test variance with NaN values matches pandas."""
        data = {'a': [1.0, np.nan, 3.0, 4.0, 5.0], 'b': [4.0, 5.0, 6.0, np.nan, 8.0]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.var()
        pd_result = pd_df.var()

        for col in ['a', 'b']:
            if np.isnan(pd_result[col]):
                assert np.isnan(jf_result._data[jf_result._index.tolist().index(col)])
            else:
                np.testing.assert_allclose(
                    jf_result._data[jf_result._index.tolist().index(col)],
                    pd_result[col],
                    rtol=1e-5
                )

    # TODO: Add cumsum and cumprod tests once those methods are implemented
    # def test_cumsum_with_nan(self):
    #     """Test cumulative sum with NaN values matches pandas."""
    #     data = {'a': [1.0, np.nan, 3.0, 4.0], 'b': [5.0, 6.0, np.nan, 8.0]}
    #
    #     jf_df = DataFrame(data)
    #     pd_df = pd.DataFrame(data)
    #
    #     jf_result = jf_df.cumsum()
    #     pd_result = pd_df.cumsum()
    #
    #     np.testing.assert_array_equal(jf_result.values, pd_result.values)
    #
    # def test_cumprod_with_nan(self):
    #     """Test cumulative product with NaN values matches pandas."""
    #     data = {'a': [1.0, np.nan, 3.0, 4.0], 'b': [2.0, 3.0, np.nan, 5.0]}
    #
    #     jf_df = DataFrame(data)
    #     pd_df = pd.DataFrame(data)
    #
    #     jf_result = jf_df.cumprod()
    #     pd_result = pd_df.cumprod()
    #
    #     np.testing.assert_array_equal(jf_result.values, pd_result.values)


class TestNaNMixedDtypes:
    """Test NaN handling with mixed dtypes (dtype blocks)."""

    def test_nan_in_different_dtype_blocks(self):
        """Test NaN handling when columns have different dtypes."""
        data = {
            'int32_col': np.array([1, 2, 3], dtype=np.int32),
            'float32_col': np.array([1.0, np.nan, 3.0], dtype=np.float32),
            'float64_col': np.array([4.0, 5.0, np.nan], dtype=np.float64),
        }

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # Verify dtypes are preserved
        assert jf_df.dtypes['int32_col'] == np.dtype('int32')
        assert jf_df.dtypes['float32_col'] == np.dtype('float32')
        assert jf_df.dtypes['float64_col'] == np.dtype('float64')

        # Verify values match
        np.testing.assert_array_equal(
            jf_df[['float32_col', 'float64_col']].values,
            pd_df[['float32_col', 'float64_col']].values
        )

    def test_arithmetic_with_nan_across_dtypes(self):
        """Test that arithmetic operations preserve NaN across different dtype blocks."""
        # Note: JAX may truncate float64 to float32 depending on configuration
        # This test verifies NaN preservation, not exact dtype preservation
        data = {
            'col_a': [1.0, np.nan, 3.0],
            'col_b': [4.0, 5.0, np.nan],
        }

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df + 10
        pd_result = pd_df + 10

        # Verify NaN locations are preserved
        np.testing.assert_array_equal(jf_result.values, pd_result.values)


class TestNaNWhereOperation:
    """Test where operation with NaN values."""

    def test_where_with_nan_condition(self):
        """Test where with NaN in condition matches pandas."""
        data = {'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # Create condition with NaN result
        condition_jf = jf_df > 2
        condition_pd = pd_df > 2

        jf_result = jf_df.where(condition_jf, -999)
        pd_result = pd_df.where(condition_pd, -999)

        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_where_with_nan_in_data(self):
        """Test where with NaN in original data matches pandas."""
        data = {'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        condition_jf = jf_df > 2
        condition_pd = pd_df > 2

        jf_result = jf_df.where(condition_jf, 0)
        pd_result = pd_df.where(condition_pd, 0)

        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_where_with_nan_fill_value(self):
        """Test where with NaN as fill_value matches pandas."""
        data = {'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]}

        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        condition_jf = jf_df > 2
        condition_pd = pd_df > 2

        jf_result = jf_df.where(condition_jf, np.nan)
        pd_result = pd_df.where(condition_pd, np.nan)

        np.testing.assert_array_equal(jf_result.values, pd_result.values)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
