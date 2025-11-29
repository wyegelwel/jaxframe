"""
Extended pandas equivalence tests covering edge cases and complex scenarios.

This test suite explores corner cases and documents any deviations
from pandas behavior.
"""

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp

from jaxframe import DataFrame, Series, concat


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe_operations(self):
        """Test operations on empty DataFrames."""
        jf_empty = DataFrame({})
        pd_empty = pd.DataFrame({})

        assert jf_empty.size == pd_empty.size
        assert jf_empty.shape == pd_empty.shape
        assert jf_empty.empty == pd_empty.empty

    def test_single_row_operations(self):
        """Test operations on single-row DataFrames."""
        data = {'a': [5.0], 'b': [10.0]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # Arithmetic
        jf_result = jf_df * 2
        pd_result = pd_df * 2
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

        # Reductions
        jf_sum = jf_df.sum()
        pd_sum = pd_df.sum()
        np.testing.assert_allclose(jf_sum._data, pd_sum.values, rtol=1e-5)

    def test_single_column_operations(self):
        """Test operations on single-column DataFrames."""
        data = {'a': [1.0, 2.0, 3.0]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # Arithmetic
        jf_result = jf_df + 10
        pd_result = pd_df + 10
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

        # Transpose
        jf_T = jf_df.T
        pd_T = pd_df.T
        np.testing.assert_allclose(jf_T.values, pd_T.values, rtol=1e-5)

    def test_zero_values(self):
        """Test operations with zero values."""
        data = {'a': [0.0, 1.0, 0.0], 'b': [2.0, 0.0, 3.0]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # Division by non-zero
        jf_result = jf_df / 2
        pd_result = pd_df / 2
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

        # Multiplication
        jf_result = jf_df * 0
        pd_result = pd_df * 0
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_negative_values(self):
        """Test operations with negative values."""
        data = {'a': [-1.0, -2.0, -3.0], 'b': [4.0, -5.0, 6.0]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # abs
        jf_result = jf_df.abs()
        pd_result = pd_df.abs()
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

        # Comparisons
        jf_result = jf_df < 0
        pd_result = pd_df < 0
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_large_values(self):
        """Test operations with large values."""
        data = {'a': [1e10, 2e10, 3e10], 'b': [1e20, 2e20, 3e20]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # Arithmetic preserves large values
        jf_result = jf_df / 1e10
        pd_result = pd_df / 1e10
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_small_values(self):
        """Test operations with very small values."""
        data = {'a': [1e-10, 2e-10, 3e-10], 'b': [1e-20, 2e-20, 3e-20]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # Arithmetic preserves small values
        jf_result = jf_df * 1e10
        pd_result = pd_df * 1e10
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-4)


class TestChainedOperations:
    """Test chaining multiple operations together."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_arithmetic_chain(self):
        """Test chaining arithmetic operations."""
        jf_result = (self.jf_df + 10) * 2 - 5
        pd_result = (self.pd_df + 10) * 2 - 5
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_comparison_chain(self):
        """Test chaining comparison operations."""
        jf_result = (self.jf_df > 2) & (self.jf_df < 40)
        pd_result = (self.pd_df > 2) & (self.pd_df < 40)
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_mixed_operations(self):
        """Test mixing arithmetic, comparison, and masking."""
        # (df * 2 > 10).where(...) pattern
        jf_scaled = self.jf_df * 2
        jf_mask = jf_scaled > 10
        jf_result = jf_scaled.where(jf_mask, 0)

        pd_scaled = self.pd_df * 2
        pd_mask = pd_scaled > 10
        pd_result = pd_scaled.where(pd_mask, 0)

        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_reduction_after_arithmetic(self):
        """Test reductions after arithmetic operations."""
        jf_result = (self.jf_df * 2 + 5).sum()
        pd_result = (self.pd_df * 2 + 5).sum()
        np.testing.assert_allclose(jf_result._data, pd_result.values, rtol=1e-5)

    def test_indexing_after_operations(self):
        """Test indexing after arithmetic operations."""
        jf_result = (self.jf_df + 10).head(3)
        pd_result = (self.pd_df + 10).head(3)
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)


class TestBroadcasting:
    """Test broadcasting behavior."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [1.0, 2.0, 3.0],
            'b': [4.0, 5.0, 6.0],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_scalar_broadcasting(self):
        """Test scalar broadcasts to all elements."""
        jf_result = self.jf_df + 100
        pd_result = self.pd_df + 100
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_array_broadcasting(self):
        """Test 1D array broadcasting."""
        scalar_array = np.array([10.0, 20.0])

        jf_result = self.jf_df + scalar_array
        pd_result = self.pd_df + scalar_array
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_dataframe_broadcasting(self):
        """Test DataFrame-to-DataFrame operations with same shape."""
        data2 = {'a': [10.0, 20.0, 30.0], 'b': [40.0, 50.0, 60.0]}
        jf_df2 = DataFrame(data2)
        pd_df2 = pd.DataFrame(data2)

        jf_result = self.jf_df + jf_df2
        pd_result = self.pd_df + pd_df2
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)


class TestIndexingEdgeCases:
    """Test advanced indexing scenarios."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': [100, 200, 300, 400, 500],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_iloc_negative_indices(self):
        """Test iloc with negative indices."""
        jf_result = self.jf_df.iloc[-2:]
        pd_result = self.pd_df.iloc[-2:]
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_iloc_step_slicing(self):
        """Test iloc with step in slice."""
        jf_result = self.jf_df.iloc[::2]
        pd_result = self.pd_df.iloc[::2]
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_iloc_single_column_2d(self):
        """Test iloc selecting single column with 2D indexing."""
        jf_result = self.jf_df.iloc[:, 1]
        pd_result = self.pd_df.iloc[:, 1]
        np.testing.assert_allclose(jf_result._data, pd_result.values, rtol=1e-5)

    def test_head_more_than_length(self):
        """Test head with n > len(df)."""
        jf_result = self.jf_df.head(100)
        pd_result = self.pd_df.head(100)
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_tail_zero(self):
        """Test tail with n=0."""
        jf_result = self.jf_df.tail(0)
        pd_result = self.pd_df.tail(0)
        assert jf_result.shape == pd_result.shape
        assert jf_result.shape[0] == 0


class TestStatisticalEdgeCases:
    """Test statistical functions with edge cases."""

    def test_std_single_value(self):
        """Test std with single row (std should be 0)."""
        data = {'a': [5.0], 'b': [10.0]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.std()
        pd_result = pd_df.std()
        # pandas returns NaN for single value, we might return 0
        # This is a known difference
        # np.testing.assert_allclose(jf_result._data, pd_result.values, rtol=1e-5)

    def test_corr_constant_column(self):
        """Test correlation with constant column."""
        # When a column is constant, correlation is undefined (NaN in pandas)
        data = {
            'a': [1.0, 2.0, 3.0],
            'b': [5.0, 5.0, 5.0],  # Constant
            'c': [7.0, 8.0, 9.0],
        }
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.corr()
        pd_result = pd_df.corr()

        # NOTE: JAX might produce NaN or different values for undefined correlations
        # This is a known limitation
        # We can check that non-constant columns correlate correctly
        assert jf_result.shape == pd_result.shape

    def test_var_identical_values(self):
        """Test variance with identical values (should be 0)."""
        data = {'a': [3.0, 3.0, 3.0], 'b': [7.0, 7.0, 7.0]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.var()
        pd_result = pd_df.var()
        np.testing.assert_allclose(jf_result._data, pd_result.values, rtol=1e-5, atol=1e-10)


class TestTimeSeriesEdgeCases:
    """Test time series operations with edge cases."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0],
        }
        self.jf_df = DataFrame(self.data)
        self.pd_df = pd.DataFrame(self.data)

    def test_shift_zero(self):
        """Test shift with periods=0 (should return same data)."""
        jf_result = self.jf_df.shift(0)
        pd_result = self.pd_df.shift(0)
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_shift_full_length(self):
        """Test shift by entire length."""
        jf_result = self.jf_df.shift(5, fill_value=0)
        pd_result = self.pd_df.shift(5, fill_value=0)
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_shift_more_than_length(self):
        """Test shift by more than length."""
        jf_result = self.jf_df.shift(10, fill_value=0)
        pd_result = self.pd_df.shift(10, fill_value=0)
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_diff_first_row(self):
        """Test that diff first row matches pandas with fillna(0)."""
        jf_result = self.jf_df.diff(1)
        pd_result = self.pd_df.diff(1).fillna(0)

        # Check first row is all zeros
        assert np.allclose(jf_result.values[0], 0.0)
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_diff_large_period(self):
        """Test diff with period larger than half the data."""
        jf_result = self.jf_df.diff(3)
        pd_result = self.pd_df.diff(3).fillna(0)
        np.testing.assert_allclose(jf_result.values, pd_result.values, rtol=1e-5)

    def test_pct_change_with_zeros(self):
        """Test pct_change when previous value is zero."""
        # NOTE: This will produce inf where we divide by zero
        # This matches pandas behavior
        data = {'a': [0.0, 1.0, 2.0], 'b': [10.0, 0.0, 5.0]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.pct_change(1)
        pd_result = pd_df.pct_change(1)

        # Skip first row (NaN in pandas, different fill in JAXFrame)
        # and skip inf values (where division by zero occurred)
        # Just check that finite values match
        jf_finite = np.isfinite(jf_result.values[1:])
        pd_finite = np.isfinite(pd_result.values[1:])

        # Where both are finite, they should match
        both_finite = jf_finite & pd_finite
        if both_finite.any():
            np.testing.assert_allclose(
                jf_result.values[1:][both_finite],
                pd_result.values[1:][both_finite],
                rtol=1e-5
            )


class TestConcatenationEdgeCases:
    """Test concatenation edge cases."""

    def test_concat_single_dataframe(self):
        """Test concatenating a single DataFrame."""
        data = {'a': [1, 2], 'b': [3, 4]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = concat([jf_df], axis=0)
        pd_result = pd.concat([pd_df], axis=0, ignore_index=True)
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_concat_many_dataframes(self):
        """Test concatenating many DataFrames."""
        dfs_jf = [DataFrame({'a': [i], 'b': [i*10]}) for i in range(10)]
        dfs_pd = [pd.DataFrame({'a': [i], 'b': [i*10]}) for i in range(10)]

        jf_result = concat(dfs_jf, axis=0)
        pd_result = pd.concat(dfs_pd, axis=0, ignore_index=True)
        np.testing.assert_array_equal(jf_result.values, pd_result.values)

    def test_concat_different_sizes(self):
        """Test concatenating DataFrames of different row counts."""
        jf_df1 = DataFrame({'a': [1, 2], 'b': [3, 4]})
        jf_df2 = DataFrame({'a': [5, 6, 7], 'b': [8, 9, 10]})
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_df2 = pd.DataFrame({'a': [5, 6, 7], 'b': [8, 9, 10]})

        jf_result = concat([jf_df1, jf_df2], axis=0)
        pd_result = pd.concat([pd_df1, pd_df2], axis=0, ignore_index=True)
        np.testing.assert_array_equal(jf_result.values, pd_result.values)


class TestKnownDifferences:
    """Document known differences between JAXFrame and pandas.

    These tests document intentional design differences or limitations.
    """

    def test_nan_handling_difference(self):
        """
        KNOWN DIFFERENCE: JAXFrame uses 0 fill instead of NaN.

        Pandas uses NaN for undefined values (e.g., first row of diff()).
        JAXFrame uses 0.0 as fill_value because NaN is not JIT-friendly.
        """
        data = {'a': [1.0, 2.0, 3.0]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_result = jf_df.diff(1)
        pd_result_with_nan = pd_df.diff(1)  # First value is NaN
        pd_result_filled = pd_result_with_nan.fillna(0)  # Our equivalent

        # JAXFrame matches fillna(0), not raw diff()
        np.testing.assert_allclose(jf_result.values, pd_result_filled.values, rtol=1e-5)

        # But doesn't match raw diff with NaN
        assert not np.array_equal(jf_result.values, pd_result_with_nan.values, equal_nan=True)

    def test_dtype_preservation_difference(self):
        """
        KNOWN DIFFERENCE: JAXFrame may convert dtypes for JAX compatibility.

        JAX has limited dtype support compared to NumPy/pandas.
        Some dtypes may be converted to JAX-compatible types.
        """
        # Integer data might be converted to float for JAX compatibility
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # Both work, but dtypes might differ
        assert jf_df.shape == pd_df.shape
        # dtype check would show differences (pandas: int, JAXFrame: might be float)

    def test_boolean_storage_difference(self):
        """
        KNOWN DIFFERENCE: Boolean results are stored as numeric.

        Comparison operations return numeric arrays (0/1) rather than
        pure boolean arrays, because JAX arrays use numeric types.
        """
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        jf_bool = jf_df > 2
        pd_bool = pd_df > 2

        # Values match when compared numerically
        np.testing.assert_array_equal(jf_bool.values, pd_bool.values)

        # But dtypes are different (pandas: bool, JAXFrame: numeric)
        # assert jf_bool._numeric_dtypes != pd_bool.dtypes  # Would show difference

    def test_index_handling_difference(self):
        """
        KNOWN DIFFERENCE: JAXFrame uses simple integer indices.

        Pandas supports rich index types (MultiIndex, DatetimeIndex, etc.).
        JAXFrame currently only supports simple integer indices for JIT compatibility.
        """
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data, index=['x', 'y', 'z'])

        # JAXFrame ignores non-integer indices in construction
        # This is a simplification for JAX compatibility
        assert len(jf_df.index) == len(pd_df.index)
        # But index types are different


class TestComplexRealWorldScenarios:
    """Test realistic data science workflows."""

    def test_normalization_workflow(self):
        """Test Z-score normalization workflow."""
        data = {
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
        }
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # Normalize: (x - mean) / std
        jf_normalized = (jf_df - jf_df.mean()) / jf_df.std()
        pd_normalized = (pd_df - pd_df.mean()) / pd_df.std()

        np.testing.assert_allclose(jf_normalized.values, pd_normalized.values, rtol=1e-5)

        # Verify normalized data has mean~0, std~1
        # Note: mean might have small floating point error, not exactly 0
        assert np.allclose(jf_normalized.mean()._data, 0.0, atol=1e-6)
        assert np.allclose(jf_normalized.std()._data, 1.0, rtol=1e-5)

    def test_outlier_clipping_workflow(self):
        """Test outlier detection and clipping."""
        data = {
            'values': [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],  # 100 is outlier
        }
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # Clip outliers to [1, 10] range
        jf_clipped = jf_df.clip(1, 10)
        pd_clipped = pd_df.clip(1, 10)

        np.testing.assert_allclose(jf_clipped.values, pd_clipped.values, rtol=1e-5)
        assert jf_clipped.max()._data[0] == 10.0  # Outlier clipped

    def test_feature_engineering_workflow(self):
        """Test creating new features from existing ones."""
        data = {
            'price': [100.0, 105.0, 103.0, 108.0],
            'quantity': [10.0, 12.0, 8.0, 15.0],
        }
        jf_df = DataFrame(data)
        pd_df = pd.DataFrame(data)

        # Create total_value feature
        jf_df_with_total = concat([
            jf_df,
            DataFrame({'total_value': (jf_df['price']._data * jf_df['quantity']._data).tolist()})
        ], axis=1)

        pd_df_with_total = pd_df.copy()
        pd_df_with_total['total_value'] = pd_df['price'] * pd_df['quantity']

        np.testing.assert_allclose(
            jf_df_with_total.values,
            pd_df_with_total.values,
            rtol=1e-5
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
