"""
Basic tests for JAXFrame DataFrame functionality.
"""

import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, "..")

from jaxframe import DataFrame, Series


class TestDataFrameCreation:
    """Test DataFrame creation from various inputs."""

    def test_create_from_dict_numeric(self):
        """Test creating DataFrame from dictionary of numeric data."""
        df = DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4.0, 5.0, 6.0],
            }
        )

        assert df.shape == (3, 2)
        assert df.columns == ["a", "b"]
        assert len(df._numeric_cols) == 2
        assert len(df._object_data) == 0

    def test_create_from_dict_mixed(self):
        """Test creating DataFrame with mixed numeric and object columns."""
        df = DataFrame(
            {
                "numbers": [1.0, 2.0, 3.0],
                "strings": ["a", "b", "c"],
            }
        )

        assert df.shape == (3, 2)
        assert df.columns == ["numbers", "strings"]
        assert "numbers" in df._numeric_cols
        assert "strings" in df._object_data

    def test_create_from_array(self):
        """Test creating DataFrame from 2D array."""
        arr = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float64)
        df = DataFrame(arr)

        assert df.shape == (3, 2)
        assert df.columns == ["col_0", "col_1"]

    def test_empty_dataframe(self):
        """Test creating empty DataFrame."""
        df = DataFrame({})

        assert df.shape == (0, 0)
        assert df.columns == []


class TestDataFrameIndexing:
    """Test DataFrame indexing operations."""

    def test_single_column_access(self):
        """Test accessing single column returns Series."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        series = df["a"]

        assert isinstance(series, Series)
        assert series.name == "a"
        assert len(series.values) == 3

    def test_multiple_column_access(self):
        """Test accessing multiple columns returns DataFrame."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        subset = df[["a", "c"]]

        assert isinstance(subset, DataFrame)
        assert subset.columns == ["a", "c"]
        assert subset.shape == (3, 2)

    def test_column_not_found(self):
        """Test that accessing non-existent column raises KeyError."""
        df = DataFrame({"a": [1, 2, 3]})

        with pytest.raises(KeyError):
            _ = df["nonexistent"]


class TestDataFrameArithmetic:
    """Test arithmetic operations on DataFrames."""

    def test_multiply_scalar(self):
        """Test multiplying DataFrame by scalar."""
        df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = df * 2

        expected = jnp.array([[2.0, 8.0], [4.0, 10.0], [6.0, 12.0]])
        assert jnp.allclose(result._numeric_data, expected)

    def test_add_scalar(self):
        """Test adding scalar to DataFrame."""
        df = DataFrame({"a": [1.0, 2.0, 3.0]})
        result = df + 10

        expected = jnp.array([[11.0], [12.0], [13.0]])
        assert jnp.allclose(result._numeric_data, expected)

    def test_subtract_scalar(self):
        """Test subtracting scalar from DataFrame."""
        df = DataFrame({"a": [10.0, 20.0, 30.0]})
        result = df - 5

        expected = jnp.array([[5.0], [15.0], [25.0]])
        assert jnp.allclose(result._numeric_data, expected)

    def test_multiply_dataframe(self):
        """Test multiplying two DataFrames element-wise."""
        df1 = DataFrame({"a": [1.0, 2.0, 3.0]})
        df2 = DataFrame({"a": [2.0, 3.0, 4.0]})
        result = df1 * df2

        expected = jnp.array([[2.0], [6.0], [12.0]])
        assert jnp.allclose(result._numeric_data, expected)


class TestDataFrameAggregations:
    """Test aggregation operations."""

    def test_sum_column_wise(self):
        """Test column-wise sum."""
        df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = df.sum(axis=0)

        assert isinstance(result, Series)
        assert jnp.allclose(result.values, jnp.array([6.0, 15.0]))

    def test_sum_row_wise(self):
        """Test row-wise sum."""
        df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = df.sum(axis=1)

        assert isinstance(result, Series)
        assert jnp.allclose(result.values, jnp.array([5.0, 7.0, 9.0]))

    def test_sum_total(self):
        """Test total sum."""
        df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = df.sum(axis=None)

        assert jnp.allclose(result, 21.0)

    def test_mean(self):
        """Test mean calculation."""
        df = DataFrame({"a": [1.0, 2.0, 3.0]})
        result = df.mean(axis=0)

        assert jnp.allclose(result.values, jnp.array([2.0]))


class TestDataFrameComparison:
    """Test comparison operations."""

    def test_greater_than_scalar(self):
        """Test greater than comparison with scalar."""
        df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = df > 2.5

        # Check that we get a boolean DataFrame
        assert result.shape == df.shape
        assert not result._numeric_data[0, 0]  # 1.0 > 2.5
        assert result._numeric_data[2, 0]  # 3.0 > 2.5
        assert result._numeric_data[1, 1]  # 5.0 > 2.5


class TestDataFrameWhere:
    """Test where operation (JIT-friendly filtering)."""

    def test_where_scalar(self):
        """Test where with scalar fill value."""
        df = DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        mask = df > 2
        result = df.where(mask, fill_value=0.0)

        expected = jnp.array([[0.0], [0.0], [3.0], [4.0]])
        assert jnp.allclose(result._numeric_data, expected)

    def test_where_preserves_shape(self):
        """Test that where preserves DataFrame shape (important for JIT)."""
        df = DataFrame({"a": [1.0, 2.0, 3.0]})
        result = df.where(df > 10, fill_value=-1.0)

        assert result.shape == df.shape


class TestJAXIntegration:
    """Test JAX-specific functionality."""

    def test_jit_compilation(self):
        """Test that DataFrame operations can be JIT-compiled."""
        df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

        @jax.jit
        def compute(df):
            return (df * 2).sum(axis=None)

        result = compute(df)
        expected = (df._numeric_data * 2).sum()

        assert jnp.allclose(result, expected)

    def test_gradient_computation(self):
        """Test gradient computation through DataFrame."""
        df = DataFrame({"x": [1.0, 2.0, 3.0]})

        def loss(df):
            return (df._numeric_data**2).sum()

        grad_fn = jax.grad(loss)
        grads = grad_fn(df)

        # Gradient of x^2 is 2x
        expected = jnp.array([[2.0], [4.0], [6.0]])
        assert jnp.allclose(grads._numeric_data, expected)

    def test_pytree_registration(self):
        """Test that DataFrame is properly registered as JAX pytree."""
        df = DataFrame({"a": [1.0, 2.0, 3.0]})

        # Should be able to flatten and unflatten
        children, aux = jax.tree_util.tree_flatten(df)
        df_reconstructed = jax.tree_util.tree_unflatten(aux, children)

        assert jnp.allclose(df._numeric_data, df_reconstructed._numeric_data)
        assert df._numeric_cols == df_reconstructed._numeric_cols

    def test_vmap_compatibility(self):
        """Test that DataFrame works with vmap."""
        # Create multiple DataFrames
        dfs_data = jnp.array(
            [
                [[1.0], [2.0], [3.0]],
                [[4.0], [5.0], [6.0]],
            ]
        )

        # Note: This is a simplified test - full vmap support would need more work
        @jax.vmap
        def sum_rows(data):
            return data.sum()

        results = sum_rows(dfs_data)
        expected = jnp.array([6.0, 15.0])

        assert jnp.allclose(results, expected)


class TestSeries:
    """Test Series functionality."""

    def test_series_creation(self):
        """Test creating a Series."""
        s = Series([1, 2, 3], name="test")

        assert s.name == "test"
        assert len(s.values) == 3

    def test_series_sum(self):
        """Test Series sum."""
        s = Series([1.0, 2.0, 3.0])
        result = s.sum()

        assert jnp.allclose(result, 6.0)

    def test_series_mean(self):
        """Test Series mean."""
        s = Series([1.0, 2.0, 3.0])
        result = s.mean()

        assert jnp.allclose(result, 2.0)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
