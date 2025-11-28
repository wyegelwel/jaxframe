"""
Equivalence Testing Framework

Tests that JAXFrame operations match pandas behavior exactly.
This framework is the foundation for ensuring API compatibility.
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal, assert_allclose
import pytest

from jaxframe import DataFrame


class EquivalenceTest:
    """Base class for pandas-jaxframe equivalence testing."""

    def compare_dataframes(self, pandas_df, jaxframe_df, rtol=1e-5, atol=1e-8):
        """
        Compare pandas DataFrame with jaxframe DataFrame.

        Args:
            pandas_df: pandas DataFrame
            jaxframe_df: jaxframe DataFrame
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        # Check shapes match
        assert pandas_df.shape == jaxframe_df.shape, \
            f"Shape mismatch: pandas {pandas_df.shape} vs jaxframe {jaxframe_df.shape}"

        # Compare numeric values
        pandas_values = pandas_df.values
        jaxframe_values = np.array(jaxframe_df._numeric_data)

        assert_allclose(
            pandas_values,
            jaxframe_values,
            rtol=rtol,
            atol=atol,
            err_msg="DataFrame values don't match"
        )

    def compare_series(self, pandas_series, jaxframe_series, rtol=1e-5, atol=1e-8):
        """Compare pandas Series with jaxframe Series."""
        pandas_values = pandas_series.values
        jaxframe_values = np.array(jaxframe_series.values)

        assert_allclose(
            pandas_values,
            jaxframe_values,
            rtol=rtol,
            atol=atol,
            err_msg="Series values don't match"
        )

    def compare_scalars(self, pandas_scalar, jaxframe_scalar, rtol=1e-5, atol=1e-8):
        """Compare scalar values."""
        assert_allclose(
            pandas_scalar,
            jaxframe_scalar,
            rtol=rtol,
            atol=atol,
            err_msg="Scalar values don't match"
        )

    def run_comparison(self, pandas_op, jaxframe_op, data):
        """
        Run operation on both pandas and jaxframe, compare results.

        Args:
            pandas_op: Function taking pandas DataFrame
            jaxframe_op: Function taking jaxframe DataFrame
            data: Dictionary of column data
        """
        # Create DataFrames
        pdf = pd.DataFrame(data)
        jdf = DataFrame(data)

        # Run operations
        pandas_result = pandas_op(pdf)
        jaxframe_result = jaxframe_op(jdf)

        # Compare based on result type
        if isinstance(pandas_result, pd.DataFrame):
            self.compare_dataframes(pandas_result, jaxframe_result)
        elif isinstance(pandas_result, pd.Series):
            self.compare_series(pandas_result, jaxframe_result)
        else:
            self.compare_scalars(pandas_result, jaxframe_result)

        return pandas_result, jaxframe_result


class TestArithmeticOperations(EquivalenceTest):
    """Test that arithmetic operations match pandas."""

    def test_addition_scalar(self):
        """Test df + scalar."""
        data = {'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]}
        self.run_comparison(
            pandas_op=lambda df: df + 10,
            jaxframe_op=lambda df: df + 10,
            data=data
        )

    def test_subtraction_scalar(self):
        """Test df - scalar."""
        data = {'a': [10.0, 20.0, 30.0], 'b': [40.0, 50.0, 60.0]}
        self.run_comparison(
            pandas_op=lambda df: df - 5,
            jaxframe_op=lambda df: df - 5,
            data=data
        )

    def test_multiplication_scalar(self):
        """Test df * scalar."""
        data = {'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]}
        self.run_comparison(
            pandas_op=lambda df: df * 2,
            jaxframe_op=lambda df: df * 2,
            data=data
        )

    def test_multiplication_dataframe(self):
        """Test df * df."""
        data = {'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]}

        pdf = pd.DataFrame(data)
        jdf = DataFrame(data)

        pandas_result = pdf * pdf
        jaxframe_result = jdf * jdf

        self.compare_dataframes(pandas_result, jaxframe_result)


class TestAggregations(EquivalenceTest):
    """Test that aggregation operations match pandas."""

    def test_sum_axis0(self):
        """Test df.sum(axis=0)."""
        data = {'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]}
        self.run_comparison(
            pandas_op=lambda df: df.sum(axis=0),
            jaxframe_op=lambda df: df.sum(axis=0),
            data=data
        )

    def test_sum_axis1(self):
        """Test df.sum(axis=1)."""
        data = {'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]}
        self.run_comparison(
            pandas_op=lambda df: df.sum(axis=1),
            jaxframe_op=lambda df: df.sum(axis=1),
            data=data
        )

    def test_mean_axis0(self):
        """Test df.mean(axis=0)."""
        data = {'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]}
        self.run_comparison(
            pandas_op=lambda df: df.mean(axis=0),
            jaxframe_op=lambda df: df.mean(axis=0),
            data=data
        )


class TestJAXTransforms:
    """Test JAX transform compatibility."""

    def test_jit_sum(self):
        """Test that sum can be JIT compiled."""
        df = DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})

        @jax.jit
        def compute_sum(df):
            return df.sum(axis=None)

        result = compute_sum(df)
        expected = 21.0

        assert_allclose(result, expected)

    def test_jit_arithmetic(self):
        """Test that arithmetic can be JIT compiled."""
        df = DataFrame({'a': [1.0, 2.0, 3.0]})

        @jax.jit
        def compute(df):
            return (df * 2 + 10).sum(axis=None)

        result = compute(df)
        # (1*2+10) + (2*2+10) + (3*2+10) = 12 + 14 + 16 = 42
        expected = 42.0

        assert_allclose(result, expected)

    def test_grad_sum(self):
        """Test that sum is differentiable."""
        df = DataFrame({'a': [1.0, 2.0, 3.0]})

        def loss(df):
            return df._numeric_data.sum()

        grad_fn = jax.grad(loss)
        grads = grad_fn(df)

        # Gradient of sum is 1 everywhere
        expected = np.ones((3, 1))
        assert_allclose(grads._numeric_data, expected)

    def test_grad_squared(self):
        """Test gradient of squared sum."""
        df = DataFrame({'a': [1.0, 2.0, 3.0]})

        def loss(df):
            return (df._numeric_data ** 2).sum()

        grad_fn = jax.grad(loss)
        grads = grad_fn(df)

        # Gradient of x^2 is 2x
        expected = np.array([[2.0], [4.0], [6.0]])
        assert_allclose(grads._numeric_data, expected)

    def test_jit_and_grad_together(self):
        """Test that JIT and grad can be composed."""
        df = DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})

        @jax.jit
        @jax.grad
        def compute(df):
            return (df._numeric_data ** 2).sum()

        grads = compute(df)

        # Should have same shape as input
        assert grads._numeric_data.shape == (3, 2)


class TestCompatibilityMatrix:
    """
    Test the full compatibility matrix for all operations.

    This uses pytest parametrize to test all combinations.
    """

    # Define operations with their compatibility
    OPERATIONS = [
        # (name, operation, supports_jit, supports_grad, expected_result)
        ('sum', lambda df: df.sum(axis=None), True, True, 21.0),
        ('mean', lambda df: df.mean(axis=None), True, True, 3.5),
        ('multiply', lambda df: (df * 2).sum(axis=None), True, True, 42.0),
        ('add', lambda df: (df + 10).sum(axis=None), True, True, 81.0),
        ('where', lambda df: df.where(df > 3, 0).sum(axis=None), True, True, 15.0),
    ]

    @pytest.mark.parametrize("name,op,jit_ok,grad_ok,expected", OPERATIONS)
    def test_operation_jit(self, name, op, jit_ok, grad_ok, expected):
        """Test if operation supports JIT."""
        df = DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})

        if jit_ok:
            jitted = jax.jit(op)
            result = jitted(df)
            assert_allclose(result, expected, err_msg=f"{name} JIT failed")
        else:
            pytest.skip(f"{name} doesn't support JIT")

    @pytest.mark.parametrize("name,op,jit_ok,grad_ok,expected", OPERATIONS)
    def test_operation_grad(self, name, op, jit_ok, grad_ok, expected):
        """Test if operation supports grad."""
        df = DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})

        if grad_ok:
            grad_fn = jax.grad(op)
            grads = grad_fn(df)
            # Just check it doesn't crash and produces finite gradients
            assert jnp.all(jnp.isfinite(grads._numeric_data)), \
                f"{name} grad produced non-finite values"
        else:
            pytest.skip(f"{name} doesn't support grad")


class TestPropertyBasedEquivalence:
    """
    Property-based testing: generate random data and ensure equivalence.

    This is more comprehensive than fixed test cases.
    """

    def generate_random_data(self, n_rows=10, n_cols=3, seed=42):
        """Generate random test data."""
        np.random.seed(seed)
        return {
            f'col_{i}': np.random.randn(n_rows)
            for i in range(n_cols)
        }

    def test_sum_random_data(self):
        """Test sum on random data."""
        for seed in range(10):  # Test 10 random datasets
            data = self.generate_random_data(seed=seed)

            pdf = pd.DataFrame(data)
            jdf = DataFrame(data)

            pandas_sum = pdf.sum().sum()
            jaxframe_sum = jdf.sum(axis=None)

            assert_allclose(pandas_sum, jaxframe_sum, rtol=1e-5)

    def test_mean_random_data(self):
        """Test mean on random data."""
        for seed in range(10):
            data = self.generate_random_data(seed=seed)

            pdf = pd.DataFrame(data)
            jdf = DataFrame(data)

            pandas_mean = pdf.mean().mean()
            jaxframe_mean = jdf.mean(axis=None)

            assert_allclose(pandas_mean, jaxframe_mean, rtol=1e-5)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
