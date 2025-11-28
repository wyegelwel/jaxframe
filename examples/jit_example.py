"""
Example: JIT Compilation with JAXFrame

This example demonstrates how to use JAX's JIT compilation
with DataFrames for significant performance improvements.
"""

import jax
import jax.numpy as jnp
import time
import sys
sys.path.insert(0, '..')

from jaxframe import DataFrame


def example_basic_jit():
    """Basic JIT compilation example."""
    print("=" * 60)
    print("Example 1: Basic JIT Compilation")
    print("=" * 60)

    # Create a DataFrame
    df = DataFrame({
        'price': [10.0, 20.0, 30.0, 40.0, 50.0],
        'quantity': [1.0, 2.0, 3.0, 4.0, 5.0],
        'discount': [0.1, 0.15, 0.2, 0.05, 0.1],
    })

    print("\nOriginal DataFrame:")
    print(df)

    # Define a computation
    def compute_revenue(df):
        """Calculate revenue with discount applied."""
        return (df['price'].values * df['quantity'].values * (1 - df['discount'].values)).sum()

    # Without JIT
    result = compute_revenue(df)
    print(f"\nRevenue (no JIT): {result}")

    # With JIT - operates on the whole DataFrame
    @jax.jit
    def compute_revenue_jit(df):
        """JIT-compiled revenue calculation."""
        # For JIT, we work with the numeric data directly
        price_qty_product = df._numeric_data[:, 0] * df._numeric_data[:, 1]
        discount_factor = 1 - df._numeric_data[:, 2]
        return (price_qty_product * discount_factor).sum()

    result_jit = compute_revenue_jit(df)
    print(f"Revenue (with JIT): {result_jit}")
    print("\n✅ JIT compilation successful!")


def example_arithmetic_operations():
    """JIT with DataFrame arithmetic operations."""
    print("\n" + "=" * 60)
    print("Example 2: JIT with DataFrame Operations")
    print("=" * 60)

    df = DataFrame({
        'x': [1.0, 2.0, 3.0, 4.0],
        'y': [2.0, 4.0, 6.0, 8.0],
    })

    print("\nOriginal DataFrame:")
    print(df)

    # JIT-compiled function using DataFrame operations
    @jax.jit
    def compute(df):
        """Complex computation on DataFrame."""
        result = (df * 2 + 10) * 0.5
        return result.sum(axis=1)

    result = compute(df)
    print(f"\nResult: {result}")
    print("✅ DataFrame operations work in JIT!")


def example_performance_comparison():
    """Compare performance with and without JIT."""
    print("\n" + "=" * 60)
    print("Example 3: Performance Comparison")
    print("=" * 60)

    # Create larger DataFrame
    n = 10000
    df = DataFrame({
        'a': jnp.array(range(n), dtype=jnp.float64),
        'b': jnp.array(range(n, 2*n), dtype=jnp.float64),
        'c': jnp.array(range(2*n, 3*n), dtype=jnp.float64),
    })

    print(f"\nDataFrame shape: {df.shape}")

    # Complex computation
    def complex_computation(df):
        """A more complex computation for benchmarking."""
        result = df._numeric_data
        for _ in range(10):
            result = jnp.sin(result) * 2 + jnp.cos(result)
        return result.sum()

    # Without JIT
    start = time.time()
    for _ in range(100):
        result_no_jit = complex_computation(df)
    time_no_jit = time.time() - start

    # With JIT
    jitted_fn = jax.jit(complex_computation)

    # Warmup (first call compiles)
    _ = jitted_fn(df)

    start = time.time()
    for _ in range(100):
        result_jit = jitted_fn(df)
    time_jit = time.time() - start

    print(f"\nTime without JIT: {time_no_jit:.4f}s")
    print(f"Time with JIT: {time_jit:.4f}s")
    print(f"Speedup: {time_no_jit / time_jit:.2f}x")
    print("\n✅ JIT provides significant speedup!")


def example_where_operation():
    """JIT-compatible conditional operations."""
    print("\n" + "=" * 60)
    print("Example 4: JIT-Compatible Filtering with where()")
    print("=" * 60)

    df = DataFrame({
        'price': [10.0, 150.0, 30.0, 200.0, 50.0],
        'quantity': [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    print("\nOriginal DataFrame:")
    print(df)

    @jax.jit
    def apply_discount(df, threshold):
        """
        Apply discount to high-value items.

        Uses where() instead of boolean indexing to maintain fixed shape.
        """
        high_value = df > threshold
        # Set low-value items to 0 (or could use any fill value)
        return df.where(high_value, fill_value=0.0)

    result = apply_discount(df, 100.0)
    print(f"\nAfter filtering (threshold=100):")
    print(result)
    print("\n✅ where() operation is JIT-compatible!")


if __name__ == "__main__":
    example_basic_jit()
    example_arithmetic_operations()
    example_performance_comparison()
    example_where_operation()

    print("\n" + "=" * 60)
    print("All JIT examples completed successfully! 🎉")
    print("=" * 60)
