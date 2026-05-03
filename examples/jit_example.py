"""
Example: JIT Compilation with JAXFrame

Demonstrates using jax.jit with DataFrames for performance.
"""

import time

import jax
import numpy as np

from jaxframe import DataFrame


def example_basic_jit():
    """Basic JIT compilation example."""
    print("=" * 60)
    print("Example 1: Basic JIT Compilation")
    print("=" * 60)

    df = DataFrame({
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        "quantity": [1.0, 2.0, 3.0, 4.0, 5.0],
        "discount": [0.1, 0.15, 0.2, 0.05, 0.1],
    })
    print("\nDataFrame:")
    print(df)

    @jax.jit
    def compute_revenue(df):
        return (df["price"] * df["quantity"] * (1 - df["discount"])).sum()

    result = compute_revenue(df)
    print(f"\nRevenue (JIT compiled): {result}")


def example_arithmetic_operations():
    """JIT with DataFrame arithmetic operations."""
    print("\n" + "=" * 60)
    print("Example 2: JIT with DataFrame Operations")
    print("=" * 60)

    df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
    print("\nDataFrame:")
    print(df)

    @jax.jit
    def compute(df):
        return ((df * 2 + 10) * 0.5).sum(axis=None)

    result = compute(df)
    print(f"\nResult: {result}")


def example_performance_comparison():
    """Compare performance with and without JIT."""
    print("\n" + "=" * 60)
    print("Example 3: Performance Comparison")
    print("=" * 60)

    n = 100_000
    data = {f"col_{i}": np.random.randn(n) for i in range(10)}
    df = DataFrame(data)
    print(f"\nDataFrame shape: {df.shape}")

    def pipeline(df):
        return ((df + 1) * 2 - 3).sum(axis=None)

    # Without JIT
    pipeline(df)  # warmup
    t0 = time.perf_counter()
    for _ in range(50):
        r = pipeline(df)
        r.block_until_ready()
    time_no_jit = (time.perf_counter() - t0) / 50 * 1000

    # With JIT
    jitted = jax.jit(pipeline)
    jitted(df).block_until_ready()  # warmup
    t0 = time.perf_counter()
    for _ in range(50):
        r = jitted(df)
        r.block_until_ready()
    time_jit = (time.perf_counter() - t0) / 50 * 1000

    print(f"\nEager: {time_no_jit:.2f}ms")
    print(f"JIT:   {time_jit:.2f}ms")
    print(f"Speedup: {time_no_jit / time_jit:.1f}x")


def example_rolling_jit():
    """JIT-compiled rolling window operations."""
    print("\n" + "=" * 60)
    print("Example 4: JIT Rolling Windows")
    print("=" * 60)

    df = DataFrame({
        "price": [10.0, 12.0, 11.0, 14.0, 13.0, 15.0, 16.0, 14.0],
    })
    print("\nPrice series:")
    print(df)

    @jax.jit
    def rolling_stats(df):
        r = df.rolling(3)
        return r.mean().sum(axis=None), r.std().sum(axis=None)

    mean_total, std_total = rolling_stats(df)
    print(f"\nRolling(3) mean total: {mean_total:.2f}")
    print(f"Rolling(3) std total:  {std_total:.2f}")

    print("\nRolling mean values:")
    print(df.rolling(3).mean())


if __name__ == "__main__":
    example_basic_jit()
    example_arithmetic_operations()
    example_performance_comparison()
    example_rolling_jit()
    print("\nAll JIT examples completed successfully!")
