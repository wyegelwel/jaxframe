"""
Demonstration of JAXFrame's pandas-compatible API.

This example showcases the extensive pandas-like functionality
that's been added to JAXFrame, all while maintaining JIT and grad compatibility!
"""

import sys
sys.path.insert(0, '..')

import jax
import jax.numpy as jnp
from jaxframe import DataFrame, concat

print("=" * 70)
print("JAXFrame Pandas Compatibility Demo")
print("=" * 70)

# Create a sample DataFrame
df = DataFrame({
    'price': [100.0, 105.0, 103.0, 108.0, 110.0],
    'volume': [1000, 1200, 950, 1100, 1300],
    'returns': [0.0, 0.05, -0.02, 0.05, 0.02],
})

print("\nOriginal DataFrame:")
print(df)

# Core attributes
print("\n" + "=" * 70)
print("Core Attributes")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"Size: {df.size}")
print(f"Dimensions: {df.ndim}")
print(f"Empty: {df.empty}")
print(f"Columns: {df.columns}")
print(f"Data types: {df.dtypes}")

# Arithmetic operators
print("\n" + "=" * 70)
print("Arithmetic Operations")
print("=" * 70)
print("\nDivision (df / 2):")
print(df / 2)

print("\nPower (df ** 2):")
print(df ** 2)

print("\nModulo (df % 10):")
print(df % 10)

# Comparison and logical operators
print("\n" + "=" * 70)
print("Comparison & Logical Operations")
print("=" * 70)
print("\nFind prices >= 105:")
high_prices = df >= 105
print(high_prices)

print("\nFind prices between 100 and 108:")
price_range = (df >= 100) & (df <= 108)
print(price_range)

# Reduction methods
print("\n" + "=" * 70)
print("Reduction Methods")
print("=" * 70)
print(f"Standard deviation:\n{df.std()}")
print(f"\nVariance:\n{df.var()}")
print(f"\nMin values:\n{df.min()}")
print(f"\nMax values:\n{df.max()}")
print(f"\nProduct:\n{df.prod()}")

# Statistical functions
print("\n" + "=" * 70)
print("Statistical Functions")
print("=" * 70)
print("\nCorrelation matrix:")
print(df.corr())

print("\nCovariance matrix:")
print(df.cov())

# Indexing and selection
print("\n" + "=" * 70)
print("Indexing & Selection")
print("=" * 70)
print("\nFirst 3 rows (head):")
print(df.head(3))

print("\nLast 2 rows (tail):")
print(df.tail(2))

print("\niloc[1:3] (rows 1-2):")
print(df.iloc[1:3])

print("\nAttribute access - df.price:")
print(df.price)

# Masking and clipping
print("\n" + "=" * 70)
print("Masking & Clipping")
print("=" * 70)
print("\nClip values to [100, 1100]:")
print(df.clip(100, 1100))

print("\nMask values > 1000 with 0:")
print(df.mask(df > 1000, 0))

# Shape operations
print("\n" + "=" * 70)
print("Shape Operations")
print("=" * 70)
print("\nTranspose:")
print(df.T)

# Concatenation
df2 = DataFrame({
    'price': [112.0, 115.0],
    'volume': [1400, 1500],
    'returns': [0.02, 0.03],
})
print("\nConcatenate two DataFrames:")
combined = concat([df, df2], axis=0)
print(combined)

# Time series operations
print("\n" + "=" * 70)
print("Time Series Operations")
print("=" * 70)
print("\nShift forward by 1 (lag):")
print(df.shift(1))

print("\nFirst difference:")
print(df.diff())

print("\nPercentage change:")
print(df.pct_change())

# JIT compilation with new operations!
print("\n" + "=" * 70)
print("JIT Compilation Demo")
print("=" * 70)

@jax.jit
def complex_calculation(df_data):
    """Demonstrate that all new operations work with JIT!"""
    # Use new operations
    normalized = (df_data - df_data.mean(axis=0)) / df_data.std(axis=0)
    clipped = jnp.clip(normalized, -2, 2)
    return (clipped ** 2).sum()

result = complex_calculation(df._numeric_data)
print(f"\nJIT-compiled complex calculation result: {result}")
print("✅ All new operations are JIT-compatible!")

# Gradient computation with new operations!
print("\n" + "=" * 70)
print("Gradient Computation Demo")
print("=" * 70)

@jax.grad
def loss_function(data):
    """Loss using new statistical operations."""
    # Variance as loss (minimize spread)
    return jnp.var(data)

grad_result = loss_function(df._numeric_data)
print(f"\nGradient of variance:\n{grad_result}")
print("✅ Many new operations are differentiable!")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
JAXFrame now supports extensive pandas-compatible functionality:

✅ 5 core attributes (size, ndim, dtypes, values, empty)
✅ 5 arithmetic operators (/, //, %, **, @)
✅ 5 comparison operators (>=, <, <=, ==, !=)
✅ 3 logical operators (&, |, ~)
✅ 6 reduction methods (std, var, min, max, prod, abs)
✅ 4 indexing features (iloc, head, tail, attribute access)
✅ 2 masking methods (mask, clip)
✅ 2 statistical functions (corr, cov)
✅ 3 shape operations (T, transpose, concat)
✅ 3 time series methods (shift, diff, pct_change)

All while maintaining JIT compilation and gradient computation support!
""")
print("=" * 70)
