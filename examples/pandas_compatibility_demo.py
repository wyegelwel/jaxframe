"""
Demonstration of JAXFrame's pandas-compatible API.

Showcases the extensive pandas-like functionality, all while
maintaining JIT and grad compatibility.
"""

import jax

from jaxframe import DataFrame, concat

print("=" * 70)
print("JAXFrame Pandas Compatibility Demo")
print("=" * 70)

# Create a sample DataFrame
df = DataFrame({
    "price": [100.0, 105.0, 103.0, 108.0, 110.0],
    "volume": [1000, 1200, 950, 1100, 1300],
    "returns": [0.0, 0.05, -0.02, 0.05, 0.02],
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
print(df**2)

# Comparison
print("\n" + "=" * 70)
print("Comparison & Logical Operations")
print("=" * 70)
print("\nFind prices >= 105:")
print(df >= 105)

# Reduction methods
print("\n" + "=" * 70)
print("Reduction Methods")
print("=" * 70)
print(f"Standard deviation:\n{df.std()}")
print(f"\nMin values:\n{df.min()}")
print(f"\nMax values:\n{df.max()}")

# Statistical functions
print("\n" + "=" * 70)
print("Statistical Functions")
print("=" * 70)
print("\nCorrelation matrix:")
print(df.corr())

# Indexing and selection
print("\n" + "=" * 70)
print("Indexing & Selection")
print("=" * 70)
print("\nFirst 3 rows (head):")
print(df.head(3))
print("\nLast 2 rows (tail):")
print(df.tail(2))
print("\niloc[1:3]:")
print(df.iloc[1:3])

# Masking and clipping
print("\n" + "=" * 70)
print("Masking & Clipping")
print("=" * 70)
print("\nClip values to [100, 1100]:")
print(df.clip(100, 1100))

# Shape operations
print("\n" + "=" * 70)
print("Shape Operations")
print("=" * 70)
print("\nTranspose:")
print(df.T)

# Concatenation
df2 = DataFrame({
    "price": [112.0, 115.0],
    "volume": [1400, 1500],
    "returns": [0.02, 0.03],
})
print("\nConcatenate two DataFrames:")
combined = concat([df, df2], axis=0)
print(combined)

# Time series operations
print("\n" + "=" * 70)
print("Time Series Operations")
print("=" * 70)
print("\nShift forward by 1:")
print(df.shift(1))
print("\nFirst difference:")
print(df.diff())
print("\nRolling mean (window=3):")
print(df.rolling(3).mean())

# JIT compilation
print("\n" + "=" * 70)
print("JIT Compilation Demo")
print("=" * 70)


@jax.jit
def normalize_and_score(df):
    """Normalize columns and compute a score."""
    normalized = (df - df.mean(axis=0)) / df.std(axis=0)
    return normalized.sum(axis=None)


result = normalize_and_score(df)
print(f"\nJIT-compiled normalize+score: {result}")

# Gradient computation
print("\n" + "=" * 70)
print("Gradient Computation Demo")
print("=" * 70)


def variance_loss(df):
    """Loss = total variance across all columns."""
    return df.var(axis=None)


grad_fn = jax.grad(variance_loss)
df_float = df.astype(float)  # grad requires float inputs
grad_result = grad_fn(df_float)
print("\nGradient of variance loss:")
print(grad_result)

print("\n" + "=" * 70)
print("Summary: JAXFrame supports the pandas API with JIT + grad!")
print("=" * 70)
