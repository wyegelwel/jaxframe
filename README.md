# JAXFrame

A JAX-based DataFrame library that mirrors the pandas API with support for automatic differentiation and JIT compilation.

## Why JAXFrame?

**Pandas** is excellent for data manipulation but:
- No automatic differentiation
- No JIT compilation
- No GPU/TPU acceleration

**JAXFrame** combines the pandas API with JAX's transformations:
- **1-1 pandas API coverage**: every public `DataFrame`/`Series` member is
  implemented (enforced by `tests/test_api_coverage.py`), except a short
  declared out-of-scope list (tz/Period semantics, MultiIndex, sparse/arrow
  accessors, resample — see `scripts/api_coverage.py`)
- JIT compilation for faster repeated computations
- Automatic differentiation (`jax.grad`) through DataFrame operations —
  including sort, top-k, min/max, quantiles, ffill/interpolate (subgradients)
- Vectorization (`jax.vmap`) across batches of DataFrames
- GPU support via JAX; compute ops are JAX-native, formatting/exotic I/O
  delegate to pandas

## Quick Start

```python
import jax
import jaxframe as jf

# Create a DataFrame (like pandas)
df = jf.DataFrame({
    'price': [10.0, 20.0, 30.0],
    'quantity': [1.0, 2.0, 3.0],
    'discount': [0.1, 0.2, 0.15]
})

# Pandas-like operations
revenue = df['price'] * df['quantity']
total = revenue.sum()

# JIT compilation — compile once, run fast on repeat calls
@jax.jit
def compute_revenue(df):
    return (df['price'] * df['quantity'] * (1 - df['discount'])).sum()

result = compute_revenue(df)

# Automatic differentiation
def loss_fn(df):
    return (df['price'] ** 2).mean()

grad_fn = jax.grad(loss_fn)
gradients = grad_fn(df)  # Returns DataFrame with gradients
```

## Key Features

### 1. Differentiable DataFrames

```python
# Gradient w.r.t. DataFrame values
def mse_loss(df):
    predicted = df['feature1'] * 2 + df['feature2'] * 3
    return ((predicted - df['target']) ** 2).sum()

grad_fn = jax.grad(mse_loss)
gradients = grad_fn(df)  # ∂L/∂(each element)
```

### 2. JIT-Compiled Operations

```python
@jax.jit
def process_batch(df):
    normalized = (df - df.mean(axis=0)) / df.std(axis=0)
    return normalized.sum(axis=None)

# First call compiles; subsequent calls with same-shaped data reuse the compiled code
result = process_batch(df)
```

### 3. Mixed Numeric/Object Columns

```python
df = jf.DataFrame({
    'price': [10.0, 20.0, 30.0],      # Numeric (JAX)
    'quantity': [1, 2, 3],             # Numeric (JAX)
    'product': ['A', 'B', 'C']         # Object (NumPy)
})

# Numeric operations use JAX
df_numeric = df[['price', 'quantity']]
fast_result = jax.jit(lambda x: x * 2)(df_numeric)

# Object columns supported but not JIT-able
products = df['product']  # Returns numpy array
```

### 4. Vectorization (`vmap`)

```python
import jax.numpy as jnp

# Stack DataFrames into a batched pytree
batch = jax.tree.map(lambda *xs: jnp.stack(xs), df1, df2, df3)

# vmap processes the batch in parallel
@jax.vmap
def process(df):
    return df.sum(axis=None)

results = process(batch)
```

### 5. JAX Transform Compatibility

DataFrames are registered as JAX pytrees, so they work with the full JAX transform stack:

| Transform | Status | Notes |
|-----------|--------|-------|
| `jax.jit` | Supported | Shape-preserving ops incl. sort_values/rank/nlargest/ffill/interpolate; shape-changing + unique-based ops (filtering, dropna, merge/groupby discovery) stay eager |
| `jax.grad` | Supported | JAX subgradient convention: min/max/median/quantile/sort/top-k differentiable; boolean/integer outputs excluded |
| `jax.vmap` | Supported | Batch a function over stacked DataFrames |
| `jax.pmap` | Supported | Multi-device parallelism (same mechanism as vmap; tested but not battle-tested in production) |

## Limitations

JAXFrame uses a **hybrid eager/JIT** architecture: structure discovery (shapes, group assignments) happens eagerly, while data computation uses JIT-compiled JAX ops. This means:

**Not supported inside `jax.jit`:**
```python
df[df['price'] > 100]      # Dynamic filtering (changes shape)
df.dropna()                 # Shape depends on data
df.merge(other)             # Key matching is structure discovery
```

**Supported inside `jax.jit`** (shape-preserving): `sort_values`, `rank`,
`nlargest`/`nsmallest`, `ffill`/`bfill`/`interpolate`, `cummax`/`cummin`,
all arithmetic/reductions/rolling/expanding/ewm and groupby aggregations.

**JIT-compatible alternatives:**
```python
df.where(df['price'] > 100, 0)  # Fixed shape, NaN/fill replacement
```

**GroupBy — hybrid eager/JIT:**
```python
# Group discovery (groupby call) must happen outside jit:
gb = df.groupby('key')['value']

# But aggregation is JIT-compiled:
@jax.jit
def fast_agg(gb):
    return gb.mean()

result = fast_agg(gb)  # JIT-compiled segment ops
```

**Not currently supported:**
- UDFs via `apply` work eagerly with arbitrary Python functions, but are only JIT-compatible if the function uses JAX operations exclusively
- String operations on object columns
- MultiIndex

## Installation

```bash
pip install jaxframe          # CPU only
pip install jaxframe[cuda12]  # with CUDA 12 GPU support
```

## Documentation

See [DESIGN.md](DESIGN.md) for detailed architecture and design decisions.

## Examples

See the [examples/](examples/) directory for more:

- [Gradient computation](examples/grad_example.py)
- [JIT compilation](examples/jit_example.py)
- [Mixed column types](examples/mixed_columns_example.py)
- [Pandas compatibility](examples/pandas_compatibility_demo.py)

## License

MIT
