# JAXFrame

A JAX-based DataFrame library that mirrors the pandas API with support for automatic differentiation and JIT compilation.

## Why JAXFrame?

**Pandas** is excellent for data manipulation but:
- No automatic differentiation
- No JIT compilation
- No GPU/TPU acceleration
- NumPy backend limits performance

**JAX** provides:
- ✅ Automatic differentiation (grad, jacobian, hessian)
- ✅ JIT compilation for speed
- ✅ Vectorization (vmap, pmap)
- ✅ GPU/TPU support

**JAXFrame** combines the best of both:
- Familiar pandas-like API
- JAX performance and transformations
- Differentiable data pipelines
- JIT-compiled analytics

## Quick Start

```python
import jax
import jax.numpy as jnp
import jaxframe as jf

# Create a DataFrame (like pandas)
df = jf.DataFrame({
    'price': [10.0, 20.0, 30.0],
    'quantity': [1, 2, 3],
    'discount': [0.1, 0.2, 0.15]
})

# Pandas-like operations
revenue = df['price'] * df['quantity']
total = revenue.sum()

# JIT compilation (100x faster!)
@jax.jit
def compute_revenue(df):
    return (df['price'] * df['quantity'] * (1 - df['discount'])).sum()

result = compute_revenue(df)

# Automatic differentiation
def loss_fn(df):
    predicted = df['price'] * 1.5
    actual = df['target']
    return ((predicted - actual) ** 2).mean()

grad_fn = jax.grad(loss_fn)
gradients = grad_fn(df)
```

## Key Features

### 1. Differentiable DataFrames

```python
# Define a loss function over DataFrames
def revenue_loss(df, weights):
    predicted_revenue = (
        df['price'] * weights['price_weight'] +
        df['quantity'] * weights['qty_weight']
    )
    return ((predicted_revenue - df['actual_revenue']) ** 2).sum()

# Compute gradients
weights = {'price_weight': 1.0, 'qty_weight': 1.0}
grads = jax.grad(revenue_loss, argnums=1)(df, weights)
```

### 2. JIT-Compiled Operations

```python
# Compile once, run fast
@jax.jit
def process_batch(df):
    normalized = (df - df.mean()) / df.std()
    return normalized.sum(axis=0)

# First call: compiles
result1 = process_batch(df1)  # ~100ms

# Subsequent calls: instant
result2 = process_batch(df2)  # ~1ms (100x faster!)
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

### 4. Vectorization

```python
# Process multiple DataFrames in parallel
@jax.vmap
def process(df):
    return df['price'].mean()

results = process(list_of_dfs)  # Parallel execution
```

## Design Principles

1. **Numeric-First**: Numeric operations get full JAX performance
2. **Gradual Degradation**: Non-JIT-able ops fall back gracefully
3. **Explicit Over Implicit**: Clear about what's JIT/grad compatible
4. **Pandas-Compatible**: Familiar API where possible

## Limitations

Not all pandas operations are supported in JIT context:

❌ **Not JIT-compatible**:
```python
df[df['price'] > 100]  # Dynamic filtering (changes shape)
df.groupby('dynamic_key')  # Unknown number of groups
```

✅ **JIT-compatible alternatives**:
```python
df.where(df['price'] > 100, fill_value=0)  # Fixed shape
df.groupby('static_key')  # Encoded categories
```

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
