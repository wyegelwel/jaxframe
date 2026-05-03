# JAXFrame Design Document

A JAX-based library mirroring the pandas API with support for automatic differentiation and JIT compilation.

## Core Architecture: Dtype Blocks

### Data Structure

```python
class DataFrame:
    _dtype_blocks: dict[np.dtype, jnp.ndarray]   # dtype → 2D JAX array (n_rows, n_cols)
    _column_to_block: dict[str, tuple[np.dtype, int]]  # col → (dtype, index_in_block)
    _object_data: dict[str, np.ndarray]           # non-numeric columns (strings, etc.)
    _index: np.ndarray                             # row index
    _column_order: tuple[str, ...]                 # preserves user column order
```

Numeric columns are grouped by dtype into 2D JAX arrays ("blocks"). Each column maps to its block via `_column_to_block`. Object (non-numeric) columns are stored separately in numpy arrays.

Example:
```python
df = DataFrame({
    'price': [10.0, 20.0],    # float64 → dtype_blocks[float64][:, 0]
    'volume': [100, 200],      # int32   → dtype_blocks[int32][:, 0]
    'name': ['A', 'B'],        # object  → object_data['name']
})
```

### Key Internals

- **`_from_parts()`** — Class method to construct DataFrames without going through `__init__` (which calls `np.asarray`, breaking JAX tracing). All internal operations use this.
- **`_apply_blockwise(fn)`** — Applies a function to each dtype block and returns a new DataFrame. Used for element-wise operations.
- **`_reduce_axis0(fn)`** — Applies a reduction function to each block along axis 0, returns a Series.

## Hybrid Eager/JIT Design

The core architectural principle: **separate structure discovery from data computation**.

- **Structure discovery** (shapes, indices, group assignments, join keys, unique values) happens **eagerly** — outside `jax.jit` traces. Use `jnp.*` (not `np.*`) so it still runs on GPU. Results become static `aux_data` in JAX pytrees.
- **Data computation** (arithmetic, reductions, gathers, scatters) uses **JAX ops** (`jnp.*`, `jax.ops.segment_*`, `jnp.take`). These are JIT-compiled and dispatch to GPU/TPU.
- **"Eager" ≠ CPU.** `jnp.unique` called eagerly on GPU data runs on GPU.

Examples:
- `groupby`: group discovery is eager → segment ops are JIT
- `sort_values`: argsort is eager → data permutation via `jnp.take` is JIT
- `rolling().mean()`: window setup is eager → prefix-sum computation is JIT

## JAX Pytree Registration

DataFrame, Series, and SeriesGroupBy are registered as JAX pytrees.

### DataFrame Pytree

```python
def _dataframe_flatten(df):
    # Children: dtype blocks (sorted by dtype name for determinism)
    children = tuple(block for _, block in sorted_dtype_items)
    # Aux: column_to_block mapping, object_data, index, column_order
    aux_data = _DataFrameAux(...)
    return children, aux_data
```

Only `_dtype_blocks` are children (participate in grad/JIT). Object data, index, and column mappings are auxiliary (static, non-differentiable).

### Series Pytree

```python
def _series_flatten(series):
    children = (series._data,)   # 1D JAX array
    aux_data = _SeriesAux(index, name)
    return children, aux_data
```

## JIT Compatibility

### Always JIT-compatible
- Element-wise: `df * 2`, `df + other_df`, comparisons
- Aggregations: `sum()`, `mean()`, `std()`, `var()`
- Window: `rolling(w).mean()`, `rolling(w).std()` (prefix-sum based, O(n))
- Transforms: `clip()`, `where()`, `fillna()`, `shift()`, `diff()`, `cumsum()`, `cumprod()`
- GroupBy aggs: `groupby().sum()`, `.mean()`, `.std()`, `.var()`

### Not JIT-compatible (structure discovery)
- `sort_values()` — uses eager `np.argsort`
- `rank()` — uses eager `np.argsort`
- `drop_duplicates()` — uses eager duplicate detection
- Dynamic filtering: `df[df['x'] > 0]` — changes shape

### JIT-compatible alternatives
```python
# Instead of: df[df['price'] > 100]
df.where(df['price'] > 100, fill_value=0)  # Fixed shape

# Instead of: df.groupby('dynamic_key')
df.groupby('static_key')  # Encoded categories, eager group discovery
```

## Automatic Differentiation

All numeric operations that produce float outputs support `jax.grad`:
- Arithmetic, sum, mean, std, var, clip, where, fillna
- cumsum, cumprod, shift, diff, apply
- groupby.sum, groupby.mean, groupby.std, groupby.var, groupby.transform

Non-differentiable operations (with reasons):
- `min`, `max` — non-smooth
- `median`, `quantile` — sort-based, non-smooth
- `isna`, `notna`, `count`, `all`, `any` — boolean/integer output
- `sort_values`, `rank` — discrete (argsort)
- `round` — step function (zero gradient)
- `idxmin`, `idxmax` — discrete

## Rolling Window Implementation

Rolling operations use **O(n) prefix sums** instead of O(n*w) gather matrices:

```python
# Shared setup
cum = cumsum(clean_data)       # prefix sums of values
cum_valid = cumsum(valid_mask)  # prefix sums of valid counts

# Rolling sum = cum[i] - cum[i - window]
rolling_sum = cum[end_indices] - cum[start_indices]
rolling_count = cum_valid[end_indices] - cum_valid[start_indices]
```

This gives O(n) time and memory for sum, mean, var, std. Rolling min/max still use O(n*w) gather (no prefix-sum equivalent).

## Missing Data

Uses `jnp.nan` sentinel values (IEEE 754 NaN). Custom fused `lax.reduce` combiners for NaN-skipping reductions (`_fast_nansum`, `_fast_nanmin`, `_fast_nanmax`, `_fast_nanmean_var_std`) that embed NaN checks in the reduction for single-pass performance.

## File Structure

| File | Purpose |
|------|---------|
| `jaxframe/dataframe.py` | Core: DataFrame, Series, GroupBy, Rolling, Expanding, EWM, pytree registration |
| `tests/test_pandas_mirror.py` | Parametrized pandas equivalence tests |
| `tests/test_jax_transforms.py` | JIT/grad compatibility matrix |
| `tests/test_benchmarks.py` | Performance comparison: jaxframe vs pandas |
| `tests/conftest.py` | Shared fixtures and `run_equiv()` helper |
