# JAXFrame Design Document

A JAX-based library mirroring the pandas API with support for automatic differentiation and JIT compilation.

## Core Design Challenges

### 1. JAX Constraints
- **Homogeneous arrays**: JAX arrays must have a single dtype
- **Static shapes**: JIT compilation requires compile-time known shapes
- **Pure functions**: No side effects, immutable operations
- **Numeric focus**: JAX is designed for numeric computation

### 2. Pandas Features to Support
- **Heterogeneous columns**: Mixed dtypes (int, float, string, datetime, object)
- **Dynamic operations**: Filtering, groupby, joins that change shape
- **Missing data**: NaN, None handling
- **Index alignment**: Automatic alignment on operations
- **Rich API**: 100+ methods on DataFrame and Series

## Architecture Overview

### Core Data Structure: `DataFrame`

```python
@dataclass
class DataFrame:
    # Numeric columns stored as a single JAX array for performance
    _numeric_data: Optional[jnp.ndarray]  # Shape: (n_rows, n_numeric_cols)
    _numeric_cols: tuple[str, ...]  # Column names for numeric data
    _numeric_dtypes: tuple[jnp.dtype, ...]  # Original dtypes

    # Non-numeric columns stored separately (not JAX arrays)
    _object_data: dict[str, np.ndarray]  # String, datetime, object columns

    # Index management
    _index: Index  # Row index (can be numeric or object)

    # Metadata for JIT/grad
    _pytree_metadata: dict  # For JAX pytree registration
    _differentiable_cols: set[str]  # Which columns support grad
```

### Key Design Principles

1. **Columnar Storage with Numeric/Object Split**
   - Numeric columns in a single JAX array for efficient computation
   - Object columns in separate numpy arrays (outside JAX)
   - Enables JIT on numeric operations while supporting mixed types

2. **Lazy Evaluation for Shape Changes**
   - Operations that change shape return "lazy" objects when inside JIT
   - Concrete evaluation happens outside JIT context
   - Static shape operations compile efficiently

3. **Pytree Registration**
   - Register DataFrame as JAX pytree for grad/vmap support
   - Only numeric data participates in differentiation
   - Object data passes through unchanged

## Handling Non-Numeric Columns

### Strategy 1: Dual Storage (Recommended)

**Approach**: Separate numeric and non-numeric data

```python
# Numeric operations use JAX
df_numeric = df[['price', 'quantity', 'discount']]
result = jax.jit(lambda x: x * 2)(df_numeric)

# Mixed operations fall back to numpy/pandas-like behavior
df['category']  # Returns regular numpy array
df.groupby('category')['price'].mean()  # JAX on numeric agg
```

**Pros**:
- Best performance for numeric operations
- Full JIT/grad support on numeric columns
- Can still support object columns for practical use

**Cons**:
- API complexity (some operations work differently based on dtypes)
- Users must be aware of numeric vs object distinction

### Strategy 2: Encoding Non-Numeric Data

**Approach**: Encode strings/categories as integers, maintain lookup tables

```python
@dataclass
class DataFrame:
    _data: jnp.ndarray  # All data as numbers (encoded)
    _encoders: dict[str, Encoder]  # String -> int mappings
    _column_types: dict[str, ColumnType]  # Original types
```

**Example**:
```python
# Under the hood:
# ['cat', 'dog', 'cat'] -> [0, 1, 0]
# Store mapping: {'cat': 0, 'dog': 1}

df['animal'] = ['cat', 'dog', 'cat']  # Encoded as integers
jax.jit(compute)(df)  # Works! All data is numeric
df['animal'].values  # [0, 1, 0] (encoded)
df['animal'].decode()  # ['cat', 'dog', 'cat'] (original)
```

**Pros**:
- Everything can be JIT compiled
- Simpler internal representation
- Enables categorical operations in JAX

**Cons**:
- Extra encoding/decoding overhead
- Loses semantic meaning in JAX operations
- String operations become awkward

### Strategy 3: Hybrid Approach (Recommended for Production)

Combine both strategies:

```python
class DataFrame:
    def __init__(self, data):
        self._numeric_block = ...  # JAX array
        self._categorical_block = ...  # Encoded JAX array
        self._object_block = ...  # Non-JAX (strings, complex objects)

    @jax.jit
    def numeric_operation(self):
        # Only operates on numeric_block
        return self._numeric_block * 2

    def mixed_operation(self):
        # Can access all blocks, no JIT
        ...
```

**Benefits**:
- Numeric operations: Full JAX performance
- Categorical operations: JIT-able with encoding
- Complex objects: Supported but not JIT-able
- Clear performance model for users

## JIT Compilation Strategy

### Challenge: Dynamic Shapes

Pandas operations often change shapes:
```python
df[df['price'] > 100]  # Unknown number of rows at compile time
df.groupby('category')  # Unknown number of groups
```

### Solution: Static Shape Subset + Dynamic Fallback

```python
class DataFrame:
    def __jit_compatible__(self) -> bool:
        """Check if current operation can be JIT compiled"""
        return not self._has_pending_dynamic_ops

    def filter(self, mask):
        if jax.core.cur_sublevel().level > 0:  # Inside JIT
            # Return lazy object or error
            raise JitCompatibilityError(
                "Dynamic filtering not supported in JIT. "
                "Use .where() with fill values for fixed shapes"
            )
        return self._eager_filter(mask)

    def where(self, mask, fill_value):
        """JIT-compatible filtering (preserves shape)"""
        # This is JIT-safe: same shape in/out
        new_data = jnp.where(mask[:, None], self._numeric_data, fill_value)
        return DataFrame(new_data, ...)
```

### JIT-Compatible Operations

✅ **Always JIT-able**:
- Element-wise operations: `df * 2`, `df + other_df`
- Fixed-shape transformations: `df.where()`, `df.clip()`
- Aggregations with known output: `df.sum()`, `df.mean()`
- Window operations: `df.rolling(3).mean()` (with padding)

⚠️ **Conditionally JIT-able**:
- GroupBy: Only if group keys are static/encoded
- Joins: Only if using fixed-size merge with fill
- Sorting: Possible with JAX sort, but indices change

❌ **Not JIT-able**:
- Dynamic filtering: `df[df.col > threshold]`
- Pivot tables with unknown dimensions
- Explode operations
- String manipulations on object columns

### API Design for JIT

Provide explicit JIT-friendly alternatives:

```python
# Not JIT-able
df_filtered = df[df['price'] > 100]

# JIT-able alternative
df_masked = df.where(df['price'] > 100, fill_value=0.0)

# Decorator to auto-handle
@jaxframe.jit_compatible
def process_dataframe(df, threshold):
    return df.where(df['price'] > threshold, 0.0).sum()

# This works!
jitted_fn = jax.jit(process_dataframe)
result = jitted_fn(df, 100.0)
```

## Automatic Differentiation (grad) Strategy

### Challenge: What Should Be Differentiable?

```python
df = DataFrame({
    'price': [10.0, 20.0, 30.0],      # Differentiable
    'quantity': [1, 2, 3],             # Differentiable (integers)
    'name': ['a', 'b', 'c'],           # Not differentiable
})

# What does this mean?
grad_fn = jax.grad(lambda df: df['price'].sum())
```

### Solution: Pytree Registration

Register DataFrame as a JAX pytree, only including numeric data in the tree:

```python
def _dataframe_flatten(df):
    """Extract arrays for JAX transformations"""
    # Only numeric data participates in grad
    children = (df._numeric_data,)  # JAX arrays
    aux_data = {
        'numeric_cols': df._numeric_cols,
        'numeric_dtypes': df._numeric_dtypes,
        'object_data': df._object_data,  # Frozen/non-differentiable
        'index': df._index,
    }
    return children, aux_data

def _dataframe_unflatten(aux_data, children):
    """Reconstruct DataFrame from JAX transformation"""
    numeric_data, = children
    return DataFrame(
        _numeric_data=numeric_data,
        **aux_data
    )

jax.tree_util.register_pytree_node(
    DataFrame,
    _dataframe_flatten,
    _dataframe_unflatten
)
```

### Gradient Examples

```python
# Example 1: Simple gradient
def loss_fn(df):
    predictions = df['price'] * 2
    targets = df['target']
    return ((predictions - targets) ** 2).sum()

grad_fn = jax.grad(loss_fn)
gradients = grad_fn(df)  # Returns DataFrame with gradients

# Example 2: Per-column gradients
def multivariate_loss(df):
    return (df['x']**2 + df['y']**2).sum()

grad_df = jax.grad(multivariate_loss)(df)
# grad_df['x'] contains ∂L/∂x
# grad_df['y'] contains ∂L/∂y

# Example 3: GroupBy with grad
def grouped_loss(df):
    # Assuming categorical encoding
    groups = df._categorical_block[:, 0]  # 'category' column
    values = df._numeric_block[:, 0]  # 'value' column

    # Differentiable groupby mean
    group_means = jax_ops.segment_mean(values, groups)
    return (group_means ** 2).sum()

jax.grad(grouped_loss)(df)  # Works!
```

### Handling Non-Differentiable Columns

```python
class DataFrame:
    def _mark_differentiable(self, cols: list[str]):
        """Mark which columns should participate in grad"""
        self._differentiable_cols = set(cols)

    def requires_grad(self, cols: Union[str, list[str]]) -> 'DataFrame':
        """PyTorch-style API for marking differentiable columns"""
        cols = [cols] if isinstance(cols, str) else cols
        new_df = self.copy()
        new_df._differentiable_cols.update(cols)
        return new_df

# Usage
df = DataFrame({'price': [1, 2, 3], 'id': [10, 20, 30]})
df = df.requires_grad(['price'])  # Only 'price' differentiable

def loss(df):
    return df['price'].sum()  # OK

def bad_loss(df):
    return df['id'].sum()  # Could warn/error if id not marked
```

## Index Management

### Challenge: Pandas Has Rich Index Support

```python
df1.loc['2020-01-01']  # DatetimeIndex
df2.loc[['a', 'b']]    # String index
df3.iloc[0:10]         # Integer position
```

### Strategy: Separate Index Implementation

```python
class Index:
    """Base class for indices"""
    def __getitem__(self, key):
        raise NotImplementedError

class NumericIndex(Index):
    """JIT-compatible integer index"""
    _values: jnp.ndarray  # Part of pytree

    @jax.jit
    def get_loc(self, key):
        return jnp.searchsorted(self._values, key)

class ObjectIndex(Index):
    """Non-JIT string/datetime index"""
    _values: np.ndarray  # Not in pytree
    _hash_map: dict  # Fast lookup

    def get_loc(self, key):
        return self._hash_map[key]

class DataFrame:
    def loc(self, key):
        if isinstance(self._index, NumericIndex):
            # JIT-compatible path
            idx = self._index.get_loc(key)
            return self.iloc[idx]
        else:
            # Non-JIT path
            idx = self._index.get_loc(key)
            return self.iloc[idx]
```

## Missing Data Handling

### Strategy: Masked Arrays

```python
@dataclass
class DataFrame:
    _numeric_data: jnp.ndarray
    _numeric_mask: Optional[jnp.ndarray]  # True = valid, False = missing

    def fillna(self, value):
        """JIT-compatible fillna"""
        filled = jnp.where(self._numeric_mask, self._numeric_data, value)
        return DataFrame(filled, numeric_mask=jnp.ones_like(self._numeric_mask))

    def dropna(self):
        """Not JIT-compatible (changes shape)"""
        if in_jit_context():
            raise JitError("dropna changes shape, use fillna or where")
        valid_rows = self._numeric_mask.all(axis=1)
        return self.iloc[valid_rows]

    @jax.jit
    def mean(self, skipna=True):
        """Masked mean"""
        if skipna:
            return jnp.where(self._numeric_mask, self._numeric_data, 0).sum() / self._numeric_mask.sum()
        else:
            return self._numeric_data.mean()
```

## API Surface Design

### Principle: Pandas-Compatible, JAX-Aware

```python
# Looks like pandas
import jaxframe as jf

df = jf.DataFrame({
    'price': [10.0, 20.0, 30.0],
    'quantity': [1, 2, 3],
    'name': ['a', 'b', 'c']
})

# Pandas-like operations
result = df['price'] * df['quantity']
result = df.groupby('name')['price'].mean()
result = df[df['price'] > 15]

# JAX-aware operations
@jax.jit
def process(df):
    # Only use JIT-compatible operations
    return df.where(df['price'] > 15, 0.0).sum()

@jax.grad
def loss(df):
    return (df['price'] ** 2).sum()

# Explicit control
df_numeric = df.select_dtypes(include='number')  # JAX-only subset
fast_result = jax.jit(lambda x: x * 2)(df_numeric)
```

### Gradual Degradation

```python
class DataFrame:
    def __call_method__(self, name, *args, **kwargs):
        """Intercept method calls"""
        if in_jit_context() and not self._is_jit_compatible(name):
            raise JitCompatibilityError(
                f"{name} is not JIT-compatible. "
                f"Suggestions: {self._jit_alternatives(name)}"
            )
        return getattr(self, name)(*args, **kwargs)
```

## Implementation Phases

### Phase 1: Core Numeric Operations
- DataFrame with numeric-only columns
- Basic indexing (iloc, loc with integers)
- Element-wise operations
- Aggregations (sum, mean, std)
- JIT support for all operations
- Pytree registration for grad

### Phase 2: Mixed Types
- Add object column support
- Categorical encoding
- String index support
- Graceful degradation for JIT

### Phase 3: Advanced Operations
- GroupBy with static groups
- Joins (merge, concat)
- Window operations
- Time series support

### Phase 4: Polish
- Better error messages
- Performance optimization
- Comprehensive docs
- Pandas compatibility layer

## Example Use Cases

### 1. Differentiable Data Processing

```python
def neural_feature_engineering(df, weights):
    """Learnable feature transformation"""
    df = jf.DataFrame(df)
    features = df[['x1', 'x2', 'x3']]

    # Differentiable operations
    scaled = features * weights['scale']
    normalized = (scaled - scaled.mean()) / scaled.std()
    interactions = normalized['x1'] * normalized['x2']

    return jf.concat([normalized, interactions], axis=1)

# Optimize feature engineering
def loss(weights, df, targets):
    features = neural_feature_engineering(df, weights)
    predictions = features.sum(axis=1)
    return ((predictions - targets) ** 2).mean()

weights = {'scale': jnp.array([1.0, 1.0, 1.0])}
grad_fn = jax.grad(loss)
gradients = grad_fn(weights, train_df, targets)
```

### 2. JIT-Compiled Analytics

```python
@jax.jit
def compute_metrics(df):
    """Fast metric computation"""
    return {
        'revenue': (df['price'] * df['quantity']).sum(),
        'avg_price': df['price'].mean(),
        'total_items': df['quantity'].sum(),
    }

# 100x faster on repeated calls
for batch in data_stream:
    metrics = compute_metrics(batch)  # Reuses compiled code
```

### 3. Vectorized Processing

```python
# Process multiple dataframes in parallel
dfs = [load_data(f) for f in files]

@jax.vmap
def process_batch(df):
    return df['price'].mean()

# Parallel processing across all dataframes
means = process_batch(dfs)
```

## Open Questions

1. **Should we support 100% pandas API?**
   - Pro: Easy migration
   - Con: Some operations don't fit JAX model
   - Proposal: 80% compatible, clear docs on differences

2. **How to handle in-place operations?**
   - Pandas: `df['new_col'] = values` (in-place)
   - JAX: Immutable (should return new DataFrame)
   - Proposal: `df.assign(new_col=values)` recommended, `df['new_col'] = values` allowed but creates copy

3. **Performance vs Compatibility trade-off?**
   - Could support more pandas features with performance cost
   - Proposal: Performance first, document incompatibilities clearly

4. **Index alignment automatic or explicit?**
   - Pandas: Automatic alignment on operations
   - JAX: No concept of alignment
   - Proposal: Optional alignment, disabled in JIT context

## Conclusion

**Recommended Architecture**:
1. Hybrid storage: numeric JAX arrays + object columns
2. Pytree registration for grad support (numeric only)
3. Explicit JIT-compatible API subset
4. Gradual degradation with helpful errors
5. Pandas-like API where possible, JAX-aware where necessary

This design enables:
- ✅ Full JIT compilation on numeric operations
- ✅ Automatic differentiation through DataFrame operations
- ✅ Support for non-numeric columns (with limitations)
- ✅ Familiar pandas-like API
- ✅ Clear performance model for users
