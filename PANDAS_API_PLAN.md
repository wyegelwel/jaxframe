# Pandas API Coverage Plan for JAXFrame

Comprehensive plan to mirror the pandas API with JAX support indicators.

## Legend

**Support Status:**
- ✅ **Implemented**: Fully working
- 🚧 **In Progress**: Partially implemented
- 📋 **Planned**: Designed, not yet implemented
- ⚠️ **Limited**: Supported with limitations
- ❌ **Not Feasible**: Incompatible with JAX design

**JAX Compatibility:**
- 🔥 **JIT**: Supports `jax.jit` compilation
- 🎯 **grad**: Supports `jax.grad` differentiation
- 📊 **vmap**: Supports `jax.vmap` vectorization
- 🚫 **None**: Not compatible with JAX transformations

---

## 1. DataFrame Construction

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `DataFrame(dict)` | ✅ | 🔥 | 🎯 | 📊 | Hybrid storage for mixed types |
| `DataFrame(array)` | ✅ | 🔥 | 🎯 | 📊 | Converts to JAX array |
| `DataFrame(Series)` | 📋 | 🔥 | 🎯 | 📊 | |
| `DataFrame.from_dict()` | 📋 | 🔥 | 🎯 | 📊 | |
| `DataFrame.from_records()` | 📋 | 🔥 | 🎯 | 📊 | |
| `DataFrame.from_csv()` | 📋 | 🚫 | 🚫 | 🚫 | I/O not JIT-able |
| `DataFrame.copy()` | 📋 | 🔥 | 🎯 | 📊 | JAX arrays immutable anyway |

---

## 2. Attributes and Properties

| Attribute | Status | JIT | grad | vmap | Notes |
|-----------|--------|-----|------|------|-------|
| `df.shape` | ✅ | 🔥 | 🚫 | 📊 | Returns tuple |
| `df.size` | 📋 | 🔥 | 🚫 | 📊 | Total elements |
| `df.ndim` | 📋 | 🔥 | 🚫 | 📊 | Always 2 |
| `df.columns` | ✅ | 🚫 | 🚫 | 🚫 | Returns Python list |
| `df.index` | ✅ | 🚫 | 🚫 | 🚫 | Metadata |
| `df.dtypes` | 📋 | 🚫 | 🚫 | 🚫 | Metadata |
| `df.values` | 📋 | 🔥 | 🎯 | 📊 | Returns numeric JAX array |
| `df.empty` | 📋 | 🔥 | 🚫 | 📊 | Boolean check |

---

## 3. Indexing and Selection

### 3.1 Column Selection

| Operation | Status | JIT | grad | vmap | Notes |
|-----------|--------|-----|------|------|-------|
| `df['col']` | ✅ | 🔥 | 🎯 | 📊 | Returns Series |
| `df[['col1', 'col2']]` | ✅ | 🔥 | 🎯 | 📊 | Returns DataFrame |
| `df.col` | 📋 | 🔥 | 🎯 | 📊 | Attribute access |

### 3.2 Row Selection

| Operation | Status | JIT | grad | vmap | Notes |
|-----------|--------|-----|------|------|-------|
| `df.loc[label]` | 📋 | ⚠️ | ⚠️ | ⚠️ | Static labels only in JIT |
| `df.iloc[int]` | 📋 | 🔥 | 🎯 | 📊 | Integer indexing |
| `df.iloc[slice]` | 📋 | 🔥 | 🎯 | 📊 | Slicing |
| `df.iloc[array]` | 📋 | 🔥 | 🎯 | 📊 | Advanced indexing |
| `df[mask]` | ❌ | 🚫 | 🚫 | 🚫 | Changes shape - use `where()` |
| `df.head(n)` | 📋 | 🔥 | 🎯 | 📊 | Fixed n |
| `df.tail(n)` | 📋 | 🔥 | 🎯 | 📊 | Fixed n |
| `df.sample(n)` | ⚠️ | ⚠️ | 🚫 | 📊 | Random not differentiable |

### 3.3 Boolean Indexing Alternatives

| Operation | Status | JIT | grad | vmap | Notes |
|-----------|--------|-----|------|------|-------|
| `df.where(cond, fill)` | ✅ | 🔥 | 🎯 | 📊 | **JIT-friendly filtering** |
| `df.mask(cond, fill)` | 📋 | 🔥 | 🎯 | 📊 | Inverse of where |
| `df.clip(lower, upper)` | 📋 | 🔥 | 🎯 | 📊 | Element-wise clipping |

---

## 4. Binary Operations

### 4.1 Arithmetic Operations

| Operation | Status | JIT | grad | vmap | Notes |
|-----------|--------|-----|------|------|-------|
| `df + other` | ✅ | 🔥 | 🎯 | 📊 | Element-wise |
| `df - other` | ✅ | 🔥 | 🎯 | 📊 | Element-wise |
| `df * other` | ✅ | 🔥 | 🎯 | 📊 | Element-wise |
| `df / other` | 📋 | 🔥 | 🎯 | 📊 | Element-wise |
| `df // other` | 📋 | 🔥 | 🎯 | 📊 | Floor division |
| `df % other` | 📋 | 🔥 | 🎯 | 📊 | Modulo |
| `df ** other` | 📋 | 🔥 | 🎯 | 📊 | Power |
| `df @ other` | 📋 | 🔥 | 🎯 | 📊 | Matrix multiplication |

### 4.2 Comparison Operations

| Operation | Status | JIT | grad | vmap | Notes |
|-----------|--------|-----|------|------|-------|
| `df > other` | ✅ | 🔥 | 🚫 | 📊 | Returns bool DataFrame |
| `df >= other` | 📋 | 🔥 | 🚫 | 📊 | |
| `df < other` | 📋 | 🔥 | 🚫 | 📊 | |
| `df <= other` | 📋 | 🔥 | 🚫 | 📊 | |
| `df == other` | 📋 | 🔥 | 🚫 | 📊 | |
| `df != other` | 📋 | 🔥 | 🚫 | 📊 | |

### 4.3 Logical Operations

| Operation | Status | JIT | grad | vmap | Notes |
|-----------|--------|-----|------|------|-------|
| `df & other` | 📋 | 🔥 | 🚫 | 📊 | Logical AND |
| `df \| other` | 📋 | 🔥 | 🚫 | 📊 | Logical OR |
| `~df` | 📋 | 🔥 | 🚫 | 📊 | Logical NOT |

---

## 5. Function Application

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `df.apply(func)` | 📋 | ⚠️ | ⚠️ | 📊 | Only if func is JIT-able |
| `df.applymap(func)` | 📋 | ⚠️ | ⚠️ | 📊 | Element-wise |
| `df.pipe(func)` | 📋 | ⚠️ | ⚠️ | 📊 | Chaining |
| `df.transform(func)` | 📋 | ⚠️ | ⚠️ | 📊 | |

---

## 6. Computations / Descriptive Stats

### 6.1 Reductions

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `df.sum(axis)` | ✅ | 🔥 | 🎯 | 📊 | Fully differentiable |
| `df.mean(axis)` | ✅ | 🔥 | 🎯 | 📊 | Fully differentiable |
| `df.std(axis)` | 📋 | 🔥 | 🎯 | 📊 | Fully differentiable |
| `df.var(axis)` | 📋 | 🔥 | 🎯 | 📊 | Fully differentiable |
| `df.min(axis)` | 📋 | 🔥 | ⚠️ | 📊 | Non-smooth gradient |
| `df.max(axis)` | 📋 | 🔥 | ⚠️ | 📊 | Non-smooth gradient |
| `df.median(axis)` | 📋 | 🔥 | 🚫 | 📊 | Not differentiable |
| `df.prod(axis)` | 📋 | 🔥 | 🎯 | 📊 | Product |
| `df.count(axis)` | 📋 | 🔥 | 🚫 | 📊 | Count non-NaN |
| `df.abs()` | 📋 | 🔥 | ⚠️ | 📊 | Non-smooth at 0 |
| `df.cumsum(axis)` | 📋 | 🔥 | 🎯 | 📊 | Cumulative sum |
| `df.cumprod(axis)` | 📋 | 🔥 | 🎯 | 📊 | Cumulative product |

### 6.2 Statistical Functions

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `df.describe()` | 📋 | 🚫 | 🚫 | 🚫 | Summary stats (I/O) |
| `df.corr()` | 📋 | 🔥 | 🎯 | 📊 | Correlation matrix |
| `df.cov()` | 📋 | 🔥 | 🎯 | 📊 | Covariance matrix |
| `df.quantile(q)` | 📋 | 🔥 | 🚫 | 📊 | Not differentiable |
| `df.rank()` | 📋 | 🔥 | 🚫 | 📊 | Not differentiable |

---

## 7. Reshaping / Sorting / Transposing

### 7.1 Shape Manipulation

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `df.T` / `df.transpose()` | 📋 | 🔥 | 🎯 | 📊 | Transpose |
| `df.stack()` | 📋 | 🔥 | 🎯 | 📊 | Pivot to long format |
| `df.unstack()` | 📋 | 🔥 | 🎯 | 📊 | Pivot to wide format |
| `df.pivot()` | ⚠️ | 🚫 | 🚫 | 🚫 | Dynamic shape |
| `df.melt()` | 📋 | 🔥 | 🎯 | 📊 | Unpivot |
| `df.explode()` | ❌ | 🚫 | 🚫 | 🚫 | Dynamic shape |

### 7.2 Sorting

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `df.sort_values(by)` | 📋 | 🔥 | 🚫 | 📊 | Uses jax.sort |
| `df.sort_index()` | 📋 | 🔥 | 🚫 | 📊 | |
| `df.argsort()` | 📋 | 🔥 | 🚫 | 📊 | Returns indices |

---

## 8. Combining / Joining / Merging

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `pd.concat([df1, df2])` | 📋 | 🔥 | 🎯 | 📊 | Fixed shapes |
| `df.append(other)` | 📋 | 🔥 | 🎯 | 📊 | Deprecated in pandas 2.0 |
| `df.join(other)` | ⚠️ | ⚠️ | ⚠️ | 📊 | Index-based, static shapes |
| `pd.merge(df1, df2)` | ⚠️ | 🚫 | 🚫 | 🚫 | Dynamic result size |

---

## 9. Time Series

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `df.shift(n)` | 📋 | 🔥 | 🎯 | 📊 | Shift with padding |
| `df.diff(n)` | 📋 | 🔥 | 🎯 | 📊 | Difference |
| `df.pct_change()` | 📋 | 🔥 | 🎯 | 📊 | Percent change |
| `df.rolling(window)` | 📋 | 🔥 | 🎯 | 📊 | Rolling window ops |
| `df.expanding()` | 📋 | 🔥 | 🎯 | 📊 | Expanding window |
| `df.ewm()` | 📋 | 🔥 | 🎯 | 📊 | Exponential weighted |
| `df.resample()` | ⚠️ | 🚫 | 🚫 | 🚫 | Dynamic groups |

---

## 10. GroupBy Operations

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `df.groupby(static_col)` | 📋 | 🔥 | 🎯 | 📊 | **Requires encoded categories** |
| `groupby().sum()` | 📋 | 🔥 | 🎯 | 📊 | Use jax segment_sum |
| `groupby().mean()` | 📋 | 🔥 | 🎯 | 📊 | Use jax segment ops |
| `groupby().count()` | 📋 | 🔥 | 🚫 | 📊 | |
| `groupby().apply(func)` | ⚠️ | ⚠️ | ⚠️ | 📊 | If func is JIT-able |
| `groupby(dynamic_col)` | ❌ | 🚫 | 🚫 | 🚫 | Unknown groups at compile time |

**Note:** GroupBy requires categorical encoding for JIT. See categorical support section.

---

## 11. Missing Data Handling

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `df.isna()` / `df.isnull()` | 📋 | 🔥 | 🚫 | 📊 | Boolean mask |
| `df.notna()` | 📋 | 🔥 | 🚫 | 📊 | Boolean mask |
| `df.fillna(value)` | 📋 | 🔥 | 🎯 | 📊 | Replace NaN |
| `df.dropna()` | ❌ | 🚫 | 🚫 | 🚫 | Changes shape |
| `df.interpolate()` | 📋 | 🔥 | 🎯 | 📊 | Linear interpolation |

**Implementation:** Use masked arrays (separate mask array) for NaN tracking.

---

## 12. String Methods (Object Columns)

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `df['col'].str.upper()` | 📋 | 🚫 | 🚫 | 🚫 | Object columns not JIT-able |
| `df['col'].str.lower()` | 📋 | 🚫 | 🚫 | 🚫 | |
| `df['col'].str.contains()` | 📋 | 🚫 | 🚫 | 🚫 | |
| `df['col'].str.replace()` | 📋 | 🚫 | 🚫 | 🚫 | |
| `df['col'].str.split()` | 📋 | 🚫 | 🚫 | 🚫 | |

**Note:** String operations work but are not JIT-compatible. Use categorical encoding for JIT.

---

## 13. Categorical Data

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `df['col'].astype('category')` | 📋 | 🔥 | 🚫 | 📊 | Encode to integers |
| `df['col'].cat.codes` | 📋 | 🔥 | 🚫 | 📊 | Integer codes |
| `df['col'].cat.categories` | 📋 | 🚫 | 🚫 | 🚫 | Category mapping |
| `pd.get_dummies(df)` | 📋 | 🔥 | 🎯 | 📊 | One-hot encoding |

**Implementation Strategy:**
```python
class CategoricalColumn:
    _codes: jnp.ndarray  # Integer codes (JIT-able)
    _categories: np.ndarray  # String mapping (metadata)

    def encode(self, strings):
        # strings -> codes

    def decode(self, codes):
        # codes -> strings
```

---

## 14. Advanced Features

### 14.1 Multi-Index

| Feature | Status | JIT | grad | vmap | Notes |
|---------|--------|-----|------|------|-------|
| MultiIndex creation | 📋 | 🚫 | 🚫 | 🚫 | Complex metadata |
| MultiIndex selection | 📋 | ⚠️ | ⚠️ | ⚠️ | Static levels only |

### 14.2 Sparse Data

| Feature | Status | JIT | grad | vmap | Notes |
|---------|--------|-----|------|------|-------|
| Sparse arrays | 📋 | 🔥 | 🎯 | 📊 | Use JAX BCOO format |

### 14.3 Extension Types

| Feature | Status | JIT | grad | vmap | Notes |
|---------|--------|-----|------|------|-------|
| Custom dtypes | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Limited support |

---

## 15. I/O Operations

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `pd.read_csv()` | 📋 | 🚫 | 🚫 | 🚫 | I/O not JIT-able |
| `df.to_csv()` | 📋 | 🚫 | 🚫 | 🚫 | |
| `pd.read_parquet()` | 📋 | 🚫 | 🚫 | 🚫 | |
| `df.to_parquet()` | 📋 | 🚫 | 🚫 | 🚫 | |
| `pd.read_json()` | 📋 | 🚫 | 🚫 | 🚫 | |
| `df.to_json()` | 📋 | 🚫 | 🚫 | 🚫 | |
| `df.to_numpy()` | 📋 | 🚫 | 🚫 | 🚫 | Convert to numpy |
| `df.to_pandas()` | 📋 | 🚫 | 🚫 | 🚫 | Convert to pandas |
| `DataFrame.from_pandas()` | 📋 | 🚫 | 🚫 | 🚫 | Create from pandas |

**Note:** I/O operations happen outside JIT context. Use for data loading/saving only.

---

## 16. Plotting

| Method | Status | JIT | grad | vmap | Notes |
|--------|--------|-----|------|------|-------|
| `df.plot()` | 📋 | 🚫 | 🚫 | 🚫 | Uses matplotlib |
| `df.plot.line()` | 📋 | 🚫 | 🚫 | 🚫 | |
| `df.plot.scatter()` | 📋 | 🚫 | 🚫 | 🚫 | |
| `df.plot.bar()` | 📋 | 🚫 | 🚫 | 🚫 | |

---

# Testing Strategy

## Test Suite Structure

```
tests/
├── test_equivalence.py       # Pandas vs JAXFrame equivalence
├── test_jax_transforms.py    # JIT/grad/vmap compatibility
├── test_dataframe.py          # Core functionality
├── test_series.py             # Series operations
├── test_categorical.py        # Categorical encoding
├── test_groupby.py            # GroupBy operations
├── test_io.py                 # I/O operations
└── test_performance.py        # Benchmarks
```

## Equivalence Testing Framework

```python
import pandas as pd
import jaxframe as jf
import numpy as np
import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal

class EquivalenceTest:
    """Test that JAXFrame matches pandas behavior."""

    def test_operation(self, pandas_op, jaxframe_op):
        """
        Compare pandas and jaxframe operations.

        Args:
            pandas_op: Function taking pandas DataFrame
            jaxframe_op: Function taking jaxframe DataFrame
        """
        # Generate test data
        data = {
            'a': [1.0, 2.0, 3.0, 4.0],
            'b': [5.0, 6.0, 7.0, 8.0],
        }

        # Create both DataFrames
        pdf = pd.DataFrame(data)
        jdf = jf.DataFrame(data)

        # Run operations
        pandas_result = pandas_op(pdf)
        jaxframe_result = jaxframe_op(jdf)

        # Compare results
        self.assert_equivalent(pandas_result, jaxframe_result)

    def assert_equivalent(self, pandas_result, jaxframe_result):
        """Assert pandas and jaxframe results are equivalent."""
        if isinstance(pandas_result, pd.DataFrame):
            # Compare DataFrames
            assert_array_almost_equal(
                pandas_result.values,
                np.array(jaxframe_result._numeric_data)
            )
        elif isinstance(pandas_result, pd.Series):
            # Compare Series
            assert_array_almost_equal(
                pandas_result.values,
                np.array(jaxframe_result.values)
            )
        else:
            # Compare scalars
            assert_array_almost_equal(pandas_result, jaxframe_result)


# Example tests
def test_sum_equivalence():
    """Test that sum matches pandas."""
    test = EquivalenceTest()
    test.test_operation(
        pandas_op=lambda df: df.sum(),
        jaxframe_op=lambda df: df.sum()
    )

def test_mean_equivalence():
    """Test that mean matches pandas."""
    test = EquivalenceTest()
    test.test_operation(
        pandas_op=lambda df: df.mean(),
        jaxframe_op=lambda df: df.mean()
    )

def test_multiplication_equivalence():
    """Test that multiplication matches pandas."""
    test = EquivalenceTest()
    test.test_operation(
        pandas_op=lambda df: df * 2,
        jaxframe_op=lambda df: df * 2
    )
```

## JAX Transform Testing

```python
import jax

class JAXTransformTest:
    """Test JAX transformations work correctly."""

    def test_jit(self, operation, data):
        """Test that operation can be JIT compiled."""
        df = jf.DataFrame(data)

        # Should not raise
        jitted_op = jax.jit(operation)
        result = jitted_op(df)

        # Should match non-JIT result
        expected = operation(df)
        assert_equivalent(result, expected)

    def test_grad(self, loss_fn, data):
        """Test that operation is differentiable."""
        df = jf.DataFrame(data)

        # Should not raise
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(df)

        # Gradients should be finite
        assert jnp.all(jnp.isfinite(grads._numeric_data))

    def test_vmap(self, operation, batched_data):
        """Test that operation can be vectorized."""
        # Should not raise
        vmapped_op = jax.vmap(operation)
        results = vmapped_op(batched_data)

        # Should have batch dimension
        assert results.shape[0] == batched_data.shape[0]


# Example tests
def test_sum_jit():
    """Test that sum is JIT-able."""
    test = JAXTransformTest()
    test.test_jit(
        operation=lambda df: df.sum(),
        data={'a': [1.0, 2.0, 3.0]}
    )

def test_sum_grad():
    """Test that sum is differentiable."""
    test = JAXTransformTest()
    test.test_grad(
        loss_fn=lambda df: df.sum().sum(),
        data={'a': [1.0, 2.0, 3.0]}
    )
```

## Compatibility Matrix Tests

```python
import pytest

# Define test matrix
OPERATIONS = [
    ('sum', lambda df: df.sum(), True, True, True),
    ('mean', lambda df: df.mean(), True, True, True),
    ('std', lambda df: df.std(), True, True, True),
    ('min', lambda df: df.min(), True, False, True),  # Non-smooth grad
    ('max', lambda df: df.max(), True, False, True),  # Non-smooth grad
    ('where', lambda df: df.where(df > 0, 0), True, True, True),
]

@pytest.mark.parametrize("name,op,jit,grad,vmap", OPERATIONS)
def test_operation_compatibility(name, op, jit, grad, vmap):
    """Test operation compatibility with JAX transforms."""
    df = jf.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})

    if jit:
        # Should be JIT-able
        jax.jit(op)(df)

    if grad:
        # Should be differentiable
        loss = lambda df: op(df)._numeric_data.sum()
        jax.grad(loss)(df)

    if vmap:
        # Should be vectorizable
        # (would need batch dimension setup)
        pass
```

---

# Implementation Roadmap

## Phase 1: Core Operations (Weeks 1-4)

**Goal:** Cover 80% of common use cases

### Week 1: Binary Operations
- [ ] All arithmetic operators (`+`, `-`, `*`, `/`, `//`, `%`, `**`)
- [ ] All comparison operators (`>`, `>=`, `<`, `<=`, `==`, `!=`)
- [ ] Logical operators (`&`, `|`, `~`)
- [ ] Equivalence tests vs pandas
- [ ] JIT/grad compatibility tests

### Week 2: Reductions & Aggregations
- [ ] All reduction operations (`sum`, `mean`, `std`, `var`, `min`, `max`)
- [ ] Cumulative operations (`cumsum`, `cumprod`)
- [ ] Statistical functions (`corr`, `cov`)
- [ ] Equivalence tests
- [ ] Gradient tests

### Week 3: Indexing & Selection
- [ ] `iloc` with integers, slices, arrays
- [ ] `loc` with static labels
- [ ] `head()`, `tail()`
- [ ] Better `where()` and `mask()`
- [ ] Equivalence tests

### Week 4: Shape Manipulation
- [ ] `transpose()`
- [ ] `stack()`, `unstack()`
- [ ] `melt()`
- [ ] Basic `concat()`
- [ ] Tests

## Phase 2: Advanced Operations (Weeks 5-8)

### Week 5: Categorical Support
- [ ] Categorical column type
- [ ] Automatic encoding/decoding
- [ ] `astype('category')`
- [ ] `get_dummies()` one-hot encoding
- [ ] Tests

### Week 6: GroupBy
- [ ] Basic groupby with encoded categories
- [ ] Segment operations using `jax.ops.segment_sum`
- [ ] `groupby().sum()`, `.mean()`, `.count()`
- [ ] Tests

### Week 7: Time Series
- [ ] `shift()`, `diff()`, `pct_change()`
- [ ] `rolling()` window operations
- [ ] `expanding()` operations
- [ ] `ewm()` exponential weighted
- [ ] Tests

### Week 8: Missing Data
- [ ] Masked array implementation
- [ ] `isna()`, `fillna()`
- [ ] `interpolate()`
- [ ] Tests

## Phase 3: I/O & Interoperability (Weeks 9-10)

### Week 9: I/O Operations
- [ ] `read_csv()`, `to_csv()`
- [ ] `read_parquet()`, `to_parquet()`
- [ ] Tests

### Week 10: Pandas Interop
- [ ] `to_pandas()` conversion
- [ ] `from_pandas()` conversion
- [ ] `to_numpy()` conversion
- [ ] Benchmark suite comparing pandas
- [ ] Tests

## Phase 4: Polish & Optimization (Weeks 11-12)

### Week 11: Performance
- [ ] Profile and optimize hot paths
- [ ] Better pytree implementation
- [ ] Memory efficiency improvements
- [ ] Benchmark suite

### Week 12: Documentation
- [ ] Complete API reference
- [ ] Migration guide from pandas
- [ ] Performance guide
- [ ] Best practices

---

# Success Metrics

## Coverage Metrics
- **API Coverage**: 80%+ of common pandas operations
- **Test Coverage**: 90%+ code coverage
- **Equivalence**: 95%+ operations match pandas exactly

## Performance Metrics
- **JIT Speedup**: 10x+ on repeated operations
- **Memory**: Comparable to pandas for numeric data
- **Compilation**: <1s for typical operations

## Compatibility Metrics
- **JIT**: 70%+ operations JIT-compatible
- **grad**: 60%+ operations differentiable
- **vmap**: 70%+ operations vectorizable

---

# Documentation Requirements

## User-Facing Docs

### 1. Migration Guide
```markdown
# Migrating from Pandas to JAXFrame

## Supported Operations (Direct Replacement)
- df.sum() ✅
- df.mean() ✅
- df * 2 ✅

## Operations Requiring Changes
- df[df > 0] ❌ → df.where(df > 0, fill_value=0) ✅
- df.groupby('dynamic') ❌ → df.groupby(encoded_col) ✅

## Unsupported Operations
- df.pivot_table() ❌ (dynamic shape)
```

### 2. Performance Guide
```markdown
# Performance Best Practices

## JIT Compilation
✅ Do: Use JIT for repeated operations
❌ Don't: JIT one-off operations (overhead)

## Gradient Computation
✅ Do: Use smooth operations (sum, mean)
⚠️ Caution: Non-smooth ops (min, max) have subgradients
```

### 3. API Reference
- Auto-generated from docstrings
- Include JAX compatibility indicators
- Show pandas equivalents

---

# Open Questions

1. **Index Alignment**: Should we support automatic index alignment like pandas?
   - Pro: Familiar pandas behavior
   - Con: Not JIT-compatible (requires dynamic lookups)
   - **Proposal**: Optional, disabled in JIT context

2. **Copy vs View**: JAX arrays are immutable, pandas has views
   - **Proposal**: Always copy (JAX way), document difference

3. **NaN Representation**: How to handle NaN in JIT?
   - **Option 1**: Masked arrays (separate mask)
   - **Option 2**: Special value (like NaN) but non-differentiable
   - **Proposal**: Masked arrays for flexibility

4. **String Operations**: Support in JIT context?
   - **Proposal**: Not supported in JIT, encourage categorical encoding

5. **Performance Guarantees**: What can we promise?
   - **Proposal**: Document "JIT gives 10x+ speedup" but don't guarantee exact numbers

---

# Summary

This plan provides:
1. ✅ **Complete API inventory** with support status
2. ✅ **JAX compatibility matrix** (JIT/grad/vmap for each operation)
3. ✅ **Comprehensive test strategy** (equivalence + transforms)
4. ✅ **12-week implementation roadmap**
5. ✅ **Clear success metrics**
6. ✅ **Documentation requirements**

**Next Steps:**
1. Review and approve this plan
2. Set up test infrastructure (equivalence framework)
3. Begin Phase 1 implementation
4. Iterate based on user feedback
