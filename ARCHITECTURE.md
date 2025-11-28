# JAXFrame Architecture

## Directory Structure

```
jaxframe/
├── jaxframe/              # Core library code
│   ├── __init__.py        # Public API exports
│   └── dataframe.py       # DataFrame and Series implementation
├── examples/              # Usage examples
│   ├── jit_example.py     # JIT compilation examples
│   ├── grad_example.py    # Automatic differentiation examples
│   └── mixed_columns_example.py  # Mixed type columns
├── tests/                 # Test suite
│   └── test_dataframe.py  # Core DataFrame tests
├── DESIGN.md              # Design document (comprehensive)
├── README.md              # User-facing documentation
├── pyproject.toml         # Project configuration
└── requirements.txt       # Dependencies
```

## Core Components

### 1. DataFrame Class (`jaxframe/dataframe.py`)

The main data structure with hybrid storage:

```python
@dataclass
class DataFrame:
    _numeric_data: Optional[jnp.ndarray]    # JAX array for numeric columns
    _numeric_cols: Tuple[str, ...]          # Column names
    _numeric_dtypes: Tuple[Any, ...]        # Original dtypes
    _object_data: Dict[str, np.ndarray]     # Non-JAX data
    _index: np.ndarray                       # Row index
    _column_order: Tuple[str, ...]          # Preserve column order
```

**Key Features**:
- Numeric data stored in single JAX array (efficient)
- Object data stored separately (supports strings, etc.)
- Pytree registration enables JAX transformations
- Pandas-like API where possible

### 2. Series Class

Simplified 1D version of DataFrame:

```python
@dataclass
class Series:
    _data: Union[jnp.ndarray, np.ndarray]
    _index: np.ndarray
    _name: Optional[str]
```

### 3. Pytree Registration

Enables JAX transformations (jit, grad, vmap):

```python
def _dataframe_flatten(df):
    """Only numeric data participates in JAX operations."""
    children = (df._numeric_data,)
    aux_data = {...}  # Metadata
    return children, aux_data
```

## Data Flow

### Creating a DataFrame

```
User Input (dict) → _init_from_dict()
    ↓
Separate numeric/object columns
    ↓
Numeric → jnp.ndarray (single array)
Object  → dict of np.ndarray
    ↓
DataFrame instance
```

### JIT Compilation

```
Python function with DataFrame ops
    ↓
@jax.jit decorator
    ↓
JAX traces through pytree
    ↓
Only _numeric_data is traced
    ↓
Compiled XLA code
    ↓
Fast execution on CPU/GPU/TPU
```

### Automatic Differentiation

```
Loss function(DataFrame) → scalar
    ↓
jax.grad() wraps function
    ↓
Forward pass: compute loss
Backward pass: compute gradients
    ↓
Returns DataFrame with gradients
```

## Design Decisions

### Why Hybrid Storage?

**Alternatives considered**:

1. **All JAX arrays** (encode strings as integers)
   - ❌ Loses semantic meaning
   - ❌ Overhead for encoding/decoding
   - ✅ Everything is JIT-able

2. **All NumPy arrays** (no JAX)
   - ❌ No JIT compilation
   - ❌ No automatic differentiation
   - ✅ Full pandas compatibility

3. **Hybrid approach** (chosen)
   - ✅ Performance where it matters (numeric ops)
   - ✅ Flexibility for non-numeric data
   - ✅ Clear mental model
   - ⚠️ API complexity (some ops work differently)

### Why Pytree Registration?

JAX transformations work on pytrees (tree-like structures):

- **Children**: Arrays that participate in JAX ops (differentiable)
- **Aux data**: Static metadata (passes through unchanged)

This allows:
```python
# This just works!
grad_fn = jax.grad(lambda df: df['price'].sum())
grads = grad_fn(dataframe)
```

### Why Not 100% Pandas Compatible?

Some pandas operations are fundamentally incompatible with JAX:

| Operation | Pandas | JAXFrame | Reason |
|-----------|--------|----------|--------|
| `df[df > 10]` | ✅ | ❌ | Changes shape (not JIT-able) |
| `df.where(df > 10, 0)` | ✅ | ✅ | Fixed shape (JIT-able) |
| `df.groupby('dynamic')` | ✅ | ⚠️ | Need static groups |
| `df['col'] = value` | ✅ | ✅ | Creates copy (JAX immutable) |

## Performance Characteristics

### When JAXFrame is Faster

- ✅ Repeated numeric operations (JIT compilation)
- ✅ Batch processing (vmap)
- ✅ GPU/TPU computation
- ✅ Automatic differentiation use cases

### When Pandas is Faster

- ❌ Single-shot operations (compilation overhead)
- ❌ Heavy string manipulation
- ❌ Dynamic schema changes
- ❌ Complex joins/groupbys

## Extension Points

### Future Enhancements

1. **Categorical Support**
   - Automatic encoding/decoding
   - Segment operations for groupby
   - Preserve semantics

2. **Index Types**
   - DatetimeIndex (for time series)
   - MultiIndex (hierarchical indexing)
   - Optimized integer index

3. **Advanced Operations**
   - Window functions (rolling, expanding)
   - Resample/pivot (with static shapes)
   - Merge/join (JIT-compatible versions)

4. **Interoperability**
   - `to_pandas()` / `from_pandas()`
   - `to_numpy()` / `from_numpy()`
   - Arrow/Parquet support

## Testing Strategy

### Test Categories

1. **Unit tests** (`tests/test_dataframe.py`)
   - Core functionality
   - Edge cases
   - Error handling

2. **Integration tests**
   - JAX transformations
   - End-to-end workflows

3. **Performance tests**
   - Benchmark against pandas
   - Verify JIT speedup

4. **Examples as tests**
   - Ensure examples run
   - Catch regressions

## Contributing

See inline documentation in:
- `DESIGN.md` - Comprehensive design rationale
- `dataframe.py` - Implementation details
- Examples - Usage patterns

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX Pytrees](https://jax.readthedocs.io/en/latest/pytrees.html)
- [Pandas API](https://pandas.pydata.org/docs/)
