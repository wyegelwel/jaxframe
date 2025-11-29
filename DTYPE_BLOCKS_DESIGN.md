# Dtype Blocks Architecture Design

## Overview

Refactor DataFrame to use **one JAX array per unique dtype** instead of converting all numeric data to float64.

## Current Architecture (Before)

```python
@dataclass
class DataFrame:
    _numeric_data: Optional[jnp.ndarray]      # Single float64 array for ALL numeric columns
    _numeric_cols: Tuple[str, ...]             # Column names
    _numeric_dtypes: Tuple[Any, ...]           # Original dtypes (stored but not used)
    _object_data: Dict[str, np.ndarray]        # Non-numeric columns
    _index: np.ndarray
    _column_order: Tuple[str, ...]
```

**Problem:** All numeric data converted to float64 (see line 206)
- Wastes memory (int8 → float64 = 8x increase)
- Loses integer precision for large ints
- Slower operations on larger dtypes

## New Architecture (After)

```python
@dataclass
class DataFrame:
    _dtype_blocks: Dict[np.dtype, jnp.ndarray]  # One array per dtype
    _column_to_block: Dict[str, Tuple[np.dtype, int]]  # col → (dtype, index_in_block)
    _object_data: Dict[str, np.ndarray]         # Non-numeric columns (unchanged)
    _index: np.ndarray                           # Row index (unchanged)
    _column_order: Tuple[str, ...]               # Column order (unchanged)
```

### Supported JAX Dtypes

Group by category for easier management:

**Integers (signed):**
- `int8`, `int16`, `int32`, `int64`

**Integers (unsigned):**
- `uint8`, `uint16`, `uint32`, `uint64`

**Floats:**
- `float16`, `float32`, `float64`

**Boolean:**
- `bool` (or `bool_`)

**Complex:**
- `complex64`, `complex128`

**Objects:**
- Already handled separately in `_object_data` (strings, etc.)

### Example Internal State

```python
df = DataFrame({
    'a': np.array([1, 2, 3], dtype=np.int32),
    'b': np.array([4.0, 5.0, 6.0], dtype=np.float32),
    'c': np.array([7.0, 8.0, 9.0], dtype=np.float32),  # Same dtype as 'b'
    'd': np.array([True, False, True], dtype=np.bool_),
    'e': ['x', 'y', 'z']  # object
})

# Internal representation:
_dtype_blocks = {
    np.dtype('int32'): jnp.array([[1], [2], [3]]),              # 1 column
    np.dtype('float32'): jnp.array([[4.0, 7.0], [5.0, 8.0], [6.0, 9.0]]),  # 2 columns
    np.dtype('bool'): jnp.array([[True], [False], [True]]),     # 1 column
}

_column_to_block = {
    'a': (np.dtype('int32'), 0),    # int32 block, column 0
    'b': (np.dtype('float32'), 0),  # float32 block, column 0
    'c': (np.dtype('float32'), 1),  # float32 block, column 1
    'd': (np.dtype('bool'), 0),     # bool block, column 0
}

_object_data = {
    'e': np.array(['x', 'y', 'z'], dtype=object)
}

_column_order = ('a', 'b', 'c', 'd', 'e')
```

## Type Promotion Rules

When mixing dtypes in operations (e.g., int32 + float32), follow NumPy's type promotion:

```python
int8 + int8 → int8
int8 + int32 → int32
int32 + float32 → float32
float32 + float64 → float64
int32 + bool → int32
float32 + bool → float32
```

Use `jnp.result_type(dtype1, dtype2)` for automatic promotion.

## Key Changes by Method Category

### 1. Initialization (`__init__`, `_init_from_dict`)

**Before:**
- Convert all to float64
- Stack into single array

**After:**
- Group columns by dtype
- Create one block per unique dtype
- Build column → block mapping

### 2. Arithmetic Operations (`__add__`, `__mul__`, etc.)

**Before:**
- Simple operation on single array

**After:**
- For scalar: apply to each block independently
- For DataFrame: align dtypes, promote types, combine results
- Return new DataFrame with appropriate dtype blocks

### 3. Reductions (`sum`, `mean`, etc.)

**Before:**
- Single reduction on _numeric_data

**After:**
- Reduce each block independently
- Combine results preserving dtypes
- Return Series with appropriate dtypes

### 4. Indexing (`iloc`, slicing)

**Before:**
- Slice single array

**After:**
- Slice each relevant block
- Preserve dtype blocks in result
- Update column → block mapping

### 5. Concatenation (`concat`)

**Before:**
- Concatenate single numeric arrays

**After:**
- Group columns by dtype across all DataFrames
- Concatenate blocks of same dtype
- Rebuild column → block mapping

## Backward Compatibility

### `values` property behavior

**Current:** Returns single float64 array
**New:** What should it return?

**Option 1:** Convert all to float64 (backward compatible but defeats purpose)
**Option 2:** Return dict of blocks (breaking change)
**Option 3:** Return array with smallest common dtype that fits all blocks

**Decision:** Use Option 3 with type promotion
- If all int types → use largest int type
- If any float → use largest float type
- Matches pandas behavior

### Pytree Registration

Must update `tree_flatten` and `tree_unflatten` to handle dict of blocks:

```python
def tree_flatten(self):
    # Flatten all dtype blocks
    block_items = sorted(self._dtype_blocks.items())  # Sort for determinism
    arrays = [block for _, block in block_items]
    dtypes = [dtype for dtype, _ in block_items]

    aux_data = {
        'dtypes': dtypes,
        'column_to_block': self._column_to_block,
        'object_data': self._object_data,
        'index': self._index,
        'column_order': self._column_order,
    }
    return arrays, aux_data
```

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Update data structures
2. Update `__init__` and `_init_from_dict`
3. Update basic properties (shape, columns, dtypes)
4. Update `_from_parts` class method

### Phase 2: Operations
5. Update arithmetic operators with type promotion
6. Update comparison and logical operators
7. Update reduction methods
8. Update indexing methods

### Phase 3: Advanced Features
9. Update statistical methods
10. Update shape operations
11. Update time series methods
12. Update pytree registration

### Phase 4: Testing & Validation
13. Run existing tests (should mostly pass)
14. Add dtype preservation tests
15. Add type promotion tests
16. Performance benchmarks

## Expected Benefits

1. **Memory Savings:**
   - int8 columns: 8x smaller
   - int32 columns: 2x smaller
   - bool columns: 64x smaller

2. **Precision:**
   - No loss of integer precision
   - Exact integer arithmetic

3. **Performance:**
   - Smaller memory footprint → better cache utilization
   - Integer ops faster than float for comparisons
   - Better GPU memory usage

4. **Pandas Compatibility:**
   - Matches pandas dtype preservation behavior
   - More predictable type behavior

## Migration Notes

- Existing code should continue to work
- `values` property behavior changes slightly (returns promoted type, not always float64)
- JIT compilation artifacts may grow slightly (multiple arrays)
- Initial implementation may be slower due to complexity, but runtime should improve
