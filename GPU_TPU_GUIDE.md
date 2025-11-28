# GPU/TPU Support and JAX Transform Composition

## GPU/TPU Support in JAXFrame

### Key Principle: JAX Arrays = Automatic Acceleration

**Any operation that works on JAX arrays automatically runs on GPU/TPU if available.**

```python
import jax
import jaxframe as jf

# JAX automatically detects and uses available devices
print(jax.devices())  # [CpuDevice(id=0)] or [GpuDevice(id=0)]

# Create DataFrame - data goes to default device
df = jf.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

# This runs on GPU if available (no code change needed!)
result = df.sum()
```

### Operations That Support GPU/TPU

**Rule of Thumb:** If it supports JIT (🔥), it supports GPU/TPU

| Operation | CPU | GPU | TPU | Notes |
|-----------|-----|-----|-----|-------|
| Arithmetic (`+`, `-`, `*`, `/`) | ✅ | ✅ | ✅ | Fully accelerated |
| Aggregations (`sum`, `mean`, `std`) | ✅ | ✅ | ✅ | Fully accelerated |
| Comparisons (`>`, `<`, `==`) | ✅ | ✅ | ✅ | Fully accelerated |
| `where()`, `clip()` | ✅ | ✅ | ✅ | Fully accelerated |
| Matrix ops (`@`) | ✅ | ✅ | ✅ | **Huge GPU speedup** |
| Sorting | ✅ | ✅ | ✅ | Accelerated |
| GroupBy (encoded) | ✅ | ✅ | ✅ | Uses segment ops |
| Rolling windows | ✅ | ✅ | ✅ | Accelerated |
| **Object columns** | ✅ | ❌ | ❌ | CPU only (NumPy arrays) |
| **String operations** | ✅ | ❌ | ❌ | CPU only |
| **I/O operations** | ✅ | ❌ | ❌ | CPU only |

### Setting the Default Device

JAX allows you to control which device is used by default:

```python
import jax
import jax.numpy as jnp
import jaxframe as jf

# Method 1: Set default device globally
jax.config.update('jax_default_device', jax.devices('gpu')[0])

# Now all DataFrames go to GPU by default
df = jf.DataFrame({'x': [1, 2, 3]})  # Automatically on GPU

# Method 2: Set via environment variable (before importing JAX)
# export JAX_PLATFORMS=gpu  # In bash
# or
import os
os.environ['JAX_PLATFORMS'] = 'gpu'
import jax  # Must import after setting env var
```

### Device Transfer Helper Methods

JAXFrame provides convenient methods to transfer DataFrames between devices:

```python
import jax
import jaxframe as jf

# Create DataFrame (on default device, usually CPU)
df = jf.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

# Check current device
print(df.device)  # CpuDevice(id=0)

# Transfer to GPU
df_gpu = df.to_device('gpu')
print(df_gpu.device)  # GpuDevice(id=0)

# Or use convenience methods
df_gpu = df.to_gpu()      # Transfer to default GPU
df_gpu = df.to_gpu(0)     # Transfer to GPU 0
df_tpu = df.to_tpu()      # Transfer to default TPU
df_cpu = df_gpu.to_cpu()  # Transfer back to CPU

# Transfer to specific device
device = jax.devices('gpu')[1]  # GPU 1
df_gpu1 = df.to_device(device)

# Multi-GPU: Distribute across devices
df_sharded = df.to_devices([
    jax.devices('gpu')[0],
    jax.devices('gpu')[1]
])  # Each chunk on different GPU
```

### Automatic Device Placement

```python
import jax
import jax.numpy as jnp
import jaxframe as jf

# Let JAX choose automatically (uses default device)
df = jf.DataFrame({'x': [1, 2, 3]})  # Goes to default device

# Operations automatically run on the DataFrame's device
result = df.sum()  # Runs on same device as df

# Explicit device placement at construction time (old way)
df_gpu = jf.DataFrame({
    'x': jax.device_put(jnp.array([1, 2, 3]), jax.devices('gpu')[0]),
    'y': jax.device_put(jnp.array([4, 5, 6]), jax.devices('gpu')[0])
})

# New way (much cleaner!)
df_gpu = jf.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}).to_gpu()
```

### Performance on Different Devices

**CPU vs GPU Speedup (typical workloads):**

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Small ops (<1K elements) | Fast | Slow | 0.1x (overhead) |
| Medium ops (1K-100K) | Baseline | 2-5x | 2-5x |
| Large ops (100K-1M) | Baseline | 10-50x | 10-50x |
| Matrix multiply | Baseline | 100-1000x | **100-1000x** |
| Repeated JIT ops | Baseline | 50-200x | **50-200x** |

**Key Insight:** GPU/TPU excel at:
- ✅ Large batch operations
- ✅ Repeated JIT-compiled functions
- ✅ Matrix operations
- ❌ Small one-off operations (overhead dominates)

---

## Mixing JIT/grad Supported and Unsupported Operations

### Rule: The Entire Function Must Be Traceable

**If ANY operation in a function is not JIT/grad compatible, the ENTIRE function fails.**

### Example 1: Mixing Supported and Unsupported (FAILS)

```python
import jax
import jaxframe as jf

df = jf.DataFrame({'x': [1, 2, 3, 4, 5]})

@jax.jit
def bad_function(df):
    # This is JIT-compatible
    doubled = df * 2

    # This is NOT JIT-compatible (dynamic shape)
    filtered = doubled[doubled > 5]  # ❌ FAILS!

    return filtered.sum()

# This will crash!
try:
    result = bad_function(df)
except Exception as e:
    print(f"Error: {e}")
    # ConcretizationTypeError: Abstract tracer value encountered...
```

**Why it fails:** JAX traces through the entire function at compile time. When it hits `doubled[doubled > 5]`, it doesn't know the output shape (depends on data), so it can't compile.

### Example 2: Splitting Into Supported Parts (WORKS)

```python
@jax.jit
def jit_part(df):
    """JIT-compatible operations only."""
    return df * 2

def non_jit_part(df):
    """Non-JIT operations."""
    # This changes shape, can't be JIT'd
    return df[df > 5]

# Use them separately
df = jf.DataFrame({'x': [1, 2, 3, 4, 5]})
doubled = jit_part(df)  # ✅ JIT compiled
filtered = non_jit_part(doubled)  # ✅ Runs eagerly
result = filtered.sum()
```

### Example 3: Using JIT-Friendly Alternatives (BEST)

```python
@jax.jit
def good_function(df, threshold):
    """All operations JIT-compatible."""
    doubled = df * 2
    # Use where() instead of boolean indexing
    masked = doubled.where(doubled > threshold, fill_value=0)
    return masked.sum()

df = jf.DataFrame({'x': [1, 2, 3, 4, 5]})
result = good_function(df, 5.0)  # ✅ Fully JIT'd!
```

### What Happens with grad?

Same principle: **entire function must be differentiable**.

```python
# FAILS: Contains non-differentiable operation
@jax.grad
def bad_loss(df):
    x = df._numeric_data
    # argmax is not differentiable
    idx = jnp.argmax(x)  # ❌ FAILS
    return x[idx]

# WORKS: All operations differentiable
@jax.grad
def good_loss(df):
    # sum, mean, etc. are all differentiable
    return (df._numeric_data ** 2).mean()
```

---

## Nested Functions and Composition

### Nested Functions Work Perfectly!

JAX traces through all nested calls, as long as all operations are traceable.

```python
import jax
import jax.numpy as jnp
import jaxframe as jf

# Level 3: Innermost function
@jax.jit
def normalize(df):
    """Normalize data."""
    return (df - df.mean()) / df.std()

# Level 2: Middle function
@jax.jit
def compute_features(df):
    """Compute features from normalized data."""
    normalized = normalize(df)  # ✅ Nested JIT call
    return normalized ** 2

# Level 1: Outer function
@jax.jit
def full_pipeline(df):
    """Full processing pipeline."""
    features = compute_features(df)  # ✅ Nested JIT call
    return features.sum()

# This works! JAX inlines all the nested calls
df = jf.DataFrame({'x': [1, 2, 3, 4, 5]})
result = full_pipeline(df)  # ✅ Entire pipeline compiled as one unit
```

**What happens:**
1. JAX traces through `full_pipeline`
2. When it hits `compute_features`, it inlines that function
3. When it hits `normalize`, it inlines that too
4. Result: One optimized compiled function

### Composition of grad and JIT

You can compose transformations in either order:

```python
# Option 1: JIT of grad
jitted_grad = jax.jit(jax.grad(loss_fn))

# Option 2: grad of JIT
grad_of_jit = jax.grad(jax.jit(loss_fn))

# Both work! Usually Option 1 is preferred (compile the gradient computation)
```

### Example: Multi-Level Nested Pipeline

```python
import jax
import jax.numpy as jnp
import jaxframe as jf

# Define reusable operations
@jax.jit
def scale(df, factor):
    return df * factor

@jax.jit
def add_offset(df, offset):
    return df + offset

@jax.jit
def compute_stats(df):
    return {
        'mean': df.mean(axis=None),
        'std': df.std(axis=None),
    }

# Compose into larger function
@jax.jit
def preprocess(df, scale_factor, offset):
    """Multi-step preprocessing."""
    step1 = scale(df, scale_factor)      # ✅ Nested
    step2 = add_offset(step1, offset)    # ✅ Nested
    return step2

# Use in differentiable pipeline
def loss(df, params):
    """Loss function with preprocessing."""
    processed = preprocess(df, params['scale'], params['offset'])  # ✅ Nested
    stats = compute_stats(processed)  # ✅ Nested
    return (stats['mean'] - params['target']) ** 2

# Get gradients through entire nested pipeline
grad_fn = jax.grad(loss, argnums=1)  # Gradient w.r.t. params

df = jf.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
params = {'scale': 2.0, 'offset': 1.0, 'target': 5.0}

grads = grad_fn(df, params)  # ✅ Works! Gradients through all nested calls
print(grads)  # {'scale': ..., 'offset': ..., 'target': ...}
```

---

## Common Patterns and Pitfalls

### ✅ Pattern 1: Separate JIT and Non-JIT Code

```python
# Good: Clear separation
def load_data():
    """I/O - not JIT-able."""
    return jf.DataFrame.from_csv('data.csv')

@jax.jit
def process_data(df):
    """Processing - JIT-able."""
    return (df - df.mean()) / df.std()

def save_results(df):
    """I/O - not JIT-able."""
    df.to_csv('results.csv')

# Usage
df = load_data()           # CPU
processed = process_data(df)  # GPU/TPU
save_results(processed)    # CPU
```

### ✅ Pattern 2: JIT the Hot Loop

```python
@jax.jit
def train_step(df, params):
    """Inner loop - fully JIT'd."""
    predictions = df._numeric_data @ params['weights']
    loss = ((predictions - df._numeric_data[:, -1]) ** 2).mean()
    return loss

# Outer loop not JIT'd (contains I/O, logging, etc.)
def train(df, params, epochs):
    for epoch in range(epochs):
        loss = train_step(df, params)  # ✅ JIT'd

        # Non-JIT operations
        print(f"Epoch {epoch}, Loss: {loss}")  # I/O
        if loss < threshold:
            break  # Data-dependent control flow
```

### ✅ Pattern 3: Static vs Dynamic Arguments

```python
from functools import partial

@partial(jax.jit, static_argnums=(1,))  # axis is static
def flexible_sum(df, axis):
    """Sum with static axis argument."""
    return df.sum(axis=axis)

df = jf.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

# These compile to different functions (different static args)
sum_rows = flexible_sum(df, axis=0)  # Compiled once
sum_cols = flexible_sum(df, axis=1)  # Compiled separately
```

### ❌ Pitfall 1: Mixing Object and Numeric Operations

```python
@jax.jit
def bad_mixed(df):
    # Numeric ops - OK
    result = df[['x', 'y']].sum()

    # Object column access - NOT OK in JIT!
    names = df['name']  # ❌ Tries to access object data

    return result

# Fix: Only use numeric columns in JIT
@jax.jit
def good_numeric_only(df_numeric):
    return df_numeric.sum()

df = jf.DataFrame({'x': [1, 2], 'y': [3, 4], 'name': ['a', 'b']})
df_numeric = df[['x', 'y']]
result = good_numeric_only(df_numeric)  # ✅
```

### ❌ Pitfall 2: Data-Dependent Shapes

```python
@jax.jit
def bad_dynamic_shape(df, threshold):
    # Shape depends on data - can't compile!
    filtered = df[df > threshold]  # ❌
    return filtered.sum()

# Fix: Use fixed-shape alternative
@jax.jit
def good_fixed_shape(df, threshold):
    # Shape stays the same
    masked = df.where(df > threshold, fill_value=0)  # ✅
    return masked.sum()
```

### ❌ Pitfall 3: Side Effects in JIT

```python
@jax.jit
def bad_side_effects(df):
    result = df.sum()
    print(f"Sum: {result}")  # ❌ Side effect (I/O)
    return result

# Fix: Move side effects outside JIT
@jax.jit
def good_no_side_effects(df):
    return df.sum()

result = good_no_side_effects(df)
print(f"Sum: {result}")  # ✅ Print outside JIT
```

---

## Decision Tree: Can My Function Be JIT'd?

```
Can your function be JIT compiled?

1. Does it change array shapes based on data?
   └─ YES → ❌ NO (use where/clip instead of boolean indexing)
   └─ NO  → Continue to 2

2. Does it access object/string columns?
   └─ YES → ❌ NO (split into numeric-only part)
   └─ NO  → Continue to 3

3. Does it do I/O (print, file read/write)?
   └─ YES → ❌ NO (move I/O outside JIT)
   └─ NO  → Continue to 4

4. Does it have data-dependent control flow (if/while with array conditions)?
   └─ YES → ❌ NO (use jnp.where or static_argnums)
   └─ NO  → Continue to 5

5. Does it only use JAX/NumPy operations?
   └─ YES → ✅ YES! Your function can be JIT'd
   └─ NO  → ❌ NO (replace with JAX equivalents)
```

---

## GPU/TPU Best Practices

### 1. Batch Your Operations

```python
# Bad: Many small operations
for i in range(1000):
    df = jf.DataFrame({'x': [i]})
    result = df.sum()  # GPU overhead kills performance

# Good: One large batched operation
df = jf.DataFrame({'x': range(1000)})
result = df.sum()  # 100x faster on GPU!
```

### 2. Use JIT for Repeated Operations

```python
# Bad: Recompiling every time
for batch in data_loader:
    result = compute(batch)  # Recompiles each time

# Good: Compile once, reuse
@jax.jit
def compute(batch):
    return (batch * 2).sum()

for batch in data_loader:
    result = compute(batch)  # Compiled once, reused
```

### 3. Profile and Optimize

```python
import jax
import jax.profiler

# Profile GPU utilization
jax.profiler.start_trace("/tmp/jax-trace")

@jax.jit
def my_function(df):
    return (df ** 2).sum()

result = my_function(large_df)

jax.profiler.stop_trace()
# View trace at chrome://tracing
```

---

## Summary

### GPU/TPU Support
- ✅ **Automatic**: If it's a JAX array operation, it runs on GPU/TPU
- ✅ **Same code**: No changes needed for GPU vs CPU
- ✅ **Huge speedup**: 10-1000x for large operations
- ❌ **Object columns**: CPU only (NumPy arrays)

### JIT/grad Composition
- ✅ **Nested functions**: Work perfectly if all operations are traceable
- ✅ **Composition**: Can compose `jax.jit`, `jax.grad`, `jax.vmap` freely
- ❌ **Mixed operations**: One non-traceable op breaks the entire function
- ✅ **Solution**: Split code or use traceable alternatives

### Key Takeaways

1. **JIT-able = GPU-able**: Same compatibility
2. **Entire function must be traceable**: Can't mix supported/unsupported ops
3. **Nested functions compose**: JAX inlines everything
4. **Separate concerns**: Keep JIT-able code separate from I/O, dynamic ops
5. **Use alternatives**: `where()` instead of `df[mask]`, static args for control flow

**Golden Rule:** If you can JIT it, you can run it on GPU/TPU and differentiate through it! 🚀
