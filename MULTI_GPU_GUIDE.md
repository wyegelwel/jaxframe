# Multi-GPU Strategies for JAXFrame

## Overview

JAX provides several approaches for multi-GPU computation. For DataFrames, the key strategies are:

1. **Data Parallelism (`pmap`)** - Split data across GPUs, replicate computation
2. **Sharding (`jax.Array`)** - Modern approach, automatic parallelism
3. **Manual Device Placement** - Explicit control over which GPU gets what data

---

## Strategy 1: Data Parallelism with `pmap` (Recommended)

### Concept
Split DataFrame rows across GPUs, run same computation on each chunk in parallel.

```
Original DataFrame (1000 rows)
         ↓
    Split rows
         ↓
GPU 0: rows 0-499     GPU 1: rows 500-999
         ↓                      ↓
    Same computation      Same computation
         ↓                      ↓
    Result 0              Result 1
         ↓                      ↓
         Combine results
```

### Implementation

```python
import jax
import jax.numpy as jnp
from jaxframe import DataFrame

# Assume 2 GPUs available
n_devices = len(jax.devices('gpu'))  # 2

# Create large DataFrame
df = DataFrame({
    'x': jnp.arange(1000),
    'y': jnp.arange(1000, 2000),
})

# Method 1: Split manually and use pmap
def split_dataframe(df, n_chunks):
    """Split DataFrame into n_chunks along rows."""
    n_rows = df.shape[0]
    chunk_size = n_rows // n_chunks

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else n_rows

        # Create chunk with subset of rows
        chunk_data = df._numeric_data[start:end]
        chunk = DataFrame._from_parts(
            numeric_data=chunk_data,
            numeric_cols=df._numeric_cols,
            numeric_dtypes=df._numeric_dtypes,
            object_data={},
            index=df._index[start:end],
            column_order=df._column_order,
        )
        chunks.append(chunk)

    return chunks

# Split into chunks (one per GPU)
df_chunks = split_dataframe(df, n_devices)

# Define computation on single chunk
def process_chunk(df_chunk):
    """Process a single chunk of data."""
    normalized = (df_chunk - df_chunk.mean(axis=0))
    return normalized.sum(axis=0)

# Use pmap to run in parallel across GPUs
@jax.pmap
def parallel_process(df_numeric_data):
    """Process data in parallel across devices."""
    # Work with raw numeric data for pmap
    normalized = (df_numeric_data - df_numeric_data.mean(axis=0))
    return normalized.sum(axis=0)

# Prepare data for pmap (stack chunks)
stacked_data = jnp.stack([chunk._numeric_data for chunk in df_chunks])

# Run in parallel!
results = parallel_process(stacked_data)  # Shape: (n_devices, n_cols)

# Combine results
final_result = results.sum(axis=0)  # Sum across devices
```

### Using JAXFrame's Built-in Multi-GPU Support

```python
import jaxframe as jf

# Create DataFrame
df = jf.DataFrame({'x': range(1000), 'y': range(1000, 2000)})

# Split across all available GPUs
df_distributed = df.distribute()  # Auto-detects GPUs

# Or specify devices explicitly
df_distributed = df.distribute(devices=jax.devices('gpu'))

# Operations automatically run in parallel
result = df_distributed.sum(axis=0)  # Parallel across GPUs!
```

---

## Strategy 2: Sharding (Modern JAX Approach)

### Concept
Use JAX's automatic sharding with `jax.Array`. JAX handles parallelism automatically.

```python
import jax
from jax.sharding import PositionalSharding

# Create sharding specification
sharding = PositionalSharding(jax.devices('gpu'))

# Create DataFrame with sharded data
data = jnp.arange(1000 * 100).reshape(1000, 100)
sharded_data = jax.device_put(data, sharding.reshape(2, 1))  # Shard first axis

df = DataFrame._from_parts(
    numeric_data=sharded_data,
    numeric_cols=tuple(f'col_{i}' for i in range(100)),
    numeric_dtypes=(jnp.float32,) * 100,
    object_data={},
    index=jnp.arange(1000),
    column_order=tuple(f'col_{i}' for i in range(100)),
)

# Operations automatically use sharding
result = df.sum(axis=0)  # JAX automatically parallelizes!
```

### JAXFrame API

```python
import jaxframe as jf

df = jf.DataFrame({'x': range(10000), 'y': range(10000, 20000)})

# Shard along rows (first axis)
df_sharded = df.shard(axis=0, devices=jax.devices('gpu'))

# Shard along columns (second axis) - for wide DataFrames
df_sharded = df.shard(axis=1, devices=jax.devices('gpu'))

# 2D sharding (rows and columns)
df_sharded = df.shard_2d(
    row_devices=jax.devices('gpu')[:2],
    col_devices=jax.devices('gpu')[2:4]
)
```

---

## Strategy 3: Manual Device Placement

### Use Case
When you need explicit control over which GPU gets which data.

```python
import jax
import jaxframe as jf

# Get available GPUs
gpus = jax.devices('gpu')

# Split DataFrame manually
df = jf.DataFrame({'x': range(1000), 'y': range(1000, 2000)})

# Method 1: Split into chunks, place on different GPUs
df_chunks = []
chunk_size = len(df) // len(gpus)

for i, gpu in enumerate(gpus):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < len(gpus) - 1 else len(df)

    # Create chunk
    chunk = df.iloc[start:end]  # (Future API)

    # Place on specific GPU
    chunk_gpu = chunk.to_device(gpu)
    df_chunks.append(chunk_gpu)

# Process each chunk independently
results = []
for chunk in df_chunks:
    result = process(chunk)  # Runs on chunk's GPU
    results.append(result)

# Combine results (may need to transfer to same device)
combined = jnp.concatenate([r.to_cpu() for r in results])
```

---

## Comparison of Strategies

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| **pmap** | Same computation on different data | Simple, explicit | Need to split/combine manually |
| **Sharding** | Automatic parallelism | Transparent, efficient | Less control |
| **Manual** | Custom workloads | Full control | More code, error-prone |

---

## Common Patterns

### Pattern 1: Distributed Training

```python
import jax
import jaxframe as jf

# Large training dataset
train_df = jf.DataFrame({'features': ..., 'labels': ...})

# Shard across GPUs
train_df_sharded = train_df.distribute()

# Training step (automatically parallel)
@jax.pmap
def train_step(batch_data, params):
    predictions = model(batch_data, params)
    loss = compute_loss(predictions, batch_data['labels'])
    grads = jax.grad(loss)(params)
    return grads

# Batch processing (each GPU gets a batch)
for epoch in range(n_epochs):
    for batch in batches(train_df_sharded):
        grads = train_step(batch, params)
        params = update(params, grads)
```

### Pattern 2: Large DataFrame Processing

```python
# Process DataFrame larger than single GPU memory
large_df = jf.DataFrame.from_parquet('huge_file.parquet')

# Shard across GPUs
large_df_sharded = large_df.shard(axis=0)  # Split rows

# Operations work transparently
result = large_df_sharded.groupby('category').mean()  # Parallel!
```

### Pattern 3: Pipeline Parallelism

```python
# Different stages on different GPUs
df = jf.DataFrame(...)

# Stage 1: Preprocessing on GPU 0
df_preprocessed = preprocess(df).to_device(jax.devices('gpu')[0])

# Stage 2: Feature extraction on GPU 1
df_features = extract_features(df_preprocessed).to_device(jax.devices('gpu')[1])

# Stage 3: Prediction on GPU 2
predictions = predict(df_features).to_device(jax.devices('gpu')[2])
```

---

## Performance Tips

### 1. Minimize Cross-Device Communication

```python
# Bad: Lots of data transfer between GPUs
for i in range(100):
    df_gpu0 = df.to_device(gpus[0])
    result = process(df_gpu0)
    df_gpu1 = result.to_device(gpus[1])  # Slow!

# Good: Keep data on same GPU
df_gpu0 = df.to_device(gpus[0])
for i in range(100):
    df_gpu0 = process(df_gpu0)  # Stays on GPU 0
```

### 2. Batch Size Per GPU

```python
# Rule of thumb: batch_size = total_batch_size // n_gpus
n_gpus = len(jax.devices('gpu'))
per_device_batch_size = 256 // n_gpus  # 128 per GPU with 2 GPUs

# Each GPU processes its batch
@jax.pmap
def process_batch(batch):
    return batch.sum()  # Each GPU sums its batch
```

### 3. Use pmap for Data Parallel, Sharding for Large Data

```python
# Data parallel (same model, different data)
@jax.pmap
def train(batch, params):
    ...

# Large data (data doesn't fit on one GPU)
df_sharded = df.shard(axis=0)  # Automatic sharding
```

---

## Recommended JAXFrame API

### Methods to Add

```python
class DataFrame:
    def distribute(self, devices=None, axis=0):
        """
        Distribute DataFrame across devices.

        Args:
            devices: List of devices (default: all GPUs)
            axis: Axis to split (0=rows, 1=columns)

        Returns:
            DistributedDataFrame
        """

    def shard(self, axis=0, devices=None):
        """
        Shard DataFrame using JAX sharding.

        Args:
            axis: Axis to shard
            devices: Devices to shard across

        Returns:
            DataFrame with sharded data
        """

    def gather(self):
        """
        Gather distributed DataFrame to single device.

        Returns:
            DataFrame on single device
        """
```

### Example Usage

```python
# Distribute automatically
df = jf.DataFrame({'x': range(10000), 'y': range(10000, 20000)})
df_dist = df.distribute()  # Auto-split across all GPUs

# Process in parallel
result = df_dist.sum(axis=0)  # Runs on all GPUs!

# Gather back to single device
df_gathered = df_dist.gather().to_cpu()
```

---

## Decision Tree: Which Strategy?

```
Do you have multiple GPUs?
├─ NO → Use single GPU (.to_gpu())
└─ YES
   │
   Do you need same computation on different data?
   ├─ YES → Use pmap (data parallelism)
   │        Example: Training, batch processing
   │
   └─ NO
      │
      Is your DataFrame too large for one GPU?
      ├─ YES → Use sharding (automatic)
      │        Example: Large dataset processing
      │
      └─ NO
         │
         Do you need fine-grained control?
         ├─ YES → Use manual placement
         │        Example: Pipeline parallelism
         │
         └─ NO → Use distribute() for simplicity
```

---

## Best Practices

1. **Start Simple**: Use `.distribute()` first
2. **Profile**: Check if you're actually GPU-bound
3. **Minimize Transfers**: Keep data on same device when possible
4. **Use pmap for Training**: Standard for distributed training
5. **Use Sharding for Large Data**: Automatic and efficient
6. **Test on Single GPU First**: Easier debugging

---

## Example: Complete Multi-GPU Workflow

```python
import jax
import jaxframe as jf

# 1. Load data (on CPU)
df = jf.DataFrame.from_parquet('data.parquet')
print(f"Loaded {len(df)} rows")

# 2. Distribute across GPUs
n_gpus = len(jax.devices('gpu'))
print(f"Distributing across {n_gpus} GPUs")

df_dist = df.distribute()

# 3. Parallel processing
@jax.pmap
def process_parallel(chunk_data):
    # Normalize
    normalized = (chunk_data - chunk_data.mean()) / chunk_data.std()
    # Compute features
    features = jnp.sin(normalized) + jnp.cos(normalized)
    return features

# Process in parallel
results = process_parallel(df_dist._numeric_data)

# 4. Gather results
final_result = jf.DataFrame._from_parts(
    numeric_data=results.reshape(-1, results.shape[-1]),
    numeric_cols=df._numeric_cols,
    numeric_dtypes=df._numeric_dtypes,
    object_data={},
    index=df._index,
    column_order=df._column_order,
)

# 5. Save (on CPU)
final_result.to_cpu().to_parquet('results.parquet')
```

---

## Summary

**For most use cases:**
- Use `df.distribute()` for automatic multi-GPU
- Use `pmap` for custom data-parallel workloads
- Use sharding for datasets larger than GPU memory

**Key principle:** JAX handles the parallelism, JAXFrame provides the convenient DataFrame API on top!
