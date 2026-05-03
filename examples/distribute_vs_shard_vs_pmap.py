"""
Example: distribute() vs shard() vs pmap

Demonstrates the differences between three multi-GPU approaches:
1. distribute() - Manual chunks + explicit pmap
2. shard() - Automatic parallelism (JAX handles it)
3. pmap - Low-level explicit data parallelism

Key differences and when to use each.
"""

import sys

sys.path.insert(0, '..')

import jax
import jax.numpy as jnp

from jaxframe import DataFrame


def example_conceptual_differences():
    """Explain the conceptual differences."""
    print("=" * 70)
    print("Conceptual Differences: distribute() vs shard() vs pmap")
    print("=" * 70)

    print("""
1. distribute() - Manual Data Parallelism
   - Splits DataFrame into separate chunks (one per GPU)
   - Returns multiple DataFrame objects (or wrapper)
   - YOU write pmap to process chunks in parallel
   - Explicit control over what runs in parallel

   Flow: DataFrame → split → [chunk1, chunk2] → pmap → results

   Example:
   ```python
   df_chunks = df.distribute()  # Returns list of DataFrames

   @jax.pmap
   def process(chunk):
       return chunk.sum()

   results = process(stack_chunks(df_chunks))  # Manual pmap
   ```

2. shard() - Automatic Parallelism
   - Data is logically distributed across GPUs
   - Returns single DataFrame (data is sharded under the hood)
   - JAX AUTOMATICALLY parallelizes operations
   - Transparent - you don't write pmap

   Flow: DataFrame → shard → ShardedDataFrame → operations (auto-parallel)

   Example:
   ```python
   df_sharded = df.shard(axis=0)  # Returns single DataFrame

   result = df_sharded.sum()  # JAX auto-parallelizes! No pmap needed
   ```

3. pmap - Low-Level Primitive
   - Explicit function for data parallelism
   - Maps a function across the first axis (device axis)
   - YOU control what runs in parallel
   - Building block for distribute()

   Flow: Write per-device function → pmap wraps it → runs in parallel

   Example:
   ```python
   @jax.pmap
   def process_batch(batch_data):  # Runs on each device
       return batch_data.sum()

   batched = stack([chunk1, chunk2])  # Stack along device axis
   results = process_batch(batched)  # Each device processes its batch
   ```

Key Insight:
- distribute() = you split data + you write pmap
- shard() = JAX splits data + JAX handles parallelism automatically
- pmap = primitive that both distribute() and shard() use internally
    """)

    print("\n" + "=" * 70)


def example_distribute_manual():
    """Example: distribute() with manual pmap."""
    print("\n" + "=" * 70)
    print("Example 1: distribute() - Manual Control")
    print("=" * 70)

    df = DataFrame({
        'x': jnp.arange(100),
        'y': jnp.arange(100, 200),
    })

    print("\nOriginal DataFrame shape:", df.shape)

    # STEP 1: Manually split (what distribute() would do)
    n_devices = len(jax.devices('cpu'))  # Using CPU for demo
    chunk_size = len(df) // n_devices

    chunks = []
    for i in range(n_devices):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_devices - 1 else len(df)

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

    print(f"\nSplit into {len(chunks)} chunks")
    print(f"Chunk 0 shape: {chunks[0].shape}")

    # STEP 2: Stack chunks for pmap (add device dimension)
    stacked = jnp.stack([c._numeric_data for c in chunks])
    print(f"\nStacked shape: {stacked.shape}")  # (n_devices, rows_per_device, cols)

    # STEP 3: Define per-device computation
    @jax.pmap
    def process_chunk(chunk_data):
        """This runs on EACH device independently."""
        # Each device gets one slice: chunk_data.shape = (rows_per_device, cols)
        normalized = (chunk_data - chunk_data.mean()) / chunk_data.std()
        return normalized.sum(axis=0)  # Sum within this device's chunk

    # STEP 4: Run in parallel
    results = process_chunk(stacked)  # Shape: (n_devices, n_cols)
    print(f"\nResults shape: {results.shape}")

    # STEP 5: Combine results across devices
    final_result = results.sum(axis=0)  # Sum across devices
    print(f"Final result: {final_result}")

    print("\n✅ distribute() requires YOU to write pmap and combine results")
    print("   Pros: Explicit control")
    print("   Cons: More code, manual split/combine")


def example_shard_automatic():
    """Example: shard() with automatic parallelism."""
    print("\n" + "=" * 70)
    print("Example 2: shard() - Automatic Parallelism (CONCEPTUAL)")
    print("=" * 70)

    df = DataFrame({
        'x': jnp.arange(100),
        'y': jnp.arange(100, 200),
    })

    print("\nOriginal DataFrame shape:", df.shape)

    print("""
CONCEPTUAL: What df.shard() would do with modern JAX:
-----------------------------------------------------
# STEP 1: Create sharding specification
devices = jax.devices('gpu')  # All GPUs
sharding = PositionalSharding(devices)

# Shard along first axis (rows)
sharded_data = jax.device_put(
    df._numeric_data,
    sharding.reshape(len(devices), 1)  # Split rows, replicate columns
)

df_sharded = DataFrame._from_parts(
    numeric_data=sharded_data,
    ...
)

# STEP 2: Operations are AUTOMATICALLY parallel!
# NO pmap needed - JAX handles it!

@jax.jit
def compute(df):
    normalized = (df._numeric_data - df._numeric_data.mean()) / df._numeric_data.std()
    return normalized.sum(axis=0)

result = compute(df_sharded)  # JAX auto-parallelizes!

Key Point: You write normal code, JAX figures out how to parallelize it
based on the sharding specification!
    """)

    # Demonstrate on single device (same logic, just not multi-device)
    @jax.jit
    def compute(data):
        """Regular function - no pmap needed!"""
        normalized = (data - data.mean()) / data.std()
        return normalized.sum(axis=0)

    result = compute(df._numeric_data)
    print(f"\nDemo result (single device): {result}")

    print("\n✅ shard() handles parallelism automatically!")
    print("   Pros: Transparent, less code, automatic optimization")
    print("   Cons: Less explicit control")
    print("   Note: Requires JAX >= 0.4.0 for PositionalSharding API")


def example_when_to_use_pmap():
    """When you need pmap explicitly."""
    print("\n" + "=" * 70)
    print("Example 3: When You NEED pmap")
    print("=" * 70)

    print("""
You need pmap when:

1. Custom Training Loops
   - Different computation per device
   - Gradient aggregation across devices
   - Parameter synchronization

2. Stateful Computations
   - Each device maintains its own state
   - Random number generators per device

3. Custom Collectives
   - All-reduce, all-gather, etc.
   - Cross-device communication

Example: Distributed Training
    """)

    # Training example
    df = DataFrame({
        'features': jnp.arange(100),
        'labels': jnp.arange(100, 200),
    })

    # Split into batches per device
    n_devices = len(jax.devices('cpu'))
    per_device_data = jnp.stack([
        df._numeric_data[i::n_devices] for i in range(n_devices)
    ])

    print(f"\nPer-device data shape: {per_device_data.shape}")
    print("  (devices, samples_per_device, features)")

    # Define training step with pmap
    @jax.pmap
    def train_step(batch, params):
        """
        This runs on EACH device with its own batch.

        pmap gives you:
        - batch for this device
        - params replicated across devices
        """
        # Simple computation on this device's batch
        # Each device computes mean of its own data
        local_mean = batch.mean()

        # Simple "gradient" computation (just for demo)
        # In real training, this would be actual gradient computation
        grad_value = (batch * params).sum()

        # pmap_axis_name enables cross-device operations
        # grad_value = jax.lax.pmean(grad_value, axis_name='devices')  # Average across devices

        return grad_value, local_mean

    # Initialize params (replicated across devices)
    params = jnp.ones((2,))
    # Replicate params for each device (add device dimension)
    params_replicated = jnp.stack([params] * n_devices)

    # Run training step in parallel
    grad_values, means = train_step(per_device_data, params_replicated)

    print(f"\nGrad values shape: {grad_values.shape}")  # (n_devices,)
    print(f"Means shape: {means.shape}")      # (n_devices,)
    print(f"Per-device means: {means}")

    print("\n✅ pmap needed for custom training logic!")
    print("   You control exactly what runs in parallel")


def example_comparison_table():
    """Side-by-side comparison."""
    print("\n" + "=" * 70)
    print("Example 4: Side-by-Side Comparison")
    print("=" * 70)

    print("""
┌──────────────────────────────────────────────────────────────────┐
│                    distribute() vs shard() vs pmap               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 1. distribute() - Manual Data Parallelism                       │
│    df_chunks = df.distribute()                                  │
│    @jax.pmap                                                     │
│    def process(chunk): ...                                      │
│    results = process(stack(chunks))                             │
│                                                                  │
│    ✓ Explicit control                                           │
│    ✓ Clear what's happening                                     │
│    ✗ More boilerplate                                           │
│    ✗ Manual combine step                                        │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 2. shard() - Automatic Parallelism                              │
│    df_sharded = df.shard(axis=0)                                │
│    result = df_sharded.sum()  # Auto-parallel!                 │
│                                                                  │
│    ✓ Transparent                                                │
│    ✓ Less code                                                  │
│    ✓ JAX optimizes automatically                                │
│    ✗ Less explicit control                                      │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 3. pmap - Low-Level Primitive                                   │
│    @jax.pmap                                                     │
│    def process(batch, params): ...                              │
│    results = process(batched_data, params)                      │
│                                                                  │
│    ✓ Full control                                               │
│    ✓ Custom collectives                                         │
│    ✓ Stateful computations                                      │
│    ✗ Requires understanding data layout                         │
│    ✗ More complex                                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

When to use each:

distribute() → When you want explicit control but convenience
               Example: Custom processing with clear split/combine

shard()      → When you want JAX to handle everything (RECOMMENDED)
               Example: Large DataFrame that doesn't fit on one GPU

pmap         → When you need custom distributed logic
               Example: Training loops, stateful computations
    """)


def example_real_world_recommendation():
    """Real-world recommendation."""
    print("\n" + "=" * 70)
    print("Example 5: Real-World Recommendation")
    print("=" * 70)

    print("""
Recommended Usage Pattern:

┌─────────────────────────────────────────────────────────────┐
│ For 90% of use cases:                                       │
│                                                             │
│ 1. Start with single GPU:                                  │
│    df_gpu = df.to_gpu()                                    │
│    result = process(df_gpu)                                │
│                                                             │
│ 2. If data doesn't fit on one GPU, use shard():           │
│    df_sharded = df.shard(axis=0)  # Auto-parallel         │
│    result = process(df_sharded)   # Same code!            │
│                                                             │
│ 3. Only use pmap if you need:                              │
│    - Custom training loops                                 │
│    - Cross-device communication                            │
│    - Device-specific state                                 │
└─────────────────────────────────────────────────────────────┘

Example: Large DataFrame Processing

```python
# Load large dataset
df = jf.DataFrame.from_parquet('huge_data.parquet')  # 100GB

# Option 1: shard() - RECOMMENDED
df_sharded = df.shard(axis=0, devices=jax.devices('gpu'))

# Process as normal - JAX handles parallelism!
normalized = (df_sharded - df_sharded.mean()) / df_sharded.std()
result = normalized.sum()  # Automatically parallel

# Option 2: distribute() with pmap - MORE CONTROL
df_chunks = df.distribute()  # Split into chunks

@jax.pmap
def process_chunk(chunk):
    normalized = (chunk - chunk.mean()) / chunk.std()
    return normalized.sum()

results = process_chunk(stack_chunks(df_chunks))
final = results.sum()  # Combine
```

Example: Custom Training

```python
# When you need pmap explicitly
@jax.pmap
def train_step(batch, params, state):
    # Custom logic per device
    predictions = model(batch, params)
    loss = compute_loss(predictions, batch['labels'])

    # Gradients on this device
    grads = jax.grad(loss)(params)

    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='devices')

    # Update params and state
    new_params = update(params, grads)
    new_state = update_state(state)

    return new_params, new_state, loss

# Training loop
for batch in data_loader:
    params, state, loss = train_step(batch, params, state)
```
    """)

    print("\n✅ Summary:")
    print("   • shard() for most cases (auto-parallel)")
    print("   • distribute() when you want explicit chunks")
    print("   • pmap when you need custom distributed logic")


if __name__ == "__main__":
    example_conceptual_differences()
    example_distribute_manual()
    example_shard_automatic()
    example_when_to_use_pmap()
    example_comparison_table()
    example_real_world_recommendation()

    print("\n" + "=" * 70)
    print("All examples completed! 🎉")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • distribute() = manual chunks + YOU write pmap")
    print("  • shard() = automatic parallelism (RECOMMENDED)")
    print("  • pmap = low-level primitive for custom logic")
    print("  • Start with shard(), use pmap only when needed")
    print("=" * 70)
