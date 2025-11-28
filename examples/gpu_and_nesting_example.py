"""
Example: GPU/TPU Support and Function Composition

Demonstrates:
1. GPU/TPU automatic acceleration
2. Nested function composition
3. What works and what doesn't with JIT/grad
4. Best practices for mixed operations
"""

import sys
sys.path.insert(0, '..')

import jax
import jax.numpy as jnp
import time
from jaxframe import DataFrame


def example_gpu_detection():
    """Show available devices and automatic GPU usage."""
    print("=" * 60)
    print("Example 1: GPU/TPU Detection")
    print("=" * 60)

    # Check available devices
    devices = jax.devices()
    print(f"\nAvailable devices: {devices}")
    print(f"Default device: {jax.devices()[0]}")

    # Create DataFrame - automatically uses default device
    df = DataFrame({'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]})

    print(f"\nDataFrame data device: {df._numeric_data.device}")
    print("\n✅ Data automatically placed on available accelerator!")


def example_nested_functions():
    """Demonstrate nested function composition."""
    print("\n" + "=" * 60)
    print("Example 2: Nested Function Composition")
    print("=" * 60)

    # Define nested operations
    @jax.jit
    def normalize(df):
        """Normalize by subtracting overall mean."""
        mean_val = df.mean(axis=None)
        return df - mean_val

    @jax.jit
    def double(df):
        """Double all values."""
        return df * 2

    @jax.jit
    def pipeline(df):
        """Multi-step pipeline with nested calls."""
        step1 = normalize(df)  # First nested call
        step2 = double(step1)  # Second nested call
        result = step2.sum(axis=None)  # Final reduction
        return result

    df = DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [10.0, 20.0, 30.0, 40.0],
    })

    print("\nOriginal DataFrame:")
    print(df)

    result = pipeline(df)
    print(f"\nPipeline result: {result:.4f}")
    print("\n✅ Nested functions compiled into single optimized function!")


def example_jit_and_grad_composition():
    """Show that JIT and grad can be composed."""
    print("\n" + "=" * 60)
    print("Example 3: JIT + grad Composition")
    print("=" * 60)

    df = DataFrame({
        'x': [1.0, 2.0, 3.0],
        'y': [4.0, 5.0, 6.0],
    })

    # Define nested differentiable functions
    @jax.jit
    def preprocess(df):
        """Preprocessing step."""
        return df * 2

    @jax.jit
    def loss_fn(df):
        """Loss function that calls preprocessing."""
        processed = preprocess(df)  # Nested call
        return (processed._numeric_data * processed._numeric_data).sum()

    # Compose JIT and grad
    grad_fn = jax.jit(jax.grad(loss_fn))  # JIT the gradient computation!

    print("\nOriginal DataFrame:")
    print(df)

    # Compute gradients
    grads = grad_fn(df)

    print("\nGradients:")
    print(grads)
    print("\n✅ JIT and grad compose perfectly!")


def example_what_breaks_jit():
    """Show what breaks JIT compilation."""
    print("\n" + "=" * 60)
    print("Example 4: What Breaks JIT")
    print("=" * 60)

    df = DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})

    # This WOULD break (commented out to keep example running)
    print("\n❌ This would fail (dynamic shape):")
    print("""
    @jax.jit
    def bad_function(df):
        doubled = df * 2
        filtered = doubled[doubled > 5]  # ❌ Shape depends on data!
        return filtered.sum()
    """)

    # This works (fixed shape)
    @jax.jit
    def good_function(df):
        doubled = df * 2
        masked = doubled.where(doubled > 5, fill_value=0)  # ✅ Fixed shape
        return masked.sum(axis=None)

    print("\n✅ This works (fixed shape):")
    print("""
    @jax.jit
    def good_function(df):
        doubled = df * 2
        masked = doubled.where(doubled > 5, fill_value=0)  # ✅ Fixed shape
        return masked.sum(axis=None)
    """)

    result = good_function(df)
    print(f"\nResult: {result}")
    print("\n✅ Use where() instead of boolean indexing in JIT!")


def example_separating_jit_and_non_jit():
    """Show how to separate JIT and non-JIT code."""
    print("\n" + "=" * 60)
    print("Example 5: Separating JIT and Non-JIT Code")
    print("=" * 60)

    # JIT-able part
    @jax.jit
    def compute_features(df):
        """Fast JIT-compiled feature computation."""
        mean_val = df.mean(axis=None)
        normalized = df - mean_val
        doubled = normalized * 2
        return doubled

    # Non-JIT part (contains I/O)
    def process_with_logging(df, name):
        """Process data with logging (can't be JIT'd due to print)."""
        print(f"\nProcessing dataset: {name}")
        print(f"Input shape: {df.shape}")

        # Call JIT-compiled function
        features = compute_features(df)

        print(f"Output range: [{features._numeric_data.min():.2f}, {features._numeric_data.max():.2f}]")

        return features

    df = DataFrame({
        'sensor1': [10.0, 20.0, 30.0],
        'sensor2': [15.0, 25.0, 35.0],
    })

    result = process_with_logging(df, "Sensor Data")

    print("\n✅ Separate JIT-able compute from I/O/logging!")


def example_gpu_performance():
    """Compare CPU vs GPU performance (if GPU available)."""
    print("\n" + "=" * 60)
    print("Example 6: GPU Performance")
    print("=" * 60)

    # Create larger dataset
    n = 10000
    df = DataFrame({
        'x': jnp.arange(n, dtype=jnp.float32),
        'y': jnp.arange(n, 2*n, dtype=jnp.float32),
        'z': jnp.arange(2*n, 3*n, dtype=jnp.float32),
    })

    @jax.jit
    def complex_computation(df):
        """Complex computation to show GPU benefits."""
        result = df._numeric_data
        for _ in range(10):
            result = jnp.sin(result) * jnp.cos(result) + result ** 2
        return result.sum()

    print(f"\nDataFrame size: {df.shape}")
    print(f"Device: {df._numeric_data.device}")

    # Warmup
    _ = complex_computation(df)

    # Time it
    start = time.time()
    for _ in range(100):
        result = complex_computation(df)
    elapsed = time.time() - start

    print(f"\n100 iterations in {elapsed:.4f}s")
    print(f"Per iteration: {elapsed/100*1000:.2f}ms")

    if 'gpu' in str(df._numeric_data.device).lower():
        print("\n🚀 Running on GPU!")
    else:
        print("\n💻 Running on CPU (GPU would be ~10-100x faster)")

    print("\n✅ JIT + GPU = massive speedup for large operations!")


def example_multi_level_nesting():
    """Show deeply nested function calls."""
    print("\n" + "=" * 60)
    print("Example 7: Multi-Level Nested Functions")
    print("=" * 60)

    # Level 3: Atomic operations
    @jax.jit
    def add(df, value):
        return df + value

    @jax.jit
    def multiply(df, value):
        return df * value

    # Level 2: Combinations
    @jax.jit
    def scale_and_shift(df, scale, shift):
        scaled = multiply(df, scale)  # Nested level 3
        shifted = add(scaled, shift)  # Nested level 3
        return shifted

    # Level 1: Full pipeline
    @jax.jit
    def preprocess_and_aggregate(df, params):
        processed = scale_and_shift(df, params['scale'], params['shift'])  # Nested level 2
        result = processed.sum(axis=None)  # Final aggregation
        return result

    df = DataFrame({'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]})
    params = {'scale': 2.0, 'shift': 10.0}

    print("\nNesting structure:")
    print("  Level 1: preprocess_and_aggregate")
    print("    Level 2: scale_and_shift")
    print("      Level 3: add, multiply")

    result = preprocess_and_aggregate(df, params)
    print(f"\nResult: {result}")
    print("\n✅ JAX inlines all nested calls into one optimized function!")


def example_grad_through_nesting():
    """Gradients flow through nested functions."""
    print("\n" + "=" * 60)
    print("Example 8: Gradients Through Nested Functions")
    print("=" * 60)

    @jax.jit
    def layer1(df, w1):
        return df._numeric_data @ w1

    @jax.jit
    def layer2(x, w2):
        return x @ w2

    @jax.jit
    def model(df, params):
        """Two-layer model with nested calls."""
        h = layer1(df, params['w1'])  # First layer
        output = layer2(h, params['w2'])  # Second layer
        return output

    def loss(df, params, target):
        """Loss function."""
        predictions = model(df, params)
        return ((predictions - target) ** 2).mean()

    # Setup
    df = DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]})
    params = {
        'w1': jnp.array([[0.5, 0.5], [0.5, 0.5]]),
        'w2': jnp.array([[1.0], [1.0]])
    }
    target = jnp.array([[5.0], [7.0]])

    print("\nModel structure:")
    print("  Input (2x2) → layer1 → (2x2) → layer2 → (2x1) output")

    # Compute gradients through all nested calls
    grad_fn = jax.grad(loss, argnums=1)  # Gradient w.r.t. params
    grads = grad_fn(df, params, target)

    print("\nGradients computed successfully!")
    print(f"Gradient shapes:")
    print(f"  ∂L/∂w1: {grads['w1'].shape}")
    print(f"  ∂L/∂w2: {grads['w2'].shape}")

    print("\n✅ Gradients flow through all nested function calls!")


if __name__ == "__main__":
    example_gpu_detection()
    example_nested_functions()
    example_jit_and_grad_composition()
    example_what_breaks_jit()
    example_separating_jit_and_non_jit()
    example_gpu_performance()
    example_multi_level_nesting()
    example_grad_through_nesting()

    print("\n" + "=" * 60)
    print("All examples completed successfully! 🎉")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  ✅ GPU/TPU support is automatic for JAX arrays")
    print("  ✅ Nested functions compose perfectly")
    print("  ✅ JIT and grad can be composed in any order")
    print("  ✅ Entire function must be traceable (no dynamic shapes)")
    print("  ✅ Separate JIT-able code from I/O and logging")
    print("  ✅ Use where() instead of boolean indexing in JIT")
    print("=" * 60)
