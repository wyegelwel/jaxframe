"""
Example: Device Management in JAXFrame

Demonstrates how to:
1. Set the default device
2. Transfer DataFrames between devices
3. Check current device
4. Use convenience methods (to_gpu, to_cpu, to_tpu)
"""

import sys
sys.path.insert(0, '..')

import jax
import jax.numpy as jnp
from jaxframe import DataFrame


def example_check_device():
    """Check which device a DataFrame is on."""
    print("=" * 60)
    print("Example 1: Check Current Device")
    print("=" * 60)

    # Create DataFrame on default device
    df = DataFrame({'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]})

    print(f"\nAvailable devices: {jax.devices()}")
    print(f"DataFrame device: {df.device}")
    print(f"Device type: {df.device.platform}")

    print("\n✅ Can easily check which device data is on!")


def example_set_default_device():
    """Set the default device globally."""
    print("\n" + "=" * 60)
    print("Example 2: Set Default Device")
    print("=" * 60)

    # Check current default
    print(f"\nCurrent default device: {jax.devices()[0]}")

    # Try to set GPU as default (if available)
    try:
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            jax.config.update('jax_default_device', gpu_devices[0])
            print(f"Set default to: {jax.devices()[0]}")

            # New DataFrames now go to GPU
            df = DataFrame({'x': [1, 2, 3]})
            print(f"New DataFrame device: {df.device}")

            # Reset to CPU
            jax.config.update('jax_default_device', jax.devices('cpu')[0])
            print(f"\nReset default to: {jax.devices()[0]}")
        else:
            print("\nNo GPU available, using CPU")
    except:
        print("\nNo GPU available, using CPU")

    print("\n✅ Can set default device globally!")


def example_to_device_methods():
    """Use device transfer methods."""
    print("\n" + "=" * 60)
    print("Example 3: Device Transfer Methods")
    print("=" * 60)

    # Create on CPU
    df = DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4.0, 5.0, 6.0],
    })

    print(f"\nOriginal device: {df.device}")

    # Method 1: to_device with string
    try:
        df_gpu = df.to_device('gpu')
        print(f"After to_device('gpu'): {df_gpu.device}")
    except RuntimeError:
        print("No GPU available, staying on CPU")
        df_gpu = df

    # Method 2: Convenience method to_gpu()
    try:
        df_gpu = df.to_gpu()
        print(f"After to_gpu(): {df_gpu.device}")
    except RuntimeError:
        print("No GPU available")

    # Method 3: Back to CPU
    df_cpu = df_gpu.to_cpu()
    print(f"After to_cpu(): {df_cpu.device}")

    # Original DataFrame unchanged
    print(f"\nOriginal still on: {df.device}")

    print("\n✅ Device transfer creates new DataFrame!")


def example_operations_on_device():
    """Operations run on the DataFrame's device."""
    print("\n" + "=" * 60)
    print("Example 4: Operations Run on DataFrame's Device")
    print("=" * 60)

    df = DataFrame({'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]})

    print(f"\nDataFrame device: {df.device}")

    # Operations run on same device
    result = df * 2
    print(f"After df * 2, result device: {result.device}")

    aggregated = df.sum()
    print(f"After df.sum(), result device: {aggregated.values.device}")

    # Transfer to different device
    try:
        df_gpu = df.to_gpu()
        result_gpu = df_gpu * 2
        print(f"\nOn GPU - df device: {df_gpu.device}")
        print(f"On GPU - result device: {result_gpu.device}")
    except RuntimeError:
        print("\nNo GPU available for this demo")

    print("\n✅ Operations preserve device placement!")


def example_clean_syntax():
    """Show clean syntax for device placement."""
    print("\n" + "=" * 60)
    print("Example 5: Clean Syntax")
    print("=" * 60)

    print("\nOLD WAY (manual device_put at construction):")
    print("""
df_gpu = jf.DataFrame({
    'x': jax.device_put(jnp.array([1, 2, 3]), jax.devices('gpu')[0]),
    'y': jax.device_put(jnp.array([4, 5, 6]), jax.devices('gpu')[0])
})
    """)

    print("\nNEW WAY (clean and readable):")
    print("""
df = jf.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
df_gpu = df.to_gpu()  # Much cleaner!
    """)

    # Demonstrate the new way
    df = DataFrame({'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]})
    print(f"\nCreated on: {df.device}")

    try:
        df_gpu = df.to_gpu()
        print(f"Transferred to: {df_gpu.device}")
    except RuntimeError:
        print("GPU not available, but syntax is still clean!")

    print("\n✅ Much cleaner API!")


def example_mixed_device_operations():
    """Operations with DataFrames on different devices."""
    print("\n" + "=" * 60)
    print("Example 6: Mixed Device Operations")
    print("=" * 60)

    df1 = DataFrame({'x': [1.0, 2.0, 3.0]})
    df2 = DataFrame({'x': [4.0, 5.0, 6.0]})

    print(f"\ndf1 device: {df1.device}")
    print(f"df2 device: {df2.device}")

    # Same device - works fine
    result = df1 + df2
    print(f"\ndf1 + df2 works (same device)")
    print(f"Result device: {result.device}")

    # Different devices - JAX will handle automatically
    try:
        df2_gpu = df2.to_gpu()
        print(f"\ndf2 moved to: {df2_gpu.device}")

        # JAX automatically moves data to compute
        result = df1 + df2_gpu
        print(f"df1 (CPU) + df2_gpu (GPU) = result on {result.device}")
    except RuntimeError:
        print("\nNo GPU to demo cross-device operations")

    print("\n✅ JAX handles cross-device operations automatically!")


def example_device_with_jit():
    """Device placement with JIT compilation."""
    print("\n" + "=" * 60)
    print("Example 7: Device Placement with JIT")
    print("=" * 60)

    df = DataFrame({'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]})

    @jax.jit
    def compute(df):
        """JIT-compiled function."""
        return (df * 2).sum(axis=None)

    # JIT works regardless of device
    result_cpu = compute(df)
    print(f"\nCPU result: {result_cpu}")

    try:
        df_gpu = df.to_gpu()
        result_gpu = compute(df_gpu)
        print(f"GPU result: {result_gpu}")
        print(f"GPU result device: {result_gpu.device}")
    except RuntimeError:
        print("No GPU available")

    print("\n✅ JIT works on any device!")


def example_practical_workflow():
    """Practical workflow: Load on CPU, compute on GPU."""
    print("\n" + "=" * 60)
    print("Example 8: Practical Workflow")
    print("=" * 60)

    print("\nTypical workflow:")
    print("1. Load data on CPU (I/O)")
    print("2. Transfer to GPU for computation")
    print("3. Transfer results back to CPU for saving")

    # Step 1: Load data (simulated)
    print("\nStep 1: Load data on CPU")
    df = DataFrame({
        'feature1': jnp.arange(100),
        'feature2': jnp.arange(100, 200),
    })
    print(f"  Loaded on: {df.device}")

    # Step 2: Transfer to GPU for computation
    print("\nStep 2: Transfer to GPU for fast computation")
    try:
        df_gpu = df.to_gpu()
        print(f"  Transferred to: {df_gpu.device}")

        # Fast GPU computation
        @jax.jit
        def process(df):
            return ((df - df.mean(axis=None)) * 2).sum(axis=None)

        result = process(df_gpu)
        print(f"  Computed result: {result}")
    except RuntimeError:
        print("  No GPU available, using CPU")
        df_gpu = df
        result = ((df - df.mean(axis=None)) * 2).sum(axis=None)

    # Step 3: Transfer back to CPU for I/O
    print("\nStep 3: Transfer back to CPU for saving")
    df_cpu = df_gpu.to_cpu()
    print(f"  Back on: {df_cpu.device}")
    print("  (Now can save to file)")

    print("\n✅ Complete GPU workflow!")


if __name__ == "__main__":
    example_check_device()
    example_set_default_device()
    example_to_device_methods()
    example_operations_on_device()
    example_clean_syntax()
    example_mixed_device_operations()
    example_device_with_jit()
    example_practical_workflow()

    print("\n" + "=" * 60)
    print("All device management examples completed! 🎉")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  ✅ Check device with df.device property")
    print("  ✅ Transfer with df.to_gpu(), df.to_cpu(), df.to_tpu()")
    print("  ✅ Set default device with jax.config.update()")
    print("  ✅ Device transfer returns new DataFrame (immutable)")
    print("  ✅ Operations preserve device placement")
    print("  ✅ JIT works on any device")
    print("  ✅ Clean syntax: df.to_gpu() instead of manual device_put")
    print("=" * 60)
