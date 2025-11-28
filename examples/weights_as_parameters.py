"""
Example: Using DataFrames with Trainable Parameters

This demonstrates how to use JAXFrame DataFrames with trainable weights,
showing that weights can (and should!) be passed as parameters.
"""

import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '..')

from jaxframe import DataFrame


def example_weights_as_parameters():
    """Show that weights can be passed as parameters."""
    print("=" * 60)
    print("Example 1: Weights as Parameters")
    print("=" * 60)

    # Training data
    df = DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [2.0, 3.0, 4.0, 5.0],
        'target': [5.0, 11.0, 17.0, 23.0],  # target = 2*f1 + 3*f2 - 1
    })

    print("\nTraining data:")
    print(df)

    # Model with weights as parameter
    def model(df, weights):
        """Linear model: prediction = w1*f1 + w2*f2 + b"""
        features = df._numeric_data[:, :2]  # First 2 columns
        prediction = features @ weights['w'] + weights['b']
        return prediction

    def loss(df, weights):
        """Mean squared error."""
        predictions = model(df, weights)
        targets = df._numeric_data[:, 2]  # Target column
        return ((predictions - targets) ** 2).mean()

    # Initial weights
    weights = {
        'w': jnp.array([1.0, 1.0]),
        'b': 0.0
    }

    print(f"\nInitial loss: {loss(df, weights):.4f}")

    # Get gradients w.r.t. weights (NOT df)
    grad_fn = jax.grad(loss, argnums=1)  # argnums=1 → second argument (weights)

    # Training loop
    learning_rate = 0.1
    for step in range(20):
        grads = grad_fn(df, weights)

        # Update weights
        weights = {
            'w': weights['w'] - learning_rate * grads['w'],
            'b': weights['b'] - learning_rate * grads['b']
        }

        if step % 5 == 0:
            print(f"Step {step}: loss={loss(df, weights):.4f}, w={weights['w']}, b={weights['b']:.2f}")

    print(f"\nFinal weights: w={weights['w']}, b={weights['b']:.2f}")
    print("✅ Weights as parameters work perfectly!")


def example_gradients_wrt_both():
    """Get gradients w.r.t. both DataFrame AND weights."""
    print("\n" + "=" * 60)
    print("Example 2: Gradients w.r.t. Both DataFrame and Weights")
    print("=" * 60)

    df = DataFrame({
        'x': [1.0, 2.0, 3.0],
        'y': [2.0, 3.0, 4.0],
    })

    weights = jnp.array([0.5, 1.5])

    def model(df, weights):
        """Simple weighted sum."""
        features = df._numeric_data
        return (features * weights).sum()

    print("\nDataFrame:")
    print(df)
    print(f"\nWeights: {weights}")
    print(f"\nModel output: {model(df, weights):.2f}")

    # Gradients w.r.t. BOTH arguments
    grad_fn = jax.grad(model, argnums=(0, 1))
    df_grads, weight_grads = grad_fn(df, weights)

    print("\n--- Gradients ---")
    print("w.r.t. DataFrame:")
    print(df_grads)
    print(f"\nw.r.t. Weights: {weight_grads}")

    print("\n✅ Can get gradients w.r.t. both!")


def example_pytree_composition():
    """Multiple DataFrames and weights compose via pytrees."""
    print("\n" + "=" * 60)
    print("Example 3: Pytree Composition (Multiple DataFrames)")
    print("=" * 60)

    # Multiple DataFrames as inputs
    df_features = DataFrame({
        'x1': [1.0, 2.0],
        'x2': [3.0, 4.0],
    })

    df_targets = DataFrame({
        'y': [7.0, 14.0],  # y = 2*x1 + 1*x2 + 1
    })

    weights = {'w': jnp.array([1.0, 1.0]), 'b': 0.0}

    print("\nFeatures:")
    print(df_features)
    print("\nTargets:")
    print(df_targets)

    def loss(df_features, df_targets, weights):
        """Loss function with multiple DataFrames."""
        predictions = df_features._numeric_data @ weights['w'] + weights['b']
        targets = df_targets._numeric_data.squeeze()
        return ((predictions - targets) ** 2).mean()

    print(f"\nInitial loss: {loss(df_features, df_targets, weights):.4f}")

    # Gradient only w.r.t. weights (most common case)
    grad_fn = jax.grad(loss, argnums=2)  # argnums=2 → third argument (weights)
    weight_grads = grad_fn(df_features, df_targets, weights)

    print(f"\nGradients w.r.t. weights:")
    print(f"  ∂L/∂w: {weight_grads['w']}")
    print(f"  ∂L/∂b: {weight_grads['b']:.4f}")

    # But you COULD get gradients w.r.t. features too (for adversarial examples!)
    grad_fn_features = jax.grad(loss, argnums=0)
    feature_grads = grad_fn_features(df_features, df_targets, weights)

    print(f"\nGradients w.r.t. features (for adversarial examples):")
    print(feature_grads)

    print("\n✅ Multiple DataFrames compose perfectly via pytrees!")


def example_jit_with_weights():
    """JIT works with weights as parameters too."""
    print("\n" + "=" * 60)
    print("Example 4: JIT Compilation with Weights as Parameters")
    print("=" * 60)

    df = DataFrame({
        'x': jnp.arange(1000, dtype=jnp.float32),
        'y': jnp.arange(1000, 2000, dtype=jnp.float32),
    })

    @jax.jit
    def compute(df, weights):
        """JIT-compiled function with DataFrame and weights."""
        return (df._numeric_data @ weights).sum()

    weights = jnp.array([0.5, 1.5])

    # First call: compiles
    print("\nFirst call (compiles)...")
    result1 = compute(df, weights)
    print(f"Result: {result1}")

    # Second call: instant!
    print("\nSecond call (instant)...")
    result2 = compute(df, weights * 2)
    print(f"Result: {result2}")

    print("\n✅ JIT works with weights as parameters!")


def example_best_practice():
    """Show the recommended pattern."""
    print("\n" + "=" * 60)
    print("Example 5: Recommended Pattern")
    print("=" * 60)

    print("""
Recommended pattern for training:

def loss_fn(params, batch):
    '''
    params: trainable parameters (weights, biases, etc.)
    batch: DataFrame with features and targets
    '''
    features = batch[['x', 'y', 'z']]
    targets = batch['target']

    predictions = model(features, params)
    return mse(predictions, targets)

# Get gradients w.r.t. params
grad_fn = jax.grad(loss_fn, argnums=0)

# Training loop
for epoch in range(num_epochs):
    for batch_df in data_loader:
        grads = grad_fn(params, batch_df)
        params = optimizer.update(params, grads)

Why this pattern?
✅ Clear separation: params (trainable) vs data (fixed)
✅ Standard ML convention (matches PyTorch, TensorFlow)
✅ Easy to swap optimizers (only params change)
✅ Can still get gradients w.r.t. data if needed (adversarial examples)
""")

    print("✅ This is the recommended pattern!")


if __name__ == "__main__":
    example_weights_as_parameters()
    example_gradients_wrt_both()
    example_pytree_composition()
    example_jit_with_weights()
    example_best_practice()

    print("\n" + "=" * 60)
    print("All examples completed successfully! 🎉")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  ✅ Weights CAN and SHOULD be passed as parameters")
    print("  ✅ Use argnums to specify which arguments to differentiate")
    print("  ✅ Multiple DataFrames compose via pytrees")
    print("  ✅ JIT works with weights as parameters")
    print("  ✅ Follow standard ML pattern: loss_fn(params, data)")
    print("=" * 60)
