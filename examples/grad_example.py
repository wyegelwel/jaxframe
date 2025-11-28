"""
Example: Automatic Differentiation with JAXFrame

This example demonstrates how to use JAX's automatic differentiation
with DataFrames for machine learning and optimization tasks.
"""

import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '..')

from jaxframe import DataFrame


def example_basic_gradient():
    """Basic gradient computation with DataFrames."""
    print("=" * 60)
    print("Example 1: Basic Gradient Computation")
    print("=" * 60)

    # Create a DataFrame
    df = DataFrame({
        'x': [1.0, 2.0, 3.0],
        'y': [2.0, 4.0, 6.0],
    })

    print("\nOriginal DataFrame:")
    print(df)

    # Define a loss function
    def loss_fn(df):
        """Simple quadratic loss: sum of squares."""
        return (df._numeric_data ** 2).sum()

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    gradients = grad_fn(df)

    print("\nGradients (∂L/∂df):")
    print(f"Shape: {gradients._numeric_data.shape}")
    print(gradients)
    print("\n✅ Gradient computation successful!")


def example_prediction_gradient():
    """Gradients for a prediction task."""
    print("\n" + "=" * 60)
    print("Example 2: Prediction Task Gradients")
    print("=" * 60)

    # Training data
    df = DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [2.0, 3.0, 4.0, 5.0],
        'target': [3.0, 5.0, 7.0, 9.0],
    })

    print("\nTraining DataFrame:")
    print(df)

    # Model parameters (separate from DataFrame for clarity)
    params = {'w1': 1.0, 'w2': 1.0, 'b': 0.0}

    def predict(df, params):
        """Linear prediction model."""
        f1 = df._numeric_data[:, 0]  # feature1
        f2 = df._numeric_data[:, 1]  # feature2
        return params['w1'] * f1 + params['w2'] * f2 + params['b']

    def mse_loss(params, df):
        """Mean squared error loss."""
        predictions = predict(df, params)
        targets = df._numeric_data[:, 2]  # target column
        return ((predictions - targets) ** 2).mean()

    # Compute gradient with respect to parameters
    grad_fn = jax.grad(mse_loss)
    gradients = grad_fn(params, df)

    print(f"\nLoss: {mse_loss(params, df):.4f}")
    print("\nGradients w.r.t. parameters:")
    for key, grad in gradients.items():
        print(f"  ∂L/∂{key}: {grad:.4f}")

    # Update parameters (simple gradient descent)
    learning_rate = 0.1
    new_params = {k: v - learning_rate * gradients[k] for k, v in params.items()}

    print("\nUpdated parameters:")
    for key, val in new_params.items():
        print(f"  {key}: {params[key]:.4f} -> {val:.4f}")

    print(f"\nNew loss: {mse_loss(new_params, df):.4f}")
    print("✅ Parameter optimization successful!")


def example_dataframe_gradient():
    """Gradient with respect to DataFrame itself (for feature learning)."""
    print("\n" + "=" * 60)
    print("Example 3: Gradient w.r.t. DataFrame (Feature Learning)")
    print("=" * 60)

    # Initial features (to be optimized)
    df = DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [1.0, 1.0, 1.0],
    })

    print("\nInitial DataFrame:")
    print(df)

    # Target values
    targets = jnp.array([5.0, 10.0, 15.0])

    def feature_loss(df):
        """
        Loss function where we want to learn optimal feature values.

        This demonstrates differentiating through DataFrame operations.
        """
        # Simple model: weighted sum of features
        predictions = df._numeric_data[:, 0] * 2 + df._numeric_data[:, 1] * 3
        return ((predictions - targets) ** 2).sum()

    # Compute gradient w.r.t. DataFrame
    grad_fn = jax.grad(feature_loss)
    feature_gradients = grad_fn(df)

    print("\nFeature gradients:")
    print(feature_gradients)

    # Update features
    learning_rate = 0.01
    updated_df = DataFrame._from_parts(
        numeric_data=df._numeric_data - learning_rate * feature_gradients._numeric_data,
        numeric_cols=df._numeric_cols,
        numeric_dtypes=df._numeric_dtypes,
        object_data=df._object_data,
        index=df._index,
        column_order=df._column_order,
    )

    print(f"\nInitial loss: {feature_loss(df):.4f}")
    print(f"Updated loss: {feature_loss(updated_df):.4f}")
    print("✅ Feature learning successful!")


def example_jacobian():
    """Computing Jacobians for sensitivity analysis."""
    print("\n" + "=" * 60)
    print("Example 4: Jacobian for Sensitivity Analysis")
    print("=" * 60)

    df = DataFrame({
        'price': [10.0, 20.0, 30.0],
        'quantity': [1.0, 2.0, 3.0],
    })

    print("\nDataFrame:")
    print(df)

    def revenue_per_item(df):
        """Calculate revenue for each item."""
        return df._numeric_data[:, 0] * df._numeric_data[:, 1]  # price * quantity

    # Compute Jacobian: how each output depends on each input
    jacobian_fn = jax.jacobian(revenue_per_item)
    jac = jacobian_fn(df)

    print("\nJacobian (∂revenue/∂inputs):")
    print(f"Shape: {jac.shape}")
    print(jac)
    print("\nInterpretation:")
    print("  - jac[i, i, 0] = ∂revenue_i/∂price_i (= quantity_i)")
    print("  - jac[i, i, 1] = ∂revenue_i/∂quantity_i (= price_i)")
    print("\n✅ Jacobian computation successful!")


def example_optimization_loop():
    """Full optimization loop: learning optimal prices."""
    print("\n" + "=" * 60)
    print("Example 5: Optimization Loop - Learning Optimal Prices")
    print("=" * 60)

    # Scenario: Find optimal prices to maximize revenue with demand constraints
    # demand = base_demand - elasticity * price

    df = DataFrame({
        'base_demand': [100.0, 150.0, 200.0],
        'elasticity': [2.0, 1.5, 3.0],
        'price': [10.0, 15.0, 12.0],  # Initial prices (to be optimized)
    })

    print("\nInitial state:")
    print(df)

    def revenue(df):
        """
        Revenue with price-dependent demand.

        demand = base_demand - elasticity * price
        revenue = price * demand
        """
        price = df._numeric_data[:, 2]
        base_demand = df._numeric_data[:, 0]
        elasticity = df._numeric_data[:, 1]

        demand = base_demand - elasticity * price
        # Add constraint: demand must be non-negative
        demand = jnp.maximum(demand, 0)

        return (price * demand).sum()

    # We want to maximize revenue, so minimize negative revenue
    def loss(df):
        return -revenue(df)

    grad_fn = jax.grad(loss)

    # Optimization loop
    print("\nOptimization progress:")
    print(f"{'Iter':>4} {'Revenue':>10} {'Prices':>30}")
    print("-" * 46)

    current_df = df
    learning_rate = 0.1

    for i in range(10):
        current_revenue = revenue(current_df)
        prices = current_df._numeric_data[:, 2]

        if i % 2 == 0:
            print(f"{i:4d} {current_revenue:10.2f} {str(prices):>30}")

        # Compute gradient and update
        grads = grad_fn(current_df)

        # Only update price column (column index 2)
        new_numeric = current_df._numeric_data.at[:, 2].add(-learning_rate * grads._numeric_data[:, 2])

        current_df = DataFrame._from_parts(
            numeric_data=new_numeric,
            numeric_cols=current_df._numeric_cols,
            numeric_dtypes=current_df._numeric_dtypes,
            object_data=current_df._object_data,
            index=current_df._index,
            column_order=current_df._column_order,
        )

    final_revenue = revenue(current_df)
    print(f"{10:4d} {final_revenue:10.2f} {str(current_df._numeric_data[:, 2]):>30}")

    print("\n✅ Optimization complete!")
    print(f"Revenue improvement: {df._numeric_data[:, 2].sum()} -> {current_df._numeric_data[:, 2].sum()}")


if __name__ == "__main__":
    example_basic_gradient()
    example_prediction_gradient()
    example_dataframe_gradient()
    example_jacobian()
    example_optimization_loop()

    print("\n" + "=" * 60)
    print("All gradient examples completed successfully! 🎉")
    print("=" * 60)
