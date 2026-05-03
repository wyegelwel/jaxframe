"""
Example: Automatic Differentiation with JAXFrame

Demonstrates using jax.grad with DataFrames for machine learning
and optimization tasks.
"""

import jax
import jax.numpy as jnp

from jaxframe import DataFrame


def example_basic_gradient():
    """Basic gradient computation with DataFrames."""
    print("=" * 60)
    print("Example 1: Basic Gradient Computation")
    print("=" * 60)

    df = DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    print("\nOriginal DataFrame:")
    print(df)

    # Loss = sum of squares of all values
    def loss_fn(df):
        return (df["x"] ** 2 + df["y"] ** 2).sum()

    grad_fn = jax.grad(loss_fn)
    gradients = grad_fn(df)

    print("\nGradients (dL/dx = 2x, dL/dy = 2y):")
    print(gradients)


def example_prediction_gradient():
    """Gradients for a prediction task."""
    print("\n" + "=" * 60)
    print("Example 2: Prediction Task Gradients")
    print("=" * 60)

    df = DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0],
        "feature2": [2.0, 3.0, 4.0, 5.0],
        "target": [3.0, 5.0, 7.0, 9.0],
    })
    print("\nTraining DataFrame:")
    print(df)

    # Differentiate w.r.t. separate weight parameters
    def mse_loss(weights, df):
        predictions = df["feature1"] * weights[0] + df["feature2"] * weights[1] + weights[2]
        return ((predictions - df["target"]) ** 2).mean()

    weights = jnp.array([1.0, 1.0, 0.0])
    grad_fn = jax.grad(mse_loss)
    gradients = grad_fn(weights, df)

    print(f"\nLoss: {mse_loss(weights, df):.4f}")
    print(f"Gradients w.r.t. [w1, w2, bias]: {gradients}")

    # One step of gradient descent
    lr = 0.1
    new_weights = weights - lr * gradients
    print(f"New loss: {mse_loss(new_weights, df):.4f}")


def example_dataframe_gradient():
    """Gradient w.r.t. DataFrame itself (for feature learning)."""
    print("\n" + "=" * 60)
    print("Example 3: Gradient w.r.t. DataFrame (Feature Learning)")
    print("=" * 60)

    df = DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [1.0, 1.0, 1.0]})
    targets = jnp.array([5.0, 10.0, 15.0])

    print("\nInitial DataFrame:")
    print(df)

    def feature_loss(df):
        predictions = df["feature1"] * 2 + df["feature2"] * 3
        return ((predictions - targets) ** 2).sum()

    # Gradient w.r.t. the entire DataFrame
    grad_fn = jax.grad(feature_loss)
    feature_gradients = grad_fn(df)

    print("\nFeature gradients:")
    print(feature_gradients)
    print(f"\nInitial loss: {feature_loss(df):.4f}")


def example_optimization_loop():
    """Full optimization loop: learning optimal prices."""
    print("\n" + "=" * 60)
    print("Example 4: Optimization Loop - Learning Optimal Prices")
    print("=" * 60)

    # demand = base_demand - elasticity * price
    # revenue = price * demand
    base_demand = jnp.array([100.0, 150.0, 200.0])
    elasticity = jnp.array([2.0, 1.5, 3.0])

    def revenue(prices):
        demand = base_demand - elasticity * prices
        demand = jnp.maximum(demand, 0)
        return (prices * demand).sum()

    # Maximize revenue = minimize -revenue
    grad_fn = jax.grad(lambda p: -revenue(p))

    prices = jnp.array([10.0, 15.0, 12.0])
    lr = 0.1

    print(f"{'Iter':>4}  {'Revenue':>10}  Prices")
    print("-" * 50)
    for i in range(10):
        if i % 2 == 0:
            print(f"{i:4d}  {revenue(prices):10.2f}  {prices}")
        prices = prices - lr * grad_fn(prices)

    print(f"{10:4d}  {revenue(prices):10.2f}  {prices}")


if __name__ == "__main__":
    example_basic_gradient()
    example_prediction_gradient()
    example_dataframe_gradient()
    example_optimization_loop()
    print("\nAll gradient examples completed successfully!")
