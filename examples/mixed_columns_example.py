"""
Example: Mixed Numeric and Object Columns

Demonstrates how JAXFrame handles DataFrames with both numeric and
non-numeric (object) columns, and what's JIT-compatible vs not.
"""

import jax

from jaxframe import DataFrame


def example_mixed_dataframe():
    """Create and work with mixed-type DataFrames."""
    print("=" * 60)
    print("Example 1: Mixed Numeric and Object Columns")
    print("=" * 60)

    df = DataFrame({
        "price": [10.99, 24.99, 15.49, 32.00],
        "quantity": [5, 3, 10, 2],
        "name": ["Widget", "Gadget", "Tool", "Device"],
        "category": ["A", "B", "A", "C"],
    })

    print("\nDataFrame with mixed types:")
    print(df)
    print(f"\nColumns: {df.columns}")
    print(f"Dtypes: {df.dtypes}")

    # Numeric columns work with JAX
    print(f"\nPrice (numeric): {df['price'].values}")
    # Object columns return numpy arrays
    print(f"Name (object):   {df['name'].values}")


def example_jit_with_mixed():
    """JIT operations only work on numeric columns."""
    print("\n" + "=" * 60)
    print("Example 2: JIT with Mixed DataFrames")
    print("=" * 60)

    df = DataFrame({
        "price": [10.0, 20.0, 30.0],
        "quantity": [2.0, 3.0, 4.0],
        "product": ["A", "B", "C"],
    })
    print("\nDataFrame:")
    print(df)

    # Select numeric columns for JIT
    df_numeric = df[["price", "quantity"]]

    @jax.jit
    def compute_total(df):
        return (df["price"] * df["quantity"]).sum()

    total = compute_total(df_numeric)
    print(f"\nTotal revenue (JIT compiled): {total}")

    # Object columns pass through unchanged in JIT
    @jax.jit
    def double_prices(df):
        return df * 2

    # This works: object columns are aux_data, not traced
    doubled = double_prices(df_numeric)
    print(f"Doubled prices: {doubled['price'].values}")


def example_groupby_with_objects():
    """GroupBy using object columns."""
    print("\n" + "=" * 60)
    print("Example 3: GroupBy with Object Keys")
    print("=" * 60)

    df = DataFrame({
        "price": [10.0, 20.0, 30.0, 15.0, 25.0],
        "quantity": [1.0, 2.0, 3.0, 4.0, 5.0],
        "category": [0.0, 1.0, 0.0, 1.0, 0.0],  # encoded as numeric
    })
    print("\nDataFrame:")
    print(df)

    # GroupBy with numeric key (JIT-compatible)
    result = df.groupby("category")["price"].mean()
    print(f"\nMean price by category:\n{result}")


if __name__ == "__main__":
    example_mixed_dataframe()
    example_jit_with_mixed()
    example_groupby_with_objects()
    print("\nAll mixed-column examples completed!")
    print("\nKey Takeaways:")
    print("  - Numeric columns: Full JAX support (JIT, grad)")
    print("  - Object columns: Supported but not JIT-traceable")
    print("  - Use select_dtypes or column selection for JIT contexts")
