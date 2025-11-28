"""
Example: Mixed Numeric and Object Columns

This example demonstrates how JAXFrame handles DataFrames with both
numeric and non-numeric (object) columns, and how to work with them
in JIT-compiled and differentiable contexts.
"""

import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '..')

from jaxframe import DataFrame


def example_mixed_dataframe():
    """Create and work with mixed-type DataFrames."""
    print("=" * 60)
    print("Example 1: Mixed Numeric and Object Columns")
    print("=" * 60)

    # Create DataFrame with mixed types
    df = DataFrame({
        'product_id': [1.0, 2.0, 3.0, 4.0],      # Numeric
        'price': [10.99, 24.99, 15.49, 32.00],   # Numeric
        'quantity': [5.0, 3.0, 10.0, 2.0],       # Numeric
        'name': ['Widget', 'Gadget', 'Tool', 'Device'],  # Object (string)
        'category': ['A', 'B', 'A', 'C'],         # Object (string)
    })

    print("\nDataFrame with mixed types:")
    print(df)
    print(f"\nNumeric columns: {df._numeric_cols}")
    print(f"Object columns: {list(df._object_data.keys())}")

    # Accessing numeric columns
    print("\n--- Accessing Columns ---")
    print(f"Price column (numeric): {df['price'].values}")
    print(f"Name column (object): {df['name'].values}")

    print("\n✅ Mixed DataFrame created successfully!")


def example_numeric_operations_only():
    """Perform JIT operations on numeric columns only."""
    print("\n" + "=" * 60)
    print("Example 2: JIT Operations on Numeric Columns")
    print("=" * 60)

    df = DataFrame({
        'price': [10.0, 20.0, 30.0],
        'quantity': [2.0, 3.0, 4.0],
        'product': ['A', 'B', 'C'],  # Object column
    })

    print("\nOriginal DataFrame:")
    print(df)

    # Select only numeric columns for JIT operations
    df_numeric = df[['price', 'quantity']]

    print("\nNumeric-only DataFrame:")
    print(df_numeric)

    @jax.jit
    def compute_total(df_numeric):
        """JIT-compiled computation on numeric data."""
        return (df_numeric._numeric_data[:, 0] * df_numeric._numeric_data[:, 1]).sum()

    total = compute_total(df_numeric)
    print(f"\nTotal revenue (JIT compiled): {total}")
    print("\n✅ JIT works on numeric-only subset!")


def example_object_columns_limitations():
    """Demonstrate limitations with object columns in JIT."""
    print("\n" + "=" * 60)
    print("Example 3: Object Column Limitations")
    print("=" * 60)

    df = DataFrame({
        'value': [10.0, 20.0, 30.0],
        'label': ['low', 'medium', 'high'],
    })

    print("\nDataFrame with object column:")
    print(df)

    # Object columns are NOT part of JAX pytree
    print("\n--- Pytree Flattening ---")
    from jaxframe.dataframe import _dataframe_flatten
    children, aux = _dataframe_flatten(df)
    print(f"Children (participates in JAX ops): {len(children)} arrays")
    print(f"Aux data (metadata): {aux['numeric_cols']} (numeric), {list(aux['object_data'].keys())} (object)")

    # Object columns pass through transformations unchanged
    @jax.jit
    def double_values(df):
        """Doubles numeric values, object columns unchanged."""
        return df * 2

    df_doubled = double_values(df[['value']])  # Only numeric column
    print("\n--- After JIT doubling (numeric only) ---")
    print(df_doubled)

    # Accessing object data (not JIT-able)
    print("\n--- Object Column Access (non-JIT) ---")
    labels = df['label'].values
    print(f"Labels: {labels}")

    print("\n✅ Object columns are supported but not JIT-able!")


def example_categorical_encoding_concept():
    """
    Demonstrate how categorical data could be encoded for JIT.

    This shows the CONCEPT - full implementation would be in the library.
    """
    print("\n" + "=" * 60)
    print("Example 4: Categorical Encoding Concept")
    print("=" * 60)

    # Original data with categories
    categories = ['small', 'medium', 'large', 'small', 'large']
    values = [10.0, 20.0, 30.0, 15.0, 35.0]

    print("\nOriginal data:")
    print(f"Categories: {categories}")
    print(f"Values: {values}")

    # Manual encoding (this would be automated in the library)
    category_map = {'small': 0, 'medium': 1, 'large': 2}
    encoded_categories = [category_map[c] for c in categories]

    print("\n--- After Encoding ---")
    print(f"Category mapping: {category_map}")
    print(f"Encoded: {encoded_categories}")

    # Create DataFrame with encoded categories
    df = DataFrame({
        'category_code': jnp.array(encoded_categories, dtype=jnp.float64),
        'value': jnp.array(values),
    })

    print("\nDataFrame (all numeric, JIT-able):")
    print(df)

    # Now we can use JIT on categorical operations
    @jax.jit
    def compute_by_category(df):
        """Compute mean value per category."""
        # This is simplified - real implementation would use segment operations
        codes = df._numeric_data[:, 0]
        vals = df._numeric_data[:, 1]
        return vals.mean()  # Simplified for demo

    result = compute_by_category(df)
    print(f"\nJIT-compiled result: {result}")

    print("\n✅ Categorical encoding enables JIT on categorical data!")
    print("💡 Full categorical support would include:")
    print("   - Automatic encoding/decoding")
    print("   - Segment operations for groupby")
    print("   - Preserving category semantics")


def example_workflow_recommendation():
    """Show recommended workflow for mixed-type DataFrames."""
    print("\n" + "=" * 60)
    print("Example 5: Recommended Workflow")
    print("=" * 60)

    # Full DataFrame with mixed types
    df_full = DataFrame({
        'price': [10.0, 20.0, 30.0, 40.0],
        'quantity': [1.0, 2.0, 3.0, 4.0],
        'discount': [0.1, 0.15, 0.2, 0.1],
        'product': ['A', 'B', 'C', 'D'],
        'region': ['North', 'South', 'North', 'West'],
    })

    print("\nFull DataFrame:")
    print(df_full)

    # Step 1: Separate concerns
    print("\n--- Step 1: Separate Numeric and Object Data ---")
    numeric_cols = ['price', 'quantity', 'discount']
    df_numeric = df_full[numeric_cols]

    print(f"Numeric columns for computation: {numeric_cols}")

    # Step 2: JIT-compiled numeric operations
    print("\n--- Step 2: JIT-Compiled Computations ---")

    @jax.jit
    def compute_revenue(df):
        """Fast revenue calculation."""
        price = df._numeric_data[:, 0]
        quantity = df._numeric_data[:, 1]
        discount = df._numeric_data[:, 2]
        return (price * quantity * (1 - discount)).sum()

    revenue = compute_revenue(df_numeric)
    print(f"Total revenue (JIT): {revenue:.2f}")

    # Step 3: Use object columns for grouping/filtering (non-JIT)
    print("\n--- Step 3: Object Columns for Metadata ---")
    products = df_full['product'].values
    regions = df_full['region'].values

    print(f"Products: {products}")
    print(f"Regions: {regions}")

    # Step 4: Combine results
    print("\n--- Step 4: Combine Results ---")
    print("Recommended pattern:")
    print("  1. Use numeric DataFrame subset for JIT/grad operations")
    print("  2. Use object columns for grouping, labeling, filtering")
    print("  3. Combine results as needed outside JIT context")

    print("\n✅ This workflow maximizes performance while supporting rich data types!")


if __name__ == "__main__":
    example_mixed_dataframe()
    example_numeric_operations_only()
    example_object_columns_limitations()
    example_categorical_encoding_concept()
    example_workflow_recommendation()

    print("\n" + "=" * 60)
    print("All mixed-column examples completed successfully! 🎉")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  ✅ Numeric columns: Full JAX support (JIT, grad, vmap)")
    print("  ✅ Object columns: Supported but not JIT-able")
    print("  ✅ Use numeric subset for performance-critical operations")
    print("  ✅ Consider categorical encoding for string categories")
    print("=" * 60)
