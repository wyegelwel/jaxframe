"""
Core DataFrame implementation with JAX backend.

This module implements a DataFrame class that:
1. Stores numeric data in JAX arrays for performance
2. Stores non-numeric data in numpy arrays
3. Supports JIT compilation via pytree registration
4. Supports automatic differentiation on numeric columns
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class DataFrame:
    """
    A DataFrame implementation with JAX backend.

    Attributes:
        _numeric_data: JAX array of shape (n_rows, n_numeric_cols) containing all numeric columns
        _numeric_cols: Tuple of column names for numeric data
        _numeric_dtypes: Original dtypes for each numeric column
        _object_data: Dictionary mapping column names to numpy arrays for non-numeric data
        _index: Row index (currently simple integer index)
        _column_order: Original column order (for preserving user's column order)
    """

    _numeric_data: Optional[jnp.ndarray]
    _numeric_cols: Tuple[str, ...]
    _numeric_dtypes: Tuple[Any, ...]
    _object_data: Dict[str, np.ndarray]
    _index: np.ndarray
    _column_order: Tuple[str, ...]

    def __init__(self, data: Union[Dict[str, Any], np.ndarray, jnp.ndarray], index=None):
        """
        Create a DataFrame from a dictionary of columns or array.

        Args:
            data: Dictionary mapping column names to arrays, or a 2D array
            index: Optional row index (defaults to range index)

        Examples:
            >>> df = DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
            >>> df = DataFrame({'x': [1, 2], 'name': ['Alice', 'Bob']})
        """
        if isinstance(data, dict):
            self._init_from_dict(data, index)
        elif isinstance(data, (np.ndarray, jnp.ndarray)):
            self._init_from_array(data, index)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _init_from_dict(self, data: Dict[str, Any], index=None):
        """Initialize from dictionary of columns."""
        if not data:
            # Empty DataFrame
            self._numeric_data = None
            self._numeric_cols = ()
            self._numeric_dtypes = ()
            self._object_data = {}
            self._index = np.array([])
            self._column_order = ()
            return

        # Determine number of rows
        first_col = next(iter(data.values()))
        n_rows = len(first_col) if hasattr(first_col, '__len__') else 1

        # Set index
        if index is None:
            self._index = np.arange(n_rows)
        else:
            self._index = np.asarray(index)

        # Separate numeric and object columns
        numeric_cols_dict = {}
        object_cols_dict = {}

        for col_name, col_data in data.items():
            arr = np.asarray(col_data)

            # Check if numeric
            if np.issubdtype(arr.dtype, np.number):
                numeric_cols_dict[col_name] = arr
            else:
                object_cols_dict[col_name] = arr

        # Store column order
        self._column_order = tuple(data.keys())

        # Create numeric block
        if numeric_cols_dict:
            self._numeric_cols = tuple(numeric_cols_dict.keys())
            arrays = [numeric_cols_dict[col] for col in self._numeric_cols]
            self._numeric_dtypes = tuple(arr.dtype for arr in arrays)

            # Stack into single JAX array
            # Convert all to float64 for now (could be optimized later)
            arrays_float = [arr.astype(np.float64) for arr in arrays]
            self._numeric_data = jnp.stack(arrays_float, axis=1)
        else:
            self._numeric_cols = ()
            self._numeric_dtypes = ()
            self._numeric_data = None

        # Store object columns
        self._object_data = object_cols_dict

    def _init_from_array(self, data: Union[np.ndarray, jnp.ndarray], index=None):
        """Initialize from 2D array (all numeric)."""
        arr = jnp.asarray(data)
        if arr.ndim != 2:
            raise ValueError("Array must be 2-dimensional")

        n_rows, n_cols = arr.shape

        # Generate column names
        self._numeric_cols = tuple(f"col_{i}" for i in range(n_cols))
        self._column_order = self._numeric_cols
        self._numeric_dtypes = tuple(arr.dtype for _ in range(n_cols))
        self._numeric_data = arr
        self._object_data = {}

        # Set index
        if index is None:
            self._index = np.arange(n_rows)
        else:
            self._index = np.asarray(index)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (n_rows, n_cols)."""
        n_rows = len(self._index)
        n_cols = len(self._column_order)
        return (n_rows, n_cols)

    @property
    def columns(self) -> List[str]:
        """Return list of column names."""
        return list(self._column_order)

    @property
    def index(self) -> np.ndarray:
        """Return row index."""
        return self._index

    def __getitem__(self, key: Union[str, List[str]]):
        """
        Get column(s) by name.

        Args:
            key: Column name or list of column names

        Returns:
            Series (single column) or DataFrame (multiple columns)

        Examples:
            >>> df['price']  # Single column -> Series
            >>> df[['price', 'quantity']]  # Multiple columns -> DataFrame
        """
        if isinstance(key, str):
            # Single column
            if key in self._numeric_cols:
                col_idx = self._numeric_cols.index(key)
                return Series(self._numeric_data[:, col_idx], index=self._index, name=key)
            elif key in self._object_data:
                return Series(self._object_data[key], index=self._index, name=key)
            else:
                raise KeyError(f"Column '{key}' not found")

        elif isinstance(key, list):
            # Multiple columns
            new_data = {}
            for col in key:
                if col in self._numeric_cols:
                    col_idx = self._numeric_cols.index(col)
                    new_data[col] = self._numeric_data[:, col_idx]
                elif col in self._object_data:
                    new_data[col] = self._object_data[col]
                else:
                    raise KeyError(f"Column '{col}' not found")
            return DataFrame(new_data, index=self._index)

        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def __repr__(self) -> str:
        """String representation of DataFrame."""
        lines = []
        lines.append(f"DataFrame(shape={self.shape})")

        # Show first few rows
        n_show = min(5, self.shape[0])
        if n_show > 0:
            # Header
            header = "  " + "  ".join(f"{col:>10}" for col in self.columns)
            lines.append(header)

            # Rows
            for i in range(n_show):
                row_data = []
                for col in self.columns:
                    if col in self._numeric_cols:
                        col_idx = self._numeric_cols.index(col)
                        val = self._numeric_data[i, col_idx]
                        # Handle both scalars and arrays
                        try:
                            if hasattr(val, 'shape') and val.shape:
                                # Multi-dimensional value, show shape instead
                                row_data.append(f"{'Array'+ str(val.shape):>10}")
                            else:
                                row_data.append(f"{float(val):>10.2f}")
                        except (TypeError, ValueError):
                            row_data.append(f"{str(val):>10}")
                    else:
                        val = self._object_data[col][i]
                        row_data.append(f"{str(val):>10}")
                lines.append(f"{i}  " + "  ".join(row_data))

            if self.shape[0] > n_show:
                lines.append("  ...")

        return "\n".join(lines)

    # ========================================
    # Numeric operations (JAX-compatible)
    # ========================================

    def __mul__(self, other: Union[float, 'DataFrame']) -> 'DataFrame':
        """Element-wise multiplication (JIT-compatible)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns to multiply")

        if isinstance(other, DataFrame):
            if self._numeric_cols != other._numeric_cols:
                raise ValueError("Column names must match")
            new_numeric = self._numeric_data * other._numeric_data
        else:
            # Handle scalars, JAX arrays, and JAX tracers
            new_numeric = self._numeric_data * other

        return DataFrame._from_parts(
            numeric_data=new_numeric,
            numeric_cols=self._numeric_cols,
            numeric_dtypes=self._numeric_dtypes,
            object_data=self._object_data,
            index=self._index,
            column_order=self._column_order,
        )

    def __add__(self, other: Union[float, 'DataFrame']) -> 'DataFrame':
        """Element-wise addition (JIT-compatible)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns to add")

        if isinstance(other, DataFrame):
            if self._numeric_cols != other._numeric_cols:
                raise ValueError("Column names must match")
            new_numeric = self._numeric_data + other._numeric_data
        else:
            # Handle scalars, JAX arrays, and JAX tracers
            new_numeric = self._numeric_data + other

        return DataFrame._from_parts(
            numeric_data=new_numeric,
            numeric_cols=self._numeric_cols,
            numeric_dtypes=self._numeric_dtypes,
            object_data=self._object_data,
            index=self._index,
            column_order=self._column_order,
        )

    def __sub__(self, other: Union[float, 'DataFrame']) -> 'DataFrame':
        """Element-wise subtraction (JIT-compatible)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns to subtract")

        if isinstance(other, DataFrame):
            if self._numeric_cols != other._numeric_cols:
                raise ValueError("Column names must match")
            new_numeric = self._numeric_data - other._numeric_data
        else:
            # Handle scalars, JAX arrays, and JAX tracers
            new_numeric = self._numeric_data - other

        return DataFrame._from_parts(
            numeric_data=new_numeric,
            numeric_cols=self._numeric_cols,
            numeric_dtypes=self._numeric_dtypes,
            object_data=self._object_data,
            index=self._index,
            column_order=self._column_order,
        )

    def sum(self, axis: Optional[int] = 0):
        """
        Sum along axis (JIT-compatible).

        Args:
            axis: 0 for column-wise sum, 1 for row-wise sum, None for total sum

        Returns:
            Series (axis=0 or 1) or scalar (axis=None)
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to sum")

        result = jnp.sum(self._numeric_data, axis=axis)

        if axis == 0:
            # Column-wise sum -> Series
            return Series(result, index=np.array(self._numeric_cols), name='sum')
        elif axis == 1:
            # Row-wise sum -> Series
            return Series(result, index=self._index, name='sum')
        else:
            # Total sum -> scalar
            return result

    def mean(self, axis: Optional[int] = 0):
        """Mean along axis (JIT-compatible)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute mean")

        result = jnp.mean(self._numeric_data, axis=axis)

        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name='mean')
        elif axis == 1:
            return Series(result, index=self._index, name='mean')
        else:
            return result

    def where(self, condition, fill_value):
        """
        Replace values where condition is False (JIT-compatible).

        This is the JIT-friendly alternative to boolean indexing.

        Args:
            condition: Boolean array or DataFrame
            fill_value: Value to use where condition is False

        Returns:
            DataFrame with same shape

        Examples:
            >>> df.where(df > 10, fill_value=0)  # Replace values <= 10 with 0
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns for where operation")

        if isinstance(condition, DataFrame):
            mask = condition._numeric_data
        else:
            mask = jnp.asarray(condition)

        # Broadcast if needed
        if mask.ndim == 1:
            mask = mask[:, None]

        new_numeric = jnp.where(mask, self._numeric_data, fill_value)

        return DataFrame._from_parts(
            numeric_data=new_numeric,
            numeric_cols=self._numeric_cols,
            numeric_dtypes=self._numeric_dtypes,
            object_data=self._object_data,
            index=self._index,
            column_order=self._column_order,
        )

    # ========================================
    # Comparison operations
    # ========================================

    def __gt__(self, other) -> 'DataFrame':
        """Greater than comparison (JIT-compatible)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compare")

        if isinstance(other, DataFrame):
            new_numeric = self._numeric_data > other._numeric_data
        else:
            # Handle scalars, JAX arrays, and JAX tracers
            new_numeric = self._numeric_data > other

        return DataFrame._from_parts(
            numeric_data=new_numeric,
            numeric_cols=self._numeric_cols,
            numeric_dtypes=tuple(jnp.bool_ for _ in self._numeric_cols),
            object_data={},
            index=self._index,
            column_order=self._numeric_cols,
        )

    # ========================================
    # Internal helpers
    # ========================================

    @classmethod
    def _from_parts(
        cls,
        numeric_data,
        numeric_cols,
        numeric_dtypes,
        object_data,
        index,
        column_order,
    ):
        """Internal constructor from pre-separated parts."""
        df = cls.__new__(cls)
        df._numeric_data = numeric_data
        df._numeric_cols = numeric_cols
        df._numeric_dtypes = numeric_dtypes
        df._object_data = object_data
        df._index = index
        df._column_order = column_order
        return df


@dataclass
class Series:
    """
    A single column (1D array).

    Simpler than DataFrame but similar interface.
    """

    _data: Union[jnp.ndarray, np.ndarray]
    _index: np.ndarray
    _name: Optional[str] = None

    def __init__(self, data, index=None, name=None):
        """Create a Series from array-like data."""
        # Convert to array, keeping as numpy for non-numeric types
        if isinstance(data, (list, np.ndarray)):
            arr = np.asarray(data)
            # Only convert to JAX if numeric
            if np.issubdtype(arr.dtype, np.number):
                self._data = jnp.asarray(arr)
            else:
                self._data = arr
        else:
            self._data = data

        if index is None:
            self._index = np.arange(len(data))
        else:
            self._index = np.asarray(index)
        self._name = name

    @property
    def values(self):
        """Return underlying array."""
        return self._data

    @property
    def name(self):
        """Return series name."""
        return self._name

    def sum(self):
        """Sum of all values (JIT-compatible)."""
        return jnp.sum(self._data)

    def mean(self):
        """Mean of all values (JIT-compatible)."""
        return jnp.mean(self._data)

    def __repr__(self):
        """String representation."""
        lines = [f"Series(name={self._name}, shape={len(self._data)})"]
        n_show = min(5, len(self._data))
        for i in range(n_show):
            lines.append(f"{self._index[i]}  {self._data[i]}")
        if len(self._data) > n_show:
            lines.append("  ...")
        return "\n".join(lines)


# ========================================
# JAX Pytree Registration
# ========================================

def _dataframe_flatten(df: DataFrame):
    """
    Flatten DataFrame for JAX transformations.

    Only numeric data participates in JAX operations (grad, jit, vmap).
    Object data is auxiliary and passes through unchanged.
    """
    # Children: arrays that participate in JAX operations
    children = (df._numeric_data,) if df._numeric_data is not None else ()

    # Aux data: metadata that doesn't participate in JAX ops
    aux_data = {
        'numeric_cols': df._numeric_cols,
        'numeric_dtypes': df._numeric_dtypes,
        'object_data': df._object_data,
        'index': df._index,
        'column_order': df._column_order,
    }

    return children, aux_data


def _dataframe_unflatten(aux_data, children):
    """Reconstruct DataFrame from flattened representation."""
    numeric_data = children[0] if children else None

    return DataFrame._from_parts(
        numeric_data=numeric_data,
        numeric_cols=aux_data['numeric_cols'],
        numeric_dtypes=aux_data['numeric_dtypes'],
        object_data=aux_data['object_data'],
        index=aux_data['index'],
        column_order=aux_data['column_order'],
    )


# Register DataFrame as a JAX pytree
jax.tree_util.register_pytree_node(
    DataFrame,
    _dataframe_flatten,
    _dataframe_unflatten,
)


def _series_flatten(series: Series):
    """Flatten Series for JAX transformations."""
    children = (series._data,)
    aux_data = {
        'index': series._index,
        'name': series._name,
    }
    return children, aux_data


def _series_unflatten(aux_data, children):
    """Reconstruct Series from flattened representation."""
    data, = children
    return Series(data, index=aux_data['index'], name=aux_data['name'])


# Register Series as a JAX pytree
jax.tree_util.register_pytree_node(
    Series,
    _series_flatten,
    _series_unflatten,
)
