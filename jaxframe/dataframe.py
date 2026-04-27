"""
Core DataFrame implementation with JAX backend.

This module implements a DataFrame class that:
1. Stores numeric data in JAX arrays for performance
2. Stores non-numeric data in numpy arrays
3. Supports JIT compilation via pytree registration
4. Supports automatic differentiation on numeric columns
"""

from dataclasses import dataclass
from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np


def concat(dataframes: list["DataFrame"], axis: int = 0):
    """
    Concatenate DataFrames along an axis (JIT-compatible with dtype preservation).

    Args:
        dataframes: List of DataFrames to concatenate
        axis: 0 for vertical (row-wise), 1 for horizontal (column-wise)

    Returns:
        DataFrame: Concatenated result

    Examples:
        >>> df1 = DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> df2 = DataFrame({'a': [5, 6], 'b': [7, 8]})
        >>> concat([df1, df2], axis=0)  # Stack vertically
    """
    if not dataframes:
        raise ValueError("Need at least one DataFrame to concatenate")

    if axis == 0:
        # Vertical concatenation (stack rows)
        # All DataFrames must have same columns
        first_cols = dataframes[0]._column_order
        if not all(df._column_order == first_cols for df in dataframes):
            raise ValueError("All DataFrames must have the same columns for axis=0")

        # Concatenate dtype blocks - concat blocks of same dtype
        new_dtype_blocks = {}
        first_column_to_block = dataframes[0]._column_to_block

        for dtype in dataframes[0]._dtype_blocks.keys():
            blocks_to_concat = [df._dtype_blocks[dtype] for df in dataframes]
            new_dtype_blocks[dtype] = jnp.concatenate(blocks_to_concat, axis=0)

        # Concatenate object data
        new_object_data = {}
        for col in dataframes[0]._object_data.keys():
            obj_arrays = [df._object_data[col] for df in dataframes]
            new_object_data[col] = np.concatenate(obj_arrays, axis=0)

        # Concatenate indices
        indices = [df._index for df in dataframes]
        new_index = np.concatenate(indices, axis=0)

        return dataframes[0].__class__._from_parts(
            dtype_blocks=new_dtype_blocks,
            column_to_block=first_column_to_block.copy(),
            object_data=new_object_data,
            index=new_index,
            column_order=dataframes[0]._column_order,
        )
    elif axis == 1:
        # Horizontal concatenation (stack columns)
        # All DataFrames must have same number of rows
        first_len = len(dataframes[0])
        if not all(len(df) == first_len for df in dataframes):
            raise ValueError("All DataFrames must have the same number of rows for axis=1")

        # Collect all columns preserving dtypes
        all_col_data = {}
        for df in dataframes:
            for col in df._column_order:
                if col in df._column_to_block:
                    dtype, idx = df._column_to_block[col]
                    all_col_data[col] = (df._dtype_blocks[dtype][:, idx], dtype)
                elif col in df._object_data:
                    all_col_data[col] = (df._object_data[col], "object")

        # Build new DataFrame from collected columns
        numeric_data = {}
        object_data = {}
        for col, (data, dtype) in all_col_data.items():
            if dtype == "object":
                object_data[col] = data
            else:
                numeric_data[col] = data

        # Column order
        tuple(col for df in dataframes for col in df._column_order)

        # Create via dict init to properly group by dtype
        result_dict = {**numeric_data, **object_data}
        return DataFrame(result_dict, index=dataframes[0]._index)

    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")


def read_csv(path, index_col=None, **kwargs):
    """Read a CSV file into a DataFrame. Not JIT-compatible (I/O)."""
    import pandas as pd

    pdf = pd.read_csv(path, index_col=index_col, **kwargs)
    return DataFrame.from_pandas(pdf)


# ========================================
# JAX Compatibility Registry
# ========================================

# Each entry: (op_name, jit_compatible, grad_compatible, reason_if_not_grad)
_JAX_COMPAT = [
    # Arithmetic
    ("__add__", True, True, None),
    ("__sub__", True, True, None),
    ("__mul__", True, True, None),
    ("__truediv__", True, True, None),
    ("__floordiv__", True, True, None),
    ("__mod__", True, True, None),
    ("__pow__", True, True, None),
    ("__radd__", True, True, None),
    ("__rsub__", True, True, None),
    ("__rmul__", True, True, None),
    ("__rtruediv__", True, True, None),
    ("__rfloordiv__", True, True, None),
    ("__rmod__", True, True, None),
    ("__rpow__", True, True, None),
    ("abs", True, True, None),
    # Reductions
    ("sum", True, True, None),
    ("mean", True, True, None),
    ("std", True, True, None),
    ("var", True, True, None),
    ("prod", True, True, None),
    ("min", True, False, "Non-smooth (gradient undefined at argmin/argmax)"),
    ("max", True, False, "Non-smooth (gradient undefined at argmin/argmax)"),
    ("median", True, False, "Non-smooth (sort-based)"),
    ("count", True, False, "Integer output — not real-valued"),
    ("all", True, False, "Boolean output — not real-valued"),
    ("any", True, False, "Boolean output — not real-valued"),
    # Statistical
    ("skew", True, True, None),
    ("kurt", True, True, None),
    ("sem", True, True, None),
    # Transforms
    ("cumsum", True, True, None),
    ("cumprod", True, True, None),
    ("shift", True, True, None),
    ("where", True, True, None),
    ("clip", True, True, None),
    ("fillna", True, True, None),
    ("apply", True, True, None),
    ("copy", True, True, None),
    ("astype", True, False, "Discrete type conversion"),
    # Boolean / discrete
    ("isna", True, False, "Boolean output — not real-valued"),
    ("notna", True, False, "Boolean output — not real-valued"),
    ("isin", True, False, "Boolean output — not real-valued"),
    ("round", True, False, "Step function — gradient zero almost everywhere"),
    ("idxmin", True, False, "Discrete (argmin) — not differentiable"),
    ("idxmax", True, False, "Discrete (argmax) — not differentiable"),
    ("sort_values", True, False, "Permutation-based (argsort) — discrete"),
    ("nlargest", True, False, "Permutation-based (argsort) — discrete"),
    ("nsmallest", True, False, "Permutation-based (argsort) — discrete"),
    ("quantile", True, False, "Non-smooth (sort-based)"),
    # Column ops
    ("drop", True, True, None),
    ("rename", True, True, None),
    # Rolling (fixed-size)
    ("rolling.sum", True, True, None),
    ("rolling.mean", True, True, None),
    ("rolling.std", True, True, None),
    ("rolling.var", True, True, None),
    ("rolling.min", True, False, "Non-smooth"),
    ("rolling.max", True, False, "Non-smooth"),
    # GroupBy (segment ops)
    ("groupby.sum", True, True, None),
    ("groupby.mean", True, True, None),
    ("groupby.std", True, True, None),
    ("groupby.var", True, True, None),
    ("groupby.transform", True, True, None),
    ("groupby.min", True, False, "Non-smooth"),
    ("groupby.max", True, False, "Non-smooth"),
    ("groupby.count", True, False, "Integer output"),
    ("groupby.prod", True, False, "JAX scatter_mul grad unimplemented"),
    ("groupby.first", True, False, "Discrete (index-based gather)"),
    ("groupby.last", True, False, "Discrete (index-based gather)"),
    # Eager-only (not JIT-compatible)
    ("describe", False, False, "Returns pandas DataFrame"),
    ("to_pandas", False, False, "I/O — outside JAX"),
    ("from_pandas", False, False, "I/O — outside JAX"),
    ("to_csv", False, False, "I/O — outside JAX"),
    ("to_numpy", False, False, "Converts to numpy"),
    ("nunique", False, False, "Uses jnp.unique (eager, not traceable)"),
    ("mode", False, False, "Uses jnp.unique (eager, not traceable)"),
    ("value_counts", False, False, "Uses jnp.unique (eager, not traceable)"),
    ("rolling(str)", False, False, "Time-based: variable window sizes"),
]


def jax_info(op_name=None):
    """Query JAX compatibility for jaxframe operations.

    Args:
        op_name: Optional operation name to query. If None, prints all.

    Returns:
        If op_name given: dict with 'jit', 'grad', 'reason' keys.
        If None: prints a formatted compatibility table.

    Examples:
        >>> jaxframe.jax_info("sum")
        {'jit': True, 'grad': True, 'reason': None}
        >>> jaxframe.jax_info("min")
        {'jit': True, 'grad': False, 'reason': 'Non-smooth ...'}
        >>> jaxframe.jax_info()  # prints full table
    """
    if op_name is not None:
        for name, jit_ok, grad_ok, reason in _JAX_COMPAT:
            if name == op_name:
                return {"jit": jit_ok, "grad": grad_ok, "reason": reason}
        return None

    # Print full table
    print(f"{'Operation':<25} {'JIT':>5} {'Grad':>5}  Reason")
    print("-" * 70)
    for name, jit_ok, grad_ok, reason in _JAX_COMPAT:
        jit_str = "Yes" if jit_ok else "No"
        grad_str = "Yes" if grad_ok else "No"
        reason_str = reason or ""
        print(f"{name:<25} {jit_str:>5} {grad_str:>5}  {reason_str}")


@dataclass
class DataFrame:
    """
    A DataFrame implementation with JAX backend.

    Uses dtype blocks architecture: one JAX array per unique dtype for memory efficiency
    and dtype preservation.

    Attributes:
        _dtype_blocks: Dict mapping dtype to JAX array of shape (n_rows, n_cols_of_that_dtype)
        _column_to_block: Dict mapping column name to (dtype, column_index_in_block)
        _object_data: Dictionary mapping column names to numpy arrays for non-numeric data
        _index: Row index (currently simple integer index)
        _column_order: Original column order (for preserving user's column order)
    """

    _dtype_blocks: dict[np.dtype, jnp.ndarray]
    _column_to_block: dict[str, tuple[np.dtype, int]]
    _object_data: dict[str, np.ndarray]
    _index: np.ndarray
    _column_order: tuple[str, ...]

    def __init__(self, data: dict[str, Any] | np.ndarray | jnp.ndarray, index=None):
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

    def _init_from_dict(self, data: dict[str, Any], index=None):
        """Initialize from dictionary of columns using dtype blocks."""
        if not data:
            # Empty DataFrame
            self._dtype_blocks = {}
            self._column_to_block = {}
            self._object_data = {}
            self._index = np.array([])
            self._column_order = ()
            return

        # Determine number of rows
        first_col = next(iter(data.values()))
        n_rows = len(first_col) if hasattr(first_col, "__len__") else 1

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

            # Check if numeric or boolean
            if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
                numeric_cols_dict[col_name] = arr
            else:
                object_cols_dict[col_name] = arr

        # Store column order
        self._column_order = tuple(data.keys())

        # Create dtype blocks: group columns by dtype
        if numeric_cols_dict:
            # Group columns by their dtype
            dtype_groups = {}  # dtype -> [(col_name, array), ...]

            for col_name, arr in numeric_cols_dict.items():
                dtype = np.dtype(arr.dtype)
                if dtype not in dtype_groups:
                    dtype_groups[dtype] = []
                dtype_groups[dtype].append((col_name, arr))

            # Create one block per dtype
            self._dtype_blocks = {}
            self._column_to_block = {}

            for dtype, col_array_pairs in dtype_groups.items():
                # Extract arrays for this dtype
                arrays = [arr for col_name, arr in col_array_pairs]

                # Stack into a 2D JAX array (n_rows, n_cols_of_this_dtype)
                if len(arrays) == 1:
                    block = jnp.asarray(arrays[0]).reshape(n_rows, 1)
                else:
                    block = jnp.stack(arrays, axis=1)

                self._dtype_blocks[dtype] = block

                # Build column -> (dtype, index) mapping
                for idx, (col_name, arr) in enumerate(col_array_pairs):
                    self._column_to_block[col_name] = (dtype, idx)
        else:
            self._dtype_blocks = {}
            self._column_to_block = {}

        # Store object columns
        self._object_data = object_cols_dict

    def _init_from_array(self, data: np.ndarray | jnp.ndarray, index=None):
        """Initialize from 2D array (all numeric) using dtype blocks."""
        arr = jnp.asarray(data)
        if arr.ndim != 2:
            raise ValueError("Array must be 2-dimensional")

        n_rows, n_cols = arr.shape

        # Generate column names
        column_names = tuple(f"col_{i}" for i in range(n_cols))
        self._column_order = column_names
        self._object_data = {}

        # All columns have the same dtype, so create a single block
        dtype = np.dtype(arr.dtype)
        self._dtype_blocks = {dtype: arr}

        # Build column -> (dtype, index) mapping
        self._column_to_block = {
            col_name: (dtype, idx) for idx, col_name in enumerate(column_names)
        }

        # Set index
        if index is None:
            self._index = np.arange(n_rows)
        else:
            self._index = np.asarray(index)

    @property
    def shape(self) -> tuple[int, int]:
        """Return (n_rows, n_cols)."""
        n_rows = len(self._index)
        n_cols = len(self._column_order)
        return (n_rows, n_cols)

    def __len__(self) -> int:
        """Return number of rows."""
        return len(self._index)

    @property
    def columns(self) -> list[str]:
        """Return list of column names."""
        return list(self._column_order)

    @property
    def index(self) -> np.ndarray:
        """Return row index."""
        return self._index

    @property
    def device(self):
        """Return the device where numeric data is stored."""
        if not self._dtype_blocks:
            return None
        # Return device of first block (all blocks should be on same device)
        first_block = next(iter(self._dtype_blocks.values()))
        return first_block.device

    @property
    def size(self) -> int:
        """Return total number of elements (rows * cols)."""
        n_rows, n_cols = self.shape
        return n_rows * n_cols

    @property
    def ndim(self) -> int:
        """Return number of dimensions (always 2 for DataFrame)."""
        return 2

    @property
    def dtypes(self) -> dict[str, Any]:
        """Return dictionary mapping column names to dtypes."""
        dtypes_dict = {}
        # Add numeric/boolean columns from dtype blocks
        for col, (dtype, idx) in self._column_to_block.items():
            dtypes_dict[col] = dtype
        # Add object columns
        for col, arr in self._object_data.items():
            dtypes_dict[col] = arr.dtype
        return dtypes_dict

    @property
    def values(self) -> jnp.ndarray:
        """
        Return numeric data as JAX array with type promotion.

        Returns:
            JAX array of shape (n_rows, n_numeric_cols) containing only numeric columns.
            All dtype blocks are promoted to a common dtype and concatenated.
            Non-numeric columns are excluded.

        Note:
            This is JIT-compatible and differentiable (unlike pandas which returns mixed types).
            Type promotion follows NumPy rules (e.g., int32 + float32 → float32).
        """
        if not self._dtype_blocks:
            raise ValueError("DataFrame has no numeric columns")

        # Get the promoted dtype for all blocks
        all_dtypes = list(self._dtype_blocks.keys())
        promoted_dtype = all_dtypes[0]
        for dtype in all_dtypes[1:]:
            promoted_dtype = jnp.result_type(promoted_dtype, dtype)

        # Convert all blocks to promoted dtype and concatenate in column order
        numeric_cols = [col for col in self._column_order if col in self._column_to_block]
        columns = []

        for col in numeric_cols:
            dtype, idx = self._column_to_block[col]
            block = self._dtype_blocks[dtype]
            col_data = block[:, idx : idx + 1]  # Keep 2D shape
            columns.append(col_data.astype(promoted_dtype))

        return jnp.concatenate(columns, axis=1)

    @property
    def empty(self) -> bool:
        """Return True if DataFrame has no elements."""
        return self.size == 0

    # Backward compatibility properties for tests
    @property
    def _numeric_cols(self) -> tuple[str, ...]:
        """Get list of numeric column names (backward compatibility)."""
        return tuple(col for col in self._column_order if col in self._column_to_block)

    @property
    def _numeric_data(self) -> jnp.ndarray | None:
        """Get numeric data as single array (backward compatibility)."""
        if not self._dtype_blocks:
            return None
        return self.values

    @property
    def _numeric_dtypes(self) -> tuple[Any, ...]:
        """Get numeric column dtypes (backward compatibility)."""
        return tuple(self._column_to_block[col][0] for col in self._numeric_cols)

    def _all_numeric(self):
        """Get all numeric data as a single flat array. Faster than values for scalar reductions."""
        if not self._dtype_blocks:
            raise ValueError("No numeric columns")
        blocks = list(self._dtype_blocks.values())
        if len(blocks) == 1:
            return blocks[0]
        return jnp.concatenate([b.reshape(-1) for b in blocks])

    # ========================================
    # Device management methods
    # ========================================

    def to_device(self, device):
        """
        Transfer DataFrame to a specific device.

        Args:
            device: Target device (JAX device object or string like 'gpu', 'cpu', 'tpu')

        Returns:
            New DataFrame with data on target device

        Examples:
            >>> df_gpu = df.to_device('gpu')
            >>> df_gpu = df.to_device(jax.devices('gpu')[0])
        """
        if not self._dtype_blocks:
            return self

        # Handle string device specifications
        if isinstance(device, str):
            device = jax.devices(device)[0]

        # Transfer all dtype blocks to device
        new_dtype_blocks = {
            dtype: jax.device_put(block, device) for dtype, block in self._dtype_blocks.items()
        }

        return DataFrame._from_parts(
            dtype_blocks=new_dtype_blocks,
            column_to_block=self._column_to_block,
            object_data=self._object_data,
            index=self._index,
            column_order=self._column_order,
        )

    def to_gpu(self, id: int = 0):
        """
        Transfer DataFrame to GPU.

        Args:
            id: GPU device ID (default 0)

        Returns:
            New DataFrame with data on GPU

        Examples:
            >>> df_gpu = df.to_gpu()      # Default GPU
            >>> df_gpu = df.to_gpu(1)     # GPU 1
        """
        try:
            gpu_devices = jax.devices("gpu")
            if id >= len(gpu_devices):
                raise ValueError(f"GPU {id} not found. Available GPUs: {len(gpu_devices)}")
            return self.to_device(gpu_devices[id])
        except RuntimeError:
            raise RuntimeError("No GPU devices available")

    def to_cpu(self):
        """
        Transfer DataFrame to CPU.

        Returns:
            New DataFrame with data on CPU

        Examples:
            >>> df_cpu = df.to_cpu()
        """
        cpu_device = jax.devices("cpu")[0]
        return self.to_device(cpu_device)

    def to_tpu(self, id: int = 0):
        """
        Transfer DataFrame to TPU.

        Args:
            id: TPU device ID (default 0)

        Returns:
            New DataFrame with data on TPU

        Examples:
            >>> df_tpu = df.to_tpu()      # Default TPU
            >>> df_tpu = df.to_tpu(1)     # TPU 1
        """
        try:
            tpu_devices = jax.devices("tpu")
            if id >= len(tpu_devices):
                raise ValueError(f"TPU {id} not found. Available TPUs: {len(tpu_devices)}")
            return self.to_device(tpu_devices[id])
        except RuntimeError:
            raise RuntimeError("No TPU devices available")

    def __getitem__(self, key: str | list[str]):
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
                            if hasattr(val, "shape") and val.shape:
                                # Multi-dimensional value, show shape instead
                                row_data.append(f"{'Array' + str(val.shape):>10}")
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

    def __mul__(self, other: Union[float, "DataFrame"]) -> "DataFrame":
        """Element-wise multiplication (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.multiply, "multiplication")

    def __add__(self, other: Union[float, "DataFrame"]) -> "DataFrame":
        """Element-wise addition (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.add, "addition")

    def __sub__(self, other: Union[float, "DataFrame", "Series"]) -> "DataFrame":
        """Element-wise subtraction (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.subtract, "subtraction")

    def __truediv__(self, other: Union[float, "DataFrame", "Series"]) -> "DataFrame":
        """Element-wise division (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.true_divide, "division")

    def __floordiv__(self, other: Union[float, "DataFrame"]) -> "DataFrame":
        """Element-wise floor division (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.floor_divide, "floor division")

    def __mod__(self, other: Union[float, "DataFrame"]) -> "DataFrame":
        """Element-wise modulo (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.mod, "modulo")

    def __pow__(self, other: Union[float, "DataFrame"]) -> "DataFrame":
        """Element-wise power (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.power, "power")

    # Reverse operators
    def __radd__(self, other):
        return self._apply_elementwise(other, lambda a, b: jnp.add(b, a), "radd")

    def __rsub__(self, other):
        return self._apply_elementwise(other, lambda a, b: jnp.subtract(b, a), "rsub")

    def __rmul__(self, other):
        return self._apply_elementwise(other, lambda a, b: jnp.multiply(b, a), "rmul")

    def __rtruediv__(self, other):
        return self._apply_elementwise(other, lambda a, b: jnp.true_divide(b, a), "rtruediv")

    def __rfloordiv__(self, other):
        return self._apply_elementwise(other, lambda a, b: jnp.floor_divide(b, a), "rfloordiv")

    def __rmod__(self, other):
        return self._apply_elementwise(other, lambda a, b: jnp.mod(b, a), "rmod")

    def __rpow__(self, other):
        return self._apply_elementwise(other, lambda a, b: jnp.power(b, a), "rpow")

    def __matmul__(self, other: Union[jnp.ndarray, "DataFrame"]) -> Union[jnp.ndarray, "DataFrame"]:
        """Matrix multiplication (JIT-compatible with type promotion)."""
        if not self._dtype_blocks:
            raise ValueError("No numeric columns for matrix multiplication")

        if isinstance(other, DataFrame):
            if not other._dtype_blocks:
                raise ValueError("Other DataFrame has no numeric columns")
            result = self.values @ other.values
            # Result is array, not DataFrame (consistent with pandas behavior)
            return result
        else:
            # Handle JAX arrays
            return self.values @ other

    def sum(self, axis: int | None = 0):
        """
        Sum along axis (JIT-compatible).

        Skips NaN values by default (matching pandas behavior).

        Args:
            axis: 0 for column-wise sum, 1 for row-wise sum, None for total sum

        Returns:
            Series (axis=0 or 1) or scalar (axis=None)
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns to sum")

        if axis is None:
            return jnp.nansum(self._all_numeric())

        result = jnp.nansum(self._numeric_data, axis=axis)
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="sum")
        else:
            return Series(result, index=self._index, name="sum")

    def mean(self, axis: int | None = 0):
        """
        Mean along axis (JIT-compatible).

        Skips NaN values by default (matching pandas behavior).
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns to compute mean")

        if axis is None:
            return jnp.nanmean(self._all_numeric())

        result = jnp.nanmean(self._numeric_data, axis=axis)
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="mean")
        else:
            return Series(result, index=self._index, name="mean")

    def std(self, axis: int | None = 0, ddof: int = 1):
        """
        Standard deviation along axis (JIT-compatible, differentiable).

        Skips NaN values by default (matching pandas behavior).

        Args:
            axis: 0 for column-wise, 1 for row-wise, None for total
            ddof: Delta degrees of freedom (default 1 for sample std)
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns to compute std")

        if axis is None:
            return jnp.nanstd(self._all_numeric(), ddof=ddof)

        result = jnp.nanstd(self._numeric_data, axis=axis, ddof=ddof)
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="std")
        else:
            return Series(result, index=self._index, name="std")

    def var(self, axis: int | None = 0, ddof: int = 1):
        """
        Variance along axis (JIT-compatible, differentiable).

        Skips NaN values by default (matching pandas behavior).

        Args:
            axis: 0 for column-wise, 1 for row-wise, None for total
            ddof: Delta degrees of freedom (default 1 for sample variance)
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns to compute variance")

        if axis is None:
            return jnp.nanvar(self._all_numeric(), ddof=ddof)

        result = jnp.nanvar(self._numeric_data, axis=axis, ddof=ddof)
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="var")
        else:
            return Series(result, index=self._index, name="var")

    def min(self, axis: int | None = 0):
        """
        Minimum along axis (JIT-compatible).

        Skips NaN values by default (matching pandas behavior).

        Note: Gradient is non-smooth at the minimum point.
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns to compute min")

        if axis is None:
            return jnp.nanmin(self._all_numeric())

        result = jnp.nanmin(self._numeric_data, axis=axis)
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="min")
        else:
            return Series(result, index=self._index, name="min")

    def max(self, axis: int | None = 0):
        """
        Maximum along axis (JIT-compatible).

        Skips NaN values by default (matching pandas behavior).

        Note: Gradient is non-smooth at the maximum point.
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns to compute max")

        if axis is None:
            return jnp.nanmax(self._all_numeric())

        result = jnp.nanmax(self._numeric_data, axis=axis)
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="max")
        else:
            return Series(result, index=self._index, name="max")

    def prod(self, axis: int | None = 0):
        """
        Product along axis (JIT-compatible, differentiable).

        Skips NaN values by default (matching pandas behavior).
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns to compute product")

        if axis is None:
            return jnp.nanprod(self._all_numeric())

        result = jnp.nanprod(self._numeric_data, axis=axis)
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="prod")
        else:
            return Series(result, index=self._index, name="prod")

    def abs(self):
        """Absolute value (JIT-compatible). Gradient is non-smooth at zero."""
        return self._apply_blockwise(jnp.abs)

    def all(self, axis: int = 0):
        """Check if all values are true along axis (JIT-compatible). Boolean output."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns")
        result = jnp.all(self._numeric_data.astype(bool), axis=axis)
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="all")
        else:
            return Series(result, index=self._index, name="all")

    def any(self, axis: int = 0):
        """Check if any value is true along axis (JIT-compatible). Boolean output."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns")
        result = jnp.any(self._numeric_data.astype(bool), axis=axis)
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="any")
        else:
            return Series(result, index=self._index, name="any")

    def round(self, decimals: int = 0):
        """Round to given number of decimals (JIT-compatible, differentiable=False)."""
        return self._apply_blockwise(lambda block: jnp.round(block, decimals))

    def idxmin(self, axis: int = 0):
        """Return index of first occurrence of minimum (JIT-compatible). Not differentiable."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns")
        indices = jnp.argmin(self._numeric_data, axis=axis)
        if axis == 0:
            # Use jnp array for index to keep JIT-compatible
            idx_arr = jnp.asarray(self._index)
            result = jnp.take(idx_arr, indices)
            return Series(result, index=np.array(self._numeric_cols), name="idxmin")
        else:
            col_names = np.array(self._numeric_cols)
            result = col_names[indices]
            return Series(result, index=self._index, name="idxmin")

    def idxmax(self, axis: int = 0):
        """Return index of first occurrence of maximum (JIT-compatible). Not differentiable."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns")
        indices = jnp.argmax(self._numeric_data, axis=axis)
        if axis == 0:
            idx_arr = jnp.asarray(self._index)
            result = jnp.take(idx_arr, indices)
            return Series(result, index=np.array(self._numeric_cols), name="idxmax")
        else:
            col_names = np.array(self._numeric_cols)
            result = col_names[indices]
            return Series(result, index=self._index, name="idxmax")

    def isin(self, values):
        """Check whether each element is in values (JIT-compatible). Boolean output."""
        values_arr = jnp.asarray(values) if len(values) > 0 else jnp.array([])

        def _check_block(block):
            if len(values_arr) == 0:
                return jnp.zeros_like(block, dtype=bool)
            # Compare each element against all values: (n_rows, n_cols, 1) == (1, 1, n_values)
            return jnp.any(block[..., None] == values_arr[None, None, :], axis=-1)

        return self._apply_blockwise(_check_block)

    def nunique(self, axis: int = 0):
        """Count unique values per column. Not JIT-compatible (eager unique)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns")
        data = self._numeric_data
        if axis == 0:
            counts = [len(jnp.unique(data[:, i])) for i in range(data.shape[1])]
            return Series(jnp.array(counts), index=np.array(self._numeric_cols), name="nunique")
        else:
            counts = [len(jnp.unique(data[i, :])) for i in range(data.shape[0])]
            return Series(jnp.array(counts), index=self._index, name="nunique")

    def skew(self, axis: int = 0):
        """Skewness (JIT-compatible, differentiable)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns")
        data = self._numeric_data
        n = data.shape[axis]
        mean = jnp.nanmean(data, axis=axis, keepdims=True)
        m2 = jnp.nansum((data - mean) ** 2, axis=axis)
        m3 = jnp.nansum((data - mean) ** 3, axis=axis)
        # Bias-corrected skewness (pandas default)
        s2 = m2 / (n - 1)
        result = (m3 / n) / (s2**1.5) * (n**2) / ((n - 1) * (n - 2))
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="skew")
        else:
            return Series(result, index=self._index, name="skew")

    def kurt(self, axis: int = 0):
        """Excess kurtosis (JIT-compatible, differentiable)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns")
        data = self._numeric_data
        n = data.shape[axis]
        mean = jnp.nanmean(data, axis=axis, keepdims=True)
        m2 = jnp.nansum((data - mean) ** 2, axis=axis)
        m4 = jnp.nansum((data - mean) ** 4, axis=axis)
        # Bias-corrected excess kurtosis (pandas default)
        s2 = m2 / (n - 1)
        adj = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        result = adj * (m4 / (s2**2)) - 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="kurt")
        else:
            return Series(result, index=self._index, name="kurt")

    kurtosis = kurt

    def sem(self, axis: int = 0, ddof: int = 1):
        """Standard error of the mean (JIT-compatible, differentiable)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns")
        std = jnp.nanstd(self._numeric_data, axis=axis, ddof=ddof)
        n = self._numeric_data.shape[axis]
        result = std / jnp.sqrt(n)
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="sem")
        else:
            return Series(result, index=self._index, name="sem")

    def mode(self, axis: int = 0):
        """Mode (most frequent value). Not JIT-compatible (eager unique)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns")
        data = self._numeric_data
        if axis == 0:
            modes = []
            for i in range(data.shape[1]):
                vals, counts = jnp.unique(data[:, i], return_counts=True)
                modes.append(vals[jnp.argmax(counts)])
            return DataFrame({col: [m] for col, m in zip(self._numeric_cols, modes)})
        else:
            raise NotImplementedError("mode(axis=1) not yet supported")

    def _apply_cross_column(self, fn):
        """Apply fn across all numeric columns (axis=1 ops). JIT-compatible."""
        data = self.values  # (n_rows, n_numeric_cols)
        result = fn(data)
        cols = self._numeric_cols
        new_dtype = result.dtype
        new_blocks = {new_dtype: result}
        new_col_to_block = {col: (new_dtype, i) for i, col in enumerate(cols)}
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._object_data,
            index=self._index,
            column_order=self._column_order,
        )

    def cumsum(self, axis: int = 0):
        """Cumulative sum along axis (JIT-compatible, differentiable)."""
        if axis == 0:
            return self._apply_blockwise(lambda b: jnp.cumsum(b, axis=0))
        elif axis == 1:
            return self._apply_cross_column(lambda d: jnp.cumsum(d, axis=1))
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")

    def cumprod(self, axis: int = 0):
        """Cumulative product along axis (JIT-compatible, differentiable)."""
        if axis == 0:
            return self._apply_blockwise(lambda b: jnp.cumprod(b, axis=0))
        elif axis == 1:
            return self._apply_cross_column(lambda d: jnp.cumprod(d, axis=1))
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")

    def count(self, axis: int = 0):
        """Count non-NaN values along axis (JIT-compatible)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns to count")
        result = jnp.sum(~jnp.isnan(self._numeric_data), axis=axis)
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="count")
        else:
            return Series(result, index=self._index, name="count")

    def median(self, axis: int = 0):
        """Median along axis (JIT-compatible). Gradient is non-smooth."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute median")
        result = jnp.nanmedian(self._numeric_data, axis=axis)
        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="median")
        else:
            return Series(result, index=self._index, name="median")

    def _apply_blockwise(self, fn):
        """Apply fn to each dtype block, return new DataFrame via _from_parts."""
        new_blocks = {dtype: fn(block) for dtype, block in self._dtype_blocks.items()}
        # Determine new dtype for column_to_block mapping
        new_column_to_block = {}
        for col, (old_dtype, idx) in self._column_to_block.items():
            new_dtype = new_blocks[old_dtype].dtype
            new_column_to_block[col] = (new_dtype, idx)
        # Re-key blocks by their actual dtype
        rekeyed = {}
        for old_dtype, block in new_blocks.items():
            rekeyed[block.dtype] = block
        return DataFrame._from_parts(
            dtype_blocks=rekeyed,
            column_to_block=new_column_to_block,
            object_data=self._object_data,
            index=self._index,
            column_order=self._column_order,
        )

    def isna(self):
        """Return boolean DataFrame indicating NaN positions (JIT-compatible)."""
        return self._apply_blockwise(jnp.isnan)

    isnull = isna

    def notna(self):
        """Return boolean DataFrame indicating non-NaN positions (JIT-compatible)."""
        return self._apply_blockwise(lambda block: ~jnp.isnan(block))

    notnull = notna

    def fillna(self, value):
        """Replace NaN with value (JIT-compatible)."""
        return self._apply_blockwise(lambda block: jnp.where(jnp.isnan(block), value, block))

    def corr(self):
        """
        Compute pairwise correlation of columns (JIT-compatible, differentiable).

        Returns:
            DataFrame: Correlation matrix

        Note:
            Only includes numeric columns.
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute correlation")

        # Center the data
        centered = self._numeric_data - self._numeric_data.mean(axis=0, keepdims=True)

        # Compute correlation matrix
        # corr = (X^T @ X) / (n-1) / (std_x * std_y)
        cov_matrix = (centered.T @ centered) / (len(self._index) - 1)
        std_devs = jnp.std(self._numeric_data, axis=0, ddof=1)
        corr_matrix = cov_matrix / jnp.outer(std_devs, std_devs)

        # Create DataFrame with column names as both index and columns
        return DataFrame(
            {col: corr_matrix[:, i] for i, col in enumerate(self._numeric_cols)},
            index=np.array(self._numeric_cols),
        )

    def cov(self):
        """
        Compute pairwise covariance of columns (JIT-compatible, differentiable).

        Returns:
            DataFrame: Covariance matrix

        Note:
            Only includes numeric columns.
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute covariance")

        # Center the data
        centered = self._numeric_data - self._numeric_data.mean(axis=0, keepdims=True)

        # Compute covariance matrix: (X^T @ X) / (n-1)
        cov_matrix = (centered.T @ centered) / (len(self._index) - 1)

        # Create DataFrame with column names as both index and columns
        return DataFrame(
            {col: cov_matrix[:, i] for i, col in enumerate(self._numeric_cols)},
            index=np.array(self._numeric_cols),
        )

    # ========================================
    # Shape manipulation
    # ========================================

    @property
    def T(self):
        """Transpose (property) - JIT-compatible, differentiable."""
        return self.transpose()

    def transpose(self):
        """
        Transpose DataFrame (JIT-compatible, differentiable).

        Returns:
            DataFrame: Transposed DataFrame with swapped rows/columns

        Note:
            Column names become the index, index becomes column names.
            Only works cleanly with numeric data.
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to transpose")

        if self._object_data:
            raise ValueError("Cannot transpose DataFrame with object columns")

        # Transpose the numeric data
        transposed_data = self._numeric_data.T

        # Swap columns and index
        # Old columns become new index, old index becomes new columns
        new_cols = tuple(str(idx) for idx in self._index)
        new_index = np.array(self._numeric_cols)

        # Create new DataFrame
        return DataFrame(
            {col: transposed_data[:, i] for i, col in enumerate(new_cols)}, index=new_index
        )

    # ========================================
    # Sorting
    # ========================================

    def sort_values(self, by, ascending=True):
        """Sort by column values using argsort. JIT-compatible."""
        if isinstance(by, str):
            by = [by]
        # Get the primary sort column
        col = by[0]
        dtype, idx = self._column_to_block[col]
        sort_col = self._dtype_blocks[dtype][:, idx]
        order = jnp.argsort(sort_col)
        if not ascending:
            order = order[::-1]
        # Apply sort order to all blocks
        new_blocks = {dt: block[order] for dt, block in self._dtype_blocks.items()}
        # Index is static aux data in pytree — keep original under JIT
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=dict(self._column_to_block),
            object_data=self._object_data,
            index=self._index,
            column_order=list(self._column_order),
        )

    def sort_index(self, ascending=True):
        """Sort by index. JIT-compatible (index is static aux data)."""
        order = np.argsort(self._index)
        if not ascending:
            order = order[::-1]
        new_blocks = {dt: block[order] for dt, block in self._dtype_blocks.items()}
        new_obj = {k: v[order] for k, v in self._object_data.items()}
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=dict(self._column_to_block),
            object_data=new_obj,
            index=self._index[order],
            column_order=list(self._column_order),
        )

    def rank(self, ascending=True, method="ordinal"):
        """Rank values along axis 0. JIT-compatible (argsort-based, not differentiable).

        Args:
            ascending: True for smallest=1 ranking
            method: 'ordinal' (default) — ties broken by position.
                    'average', 'min', 'max', 'dense' also supported.
        """

        def _rank_block(block):
            n = block.shape[0]
            if method == "ordinal":
                if ascending:
                    order = jnp.argsort(block, axis=0)
                    ranks = jnp.argsort(order, axis=0) + 1
                else:
                    order = jnp.argsort(-block, axis=0)
                    ranks = jnp.argsort(order, axis=0) + 1
                return ranks.astype(jnp.float32)
            elif method == "average":
                order = jnp.argsort(block, axis=0)
                sorted_block = jnp.take_along_axis(block, order, axis=0)
                ranks = jnp.argsort(order, axis=0).astype(jnp.float32) + 1
                # For ties, average the ranks
                for c in range(block.shape[1]):
                    vals = sorted_block[:, c]
                    # Group equal values and assign average rank
                    same_as_next = jnp.concatenate([vals[:-1] == vals[1:], jnp.array([False])])
                    # Use cumulative sum of group boundaries
                    group_id = jnp.cumsum(~same_as_next)
                    group_id = jnp.concatenate([jnp.array([0]), group_id[:-1]])
                    # Compute average rank per group using segment operations
                    ordinal_ranks = jnp.arange(1, n + 1, dtype=jnp.float32)
                    rank_sums = jnp.zeros(n, dtype=jnp.float32).at[group_id].add(ordinal_ranks)
                    rank_counts = jnp.zeros(n, dtype=jnp.float32).at[group_id].add(1.0)
                    avg_ranks = rank_sums / rank_counts
                    # Map back to original positions
                    sorted_avg = avg_ranks[group_id]
                    result_col = jnp.empty(n, dtype=jnp.float32)
                    result_col = result_col.at[order[:, c]].set(sorted_avg)
                    ranks = ranks.at[:, c].set(result_col)
                if not ascending:
                    ranks = n + 1 - ranks
                return ranks
            else:
                # Fallback: ordinal
                order = jnp.argsort(block, axis=0)
                ranks = jnp.argsort(order, axis=0) + 1
                if not ascending:
                    ranks = block.shape[0] + 1 - ranks
                return ranks.astype(jnp.float32)

        return self._apply_blockwise(_rank_block)

    def duplicated(self, subset=None, keep="first"):
        """Return boolean Series denoting duplicate rows (eager, not JIT-compatible).

        Args:
            subset: column labels to consider. Default: all columns.
            keep: 'first', 'last', or False (mark all duplicates)
        """
        if subset is None:
            subset = list(self._column_order)
        elif isinstance(subset, str):
            subset = [subset]

        # Build (n_rows, n_subset_cols) array for comparison
        cols = [np.asarray(self._get_column_data(c)) for c in subset]
        data = np.column_stack(cols)

        # Use structured array for row-wise comparison
        dt = [(f"f{i}", data.dtype) for i in range(data.shape[1])]
        structured = np.array([tuple(row) for row in data], dtype=dt)
        _, idx, counts = np.unique(structured, return_index=True, return_counts=True)

        mask = np.zeros(len(data), dtype=bool)
        if keep == "first":
            # Mark all but first occurrence
            seen = {}
            for i, row in enumerate(structured):
                key = row.item() if hasattr(row, "item") else tuple(row)
                if key in seen:
                    mask[i] = True
                else:
                    seen[key] = i
        elif keep == "last":
            seen = {}
            for i in range(len(structured) - 1, -1, -1):
                row = structured[i]
                key = row.item() if hasattr(row, "item") else tuple(row)
                if key in seen:
                    mask[i] = True
                else:
                    seen[key] = i
        else:  # keep=False
            for i, row in enumerate(structured):
                key = row.item() if hasattr(row, "item") else tuple(row)
                # Mark if appears more than once
                pass
            seen_count = {}
            for row in structured:
                key = row.item() if hasattr(row, "item") else tuple(row)
                seen_count[key] = seen_count.get(key, 0) + 1
            for i, row in enumerate(structured):
                key = row.item() if hasattr(row, "item") else tuple(row)
                if seen_count[key] > 1:
                    mask[i] = True

        return Series(jnp.array(mask), index=self._index, name=None)

    def drop_duplicates(self, subset=None, keep="first"):
        """Remove duplicate rows (eager, not JIT-compatible).

        Structure discovery (finding duplicates) is eager.
        Row selection uses jnp.take (JIT-compatible data op).
        """
        dup_mask = self.duplicated(subset=subset, keep=keep)
        keep_mask = ~np.asarray(dup_mask.values)
        keep_indices = np.where(keep_mask)[0]

        new_blocks = {dt: block[keep_indices] for dt, block in self._dtype_blocks.items()}
        new_obj = {k: v[keep_indices] for k, v in self._object_data.items()}
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=dict(self._column_to_block),
            object_data=new_obj,
            index=self._index[keep_indices],
            column_order=self._column_order,
        )

    def reset_index(self, drop=True):
        """Reset index to default integer range.

        Args:
            drop: If True, discard the old index. If False, insert it as a column.
        """
        if drop:
            return DataFrame._from_parts(
                dtype_blocks={dt: block.copy() for dt, block in self._dtype_blocks.items()},
                column_to_block=dict(self._column_to_block),
                object_data=dict(self._object_data),
                index=np.arange(len(self._index)),
                column_order=list(self._column_order),
            )
        else:
            # Insert old index as a column
            idx_name = getattr(self, "_index_name", "index")
            data = {}
            data[idx_name] = np.asarray(self._index)
            for col in self._column_order:
                if col in self._column_to_block:
                    dtype, ci = self._column_to_block[col]
                    data[col] = self._dtype_blocks[dtype][:, ci]
                elif col in self._object_data:
                    data[col] = self._object_data[col]
            return DataFrame(data)

    def set_index(self, keys):
        """Set a column as the index, removing it from columns."""
        if isinstance(keys, str):
            keys = keys
        # Get the column data for the new index
        if keys in self._column_to_block:
            dtype, idx = self._column_to_block[keys]
            new_index = np.asarray(self._dtype_blocks[dtype][:, idx])
        elif keys in self._object_data:
            new_index = self._object_data[keys]
        else:
            raise KeyError(f"Column '{keys}' not found")
        return self.drop(columns=[keys])._replace_index(new_index)

    def _replace_index(self, new_index):
        """Return a copy with a new index."""
        return DataFrame._from_parts(
            dtype_blocks=dict(self._dtype_blocks),
            column_to_block=dict(self._column_to_block),
            object_data=dict(self._object_data),
            index=new_index,
            column_order=list(self._column_order),
        )

    # ========================================
    # Describe & quantile
    # ========================================

    def quantile(self, q=0.5):
        """Compute quantile along axis 0. JIT-compatible."""
        data = self.values
        result = jnp.nanquantile(data, q, axis=0)
        return Series(result, index=np.array(self._numeric_cols), name=q)

    def nlargest(self, n, columns):
        """Return top n rows by column. JIT-compatible."""
        return self.sort_values(columns, ascending=False).head(n)

    def nsmallest(self, n, columns):
        """Return bottom n rows by column. JIT-compatible."""
        return self.sort_values(columns, ascending=True).head(n)

    def describe(self):
        """Generate descriptive statistics. Not JIT-compatible (returns mixed)."""
        import pandas as pd

        data = self.values
        cols = self._numeric_cols
        stats = {}
        for i, col in enumerate(cols):
            col_data = data[:, i]
            stats[col] = {
                "count": float(jnp.sum(~jnp.isnan(col_data))),
                "mean": float(jnp.nanmean(col_data)),
                "std": float(jnp.nanstd(col_data, ddof=1)),
                "min": float(jnp.nanmin(col_data)),
                "25%": float(jnp.nanquantile(col_data, 0.25)),
                "50%": float(jnp.nanquantile(col_data, 0.50)),
                "75%": float(jnp.nanquantile(col_data, 0.75)),
                "max": float(jnp.nanmax(col_data)),
            }
        return pd.DataFrame(stats)

    # ========================================
    # Column manipulation
    # ========================================

    def drop(self, columns=None, **kwargs):
        """Drop columns. Returns new DataFrame via _from_parts. JIT-compatible."""
        if columns is None:
            columns = kwargs.get("labels", [])
        if isinstance(columns, str):
            columns = [columns]
        drop_set = set(columns)
        new_order = [c for c in self._column_order if c not in drop_set]
        # Rebuild blocks keeping only remaining columns
        new_blocks = {}
        new_col_to_block = {}
        col_counts = {}  # dtype -> next index
        for col in new_order:
            if col in self._column_to_block:
                old_dtype, old_idx = self._column_to_block[col]
                block = self._dtype_blocks[old_dtype]
                col_data = block[:, old_idx : old_idx + 1]
                if old_dtype not in new_blocks:
                    new_blocks[old_dtype] = []
                    col_counts[old_dtype] = 0
                new_blocks[old_dtype].append(col_data)
                new_col_to_block[col] = (old_dtype, col_counts[old_dtype])
                col_counts[old_dtype] += 1
        # Concatenate column slices into blocks
        final_blocks = {dt: jnp.concatenate(cols, axis=1) for dt, cols in new_blocks.items()}
        new_object = {k: v for k, v in self._object_data.items() if k not in drop_set}
        return DataFrame._from_parts(
            dtype_blocks=final_blocks,
            column_to_block=new_col_to_block,
            object_data=new_object,
            index=self._index,
            column_order=new_order,
        )

    def rename(self, columns=None, **kwargs):
        """Rename columns. JIT-compatible (metadata-only change)."""
        if columns is None:
            return self.copy()
        mapping = columns
        new_order = [mapping.get(c, c) for c in self._column_order]
        new_col_to_block = {mapping.get(c, c): v for c, v in self._column_to_block.items()}
        new_object = {mapping.get(k, k): v for k, v in self._object_data.items()}
        return DataFrame._from_parts(
            dtype_blocks=self._dtype_blocks,
            column_to_block=new_col_to_block,
            object_data=new_object,
            index=self._index,
            column_order=new_order,
        )

    def assign(self, **kwargs):
        """Assign new columns. Not JIT-compatible (creates via __init__)."""
        # Build data dict from current columns + new ones
        data = {}
        for col in self._column_order:
            if col in self._column_to_block:
                dtype, idx = self._column_to_block[col]
                data[col] = self._dtype_blocks[dtype][:, idx]
            elif col in self._object_data:
                data[col] = self._object_data[col]
        for col, val in kwargs.items():
            if isinstance(val, Series):
                data[col] = val._data
            elif hasattr(val, "__jax_array__") or hasattr(val, "shape"):
                data[col] = val
            else:
                data[col] = val
        return DataFrame(data, index=self._index)

    def astype(self, dtype):
        """Cast all numeric columns to dtype. JIT-compatible."""
        target = jnp.dtype(dtype)
        return self._apply_blockwise(lambda block: block.astype(target))

    def select_dtypes(self, include=None, exclude=None):
        """Select columns by dtype. Not JIT-compatible (metadata op)."""
        include_set = {np.dtype(d) for d in include} if include else None
        exclude_set = {np.dtype(d) for d in exclude} if exclude else set()

        keep_cols = []
        for col in self._column_order:
            if col in self._column_to_block:
                dtype, _ = self._column_to_block[col]
                if include_set and dtype not in include_set:
                    continue
                if dtype in exclude_set:
                    continue
                keep_cols.append(col)
            elif col in self._object_data:
                obj_dtype = np.dtype("object")
                if include_set and obj_dtype not in include_set:
                    continue
                if obj_dtype in exclude_set:
                    continue
                keep_cols.append(col)

        return self[keep_cols]

    # ========================================
    # Apply
    # ========================================

    def apply(self, func, axis=0):
        """Apply function along axis. JIT-compatible if func is JIT-compatible."""
        data = self.values
        if axis == 0:
            # Apply func to each column -> Series
            results = []
            for i in range(data.shape[1]):
                results.append(func(data[:, i]))
            return Series(
                jnp.array(results),
                index=np.array(self._numeric_cols),
            )
        else:
            # Apply func to each row -> Series
            results = []
            for i in range(data.shape[0]):
                results.append(func(data[i, :]))
            return Series(
                jnp.array(results),
                index=self._index,
            )

    def pipe(self, func, *args, **kwargs):
        """Apply a function to the DataFrame. JIT-compatible if func is."""
        return func(self, *args, **kwargs)

    def between(self, left, right, inclusive="both"):
        """Boolean mask for values between left and right (JIT-compatible).

        Args:
            left: Lower bound (scalar)
            right: Upper bound (scalar)
            inclusive: 'both', 'neither', 'left', 'right'
        """
        if inclusive == "both":
            return (self >= left) & (self <= right)
        elif inclusive == "left":
            return (self >= left) & (self < right)
        elif inclusive == "right":
            return (self > left) & (self <= right)
        else:  # neither
            return (self > left) & (self < right)

    # ========================================
    # Merge / Join
    # ========================================

    def merge(
        self,
        right,
        on=None,
        left_on=None,
        right_on=None,
        how="inner",
        suffixes=("_x", "_y"),
    ):
        """Merge two DataFrames on key columns.

        Structure discovery (key matching) is eager.
        Data gathering uses array indexing.

        Args:
            right: DataFrame to merge with
            on: Column name(s) to join on (if same in both)
            left_on/right_on: Column name(s) in left/right DataFrames
            how: 'inner', 'left', 'right', 'outer'
            suffixes: Suffixes for overlapping non-key column names
        """
        # Resolve key columns
        if on is not None:
            left_on = on if isinstance(on, list) else [on]
            right_on = left_on
        else:
            if isinstance(left_on, str):
                left_on = [left_on]
            if isinstance(right_on, str):
                right_on = [right_on]

        # Extract key arrays (eager — needs numpy for dict-based matching)
        left_keys = np.column_stack([np.asarray(self._get_column_data(k)) for k in left_on])
        right_keys = np.column_stack([np.asarray(right._get_column_data(k)) for k in right_on])

        # Build index pairs (eager)
        left_idx, right_idx = _merge_indices(left_keys, right_keys, how)

        # Determine output columns and suffixes
        left_suffix, right_suffix = suffixes
        right_non_key = [c for c in right._column_order if c not in right_on]
        left_non_key = [c for c in self._column_order if c not in left_on]

        result_data = {}

        # Columns from left
        for col in self._column_order:
            out_col = col
            if col not in left_on and col in right_non_key:
                out_col = col + left_suffix
            if col in self._column_to_block:
                result_data[out_col] = self._get_column_data(col)[left_idx]
            elif col in self._object_data:
                result_data[out_col] = self._object_data[col][left_idx]

        # Columns from right (skip shared key)
        for col in right._column_order:
            if col in right_on and col in left_on:
                continue
            out_col = col
            if col not in right_on and col in left_non_key:
                out_col = col + right_suffix
            if col in right._column_to_block:
                result_data[out_col] = right._get_column_data(col)[right_idx]
            elif col in right._object_data:
                result_data[out_col] = right._object_data[col][right_idx]

        return DataFrame(result_data)

    # ========================================
    # GroupBy
    # ========================================

    def groupby(self, by):
        """Group by column. Returns a DataFrameGroupBy object.

        Group discovery is eager (computes unique groups and segment IDs).
        All aggregation methods use jax.ops.segment_* and are JIT+grad compatible.
        """
        if isinstance(by, str):
            by = [by]
        # Extract key column(s) — eager jnp for group discovery (stays on GPU)
        key_col = by[0]
        dtype, idx = self._column_to_block[key_col]
        key_data = self._dtype_blocks[dtype][:, idx]
        # Discover groups eagerly (jnp.unique works eagerly on GPU)
        unique_keys, inverse = jnp.unique(key_data, return_inverse=True)
        segment_ids = inverse.astype(jnp.int32)
        num_groups = len(unique_keys)
        group_keys = unique_keys
        # Value columns = all columns except group key(s)
        val_cols = [c for c in self._column_order if c not in by]
        return DataFrameGroupBy(
            df=self,
            by=by,
            segment_ids=segment_ids,
            num_groups=num_groups,
            group_keys=group_keys,
            val_cols=val_cols,
        )

    def rolling(self, window, min_periods: int | None = None, on: str | None = None):
        """Return a Rolling object for rolling window calculations.

        Args:
            window: int for fixed-size window (JIT-compatible),
                    or str like '7D' for time-based window (not JIT-compatible)
            min_periods: minimum number of valid observations
            on: column to use for time-based windows (must be datetime64)

        Fixed-size windows are JIT-compatible.
        Time-based windows are NOT JIT-compatible (variable window sizes).
        """
        if isinstance(window, str):
            return TimeRolling(self, window=window, on=on)
        if min_periods is None:
            min_periods = window
        return Rolling(self, window=window, min_periods=min_periods)

    def expanding(self, min_periods: int = 1):
        """Return an Expanding object for expanding window calculations (JIT-compatible)."""
        return Expanding(self, min_periods=min_periods)

    def ewm(self, alpha=None, span=None, com=None, halflife=None, min_periods: int = 0):
        """Return an EWM object for exponentially weighted calculations (JIT-compatible).

        Exactly one of alpha, span, com, halflife must be specified.
        """
        if alpha is None:
            if span is not None:
                alpha = 2.0 / (span + 1.0)
            elif com is not None:
                alpha = 1.0 / (com + 1.0)
            elif halflife is not None:
                alpha = 1.0 - jnp.exp(-jnp.log(2.0) / halflife)
            else:
                raise ValueError("Must specify one of alpha, span, com, halflife")
        return EWM(self, alpha=float(alpha), min_periods=min_periods)

    # ========================================
    # Pandas interop & copy
    # ========================================

    def to_pandas(self):
        """Convert to pandas DataFrame."""
        import pandas as pd

        data = {}
        for col in self._column_order:
            if col in self._column_to_block:
                dtype, idx = self._column_to_block[col]
                data[col] = np.asarray(self._dtype_blocks[dtype][:, idx])
            elif col in self._object_data:
                data[col] = self._object_data[col]
        return pd.DataFrame(data, index=np.asarray(self._index))

    @classmethod
    def from_pandas(cls, pdf):
        """Create DataFrame from a pandas DataFrame."""
        import pandas as pd

        data = {col: pdf[col].values.tolist() for col in pdf.columns}
        index = pdf.index.values if not isinstance(pdf.index, pd.RangeIndex) else None
        return cls(data, index=index)

    def to_numpy(self):
        """Return numeric values as a plain numpy array."""
        return np.asarray(self.values)

    def copy(self):
        """Return a deep copy of this DataFrame."""
        new_blocks = {dt: block.copy() for dt, block in self._dtype_blocks.items()}
        new_col_to_block = dict(self._column_to_block)
        new_object = {k: list(v) for k, v in self._object_data.items()}
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=new_object,
            index=self._index.copy(),
            column_order=list(self._column_order),
        )

    def melt(
        self,
        id_vars=None,
        value_vars=None,
        var_name="variable",
        value_name="value",
    ):
        """Unpivot from wide to long format.

        Structure discovery (which columns become rows) is eager.
        Data gathering uses array operations.

        Args:
            id_vars: Column(s) to keep as identifier
            value_vars: Column(s) to unpivot (default: all non-id columns)
            var_name: Name for the variable column
            value_name: Name for the value column
        """
        if id_vars is None:
            id_vars = []
        elif isinstance(id_vars, str):
            id_vars = [id_vars]
        if value_vars is None:
            value_vars = [c for c in self._column_order if c not in id_vars]
        elif isinstance(value_vars, str):
            value_vars = [value_vars]

        n_rows = len(self._index)
        n_val_cols = len(value_vars)

        result_data = {}

        # Repeat id columns n_val_cols times
        for col in id_vars:
            col_data = self._get_column_data(col)
            result_data[col] = jnp.tile(col_data, n_val_cols)

        # Variable column (object — column names repeated)
        var_col = np.repeat(value_vars, n_rows)
        result_data[var_name] = var_col

        # Value column — stack all value columns
        val_arrays = []
        for col in value_vars:
            val_arrays.append(self._get_column_data(col))
        result_data[value_name] = jnp.concatenate(val_arrays)

        return DataFrame(result_data)

    def interpolate(self, method="linear"):
        """Interpolate NaN values. JIT-compatible for 'linear' method.

        Uses linear interpolation between nearest valid observations.
        """
        if method != "linear":
            raise ValueError(f"Only 'linear' interpolation supported, got {method!r}")

        def _interp_block(block):
            n = block.shape[0]
            idx = jnp.arange(n, dtype=jnp.float32)
            result = block.copy()
            for c in range(block.shape[1]):
                col = block[:, c]
                valid = jnp.isfinite(col)
                # If all valid or all NaN, skip
                n_valid = jnp.sum(valid)
                # Use jnp.interp for linear interpolation
                valid_idx = jnp.where(valid, idx, jnp.nan)
                valid_vals = jnp.where(valid, col, jnp.nan)
                # Compact valid values
                order = jnp.argsort(~valid)  # valid first
                sorted_idx = valid_idx[order][:n_valid]
                sorted_vals = valid_vals[order][:n_valid]
                interped = jnp.interp(idx, sorted_idx, sorted_vals)
                result = result.at[:, c].set(jnp.where(n_valid > 0, interped, col))
            return result

        return self._apply_blockwise(_interp_block)

    def pivot_table(self, values=None, index=None, columns=None, aggfunc="mean"):
        """Create pivot table. Delegates to pandas (not JIT-compatible)."""
        import pandas as pd

        pdf = self.to_pandas()
        result = pdf.pivot_table(values=values, index=index, columns=columns, aggfunc=aggfunc)
        if isinstance(result, pd.DataFrame):
            return DataFrame.from_pandas(result.reset_index())
        return result

    def to_csv(self, path, index=False, **kwargs):
        """Write DataFrame to CSV file. Not JIT-compatible (I/O)."""
        self.to_pandas().to_csv(path, index=index, **kwargs)

    # ========================================
    # Time series operations
    # ========================================

    def shift(self, periods: int = 1, fill_value=None):
        """
        Shift data by n periods (JIT-compatible, differentiable).

        Args:
            periods: Number of periods to shift (positive for forward, negative for backward)
            fill_value: Value to use for padded entries (default: jnp.nan)

        Returns:
            DataFrame with shifted values

        Examples:
            >>> df.shift(1)  # Shift forward by 1 (introduce lag)
            >>> df.shift(-1)  # Shift backward by 1 (lead)
        """
        if periods == 0:
            return self

        if fill_value is None:
            fill_value = jnp.nan

        n_rows = len(self._index)

        def _shift_block(block):
            if periods > 0:
                if periods >= n_rows:
                    return jnp.full_like(block, fill_value)
                pad = jnp.full((periods, block.shape[1]), fill_value, dtype=block.dtype)
                return jnp.concatenate([pad, block[:-periods]], axis=0)
            else:
                ap = abs(periods)
                if ap >= n_rows:
                    return jnp.full_like(block, fill_value)
                pad = jnp.full((ap, block.shape[1]), fill_value, dtype=block.dtype)
                return jnp.concatenate([block[ap:], pad], axis=0)

        return self._apply_blockwise(_shift_block)

    def diff(self, periods: int = 1):
        """
        Calculate first discrete difference (JIT-compatible, differentiable).

        Computes self - self.shift(periods). NaN where shift introduces gaps.
        """
        shifted = self.shift(periods)
        col_arrays = {}
        for col in self._column_order:
            if col in self._column_to_block:
                col_arrays[col] = self._get_column_data(col) - shifted._get_column_data(col)
        return DataFrame._from_column_arrays(
            col_arrays, self._column_order, self._index.copy(), self._object_data.copy()
        )

    def pct_change(self, periods: int = 1):
        """
        Calculate percentage change (JIT-compatible, differentiable).

        Computes (self - self.shift(periods)) / self.shift(periods).
        """
        shifted = self.shift(periods)
        col_arrays = {}
        for col in self._column_order:
            if col in self._column_to_block:
                s = shifted._get_column_data(col)
                col_arrays[col] = (self._get_column_data(col) - s) / s
        return DataFrame._from_column_arrays(
            col_arrays, self._column_order, self._index.copy(), self._object_data.copy()
        )

    def where(self, condition, fill_value):
        """
        Replace values where condition is False (JIT-compatible with dtype preservation).

        This is the JIT-friendly alternative to boolean indexing.

        Args:
            condition: Boolean array or DataFrame
            fill_value: Value to use where condition is False

        Returns:
            DataFrame with same shape

        Examples:
            >>> df.where(df > 10, fill_value=0)  # Replace values <= 10 with 0
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns for where operation")

        if isinstance(condition, DataFrame):
            # Apply where column by column, preserving dtype blocks
            new_dtype_blocks = {}
            for dtype, block in self._dtype_blocks.items():
                # Apply where to all columns in this block
                new_block_cols = []
                for col in self._column_order:
                    if col in self._column_to_block:
                        col_dtype, col_idx = self._column_to_block[col]
                        if col_dtype == dtype and col in condition._column_to_block:
                            col_data = block[:, col_idx]
                            cond_data = condition._get_column_data(col)
                            new_col = jnp.where(cond_data, col_data, fill_value)
                            new_block_cols.append(new_col.reshape(-1, 1))

                if new_block_cols:
                    new_dtype_blocks[dtype] = jnp.concatenate(new_block_cols, axis=1)

            return DataFrame._from_parts(
                dtype_blocks=new_dtype_blocks,
                column_to_block=self._column_to_block.copy(),
                object_data=self._object_data.copy(),
                index=self._index,
                column_order=self._column_order,
            )
        else:
            # Scalar or array condition - apply to all columns
            mask = jnp.asarray(condition)
            new_dtype_blocks = {}
            for dtype, block in self._dtype_blocks.items():
                new_dtype_blocks[dtype] = jnp.where(mask[:, None], block, fill_value)

            return DataFrame._from_parts(
                dtype_blocks=new_dtype_blocks,
                column_to_block=self._column_to_block.copy(),
                object_data=self._object_data.copy(),
                index=self._index,
                column_order=self._column_order,
            )

    def mask(self, condition, fill_value):
        """
        Replace values where condition is True (JIT-compatible).

        This is the inverse of where().

        Args:
            condition: Boolean array or DataFrame
            fill_value: Value to use where condition is True

        Returns:
            DataFrame with same shape

        Examples:
            >>> df.mask(df > 10, fill_value=0)  # Replace values > 10 with 0
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns for mask operation")

        if isinstance(condition, DataFrame):
            # Apply mask column by column
            result_data = {}
            for col in self._column_order:
                if col in self._column_to_block and col in condition._column_to_block:
                    col_data = self._get_column_data(col)
                    cond_data = condition._get_column_data(col)
                    # Convert condition to boolean (handles 0/1 as int/float from comparisons)
                    cond_bool = cond_data.astype(bool)
                    # mask is inverse of where: replace where condition is True
                    result_data[col] = jnp.where(~cond_bool, col_data, fill_value)
            return DataFrame(result_data, index=self._index.copy())
        else:
            # Scalar or array condition - apply to all columns
            mask = jnp.asarray(condition).astype(bool)
            result_data = {}
            for col in self._column_order:
                if col in self._column_to_block:
                    col_data = self._get_column_data(col)
                    result_data[col] = jnp.where(~mask, col_data, fill_value)
            return DataFrame(result_data, index=self._index.copy())

    def clip(self, lower=None, upper=None):
        """
        Clip values to range [lower, upper] (JIT-compatible, differentiable).

        Args:
            lower: Minimum value (None for no lower bound)
            upper: Maximum value (None for no upper bound)

        Returns:
            DataFrame with clipped values

        Examples:
            >>> df.clip(0, 10)  # Clip to [0, 10]
            >>> df.clip(lower=0)  # Clip minimum to 0
        """

        def _clip_block(block):
            result = block
            if lower is not None:
                result = jnp.maximum(result, lower)
            if upper is not None:
                result = jnp.minimum(result, upper)
            return result

        return self._apply_blockwise(_clip_block)

    # ========================================
    # Indexing and selection
    # ========================================

    @property
    def iloc(self):
        """Integer-location based indexing (JIT-compatible)."""
        return _ILocIndexer(self)

    def head(self, n: int = 5):
        """
        Return first n rows (JIT-compatible).

        Args:
            n: Number of rows to return (must be static for JIT)

        Returns:
            DataFrame with first n rows
        """
        # Slice dtype blocks
        new_dtype_blocks = {}
        for dtype, block in self._dtype_blocks.items():
            new_dtype_blocks[dtype] = block[:n]

        # Slice object data
        new_object_data = {}
        for col, arr in self._object_data.items():
            new_object_data[col] = arr[:n]

        return DataFrame._from_parts(
            dtype_blocks=new_dtype_blocks,
            column_to_block=self._column_to_block.copy(),
            object_data=new_object_data,
            index=self._index[:n],
            column_order=self._column_order,
        )

    def tail(self, n: int = 5):
        """
        Return last n rows (JIT-compatible).

        Args:
            n: Number of rows to return (must be static for JIT)

        Returns:
            DataFrame with last n rows
        """
        # Handle n=0 edge case (return empty DataFrame)
        if n == 0:
            new_dtype_blocks = {}
            for dtype, block in self._dtype_blocks.items():
                new_dtype_blocks[dtype] = block[:0]

            new_object_data = {}
            for col, arr in self._object_data.items():
                new_object_data[col] = arr[:0]

            return DataFrame._from_parts(
                dtype_blocks=new_dtype_blocks,
                column_to_block=self._column_to_block.copy(),
                object_data=new_object_data,
                index=self._index[:0],
                column_order=self._column_order,
            )

        # Slice dtype blocks from the end
        new_dtype_blocks = {}
        for dtype, block in self._dtype_blocks.items():
            new_dtype_blocks[dtype] = block[-n:]

        # Slice object data from the end
        new_object_data = {}
        for col, arr in self._object_data.items():
            new_object_data[col] = arr[-n:]

        return DataFrame._from_parts(
            dtype_blocks=new_dtype_blocks,
            column_to_block=self._column_to_block.copy(),
            object_data=new_object_data,
            index=self._index[-n:],
            column_order=self._column_order,
        )

    def __getattr__(self, name):
        """
        Attribute access for column selection (df.col).

        This enables df.x to access column 'x'.
        """
        # Avoid infinite recursion for internal attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if it's a column
        if name in self._column_order:
            return self[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ========================================
    # Comparison operations
    # ========================================

    def __gt__(self, other) -> "DataFrame":
        """Greater than comparison (JIT-compatible, returns boolean)."""
        return self._apply_comparison(other, jnp.greater, "greater than")

    def __ge__(self, other) -> "DataFrame":
        """Greater than or equal comparison (JIT-compatible, returns boolean)."""
        return self._apply_comparison(other, jnp.greater_equal, "greater than or equal")

    def __lt__(self, other) -> "DataFrame":
        """Less than comparison (JIT-compatible, returns boolean)."""
        return self._apply_comparison(other, jnp.less, "less than")

    def __le__(self, other) -> "DataFrame":
        """Less than or equal comparison (JIT-compatible, returns boolean)."""
        return self._apply_comparison(other, jnp.less_equal, "less than or equal")

    def __eq__(self, other) -> "DataFrame":
        """Equality comparison (JIT-compatible, returns boolean)."""
        return self._apply_comparison(other, jnp.equal, "equality")

    def __ne__(self, other) -> "DataFrame":
        """Not equal comparison (JIT-compatible, returns boolean)."""
        return self._apply_comparison(other, jnp.not_equal, "not equal")

    # ========================================
    # Logical operations
    # ========================================

    def __and__(self, other) -> "DataFrame":
        """Logical AND (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.bitwise_and, "logical AND")

    def __or__(self, other) -> "DataFrame":
        """Logical OR (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.bitwise_or, "logical OR")

    def __invert__(self) -> "DataFrame":
        """Logical NOT (JIT-compatible)."""
        if not self._dtype_blocks:
            raise ValueError("No numeric columns for logical NOT")

        # Apply inversion to each block independently
        result_data = {}
        for col in self._column_order:
            if col in self._column_to_block:
                col_data = self._get_column_data(col)
                # Convert to boolean if needed (handles 0/1 as ints/floats)
                bool_data = col_data.astype(bool)
                result_data[col] = ~bool_data

        return DataFrame(result_data, index=self._index.copy())

    # ========================================
    # Internal helpers
    # ========================================

    def _get_column_data(self, col_name: str) -> jnp.ndarray:
        """Get data for a single column as 1D array."""
        if col_name in self._column_to_block:
            dtype, idx = self._column_to_block[col_name]
            return self._dtype_blocks[dtype][:, idx]
        elif col_name in self._object_data:
            return self._object_data[col_name]
        else:
            raise KeyError(f"Column '{col_name}' not found")

    def _apply_comparison(self, other, op, op_name: str = "comparison"):
        """
        Apply comparison operation that returns boolean results (JIT-compatible).

        Args:
            other: scalar, DataFrame, or array
            op: Comparison operation function (e.g., jnp.greater)
            op_name: Name of operation for error messages

        Returns:
            New DataFrame with boolean results
        """
        if isinstance(other, (int, float, bool, np.number)):
            # Scalar comparison - apply to each dtype block
            new_dtype_blocks = {}
            bool_dtype = np.dtype(bool)
            bool_cols = []

            for col in self._column_order:
                if col in self._column_to_block:
                    col_data = self._get_column_data(col)
                    bool_result = op(col_data, other)  # Returns bool
                    bool_cols.append(bool_result.reshape(-1, 1))

            if bool_cols:
                new_dtype_blocks[bool_dtype] = jnp.concatenate(bool_cols, axis=1)

            # Update column_to_block mapping - all columns now have bool dtype
            new_column_to_block = {}
            numeric_cols = [c for c in self._column_order if c in self._column_to_block]
            for idx, col in enumerate(numeric_cols):
                new_column_to_block[col] = (bool_dtype, idx)

            return DataFrame._from_parts(
                dtype_blocks=new_dtype_blocks,
                column_to_block=new_column_to_block,
                object_data={},
                index=self._index,
                column_order=self._column_order,
            )

        elif isinstance(other, DataFrame):
            # DataFrame comparison
            if self._column_order != other._column_order:
                raise ValueError(f"Column mismatch for {op_name}")

            new_dtype_blocks = {}
            bool_dtype = np.dtype(bool)
            bool_cols = []
            col_names = []

            for col in self._column_order:
                if col in self._column_to_block and col in other._column_to_block:
                    left_data = self._get_column_data(col)
                    right_data = other._get_column_data(col)
                    bool_result = op(left_data, right_data)  # Returns bool
                    bool_cols.append(bool_result.reshape(-1, 1))
                    col_names.append(col)

            if bool_cols:
                new_dtype_blocks[bool_dtype] = jnp.concatenate(bool_cols, axis=1)

            new_column_to_block = {col: (bool_dtype, idx) for idx, col in enumerate(col_names)}

            return DataFrame._from_parts(
                dtype_blocks=new_dtype_blocks,
                column_to_block=new_column_to_block,
                object_data={},
                index=self._index,
                column_order=tuple(col_names),
            )

        elif isinstance(other, (np.ndarray, jnp.ndarray)):
            # Array comparison
            other_arr = jnp.asarray(other)
            new_dtype_blocks = {}
            bool_dtype = np.dtype(bool)
            bool_cols = []

            for col in self._column_order:
                if col in self._column_to_block:
                    col_data = self._get_column_data(col)
                    bool_result = op(col_data, other_arr)  # Returns bool
                    bool_cols.append(bool_result.reshape(-1, 1))

            if bool_cols:
                new_dtype_blocks[bool_dtype] = jnp.concatenate(bool_cols, axis=1)

            new_column_to_block = {}
            numeric_cols = [c for c in self._column_order if c in self._column_to_block]
            for idx, col in enumerate(numeric_cols):
                new_column_to_block[col] = (bool_dtype, idx)

            return DataFrame._from_parts(
                dtype_blocks=new_dtype_blocks,
                column_to_block=new_column_to_block,
                object_data={},
                index=self._index,
                column_order=self._column_order,
            )

        else:
            raise TypeError(f"Unsupported operand type for {op_name}: {type(other)}")

    def _apply_elementwise(self, other, op, op_name: str = "operation"):
        """
        Apply element-wise operation with type promotion.

        Args:
            other: scalar, DataFrame, Series, or array
            op: Binary operation function (e.g., jnp.add)
            op_name: Name of operation for error messages

        Returns:
            New DataFrame with result
        """
        if isinstance(other, (int, float, bool, np.number)):
            # Scalar operation: apply to each block independently
            new_blocks = {}
            dtype_map = {}  # old_dtype -> new_dtype
            for dtype, block in self._dtype_blocks.items():
                result = op(block, other)
                new_dtype = np.dtype(result.dtype)
                new_blocks[new_dtype] = result
                dtype_map[dtype] = new_dtype

            if all(k == v for k, v in dtype_map.items()):
                new_column_to_block = self._column_to_block
            else:
                new_column_to_block = {
                    col: (dtype_map[old_dtype], idx)
                    for col, (old_dtype, idx) in self._column_to_block.items()
                }

            return DataFrame._from_parts(
                dtype_blocks=new_blocks,
                column_to_block=new_column_to_block,
                object_data=self._object_data,
                index=self._index,
                column_order=self._column_order,
            )

        elif isinstance(other, DataFrame):
            # DataFrame operation: align and promote types
            if self._column_order != other._column_order:
                raise ValueError(f"Column mismatch for {op_name}")

            # Fast path: same block structure — operate block-by-block
            if self._column_to_block == other._column_to_block:
                new_blocks = {}
                dtype_map = {}
                for dtype, block in self._dtype_blocks.items():
                    result = op(block, other._dtype_blocks[dtype])
                    new_dtype = np.dtype(result.dtype)
                    new_blocks[new_dtype] = result
                    dtype_map[dtype] = new_dtype

                if all(k == v for k, v in dtype_map.items()):
                    new_col_to_block = self._column_to_block
                else:
                    new_col_to_block = {
                        col: (dtype_map[old_dtype], idx)
                        for col, (old_dtype, idx) in self._column_to_block.items()
                    }
                return DataFrame._from_parts(
                    dtype_blocks=new_blocks,
                    column_to_block=new_col_to_block,
                    object_data=self._object_data,
                    index=self._index,
                    column_order=self._column_order,
                )

            # Slow path: column-by-column with repack
            col_arrays = {}
            for col in self._column_order:
                if col in self._column_to_block and col in other._column_to_block:
                    left_data = self._get_column_data(col)
                    right_data = other._get_column_data(col)
                    left_dtype, _ = self._column_to_block[col]
                    right_dtype, _ = other._column_to_block[col]
                    result_dtype = jnp.result_type(left_dtype, right_dtype)
                    col_arrays[col] = op(left_data, right_data).astype(result_dtype)

            return DataFrame._from_column_arrays(
                col_arrays, self._column_order, self._index.copy(), self._object_data.copy()
            )

        elif type(other).__name__ == "Series":
            # Series broadcasts across rows (like pandas)
            # Each column gets the corresponding value from the Series
            # Build index lookup eagerly (structure discovery)
            if hasattr(other, "_index"):
                series_lookup = {name: i for i, name in enumerate(other._index)}
            else:
                series_lookup = None

            col_arrays = {}
            for col in self._column_order:
                if col in self._column_to_block:
                    col_data = self._get_column_data(col)
                    dtype, _ = self._column_to_block[col]

                    if series_lookup is not None:
                        if col not in series_lookup:
                            continue
                        series_value = other._data[series_lookup[col]]
                    else:
                        numeric_cols = [c for c in self._column_order if c in self._column_to_block]
                        col_idx = numeric_cols.index(col)
                        series_value = other._data[col_idx]

                    result_dtype = jnp.result_type(dtype, other._data.dtype)
                    col_arrays[col] = op(col_data, series_value).astype(result_dtype)

            return DataFrame._from_column_arrays(
                col_arrays, self._column_order, self._index.copy(), self._object_data.copy()
            )

        elif isinstance(other, (np.ndarray, jnp.ndarray)):
            # Array operation
            other_arr = jnp.asarray(other)

            if other_arr.ndim == 1 and len(other_arr) == len(self._column_order):
                # 1D array matching columns: broadcast across columns
                col_arrays = {}
                numeric_cols = [c for c in self._column_order if c in self._column_to_block]
                for idx, col in enumerate(numeric_cols):
                    col_data = self._get_column_data(col)
                    dtype, _ = self._column_to_block[col]
                    result_dtype = jnp.result_type(dtype, other_arr.dtype)
                    col_arrays[col] = op(col_data, other_arr[idx]).astype(result_dtype)
                return DataFrame._from_column_arrays(
                    col_arrays, self._column_order, self._index.copy(), self._object_data.copy()
                )
            else:
                # Array broadcasts element-wise across all blocks
                new_blocks = {}
                new_col_to_block = {}
                for dtype, block in self._dtype_blocks.items():
                    result_dtype = jnp.result_type(dtype, other_arr.dtype)
                    new_blocks[np.dtype(result_dtype)] = op(block, other_arr)
                for col, (old_dtype, idx) in self._column_to_block.items():
                    result_dtype = jnp.result_type(old_dtype, other_arr.dtype)
                    new_col_to_block[col] = (np.dtype(result_dtype), idx)
                return DataFrame._from_parts(
                    dtype_blocks=new_blocks,
                    column_to_block=new_col_to_block,
                    object_data=self._object_data.copy(),
                    index=self._index.copy(),
                    column_order=self._column_order,
                )

        else:
            raise TypeError(f"Unsupported operand type for {op_name}: {type(other)}")

    @classmethod
    def _from_parts(
        cls,
        dtype_blocks,
        column_to_block,
        object_data,
        index,
        column_order,
    ):
        """Internal constructor from pre-separated parts (dtype blocks)."""
        df = cls.__new__(cls)
        df._dtype_blocks = dtype_blocks
        df._column_to_block = column_to_block
        df._object_data = object_data
        df._index = index
        df._column_order = column_order
        return df

    @classmethod
    def _from_column_arrays(cls, col_arrays, column_order, index, object_data=None):
        """Build DataFrame from {col: 1d_array} dict using _from_parts (JIT-safe)."""
        # Group columns by dtype
        dtype_groups: dict[np.dtype, list[tuple[str, jnp.ndarray]]] = {}
        for col in column_order:
            if col in col_arrays:
                arr = col_arrays[col]
                dt = np.dtype(arr.dtype)
                if dt not in dtype_groups:
                    dtype_groups[dt] = []
                dtype_groups[dt].append((col, arr))

        dtype_blocks = {}
        column_to_block = {}
        for dt, pairs in dtype_groups.items():
            arrays = [a.reshape(-1, 1) for _, a in pairs]
            dtype_blocks[dt] = jnp.concatenate(arrays, axis=1)
            for idx, (col, _) in enumerate(pairs):
                column_to_block[col] = (dt, idx)

        return cls._from_parts(
            dtype_blocks=dtype_blocks,
            column_to_block=column_to_block,
            object_data=object_data if object_data is not None else {},
            index=index,
            column_order=column_order,
        )


class _ILocIndexer:
    """
    Integer-location based indexing for DataFrame.

    This class enables df.iloc[...] syntax.
    """

    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key):
        """
        Integer-location based indexing.

        Args:
            key: Integer, slice, or array of integers

        Returns:
            DataFrame or Series depending on selection

        Examples:
            >>> df.iloc[0]  # First row as Series
            >>> df.iloc[0:5]  # First 5 rows
            >>> df.iloc[[0, 2, 4]]  # Rows 0, 2, 4
            >>> df.iloc[:, 0]  # First column
        """
        if isinstance(key, tuple):
            # Two-dimensional indexing: df.iloc[rows, cols]
            row_key, col_key = key

            # Index columns first to determine selected columns
            if isinstance(col_key, int):
                # Single column selection -> Series
                col_name = self._df._column_order[col_key]
                if col_name in self._df._column_to_block:
                    data = self._df._get_column_data(col_name)[row_key]
                    return Series(data, index=self._df._index[row_key], name=col_name)
                else:
                    # Object column
                    data = self._df._object_data[col_name][row_key]
                    return Series(data, index=self._df._index[row_key], name=col_name)
            else:
                # Multiple column selection -> DataFrame
                if isinstance(col_key, slice):
                    selected_cols = self._df._column_order[col_key]
                else:
                    selected_cols = tuple(self._df._column_order[i] for i in col_key)

                # Build result by selecting and indexing each column
                result_data = {}
                for col in selected_cols:
                    if col in self._df._column_to_block:
                        result_data[col] = self._df._get_column_data(col)[row_key]
                    elif col in self._df._object_data:
                        result_data[col] = self._df._object_data[col][row_key]

                # Handle different key types for index
                if isinstance(row_key, slice):
                    new_index = self._df._index[row_key]
                elif isinstance(row_key, int):
                    new_index = np.array([self._df._index[row_key]])
                else:
                    new_index = self._df._index[jnp.asarray(row_key)]

                return DataFrame(result_data, index=new_index)
        else:
            # One-dimensional indexing: df.iloc[rows]
            if isinstance(key, int):
                # Single row -> Series
                # Collect all column data for this row
                data_dict = {}
                for col in self._df._column_order:
                    if col in self._df._column_to_block:
                        data_dict[col] = self._df._get_column_data(col)[key]
                    elif col in self._df._object_data:
                        data_dict[col] = self._df._object_data[col][key]

                # Convert to array for Series (only numeric columns)
                if self._df._dtype_blocks:
                    # Build array in column order from numeric columns only
                    numeric_data = []
                    for col in self._df._column_order:
                        if col in self._df._column_to_block:
                            numeric_data.append(data_dict[col])
                    data = jnp.array(numeric_data)
                    idx_name = self._df._index[key]
                    return Series(
                        data,
                        index=np.array(self._df._numeric_cols),
                        name=idx_name,
                    )
                else:
                    # Only object columns
                    raise NotImplementedError(
                        "Single row selection with only object columns not yet supported"
                    )
            else:
                # Multiple rows -> DataFrame
                # Index all dtype blocks
                new_dtype_blocks = {}
                for dtype, block in self._df._dtype_blocks.items():
                    new_dtype_blocks[dtype] = block[key]

                # Index object data
                new_object_data = {}
                for col, arr in self._df._object_data.items():
                    new_object_data[col] = arr[key]

                # Handle different key types for index
                if isinstance(key, slice):
                    new_index = self._df._index[key]
                else:
                    new_index = self._df._index[jnp.asarray(key)]

                return DataFrame._from_parts(
                    dtype_blocks=new_dtype_blocks,
                    column_to_block=self._df._column_to_block.copy(),
                    object_data=new_object_data,
                    index=new_index,
                    column_order=self._df._column_order,
                )


@dataclass
class Series:
    """
    A single column (1D array).

    Simpler than DataFrame but similar interface.
    """

    _data: jnp.ndarray | np.ndarray
    _index: np.ndarray
    _name: str | None = None

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

    @property
    def index(self):
        """Return series index."""
        return self._index

    def sum(self):
        """Sum of all values (JIT-compatible)."""
        return jnp.sum(self._data)

    def mean(self):
        """Mean of all values (JIT-compatible)."""
        return jnp.mean(self._data)

    def abs(self):
        """Absolute value."""
        return Series(jnp.abs(self._data), index=self._index, name=self._name)

    def shift(self, periods: int = 1, fill_value=None):
        """Shift data by n periods (JIT-compatible)."""
        if fill_value is None:
            fill_value = jnp.nan
        n = len(self._data)
        if periods == 0:
            return Series(self._data, index=self._index, name=self._name)
        if abs(periods) >= n:
            return Series(jnp.full_like(self._data, fill_value), index=self._index, name=self._name)
        if periods > 0:
            pad = jnp.full((periods,), fill_value, dtype=self._data.dtype)
            data = jnp.concatenate([pad, self._data[:-periods]])
        else:
            ap = abs(periods)
            pad = jnp.full((ap,), fill_value, dtype=self._data.dtype)
            data = jnp.concatenate([self._data[ap:], pad])
        return Series(data, index=self._index, name=self._name)

    def diff(self, periods: int = 1):
        """First discrete difference (JIT-compatible)."""
        shifted = self.shift(periods)
        return Series(self._data - shifted._data, index=self._index, name=self._name)

    def pct_change(self, periods: int = 1):
        """Percentage change (JIT-compatible)."""
        shifted = self.shift(periods)
        result = (self._data - shifted._data) / shifted._data
        return Series(result, index=self._index, name=self._name)

    def pipe(self, func, *args, **kwargs):
        """Apply a function to the Series. JIT-compatible if func is."""
        return func(self, *args, **kwargs)

    def between(self, left, right, inclusive="both"):
        """Boolean mask for values between left and right (JIT-compatible)."""
        d = self._data
        if inclusive == "both":
            mask = (d >= left) & (d <= right)
        elif inclusive == "left":
            mask = (d >= left) & (d < right)
        elif inclusive == "right":
            mask = (d > left) & (d <= right)
        else:
            mask = (d > left) & (d < right)
        return Series(mask, index=self._index, name=self._name)

    def map(self, func):
        """Apply a function element-wise. JIT-compatible if func is JAX-compatible."""
        return Series(func(self._data), index=self._index, name=self._name)

    def replace(self, to_replace, value):
        """Replace values. JIT-compatible for scalar replacements."""
        return Series(
            jnp.where(self._data == to_replace, value, self._data),
            index=self._index,
            name=self._name,
        )

    def value_counts(self, sort=True):
        """Count occurrences of each unique value. Not JIT-compatible (eager)."""
        unique_vals, counts = jnp.unique(self._data, return_counts=True)
        if sort:
            order = jnp.argsort(-counts)
            unique_vals = unique_vals[order]
            counts = counts[order]
        return Series(counts, index=np.asarray(unique_vals), name=self._name)

    # Arithmetic operators
    def _binop(self, other, op):
        other_data = other._data if isinstance(other, Series) else other
        return Series(op(self._data, other_data), index=self._index, name=self._name)

    def __add__(self, other):
        return self._binop(other, jnp.add)

    def __sub__(self, other):
        return self._binop(other, jnp.subtract)

    def __mul__(self, other):
        return self._binop(other, jnp.multiply)

    def __truediv__(self, other):
        return self._binop(other, jnp.true_divide)

    def __floordiv__(self, other):
        return self._binop(other, jnp.floor_divide)

    def __mod__(self, other):
        return self._binop(other, jnp.mod)

    def __pow__(self, other):
        return self._binop(other, jnp.power)

    def __neg__(self):
        return Series(jnp.negative(self._data), index=self._index, name=self._name)

    # Reverse operators
    def __radd__(self, other):
        return self._binop(other, lambda a, b: jnp.add(b, a))

    def __rsub__(self, other):
        return self._binop(other, lambda a, b: jnp.subtract(b, a))

    def __rmul__(self, other):
        return self._binop(other, lambda a, b: jnp.multiply(b, a))

    def __rtruediv__(self, other):
        return self._binop(other, lambda a, b: jnp.true_divide(b, a))

    def __rfloordiv__(self, other):
        return self._binop(other, lambda a, b: jnp.floor_divide(b, a))

    def __rmod__(self, other):
        return self._binop(other, lambda a, b: jnp.mod(b, a))

    def __rpow__(self, other):
        return self._binop(other, lambda a, b: jnp.power(b, a))

    # Comparison operators
    def __gt__(self, other):
        return self._binop(other, jnp.greater)

    def __ge__(self, other):
        return self._binop(other, jnp.greater_equal)

    def __lt__(self, other):
        return self._binop(other, jnp.less)

    def __le__(self, other):
        return self._binop(other, jnp.less_equal)

    def __eq__(self, other):
        return self._binop(other, jnp.equal)

    def __ne__(self, other):
        return self._binop(other, jnp.not_equal)

    @property
    def dt(self):
        """Datetime accessor for Series containing datetime64 data."""
        return _DatetimeAccessor(self._data)

    @property
    def str(self):
        """String accessor for Series containing string/object data."""
        return _StringAccessor(self._data)

    def __repr__(self):
        """String representation."""
        lines = [f"Series(name={self._name}, shape={len(self._data)})"]
        n_show = min(5, len(self._data))
        for i in range(n_show):
            lines.append(f"{self._index[i]}  {self._data[i]}")
        if len(self._data) > n_show:
            lines.append("  ...")
        return "\n".join(lines)


class _DatetimeAccessor:
    """Accessor for datetime64 Series data. Extracts date components as numpy arrays."""

    def __init__(self, data):
        # Ensure we have numpy datetime64 data
        self._data = np.asarray(data)
        if not np.issubdtype(self._data.dtype, np.datetime64):
            raise AttributeError("Can only use .dt accessor with datetime data")

    @property
    def year(self):
        return self._data.astype("datetime64[Y]").astype(int) + 1970

    @property
    def month(self):
        return (self._data.astype("datetime64[M]").astype(int) % 12) + 1

    @property
    def day(self):
        return (self._data.astype("datetime64[D]") - self._data.astype("datetime64[M]")).astype(
            int
        ) + 1

    @property
    def hour(self):
        return (self._data.astype("datetime64[h]") - self._data.astype("datetime64[D]")).astype(int)

    @property
    def minute(self):
        return (self._data.astype("datetime64[m]") - self._data.astype("datetime64[h]")).astype(int)

    @property
    def second(self):
        return (self._data.astype("datetime64[s]") - self._data.astype("datetime64[m]")).astype(int)

    @property
    def dayofweek(self):
        # Monday=0, Sunday=6 — datetime64 epoch (1970-01-01) was Thursday (3)
        return (self._data.astype("datetime64[D]").astype(int) - 4) % 7

    @property
    def date(self):
        return self._data.astype("datetime64[D]")


class _StringAccessor:
    """Accessor for string Series data. Operations return numpy arrays."""

    def __init__(self, data):
        self._data = np.asarray(data)

    def lower(self):
        return np.array([s.lower() for s in self._data], dtype=object)

    def upper(self):
        return np.array([s.upper() for s in self._data], dtype=object)

    def strip(self):
        return np.array([s.strip() for s in self._data], dtype=object)

    def lstrip(self):
        return np.array([s.lstrip() for s in self._data], dtype=object)

    def rstrip(self):
        return np.array([s.rstrip() for s in self._data], dtype=object)

    def len(self):
        return np.array([len(s) for s in self._data])

    def contains(self, pat):
        return np.array([pat in s for s in self._data])

    def startswith(self, pat):
        return np.array([s.startswith(pat) for s in self._data])

    def endswith(self, pat):
        return np.array([s.endswith(pat) for s in self._data])

    def replace(self, old, new):
        return np.array([s.replace(old, new) for s in self._data], dtype=object)

    def split(self, sep=None):
        return [s.split(sep) for s in self._data]


# ========================================
# Rolling
# ========================================


def _merge_indices(left_keys, right_keys, how):
    """Compute row index pairs for merge (eager, structure discovery).

    Returns (left_idx, right_idx) — numpy arrays of row indices.
    """
    # Build lookup from right keys to row indices
    right_lookup = {}
    for i, key in enumerate(map(tuple, right_keys)):
        right_lookup.setdefault(key, []).append(i)

    left_indices = []
    right_indices = []

    if how == "inner":
        for i, key in enumerate(map(tuple, left_keys)):
            if key in right_lookup:
                for j in right_lookup[key]:
                    left_indices.append(i)
                    right_indices.append(j)
    elif how == "left":
        for i, key in enumerate(map(tuple, left_keys)):
            if key in right_lookup:
                for j in right_lookup[key]:
                    left_indices.append(i)
                    right_indices.append(j)
            else:
                left_indices.append(i)
                right_indices.append(-1)  # Will need NaN handling
    elif how == "right":
        left_lookup = {}
        for i, key in enumerate(map(tuple, left_keys)):
            left_lookup.setdefault(key, []).append(i)
        for j, key in enumerate(map(tuple, right_keys)):
            if key in left_lookup:
                for i in left_lookup[key]:
                    left_indices.append(i)
                    right_indices.append(j)
            else:
                left_indices.append(-1)
                right_indices.append(j)
    elif how == "outer":
        matched_right = set()
        for i, key in enumerate(map(tuple, left_keys)):
            if key in right_lookup:
                for j in right_lookup[key]:
                    left_indices.append(i)
                    right_indices.append(j)
                    matched_right.add(j)
            else:
                left_indices.append(i)
                right_indices.append(-1)
        for j in range(len(right_keys)):
            if j not in matched_right:
                left_indices.append(-1)
                right_indices.append(j)

    return np.array(left_indices), np.array(right_indices)


def _rolling_window(x, window):
    """Create rolling windows over axis 0 of a 2D array. JIT-compatible.

    Returns array of shape (n_rows, n_cols, window) via gather.
    """
    n = x.shape[0]
    # indices[i] = [i-window+1, ..., i] clipped to [0, n-1]
    idx = jnp.arange(n)[:, None] - jnp.arange(window - 1, -1, -1)[None, :]
    idx = jnp.clip(idx, 0, n - 1)
    # Mask: positions where idx < 0 before clipping (not enough history)
    valid = jnp.arange(n)[:, None] - jnp.arange(window - 1, -1, -1)[None, :] >= 0
    return idx, valid


class Rolling:
    """Rolling window operations. JIT-compatible with fixed window size."""

    def __init__(self, df, window: int, min_periods: int):
        self._df = df
        self._window = window
        self._min_periods = min_periods

    def _apply(self, fn):
        """Apply rolling fn to each dtype block. Returns DataFrame."""
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            idx, valid = _rolling_window(block, self._window)
            # Gather: (n_rows, window, n_cols) then apply fn
            # block[idx] shape: (n_rows, window, n_cols)
            gathered = block[idx]  # (n_rows, n_cols, window) via advanced idx
            # Mask invalid positions with NaN
            mask = valid[:, :, None]  # (n_rows, window, 1)
            gathered = jnp.where(mask, gathered, jnp.nan)
            # Count valid per window
            n_valid = jnp.sum(valid, axis=1)  # (n_rows,)
            result = fn(gathered, n_valid)  # (n_rows, n_cols)
            # NaN where fewer than min_periods valid values
            result = jnp.where(n_valid[:, None] >= self._min_periods, result, jnp.nan)
            new_blocks[result.dtype] = result

        new_col_to_block = {}
        for col, (old_dtype, col_idx) in self._df._column_to_block.items():
            new_dtype = new_blocks[
                next(
                    nd
                    for od, nd in zip(self._df._dtype_blocks.keys(), new_blocks.keys())
                    if od == old_dtype
                )
            ].dtype
            new_col_to_block[col] = (new_dtype, col_idx)

        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._df._object_data,
            index=self._df._index,
            column_order=self._df._column_order,
        )

    def sum(self):
        """Rolling sum."""
        return self._apply(lambda g, nv: jnp.nansum(g, axis=1))

    def mean(self):
        """Rolling mean."""
        return self._apply(lambda g, nv: jnp.nanmean(g, axis=1))

    def var(self, ddof: int = 1):
        """Rolling variance."""
        return self._apply(lambda g, nv: jnp.nanvar(g, axis=1, ddof=ddof))

    def std(self, ddof: int = 1):
        """Rolling standard deviation."""
        return self._apply(lambda g, nv: jnp.nanstd(g, axis=1, ddof=ddof))

    def min(self):
        """Rolling minimum."""
        return self._apply(lambda g, nv: jnp.nanmin(g, axis=1))

    def max(self):
        """Rolling maximum."""
        return self._apply(lambda g, nv: jnp.nanmax(g, axis=1))


class Expanding:
    """Expanding window operations. JIT-compatible — window grows from start."""

    def __init__(self, df, min_periods: int = 1):
        self._df = df
        self._min_periods = min_periods

    def _apply(self, fn):
        """Apply expanding fn to each dtype block. Returns DataFrame."""
        new_blocks = {}
        n_rows = len(self._df._index)
        # For expanding, window = row_index + 1, so row 0 has window 1, row 1 has window 2, etc.
        # Build gather indices: for row i, gather rows [0..i], padding rest with 0
        row_idx = jnp.arange(n_rows)
        # (n_rows, n_rows) index matrix: each row i has [0,1,...,i, 0,0,...,0]
        idx = jnp.broadcast_to(jnp.arange(n_rows)[None, :], (n_rows, n_rows))
        idx = jnp.minimum(idx, row_idx[:, None])
        # valid mask: position j is valid for row i if j <= i
        valid = jnp.arange(n_rows)[None, :] <= row_idx[:, None]  # (n_rows, n_rows)
        n_valid = row_idx + 1  # (n_rows,)

        for dtype, block in self._df._dtype_blocks.items():
            # gathered: (n_rows, n_rows, n_cols)
            gathered = block[idx]
            mask = valid[:, :, None]
            gathered = jnp.where(mask, gathered, jnp.nan)
            result = fn(gathered, n_valid)
            result = jnp.where(n_valid[:, None] >= self._min_periods, result, jnp.nan)
            new_blocks[result.dtype] = result

        new_col_to_block = {}
        for col, (old_dtype, col_idx) in self._df._column_to_block.items():
            new_dtype = new_blocks[
                next(
                    nd
                    for od, nd in zip(self._df._dtype_blocks.keys(), new_blocks.keys())
                    if od == old_dtype
                )
            ].dtype
            new_col_to_block[col] = (new_dtype, col_idx)

        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._df._object_data,
            index=self._df._index,
            column_order=self._df._column_order,
        )

    def sum(self):
        """Expanding sum."""
        return self._apply(lambda g, nv: jnp.nansum(g, axis=1))

    def mean(self):
        """Expanding mean."""
        return self._apply(lambda g, nv: jnp.nanmean(g, axis=1))

    def var(self, ddof: int = 1):
        """Expanding variance."""
        return self._apply(lambda g, nv: jnp.nanvar(g, axis=1, ddof=ddof))

    def std(self, ddof: int = 1):
        """Expanding standard deviation."""
        return self._apply(lambda g, nv: jnp.nanstd(g, axis=1, ddof=ddof))

    def min(self):
        """Expanding minimum."""
        return self._apply(lambda g, nv: jnp.nanmin(g, axis=1))

    def max(self):
        """Expanding maximum."""
        return self._apply(lambda g, nv: jnp.nanmax(g, axis=1))


class EWM:
    """Exponentially weighted moving operations. JIT-compatible via scan."""

    def __init__(self, df, alpha: float, min_periods: int = 0):
        self._df = df
        self._alpha = alpha
        self._min_periods = min_periods

    def _ewm_mean_block(self, block):
        """Compute EWM mean using scan (JIT-compatible)."""
        alpha = self._alpha

        def _scan_fn(carry, x):
            weighted_sum, total_weight = carry
            weighted_sum = alpha * x + (1 - alpha) * weighted_sum
            total_weight = alpha + (1 - alpha) * total_weight
            return (weighted_sum, total_weight), weighted_sum / total_weight

        n_cols = block.shape[1]
        init = (jnp.zeros(n_cols), jnp.zeros(n_cols))
        _, result = jax.lax.scan(_scan_fn, init, block)
        return result

    def _ewm_var_block(self, block):
        """Compute EWM variance using scan (JIT-compatible)."""
        alpha = self._alpha

        def _scan_fn(carry, x):
            old_mean, old_var, total_weight = carry
            total_weight = alpha + (1 - alpha) * total_weight
            new_mean = alpha * x + (1 - alpha) * old_mean
            new_var = (1 - alpha) * (old_var + alpha * (x - old_mean) ** 2)
            return (new_mean, new_var, total_weight), (new_var / total_weight, total_weight)

        n_cols = block.shape[1]
        init = (block[0], jnp.zeros(n_cols), jnp.zeros(n_cols))
        _, (var_raw, weights) = jax.lax.scan(_scan_fn, init, block)
        # Mask based on min_periods
        row_idx = jnp.arange(block.shape[0])
        result = jnp.where(row_idx[:, None] >= self._min_periods, var_raw, jnp.nan)
        return result

    def mean(self):
        """Exponentially weighted moving mean."""
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            new_blocks[block.dtype] = self._ewm_mean_block(block)

        new_col_to_block = {}
        for col, (old_dtype, col_idx) in self._df._column_to_block.items():
            new_dtype = new_blocks[
                next(
                    nd
                    for od, nd in zip(self._df._dtype_blocks.keys(), new_blocks.keys())
                    if od == old_dtype
                )
            ].dtype
            new_col_to_block[col] = (new_dtype, col_idx)

        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._df._object_data,
            index=self._df._index,
            column_order=self._df._column_order,
        )

    def std(self, ddof: int = 1):
        """Exponentially weighted moving standard deviation."""
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            var = self._ewm_var_block(block)
            new_blocks[var.dtype] = jnp.sqrt(jnp.maximum(var, 0))

        new_col_to_block = {}
        for col, (old_dtype, col_idx) in self._df._column_to_block.items():
            new_dtype = new_blocks[
                next(
                    nd
                    for od, nd in zip(self._df._dtype_blocks.keys(), new_blocks.keys())
                    if od == old_dtype
                )
            ].dtype
            new_col_to_block[col] = (new_dtype, col_idx)

        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._df._object_data,
            index=self._df._index,
            column_order=self._df._column_order,
        )

    def var(self, ddof: int = 1):
        """Exponentially weighted moving variance."""
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            new_blocks[block.dtype] = self._ewm_var_block(block)

        new_col_to_block = {}
        for col, (old_dtype, col_idx) in self._df._column_to_block.items():
            new_dtype = new_blocks[
                next(
                    nd
                    for od, nd in zip(self._df._dtype_blocks.keys(), new_blocks.keys())
                    if od == old_dtype
                )
            ].dtype
            new_col_to_block[col] = (new_dtype, col_idx)

        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._df._object_data,
            index=self._df._index,
            column_order=self._df._column_order,
        )


_TIME_UNITS = {
    "s": "s",
    "T": "m",
    "min": "m",
    "H": "h",
    "h": "h",
    "D": "D",
    "W": "W",
}


def _parse_time_window(window: str):
    """Parse pandas-style offset string to numpy timedelta64."""
    # Extract number and unit: '7D' -> (7, 'D'), '2H' -> (2, 'H')
    i = 0
    while i < len(window) and (window[i].isdigit() or window[i] == "."):
        i += 1
    num = int(window[:i]) if i > 0 else 1
    unit = window[i:]
    np_unit = _TIME_UNITS.get(unit, unit)
    return np.timedelta64(num, np_unit)


class TimeRolling:
    """Time-based rolling windows. NOT JIT-compatible (variable window sizes)."""

    def __init__(self, df, window: str, on: str | None = None):
        self._df = df
        self._delta = _parse_time_window(window)
        # Get timestamps
        if on is not None:
            self._times = np.asarray(df._object_data.get(on, df[on].values))
        else:
            self._times = np.asarray(df._index)
        if not np.issubdtype(self._times.dtype, np.datetime64):
            raise ValueError("Time-based rolling requires datetime index or 'on' column")
        # Pre-compute window boundaries (eager)
        # Pandas uses (t - delta, t] window (open left, closed right)
        self._starts = np.searchsorted(self._times, self._times - self._delta, side="right")

    def _apply_np(self, fn):
        """Apply fn per-row using variable-size windows. Returns DataFrame."""
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            data = np.asarray(block)
            n_rows, n_cols = data.shape
            result = np.empty_like(data, dtype=np.float64)
            for i in range(n_rows):
                window_data = data[self._starts[i] : i + 1]
                result[i] = fn(window_data, axis=0)
            new_blocks[jnp.float32] = jnp.asarray(result, dtype=jnp.float32)

        new_col_to_block = {
            col: (jnp.float32, idx) for col, (_, idx) in self._df._column_to_block.items()
        }
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._df._object_data,
            index=self._df._index,
            column_order=self._df._column_order,
        )

    def sum(self):
        return self._apply_np(np.nansum)

    def mean(self):
        return self._apply_np(np.nanmean)

    def var(self, ddof=1):
        return self._apply_np(lambda d, axis: np.nanvar(d, axis=axis, ddof=ddof))

    def std(self, ddof=1):
        return self._apply_np(lambda d, axis: np.nanstd(d, axis=axis, ddof=ddof))

    def min(self):
        return self._apply_np(np.nanmin)

    def max(self):
        return self._apply_np(np.nanmax)


# ========================================
# GroupBy
# ========================================


class SeriesGroupBy:
    """GroupBy on a single column. Aggregations use jax.ops.segment_* (JIT+grad)."""

    def __init__(self, data, segment_ids, num_groups, group_keys, name=None):
        self._data = data  # 1-D JAX array
        self._segment_ids = segment_ids  # int32 array
        self._num_groups = num_groups  # static int
        self._group_keys = group_keys  # JAX array of unique key values
        self._name = name

    def _result_series(self, values):
        return Series(values, index=np.asarray(self._group_keys), name=self._name)

    def sum(self):
        result = jax.ops.segment_sum(self._data, self._segment_ids, self._num_groups)
        return self._result_series(result)

    def mean(self):
        sums = jax.ops.segment_sum(self._data, self._segment_ids, self._num_groups)
        counts = jax.ops.segment_sum(jnp.ones_like(self._data), self._segment_ids, self._num_groups)
        return self._result_series(sums / counts)

    def count(self):
        counts = jax.ops.segment_sum(jnp.ones_like(self._data), self._segment_ids, self._num_groups)
        return self._result_series(counts)

    def min(self):
        result = jax.ops.segment_min(self._data, self._segment_ids, self._num_groups)
        return self._result_series(result)

    def max(self):
        result = jax.ops.segment_max(self._data, self._segment_ids, self._num_groups)
        return self._result_series(result)

    def std(self, ddof=1):
        return self._result_series(jnp.sqrt(self.var(ddof=ddof).values))

    def var(self, ddof=1):
        counts = jax.ops.segment_sum(jnp.ones_like(self._data), self._segment_ids, self._num_groups)
        sums = jax.ops.segment_sum(self._data, self._segment_ids, self._num_groups)
        means = sums / counts
        # Expand means back to per-row, compute squared deviations
        row_means = means[self._segment_ids]
        sq_devs = (self._data - row_means) ** 2
        sum_sq = jax.ops.segment_sum(sq_devs, self._segment_ids, self._num_groups)
        return self._result_series(sum_sq / (counts - ddof))

    def prod(self):
        result = jax.ops.segment_prod(self._data, self._segment_ids, self._num_groups)
        return self._result_series(result)

    def first(self):
        # Take the first element per group (lowest index)
        # Use segment_min on indices, then gather
        indices = jnp.arange(len(self._data))
        first_idx = jax.ops.segment_min(indices, self._segment_ids, self._num_groups)
        return self._result_series(self._data[first_idx])

    def last(self):
        indices = jnp.arange(len(self._data))
        last_idx = jax.ops.segment_max(indices, self._segment_ids, self._num_groups)
        return self._result_series(self._data[last_idx])

    def agg(self, func):
        """Aggregate using one or more functions.

        Args:
            func: string name ('sum', 'mean', etc.) or list of strings.
        """
        if isinstance(func, str):
            return getattr(self, func)()
        if isinstance(func, list):
            # Multiple functions -> DataFrame
            results = {}
            for f in func:
                results[f] = getattr(self, f)().values
            return DataFrame(results, index=np.asarray(self._group_keys))
        raise ValueError(f"Unsupported agg func type: {type(func)}")

    def transform(self, func):
        """Apply aggregation and broadcast back to original shape.

        Args:
            func: string name ('sum', 'mean', etc.)
        """
        if isinstance(func, str):
            agg_result = getattr(self, func)().values
        else:
            raise ValueError("transform requires a string function name")
        # Broadcast group results back to each row
        return Series(
            agg_result[self._segment_ids],
            index=np.arange(len(self._data)),
            name=self._name,
        )


class DataFrameGroupBy:
    """GroupBy on a DataFrame. Aggregations use jax.ops.segment_* (JIT+grad)."""

    def __init__(self, df, by, segment_ids, num_groups, group_keys, val_cols):
        self._df = df
        self._by = by
        self._segment_ids = segment_ids
        self._num_groups = num_groups
        self._group_keys = group_keys
        self._val_cols = val_cols

    def __getitem__(self, key):
        """Select column(s) to aggregate."""
        if isinstance(key, str):
            # Return SeriesGroupBy
            dtype, idx = self._df._column_to_block[key]
            col_data = self._df._dtype_blocks[dtype][:, idx]
            return SeriesGroupBy(
                data=col_data,
                segment_ids=self._segment_ids,
                num_groups=self._num_groups,
                group_keys=self._group_keys,
                name=key,
            )
        raise NotImplementedError("Multi-column groupby selection not yet supported")

    def _apply_segment_op(self, op_fn):
        """Apply a segment op to all value columns, return DataFrame."""
        results = {}
        for col in self._val_cols:
            dtype, idx = self._df._column_to_block[col]
            col_data = self._df._dtype_blocks[dtype][:, idx]
            results[col] = op_fn(col_data)
        return DataFrame(results, index=np.asarray(self._group_keys))

    def sum(self):
        return self._apply_segment_op(
            lambda d: jax.ops.segment_sum(d, self._segment_ids, self._num_groups)
        )

    def mean(self):
        counts = jax.ops.segment_sum(
            jnp.ones(len(self._segment_ids)), self._segment_ids, self._num_groups
        )
        return self._apply_segment_op(
            lambda d: jax.ops.segment_sum(d, self._segment_ids, self._num_groups) / counts
        )

    def count(self):
        counts = jax.ops.segment_sum(
            jnp.ones(len(self._segment_ids)), self._segment_ids, self._num_groups
        )
        # All columns have same count
        results = {col: counts for col in self._val_cols}
        return DataFrame(results, index=np.asarray(self._group_keys))

    def min(self):
        return self._apply_segment_op(
            lambda d: jax.ops.segment_min(d, self._segment_ids, self._num_groups)
        )

    def max(self):
        return self._apply_segment_op(
            lambda d: jax.ops.segment_max(d, self._segment_ids, self._num_groups)
        )

    def std(self, ddof=1):
        counts = jax.ops.segment_sum(
            jnp.ones(len(self._segment_ids)), self._segment_ids, self._num_groups
        )

        def _std_col(d):
            sums = jax.ops.segment_sum(d, self._segment_ids, self._num_groups)
            means = sums / counts
            sq_devs = (d - means[self._segment_ids]) ** 2
            sum_sq = jax.ops.segment_sum(sq_devs, self._segment_ids, self._num_groups)
            return jnp.sqrt(sum_sq / (counts - ddof))

        return self._apply_segment_op(_std_col)

    def var(self, ddof=1):
        counts = jax.ops.segment_sum(
            jnp.ones(len(self._segment_ids)), self._segment_ids, self._num_groups
        )

        def _var_col(d):
            sums = jax.ops.segment_sum(d, self._segment_ids, self._num_groups)
            means = sums / counts
            sq_devs = (d - means[self._segment_ids]) ** 2
            sum_sq = jax.ops.segment_sum(sq_devs, self._segment_ids, self._num_groups)
            return sum_sq / (counts - ddof)

        return self._apply_segment_op(_var_col)

    def prod(self):
        return self._apply_segment_op(
            lambda d: jax.ops.segment_prod(d, self._segment_ids, self._num_groups)
        )

    def first(self):
        indices = jnp.arange(len(self._segment_ids))
        first_idx = jax.ops.segment_min(indices, self._segment_ids, self._num_groups)

        def _first_col(d):
            return d[first_idx]

        return self._apply_segment_op(_first_col)

    def last(self):
        indices = jnp.arange(len(self._segment_ids))
        last_idx = jax.ops.segment_max(indices, self._segment_ids, self._num_groups)

        def _last_col(d):
            return d[last_idx]

        return self._apply_segment_op(_last_col)

    def agg(self, func):
        """Aggregate using a function name string."""
        if isinstance(func, str):
            return getattr(self, func)()
        raise ValueError(f"Unsupported agg func type: {type(func)}")


# ========================================
# JAX Pytree Registration
# ========================================


class _DataFrameAux:
    """Hashable aux_data for DataFrame pytree registration.

    JAX requires aux_data to be hashable and have simple equality semantics
    for its JIT compilation cache. This wraps DataFrame metadata into a
    hashable, comparable container.
    """

    __slots__ = (
        "dtypes_order",
        "column_to_block",
        "object_data_keys",
        "index_tuple",
        "column_order",
        "_object_data_arrays",
    )

    def __init__(self, dtypes_order, column_to_block, object_data, index, column_order):
        self.dtypes_order = dtypes_order
        # Convert to frozenset of tuples for hashability
        self.column_to_block = tuple(sorted(column_to_block.items()))
        self.object_data_keys = tuple(sorted(object_data.keys()))
        self.index_tuple = tuple(index.tolist()) if len(index) <= 10000 else (len(index),)
        self.column_order = tuple(column_order) if not isinstance(column_order, tuple) else column_order
        # Keep original references for unflatten
        self._object_data_arrays = object_data

    def __hash__(self):
        return hash((self.dtypes_order, self.column_to_block, self.index_tuple, self.column_order))

    def __eq__(self, other):
        if not isinstance(other, _DataFrameAux):
            return NotImplemented
        return (
            self.dtypes_order == other.dtypes_order
            and self.column_to_block == other.column_to_block
            and self.index_tuple == other.index_tuple
            and self.column_order == other.column_order
        )


def _dataframe_flatten(df: DataFrame):
    """
    Flatten DataFrame for JAX transformations (dtype blocks version).

    Only numeric data participates in JAX operations (grad, jit, vmap).
    Object data is auxiliary and passes through unchanged.
    """
    # Children: all dtype blocks (arrays that participate in JAX operations)
    # Sort dtypes for deterministic ordering
    dtype_items = sorted(df._dtype_blocks.items(), key=lambda x: str(x[0]))
    children = tuple(block for _, block in dtype_items)
    dtypes_order = tuple(dtype for dtype, _ in dtype_items)

    aux_data = _DataFrameAux(
        dtypes_order=dtypes_order,
        column_to_block=df._column_to_block,
        object_data=df._object_data,
        index=df._index,
        column_order=df._column_order,
    )

    return children, aux_data


def _dataframe_unflatten(aux_data, children):
    """Reconstruct DataFrame from flattened representation (dtype blocks version)."""
    dtype_blocks = {}
    for dtype, block in zip(aux_data.dtypes_order, children):
        dtype_blocks[dtype] = block

    return DataFrame._from_parts(
        dtype_blocks=dtype_blocks,
        column_to_block=dict(aux_data.column_to_block),
        object_data=aux_data._object_data_arrays,
        index=np.arange(children[0].shape[0]) if children else np.array([]),
        column_order=list(aux_data.column_order),
    )


# Register DataFrame as a JAX pytree
jax.tree_util.register_pytree_node(
    DataFrame,
    _dataframe_flatten,
    _dataframe_unflatten,
)


class _SeriesAux:
    """Hashable aux_data for Series pytree registration."""

    __slots__ = ("index_tuple", "name", "_index_array")

    def __init__(self, index, name):
        self.index_tuple = tuple(index.tolist()) if len(index) <= 10000 else (len(index),)
        self.name = name
        self._index_array = index

    def __hash__(self):
        return hash((self.index_tuple, self.name))

    def __eq__(self, other):
        if not isinstance(other, _SeriesAux):
            return NotImplemented
        return self.index_tuple == other.index_tuple and self.name == other.name


def _series_flatten(series: Series):
    """Flatten Series for JAX transformations."""
    children = (series._data,)
    aux_data = _SeriesAux(series._index, series._name)
    return children, aux_data


def _series_unflatten(aux_data, children):
    """Reconstruct Series from flattened representation."""
    (data,) = children
    return Series(data, index=aux_data._index_array, name=aux_data.name)


# Register Series as a JAX pytree
jax.tree_util.register_pytree_node(
    Series,
    _series_flatten,
    _series_unflatten,
)


class _SeriesGroupByAux:
    """Hashable aux_data for SeriesGroupBy pytree registration."""

    __slots__ = ("_segment_ids", "_num_groups", "_group_keys", "_name")

    def __init__(self, segment_ids, num_groups, group_keys, name):
        self._segment_ids = segment_ids
        self._num_groups = num_groups
        self._group_keys = group_keys
        self._name = name

    def __hash__(self):
        return hash((self._num_groups, self._name, len(self._segment_ids)))

    def __eq__(self, other):
        if not isinstance(other, _SeriesGroupByAux):
            return NotImplemented
        return (
            self._num_groups == other._num_groups
            and self._name == other._name
            and len(self._segment_ids) == len(other._segment_ids)
        )


def _series_groupby_flatten(sgb):
    """Flatten SeriesGroupBy for JAX — only data is a differentiable leaf.
    segment_ids goes in aux (static) since it's integer indexing, not differentiable."""
    children = (sgb._data,)
    aux = _SeriesGroupByAux(sgb._segment_ids, sgb._num_groups, sgb._group_keys, sgb._name)
    return children, aux


def _series_groupby_unflatten(aux, children):
    (data,) = children
    return SeriesGroupBy(
        data=data,
        segment_ids=aux._segment_ids,
        num_groups=aux._num_groups,
        group_keys=aux._group_keys,
        name=aux._name,
    )


jax.tree_util.register_pytree_node(
    SeriesGroupBy,
    _series_groupby_flatten,
    _series_groupby_unflatten,
)
