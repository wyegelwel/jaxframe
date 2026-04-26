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
        if self._numeric_data is None:
            raise ValueError("No numeric columns to sum")

        result = jnp.nansum(self._numeric_data, axis=axis)

        if axis == 0:
            # Column-wise sum -> Series
            return Series(result, index=np.array(self._numeric_cols), name="sum")
        elif axis == 1:
            # Row-wise sum -> Series
            return Series(result, index=self._index, name="sum")
        else:
            # Total sum -> scalar
            return result

    def mean(self, axis: int | None = 0):
        """
        Mean along axis (JIT-compatible).

        Skips NaN values by default (matching pandas behavior).
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute mean")

        result = jnp.nanmean(self._numeric_data, axis=axis)

        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="mean")
        elif axis == 1:
            return Series(result, index=self._index, name="mean")
        else:
            return result

    def std(self, axis: int | None = 0, ddof: int = 1):
        """
        Standard deviation along axis (JIT-compatible, differentiable).

        Skips NaN values by default (matching pandas behavior).

        Args:
            axis: 0 for column-wise, 1 for row-wise, None for total
            ddof: Delta degrees of freedom (default 1 for sample std)
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute std")

        result = jnp.nanstd(self._numeric_data, axis=axis, ddof=ddof)

        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="std")
        elif axis == 1:
            return Series(result, index=self._index, name="std")
        else:
            return result

    def var(self, axis: int | None = 0, ddof: int = 1):
        """
        Variance along axis (JIT-compatible, differentiable).

        Skips NaN values by default (matching pandas behavior).

        Args:
            axis: 0 for column-wise, 1 for row-wise, None for total
            ddof: Delta degrees of freedom (default 1 for sample variance)
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute variance")

        result = jnp.nanvar(self._numeric_data, axis=axis, ddof=ddof)

        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="var")
        elif axis == 1:
            return Series(result, index=self._index, name="var")
        else:
            return result

    def min(self, axis: int | None = 0):
        """
        Minimum along axis (JIT-compatible).

        Skips NaN values by default (matching pandas behavior).

        Note: Gradient is non-smooth at the minimum point.
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute min")

        result = jnp.nanmin(self._numeric_data, axis=axis)

        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="min")
        elif axis == 1:
            return Series(result, index=self._index, name="min")
        else:
            return result

    def max(self, axis: int | None = 0):
        """
        Maximum along axis (JIT-compatible).

        Skips NaN values by default (matching pandas behavior).

        Note: Gradient is non-smooth at the maximum point.
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute max")

        result = jnp.nanmax(self._numeric_data, axis=axis)

        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="max")
        elif axis == 1:
            return Series(result, index=self._index, name="max")
        else:
            return result

    def prod(self, axis: int | None = 0):
        """
        Product along axis (JIT-compatible, differentiable).

        Skips NaN values by default (matching pandas behavior).
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute product")

        result = jnp.nanprod(self._numeric_data, axis=axis)

        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name="prod")
        elif axis == 1:
            return Series(result, index=self._index, name="prod")
        else:
            return result

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
        """Count unique values per column. Not JIT-compatible (uses np.unique)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns")
        data = np.asarray(self._numeric_data)
        if axis == 0:
            counts = [len(np.unique(data[:, i])) for i in range(data.shape[1])]
            return Series(jnp.array(counts), index=np.array(self._numeric_cols), name="nunique")
        else:
            counts = [len(np.unique(data[i, :])) for i in range(data.shape[0])]
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
        """Mode (most frequent value). Not JIT-compatible."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns")
        data = np.asarray(self._numeric_data)
        if axis == 0:
            modes = []
            for i in range(data.shape[1]):
                vals, counts = np.unique(data[:, i], return_counts=True)
                modes.append(vals[np.argmax(counts)])
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
        # Extract key column(s) — eager numpy for group discovery
        key_col = by[0]
        dtype, idx = self._column_to_block[key_col]
        key_data = np.asarray(self._dtype_blocks[dtype][:, idx])
        # Discover groups eagerly
        unique_keys, inverse = np.unique(key_data, return_inverse=True)
        segment_ids = jnp.array(inverse, dtype=jnp.int32)
        num_groups = len(unique_keys)
        group_keys = jnp.array(unique_keys)
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

        Args:
            periods: Number of periods for differencing

        Returns:
            DataFrame with differences (NaN for undefined values, matching pandas)

        Examples:
            >>> df.diff()  # df[i] - df[i-1]
            >>> df.diff(2)  # df[i] - df[i-2]
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns for diff")

        # Use NaN for shifted values (default behavior)
        shifted = self.shift(periods)

        # Calculate differences for each column
        result_data = {}
        for col in self._column_order:
            if col in self._column_to_block:
                col_data = self._get_column_data(col)
                shifted_data = shifted._get_column_data(col)
                diff_col = col_data - shifted_data

                # Set first 'periods' rows to NaN (matching pandas behavior)
                if periods > 0:
                    mask = jnp.arange(len(self._index)) < periods
                    diff_col = jnp.where(mask, jnp.nan, diff_col)
                elif periods < 0:
                    # For negative periods, mask the last abs(periods) rows
                    abs_periods = abs(periods)
                    mask = jnp.arange(len(self._index)) >= (len(self._index) - abs_periods)
                    diff_col = jnp.where(mask, jnp.nan, diff_col)

                result_data[col] = diff_col

        return DataFrame(result_data, index=self._index.copy())

    def pct_change(self, periods: int = 1):
        """
        Calculate percentage change (JIT-compatible, differentiable).

        Args:
            periods: Number of periods for percentage change

        Returns:
            DataFrame with percentage changes (NaN for undefined values, matching pandas)

        Formula: (current - previous) / previous

        Examples:
            >>> df.pct_change()  # (df[i] - df[i-1]) / df[i-1]
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns for pct_change")

        # Use NaN for shifted values (default behavior)
        # This will naturally produce NaN in the output for undefined values
        shifted = self.shift(periods)

        # Calculate percentage change for each column
        result_data = {}
        for col in self._column_order:
            if col in self._column_to_block:
                col_data = self._get_column_data(col)
                shifted_data = shifted._get_column_data(col)
                result_data[col] = (col_data - shifted_data) / shifted_data

        return DataFrame(result_data, index=self._index.copy())

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
            for dtype, block in self._dtype_blocks.items():
                # Determine result dtype
                result_dtype = jnp.result_type(dtype, type(other))
                new_blocks[np.dtype(result_dtype)] = op(block, other).astype(result_dtype)

            # Rebuild column mapping if dtype changed
            new_column_to_block = {}
            for col, (old_dtype, idx) in self._column_to_block.items():
                result_dtype = jnp.result_type(old_dtype, type(other))
                new_column_to_block[col] = (np.dtype(result_dtype), idx)

            return DataFrame._from_parts(
                dtype_blocks=new_blocks,
                column_to_block=new_column_to_block,
                object_data=self._object_data.copy(),
                index=self._index.copy(),
                column_order=self._column_order,
            )

        elif isinstance(other, DataFrame):
            # DataFrame operation: align and promote types
            if self._column_order != other._column_order:
                raise ValueError(f"Column mismatch for {op_name}")

            # Collect results for each column
            result_data = {}
            for col in self._column_order:
                if col in self._column_to_block and col in other._column_to_block:
                    left_data = self._get_column_data(col)
                    right_data = other._get_column_data(col)

                    # Determine result dtype
                    left_dtype, _ = self._column_to_block[col]
                    right_dtype, _ = other._column_to_block[col]
                    result_dtype = jnp.result_type(left_dtype, right_dtype)

                    result_data[col] = op(left_data, right_data).astype(result_dtype)

            return DataFrame(result_data, index=self._index.copy())

        elif type(other).__name__ == "Series":
            # Series broadcasts across rows (like pandas)
            # Each column gets the corresponding value from the Series
            result_data = {}
            for col in self._column_order:
                if col in self._column_to_block:
                    col_data = self._get_column_data(col)
                    dtype, _ = self._column_to_block[col]

                    # Find this column's value in the Series
                    # Series._index contains column names, Series._data contains values
                    if hasattr(other, "_index"):
                        # Find column position in Series index
                        series_idx = np.where(other._index == col)[0]
                        if len(series_idx) > 0:
                            series_value = other._data[series_idx[0]]
                        else:
                            continue  # Column not in Series, skip it
                    else:
                        # Fallback: use index in column_order
                        numeric_cols = [c for c in self._column_order if c in self._column_to_block]
                        col_idx = numeric_cols.index(col)
                        series_value = other._data[col_idx]

                    result_dtype = jnp.result_type(dtype, other._data.dtype)
                    result_data[col] = op(col_data, series_value).astype(result_dtype)

            return DataFrame(result_data, index=self._index.copy())

        elif isinstance(other, (np.ndarray, jnp.ndarray)):
            # Array operation
            other_arr = jnp.asarray(other)

            # Handle 1D arrays: broadcast across columns (like pandas)
            if other_arr.ndim == 1 and len(other_arr) == len(self._column_order):
                # Array length matches number of columns - broadcast across columns
                result_data = {}
                numeric_cols = [c for c in self._column_order if c in self._column_to_block]
                for idx, col in enumerate(numeric_cols):
                    col_data = self._get_column_data(col)
                    dtype, _ = self._column_to_block[col]
                    result_dtype = jnp.result_type(dtype, other_arr.dtype)
                    result_data[col] = op(col_data, other_arr[idx]).astype(result_dtype)
                return DataFrame(result_data, index=self._index.copy())
            else:
                # Array broadcasts element-wise
                result_data = {}
                for col in self._column_order:
                    if col in self._column_to_block:
                        col_data = self._get_column_data(col)
                        dtype, _ = self._column_to_block[col]
                        result_dtype = jnp.result_type(dtype, other_arr.dtype)
                        result_data[col] = op(col_data, other_arr).astype(result_dtype)
                return DataFrame(result_data, index=self._index.copy())

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

    def value_counts(self, sort=True):
        """Count occurrences of each unique value. Not JIT-compatible."""
        data = np.asarray(self._data)
        unique_vals, counts = np.unique(data, return_counts=True)
        if sort:
            order = np.argsort(-counts)
            unique_vals = unique_vals[order]
            counts = counts[order]
        return Series(jnp.array(counts), index=unique_vals, name=self._name)

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

    # Aux data: metadata that doesn't participate in JAX ops
    aux_data = {
        "dtypes_order": dtypes_order,
        "column_to_block": df._column_to_block,
        "object_data": df._object_data,
        "index": df._index,
        "column_order": df._column_order,
    }

    return children, aux_data


def _dataframe_unflatten(aux_data, children):
    """Reconstruct DataFrame from flattened representation (dtype blocks version)."""
    # Reconstruct dtype_blocks from children and dtypes_order
    dtype_blocks = {}
    for dtype, block in zip(aux_data["dtypes_order"], children):
        dtype_blocks[dtype] = block

    return DataFrame._from_parts(
        dtype_blocks=dtype_blocks,
        column_to_block=aux_data["column_to_block"],
        object_data=aux_data["object_data"],
        index=aux_data["index"],
        column_order=aux_data["column_order"],
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
        "index": series._index,
        "name": series._name,
    }
    return children, aux_data


def _series_unflatten(aux_data, children):
    """Reconstruct Series from flattened representation."""
    (data,) = children
    return Series(data, index=aux_data["index"], name=aux_data["name"])


# Register Series as a JAX pytree
jax.tree_util.register_pytree_node(
    Series,
    _series_flatten,
    _series_unflatten,
)


def _series_groupby_flatten(sgb):
    """Flatten SeriesGroupBy for JAX — only data is a differentiable leaf.
    segment_ids goes in aux (static) since it's integer indexing, not differentiable."""
    children = (sgb._data,)
    aux = {
        "segment_ids": sgb._segment_ids,
        "num_groups": sgb._num_groups,
        "group_keys": sgb._group_keys,
        "name": sgb._name,
    }
    return children, aux


def _series_groupby_unflatten(aux, children):
    (data,) = children
    return SeriesGroupBy(
        data=data,
        segment_ids=aux["segment_ids"],
        num_groups=aux["num_groups"],
        group_keys=aux["group_keys"],
        name=aux["name"],
    )


jax.tree_util.register_pytree_node(
    SeriesGroupBy,
    _series_groupby_flatten,
    _series_groupby_unflatten,
)
