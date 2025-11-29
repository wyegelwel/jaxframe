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


def concat(dataframes: List['DataFrame'], axis: int = 0):
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
                    all_col_data[col] = (df._object_data[col], 'object')

        # Build new DataFrame from collected columns
        numeric_data = {}
        object_data = {}
        for col, (data, dtype) in all_col_data.items():
            if dtype == 'object':
                object_data[col] = data
            else:
                numeric_data[col] = data

        # Column order
        new_column_order = tuple(
            col for df in dataframes for col in df._column_order
        )

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

    _dtype_blocks: Dict[np.dtype, jnp.ndarray]
    _column_to_block: Dict[str, Tuple[np.dtype, int]]
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

    def _init_from_array(self, data: Union[np.ndarray, jnp.ndarray], index=None):
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
    def shape(self) -> Tuple[int, int]:
        """Return (n_rows, n_cols)."""
        n_rows = len(self._index)
        n_cols = len(self._column_order)
        return (n_rows, n_cols)

    def __len__(self) -> int:
        """Return number of rows."""
        return len(self._index)

    @property
    def columns(self) -> List[str]:
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
    def dtypes(self) -> Dict[str, Any]:
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
            col_data = block[:, idx:idx+1]  # Keep 2D shape
            columns.append(col_data.astype(promoted_dtype))

        return jnp.concatenate(columns, axis=1)

    @property
    def empty(self) -> bool:
        """Return True if DataFrame has no elements."""
        return self.size == 0

    # Backward compatibility properties for tests
    @property
    def _numeric_cols(self) -> Tuple[str, ...]:
        """Get list of numeric column names (backward compatibility)."""
        return tuple(col for col in self._column_order if col in self._column_to_block)

    @property
    def _numeric_data(self) -> Optional[jnp.ndarray]:
        """Get numeric data as single array (backward compatibility)."""
        if not self._dtype_blocks:
            return None
        return self.values

    @property
    def _numeric_dtypes(self) -> Tuple[Any, ...]:
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
            dtype: jax.device_put(block, device)
            for dtype, block in self._dtype_blocks.items()
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
            gpu_devices = jax.devices('gpu')
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
        cpu_device = jax.devices('cpu')[0]
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
            tpu_devices = jax.devices('tpu')
            if id >= len(tpu_devices):
                raise ValueError(f"TPU {id} not found. Available TPUs: {len(tpu_devices)}")
            return self.to_device(tpu_devices[id])
        except RuntimeError:
            raise RuntimeError("No TPU devices available")

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
        """Element-wise multiplication (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.multiply, "multiplication")

    def __add__(self, other: Union[float, 'DataFrame']) -> 'DataFrame':
        """Element-wise addition (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.add, "addition")

    def __sub__(self, other: Union[float, 'DataFrame', 'Series']) -> 'DataFrame':
        """Element-wise subtraction (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.subtract, "subtraction")

    def __truediv__(self, other: Union[float, 'DataFrame', 'Series']) -> 'DataFrame':
        """Element-wise division (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.true_divide, "division")

    def __floordiv__(self, other: Union[float, 'DataFrame']) -> 'DataFrame':
        """Element-wise floor division (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.floor_divide, "floor division")

    def __mod__(self, other: Union[float, 'DataFrame']) -> 'DataFrame':
        """Element-wise modulo (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.mod, "modulo")

    def __pow__(self, other: Union[float, 'DataFrame']) -> 'DataFrame':
        """Element-wise power (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.power, "power")

    def __matmul__(self, other: Union[jnp.ndarray, 'DataFrame']) -> Union[jnp.ndarray, 'DataFrame']:
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

    def std(self, axis: Optional[int] = 0, ddof: int = 1):
        """
        Standard deviation along axis (JIT-compatible, differentiable).

        Args:
            axis: 0 for column-wise, 1 for row-wise, None for total
            ddof: Delta degrees of freedom (default 1 for sample std)
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute std")

        result = jnp.std(self._numeric_data, axis=axis, ddof=ddof)

        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name='std')
        elif axis == 1:
            return Series(result, index=self._index, name='std')
        else:
            return result

    def var(self, axis: Optional[int] = 0, ddof: int = 1):
        """
        Variance along axis (JIT-compatible, differentiable).

        Args:
            axis: 0 for column-wise, 1 for row-wise, None for total
            ddof: Delta degrees of freedom (default 1 for sample variance)
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute variance")

        result = jnp.var(self._numeric_data, axis=axis, ddof=ddof)

        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name='var')
        elif axis == 1:
            return Series(result, index=self._index, name='var')
        else:
            return result

    def min(self, axis: Optional[int] = 0):
        """
        Minimum along axis (JIT-compatible).

        Note: Gradient is non-smooth at the minimum point.
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute min")

        result = jnp.min(self._numeric_data, axis=axis)

        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name='min')
        elif axis == 1:
            return Series(result, index=self._index, name='min')
        else:
            return result

    def max(self, axis: Optional[int] = 0):
        """
        Maximum along axis (JIT-compatible).

        Note: Gradient is non-smooth at the maximum point.
        """
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute max")

        result = jnp.max(self._numeric_data, axis=axis)

        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name='max')
        elif axis == 1:
            return Series(result, index=self._index, name='max')
        else:
            return result

    def prod(self, axis: Optional[int] = 0):
        """Product along axis (JIT-compatible, differentiable)."""
        if self._numeric_data is None:
            raise ValueError("No numeric columns to compute product")

        result = jnp.prod(self._numeric_data, axis=axis)

        if axis == 0:
            return Series(result, index=np.array(self._numeric_cols), name='prod')
        elif axis == 1:
            return Series(result, index=self._index, name='prod')
        else:
            return result

    def abs(self):
        """
        Absolute value (JIT-compatible).

        Note: Gradient is non-smooth at zero.
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns for absolute value")

        # Apply abs to each column, preserving dtype
        result_data = {}
        for col in self._column_order:
            if col in self._column_to_block:
                col_data = self._get_column_data(col)
                result_data[col] = jnp.abs(col_data)

        return DataFrame(result_data, index=self._index.copy())

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
        return DataFrame({
            col: corr_matrix[:, i]
            for i, col in enumerate(self._numeric_cols)
        }, index=np.array(self._numeric_cols))

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
        return DataFrame({
            col: cov_matrix[:, i]
            for i, col in enumerate(self._numeric_cols)
        }, index=np.array(self._numeric_cols))

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
        return DataFrame({
            col: transposed_data[:, i]
            for i, col in enumerate(new_cols)
        }, index=new_index)

    # ========================================
    # Time series operations
    # ========================================

    def shift(self, periods: int = 1, fill_value=0.0):
        """
        Shift data by n periods (JIT-compatible, differentiable).

        Args:
            periods: Number of periods to shift (positive for forward, negative for backward)
            fill_value: Value to use for padded entries

        Returns:
            DataFrame with shifted values

        Examples:
            >>> df.shift(1)  # Shift forward by 1 (introduce lag)
            >>> df.shift(-1)  # Shift backward by 1 (lead)
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns to shift")

        if periods == 0:
            return self

        n_rows = len(self._index)

        # Apply shift to each column, preserving dtype
        result_data = {}
        for col in self._column_order:
            if col in self._column_to_block:
                col_data = self._get_column_data(col)

                if periods > 0:
                    # Shift forward (lag) - add fill_value at the beginning
                    if periods >= n_rows:
                        # If shifting by more than length, return all fill_value
                        new_col_data = jnp.full_like(col_data, fill_value)
                    else:
                        padding = jnp.full((periods,), fill_value, dtype=col_data.dtype)
                        new_col_data = jnp.concatenate([padding, col_data[:-periods]], axis=0)
                else:
                    # Shift backward (lead) - add fill_value at the end
                    abs_periods = abs(periods)
                    if abs_periods >= n_rows:
                        # If shifting by more than length, return all fill_value
                        new_col_data = jnp.full_like(col_data, fill_value)
                    else:
                        padding = jnp.full((abs_periods,), fill_value, dtype=col_data.dtype)
                        new_col_data = jnp.concatenate([col_data[abs_periods:], padding], axis=0)

                result_data[col] = new_col_data

        # Note: object data is not shifted, similar to old behavior
        return DataFrame(result_data, index=self._index.copy())

    def diff(self, periods: int = 1):
        """
        Calculate first discrete difference (JIT-compatible, differentiable).

        Args:
            periods: Number of periods for differencing

        Returns:
            DataFrame with differences

        Examples:
            >>> df.diff()  # df[i] - df[i-1]
            >>> df.diff(2)  # df[i] - df[i-2]
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns for diff")

        shifted = self.shift(periods, fill_value=0.0)

        # Calculate differences for each column
        result_data = {}
        for col in self._column_order:
            if col in self._column_to_block:
                col_data = self._get_column_data(col)
                shifted_data = shifted._get_column_data(col)
                diff_col = col_data - shifted_data

                # Set first 'periods' rows to 0 (matching pandas behavior with fillna(0))
                # This is because pandas returns NaN for the first 'periods' rows
                if periods > 0:
                    mask = jnp.arange(len(self._index)) < periods
                    diff_col = jnp.where(mask, 0.0, diff_col)
                elif periods < 0:
                    # For negative periods, mask the last abs(periods) rows
                    abs_periods = abs(periods)
                    mask = jnp.arange(len(self._index)) >= (len(self._index) - abs_periods)
                    diff_col = jnp.where(mask, 0.0, diff_col)

                result_data[col] = diff_col

        return DataFrame(result_data, index=self._index.copy())

    def pct_change(self, periods: int = 1):
        """
        Calculate percentage change (JIT-compatible, differentiable).

        Args:
            periods: Number of periods for percentage change

        Returns:
            DataFrame with percentage changes

        Formula: (current - previous) / previous

        Examples:
            >>> df.pct_change()  # (df[i] - df[i-1]) / df[i-1]
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns for pct_change")

        shifted = self.shift(periods, fill_value=1.0)  # Use 1.0 to avoid division by zero in padding

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
            # Apply where column by column
            result_data = {}
            for col in self._column_order:
                if col in self._column_to_block and col in condition._column_to_block:
                    col_data = self._get_column_data(col)
                    cond_data = condition._get_column_data(col)
                    result_data[col] = jnp.where(cond_data, col_data, fill_value)
            return DataFrame(result_data, index=self._index.copy())
        else:
            # Scalar or array condition - apply to all columns
            mask = jnp.asarray(condition)
            result_data = {}
            for col in self._column_order:
                if col in self._column_to_block:
                    col_data = self._get_column_data(col)
                    result_data[col] = jnp.where(mask, col_data, fill_value)
            return DataFrame(result_data, index=self._index.copy())

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
        if not self._dtype_blocks:
            raise ValueError("No numeric columns to clip")

        # Apply clipping to each column
        result_data = {}
        for col in self._column_order:
            if col in self._column_to_block:
                col_data = self._get_column_data(col)

                if lower is not None:
                    col_data = jnp.maximum(col_data, lower)
                if upper is not None:
                    col_data = jnp.minimum(col_data, upper)

                result_data[col] = col_data

        return DataFrame(result_data, index=self._index.copy())

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
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if it's a column
        if name in self._column_order:
            return self[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ========================================
    # Comparison operations
    # ========================================

    def __gt__(self, other) -> 'DataFrame':
        """Greater than comparison (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.greater, "greater than")

    def __ge__(self, other) -> 'DataFrame':
        """Greater than or equal comparison (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.greater_equal, "greater than or equal")

    def __lt__(self, other) -> 'DataFrame':
        """Less than comparison (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.less, "less than")

    def __le__(self, other) -> 'DataFrame':
        """Less than or equal comparison (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.less_equal, "less than or equal")

    def __eq__(self, other) -> 'DataFrame':
        """Equality comparison (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.equal, "equality")

    def __ne__(self, other) -> 'DataFrame':
        """Not equal comparison (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.not_equal, "not equal")

    # ========================================
    # Logical operations
    # ========================================

    def __and__(self, other) -> 'DataFrame':
        """Logical AND (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.bitwise_and, "logical AND")

    def __or__(self, other) -> 'DataFrame':
        """Logical OR (JIT-compatible with type promotion)."""
        return self._apply_elementwise(other, jnp.bitwise_or, "logical OR")

    def __invert__(self) -> 'DataFrame':
        """Logical NOT (JIT-compatible)."""
        if not self._dtype_blocks:
            raise ValueError("No numeric columns for logical NOT")

        # Apply inversion to each block independently
        result_data = {}
        for col in self._column_order:
            if col in self._column_to_block:
                col_data = self._get_column_data(col)
                # Convert to boolean if needed (handles comparison results that might be 0/1 as ints/floats)
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

        elif type(other).__name__ == 'Series':
            # Series broadcasts across rows (like pandas)
            result_data = {}
            for col in self._column_order:
                if col in self._column_to_block:
                    col_data = self._get_column_data(col)
                    dtype, _ = self._column_to_block[col]
                    result_dtype = jnp.result_type(dtype, other._data.dtype)
                    result_data[col] = op(col_data, other._data).astype(result_dtype)

            return DataFrame(result_data, index=self._index.copy())

        elif isinstance(other, (np.ndarray, jnp.ndarray)):
            # Array operation: broadcast to each column
            other_arr = jnp.asarray(other)
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
                    return Series(data, index=np.array(self._df._numeric_cols), name=self._df._index[key])
                else:
                    # Only object columns
                    raise NotImplementedError("Single row selection with only object columns not yet supported")
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
        'dtypes_order': dtypes_order,
        'column_to_block': df._column_to_block,
        'object_data': df._object_data,
        'index': df._index,
        'column_order': df._column_order,
    }

    return children, aux_data


def _dataframe_unflatten(aux_data, children):
    """Reconstruct DataFrame from flattened representation (dtype blocks version)."""
    # Reconstruct dtype_blocks from children and dtypes_order
    dtype_blocks = {}
    for dtype, block in zip(aux_data['dtypes_order'], children):
        dtype_blocks[dtype] = block

    return DataFrame._from_parts(
        dtype_blocks=dtype_blocks,
        column_to_block=aux_data['column_to_block'],
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
