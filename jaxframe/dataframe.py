"""
Core DataFrame implementation with JAX backend.

This module implements a DataFrame class that:
1. Stores numeric data in JAX arrays for performance
2. Stores non-numeric data in numpy arrays
3. Supports JIT compilation via pytree registration
4. Supports automatic differentiation on numeric columns
"""

import functools
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
    ("min", True, True, "Subgradient at argmin (JAX convention)"),
    ("max", True, True, "Subgradient at argmax (JAX convention)"),
    ("median", True, True, "Differentiable via quantile interpolation"),
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
    ("sort_values", True, True, "jnp.argsort + gather; grad flows through the permutation"),
    ("nlargest", True, True, "top_k gather; grad flows to selected rows"),
    ("nsmallest", True, True, "top_k gather; grad flows to selected rows"),
    ("quantile", True, True, "Differentiable via interpolation weights"),
    ("rank", True, False, "Step function — gradient zero a.e."),
    ("ffill", True, True, None),
    ("bfill", True, True, None),
    ("interpolate", True, True, None),
    ("cummax", True, True, "Grad flows to running-max positions"),
    ("cummin", True, True, "Grad flows to running-min positions"),
    ("take", True, True, "Gather — grad flows"),
    ("mask", True, True, None),
    ("replace", True, True, "Grad flows through unreplaced values"),
    ("combine_first", True, True, None),
    ("transform", True, True, None),
    ("agg", True, True, "For differentiable aggregators"),
    ("eq/ne/lt/le/gt/ge", True, False, "Boolean output — not real-valued"),
    # Column ops
    ("drop", True, True, None),
    ("rename", True, True, None),
    # Rolling (fixed-size)
    ("rolling.sum", True, True, None),
    ("rolling.mean", True, True, None),
    ("rolling.std", True, True, None),
    ("rolling.var", True, True, None),
    ("rolling.min", True, True, "Subgradient"),
    ("rolling.max", True, True, "Subgradient"),
    # GroupBy (segment ops)
    ("groupby.sum", True, True, None),
    ("groupby.mean", True, True, None),
    ("groupby.std", True, True, None),
    ("groupby.var", True, True, None),
    ("groupby.transform", True, True, None),
    ("groupby.min", True, True, "segment_min grad (subgradient)"),
    ("groupby.max", True, True, "segment_max grad (subgradient)"),
    ("groupby.count", True, False, "Integer output"),
    ("groupby.prod", True, False, "JAX scatter_mul grad unimplemented"),
    ("groupby.first", True, True, "Gather — grad flows to first element per group"),
    ("groupby.last", True, True, "Gather — grad flows to last element per group"),
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
        elif isinstance(data, np.ndarray | jnp.ndarray):
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

        # Cached like _get_column_data: frames are immutable except via
        # _replace_self, which clears this cache.
        cached = self.__dict__.get("_values_cache")
        if cached is not None:
            return cached

        # Single block in column order: no promotion or concat needed
        if len(self._dtype_blocks) == 1:
            block = next(iter(self._dtype_blocks.values()))
            numeric_cols = [col for col in self._column_order if col in self._column_to_block]
            in_order = all(self._column_to_block[col][1] == i for i, col in enumerate(numeric_cols))
            if in_order and block.shape[1] == len(numeric_cols):
                self.__dict__["_values_cache"] = block
                return block

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

        result = jnp.concatenate(columns, axis=1)
        self.__dict__["_values_cache"] = result
        return result

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

    def block_until_ready(self):
        """Block until all underlying JAX arrays are materialized."""
        for block in self._dtype_blocks.values():
            block.block_until_ready()
        return self

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
            # Single column — slice its dtype block directly (one dispatch);
            # never build the full promoted matrix here.
            if key in self._column_to_block:
                return Series(self._get_column_data(key), index=self._index, name=key)
            elif key in self._object_data:
                return Series(self._object_data[key], index=self._index, name=key)
            else:
                raise KeyError(f"Column '{key}' not found")

        elif isinstance(key, list):
            # Multiple columns — per-column block slices, no promotion
            col_arrays = {}
            object_data = {}
            for col in key:
                if col in self._column_to_block:
                    col_arrays[col] = self._get_column_data(col)
                elif col in self._object_data:
                    object_data[col] = self._object_data[col]
                else:
                    raise KeyError(f"Column '{col}' not found")
            return DataFrame._from_column_arrays(
                col_arrays, list(key), self._index, object_data=object_data
            )

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

    # ---- Fast NaN-aware reductions via jax.lax.reduce ----
    # XLA CPU doesn't fuse isnan+where+reduce into one kernel, so jnp.nansum
    # is ~6x slower than jnp.sum.  By embedding the NaN check in the
    # lax.reduce combiner, XLA produces a single fused reduction — nearly
    # as fast as the non-NaN version.  Each is self-JIT'd so the fusion
    # kicks in even for eager DataFrame calls.

    @staticmethod
    @jax.jit
    def _fast_nansum(block):
        """NaN-skipping sum via single-pass fused lax.reduce."""
        block = block.astype(jnp.float32)

        def combiner(a, b):
            return jnp.where(jnp.isnan(a), 0.0, a) + jnp.where(jnp.isnan(b), 0.0, b)

        return jax.lax.reduce(block, jnp.float32(0.0), combiner, [0])

    @staticmethod
    @jax.jit
    def _fast_nanmin(block):
        """NaN-skipping min via single-pass fused lax.reduce."""
        block = block.astype(jnp.float32)

        def combiner(a, b):
            return jnp.minimum(
                jnp.where(jnp.isnan(a), jnp.inf, a),
                jnp.where(jnp.isnan(b), jnp.inf, b),
            )

        return jax.lax.reduce(block, jnp.float32(jnp.inf), combiner, [0])

    @staticmethod
    @jax.jit
    def _fast_nanmax(block):
        """NaN-skipping max via single-pass fused lax.reduce."""
        block = block.astype(jnp.float32)

        def combiner(a, b):
            return jnp.maximum(
                jnp.where(jnp.isnan(a), -jnp.inf, a),
                jnp.where(jnp.isnan(b), -jnp.inf, b),
            )

        return jax.lax.reduce(block, jnp.float32(-jnp.inf), combiner, [0])

    @staticmethod
    @jax.jit
    def _fast_nanprod(block):
        """NaN-skipping product via single-pass fused lax.reduce."""

        def combiner(a, b):
            return jnp.where(jnp.isnan(a), 1.0, a) * jnp.where(jnp.isnan(b), 1.0, b)

        return jax.lax.reduce(block, jnp.float32(1.0), combiner, [0])

    @staticmethod
    @jax.jit
    def _fast_nansum_and_count(block):
        """Total NaN-skipping sum + valid count for a block, both scalar.

        Fuses nansum + count into a single JIT call to avoid Python dispatch overhead.
        Uses lax.reduce axis=0 (well-optimized) then .sum() on the small column vector.
        """
        block = block.astype(jnp.float32)

        def nan_add(a, b):
            return jnp.where(jnp.isnan(a), 0.0, a) + jnp.where(jnp.isnan(b), 0.0, b)

        col_sums = jax.lax.reduce(block, jnp.float32(0.0), nan_add, [0])
        valid = jnp.where(jnp.isnan(block), 0.0, 1.0)
        col_counts = jax.lax.reduce(valid, jnp.float32(0.0), jax.lax.add, [0])
        return col_sums.sum(), col_counts.sum()

    @staticmethod
    @jax.jit
    def _fast_nansumsq_and_count(block):
        """Total NaN-skipping sum, sum-of-squares, and count for a block, all scalar."""
        block = block.astype(jnp.float32)

        def nan_add(a, b):
            return jnp.where(jnp.isnan(a), 0.0, a) + jnp.where(jnp.isnan(b), 0.0, b)

        col_sums = jax.lax.reduce(block, jnp.float32(0.0), nan_add, [0])
        clean = jnp.where(jnp.isnan(block), 0.0, block)
        col_sumsq = jax.lax.reduce(clean * clean, jnp.float32(0.0), jax.lax.add, [0])
        valid = jnp.where(jnp.isnan(block), 0.0, 1.0)
        col_counts = jax.lax.reduce(valid, jnp.float32(0.0), jax.lax.add, [0])
        return col_sums.sum(), col_sumsq.sum(), col_counts.sum()

    @staticmethod
    @jax.jit
    def _fast_nanmean(block):
        """NaN-skipping mean via fused single-operand reduces.

        Single-operand lax.reduce is differentiable (tuple-operand is not:
        its transpose can't handle symbolic Zero cotangents); XLA fuses the
        two passes under JIT anyway."""
        clean = jnp.where(jnp.isnan(block), 0.0, block)
        valid = jnp.where(jnp.isnan(block), 0.0, 1.0)
        total = jax.lax.reduce(clean, jnp.float32(0.0), jax.lax.add, [0])
        count = jax.lax.reduce(valid, jnp.float32(0.0), jax.lax.add, [0])
        return total / jnp.maximum(count, 1)

    @staticmethod
    @jax.jit
    def _fast_nanvar(block, ddof):
        """NaN-skipping variance via fused single-operand reduces (differentiable)."""
        clean = jnp.where(jnp.isnan(block), 0.0, block)
        valid = jnp.where(jnp.isnan(block), 0.0, 1.0)
        total = jax.lax.reduce(clean, jnp.float32(0.0), jax.lax.add, [0])
        total_sq = jax.lax.reduce(clean * clean, jnp.float32(0.0), jax.lax.add, [0])
        count = jax.lax.reduce(valid, jnp.float32(0.0), jax.lax.add, [0])
        mean = total / jnp.maximum(count, 1)
        return (total_sq - count * mean * mean) / jnp.maximum(count - ddof, 1)

    def _reduce_axis0(self, fn, name):
        """Reduce each dtype block along axis=0 and assemble into a Series.

        Avoids building a full promoted matrix via .values — reduces each
        block independently and concatenates the small 1D results.
        """
        numeric_cols = self._numeric_cols
        if len(self._dtype_blocks) == 1:
            block = next(iter(self._dtype_blocks.values()))
            return Series(fn(block), index=np.array(numeric_cols), name=name)
        # Multi-dtype: reduce each block, build result in column order
        block_results = {}
        for dtype, block in self._dtype_blocks.items():
            block_results[dtype] = fn(block)
        parts = []
        for col in numeric_cols:
            dtype, idx = self._column_to_block[col]
            parts.append(block_results[dtype][idx : idx + 1])
        return Series(jnp.concatenate(parts), index=np.array(numeric_cols), name=name)

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
            blocks = list(self._dtype_blocks.values())
            if len(blocks) == 1:
                return self._fast_nansum(blocks[0]).sum()
            total = jnp.float32(0.0)
            for block in blocks:
                total = total + self._fast_nansum(block).sum()
            return total

        if axis == 0:
            return self._reduce_axis0(self._fast_nansum, "sum")
        result = jnp.nansum(self._numeric_data, axis=axis)
        return Series(result, index=self._index, name="sum")

    def mean(self, axis: int | None = 0):
        """
        Mean along axis (JIT-compatible).

        Skips NaN values by default (matching pandas behavior).
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns to compute mean")

        if axis is None:
            blocks = list(self._dtype_blocks.values())
            if len(blocks) == 1:
                s, c = self._fast_nansum_and_count(blocks[0])
                return s / jnp.maximum(c, 1)
            # Multi-dtype: concat all blocks (promoted to float32) and reduce once
            combined = jnp.concatenate([b.astype(jnp.float32).reshape(-1) for b in blocks])
            s, c = self._fast_nansum_and_count(combined.reshape(1, -1))
            return s / jnp.maximum(c, 1)

        if axis == 0:
            return self._reduce_axis0(self._fast_nanmean, "mean")
        result = jnp.nanmean(self._numeric_data, axis=axis)
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
            return jnp.sqrt(self.var(axis=None, ddof=ddof))

        if axis == 0:
            ddof_arr = jnp.float32(ddof)
            return self._reduce_axis0(
                lambda b: jnp.sqrt(jnp.maximum(self._fast_nanvar(b, ddof_arr), 0)), "std"
            )
        result = jnp.nanstd(self._numeric_data, axis=axis, ddof=ddof)
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
            blocks = list(self._dtype_blocks.values())
            if len(blocks) == 1:
                s, sq, c = self._fast_nansumsq_and_count(blocks[0])
            else:
                s = jnp.float32(0.0)
                sq = jnp.float32(0.0)
                c = jnp.float32(0.0)
                for block in blocks:
                    bs, bsq, bc = self._fast_nansumsq_and_count(block)
                    s = s + bs
                    sq = sq + bsq
                    c = c + bc
            mean = s / jnp.maximum(c, 1)
            return (sq - c * mean * mean) / jnp.maximum(c - ddof, 1)

        if axis == 0:
            ddof_arr = jnp.float32(ddof)
            return self._reduce_axis0(lambda b: self._fast_nanvar(b, ddof_arr), "var")
        result = jnp.nanvar(self._numeric_data, axis=axis, ddof=ddof)
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
            result = jnp.float32(jnp.inf)
            for block in self._dtype_blocks.values():
                result = jnp.minimum(result, self._fast_nanmin(block).min())
            return result

        if axis == 0:
            return self._reduce_axis0(self._fast_nanmin, "min")
        result = jnp.nanmin(self._numeric_data, axis=axis)
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
            result = jnp.float32(-jnp.inf)
            for block in self._dtype_blocks.values():
                result = jnp.maximum(result, self._fast_nanmax(block).max())
            return result

        if axis == 0:
            return self._reduce_axis0(self._fast_nanmax, "max")
        result = jnp.nanmax(self._numeric_data, axis=axis)
        return Series(result, index=self._index, name="max")

    def prod(self, axis: int | None = 0):
        """
        Product along axis (JIT-compatible, differentiable).

        Skips NaN values by default (matching pandas behavior).
        """
        if not self._dtype_blocks:
            raise ValueError("No numeric columns to compute product")

        if axis is None:
            total = jnp.float32(1.0)
            for block in self._dtype_blocks.values():
                total = total * self._fast_nanprod(block).prod()
            return total

        if axis == 0:
            return self._reduce_axis0(self._fast_nanprod, "prod")
        result = jnp.nanprod(self._numeric_data, axis=axis)
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
        if not self._dtype_blocks:
            raise ValueError("No numeric columns to compute median")
        if axis == 0:
            return self._reduce_axis0(lambda b: jnp.nanmedian(b, axis=0), "median")
        result = jnp.nanmedian(self._numeric_data, axis=axis)
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
        new_blocks = {dt: _fillna_block(block, value) for dt, block in self._dtype_blocks.items()}
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=dict(self._column_to_block),
            object_data=self._object_data,
            index=self._index,
            column_order=list(self._column_order),
        )

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
        """Sort by column values (multi-key lexicographic). JIT-compatible:
        jnp.argsort/lexsort + gather; grad flows through the gather. Index and
        object columns follow the sort when eager, stay put under a trace."""
        if isinstance(by, str):
            by = [by]
        if isinstance(ascending, bool):
            ascending = [ascending] * len(by)
        keys = []
        for col, asc in zip(by, ascending):
            dtype, idx = self._column_to_block[col]
            k = self._dtype_blocks[dtype][:, idx]
            keys.append(k if asc else -k)
        dtypes_order = list(self._dtype_blocks.keys())
        blocks = tuple(self._dtype_blocks[dt] for dt in dtypes_order)
        if len(keys) == 1:
            sorted_blocks, order = _argsort_take_blocks(blocks, keys[0])
        else:
            # lexsort: last key is primary
            sorted_blocks, order = _lexsort_take_blocks(blocks, jnp.stack(keys[::-1]))
        new_blocks = dict(zip(dtypes_order, sorted_blocks))
        if _is_tracer(order):
            new_index, new_obj = self._index, self._object_data
        else:
            pos = np.asarray(order)
            new_index = self._index[pos]
            new_obj = {k: v[pos] for k, v in self._object_data.items()}
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=dict(self._column_to_block),
            object_data=new_obj,
            index=new_index,
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

    def rank(self, ascending=True, method="average", axis=0):
        """Rank values along axis 0, pandas semantics (NaNs get NaN rank).

        JIT-compatible: sort + searchsorted per column (vmapped), no eager
        argsort. Not differentiable (ranks are a step function of the data).

        Args:
            ascending: True for smallest=1 ranking
            method: 'average' (pandas default), 'min', 'max', 'dense',
                    'first'/'ordinal'.
        """

        def _rank_block(block):
            return jax.vmap(
                lambda col: _rank_1d(col, method=method, ascending=ascending),
                in_axes=1,
                out_axes=1,
            )(block)

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
        """Apply function along axis. JIT-compatible if func is JIT-compatible.

        Reductions return a Series; shape-preserving functions return a
        DataFrame (pandas semantics)."""
        data = self.values
        if axis == 0:
            # Apply func to each column
            results = []
            for i in range(data.shape[1]):
                results.append(func(data[:, i]))
            if all(hasattr(r, "shape") and r.shape == (data.shape[0],) for r in results):
                return DataFrame._from_column_arrays(
                    dict(zip(self._numeric_cols, results)),
                    list(self._numeric_cols),
                    self._index,
                )
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
                gathered = _gather_rows_nanfill(self._get_column_data(col), left_idx)
                # Shared key column: coalesce from right for right-only rows
                if col in left_on and col in right_on and col in right._column_to_block:
                    right_gathered = _gather_rows_nanfill(right._get_column_data(col), right_idx)
                    gathered = jnp.where(
                        jnp.asarray(np.asarray(left_idx) < 0), right_gathered, gathered
                    )
                result_data[out_col] = gathered
            elif col in self._object_data:
                result_data[out_col] = _gather_rows_nanfill(self._object_data[col], left_idx)

        # Columns from right (skip shared key)
        for col in right._column_order:
            if col in right_on and col in left_on:
                continue
            out_col = col
            if col not in right_on and col in left_non_key:
                out_col = col + right_suffix
            if col in right._column_to_block:
                result_data[out_col] = _gather_rows_nanfill(right._get_column_data(col), right_idx)
            elif col in right._object_data:
                result_data[out_col] = _gather_rows_nanfill(right._object_data[col], right_idx)

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

    # ========================================
    # Named arithmetic / comparisons (pandas parity)
    # ========================================

    def add(self, other, axis="columns", fill_value=None):
        """Elementwise addition (JIT-compatible, differentiable)."""
        return self._apply_elementwise(other, jnp.add, "add")

    radd = add

    def sub(self, other, axis="columns", fill_value=None):
        """Elementwise subtraction (JIT-compatible, differentiable)."""
        return self._apply_elementwise(other, jnp.subtract, "sub")

    subtract = sub

    def rsub(self, other, axis="columns", fill_value=None):
        return self._apply_elementwise(other, lambda a, b: b - a, "rsub")

    def mul(self, other, axis="columns", fill_value=None):
        """Elementwise multiplication (JIT-compatible, differentiable)."""
        return self._apply_elementwise(other, jnp.multiply, "mul")

    multiply = mul
    rmul = mul

    def div(self, other, axis="columns", fill_value=None):
        """Elementwise division (JIT-compatible, differentiable)."""
        return self._apply_elementwise(other, jnp.true_divide, "div")

    divide = div
    truediv = div

    def rdiv(self, other, axis="columns", fill_value=None):
        return self._apply_elementwise(other, lambda a, b: b / a, "rdiv")

    rtruediv = rdiv

    def floordiv(self, other, axis="columns", fill_value=None):
        return self._apply_elementwise(other, jnp.floor_divide, "floordiv")

    def rfloordiv(self, other, axis="columns", fill_value=None):
        return self._apply_elementwise(other, lambda a, b: b // a, "rfloordiv")

    def mod(self, other, axis="columns", fill_value=None):
        return self._apply_elementwise(other, jnp.mod, "mod")

    def rmod(self, other, axis="columns", fill_value=None):
        return self._apply_elementwise(other, lambda a, b: b % a, "rmod")

    def pow(self, other, axis="columns", fill_value=None):
        return self._apply_elementwise(other, jnp.power, "pow")

    def rpow(self, other, axis="columns", fill_value=None):
        return self._apply_elementwise(other, lambda a, b: b**a, "rpow")

    product = prod

    def dot(self, other):
        """Matrix multiplication (JIT-compatible, differentiable)."""
        return self @ other

    def eq(self, other, axis="columns"):
        """Elementwise equality (JIT-compatible, boolean output)."""
        return self._apply_comparison(other, jnp.equal, "eq")

    def ne(self, other, axis="columns"):
        return self._apply_comparison(other, jnp.not_equal, "ne")

    def lt(self, other, axis="columns"):
        return self._apply_comparison(other, jnp.less, "lt")

    def le(self, other, axis="columns"):
        return self._apply_comparison(other, jnp.less_equal, "le")

    def gt(self, other, axis="columns"):
        return self._apply_comparison(other, jnp.greater, "gt")

    def ge(self, other, axis="columns"):
        return self._apply_comparison(other, jnp.greater_equal, "ge")

    # ========================================
    # Cumulative / missing-data ops
    # ========================================

    def cummax(self, axis: int = 0, skipna=True):
        """Cumulative max, pandas NaN semantics (JIT-compatible)."""
        return self._apply_blockwise(lambda b: _cummax_pandas(b, axis=0))

    def cummin(self, axis: int = 0, skipna=True):
        """Cumulative min, pandas NaN semantics (JIT-compatible)."""
        return self._apply_blockwise(lambda b: _cummin_pandas(b, axis=0))

    def ffill(self, axis: int = 0):
        """Forward-fill NaNs down each column (JIT-compatible, differentiable)."""
        return self._apply_blockwise(lambda b: _ffill_array(b, axis=0))

    pad = ffill

    def bfill(self, axis: int = 0):
        """Backward-fill NaNs down each column (JIT-compatible, differentiable)."""
        return self._apply_blockwise(lambda b: _bfill_array(b, axis=0))

    backfill = bfill

    def dropna(self, axis: int = 0, how="any", subset=None):
        """Drop rows (axis=0) or columns (axis=1) containing NaN. Eager (shape change)."""
        if axis == 1:
            keep = [
                col
                for col in self._column_order
                if col in self._object_data or not bool(jnp.isnan(self._get_column_data(col)).any())
            ]
            return self[keep]
        cols = (
            subset
            if subset is not None
            else [c for c in self._column_order if c in self._column_to_block]
        )
        masks = [np.isnan(np.asarray(self._get_column_data(c))) for c in cols]
        if not masks:
            return self.copy()
        stacked = np.stack(masks, axis=1)
        bad = stacked.any(axis=1) if how == "any" else stacked.all(axis=1)
        pos = np.flatnonzero(~bad)
        return self.take(pos)

    def first_valid_index(self):
        """First index label with any non-NaN value. Eager."""
        valid = ~np.isnan(np.asarray(self._numeric_data)).all(axis=1)
        pos = np.flatnonzero(valid)
        return self._index[pos[0]] if len(pos) else None

    def last_valid_index(self):
        """Last index label with any non-NaN value. Eager."""
        valid = ~np.isnan(np.asarray(self._numeric_data)).all(axis=1)
        pos = np.flatnonzero(valid)
        return self._index[pos[-1]] if len(pos) else None

    def asof(self, where):
        """Per-column last non-NaN value at index <= where. Eager."""
        mask = self._index <= where
        vals = {}
        for col in self._numeric_cols:
            arr = np.asarray(self._get_column_data(col))
            valid = np.flatnonzero(mask & ~np.isnan(arr))
            vals[col] = arr[valid[-1]] if len(valid) else np.nan
        return Series(
            jnp.asarray(list(vals.values())), index=np.asarray(list(vals.keys())), name=where
        )

    # ========================================
    # Function application
    # ========================================

    def agg(self, func=None, axis=0):
        """Aggregate by name, callable, list, or {column: func} dict."""
        if isinstance(func, str):
            return getattr(self, func)()
        if isinstance(func, dict):
            results = {}
            for col, f in func.items():
                s = self[col]
                results[col] = s.agg(f) if not isinstance(f, list | tuple) else s.agg(list(f))
            scalars = {c: v for c, v in results.items() if not isinstance(v, Series)}
            if len(scalars) == len(results):
                return Series(
                    jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in scalars.values()]),
                    index=np.asarray(list(scalars.keys())),
                )
            return results
        if isinstance(func, list | tuple):
            rows = {}
            for f in func:
                fname = f if isinstance(f, str) else getattr(f, "__name__", str(f))
                s = getattr(self, f)() if isinstance(f, str) else self.apply(f)
                rows[fname] = {c: s[c] for c in s.index.tolist()} if isinstance(s, Series) else {}
            cols = list(self._numeric_cols)
            data = {c: [float(rows[fn].get(c, np.nan)) for fn in rows] for c in cols}
            return DataFrame(data, index=np.asarray(list(rows.keys())))
        return func(self)

    aggregate = agg

    def transform(self, func, axis=0):
        """Apply a shape-preserving function columnwise (JIT-compatible if func is)."""
        if isinstance(func, str):
            result = getattr(self, func)()
        else:
            new_cols = {}
            for col in self._numeric_cols:
                res = func(self._get_column_data(col))
                new_cols[col] = res._data if isinstance(res, Series) else res
            result = DataFrame._from_column_arrays(new_cols, list(self._numeric_cols), self._index)
        if isinstance(result, DataFrame) and result.shape[0] == self.shape[0]:
            return result
        raise ValueError("transform must return a result with the same shape")

    def map(self, func, na_action=None):
        """Apply func elementwise to every value (JIT-compatible when func is
        vectorizable over arrays; falls back to eager loop)."""
        try:
            return self._apply_blockwise(func)
        except Exception:
            new_cols = {}
            for col in self._column_order:
                arr = np.asarray(self._get_column_data(col))
                new_cols[col] = np.asarray([func(v) for v in arr])
            return DataFrame(new_cols, index=self._index)

    # ========================================
    # Selection / structure
    # ========================================

    @property
    def loc(self):
        """Label-based indexing (eager)."""
        return _DataFrameLocIndexer(self)

    @property
    def at(self):
        """Fast label-based scalar access (eager)."""
        return _DataFrameAtIndexer(self)

    @property
    def iat(self):
        """Fast integer-position scalar access (eager)."""
        return _DataFrameIatIndexer(self)

    @property
    def axes(self):
        return [self._index, self._column_order]

    @property
    def attrs(self) -> dict:
        if "_attrs" not in self.__dict__:
            self.__dict__["_attrs"] = {}
        return self.__dict__["_attrs"]

    @attrs.setter
    def attrs(self, value):
        self.__dict__["_attrs"] = dict(value)

    @property
    def flags(self):
        return _Flags()

    def set_flags(self, **kwargs):
        """Return a copy (flags are unused by jaxframe)."""
        return self.copy()

    def keys(self):
        return self._column_order

    def items(self):
        """Iterate over (column, Series) pairs (eager)."""
        for col in self._column_order:
            yield col, self[col]

    def iterrows(self):
        """Iterate over (label, row Series) pairs. Eager."""
        arrays = {col: np.asarray(self._get_column_data(col)) for col in self._column_to_block}
        arrays.update({col: arr for col, arr in self._object_data.items()})
        cols = np.asarray(self._column_order)
        for i, label in enumerate(self._index):
            row = np.asarray([arrays[c][i] for c in self._column_order])
            yield label, Series(row, index=cols)

    def itertuples(self, index=True, name="Pandas"):
        """Iterate over rows as namedtuples. Eager."""
        import collections

        fields = (["Index"] if index else []) + [str(c) for c in self._column_order]
        Row = collections.namedtuple(name, fields, rename=True)
        arrays = {col: np.asarray(self._get_column_data(col)) for col in self._column_to_block}
        arrays.update({col: arr for col, arr in self._object_data.items()})
        for i, label in enumerate(self._index):
            vals = [arrays[c][i] for c in self._column_order]
            yield Row(*(([label] if index else []) + vals))

    def get(self, key, default=None):
        """Column by name, or default."""
        try:
            return self[key]
        except (KeyError, TypeError):
            return default

    def xs(self, key, axis=0):
        """Row by index label (axis=0) or column (axis=1). Eager."""
        if axis == 1:
            return self[key]
        pos = np.flatnonzero(self._index == key)
        if len(pos) == 0:
            raise KeyError(key)
        return self.iloc[int(pos[0])]

    def take(self, indices, axis: int = 0):
        """Gather rows (axis=0) or columns (axis=1) by position.
        JIT-compatible for axis=0 (gather; grad flows)."""
        if axis == 1:
            return self[[self._column_order[int(i)] for i in np.asarray(indices)]]
        idx = jnp.asarray(indices)
        new_blocks = {dt: jnp.take(block, idx, axis=0) for dt, block in self._dtype_blocks.items()}
        if _is_tracer(idx):
            new_index, new_obj = self._index, self._object_data
        else:
            pos = np.asarray(indices)
            new_index = self._index[pos]
            new_obj = {k: v[pos] for k, v in self._object_data.items()}
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=dict(self._column_to_block),
            object_data=new_obj,
            index=new_index,
            column_order=list(self._column_order),
        )

    def sample(self, n=None, frac=None, replace=False, random_state=None, axis=0):
        """Random row sample. Eager, not differentiable."""
        rng = np.random.default_rng(random_state)
        if n is None:
            n = int(round((frac if frac is not None else 1.0) * len(self)))
        pos = rng.choice(len(self), size=n, replace=replace)
        return self.take(pos)

    def filter(self, items=None, like=None, regex=None, axis=1):
        """Filter columns (axis=1) or index labels (axis=0). Eager."""
        if axis == 0:
            labels = self._index
            if items is not None:
                mask = np.isin(labels, np.asarray(items))
            elif like is not None:
                mask = np.array([like in str(x) for x in labels])
            else:
                import re

                pat = re.compile(regex)
                mask = np.array([bool(pat.search(str(x))) for x in labels])
            return self.take(np.flatnonzero(mask))
        cols = self._column_order
        if items is not None:
            keep = [c for c in cols if c in set(items)]
        elif like is not None:
            keep = [c for c in cols if like in str(c)]
        else:
            import re

            pat = re.compile(regex)
            keep = [c for c in cols if pat.search(str(c))]
        return self[keep]

    def squeeze(self, axis=None):
        """Collapse a 1-column frame to a Series (and 1x1 to a scalar)."""
        if len(self._column_order) == 1:
            s = self[self._column_order[0]]
            if len(s) == 1:
                return s._data[0]
            return s
        return self

    def truncate(self, before=None, after=None):
        """Keep rows with index label in [before, after]. Eager."""
        mask = np.ones(len(self._index), dtype=bool)
        if before is not None:
            mask &= self._index >= before
        if after is not None:
            mask &= self._index <= after
        return self.take(np.flatnonzero(mask))

    def add_prefix(self, prefix: str):
        """Prefix all column names."""
        return self.rename(columns={c: f"{prefix}{c}" for c in self._column_order})

    def add_suffix(self, suffix: str):
        """Suffix all column names."""
        return self.rename(columns={c: f"{c}{suffix}" for c in self._column_order})

    def rename_axis(self, mapper=None, **kwargs):
        """No-op copy (axis names are not tracked)."""
        return self.copy()

    def set_axis(self, labels, axis=0):
        """Replace the index (axis=0) or column names (axis=1)."""
        if axis in (1, "columns"):
            return self.rename(columns=dict(zip(self._column_order, labels)))
        return self._replace_index(np.asarray(labels))

    def reindex(self, index=None, columns=None, fill_value=jnp.nan):
        """Conform to new index labels/columns; missing entries get fill_value. Eager."""
        result = self
        if columns is not None:
            data = {}
            for col in columns:
                if col in result._column_order:
                    data[col] = np.asarray(result._get_column_data(col))
                else:
                    data[col] = np.full(len(result), np.nan)
            result = DataFrame(data, index=result._index)
        if index is not None:
            new_index = np.asarray(index)
            lookup = {label: i for i, label in enumerate(result._index.tolist())}
            pos = np.array([lookup.get(label, -1) for label in new_index.tolist()])
            gathered = result.take(np.maximum(pos, 0))
            missing = jnp.asarray(pos < 0)[:, None]
            new_blocks = {
                dt: jnp.where(missing, fill_value, block)
                for dt, block in gathered._dtype_blocks.items()
            }
            result = DataFrame._from_parts(
                dtype_blocks=new_blocks,
                column_to_block=dict(gathered._column_to_block),
                object_data=gathered._object_data,
                index=new_index,
                column_order=list(gathered._column_order),
            )
        return result

    def reindex_like(self, other, fill_value=jnp.nan):
        """Conform to another DataFrame's index and columns. Eager."""
        return self.reindex(index=other._index, columns=other._column_order, fill_value=fill_value)

    # ========================================
    # Mutating column ops (pandas parity)
    # ========================================

    def _replace_self(self, df: "DataFrame"):
        self._dtype_blocks = df._dtype_blocks
        self._column_to_block = df._column_to_block
        self._object_data = df._object_data
        self._index = df._index
        self._column_order = df._column_order
        self.__dict__.pop("_col_cache", None)  # cached column slices are stale now
        self.__dict__.pop("_values_cache", None)

    def __setitem__(self, key, value):
        """Add or replace a column. Eager, mutates self."""
        if not isinstance(key, str):
            raise TypeError(f"column key must be str, got {type(key)}")
        if isinstance(value, Series):
            value = value._data
        elif isinstance(value, DataFrame):
            value = value._get_column_data(value._column_order[0])
        elif np.isscalar(value) or value is None:
            value = np.full(len(self), value)
        self._replace_self(self.assign(**{key: value}))

    def insert(self, loc: int, column, value):
        """Insert a column at position loc. Eager, mutates self."""
        if isinstance(value, Series):
            value = value._data
        new_df = self.assign(**{column: value})
        order = [c for c in new_df._column_order if c != column]
        order.insert(loc, column)
        reordered = new_df[order]
        self._replace_self(reordered)

    def isetitem(self, loc: int, value):
        """Set column by position. Eager, mutates self."""
        self[self._column_order[loc]] = value

    def pop(self, item):
        """Remove a column and return it as a Series. Eager, mutates self."""
        s = self[item]
        self._replace_self(self.drop(columns=[item]))
        return s

    def update(self, other: "DataFrame"):
        """Overwrite with non-NaN values from other (shared columns). Mutates self."""
        new = {}
        for col in self._column_order:
            if col in other._column_order and col in self._column_to_block:
                a = self._get_column_data(col)
                b = other._get_column_data(col)
                new[col] = jnp.where(jnp.isnan(b), a, b)
        result = self.assign(**new)
        self._replace_self(result)

    # ========================================
    # Combining / comparing
    # ========================================

    def join(self, other: "DataFrame", on=None, how="left", lsuffix="", rsuffix=""):
        """Join on index (or a key column with on=). Eager structure discovery,
        JIT-compatible gathers via merge."""
        index_col = "__jaxframe_join_index__"
        right = other.copy()
        right_data = {index_col: np.asarray(other._index)}
        for col in other._column_order:
            arr = (
                other._get_column_data(col)
                if col in other._column_to_block
                else other._object_data[col]
            )
            right_data[col] = np.asarray(arr)
        right = DataFrame(right_data)
        if on is not None:
            left = self.copy()
            merged = left.merge(
                right,
                left_on=on,
                right_on=index_col,
                how=how,
                suffixes=(lsuffix or "_x", rsuffix or "_y"),
            )
        else:
            left_data = {index_col: np.asarray(self._index)}
            for col in self._column_order:
                arr = (
                    self._get_column_data(col)
                    if col in self._column_to_block
                    else self._object_data[col]
                )
                left_data[col] = np.asarray(arr)
            left = DataFrame(left_data)
            merged = left.merge(
                right, on=index_col, how=how, suffixes=(lsuffix or "_x", rsuffix or "_y")
            )
        keep = [c for c in merged._column_order if c != index_col]
        return merged[keep]

    def align(self, other: "DataFrame", join="outer", axis=None, fill_value=jnp.nan):
        """Align two frames on index and columns. Eager."""
        if join == "outer":
            idx = np.union1d(self._index, other._index)
            cols = sorted(set(self._column_order) | set(other._column_order), key=str)
        elif join == "inner":
            idx = np.intersect1d(self._index, other._index)
            cols = [c for c in self._column_order if c in set(other._column_order)]
        elif join == "left":
            idx, cols = self._index, list(self._column_order)
        else:
            idx, cols = other._index, list(other._column_order)
        return (
            self.reindex(index=idx, columns=cols, fill_value=fill_value),
            other.reindex(index=idx, columns=cols, fill_value=fill_value),
        )

    def combine(self, other: "DataFrame", func, fill_value=None):
        """Combine columnwise with another frame via func."""
        cols = sorted(set(self._column_order) | set(other._column_order), key=str)
        data = {}
        for col in cols:
            if col in self._column_order and col in other._column_order:
                a, b = self._get_column_data(col), other._get_column_data(col)
                if fill_value is not None:
                    a = jnp.where(jnp.isnan(a), fill_value, a)
                    b = jnp.where(jnp.isnan(b), fill_value, b)
                result = func(Series(a), Series(b))
                data[col] = result._data if isinstance(result, Series) else result
            elif col in self._column_order:
                data[col] = self._get_column_data(col)
            else:
                data[col] = other._get_column_data(col)
        return DataFrame({c: np.asarray(v) for c, v in data.items()}, index=self._index)

    def combine_first(self, other: "DataFrame"):
        """Fill NaNs from other, union of columns (JIT-compatible data path)."""
        return self.combine(
            other, lambda a, b: Series(jnp.where(jnp.isnan(a._data), b._data, a._data))
        )

    def corrwith(self, other, axis=0):
        """Pairwise Pearson correlation with matching columns of other."""
        if isinstance(other, Series):
            cols = list(self._numeric_cols)
            vals = [_nan_pearson(self._get_column_data(c), other._data) for c in cols]
        else:
            cols = [c for c in self._numeric_cols if c in set(other._column_order)]
            vals = [_nan_pearson(self._get_column_data(c), other._get_column_data(c)) for c in cols]
        return Series(jnp.stack(vals), index=np.asarray(cols))

    def equals(self, other) -> bool:
        """Exact equality including NaN positions. Eager."""
        if not isinstance(other, DataFrame):
            return False
        if list(self._column_order) != list(other._column_order) or self.shape != other.shape:
            return False
        for col in self._column_order:
            a = np.asarray(
                self._get_column_data(col)
                if col in self._column_to_block
                else self._object_data[col]
            )
            b = np.asarray(
                other._get_column_data(col)
                if col in other._column_to_block
                else other._object_data[col]
            )
            try:
                if not np.array_equal(a, b, equal_nan=True):
                    return False
            except TypeError:
                if not np.array_equal(a, b):
                    return False
        return True

    def compare(self, other: "DataFrame"):
        """Cells that differ, as a frame with <col>_self/<col>_other columns. Eager."""
        diff_rows = np.zeros(len(self), dtype=bool)
        for col in self._numeric_cols:
            a = np.asarray(self._get_column_data(col))
            b = np.asarray(other._get_column_data(col))
            diff_rows |= ~((a == b) | (np.isnan(a) & np.isnan(b)))
        pos = np.flatnonzero(diff_rows)
        data = {}
        for col in self._numeric_cols:
            a = np.asarray(self._get_column_data(col))[pos]
            b = np.asarray(other._get_column_data(col))[pos]
            cell_diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            data[f"{col}_self"] = np.where(cell_diff, a, np.nan)
            data[f"{col}_other"] = np.where(cell_diff, b, np.nan)
        return DataFrame(data, index=self._index[pos])

    def replace(self, to_replace, value=None):
        """Replace values (scalar or {old: new} dict). JIT-compatible data path."""
        if isinstance(to_replace, dict):
            pairs = list(to_replace.items())
        else:
            pairs = [(to_replace, value)]

        def _replace_block(block):
            out = block
            for old, new in pairs:
                out = jnp.where(out == old, new, out)
            return out

        return self._apply_blockwise(_replace_block)

    # ========================================
    # Reshaping
    # ========================================

    def stack(self, level=-1):
        """Stack columns into a Series with (row, col) tuple index. Eager."""
        vals, labels = [], []
        for i, label in enumerate(self._index):
            for col in self._column_order:
                vals.append(float(np.asarray(self._get_column_data(col))[i]))
                labels.append((label, col))
        return Series(jnp.asarray(vals), index=np.asarray(labels, dtype=object), name=None)

    def unstack(self, level=-1):
        """Unstack into a Series with (col, row) tuple index. Eager."""
        vals, labels = [], []
        for col in self._column_order:
            arr = np.asarray(self._get_column_data(col))
            for i, label in enumerate(self._index):
                vals.append(float(arr[i]))
                labels.append((col, label))
        return Series(jnp.asarray(vals), index=np.asarray(labels, dtype=object), name=None)

    def _column_values(self, col) -> np.ndarray:
        """Numeric or object column as a numpy array (eager helper)."""
        if col in self._column_to_block:
            return np.asarray(self._get_column_data(col))
        return np.asarray(self._object_data[col])

    def pivot(self, columns=None, index=None, values=None):
        """Reshape by unique column values. Eager (structure discovery)."""
        col_keys = self._column_values(columns)
        idx_vals = self._column_values(index) if index is not None else np.asarray(self._index)
        val_cols = (
            values
            if values is not None
            else [c for c in self._column_order if c not in (columns, index)]
        )
        if isinstance(val_cols, str):
            val_cols = [val_cols]
        uniq_cols = np.unique(col_keys)
        uniq_idx = np.unique(idx_vals)
        data = {}
        for vc in val_cols:
            src = np.asarray(self._get_column_data(vc))
            for uc in uniq_cols:
                out = np.full(len(uniq_idx), np.nan)
                sel = col_keys == uc
                row_pos = np.searchsorted(uniq_idx, idx_vals[sel])
                out[row_pos] = src[sel]
                name = uc if len(val_cols) == 1 else f"{vc}_{uc}"
                data[str(name)] = out
        return DataFrame(data, index=uniq_idx)

    def explode(self, column, ignore_index=False):
        """Flatten a list-valued object column into rows. Eager."""
        arr = self._object_data[column]
        counts = [len(v) if isinstance(v, list | tuple | np.ndarray) else 1 for v in arr]
        row_pos = np.repeat(np.arange(len(arr)), counts)
        exploded = []
        for v in arr:
            if isinstance(v, list | tuple | np.ndarray):
                exploded.extend(v)
            else:
                exploded.append(v)
        base = self.take(row_pos)
        data = {}
        for col in self._column_order:
            if col == column:
                data[col] = np.asarray(exploded, dtype=object)
            elif col in base._column_to_block:
                data[col] = np.asarray(base._get_column_data(col))
            else:
                data[col] = base._object_data[col]
        index = np.arange(len(row_pos)) if ignore_index else self._index[row_pos]
        return DataFrame(data, index=index)

    def value_counts(self, subset=None, sort=True, ascending=False, dropna=True):
        """Count unique rows. Eager (structure discovery)."""
        cols = subset if subset is not None else list(self._column_order)
        arrays = [np.asarray(self._get_column_data(c)) for c in cols]
        rows = np.stack(arrays, axis=1)
        if dropna:
            rows = rows[~np.isnan(rows).any(axis=1)]
        uniq, counts = np.unique(rows, axis=0, return_counts=True)
        if sort:
            order = np.argsort(counts if ascending else -counts, kind="stable")
            uniq, counts = uniq[order], counts[order]
        labels = np.asarray([tuple(r) for r in uniq], dtype=object) if len(cols) > 1 else uniq[:, 0]
        return Series(jnp.asarray(counts), index=labels, name="count")

    # ========================================
    # Expression evaluation
    # ========================================

    def _eval_env(self, local_dict=None):
        env = {"jnp": jnp, "np": np, "abs": abs, "min": min, "max": max}
        for col in self._column_order:
            if str(col).isidentifier():
                env[str(col)] = self[col]
        if local_dict:
            env.update(local_dict)
        return env

    def eval(self, expr: str, local_dict=None, **kwargs):
        """Evaluate an expression over columns. Supports 'new_col = expr'
        assignment form. JIT-compatible data path (Python-parsed, JAX ops)."""
        expr = expr.strip()
        # Assignment form (single =, not ==/<=/>=/!=)
        import re

        m = re.match(r"^([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$", expr, re.DOTALL)
        if m:
            target, rhs = m.group(1), m.group(2)
            result = eval(rhs, {"__builtins__": {}}, self._eval_env(local_dict))  # noqa: S307
            value = result._data if isinstance(result, Series) else result
            return self.assign(**{target: value})
        result = eval(expr, {"__builtins__": {}}, self._eval_env(local_dict))  # noqa: S307
        return result

    def query(self, expr: str, local_dict=None, **kwargs):
        """Filter rows by a boolean expression over columns. Eager (shape change)."""
        result = eval(  # noqa: S307
            expr, {"__builtins__": {}}, self._eval_env(local_dict)
        )
        mask = np.asarray(result._data if isinstance(result, Series) else result)
        return self.take(np.flatnonzero(mask))

    # ========================================
    # Construction / conversion
    # ========================================

    @classmethod
    def from_dict(cls, data, orient="columns"):
        """Construct from a dict of columns (or rows with orient='index')."""
        if orient == "columns":
            return cls(dict(data))
        if orient == "index":
            rows = list(data.values())
            index = np.asarray(list(data.keys()))
            n_cols = len(rows[0])
            cols = {i: np.asarray([r[i] for r in rows]) for i in range(n_cols)}
            return cls({str(k): v for k, v in cols.items()}, index=index)
        raise ValueError(f"orient must be 'columns' or 'index', got {orient!r}")

    @classmethod
    def from_records(cls, data, columns=None):
        """Construct from a list of records (dicts or tuples)."""
        if len(data) == 0:
            return cls({})
        first = data[0]
        if isinstance(first, dict):
            cols = columns if columns is not None else list(first.keys())
            return cls({c: np.asarray([r.get(c) for r in data]) for c in cols})
        n = len(first)
        cols = columns if columns is not None else list(range(n))
        return cls({str(c): np.asarray([r[i] for r in data]) for i, c in enumerate(cols)})

    def to_dict(self, orient="dict"):
        """Convert to dict. Supports 'dict', 'list', 'records' orients. Eager."""
        cols = {
            col: np.asarray(
                self._get_column_data(col)
                if col in self._column_to_block
                else self._object_data[col]
            ).tolist()
            for col in self._column_order
        }
        if orient == "list":
            return cols
        if orient == "records":
            return [{c: cols[c][i] for c in self._column_order} for i in range(len(self))]
        return {c: dict(zip(self._index.tolist(), cols[c])) for c in self._column_order}

    def to_records(self, index=True):
        """Convert to a numpy record array. Eager."""
        return self.to_pandas().to_records(index=index)

    def memory_usage(self, index=True, deep=False):
        """Bytes per column, as a Series."""
        cols, sizes = [], []
        if index:
            cols.append("Index")
            sizes.append(self._index.nbytes)
        for col in self._column_order:
            arr = np.asarray(
                self._get_column_data(col)
                if col in self._column_to_block
                else self._object_data[col]
            )
            cols.append(col)
            sizes.append(arr.nbytes)
        return Series(jnp.asarray(sizes), index=np.asarray(cols))

    def info(self, **kwargs):
        """Print a concise summary (delegates to pandas)."""
        return self.to_pandas().info(**kwargs)

    def convert_dtypes(self):
        """No-op copy (jaxframe already uses concrete dtypes)."""
        return self.copy()

    def infer_objects(self):
        """No-op copy (jaxframe already uses concrete dtypes)."""
        return self.copy()

    def _delegate_pandas(self, method, *args, **kwargs):
        return getattr(self.to_pandas(), method)(*args, **kwargs)

    def to_json(self, *a, **k):
        """Write JSON via pandas (eager I/O)."""
        return self._delegate_pandas("to_json", *a, **k)

    def to_html(self, *a, **k):
        return self._delegate_pandas("to_html", *a, **k)

    def to_string(self, *a, **k):
        return self._delegate_pandas("to_string", *a, **k)

    def to_markdown(self, *a, **k):
        return self._delegate_pandas("to_markdown", *a, **k)

    def to_latex(self, *a, **k):
        return self._delegate_pandas("to_latex", *a, **k)

    def to_excel(self, *a, **k):
        return self._delegate_pandas("to_excel", *a, **k)

    def to_parquet(self, *a, **k):
        return self._delegate_pandas("to_parquet", *a, **k)

    def to_feather(self, *a, **k):
        return self._delegate_pandas("to_feather", *a, **k)

    def to_pickle(self, *a, **k):
        return self._delegate_pandas("to_pickle", *a, **k)

    def to_sql(self, *a, **k):
        return self._delegate_pandas("to_sql", *a, **k)

    def to_hdf(self, *a, **k):
        return self._delegate_pandas("to_hdf", *a, **k)

    def to_stata(self, *a, **k):
        return self._delegate_pandas("to_stata", *a, **k)

    def to_orc(self, *a, **k):
        return self._delegate_pandas("to_orc", *a, **k)

    def to_xml(self, *a, **k):
        return self._delegate_pandas("to_xml", *a, **k)

    def to_clipboard(self, *a, **k):
        return self._delegate_pandas("to_clipboard", *a, **k)

    def to_xarray(self):
        return self._delegate_pandas("to_xarray")

    @property
    def plot(self):
        """Plotting accessor (delegates to pandas)."""
        return self.to_pandas().plot

    def hist(self, *a, **k):
        """Histogram plot (delegates to pandas)."""
        return self._delegate_pandas("hist", *a, **k)

    def boxplot(self, *a, **k):
        """Box plot (delegates to pandas)."""
        return self._delegate_pandas("boxplot", *a, **k)

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
        """Interpolate NaN values, pandas semantics: interior NaNs linear,
        trailing NaNs forward-filled, leading NaNs kept. JIT-compatible and
        differentiable (no dynamic shapes)."""
        if method != "linear":
            raise ValueError(f"Only 'linear' interpolation supported, got {method!r}")
        return self._apply_blockwise(lambda b: _interpolate_array(b, axis=0))

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
            if abs(periods) >= n_rows:
                return jnp.full_like(block, fill_value)
            rolled = jnp.roll(block, periods, axis=0)
            if periods > 0:
                return rolled.at[:periods].set(fill_value)
            else:
                return rolled.at[periods:].set(fill_value)

        return self._apply_blockwise(_shift_block)

    def diff(self, periods: int = 1):
        """
        Calculate first discrete difference (JIT-compatible, differentiable).

        Computes self - self.shift(periods). NaN where shift introduces gaps.
        """
        shifted = self.shift(periods)
        new_blocks = {}
        new_col_to_block = {}
        for (orig_dtype, block), (shifted_dtype, shifted_block) in zip(
            self._dtype_blocks.items(), shifted._dtype_blocks.items()
        ):
            result = block - shifted_block
            new_blocks[result.dtype] = result
            for col, (col_dtype, idx) in self._column_to_block.items():
                if col_dtype == orig_dtype:
                    new_col_to_block[col] = (result.dtype, idx)
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._object_data,
            index=self._index,
            column_order=self._column_order,
        )

    def pct_change(self, periods: int = 1):
        """
        Calculate percentage change (JIT-compatible, differentiable).

        Computes (self - self.shift(periods)) / self.shift(periods).
        """
        shifted = self.shift(periods)
        new_blocks = {}
        new_col_to_block = {}
        for (orig_dtype, block), (shifted_dtype, shifted_block) in zip(
            self._dtype_blocks.items(), shifted._dtype_blocks.items()
        ):
            result = (block - shifted_block) / shifted_block
            new_blocks[result.dtype] = result
            for col, (col_dtype, idx) in self._column_to_block.items():
                if col_dtype == orig_dtype:
                    new_col_to_block[col] = (result.dtype, idx)
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._object_data,
            index=self._index,
            column_order=self._column_order,
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

        result_data = {}
        cols = []
        if isinstance(condition, DataFrame):
            # Apply mask column by column (JIT-safe: no __init__ reconstruction)
            for col in self._column_order:
                if col in self._column_to_block and col in condition._column_to_block:
                    col_data = self._get_column_data(col)
                    cond_bool = condition._get_column_data(col).astype(bool)
                    # mask is inverse of where: replace where condition is True
                    result_data[col] = jnp.where(~cond_bool, col_data, fill_value)
                    cols.append(col)
        else:
            # Scalar or array condition - apply to all columns
            mask = jnp.asarray(condition).astype(bool)
            for col in self._column_order:
                if col in self._column_to_block:
                    col_data = self._get_column_data(col)
                    result_data[col] = jnp.where(~mask, col_data, fill_value)
                    cols.append(col)
        return DataFrame._from_column_arrays(result_data, cols, self._index)

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
        """Get data for a single column as 1D array.

        Column slices are cached per frame: JAX arrays are immutable and every
        mutating DataFrame op goes through _replace_self (which clears the
        cache), so a cached slice can never go stale. This turns repeated
        column access from a device dispatch into a dict lookup."""
        if col_name in self._column_to_block:
            cache = self.__dict__.get("_col_cache")
            if cache is None:
                cache = self.__dict__["_col_cache"] = {}
            arr = cache.get(col_name)
            if arr is None:
                dtype, idx = self._column_to_block[col_name]
                arr = cache[col_name] = self._dtype_blocks[dtype][:, idx]
            return arr
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
        if isinstance(other, int | float | bool | np.number):
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

        elif isinstance(other, np.ndarray | jnp.ndarray):
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
        if isinstance(other, int | float | bool | np.number):
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

        elif isinstance(other, np.ndarray | jnp.ndarray):
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


class _DataFrameLocIndexer:
    """Label-based indexing for DataFrame (eager)."""

    def __init__(self, df):
        self._df = df

    def _row_positions(self, key):
        df = self._df
        if isinstance(key, Series):
            key = np.asarray(key._data)
        if isinstance(key, np.ndarray | jnp.ndarray) and np.asarray(key).dtype == bool:
            return np.flatnonzero(np.asarray(key)), False
        if isinstance(key, slice):
            if key.start is None and key.stop is None:
                return np.arange(len(df._index)), False
            start = 0 if key.start is None else int(np.flatnonzero(df._index == key.start)[0])
            stop = (
                len(df._index)
                if key.stop is None
                else int(np.flatnonzero(df._index == key.stop)[0]) + 1
            )
            return np.arange(start, stop), False
        if isinstance(key, list | np.ndarray):
            lookup = {label: i for i, label in enumerate(df._index.tolist())}
            return np.array([lookup[k] for k in key]), False
        pos = np.flatnonzero(df._index == key)
        if len(pos) == 0:
            raise KeyError(key)
        return np.array([pos[0]]), True

    def __getitem__(self, key):
        df = self._df
        col_key = None
        if isinstance(key, tuple) and len(key) == 2:
            key, col_key = key
        pos, scalar_row = self._row_positions(key)
        result = df.take(pos)
        if col_key is not None:
            if isinstance(col_key, str):
                series = result[col_key]
                return series._data[0] if scalar_row else series
            if isinstance(col_key, slice):
                cols = df._column_order
                start = 0 if col_key.start is None else cols.index(col_key.start)
                stop = len(cols) if col_key.stop is None else cols.index(col_key.stop) + 1
                result = result[cols[start:stop]]
            else:
                result = result[list(col_key)]
        if scalar_row:
            return result.iloc[0]
        return result


class _DataFrameAtIndexer:
    """Label-based scalar access (eager)."""

    def __init__(self, df):
        self._df = df

    def __getitem__(self, key):
        row_label, col = key
        pos = np.flatnonzero(self._df._index == row_label)
        if len(pos) == 0:
            raise KeyError(row_label)
        return self._df._get_column_data(col)[int(pos[0])]


class _DataFrameIatIndexer:
    """Integer-position scalar access (eager)."""

    def __init__(self, df):
        self._df = df

    def __getitem__(self, key):
        i, j = key
        col = self._df._column_order[int(j)]
        return self._df._get_column_data(col)[int(i)]


# ============================
# Shared JIT-compatible kernels
# ============================

_MISSING = object()  # sentinel: distinguish "not passed" from None


class _Flags:
    """Minimal pandas.Flags stand-in (jaxframe does not use flags)."""

    allows_duplicate_labels = True

    def __repr__(self):
        return "<Flags(allows_duplicate_labels=True)>"


def _is_tracer(x) -> bool:
    """True if x is a JAX tracer (inside a jit/grad trace)."""
    return isinstance(x, jax.core.Tracer)


@jax.jit
def _argsort_take_blocks(blocks, key):
    """Fused argsort + gather for sort_values: one compiled kernel instead of
    one dispatch per block. blocks is a tuple pytree; returns (blocks, order)."""
    order = jnp.argsort(key, stable=True)
    return tuple(jnp.take(b, order, axis=0) for b in blocks), order


@jax.jit
def _lexsort_take_blocks(blocks, keys):
    """Fused lexsort + gather (multi-key sort_values)."""
    order = jnp.lexsort(keys)
    return tuple(jnp.take(b, order, axis=0) for b in blocks), order


@functools.partial(jax.jit, static_argnames=("axis",))
def _ffill_array(arr, axis: int = 0):
    """Forward-fill NaNs along axis. JIT-compatible, grad flows through gather.

    Uses the running-max-of-valid-indices trick: no dynamic shapes.
    """
    isnan = jnp.isnan(arr)
    n = arr.shape[axis]
    shape = [1] * arr.ndim
    shape[axis] = n
    idx = jnp.arange(n).reshape(shape)
    idx = jnp.broadcast_to(idx, arr.shape)
    valid_idx = jnp.where(isnan, -1, idx)
    last_valid = jax.lax.cummax(valid_idx, axis=axis)
    gathered = jnp.take_along_axis(arr, jnp.maximum(last_valid, 0), axis=axis)
    return jnp.where(last_valid < 0, jnp.nan, gathered)


@functools.partial(jax.jit, static_argnames=("axis",))
def _bfill_array(arr, axis: int = 0):
    """Backward-fill NaNs along axis. JIT-compatible, grad flows through gather."""
    flipped = jnp.flip(arr, axis=axis)
    return jnp.flip(_ffill_array(flipped, axis=axis), axis=axis)


@functools.partial(jax.jit, static_argnames=("axis",))
def _interpolate_array(arr, axis: int = 0):
    """Linear interpolation of interior NaNs along axis; trailing NaNs are
    forward-filled, leading NaNs kept (pandas .interpolate() default).
    JIT-compatible and differentiable."""
    isnan = jnp.isnan(arr)
    n = arr.shape[axis]
    shape = [1] * arr.ndim
    shape[axis] = n
    idx = jnp.broadcast_to(jnp.arange(n).reshape(shape), arr.shape)
    prev_idx = jax.lax.cummax(jnp.where(isnan, -1, idx), axis=axis)
    next_rev = jax.lax.cummax(jnp.flip(jnp.where(isnan, -1, n - 1 - idx), axis=axis), axis=axis)
    next_idx = jnp.flip(next_rev, axis=axis)
    has_prev = prev_idx >= 0
    has_next = next_idx >= 0
    next_pos = jnp.where(has_next, n - 1 - next_idx, 0)
    prev_pos = jnp.maximum(prev_idx, 0)
    prev_val = jnp.take_along_axis(arr, prev_pos, axis=axis)
    next_val = jnp.take_along_axis(arr, next_pos, axis=axis)
    span = jnp.maximum(next_pos - prev_pos, 1)
    frac = (idx - prev_pos) / span
    interp = prev_val + (next_val - prev_val) * frac
    # interior NaN: interpolate; trailing NaN (no next): hold prev; leading NaN: keep NaN
    filled = jnp.where(has_next, interp, prev_val)
    filled = jnp.where(has_prev, filled, jnp.nan)
    return jnp.where(isnan, filled, arr)


@functools.partial(jax.jit, static_argnames=("method", "ascending"))
def _rank_1d(x, method: str = "average", ascending: bool = True):
    """Rank a 1D array, pandas semantics (NaNs get NaN rank).

    JIT-compatible: uses sort + searchsorted, no dynamic shapes.
    """
    v = x if ascending else -x
    isnan = jnp.isnan(v)
    sorted_v = jnp.sort(v)  # NaNs sort to the end
    rank_min = jnp.searchsorted(sorted_v, v, side="left") + 1
    rank_max = jnp.searchsorted(sorted_v, v, side="right")
    if method == "average":
        ranks = (rank_min + rank_max) / 2.0
    elif method == "min":
        ranks = rank_min.astype(jnp.float32)
    elif method == "max":
        ranks = rank_max.astype(jnp.float32)
    elif method == "dense":
        boundary = jnp.concatenate(
            [jnp.ones(1, dtype=jnp.int32), (sorted_v[1:] != sorted_v[:-1]).astype(jnp.int32)]
        )
        dense_sorted = jnp.cumsum(boundary)
        pos = jnp.searchsorted(sorted_v, v, side="left")
        ranks = dense_sorted[pos].astype(jnp.float32)
    elif method in ("first", "ordinal"):
        order = jnp.argsort(v, stable=True)
        ranks = (
            jnp.zeros(v.shape[0], dtype=jnp.float32)
            .at[order]
            .set(jnp.arange(1, v.shape[0] + 1, dtype=jnp.float32))
        )
    else:
        raise ValueError(f"unsupported rank method: {method}")
    return jnp.where(isnan, jnp.nan, ranks.astype(jnp.float32))


@jax.jit
def _nan_pearson(x, y):
    """Pearson correlation over pairwise-complete observations. JIT-compatible."""
    valid = ~(jnp.isnan(x) | jnp.isnan(y))
    n = jnp.sum(valid)
    xv = jnp.where(valid, x, 0.0)
    yv = jnp.where(valid, y, 0.0)
    mx = jnp.sum(xv) / n
    my = jnp.sum(yv) / n
    dx = jnp.where(valid, x - mx, 0.0)
    dy = jnp.where(valid, y - my, 0.0)
    cov = jnp.sum(dx * dy)
    sx = jnp.sqrt(jnp.sum(dx * dx))
    sy = jnp.sqrt(jnp.sum(dy * dy))
    return cov / (sx * sy)


@functools.partial(jax.jit, static_argnames=("ddof",))
def _nan_cov(x, y, ddof: int = 1):
    """Covariance over pairwise-complete observations. JIT-compatible."""
    valid = ~(jnp.isnan(x) | jnp.isnan(y))
    n = jnp.sum(valid)
    xv = jnp.where(valid, x, 0.0)
    yv = jnp.where(valid, y, 0.0)
    mx = jnp.sum(xv) / n
    my = jnp.sum(yv) / n
    dx = jnp.where(valid, x - mx, 0.0)
    dy = jnp.where(valid, y - my, 0.0)
    return jnp.sum(dx * dy) / (n - ddof)


@functools.partial(jax.jit, static_argnames=("axis",))
def _cummax_pandas(arr, axis: int = 0):
    """Cumulative max, pandas NaN semantics (NaN stays NaN, max carries through)."""
    isnan = jnp.isnan(arr)
    filled = jnp.where(isnan, -jnp.inf, arr)
    cm = jax.lax.cummax(filled, axis=axis)
    return jnp.where(isnan, jnp.nan, cm)


@functools.partial(jax.jit, static_argnames=("axis",))
def _cummin_pandas(arr, axis: int = 0):
    """Cumulative min, pandas NaN semantics."""
    isnan = jnp.isnan(arr)
    filled = jnp.where(isnan, jnp.inf, arr)
    cm = jax.lax.cummin(filled, axis=axis)
    return jnp.where(isnan, jnp.nan, cm)


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
        if isinstance(data, list | np.ndarray):
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

    def block_until_ready(self):
        """Block until underlying JAX array is materialized."""
        if hasattr(self._data, "block_until_ready"):
            self._data.block_until_ready()
        return self

    @property
    def name(self):
        """Return series name."""
        return self._name

    @property
    def index(self):
        """Return series index."""
        return self._index

    def sum(self, axis=0, skipna=True):
        """Sum of all values, NaN-aware (JIT-compatible)."""
        return jnp.nansum(self._data) if skipna else jnp.sum(self._data)

    def mean(self, axis=0, skipna=True):
        """Mean of all values, NaN-aware (JIT-compatible)."""
        return jnp.nanmean(self._data) if skipna else jnp.mean(self._data)

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

    # ============================
    # Internal helpers
    # ============================

    def _new(self, data, index=None, name=_MISSING):
        """Build a Series preserving index/name unless overridden."""
        return Series(
            data,
            index=self._index if index is None else index,
            name=self._name if name is _MISSING else name,
        )

    # ============================
    # Properties (pandas parity)
    # ============================

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def shape(self):
        return (len(self._data),)

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def dtypes(self):
        return self._data.dtype

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    @property
    def T(self):
        return self

    def transpose(self):
        """Return self — a Series is 1D (JIT-compatible)."""
        return self

    @property
    def axes(self):
        return [self._index]

    @property
    def array(self):
        return self._data

    @property
    def nbytes(self) -> int:
        return np.asarray(self._data).nbytes

    @property
    def hasnans(self) -> bool:
        if not jnp.issubdtype(self._data.dtype, jnp.floating):
            return False
        return bool(jnp.any(jnp.isnan(self._data)))

    @property
    def is_unique(self) -> bool:
        arr = np.asarray(self._data)
        return len(np.unique(arr)) == len(arr)

    @property
    def is_monotonic_increasing(self) -> bool:
        arr = self._data
        if len(arr) <= 1:
            return True
        return bool(jnp.all(arr[1:] >= arr[:-1]))

    @property
    def is_monotonic_decreasing(self) -> bool:
        arr = self._data
        if len(arr) <= 1:
            return True
        return bool(jnp.all(arr[1:] <= arr[:-1]))

    @property
    def attrs(self) -> dict:
        if not hasattr(self, "_attrs"):
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        self._attrs = dict(value)

    @property
    def flags(self):
        return _Flags()

    def set_flags(self, **kwargs):
        """Return a copy (flags are not used by jaxframe)."""
        return self.copy()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(np.asarray(self._data).tolist())

    def __getitem__(self, key):
        if isinstance(key, Series):
            key = np.asarray(key._data)
        if isinstance(key, np.ndarray | jnp.ndarray) and key.dtype == bool:
            mask = np.asarray(key)
            return Series(self._data[jnp.asarray(mask)], index=self._index[mask], name=self._name)
        if isinstance(key, slice):
            return Series(self._data[key], index=self._index[key], name=self._name)
        if isinstance(key, list | np.ndarray | jnp.ndarray):
            idx = jnp.asarray(key)
            return Series(
                jnp.take(self._data, idx, axis=0),
                index=self._index[np.asarray(key)],
                name=self._name,
            )
        # Label lookup first; fall back to positional for default integer index
        matches = np.flatnonzero(self._index == key)
        if len(matches) == 1:
            return self._data[int(matches[0])]
        if isinstance(key, int | np.integer):
            return self._data[int(key)]
        raise KeyError(key)

    @property
    def iloc(self):
        return _SeriesILocIndexer(self)

    @property
    def loc(self):
        return _SeriesLocIndexer(self)

    @property
    def at(self):
        return _SeriesLocIndexer(self)

    @property
    def iat(self):
        return _SeriesILocIndexer(self)

    def keys(self):
        return self._index

    def items(self):
        vals = np.asarray(self._data).tolist()
        return zip(self._index.tolist(), vals)

    def item(self):
        if len(self._data) != 1:
            raise ValueError("can only convert an array of size 1 to a Python scalar")
        return np.asarray(self._data).item()

    # ============================
    # Reductions (NaN-aware, JIT-compatible)
    # ============================

    def min(self, axis=0, skipna=True):
        """Minimum (JIT-compatible)."""
        return jnp.nanmin(self._data) if skipna else jnp.min(self._data)

    def max(self, axis=0, skipna=True):
        """Maximum (JIT-compatible)."""
        return jnp.nanmax(self._data) if skipna else jnp.max(self._data)

    def prod(self, axis=0, skipna=True):
        """Product (JIT-compatible, differentiable)."""
        return jnp.nanprod(self._data) if skipna else jnp.prod(self._data)

    product = prod

    def std(self, axis=0, ddof: int = 1, skipna=True):
        """Standard deviation (JIT-compatible, differentiable)."""
        return jnp.nanstd(self._data, ddof=ddof)

    def var(self, axis=0, ddof: int = 1, skipna=True):
        """Variance (JIT-compatible, differentiable)."""
        return jnp.nanvar(self._data, ddof=ddof)

    def sem(self, axis=0, ddof: int = 1):
        """Standard error of the mean (JIT-compatible, differentiable)."""
        n = jnp.sum(~jnp.isnan(self._data))
        return jnp.nanstd(self._data, ddof=ddof) / jnp.sqrt(n)

    def median(self, axis=0, skipna=True):
        """Median (JIT-compatible)."""
        return jnp.nanmedian(self._data)

    def quantile(self, q=0.5):
        """Quantile(s) (JIT-compatible). Scalar q -> scalar, list q -> Series."""
        if isinstance(q, list | tuple | np.ndarray):
            vals = jnp.nanquantile(self._data, jnp.asarray(q))
            return Series(vals, index=np.asarray(q), name=self._name)
        return jnp.nanquantile(self._data, q)

    def count(self):
        """Number of non-NaN observations (JIT-compatible)."""
        if not jnp.issubdtype(self._data.dtype, jnp.floating):
            return len(self._data)
        return jnp.sum(~jnp.isnan(self._data))

    def nunique(self, dropna=True) -> int:
        """Number of unique values. Eager (structure discovery)."""
        arr = np.asarray(self._data)
        if dropna and np.issubdtype(arr.dtype, np.floating):
            arr = arr[~np.isnan(arr)]
        return len(np.unique(arr))

    def all(self, axis=0):
        """Whether all elements are truthy (JIT-compatible, boolean output)."""
        return jnp.all(self._data != 0)

    def any(self, axis=0):
        """Whether any element is truthy (JIT-compatible, boolean output)."""
        return jnp.any(self._data != 0)

    def skew(self, axis=0):
        """Bias-corrected skewness, pandas semantics (JIT-compatible, differentiable)."""
        data = self._data
        n = jnp.sum(~jnp.isnan(data))
        mean = jnp.nanmean(data)
        m2 = jnp.nansum((data - mean) ** 2)
        m3 = jnp.nansum((data - mean) ** 3)
        s2 = m2 / (n - 1)
        return (m3 / n) / (s2**1.5) * (n**2) / ((n - 1) * (n - 2))

    def kurt(self, axis=0):
        """Bias-corrected excess kurtosis, pandas semantics (JIT-compatible)."""
        data = self._data
        n = jnp.sum(~jnp.isnan(data))
        mean = jnp.nanmean(data)
        m2 = jnp.nansum((data - mean) ** 2)
        m4 = jnp.nansum((data - mean) ** 4)
        s2 = m2 / (n - 1)
        adj = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        return adj * (m4 / (s2**2)) - 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))

    kurtosis = kurt

    def mode(self):
        """All modal values, ascending (eager unique)."""
        arr = np.asarray(self._data)
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr[~np.isnan(arr)]
        vals, counts = np.unique(arr, return_counts=True)
        modes = np.sort(vals[counts == counts.max()])
        return Series(modes, index=np.arange(len(modes)), name=self._name)

    def argmin(self, axis=0, skipna=True):
        """Position of minimum (JIT-compatible, discrete output)."""
        return jnp.nanargmin(self._data)

    def argmax(self, axis=0, skipna=True):
        """Position of maximum (JIT-compatible, discrete output)."""
        return jnp.nanargmax(self._data)

    def idxmin(self, axis=0, skipna=True):
        """Index label of minimum. Eager label lookup."""
        return self._index[int(jnp.nanargmin(self._data))]

    def idxmax(self, axis=0, skipna=True):
        """Index label of maximum. Eager label lookup."""
        return self._index[int(jnp.nanargmax(self._data))]

    def argsort(self, ascending=True):
        """Positions that would sort the Series (JIT-compatible, discrete)."""
        order = jnp.argsort(self._data if ascending else -self._data, stable=True)
        return self._new(order)

    def autocorr(self, lag: int = 1):
        """Lag-N autocorrelation, pairwise-complete (JIT-compatible for static lag)."""
        x = self._data[:-lag] if lag > 0 else self._data
        y = self._data[lag:] if lag > 0 else self._data
        return _nan_pearson(x, y)

    def corr(self, other, method="pearson"):
        """Pearson correlation with another Series (JIT-compatible)."""
        if method != "pearson":
            raise NotImplementedError(f"corr method {method!r} not supported")
        other_data = other._data if isinstance(other, Series) else jnp.asarray(other)
        return _nan_pearson(self._data, other_data)

    def cov(self, other, ddof: int = 1):
        """Covariance with another Series (JIT-compatible, differentiable)."""
        other_data = other._data if isinstance(other, Series) else jnp.asarray(other)
        return _nan_cov(self._data, other_data, ddof=ddof)

    def dot(self, other):
        """Dot product (JIT-compatible, differentiable)."""
        other_data = other._data if isinstance(other, Series) else jnp.asarray(other)
        return jnp.dot(self._data, other_data)

    def __matmul__(self, other):
        return self.dot(other)

    def describe(self):
        """Summary statistics (eager assembly of JIT-computed stats)."""
        d = self._data
        stats = [
            jnp.sum(~jnp.isnan(d)),
            jnp.nanmean(d),
            jnp.nanstd(d, ddof=1),
            jnp.nanmin(d),
            jnp.nanquantile(d, 0.25),
            jnp.nanquantile(d, 0.5),
            jnp.nanquantile(d, 0.75),
            jnp.nanmax(d),
        ]
        labels = np.array(["count", "mean", "std", "min", "25%", "50%", "75%", "max"])
        return Series(
            jnp.stack([jnp.asarray(s, dtype=jnp.float32) for s in stats]),
            index=labels,
            name=self._name,
        )

    # ============================
    # Cumulative ops (JIT-compatible)
    # ============================

    def cumsum(self, axis=0, skipna=True):
        """Cumulative sum, NaN-aware (JIT-compatible, differentiable)."""
        d = self._data
        if jnp.issubdtype(d.dtype, jnp.floating):
            isnan = jnp.isnan(d)
            out = jnp.cumsum(jnp.where(isnan, 0.0, d))
            return self._new(jnp.where(isnan, jnp.nan, out))
        return self._new(jnp.cumsum(d))

    def cumprod(self, axis=0, skipna=True):
        """Cumulative product, NaN-aware (JIT-compatible, differentiable)."""
        d = self._data
        if jnp.issubdtype(d.dtype, jnp.floating):
            isnan = jnp.isnan(d)
            out = jnp.cumprod(jnp.where(isnan, 1.0, d))
            return self._new(jnp.where(isnan, jnp.nan, out))
        return self._new(jnp.cumprod(d))

    def cummax(self, axis=0, skipna=True):
        """Cumulative max, NaN-aware (JIT-compatible)."""
        if jnp.issubdtype(self._data.dtype, jnp.floating):
            return self._new(_cummax_pandas(self._data))
        return self._new(jax.lax.cummax(self._data, axis=0))

    def cummin(self, axis=0, skipna=True):
        """Cumulative min, NaN-aware (JIT-compatible)."""
        if jnp.issubdtype(self._data.dtype, jnp.floating):
            return self._new(_cummin_pandas(self._data))
        return self._new(jax.lax.cummin(self._data, axis=0))

    # ============================
    # Elementwise ops (JIT-compatible)
    # ============================

    def round(self, decimals: int = 0):
        """Round to given decimals (JIT-compatible)."""
        return self._new(jnp.round(self._data, decimals))

    def clip(self, lower=None, upper=None):
        """Clip values (JIT-compatible, differentiable)."""
        return self._new(jnp.clip(self._data, lower, upper))

    def isna(self):
        """Boolean mask of NaNs (JIT-compatible)."""
        if not jnp.issubdtype(self._data.dtype, jnp.floating):
            return self._new(jnp.zeros(len(self._data), dtype=bool))
        return self._new(jnp.isnan(self._data))

    isnull = isna

    def notna(self):
        """Boolean mask of non-NaNs (JIT-compatible)."""
        return self._new(~self.isna()._data)

    notnull = notna

    def fillna(self, value):
        """Fill NaNs with a scalar or aligned Series (JIT-compatible)."""
        fill = value._data if isinstance(value, Series) else value
        return self._new(jnp.where(jnp.isnan(self._data), fill, self._data))

    def ffill(self, axis=0):
        """Forward-fill NaNs (JIT-compatible, differentiable)."""
        return self._new(_ffill_array(self._data))

    pad = ffill

    def bfill(self, axis=0):
        """Backward-fill NaNs (JIT-compatible, differentiable)."""
        return self._new(_bfill_array(self._data))

    backfill = bfill

    def interpolate(self, method="linear"):
        """Linear interpolation of NaNs (JIT-compatible, differentiable)."""
        if method != "linear":
            raise NotImplementedError(f"interpolate method {method!r} not supported")
        return self._new(_interpolate_array(self._data))

    def where(self, cond, other=jnp.nan):
        """Keep values where cond is True, else other (JIT-compatible)."""
        if callable(cond):
            cond = cond(self)
        cond_data = cond._data if isinstance(cond, Series) else jnp.asarray(cond)
        other_data = other._data if isinstance(other, Series) else other
        return self._new(jnp.where(cond_data, self._data, other_data))

    def mask(self, cond, other=jnp.nan):
        """Replace values where cond is True (JIT-compatible)."""
        if callable(cond):
            cond = cond(self)
        cond_data = cond._data if isinstance(cond, Series) else jnp.asarray(cond)
        other_data = other._data if isinstance(other, Series) else other
        return self._new(jnp.where(cond_data, other_data, self._data))

    def case_when(self, caselist):
        """Replace values where each condition is True; first match wins
        (JIT-compatible)."""
        out = self._data
        for cond, repl in reversed(caselist):
            if callable(cond):
                cond = cond(self)
            cond_data = cond._data if isinstance(cond, Series) else jnp.asarray(cond)
            repl_data = repl._data if isinstance(repl, Series) else repl
            out = jnp.where(cond_data, repl_data, out)
        return self._new(out)

    def isin(self, values):
        """Boolean mask of membership (JIT-compatible, boolean output)."""
        vals = jnp.asarray(values._data if isinstance(values, Series) else np.asarray(list(values)))
        return self._new(jnp.isin(self._data, vals))

    def astype(self, dtype):
        """Cast to dtype (JIT-compatible)."""
        return self._new(self._data.astype(dtype))

    def copy(self, deep=True):
        """Copy (JAX arrays are immutable; shares data)."""
        return self._new(self._data)

    def convert_dtypes(self):
        """No-op copy (jaxframe already uses concrete dtypes)."""
        return self.copy()

    def infer_objects(self):
        """No-op copy (jaxframe already uses concrete dtypes)."""
        return self.copy()

    # ============================
    # Named arithmetic / comparison (pandas parity)
    # ============================

    def add(self, other, fill_value=None):
        """Elementwise addition (JIT-compatible, differentiable)."""
        return self._binop_filled(other, jnp.add, fill_value)

    radd = add

    def sub(self, other, fill_value=None):
        """Elementwise subtraction (JIT-compatible, differentiable)."""
        return self._binop_filled(other, jnp.subtract, fill_value)

    subtract = sub

    def rsub(self, other, fill_value=None):
        return self._binop_filled(other, lambda a, b: b - a, fill_value)

    def mul(self, other, fill_value=None):
        """Elementwise multiplication (JIT-compatible, differentiable)."""
        return self._binop_filled(other, jnp.multiply, fill_value)

    multiply = mul
    rmul = mul

    def div(self, other, fill_value=None):
        """Elementwise division (JIT-compatible, differentiable)."""
        return self._binop_filled(other, jnp.true_divide, fill_value)

    divide = div
    truediv = div

    def rdiv(self, other, fill_value=None):
        return self._binop_filled(other, lambda a, b: b / a, fill_value)

    rtruediv = rdiv

    def floordiv(self, other, fill_value=None):
        return self._binop_filled(other, jnp.floor_divide, fill_value)

    def rfloordiv(self, other, fill_value=None):
        return self._binop_filled(other, lambda a, b: b // a, fill_value)

    def mod(self, other, fill_value=None):
        return self._binop_filled(other, jnp.mod, fill_value)

    def rmod(self, other, fill_value=None):
        return self._binop_filled(other, lambda a, b: b % a, fill_value)

    def pow(self, other, fill_value=None):
        return self._binop_filled(other, jnp.power, fill_value)

    def rpow(self, other, fill_value=None):
        return self._binop_filled(other, lambda a, b: b**a, fill_value)

    def divmod(self, other):
        """Return (floordiv, mod) pair of Series (JIT-compatible)."""
        return self.floordiv(other), self.mod(other)

    def rdivmod(self, other):
        return self.rfloordiv(other), self.rmod(other)

    def _binop_filled(self, other, op, fill_value):
        a = self._data
        b = other._data if isinstance(other, Series) else other
        if fill_value is not None:
            a = jnp.where(jnp.isnan(a), fill_value, a)
            if isinstance(b, jnp.ndarray | np.ndarray):
                b = jnp.where(jnp.isnan(b), fill_value, b)
        return self._new(op(a, b))

    def eq(self, other):
        """Elementwise equality (JIT-compatible, boolean output)."""
        return self._binop(other, jnp.equal)

    def ne(self, other):
        return self._binop(other, jnp.not_equal)

    def lt(self, other):
        return self._binop(other, jnp.less)

    def le(self, other):
        return self._binop(other, jnp.less_equal)

    def gt(self, other):
        return self._binop(other, jnp.greater)

    def ge(self, other):
        return self._binop(other, jnp.greater_equal)

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

    # ============================
    # Selection / reshaping
    # ============================

    def head(self, n: int = 5):
        """First n rows (JIT-compatible for static n)."""
        return Series(self._data[:n], index=self._index[:n], name=self._name)

    def tail(self, n: int = 5):
        """Last n rows (JIT-compatible for static n)."""
        if n == 0:
            return Series(self._data[:0], index=self._index[:0], name=self._name)
        return Series(self._data[-n:], index=self._index[-n:], name=self._name)

    def take(self, indices, axis=0):
        """Gather by integer position (JIT-compatible, grad flows through gather)."""
        idx = jnp.asarray(indices)
        new_index = self._index[np.asarray(indices)] if not _is_tracer(idx) else self._index
        return Series(jnp.take(self._data, idx, axis=0), index=new_index, name=self._name)

    def repeat(self, repeats, axis=None):
        """Repeat each element (JIT-compatible for static scalar repeats)."""
        return Series(
            jnp.repeat(self._data, repeats),
            index=np.repeat(self._index, repeats),
            name=self._name,
        )

    def drop(self, labels=None, index=None):
        """Drop by index label. Eager (shape change)."""
        labels = labels if labels is not None else index
        if not isinstance(labels, list | np.ndarray | tuple):
            labels = [labels]
        mask = ~np.isin(self._index, np.asarray(labels))
        return Series(self._data[jnp.asarray(mask)], index=self._index[mask], name=self._name)

    def dropna(self):
        """Drop NaN entries. Eager (shape change)."""
        mask = ~np.isnan(np.asarray(self._data))
        return Series(self._data[jnp.asarray(mask)], index=self._index[mask], name=self._name)

    def get(self, key, default=None):
        """Value at index label, or default (eager label lookup)."""
        matches = np.flatnonzero(self._index == key)
        if len(matches) == 0:
            return default
        return self._data[int(matches[0])]

    def xs(self, key, axis=0):
        """Value at index label (eager label lookup)."""
        return self[key]

    def pop(self, item):
        """Remove label and return its value. Eager, mutates self."""
        value = self[item]
        mask = self._index != item
        self._data = self._data[jnp.asarray(mask)]
        self._index = self._index[mask]
        return value

    def filter(self, items=None, like=None, regex=None):
        """Filter by index label. Eager."""
        idx = self._index
        if items is not None:
            mask = np.isin(idx, np.asarray(items))
        elif like is not None:
            mask = np.array([like in str(label) for label in idx])
        elif regex is not None:
            import re

            pat = re.compile(regex)
            mask = np.array([bool(pat.search(str(label))) for label in idx])
        else:
            raise TypeError("Must pass either items, like, or regex")
        return Series(self._data[jnp.asarray(mask)], index=idx[mask], name=self._name)

    def truncate(self, before=None, after=None):
        """Keep rows with label in [before, after]. Eager."""
        mask = np.ones(len(self._index), dtype=bool)
        if before is not None:
            mask &= self._index >= before
        if after is not None:
            mask &= self._index <= after
        return Series(self._data[jnp.asarray(mask)], index=self._index[mask], name=self._name)

    def squeeze(self, axis=None):
        """Scalar if length 1, else self (JIT-compatible)."""
        if len(self._data) == 1:
            return self._data[0]
        return self

    def explode(self, ignore_index=False):
        """Flatten list-like elements. Eager (object data)."""
        arr = np.asarray(self._data, dtype=object)
        out_vals, out_idx = [], []
        for label, v in zip(self._index, arr):
            if isinstance(v, list | tuple | np.ndarray) and len(v) > 0:
                out_vals.extend(v)
                out_idx.extend([label] * len(v))
            else:
                out_vals.append(np.nan if isinstance(v, list | tuple) else v)
                out_idx.append(label)
        index = np.arange(len(out_vals)) if ignore_index else np.asarray(out_idx)
        return Series(np.asarray(out_vals), index=index, name=self._name)

    def sample(self, n=None, frac=None, replace=False, random_state=None):
        """Random sample. Eager, not differentiable."""
        rng = np.random.default_rng(random_state)
        if n is None:
            n = int(round((frac if frac is not None else 1.0) * len(self._data)))
        pos = rng.choice(len(self._data), size=n, replace=replace)
        return Series(
            jnp.take(self._data, jnp.asarray(pos), axis=0),
            index=self._index[pos],
            name=self._name,
        )

    def reindex(self, index=None, fill_value=jnp.nan):
        """Conform to new index labels; missing labels get fill_value. Eager."""
        new_index = np.asarray(index)
        lookup = {label: i for i, label in enumerate(self._index.tolist())}
        pos = np.array([lookup.get(label, -1) for label in new_index.tolist()])
        gathered = jnp.take(self._data, jnp.asarray(np.maximum(pos, 0)), axis=0)
        data = jnp.where(jnp.asarray(pos) < 0, fill_value, gathered)
        return Series(data, index=new_index, name=self._name)

    def reindex_like(self, other, fill_value=jnp.nan):
        """Conform to another Series' index. Eager."""
        return self.reindex(index=other._index, fill_value=fill_value)

    def rename(self, index=None, **kwargs):
        """Rename the Series (scalar) or transform index labels (dict/callable)."""
        if callable(index):
            return Series(
                self._data, index=np.asarray([index(i) for i in self._index]), name=self._name
            )
        if isinstance(index, dict):
            return Series(
                self._data,
                index=np.asarray([index.get(i, i) for i in self._index.tolist()]),
                name=self._name,
            )
        return Series(self._data, index=self._index, name=index)

    def rename_axis(self, mapper=None, **kwargs):
        """No-op copy (index names are not tracked)."""
        return self.copy()

    def add_prefix(self, prefix: str):
        """Prefix index labels."""
        return Series(
            self._data,
            index=np.asarray([f"{prefix}{i}" for i in self._index]),
            name=self._name,
        )

    def add_suffix(self, suffix: str):
        """Suffix index labels."""
        return Series(
            self._data,
            index=np.asarray([f"{i}{suffix}" for i in self._index]),
            name=self._name,
        )

    def set_axis(self, labels, axis=0):
        """Replace the index (JIT-compatible)."""
        return Series(self._data, index=np.asarray(labels), name=self._name)

    def reset_index(self, drop=False, name=None):
        """Reset index to a default RangeIndex."""
        new_index = np.arange(len(self._data))
        if drop:
            return Series(self._data, index=new_index, name=self._name)
        return DataFrame(
            {"index": self._index, self._name or (name or 0): self._data},
            index=new_index,
        )

    def asof(self, where):
        """Last non-NaN value with index label <= where. Eager."""
        mask = (self._index <= where) & ~np.isnan(np.asarray(self._data))
        pos = np.flatnonzero(mask)
        if len(pos) == 0:
            return jnp.nan
        return self._data[int(pos[-1])]

    def first_valid_index(self):
        """Label of first non-NaN entry, or None. Eager."""
        valid = np.flatnonzero(~np.isnan(np.asarray(self._data)))
        return self._index[valid[0]] if len(valid) else None

    def last_valid_index(self):
        """Label of last non-NaN entry, or None. Eager."""
        valid = np.flatnonzero(~np.isnan(np.asarray(self._data)))
        return self._index[valid[-1]] if len(valid) else None

    # ============================
    # Sorting / ranking / uniqueness
    # ============================

    def sort_values(self, ascending=True, na_position="last"):
        """Sort by values. JIT-compatible (jnp.argsort + gather); index follows
        the sort when eager, stays put under a trace (static aux data)."""
        order = jnp.argsort(self._data if ascending else -self._data, stable=True)
        data = jnp.take(self._data, order, axis=0)
        index = self._index if _is_tracer(order) else self._index[np.asarray(order)]
        return Series(data, index=index, name=self._name)

    def sort_index(self, ascending=True):
        """Sort by index labels. Structure discovery is eager; gather is JIT."""
        order = np.argsort(self._index)
        if not ascending:
            order = order[::-1]
        return Series(
            jnp.take(self._data, jnp.asarray(order), axis=0),
            index=self._index[order],
            name=self._name,
        )

    def rank(self, method="average", ascending=True):
        """Rank values, pandas semantics (JIT-compatible)."""
        return self._new(_rank_1d(self._data, method=method, ascending=ascending))

    def nlargest(self, n: int = 5):
        """Top n values, descending. JIT-compatible via lax.top_k (static n)."""
        key = jnp.where(jnp.isnan(self._data), -jnp.inf, self._data)
        _, pos = jax.lax.top_k(key, n)
        values = jnp.take(self._data, pos)
        index = self._index if _is_tracer(pos) else self._index[np.asarray(pos)]
        return Series(values, index=index, name=self._name)

    def nsmallest(self, n: int = 5):
        """Bottom n values, ascending. JIT-compatible via lax.top_k (static n)."""
        key = jnp.where(jnp.isnan(self._data), -jnp.inf, -self._data)
        _, pos = jax.lax.top_k(key, n)
        values = jnp.take(self._data, pos)
        index = self._index if _is_tracer(pos) else self._index[np.asarray(pos)]
        return Series(values, index=index, name=self._name)

    def unique(self):
        """Unique values in order of appearance. Eager (structure discovery)."""
        arr = np.asarray(self._data)
        _, first_pos = np.unique(arr, return_index=True)
        return jnp.asarray(arr[np.sort(first_pos)])

    def duplicated(self, keep="first"):
        """Boolean mask of duplicate values. Eager (structure discovery)."""
        arr = np.asarray(self._data)
        n = len(arr)
        u, first_pos, inv, counts = np.unique(
            arr, return_index=True, return_inverse=True, return_counts=True
        )
        if keep == "first":
            mask = first_pos[inv] != np.arange(n)
        elif keep == "last":
            rev = arr[::-1]
            _, first_pos_r, inv_r = np.unique(rev, return_index=True, return_inverse=True)[:3]
            mask = (first_pos_r[inv_r] != np.arange(n))[::-1]
        else:  # keep=False
            mask = counts[inv] > 1
        return self._new(jnp.asarray(mask))

    def drop_duplicates(self, keep="first"):
        """Drop duplicate values. Eager (shape change)."""
        mask = ~np.asarray(self.duplicated(keep=keep)._data)
        return Series(self._data[jnp.asarray(mask)], index=self._index[mask], name=self._name)

    def factorize(self, sort=False):
        """Encode values as (codes, uniques) in appearance order; NaN gets
        code -1 (pandas semantics). Eager."""
        arr = np.asarray(self._data)
        nan_mask = (
            np.isnan(arr) if np.issubdtype(arr.dtype, np.floating) else np.zeros(len(arr), bool)
        )
        valid = arr[~nan_mask]
        uniques, first_pos, inv = np.unique(valid, return_index=True, return_inverse=True)
        if not sort:
            order = np.argsort(first_pos)
            remap = np.empty_like(order)
            remap[order] = np.arange(len(order))
            inv = remap[inv]
            uniques = uniques[order]
        codes = np.full(len(arr), -1, dtype=np.int64)
        codes[~nan_mask] = inv
        return codes, uniques

    def searchsorted(self, value, side="left"):
        """Positions where value(s) would be inserted (JIT-compatible)."""
        return jnp.searchsorted(self._data, jnp.asarray(value), side=side)

    # ============================
    # Function application / grouping / windows
    # ============================

    def apply(self, func, convert_dtype=True, args=(), **kwargs):
        """Apply func elementwise. Vectorized (JIT-compatible) when func accepts
        arrays; falls back to an eager Python loop otherwise."""
        try:
            result = func(self._data, *args, **kwargs)
            if hasattr(result, "shape") and result.shape == self._data.shape:
                return self._new(result)
        except Exception:
            pass
        out = np.asarray([func(v, *args, **kwargs) for v in np.asarray(self._data)])
        return self._new(out)

    def agg(self, func=None):
        """Aggregate by name, callable, or list thereof."""
        if isinstance(func, str):
            return getattr(self, func)()
        if isinstance(func, list | tuple):
            vals = [self.agg(f) for f in func]
            labels = [f if isinstance(f, str) else getattr(f, "__name__", str(f)) for f in func]
            return Series(
                jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in vals]),
                index=np.asarray(labels),
                name=self._name,
            )
        return func(self)

    aggregate = agg

    def transform(self, func):
        """Apply a shape-preserving function (JIT-compatible if func is)."""
        if isinstance(func, str):
            result = getattr(self, func)()
        else:
            result = func(self)
        if isinstance(result, Series):
            return result
        return self._new(result)

    def groupby(self, by):
        """Group by another Series/array. Structure discovery is eager;
        aggregations are JIT-compatible segment ops."""
        keys_arr = np.asarray(by._data if isinstance(by, Series) else by)
        group_keys, segment_ids = np.unique(keys_arr, return_inverse=True)
        return SeriesGroupBy(
            data=self._data,
            segment_ids=jnp.asarray(segment_ids),
            num_groups=len(group_keys),
            group_keys=group_keys,
            name=self._name,
        )

    def _to_frame_internal(self):
        col = self._name if isinstance(self._name, str) else "__series__"
        return DataFrame._from_column_arrays(
            {col: jnp.asarray(self._data)}, [col], self._index
        ), col

    def rolling(self, window, min_periods=None):
        """Rolling window (JIT-compatible aggregations)."""
        df, col = self._to_frame_internal()
        return _SeriesWindowProxy(df.rolling(window, min_periods=min_periods), col, self._name)

    def expanding(self, min_periods: int = 1):
        """Expanding window (JIT-compatible aggregations)."""
        df, col = self._to_frame_internal()
        return _SeriesWindowProxy(df.expanding(min_periods=min_periods), col, self._name)

    def ewm(self, alpha=None, span=None, com=None, halflife=None, min_periods: int = 0):
        """Exponentially weighted window (JIT-compatible aggregations)."""
        df, col = self._to_frame_internal()
        return _SeriesWindowProxy(
            df.ewm(alpha=alpha, span=span, com=com, halflife=halflife, min_periods=min_periods),
            col,
            self._name,
        )

    # ============================
    # Combining / comparing
    # ============================

    def combine(self, other, func, fill_value=None):
        """Combine elementwise with another Series via func (JIT-compatible
        when func is JAX-compatible)."""
        other_data = other._data if isinstance(other, Series) else other
        a, b = self._data, other_data
        if fill_value is not None:
            a = jnp.where(jnp.isnan(a), fill_value, a)
            if isinstance(b, jnp.ndarray | np.ndarray):
                b = jnp.where(jnp.isnan(b), fill_value, b)
        return self._new(func(a, b))

    def combine_first(self, other):
        """Fill NaNs with values from other (JIT-compatible)."""
        return self._new(jnp.where(jnp.isnan(self._data), other._data, self._data))

    def update(self, other):
        """Overwrite with non-NaN values from other. Mutates self (eager)."""
        self._data = jnp.where(jnp.isnan(other._data), self._data, other._data)

    def align(self, other, join="outer", fill_value=jnp.nan):
        """Align two Series on their index union/intersection. Eager."""
        if np.array_equal(self._index, other._index):
            return self.copy(), other.copy()
        if join == "outer":
            labels = np.union1d(self._index, other._index)
        elif join == "inner":
            labels = np.intersect1d(self._index, other._index)
        elif join == "left":
            labels = self._index
        else:
            labels = other._index
        return self.reindex(labels, fill_value=fill_value), other.reindex(
            labels, fill_value=fill_value
        )

    def equals(self, other) -> bool:
        """Exact equality including NaN positions. Eager."""
        a = np.asarray(self._data)
        b = np.asarray(other._data if isinstance(other, Series) else other)
        if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
            return False
        try:
            return bool(np.array_equal(a, b, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(a, b))

    def compare(self, other, keep_shape=False):
        """Rows where self and other differ, as a DataFrame. Eager."""
        a = np.asarray(self._data)
        b = np.asarray(other._data)
        diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        if not keep_shape:
            pos = np.flatnonzero(diff)
            return DataFrame({"self": a[pos], "other": b[pos]}, index=self._index[pos])
        a2, b2 = a.astype(np.float64).copy(), b.astype(np.float64).copy()
        a2[~diff] = np.nan
        b2[~diff] = np.nan
        return DataFrame({"self": a2, "other": b2}, index=self._index)

    # ============================
    # Conversion / I/O (eager; delegates to pandas where sensible)
    # ============================

    def to_numpy(self, dtype=None):
        """Convert to a numpy array (eager)."""
        arr = np.asarray(self._data)
        return arr.astype(dtype) if dtype is not None else arr

    def tolist(self):
        """Convert to a Python list (eager)."""
        return np.asarray(self._data).tolist()

    to_list = tolist

    def to_dict(self):
        """Convert to {label: value} dict (eager)."""
        return dict(zip(self._index.tolist(), np.asarray(self._data).tolist()))

    def to_frame(self, name=None):
        """Convert to a single-column DataFrame."""
        col = name if name is not None else (self._name if self._name is not None else 0)
        return DataFrame({col: np.asarray(self._data)}, index=self._index)

    def to_pandas(self):
        """Convert to a pandas Series (eager)."""
        import pandas as pd

        return pd.Series(np.asarray(self._data), index=self._index, name=self._name)

    def memory_usage(self, index=True, deep=False) -> int:
        """Bytes used by the underlying data (plus index)."""
        total = np.asarray(self._data).nbytes
        if index:
            total += self._index.nbytes
        return total

    def info(self, **kwargs):
        """Print a concise summary (delegates to pandas)."""
        return self.to_pandas().info(**kwargs)

    def _delegate_pandas(self, method, *args, **kwargs):
        return getattr(self.to_pandas(), method)(*args, **kwargs)

    def to_csv(self, *a, **k):
        """Write CSV via pandas (eager I/O)."""
        return self._delegate_pandas("to_csv", *a, **k)

    def to_json(self, *a, **k):
        """Write JSON via pandas (eager I/O)."""
        return self._delegate_pandas("to_json", *a, **k)

    def to_string(self, *a, **k):
        return self._delegate_pandas("to_string", *a, **k)

    def to_markdown(self, *a, **k):
        return self._delegate_pandas("to_markdown", *a, **k)

    def to_latex(self, *a, **k):
        return self._delegate_pandas("to_latex", *a, **k)

    def to_excel(self, *a, **k):
        return self._delegate_pandas("to_excel", *a, **k)

    def to_pickle(self, *a, **k):
        return self._delegate_pandas("to_pickle", *a, **k)

    def to_sql(self, *a, **k):
        return self._delegate_pandas("to_sql", *a, **k)

    def to_hdf(self, *a, **k):
        return self._delegate_pandas("to_hdf", *a, **k)

    def to_clipboard(self, *a, **k):
        return self._delegate_pandas("to_clipboard", *a, **k)

    def to_xarray(self):
        return self._delegate_pandas("to_xarray")

    @property
    def plot(self):
        """Plotting accessor (delegates to pandas)."""
        return self.to_pandas().plot

    def hist(self, *a, **k):
        """Histogram plot (delegates to pandas)."""
        return self._delegate_pandas("hist", *a, **k)

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


class _SeriesILocIndexer:
    """Integer-position indexing for Series."""

    def __init__(self, series: "Series"):
        self._s = series

    def __getitem__(self, key):
        if isinstance(key, int | np.integer):
            return self._s._data[int(key)]
        if isinstance(key, slice):
            return Series(self._s._data[key], index=self._s._index[key], name=self._s._name)
        arr = np.asarray(key)
        if arr.dtype == bool:
            return Series(
                self._s._data[jnp.asarray(arr)], index=self._s._index[arr], name=self._s._name
            )
        return Series(
            jnp.take(self._s._data, jnp.asarray(arr), axis=0),
            index=self._s._index[arr],
            name=self._s._name,
        )


class _SeriesLocIndexer:
    """Label-based indexing for Series."""

    def __init__(self, series: "Series"):
        self._s = series

    def __getitem__(self, key):
        s = self._s
        if isinstance(key, Series):
            key = np.asarray(key._data)
        if isinstance(key, np.ndarray | jnp.ndarray) and np.asarray(key).dtype == bool:
            mask = np.asarray(key)
            return Series(s._data[jnp.asarray(mask)], index=s._index[mask], name=s._name)
        if isinstance(key, slice):
            # label-based inclusive slice
            start = 0 if key.start is None else int(np.flatnonzero(s._index == key.start)[0])
            stop = (
                len(s._index)
                if key.stop is None
                else int(np.flatnonzero(s._index == key.stop)[0]) + 1
            )
            return Series(s._data[start:stop], index=s._index[start:stop], name=s._name)
        if isinstance(key, list | np.ndarray):
            lookup = {label: i for i, label in enumerate(s._index.tolist())}
            pos = np.array([lookup[k] for k in key])
            return Series(
                jnp.take(s._data, jnp.asarray(pos), axis=0), index=s._index[pos], name=s._name
            )
        matches = np.flatnonzero(s._index == key)
        if len(matches) == 0:
            raise KeyError(key)
        return s._data[int(matches[0])]


class _SeriesWindowProxy:
    """Wraps a DataFrame window object (Rolling/Expanding/EWM) built from a
    single-column frame, unwrapping results back to Series."""

    def __init__(self, window_obj, col, name):
        self._window = window_obj
        self._col = col
        self._name = name

    def __getattr__(self, attr):
        method = getattr(self._window, attr)

        def _call(*args, **kwargs):
            result_df = method(*args, **kwargs)
            series = result_df[self._col]
            series._name = self._name
            return series

        return _call


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


def _gather_rows_nanfill(arr, idx):
    """Gather rows by index; idx == -1 means 'unmatched' and yields NaN
    (numeric, promoting ints to float — pandas semantics) or None (object)."""
    idx = np.asarray(idx)
    has_missing = bool((idx < 0).any())
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        out = arr[np.maximum(idx, 0)].copy()
        if has_missing:
            out[idx < 0] = None
        return out
    gathered = jnp.take(jnp.asarray(arr), jnp.asarray(np.maximum(idx, 0)), axis=0)
    if not has_missing:
        return gathered
    if not jnp.issubdtype(gathered.dtype, jnp.floating):
        gathered = gathered.astype(jnp.float32)
    return jnp.where(jnp.asarray(idx < 0), jnp.nan, gathered)


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


@jax.jit
def _fillna_block(block, value):
    """JIT'd fillna — replaces NaN with value."""
    return jnp.where(jnp.isnan(block), value, block)


def _make_rolling_fn(reduce_fn):
    """Create a JIT'd rolling block function for a given reduction."""

    @jax.jit
    def _fn(block, idx, valid, min_periods):
        gathered = block[idx]
        mask = valid[:, :, None]
        gathered = jnp.where(mask, gathered, jnp.nan)
        n_valid = jnp.sum(valid, axis=1)
        result = reduce_fn(gathered)
        return jnp.where(n_valid[:, None] >= min_periods, result, jnp.nan)

    return _fn


_rolling_min_block = _make_rolling_fn(lambda g: jnp.nanmin(g, axis=1))
_rolling_max_block = _make_rolling_fn(lambda g: jnp.nanmax(g, axis=1))


def _rolling_prefix_sums(block, window):
    """Shared prefix-sum setup for O(n) rolling ops."""
    valid = ~jnp.isnan(block)
    clean = jnp.where(valid, block, 0.0)
    n, cols = block.shape
    z = jnp.zeros((1, cols), dtype=clean.dtype)
    cum = jnp.concatenate([z, jnp.cumsum(clean, axis=0)], axis=0)
    cum_valid = jnp.concatenate(
        [jnp.zeros((1, cols), dtype=jnp.float32), jnp.cumsum(valid.astype(jnp.float32), axis=0)],
        axis=0,
    )
    end = jnp.arange(1, n + 1)
    start = jnp.maximum(end - window, 0)
    rolling_sum = cum[end] - cum[start]
    rolling_count = cum_valid[end] - cum_valid[start]
    return clean, rolling_sum, rolling_count


@jax.jit
def _rolling_sum_cumsum(block, window, min_periods):
    """Rolling sum via prefix sums — O(n) memory."""
    _, rolling_sum, rolling_count = _rolling_prefix_sums(block, window)
    return jnp.where(rolling_count >= min_periods, rolling_sum, jnp.nan)


@jax.jit
def _rolling_mean_cumsum(block, window, min_periods):
    """Rolling mean via prefix sums — O(n) memory."""
    _, rolling_sum, rolling_count = _rolling_prefix_sums(block, window)
    mean = rolling_sum / jnp.maximum(rolling_count, 1)
    return jnp.where(rolling_count >= min_periods, mean, jnp.nan)


@jax.jit
def _rolling_var_cumsum(block, window, min_periods):
    """Rolling variance via prefix sums — O(n) memory."""
    clean, rolling_sum, rolling_count = _rolling_prefix_sums(block, window)
    n, cols = block.shape
    z = jnp.zeros((1, cols), dtype=clean.dtype)
    cum_sq = jnp.concatenate([z, jnp.cumsum(clean * clean, axis=0)], axis=0)
    end = jnp.arange(1, n + 1)
    start = jnp.maximum(end - window, 0)
    rolling_sq = cum_sq[end] - cum_sq[start]
    mean = rolling_sum / jnp.maximum(rolling_count, 1)
    var = (rolling_sq - rolling_count * mean * mean) / jnp.maximum(rolling_count - 1, 1)
    return jnp.where(rolling_count >= min_periods, var, jnp.nan)


@jax.jit
def _rolling_std_cumsum(block, window, min_periods):
    """Rolling std via prefix sums — O(n) memory."""
    clean, rolling_sum, rolling_count = _rolling_prefix_sums(block, window)
    n, cols = block.shape
    z = jnp.zeros((1, cols), dtype=clean.dtype)
    cum_sq = jnp.concatenate([z, jnp.cumsum(clean * clean, axis=0)], axis=0)
    end = jnp.arange(1, n + 1)
    start = jnp.maximum(end - window, 0)
    rolling_sq = cum_sq[end] - cum_sq[start]
    mean = rolling_sum / jnp.maximum(rolling_count, 1)
    var = (rolling_sq - rolling_count * mean * mean) / jnp.maximum(rolling_count - 1, 1)
    return jnp.where(rolling_count >= min_periods, jnp.sqrt(jnp.maximum(var, 0)), jnp.nan)


class Rolling:
    """Rolling window operations. JIT-compatible with fixed window size."""

    def __init__(self, df, window: int, min_periods: int):
        self._df = df
        self._window = window
        self._min_periods = min_periods

    def _build_result(self, new_blocks):
        """Build DataFrame from new dtype blocks, remapping column_to_block."""
        dtype_map = dict(zip(self._df._dtype_blocks.keys(), new_blocks.keys()))
        new_col_to_block = {}
        for col, (old_dtype, col_idx) in self._df._column_to_block.items():
            new_col_to_block[col] = (dtype_map[old_dtype], col_idx)
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._df._object_data,
            index=self._df._index,
            column_order=self._df._column_order,
        )

    def _apply_cumsum(self, block_fn):
        """Apply a prefix-sum rolling function — O(n) memory.

        All results are float32 (rolling stats are always float).
        Merges multiple dtype blocks into a single float32 block.
        """
        parts = []  # (result_block, [(col, original_col_idx), ...])
        window = jnp.int32(self._window)
        min_periods = jnp.int32(self._min_periods)
        for dtype, block in self._df._dtype_blocks.items():
            result = block_fn(block.astype(jnp.float32), window, min_periods)
            parts.append((result, dtype))

        # Merge all result blocks into one float32 block
        merged = jnp.concatenate([p[0] for p in parts], axis=1)
        new_blocks = {jnp.float32: merged}

        # Rebuild column_to_block with correct indices into merged block
        new_col_to_block = {}
        offset = 0
        for result, orig_dtype in parts:
            for col, (dt, col_idx) in self._df._column_to_block.items():
                if dt == orig_dtype:
                    new_col_to_block[col] = (jnp.float32, offset + col_idx)
            offset += result.shape[1]

        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._df._object_data,
            index=self._df._index,
            column_order=self._df._column_order,
        )

    def _apply_gather(self, block_fn):
        """Apply a gather-based rolling function — O(n*w) memory. Used for min/max."""
        parts = []
        min_periods = jnp.int32(self._min_periods)
        for dtype, block in self._df._dtype_blocks.items():
            fblock = block.astype(jnp.float32)
            idx, valid = _rolling_window(fblock, self._window)
            idx = jnp.array(idx)
            valid = jnp.array(valid)
            result = block_fn(fblock, idx, valid, min_periods)
            parts.append((result, dtype))

        merged = jnp.concatenate([p[0] for p in parts], axis=1)
        new_blocks = {jnp.float32: merged}

        new_col_to_block = {}
        offset = 0
        for result, orig_dtype in parts:
            for col, (dt, col_idx) in self._df._column_to_block.items():
                if dt == orig_dtype:
                    new_col_to_block[col] = (jnp.float32, offset + col_idx)
            offset += result.shape[1]

        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._df._object_data,
            index=self._df._index,
            column_order=self._df._column_order,
        )

    def sum(self):
        """Rolling sum."""
        return self._apply_cumsum(_rolling_sum_cumsum)

    def mean(self):
        """Rolling mean."""
        return self._apply_cumsum(_rolling_mean_cumsum)

    def var(self, ddof: int = 1):
        """Rolling variance."""
        return self._apply_cumsum(_rolling_var_cumsum)

    def std(self, ddof: int = 1):
        """Rolling standard deviation."""
        return self._apply_cumsum(_rolling_std_cumsum)

    def min(self):
        """Rolling minimum."""
        return self._apply_gather(_rolling_min_block)

    def max(self):
        """Rolling maximum."""
        return self._apply_gather(_rolling_max_block)


@jax.jit
def _expanding_sum_block(block, min_periods):
    """JIT'd expanding sum via scan."""

    def scan_fn(carry, x):
        running_sum, count = carry
        is_valid = ~jnp.isnan(x)
        running_sum = running_sum + jnp.where(is_valid, x, 0.0)
        count = count + is_valid.astype(jnp.int32)
        result = jnp.where(count >= min_periods, running_sum, jnp.nan)
        return (running_sum, count), result

    n_cols = block.shape[1]
    init = (jnp.zeros(n_cols, dtype=block.dtype), jnp.zeros(n_cols, dtype=jnp.int32))
    _, result = jax.lax.scan(scan_fn, init, block)
    return result


@jax.jit
def _expanding_mean_block(block, min_periods):
    """JIT'd expanding mean via scan."""

    def scan_fn(carry, x):
        running_sum, count = carry
        is_valid = ~jnp.isnan(x)
        running_sum = running_sum + jnp.where(is_valid, x, 0.0)
        count = count + is_valid.astype(jnp.int32)
        mean = running_sum / jnp.maximum(count, 1)
        result = jnp.where(count >= min_periods, mean, jnp.nan)
        return (running_sum, count), result

    n_cols = block.shape[1]
    init = (jnp.zeros(n_cols, dtype=block.dtype), jnp.zeros(n_cols, dtype=jnp.int32))
    _, result = jax.lax.scan(scan_fn, init, block)
    return result


@jax.jit
def _expanding_var_block(block, min_periods, ddof):
    """JIT'd expanding variance via Welford's algorithm."""

    def scan_fn(carry, x):
        mean, m2, count = carry
        is_valid = ~jnp.isnan(x)
        safe_x = jnp.where(is_valid, x, 0.0)
        new_count = count + is_valid.astype(count.dtype)
        delta = safe_x - mean
        safe_count = jnp.maximum(new_count, 1)
        new_mean = jnp.where(is_valid, mean + delta / safe_count, mean)
        delta2 = safe_x - new_mean
        new_m2 = jnp.where(is_valid, m2 + delta * delta2, m2)
        denom = jnp.maximum(new_count - ddof, 1)
        var = new_m2 / denom
        result = jnp.where((new_count >= min_periods) & (new_count > ddof), var, jnp.nan)
        return (new_mean, new_m2, new_count), result

    n_cols = block.shape[1]
    init = (
        jnp.zeros(n_cols, dtype=block.dtype),
        jnp.zeros(n_cols, dtype=block.dtype),
        jnp.zeros(n_cols, dtype=jnp.float32),
    )
    _, result = jax.lax.scan(scan_fn, init, block)
    return result


@jax.jit
def _expanding_min_block(block, min_periods):
    """JIT'd expanding min via scan."""

    def scan_fn(carry, x):
        running_min, count = carry
        is_valid = ~jnp.isnan(x)
        candidate = jnp.where(is_valid, jnp.minimum(running_min, x), running_min)
        count = count + is_valid.astype(jnp.int32)
        result = jnp.where(count >= min_periods, candidate, jnp.nan)
        return (candidate, count), result

    n_cols = block.shape[1]
    init = (jnp.full(n_cols, jnp.inf, dtype=block.dtype), jnp.zeros(n_cols, dtype=jnp.int32))
    _, result = jax.lax.scan(scan_fn, init, block)
    return result


@jax.jit
def _expanding_max_block(block, min_periods):
    """JIT'd expanding max via scan."""

    def scan_fn(carry, x):
        running_max, count = carry
        is_valid = ~jnp.isnan(x)
        candidate = jnp.where(is_valid, jnp.maximum(running_max, x), running_max)
        count = count + is_valid.astype(jnp.int32)
        result = jnp.where(count >= min_periods, candidate, jnp.nan)
        return (candidate, count), result

    n_cols = block.shape[1]
    init = (
        jnp.full(n_cols, -jnp.inf, dtype=block.dtype),
        jnp.zeros(n_cols, dtype=jnp.int32),
    )
    _, result = jax.lax.scan(scan_fn, init, block)
    return result


class Expanding:
    """Expanding window operations. JIT-compatible via jax.lax.scan — O(n)."""

    def __init__(self, df, min_periods: int = 1):
        self._df = df
        self._min_periods = min_periods

    def _build_result(self, new_blocks):
        """Build DataFrame from new dtype blocks, remapping column_to_block."""
        dtype_map = dict(zip(self._df._dtype_blocks.keys(), new_blocks.keys()))
        new_col_to_block = {}
        for col, (old_dtype, col_idx) in self._df._column_to_block.items():
            new_col_to_block[col] = (dtype_map[old_dtype], col_idx)
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._df._object_data,
            index=self._df._index,
            column_order=self._df._column_order,
        )

    def _apply_block(self, fn, block, *args):
        """Apply a JIT'd block function with min_periods."""
        return fn(block, jnp.int32(self._min_periods), *args)

    def sum(self):
        """Expanding sum via scan — O(n)."""
        mp = jnp.int32(self._min_periods)
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            new_blocks[block.dtype] = _expanding_sum_block(block, mp)
        return self._build_result(new_blocks)

    def mean(self):
        """Expanding mean via scan — O(n)."""
        mp = jnp.int32(self._min_periods)
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            new_blocks[block.dtype] = _expanding_mean_block(block, mp)
        return self._build_result(new_blocks)

    def var(self, ddof: int = 1):
        """Expanding variance via Welford's algorithm — O(n)."""
        mp = jnp.int32(self._min_periods)
        ddof_arr = jnp.float32(ddof)
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            new_blocks[block.dtype] = _expanding_var_block(block, mp, ddof_arr)
        return self._build_result(new_blocks)

    def std(self, ddof: int = 1):
        """Expanding std via Welford's algorithm — O(n)."""
        mp = jnp.int32(self._min_periods)
        ddof_arr = jnp.float32(ddof)
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            var = _expanding_var_block(block, mp, ddof_arr)
            new_blocks[var.dtype] = jnp.sqrt(jnp.maximum(var, 0))
        return self._build_result(new_blocks)

    def min(self):
        """Expanding min via scan — O(n)."""
        mp = jnp.int32(self._min_periods)
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            new_blocks[block.dtype] = _expanding_min_block(block, mp)
        return self._build_result(new_blocks)

    def max(self):
        """Expanding max via scan — O(n)."""
        mp = jnp.int32(self._min_periods)
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            new_blocks[block.dtype] = _expanding_max_block(block, mp)
        return self._build_result(new_blocks)


@jax.jit
def _ewm_mean_block(block, alpha):
    """JIT'd EWM mean via scan."""

    def scan_fn(carry, x):
        weighted_sum, total_weight = carry
        valid = ~jnp.isnan(x)
        xv = jnp.where(valid, x, 0.0)
        # NaN entries contribute no weight but decay prior weights (pandas ignore_na=False)
        weighted_sum = jnp.where(valid, alpha * xv, 0.0) + (1 - alpha) * weighted_sum
        total_weight = jnp.where(valid, alpha, 0.0) + (1 - alpha) * total_weight
        return (weighted_sum, total_weight), weighted_sum / total_weight

    n_cols = block.shape[1]
    init = (jnp.zeros(n_cols), jnp.zeros(n_cols))
    _, result = jax.lax.scan(scan_fn, init, block)
    return result


@jax.jit
def _ewm_var_block(block, alpha, min_periods):
    """JIT'd EWM variance via scan."""

    def scan_fn(carry, x):
        old_mean, old_var, total_weight = carry
        # NaN entries don't move the mean/var (pandas skips them)
        x = jnp.where(jnp.isnan(x), old_mean, x)
        total_weight = alpha + (1 - alpha) * total_weight
        new_mean = alpha * x + (1 - alpha) * old_mean
        new_var = (1 - alpha) * (old_var + alpha * (x - old_mean) ** 2)
        return (new_mean, new_var, total_weight), (new_var / total_weight, total_weight)

    n_cols = block.shape[1]
    init = (block[0], jnp.zeros(n_cols), jnp.zeros(n_cols))
    _, (var_raw, _weights) = jax.lax.scan(scan_fn, init, block)
    row_idx = jnp.arange(block.shape[0])
    return jnp.where(row_idx[:, None] >= min_periods, var_raw, jnp.nan)


class EWM:
    """Exponentially weighted moving operations. JIT-compatible via scan."""

    def __init__(self, df, alpha: float, min_periods: int = 0):
        self._df = df
        self._alpha = alpha
        self._min_periods = min_periods

    def _build_result(self, new_blocks):
        """Build DataFrame from new dtype blocks, remapping column_to_block."""
        dtype_map = dict(zip(self._df._dtype_blocks.keys(), new_blocks.keys()))
        new_col_to_block = {}
        for col, (old_dtype, col_idx) in self._df._column_to_block.items():
            new_col_to_block[col] = (dtype_map[old_dtype], col_idx)
        return DataFrame._from_parts(
            dtype_blocks=new_blocks,
            column_to_block=new_col_to_block,
            object_data=self._df._object_data,
            index=self._df._index,
            column_order=self._df._column_order,
        )

    def mean(self):
        """Exponentially weighted moving mean."""
        alpha = jnp.float32(self._alpha)
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            new_blocks[block.dtype] = _ewm_mean_block(block, alpha)
        return self._build_result(new_blocks)

    def std(self, ddof: int = 1):
        """Exponentially weighted moving standard deviation."""
        alpha = jnp.float32(self._alpha)
        mp = jnp.int32(self._min_periods)
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            var = _ewm_var_block(block, alpha, mp)
            new_blocks[var.dtype] = jnp.sqrt(jnp.maximum(var, 0))
        return self._build_result(new_blocks)

    def var(self, ddof: int = 1):
        """Exponentially weighted moving variance."""
        alpha = jnp.float32(self._alpha)
        mp = jnp.int32(self._min_periods)
        new_blocks = {}
        for dtype, block in self._df._dtype_blocks.items():
            new_blocks[block.dtype] = _ewm_var_block(block, alpha, mp)
        return self._build_result(new_blocks)


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


@functools.partial(jax.jit, static_argnums=(2,))
def _segment_sum(data, ids, num):
    # NaN-aware: pandas groupby skips NaN
    return jax.ops.segment_sum(jnp.where(jnp.isnan(data), 0, data), ids, num)


@functools.partial(jax.jit, static_argnums=(2,))
def _segment_mean(data, ids, num):
    valid = ~jnp.isnan(data)
    sums = jax.ops.segment_sum(jnp.where(valid, data, 0), ids, num)
    counts = jax.ops.segment_sum(valid.astype(data.dtype), ids, num)
    return sums / counts


@functools.partial(jax.jit, static_argnums=(2,))
def _segment_var(data, ids, num, ddof):
    valid = ~jnp.isnan(data)
    counts = jax.ops.segment_sum(valid.astype(data.dtype), ids, num)
    sums = jax.ops.segment_sum(jnp.where(valid, data, 0), ids, num)
    means = sums / counts
    sq_devs = jnp.where(valid, (data - means[ids]) ** 2, 0)
    sum_sq = jax.ops.segment_sum(sq_devs, ids, num)
    return sum_sq / (counts - ddof)


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
        return self._result_series(_segment_sum(self._data, self._segment_ids, self._num_groups))

    def mean(self):
        return self._result_series(_segment_mean(self._data, self._segment_ids, self._num_groups))

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
        var = _segment_var(self._data, self._segment_ids, self._num_groups, jnp.float32(ddof))
        return self._result_series(jnp.sqrt(var))

    def var(self, ddof=1):
        return self._result_series(
            _segment_var(self._data, self._segment_ids, self._num_groups, jnp.float32(ddof))
        )

    def prod(self):
        result = jax.ops.segment_prod(self._data, self._segment_ids, self._num_groups)
        return self._result_series(result)

    def first(self):
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
            lambda d: _segment_sum(d, self._segment_ids, self._num_groups)
        )

    def mean(self):
        return self._apply_segment_op(
            lambda d: _segment_mean(d, self._segment_ids, self._num_groups)
        )

    def count(self):
        counts = jax.ops.segment_sum(
            jnp.ones(len(self._segment_ids)), self._segment_ids, self._num_groups
        )
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
        ddof_arr = jnp.float32(ddof)
        return self._apply_segment_op(
            lambda d: jnp.sqrt(_segment_var(d, self._segment_ids, self._num_groups, ddof_arr))
        )

    def var(self, ddof=1):
        ddof_arr = jnp.float32(ddof)
        return self._apply_segment_op(
            lambda d: _segment_var(d, self._segment_ids, self._num_groups, ddof_arr)
        )

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
        self.column_order = (
            tuple(column_order) if not isinstance(column_order, tuple) else column_order
        )
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
