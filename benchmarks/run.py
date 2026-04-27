#!/usr/bin/env python3
"""
JaxFrame benchmark runner.

Collects timing data for pandas vs jaxframe (eager and JIT) across
multiple operation categories and data sizes. Outputs structured JSON.

Usage:
    uv run python benchmarks/run.py              # full suite
    uv run python benchmarks/run.py --quick      # fast subset (fewer sizes)
    uv run python benchmarks/run.py --category scalar_reductions
"""

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Add parent to path so we can import jaxframe
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jaxframe import DataFrame  # noqa: E402

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class BenchOp:
    """A single benchmarkable operation."""

    name: str
    pandas_fn: Callable
    jaxframe_fn: Callable
    jit_fn: Callable | None = None  # None = not JIT-compatible


@dataclass
class GroupByOp:
    """A GroupBy aggregation to benchmark."""

    label: str
    agg_name: str


@dataclass
class Category:
    """A benchmark category with its operations and constraints."""

    key: str
    label: str
    ops: list[BenchOp]
    max_rows: int | None = None  # Cap row sizes (e.g. O(n²) ops)


@dataclass
class BenchResult:
    """A single benchmark measurement."""

    category: str
    op: str
    scaling: str  # "rows" or "cols"
    n_rows: int
    n_cols: int
    pandas_ms: float
    jaxframe_ms: float
    jit_ms: float | None = None


@dataclass
class OverheadResult:
    """Overhead comparison: raw JAX vs jaxframe JIT."""

    category: str = "overhead"
    op: str = "chain"
    scaling: str = "rows"
    n_rows: int = 0
    n_cols: int = 0
    raw_jnp_ms: float = 0.0
    jaxframe_jit_ms: float = 0.0
    overhead_ratio: float | None = None


@dataclass
class BenchMetadata:
    """Environment info captured at benchmark time."""

    timestamp: str = ""
    device: str = ""
    device_count: int = 0
    jax_version: str = ""
    numpy_version: str = ""
    pandas_version: str = ""
    python_version: str = ""
    platform: str = ""


# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------


QUICK_MODE = False


def bench(fn, warmup=None, rounds=None):
    """Run fn, return median time in ms."""
    if warmup is None:
        warmup = 2 if QUICK_MODE else 3
    if rounds is None:
        rounds = 7 if QUICK_MODE else 20
    for _ in range(warmup):
        result = fn()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        result = fn()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] * 1000


def make_data(n_rows, n_cols, seed=42):
    """Generate random float data as a dict of columns."""
    rng = np.random.default_rng(seed)
    return {f"col_{i}": rng.standard_normal(n_rows) for i in range(n_cols)}


def make_groupby_data(n_rows, n_cols, n_groups=100, seed=42):
    """Generate data with a group key column."""
    rng = np.random.default_rng(seed)
    data = {f"val_{i}": rng.standard_normal(n_rows) for i in range(n_cols)}
    data["key"] = rng.integers(0, n_groups, size=n_rows).astype(float)
    return data


# ---------------------------------------------------------------------------
# Operation definitions
# ---------------------------------------------------------------------------


def _make_scalar_reductions() -> list[BenchOp]:
    """Scalar reductions (axis=None) — JIT-compatible."""
    ops = []
    for name in ["sum", "mean", "std", "var", "min", "max", "prod"]:

        def pd_fn(df, _n=name):
            return getattr(df.values, _n)()

        def jf_fn(df, _n=name):
            return getattr(df, _n)(axis=None)

        ops.append(
            BenchOp(
                name=name,
                pandas_fn=pd_fn,
                jaxframe_fn=jf_fn,
                jit_fn=jf_fn,
            )
        )
    return ops


def _make_arithmetic_chains() -> list[BenchOp]:
    """Arithmetic chain operations — JIT-compatible."""
    return [
        BenchOp(
            name="mul+sum",
            pandas_fn=lambda df: (df * 2).values.sum(),
            jaxframe_fn=lambda df: (df * 2).sum(axis=None),
            jit_fn=lambda df: (df * 2).sum(axis=None),
        ),
        BenchOp(
            name="pow+sum",
            pandas_fn=lambda df: (df**2).values.sum(),
            jaxframe_fn=lambda df: (df**2).sum(axis=None),
            jit_fn=lambda df: (df**2).sum(axis=None),
        ),
        BenchOp(
            name="chain",
            pandas_fn=lambda df: ((df + 1) * 2 - 3).values.sum(),
            jaxframe_fn=lambda df: ((df + 1) * 2 - 3).sum(axis=None),
            jit_fn=lambda df: ((df + 1) * 2 - 3).sum(axis=None),
        ),
    ]


def _make_column_reductions() -> list[BenchOp]:
    """Per-column reductions (axis=0) — eager only."""
    return [
        BenchOp(name="col_sum", pandas_fn=lambda df: df.sum(), jaxframe_fn=lambda df: df.sum()),
        BenchOp(name="col_mean", pandas_fn=lambda df: df.mean(), jaxframe_fn=lambda df: df.mean()),
        BenchOp(name="col_std", pandas_fn=lambda df: df.std(), jaxframe_fn=lambda df: df.std()),
    ]


def _make_cumulative() -> list[BenchOp]:
    """Cumulative operations — blockwise."""
    return [
        BenchOp(
            name="cumsum",
            pandas_fn=lambda df: df.cumsum(),
            jaxframe_fn=lambda df: df.cumsum(),
        ),
        BenchOp(
            name="cumprod",
            pandas_fn=lambda df: df.cumprod(),
            jaxframe_fn=lambda df: df.cumprod(),
        ),
    ]


def _make_shift_diff() -> list[BenchOp]:
    """Shift and diff — column-by-column."""
    return [
        BenchOp(name="diff", pandas_fn=lambda df: df.diff(), jaxframe_fn=lambda df: df.diff()),
        BenchOp(name="shift", pandas_fn=lambda df: df.shift(1), jaxframe_fn=lambda df: df.shift(1)),
    ]


def _make_rolling() -> list[BenchOp]:
    """Rolling window ops — JIT-compatible."""
    return [
        BenchOp(
            name="rolling_sum",
            pandas_fn=lambda df: df.rolling(10).sum(),
            jaxframe_fn=lambda df: df.rolling(10).sum(),
            jit_fn=lambda df: df.rolling(10).sum().sum(axis=None),
        ),
        BenchOp(
            name="rolling_mean",
            pandas_fn=lambda df: df.rolling(10).mean(),
            jaxframe_fn=lambda df: df.rolling(10).mean(),
            jit_fn=lambda df: df.rolling(10).mean().sum(axis=None),
        ),
    ]


def _make_expanding() -> list[BenchOp]:
    """Expanding window ops — JIT-compatible. O(n²) memory."""
    return [
        BenchOp(
            name="expanding_sum",
            pandas_fn=lambda df: df.expanding().sum(),
            jaxframe_fn=lambda df: df.expanding().sum(),
            jit_fn=lambda df: df.expanding().sum().sum(axis=None),
        ),
        BenchOp(
            name="expanding_mean",
            pandas_fn=lambda df: df.expanding().mean(),
            jaxframe_fn=lambda df: df.expanding().mean(),
            jit_fn=lambda df: df.expanding().mean().sum(axis=None),
        ),
    ]


def _make_ewm() -> list[BenchOp]:
    """EWM ops — JIT-compatible."""
    return [
        BenchOp(
            name="ewm_mean",
            pandas_fn=lambda df: df.ewm(span=10).mean(),
            jaxframe_fn=lambda df: df.ewm(span=10).mean(),
            jit_fn=lambda df: df.ewm(span=10).mean().sum(axis=None),
        ),
    ]


def _make_data_cleaning() -> list[BenchOp]:
    """Data cleaning ops — JIT-compatible."""
    return [
        BenchOp(
            name="fillna",
            pandas_fn=lambda df: df.fillna(0),
            jaxframe_fn=lambda df: df.fillna(0),
            jit_fn=lambda df: df.fillna(0).sum(axis=None),
        ),
        BenchOp(
            name="where",
            pandas_fn=lambda df: df.where(df > 0, 0),
            jaxframe_fn=lambda df: df.where(df > 0, 0),
            jit_fn=lambda df: df.where(df > 0, 0).sum(axis=None),
        ),
        BenchOp(
            name="clip",
            pandas_fn=lambda df: df.clip(-1, 1),
            jaxframe_fn=lambda df: df.clip(-1, 1),
            jit_fn=lambda df: df.clip(-1, 1).sum(axis=None),
        ),
    ]


def _make_sorting() -> list[BenchOp]:
    """Sorting ops — eager only."""
    return [
        BenchOp(
            name="sort_values",
            pandas_fn=lambda df: df.sort_values("col_0"),
            jaxframe_fn=lambda df: df.sort_values("col_0"),
        ),
        BenchOp(
            name="rank",
            pandas_fn=lambda df: df.rank(),
            jaxframe_fn=lambda df: df.rank(),
        ),
    ]


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

CATEGORIES = [
    Category(
        key="scalar_reductions",
        label="Scalar Reductions (axis=None)",
        ops=_make_scalar_reductions(),
    ),
    Category(
        key="arithmetic_chains",
        label="Arithmetic Chains",
        ops=_make_arithmetic_chains(),
    ),
    Category(
        key="column_reductions",
        label="Per-Column Reductions (axis=0)",
        ops=_make_column_reductions(),
    ),
    Category(key="cumulative", label="Cumulative Operations", ops=_make_cumulative()),
    Category(key="shift_diff", label="Shift & Diff", ops=_make_shift_diff()),
    Category(key="rolling", label="Rolling Windows", ops=_make_rolling()),
    Category(key="expanding", label="Expanding Windows", ops=_make_expanding(), max_rows=10_000),
    Category(key="ewm", label="Exponentially Weighted", ops=_make_ewm()),
    Category(key="data_cleaning", label="Data Cleaning", ops=_make_data_cleaning()),
    Category(key="sorting", label="Sorting", ops=_make_sorting()),
]

GROUPBY_OPS = [
    GroupByOp(label="gb_sum", agg_name="sum"),
    GroupByOp(label="gb_mean", agg_name="mean"),
    GroupByOp(label="gb_std", agg_name="std"),
]

ROW_SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]
ROW_SIZES_QUICK = [100, 10_000, 1_000_000]
COL_SIZES = [5, 10, 25, 50, 100]
COL_SIZES_QUICK = [5, 25, 100]
FIXED_COLS = 10
FIXED_ROWS = 1_000

GROUPBY_ROW_SIZES = [100, 1_000, 10_000, 100_000]
GROUPBY_ROW_SIZES_QUICK = [100, 10_000, 100_000]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _run_single(pd_fn, jf_fn, jit_fn, pdf, jdf, verbose, label=""):
    """Run pandas, jaxframe, and optionally JIT benchmarks for one data size."""
    pd_ms = bench(lambda: pd_fn(pdf))
    jf_ms = bench(lambda: jf_fn(jdf))

    jit_ms = None
    if jit_fn is not None:
        try:
            jit_op = jax.jit(jit_fn)
            jit_op(jdf)  # compile
            result = jit_op(jdf)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            jit_ms = bench(lambda: jit_op(jdf))
        except Exception as e:
            if verbose:
                print(f"    JIT failed: {e}")

    if verbose:
        jit_str = f"{jit_ms:8.2f}ms" if jit_ms is not None else "     N/A"
        speedup = pd_ms / jit_ms if jit_ms and jit_ms > 0 else None
        sp_str = f"{speedup:6.1f}x" if speedup else "   N/A"
        print(f"    {label}: pd={pd_ms:8.2f}ms  jf={jf_ms:8.2f}ms  jit={jit_str}  speedup={sp_str}")

    return pd_ms, jf_ms, jit_ms


def run_category(cat: Category, row_sizes, col_sizes, verbose=True) -> list[BenchResult]:
    """Run benchmarks for a single category."""
    effective_rows = [r for r in row_sizes if cat.max_rows is None or r <= cat.max_rows]
    results = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  {cat.label}")
        if cat.max_rows:
            print(f"  (capped at {cat.max_rows:,} rows — O(n²) memory)")
        print(f"{'=' * 60}")

    # Row scaling (fixed cols)
    for op in cat.ops:
        if verbose:
            print(f"\n  {op.name} (row scaling, {FIXED_COLS} cols):")
        for n_rows in effective_rows:
            data = make_data(n_rows, FIXED_COLS)
            pd_ms, jf_ms, jit_ms = _run_single(
                op.pandas_fn,
                op.jaxframe_fn,
                op.jit_fn,
                pd.DataFrame(data),
                DataFrame(data),
                verbose,
                label=f"{n_rows:>10,} rows",
            )
            results.append(
                BenchResult(
                    category=cat.key,
                    op=op.name,
                    scaling="rows",
                    n_rows=n_rows,
                    n_cols=FIXED_COLS,
                    pandas_ms=round(pd_ms, 4),
                    jaxframe_ms=round(jf_ms, 4),
                    jit_ms=round(jit_ms, 4) if jit_ms is not None else None,
                )
            )

    # Column scaling (fixed rows)
    for op in cat.ops:
        if verbose:
            print(f"\n  {op.name} (col scaling, {FIXED_ROWS:,} rows):")
        for n_cols in col_sizes:
            data = make_data(FIXED_ROWS, n_cols)
            pd_ms, jf_ms, jit_ms = _run_single(
                op.pandas_fn,
                op.jaxframe_fn,
                op.jit_fn,
                pd.DataFrame(data),
                DataFrame(data),
                verbose,
                label=f"{n_cols:>5} cols",
            )
            results.append(
                BenchResult(
                    category=cat.key,
                    op=op.name,
                    scaling="cols",
                    n_rows=FIXED_ROWS,
                    n_cols=n_cols,
                    pandas_ms=round(pd_ms, 4),
                    jaxframe_ms=round(jf_ms, 4),
                    jit_ms=round(jit_ms, 4) if jit_ms is not None else None,
                )
            )

    return results


def run_groupby(gb_ops, row_sizes, verbose=True) -> list[BenchResult]:
    """Run GroupBy benchmarks (needs special data setup)."""
    results = []

    if verbose:
        print(f"\n{'=' * 60}")
        print("  GroupBy Aggregations")
        print(f"{'=' * 60}")

    for gb_op in gb_ops:
        if verbose:
            print(f"\n  groupby.{gb_op.agg_name} (row scaling, {FIXED_COLS} value cols):")
        for n_rows in row_sizes:
            n_groups = max(10, n_rows // 100)
            data = make_groupby_data(n_rows, FIXED_COLS, n_groups=n_groups)
            pdf = pd.DataFrame(data)
            jdf = DataFrame(data)

            def pd_fn(_pdf=pdf, _a=gb_op.agg_name):
                return getattr(_pdf.groupby("key"), _a)()

            def jf_fn(_jdf=jdf, _a=gb_op.agg_name):
                return getattr(_jdf.groupby("key")["val_0"], _a)()

            pd_ms = bench(pd_fn)
            jf_ms = bench(jf_fn)

            # JIT version
            def _make_jit_fn(agg):
                return lambda gb: getattr(gb, agg)().values.sum()

            jit_fn = _make_jit_fn(gb_op.agg_name)
            jit_ms = None
            try:
                gb_obj = jdf.groupby("key")["val_0"]
                jit_op = jax.jit(jit_fn)
                jit_op(gb_obj)
                jit_ms = bench(lambda: jit_op(gb_obj))
            except Exception as e:
                if verbose:
                    print(f"    JIT failed: {e}")

            if verbose:
                jit_str = f"{jit_ms:8.2f}ms" if jit_ms is not None else "     N/A"
                speedup = pd_ms / jit_ms if jit_ms and jit_ms > 0 else None
                sp_str = f"{speedup:6.1f}x" if speedup else "   N/A"
                print(
                    f"    {n_rows:>10,} rows: pd={pd_ms:8.2f}ms  "
                    f"jf={jf_ms:8.2f}ms  jit={jit_str}  speedup={sp_str}"
                )

            results.append(
                BenchResult(
                    category="groupby",
                    op=gb_op.label,
                    scaling="rows",
                    n_rows=n_rows,
                    n_cols=FIXED_COLS,
                    pandas_ms=round(pd_ms, 4),
                    jaxframe_ms=round(jf_ms, 4),
                    jit_ms=round(jit_ms, 4) if jit_ms is not None else None,
                )
            )

    return results


def run_overhead(row_sizes, verbose=True) -> list[OverheadResult]:
    """Measure abstraction overhead: raw jnp vs jaxframe JIT."""
    results = []

    if verbose:
        print(f"\n{'=' * 60}")
        print("  Abstraction Overhead (raw JAX vs jaxframe JIT)")
        print(f"{'=' * 60}")

    for n_rows in row_sizes:
        data = make_data(n_rows, FIXED_COLS)
        raw = jnp.array(np.column_stack(list(data.values())), dtype=jnp.float32)
        jdf = DataFrame(data)

        raw_jit = jax.jit(lambda x: ((x + 1) * 2 - 3).sum())
        raw_jit(raw).block_until_ready()
        raw_ms = bench(lambda: raw_jit(raw))

        jf_jit = jax.jit(lambda df: ((df + 1) * 2 - 3).sum(axis=None))
        jf_jit(jdf).block_until_ready()
        jf_ms = bench(lambda: jf_jit(jdf))

        overhead = round(jf_ms / raw_ms, 2) if raw_ms > 0 else None
        results.append(
            OverheadResult(
                n_rows=n_rows,
                n_cols=FIXED_COLS,
                raw_jnp_ms=round(raw_ms, 4),
                jaxframe_jit_ms=round(jf_ms, 4),
                overhead_ratio=overhead,
            )
        )

        if verbose:
            oh = jf_ms / raw_ms if raw_ms > 0 else float("nan")
            print(
                f"    {n_rows:>10,} rows: raw={raw_ms:8.2f}ms  "
                f"jf_jit={jf_ms:8.2f}ms  overhead={oh:.1f}x"
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def collect_metadata() -> BenchMetadata:
    """Gather environment info."""
    devices = jax.devices()
    return BenchMetadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        device=devices[0].platform if devices else "unknown",
        device_count=len(devices),
        jax_version=jax.__version__,
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
    )


def main():
    parser = argparse.ArgumentParser(description="JaxFrame benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Fewer data sizes for faster run")
    parser.add_argument("--category", type=str, help="Run only this category")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    args = parser.parse_args()

    global QUICK_MODE
    QUICK_MODE = args.quick
    verbose = not args.quiet
    row_sizes = ROW_SIZES_QUICK if args.quick else ROW_SIZES
    col_sizes = COL_SIZES_QUICK if args.quick else COL_SIZES
    gb_row_sizes = GROUPBY_ROW_SIZES_QUICK if args.quick else GROUPBY_ROW_SIZES

    if verbose:
        print("JaxFrame Benchmark Suite")
        print(f"Device: {jax.devices()[0].platform}")
        print(f"Row sizes: {row_sizes}")
        print(f"Col sizes: {col_sizes}")

    all_results: list[dict] = []
    metadata = collect_metadata()

    # Run standard categories
    for cat in CATEGORIES:
        if args.category and args.category != cat.key:
            continue
        results = run_category(cat, row_sizes, col_sizes, verbose=verbose)
        all_results.extend(asdict(r) for r in results)

    # GroupBy (special data setup)
    if not args.category or args.category == "groupby":
        results = run_groupby(GROUPBY_OPS, gb_row_sizes, verbose=verbose)
        all_results.extend(asdict(r) for r in results)

    # Overhead measurement
    if not args.category or args.category == "overhead":
        results = run_overhead(row_sizes, verbose=verbose)
        all_results.extend(asdict(r) for r in results)

    # Write results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{timestamp}.json"
    latest_path = output_dir / "latest.json"

    output = {"metadata": asdict(metadata), "results": all_results}
    for path in [output_path, latest_path]:
        path.write_text(json.dumps(output, indent=2))

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Results written to: {output_path}")
        print(f"Latest symlink: {latest_path}")
        print(f"Total benchmarks: {len(all_results)}")
        print("\nGenerate report with: uv run python benchmarks/report.py")

    return output_path


if __name__ == "__main__":
    main()
