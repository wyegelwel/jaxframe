"""
Performance benchmarks: pandas vs jaxframe (eager and JIT).

Run with: uv run python tests/test_benchmarks.py

Benchmarks two categories:
1. Scalar reductions (axis=None) — JIT-compatible, should show speedup at scale
2. Per-column reductions (axis=0) — returns Series, eager-only
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxframe import DataFrame


SIZES = [
    (100, 10),
    (1_000, 10),
    (10_000, 10),
    (100_000, 10),
    (1_000_000, 10),
]


def make_data(n_rows, n_cols, seed=42):
    rng = np.random.default_rng(seed)
    return {f"col_{i}": rng.standard_normal(n_rows) for i in range(n_cols)}


def bench(fn, warmup=3, rounds=20):
    """Run fn, return median time in ms."""
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


# --- Scalar reduction ops (JIT-compatible) ---


def _sum(df):
    return df.sum(axis=None)


def _mean(df):
    return df.mean(axis=None)


def _std(df):
    return df.std(axis=None)


def _mul_sum(df):
    return (df * 2).sum(axis=None)


def _chain(df):
    return ((df + 1) * 2 - 3).sum(axis=None)


def _pow_sum(df):
    return (df**2).sum(axis=None)


# --- Per-column ops (eager only, returns Series/DataFrame) ---


def _col_sum(df):
    return df.sum()


def _col_mean(df):
    return df.mean()


def _col_std(df):
    return df.std()


def _cumsum(df):
    return df.cumsum()


def _diff(df):
    return df.diff()


SCALAR_OPS = {
    "sum": _sum,
    "mean": _mean,
    "std": _std,
    "mul+sum": _mul_sum,
    "chain": _chain,
    "pow+sum": _pow_sum,
}

COLUMN_OPS = {
    "col_sum": _col_sum,
    "col_mean": _col_mean,
    "col_std": _col_std,
    "cumsum": _cumsum,
    "diff": _diff,
}


def run_scalar_benchmarks():
    """Scalar reductions: pandas vs jaxframe eager vs JIT."""
    print("=" * 80)
    print("SCALAR REDUCTIONS (axis=None) — JIT-compatible")
    print("=" * 80)
    print(
        f"{'Op':<12} {'Size':>12} {'pandas':>10} {'jaxframe':>10}"
        f" {'jf+jit':>10} {'jit speedup':>12}"
    )
    print("-" * 72)

    for op_name, op in SCALAR_OPS.items():
        for n_rows, n_cols in SIZES:
            data = make_data(n_rows, n_cols)
            pdf = pd.DataFrame(data)
            jdf = DataFrame(data)

            # Pandas baseline (sum(axis=None) = df.values.sum())
            if op_name in ("sum", "mean", "std"):
                # pandas axis=None doesn't exist for sum/mean/std;
                # use .values to get equivalent scalar
                pd_op = {
                    "sum": lambda d: d.values.sum(),
                    "mean": lambda d: d.values.mean(),
                    "std": lambda d: d.values.std(),
                }.get(op_name, op)
                pd_ms = bench(lambda pd_op=pd_op: pd_op(pdf))
            else:
                pd_ms = bench(lambda op=op: op(pdf))

            # jaxframe eager
            jf_ms = bench(lambda op=op: op(jdf))

            # JIT version
            jit_op = jax.jit(op)
            try:
                # Warmup (includes compilation)
                jit_op(jdf).block_until_ready()
                jit_ms = bench(lambda: jit_op(jdf))
            except Exception as e:
                jit_ms = float("nan")
                print(f"  JIT failed for {op_name}: {e}")

            speedup = pd_ms / jit_ms if jit_ms > 0 else float("nan")
            size_str = f"{n_rows}x{n_cols}"
            print(
                f"{op_name:<12} {size_str:>12} {pd_ms:>9.2f}ms"
                f" {jf_ms:>9.2f}ms {jit_ms:>9.2f}ms {speedup:>10.1f}x"
            )
        print()


def run_column_benchmarks():
    """Per-column ops: pandas vs jaxframe (no JIT)."""
    print("=" * 80)
    print("PER-COLUMN OPERATIONS (axis=0) — Eager only")
    print("=" * 80)
    print(f"{'Op':<12} {'Size':>12} {'pandas':>10} {'jaxframe':>10} {'ratio':>8}")
    print("-" * 56)

    for op_name, op in COLUMN_OPS.items():
        for n_rows, n_cols in SIZES:
            data = make_data(n_rows, n_cols)
            pdf = pd.DataFrame(data)
            jdf = DataFrame(data)

            pd_ms = bench(lambda op=op: op(pdf))
            jf_ms = bench(lambda op=op: op(jdf))

            ratio = pd_ms / jf_ms if jf_ms > 0 else float("nan")
            size_str = f"{n_rows}x{n_cols}"
            print(f"{op_name:<12} {size_str:>12} {pd_ms:>9.2f}ms {jf_ms:>9.2f}ms {ratio:>7.2f}x")
        print()


def run_raw_jax_comparison():
    """Compare jaxframe JIT vs raw jnp operations (overhead measurement)."""
    print("=" * 80)
    print("JAXFRAME JIT vs RAW JAX (overhead measurement)")
    print("=" * 80)
    print(f"{'Size':>12} {'raw jnp':>10} {'jf+jit':>10} {'overhead':>10}")
    print("-" * 48)

    for n_rows, n_cols in SIZES:
        data = make_data(n_rows, n_cols)
        raw = jnp.array(np.column_stack(list(data.values())), dtype=jnp.float32)
        jdf = DataFrame(data)

        raw_jit = jax.jit(lambda x: ((x + 1) * 2 - 3).sum())
        raw_jit(raw).block_until_ready()
        raw_ms = bench(lambda: raw_jit(raw))

        jf_jit = jax.jit(lambda df: ((df + 1) * 2 - 3).sum(axis=None))
        jf_jit(jdf).block_until_ready()
        jf_ms = bench(lambda: jf_jit(jdf))

        overhead = jf_ms / raw_ms if raw_ms > 0 else float("nan")
        size_str = f"{n_rows}x{n_cols}"
        print(f"{size_str:>12} {raw_ms:>9.2f}ms {jf_ms:>9.2f}ms {overhead:>9.1f}x")
    print()


if __name__ == "__main__":
    run_raw_jax_comparison()
    run_scalar_benchmarks()
    run_column_benchmarks()
