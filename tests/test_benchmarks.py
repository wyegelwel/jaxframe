"""
Performance benchmarks: pandas vs jaxframe (eager and JIT).

Run with: uv run python tests/test_benchmarks.py
"""

import time

import jax
import numpy as np
import pandas as pd

from jaxframe import DataFrame

SIZES = [
    (100, 10),
    (1_000, 10),
    (10_000, 10),
    (100_000, 10),
    (1_000, 100),
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


def run_benchmarks():
    ops = {
        "sum": lambda df: df.sum(),
        "mean": lambda df: df.mean(),
        "std": lambda df: df.std(),
        "mul+sum": lambda df: (df * 2).sum(),
        "chain": lambda df: ((df + 1) * 2 - 3).sum(),
    }

    print(f"{'Op':<12} {'Size':>12} {'pandas':>10} {'jaxframe':>10} {'jf+jit':>10} {'speedup':>8}")
    print("-" * 72)

    for op_name, op in ops.items():
        for n_rows, n_cols in SIZES:
            data = make_data(n_rows, n_cols)
            pdf = pd.DataFrame(data)
            jdf = DataFrame(data)

            pd_ms = bench(lambda: op(pdf))
            jf_ms = bench(lambda: op(jdf))

            # JIT version
            jit_op = jax.jit(lambda df: op(df).sum(axis=None) if hasattr(op(df), "sum") else op(df))
            try:
                jit_op(jdf)  # warmup/compile
                jit_ms = bench(lambda: jit_op(jdf))
            except Exception:
                jit_ms = float("nan")

            speedup = pd_ms / jit_ms if jit_ms > 0 else float("nan")
            size_str = f"{n_rows}x{n_cols}"
            line = (
                f"{op_name:<12} {size_str:>12} {pd_ms:>9.2f}ms"
                f" {jf_ms:>9.2f}ms {jit_ms:>9.2f}ms {speedup:>7.1f}x"
            )
            print(line)
        print()


if __name__ == "__main__":
    run_benchmarks()
