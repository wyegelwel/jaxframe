# JaxFrame

JAX-based DataFrame library with pandas-compatible API. Owner: Wil Yegelwel.

## Core Goal

**Maximize JAX operation support to 100%.** Every DataFrame and Series method must be JIT-compilable and, where mathematically possible, differentiable via `jax.grad`. The only acceptable skips are mathematically necessary (e.g., `min`/`max` grad is non-smooth, boolean outputs like `isna` are not differentiable).

## Architecture: Hybrid Eager/JIT

The core architectural principle: **separate structure discovery from data computation**.

- **Structure discovery** (shapes, indices, group assignments, join keys, unique values) happens **eagerly** — outside `jax.jit` traces. Use `jnp.*` (not `np.*`) so it still runs on GPU. Results are concrete arrays that become static `aux_data` in JAX pytrees.
- **Data computation** (arithmetic, reductions, gathers, scatters) uses **JAX ops** (`jnp.*`, `jax.ops.segment_*`, `jnp.take`). These are JIT-compiled and dispatch to GPU/TPU.
- **"Eager" ≠ CPU.** Eager just means "not inside a JIT trace." `jnp.unique` called eagerly on GPU data runs on GPU and returns a concrete GPU array. Only `np.*` forces data to CPU — avoid it for data ops.

This means **every operation is GPU-compatible**:
- `groupby`: group discovery is eager → segment ops are JIT (already works on GPU)
- `merge/join`: key matching computes indices eagerly → data gathering via `jnp.take` is JIT
- `value_counts`: unique value discovery is eager → counting via segment_sum is JIT
- `sort_values`: argsort is eager → data permutation via `jnp.take` is JIT
- `drop_duplicates`: duplicate detection is eager → row selection via `jnp.take` is JIT

When adding new ops, always ask: "What is structure, and what is computation?" Put structure in eager code, computation in JAX ops.

## Implementation Rules

- **Always use `_apply_blockwise` or `_from_parts`** when returning new DataFrames. Never reconstruct through `DataFrame.__init__` from inside operations — it calls `np.asarray` which breaks JAX tracing.
- **Test-first**: write failing tests in `test_pandas_mirror.py` before implementing.
- **Same lambda for both libraries**: equivalence tests use a single lambda that works on both `pd.DataFrame` and `jaxframe.DataFrame`.

## Verification Scheme

Three pillars, all must pass before committing:

### 1. Correctness — `uv run pytest tests/test_pandas_mirror.py -v`
Parametrized equivalence tests. Each test runs the same operation on pandas and jaxframe, compares results. Adding a test = adding a tuple.

### 2. JAX Compatibility — `uv run pytest tests/test_jax_transforms.py -v`
Every operation tested under JIT and grad. Each operation is a tuple: `(name, op, supports_jit, supports_grad)`. Goal: 100% JIT support, grad support wherever mathematically valid.

### 3. Performance — `uv run python tests/test_benchmarks.py`
Pandas vs jaxframe (eager and JIT) at various data sizes. JaxFrame+JIT should be faster than pandas for numeric operations.

### Full suite: `uv run pytest tests/ -v`

## Session Workflow

1. Run existing tests (green baseline)
2. Write failing tests in `test_pandas_mirror.py`
3. Implement in `dataframe.py` using `_apply_blockwise` / `_from_parts`
4. Make tests green
5. Run `ruff check --fix` and `ruff format`
6. Run `ty check`
7. Run code simplification review (reuse, quality, efficiency)
8. Add JAX transform entries to `test_jax_transforms.py`
9. Run full suite, commit (at least once mid-phase, always at end)

## Key Files

| File | Purpose |
|------|---------|
| `jaxframe/dataframe.py` | Core implementation — DataFrame, Series, pytree registration |
| `tests/conftest.py` | Shared data fixtures and `run_equiv()` helper |
| `tests/test_pandas_mirror.py` | Parametrized pandas equivalence tests |
| `tests/test_jax_transforms.py` | JIT/grad compatibility matrix |
| `tests/test_benchmarks.py` | Performance comparison runner |

## Non-Differentiable Operations

All ops are JIT-compatible. The following are **not** differentiable (`jax.grad`), with reasons:

| Operation | Reason |
|-----------|--------|
| `min`, `max` | Non-smooth (gradient undefined at argmin/argmax) |
| `median`, `quantile` | Non-smooth (sort-based, gradient undefined at boundaries) |
| `isna`, `notna`, `isnull`, `notnull` | Boolean output — not a real-valued function |
| `count` | Integer output (sum of booleans) — not real-valued |
| `sort_values`, `nlargest`, `nsmallest` | Permutation-based (argsort) — discrete, not differentiable |
| `groupby.min`, `groupby.max` | Same as min/max — non-smooth |
| `groupby.count` | Same as count — integer output |
| `groupby.prod` | JAX limitation: `scatter_mul` gradients require `unique_indices=True` |
| `groupby.first`, `groupby.last` | Index-based gather via segment_min/max on indices — discrete |
| `all`, `any` | Boolean output — not a real-valued function |
| `round` | Step function — gradient is zero almost everywhere |
| `idxmin`, `idxmax` | Discrete (argmin/argmax) — not differentiable |
| `isin` | Boolean output — not a real-valued function |

Everything else (sum, mean, std, var, arithmetic, clip, where, fillna, cumsum, cumprod, shift, apply, reverse ops, groupby.sum/mean/std/var, transform) **is differentiable**.

When adding new ops, update this table. If an op is non-differentiable, document why.

## Tools

- **Package manager**: uv (`uv sync`, `uv run`)
- **Linter/formatter**: ruff
- **Type checker**: ty
- **Tests**: pytest
