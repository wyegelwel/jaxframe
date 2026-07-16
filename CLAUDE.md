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
| `docs/JOURNAL.md` | Development journal — optimization findings, lessons learned |

**Keep `docs/JOURNAL.md` updated** during and after tasks. Record what was tried, what worked, what didn't, and why. This is how we learn from past work and avoid repeating mistakes.

## Non-Differentiable Operations

Nearly everything is JIT-compatible AND differentiable. `jax.grad` follows the
JAX convention: subgradients at non-smooth points (min/max/sort/top-k), grads
flow through gathers/permutations. The remaining non-differentiable ops are
those whose output is not real-valued or whose gradient is zero a.e.:

| Operation | Reason |
|-----------|--------|
| `isna`, `notna`, `isnull`, `notnull`, `isin`, `all`, `any`, comparisons (`eq`,`ne`,`lt`,`le`,`gt`,`ge`) | Boolean output — not a real-valued function |
| `count`, `nunique`, `argsort`, `argmin`, `argmax`, `idxmin`, `idxmax`, `searchsorted` | Integer/discrete output |
| `rank` | Step function — gradient is zero a.e. |
| `round` | Step function — gradient is zero a.e. |
| `astype` | Discrete type conversion |
| `groupby.prod` | JAX limitation: `scatter_mul` gradients require `unique_indices=True` |
| `groupby.count` | Integer output |

Everything else — including `min`/`max`/`median`/`quantile` (subgradient /
interpolation weights), `sort_values`/`nlargest`/`nsmallest` (grad through the
gather), `ffill`/`bfill`/`interpolate`, `cummax`/`cummin`, rolling/expanding
min/max, `groupby.min/max/first/last` — **is differentiable**.

Eager-only ops (not JIT-compatible, by necessity): shape-changers
(`dropna`, `drop_duplicates`, `query`, boolean-mask indexing, `merge`/`join`
structure discovery) and unique-based ops (`unique`, `value_counts`, `mode`,
`nunique`, `factorize`, `duplicated`, `groupby` discovery). Their *data paths*
still use JAX gathers, so downstream compute stays on-device.

When adding new ops, update this table and `_JAX_COMPAT` in dataframe.py.

## Roadmap

Priority order:

1. **Real-world dogfood** — Use jaxframe for an actual ML workflow (load → clean → feature engineer → train with grad) to find gaps. API is at 100% pandas parity; dogfooding is the best remaining bug-finder.
2. **Merge to main + packaging** — README claims are current; publish to PyPI.
3. **np → jnp residual audit** — Most data paths are jnp; remaining np is in eager structure discovery (unique/duplicated/merge key matching), which is intentional. Revisit only if GPU-resident structure discovery becomes a bottleneck.
4. **vmap coverage for new ops** — sort_values/rank/ffill are jit+grad tested; extend the vmap matrix to them.
5. **Small-n dispatch floor** — eager ops cost ~250us/dispatch on this box (WSL2 GPU). If small-data latency matters, consider a CPU-array fast path below a size threshold.

Done (2026-07): 1-1 pandas API coverage (enforced by tests/test_api_coverage.py), JIT+grad maximization (sort/rank/top-k/fills traceable; subgradient convention), latency pass (column cache, self-JIT kernels, fused sort; eager beats pandas at 100k rows on most numeric ops).

## Tools

- **Package manager**: uv (`uv sync`, `uv run`)
- **Linter/formatter**: ruff
- **Type checker**: ty
- **Tests**: pytest
