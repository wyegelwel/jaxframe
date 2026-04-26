# JaxFrame

JAX-based DataFrame library with pandas-compatible API. Owner: Wil Yegelwel.

## Core Goal

**Maximize JAX operation support to 100%.** Every DataFrame and Series method must be JIT-compilable and, where mathematically possible, differentiable via `jax.grad`. The only acceptable skips are mathematically necessary (e.g., `min`/`max` grad is non-smooth, boolean outputs like `isna` are not differentiable).

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

## Tools

- **Package manager**: uv (`uv sync`, `uv run`)
- **Linter/formatter**: ruff
- **Type checker**: ty
- **Tests**: pytest
