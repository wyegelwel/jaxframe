# JaxFrame Development Journal

Record of optimization work, findings, and lessons learned.

---

## 2026-04-29: Benchmark Honesty + Expanding Rewrite

### Problem
CPU benchmarks were dishonest — ops returning DataFrame/Series skipped `block_until_ready()`, so cumsum/fillna/clip reported 0.01ms regardless of size. Expanding window used O(n^2) gather matrix (1000-3000x slower than pandas).

### Changes
- Added `block_until_ready()` to DataFrame and Series classes
- Rewrote Expanding from O(n^2) gather matrix to O(n) `jax.lax.scan`-based implementation
- Vectorized `diff()` and `pct_change()` — replaced per-column Python loops with block-level ops
- Added `_reduce_axis0` to avoid expensive `.values` concatenation for column reductions

### Findings
- `block_until_ready()` is essential for honest benchmarking on JAX — without it, async dispatch makes everything look instant
- `lax.scan` is the right pattern for expanding windows (same as EWM class already used)
- Per-column Python loops are a major bottleneck — always operate on blocks

---

## 2026-04-29: NaN Handling via Fused lax.reduce

### Problem
`jnp.nansum` is 6-8x slower than `jnp.sum` on CPU because XLA doesn't fuse `isnan + where + reduce` into a single pass.

### Key Discovery
Embedding the NaN check directly in a `lax.reduce` combiner forces XLA to fuse everything:
```python
def combiner(a, b):
    return jnp.where(jnp.isnan(a), 0.0, a) + jnp.where(jnp.isnan(b), 0.0, b)
jax.lax.reduce(block, jnp.float32(0.0), combiner, [0])
```
This is nearly as fast as plain `jnp.sum` (1.5ms vs 1.3ms at 1M rows), vs 7.5ms for `jnp.nansum`.

### Critical Gotcha: Eager Mode
`lax.reduce` with a custom combiner is **76ms in eager mode** but 1.5ms under JIT. The custom combiner dispatches each element through Python when not JIT'd.

**Solution: Self-JIT pattern** — decorate the function with `@jax.jit` so it's always compiled:
```python
@staticmethod
@jax.jit
def _fast_nansum(block):
    ...
```
This gives 1.5ms even when called eagerly from DataFrame methods.

### Results
- `_fast_nansum/min/max/prod`: 5-7x speedup over `jnp.nan*` equivalents
- col_sum: 7.4x faster than pandas (was slower)

---

## 2026-04-29: Multi-Operand lax.reduce

### Key Discovery
`lax.reduce` supports **multiple operands via tuples**! The combiner receives `(tuple_a, tuple_b) -> tuple`:
```python
total, count = jax.lax.reduce(
    (clean, valid),                           # tuple of operands
    (jnp.float32(0.0), jnp.float32(0.0)),    # tuple of inits
    lambda a, b: (a[0]+b[0], a[1]+b[1]),      # combiner on tuples
    [0],                                       # reduction dims
)
```
This computes sum AND count in a **single fused pass**.

### Results
- nanmean: 4.6ms (was 14.3ms jnp.nanmean, 10.7ms pandas) — 2.3x faster than pandas
- nanvar: 5.1ms (was 26.3ms jnp.nanvar, 51.7ms pandas) — 10x faster than pandas
- col_std improved from 2.0x to **11.3x** faster than pandas

### Gotcha: No Backward Pass
Multi-operand `lax.reduce` does **not support `jax.grad`**. During gradient tracing, blocks become `Zero` types that can't be passed to the tuple reduce. Single-operand `lax.reduce` works fine with grad.

**Workaround**: Use multi-operand for axis=0 (column) reductions (not grad-tested), use single-operand `_fast_nansum` for scalar (axis=None) paths that need grad support.

---

## 2026-04-29: Systematic JIT-ing

### Approach
After discovering the self-JIT pattern, applied it everywhere:
- **Rolling**: `_make_rolling_fn` factory creates JIT'd block functions
- **Expanding**: Module-level `@jax.jit` scan functions (`_expanding_sum_block`, etc.)
- **EWM**: Module-level `@jax.jit` scan functions (`_ewm_mean_block`, `_ewm_var_block`)
- **GroupBy**: `@functools.partial(jax.jit, static_argnums=(2,))` for segment ops

### Gotcha: static_argnums for GroupBy
`num_segments` in `jax.ops.segment_sum` must be a concrete int, not a traced value. Fix:
```python
@functools.partial(jax.jit, static_argnums=(2,))
def _segment_sum(data, ids, num):
    return jax.ops.segment_sum(data, ids, num)
```

### Gotcha: Bool dtype in _fast_nansum
`df.isna().sum()` passes bool blocks to `lax.reduce` with float32 init — type mismatch. Fix: `block = block.astype(jnp.float32)` at the start.

---

## 2026-04-29: Rolling Mean/Var/Std Optimization

### Problem
Rolling mean used `jnp.nanmean` on gathered windows — same 6-8x NaN overhead as column reductions.

### Fix
Replaced `jnp.nanmean/nanvar/nanstd` with explicit `sum/count` (for mean) and `sum/sumsq/count` (for var/std):
```python
clean = jnp.where(mask, gathered, 0.0)
count = jnp.sum(mask, axis=1)
total = jnp.sum(clean, axis=1)
result = total / jnp.maximum(count, 1)
```

### Results
- rolling_mean: 0.83x → **1.6x** faster than pandas at 1M rows

---

## 2026-04-29: Shift Optimization

### Problem
`shift()` used `jnp.concatenate([pad, block[:-periods]])` — creates a pad array + slices + concatenates.

### Fix
Replaced with `jnp.roll` + `.at[].set()`:
```python
rolled = jnp.roll(block, periods, axis=0)
return rolled.at[:periods].set(fill_value)
```

### Results
- shift: 20ms → 14.5ms at 1M rows (still slightly behind pandas at 13.4ms due to `.at[].set()` overhead)

---

## 2026-04-29: Scalar Reduction Dispatch Overhead

### Problem
Scalar mean (axis=None) does 4 separate Python→JAX calls per block: `_fast_nansum(block)`, `.sum()`, `jnp.where(isnan, 0, 1)`, `valid.sum()`.

### Fix
Created `_fast_nansum_and_count` that fuses all 4 operations into one JIT boundary.

### Results
- scalar mean: 10.3ms → 8.0ms
- scalar var: 17.8ms → 13.5ms

### Lesson: Don't Over-Fuse
Tried `_block_sum_count` using `lax.reduce([0, 1])` to reduce both axes at once — actually **slower** than reducing axis=0 then `.sum()`. XLA optimizes single-axis reductions better.

Also tried using the fused function for scalar sum (which only needs sum, not count) — **5x regression** because it computes the unused count. Use the right tool for each path.

---

## 2026-04-30: Eager np.argsort for Sort/Rank

### Problem
sort_values (0.25x) and rank (0.25x) were 4x slower than pandas because `jnp.argsort` on CPU goes through XLA's sort backend, which is much slower than NumPy's optimized introsort.

### Key Insight
Sort key computation and rank ordering are **structure discovery** — they determine permutation indices, not data values. Per our architecture, structure discovery should be eager. The data gathering (applying the permutation) is the computation that stays as JAX ops.

### Changes
- **sort_values**: `jnp.argsort` → `np.argsort(np.asarray(sort_col))` for key, `jnp.take(block, order, axis=0)` for data gather
- **rank ordinal**: Full `np.argsort` + `np.put_along_axis` for inversion (avoids double argsort)
- **rank average**: Switched to numpy for the entire tie-averaging logic — `np.bincount` is much faster than `jnp.zeros.at[].add()`
- **fillna**: Inlined `_from_parts` to skip `_apply_blockwise` re-keying overhead
- **scalar reductions**: Single-block fast path to skip loop when only 1 dtype block

### Trade-off
sort_values, rank, nlargest, nsmallest are now **not JIT-compatible** (they call `np.asarray` which breaks tracing). This is acceptable because:
1. They were already non-differentiable
2. They follow the eager structure-discovery architecture
3. The 10-16x speedup is massive

### Results (1M rows, 10 cols)
| Op | Before | After | Change |
|----|--------|-------|--------|
| sort_values | pd=82ms, jf=328ms (0.25x) | jf=33ms (**2.6x faster**) | 10x |
| rank | pd=1623ms, jf=6400ms (0.25x) | jf=410ms (**4.0x faster**) | 16x |
| shift | pd=13.4ms, jf=14.5ms (0.92x) | jf=13.0ms (**1.13x faster**) | now wins |
| fillna | pd=5.4ms, jf=6.5ms (0.82x) | jf=6.2ms (0.87x) | marginal |
| scalar mean | pd=4.3ms, jf=8.0ms (0.54x) | jf=9.2ms (0.46x) | no change |

### Lesson
**Don't fight the hardware.** NumPy's sort is 10x faster than JAX's on CPU. When an operation is inherently eager (structure discovery), use NumPy. Reserve JAX for data computation that benefits from JIT/GPU.

---

## Summary: Performance Lessons

1. **lax.reduce with custom combiner** is the key to fast NaN-aware reductions on CPU
2. **Self-JIT pattern** (`@staticmethod @jax.jit`) eliminates eager mode overhead for lax.reduce
3. **Multi-operand lax.reduce** enables single-pass mean/var but breaks `jax.grad`
4. **Avoid jnp.nan* functions** — they're 5-8x slower than equivalent fused reductions
5. **lax.reduce axis=0 then .sum() beats lax.reduce([0,1])** — XLA optimizes better
6. **Python dispatch overhead matters** — fuse operations into single JIT calls
7. **Block-level operations beat per-column loops** — always operate on dtype blocks
8. **jnp.roll + .at[].set() beats jnp.concatenate** for shift-like ops
9. **Don't compute unused outputs** — even inside JIT, extra work costs time
10. **Use NumPy for eager structure discovery** — `np.argsort` is 10x faster than `jnp.argsort` on CPU

## Current Benchmark Score (1M rows, 10 cols)

| Category | Operations | JF wins | PD wins |
|----------|-----------|---------|---------|
| Scalar reductions | sum, min, max, prod, std, var | 6 | 1 (mean) |
| Column reductions | col_sum, col_mean, col_std | 3 | 0 |
| Arithmetic chains | mul+sum, pow+sum, chain | 3 | 0 |
| Cumulative | cumsum, cumprod | 2 | 0 |
| Shift/diff | diff, shift | 2 | 0 |
| Rolling | rolling_sum, rolling_mean | 2 | 0 |
| Expanding | expanding_sum, expanding_mean | 2 | 0 |
| EWM | ewm_mean | 1 | 0 |
| Data cleaning | clip, where | 2 | 1 (fillna ~parity) |
| GroupBy | sum, mean, std | 3 | 0 |
| Sorting | sort_values, rank | 2 | 0 |
| **Total** | | **28** | **2** |

Remaining pandas wins: scalar mean (Python dispatch overhead — 0.46x), fillna (~parity at 0.87x). Scalar min/max lose at 100K rows only (small data overhead, win at other sizes).
