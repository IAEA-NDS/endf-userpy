# Playbook: ranked attempts

> **Status 2026-10-09.** H1 (all four formalisms: #353 / #354 RM, MLBW,
> RML; URR via #359), H2 (#360-#362, extended with a host-side panel
> lookup + jitted arithmetic) and H3 (checkpointed scan bodies in every
> new kernel) are **done**; H5 was superseded by H1. Not done: H4
> (pytree registration; nothing needed it yet) and H6 (jitted
> `get_reaction_xs`). Additional landed work outside this list:
> composition slicing on jax (#356), jax preproc bulk gather (#357),
> scalar-L factor fast path (#358). U-235 ppxs jax at 1M points: ~2 s
> (main before the arc: 47.7 s; numba ~3.3 s). Remaining opportunities
> are listed in `PROGRESS.md` (last entries) and summarised as H7-H10
> below.

Hypotheses, ranked by expected impact. Each entry lists what, why,
where, how to measure, risks, and a template to copy from.

Order matters: start at the top. If an attempt lands and improves
the baseline, update `FINDINGS.md`, log in `PROGRESS.md`, and move
on to the next one.

---

## H1 - Per-resonance `lax.scan` in Reich-Moore `reconstruct` [BIG] [DONE for RM 2026-10-08, see PROGRESS.md; MLBW/RML mirror open]

**What**
Rewrite the inner `_reconstruct_group` body in
`mf2_interpretation_reichmoore.py` so it scans **over the resonance
axis** with an `(ne,)` accumulator, instead of materialising the
`(ne, nres)` complex matrix.

**Why**
The 24x per-chunk gap vs. numba (see `FINDINGS.md`) is explained by
the matrix form. The numba kernel sidesteps it with a per-resonance
scalar loop. `lax.scan` over the resonance axis with a running
`(ne,)` buffer replicates that memory profile under JAX while
staying `jax.jit`-safe and `jax.grad`-differentiable through the
scanned resonance parameters. Potential 5-20x on the kernel body
(numba does ~24x, scan loses some to XLA overhead).

**Where**
- `endf_userpy/mfsec_interpretation/mf2_interpretation_reichmoore.py`:
  function `_reconstruct_group` (around line 172). This builds the
  `(ne, nres)` matrix today.
- The outer `reconstruct` function stays mostly the same. The scan
  replaces the matrix-forming body.
- Mirror in `mf2_interpretation_mlbw.py` and
  `mf2_interpretation_rml.py` once the pattern lands cleanly.

**How to measure**
Run `MEASURE.md::reconstruct_only` at 10 K and 100 K points. Target:
<= 500 ms at 10 K (down from 2.1 s), <= 5 s at 100 K (down from
~21 s). Run full `MEASURE.md::ppxs_scan` at every size for the
end-to-end check.

**Template to copy**
`endf_userpy/primitives/fit_helpers.py::_chunked_chi2_jax`. It
shows the scan + checkpoint pattern. For this hypothesis the inputs
to the scan are the per-resonance `(res_er, res_gn, res_gg, res_gf1,
res_gf2, res_group)` tuples; the carry is the accumulator dict
`{sct, cap, fis}` of shape `(ne,)` each.

Rough skeleton:

```python
def body(carry, res_i):
    er_i, gn_i, gg_i, gf1_i, gf2_i, grp_i = res_i
    # per-resonance contribution to sct / cap / fis at every e
    # (uses jnp ops; shapes known at trace time)
    dsct, dcap, dfis = _one_resonance_contrib(
        er_i, gn_i, gg_i, gf1_i, gf2_i, grp_i,
        e, group_params, ...,
    )
    sct, cap, fis = carry
    return (sct + dsct, cap + dcap, fis + dfis), None

init = (jnp.zeros_like(e), jnp.zeros_like(e), jnp.zeros_like(e))
(sct, cap, fis), _ = lax.scan(
    body, init,
    (data.res_er, data.res_gn, data.res_gg,
     data.res_gf1, data.res_gf2, data.res_group),
)
```

**Risks**
- `_one_resonance_contrib` has to be a traceable function with
  shapes known at trace time; it may need `lax.switch` over group
  index if different groups follow different code paths.
- Autodiff through `jnp.where`-masked branches needs the same `_safe`
  guards `_apply_law_vectorised` uses (positive-stand-in inputs
  before every `log`/`division`). See
  `endf_userpy/primitives/tab1.py` lines 185-220 for the pattern.
- Correctness: run the full autodiff test suite (see
  `MISSION.md::acceptance`). The `(nres, ngroups)` indicator mask is
  what makes `jax.jit(reconstruct)` work; a scan replacement must
  preserve that behaviour (both tests in
  `tests/test_mf2_interpretation_reichmoore.py` section "jax.jit
  compatibility").

**Fallback if blocked**
Try `lax.map` (independent-iteration map) in place of scan if group
dispatch can be unified. Memory is worse (stacks per-resonance
contributions) but it may trace cleaner.

---

## H2 - JAX-friendly `tab1.interp` fast path [MEDIUM]

**What**
Add a JAX-specific interp kernel analogous to
`tab1_numba._interp_full_numba`: per-point branch so dead
interpolation laws don't allocate.

**Why**
`_apply_law_vectorised` evaluates all five candidate laws
(const / lin-lin / lin-log / log-lin / log-log) for every point and
`jnp.select`s. Four wasted `log`/`exp` per point. Not a huge factor
(~4x on this one function) but it is called 93 x per
`get_particle_production_xs`, so cumulative savings scale with MT
count.

**Where**
- `endf_userpy/primitives/tab1.py`: `_apply_law_vectorised` body.
- Create `endf_userpy/primitives/tab1_jax.py` as a sibling to
  `tab1_numba.py`, with a `@jax.jit`-friendly kernel.
- The gate in `tab1.interp` is already shaped as `if xp.wants + xp.has:
  ... tab1_numba._interp_full_numba(...)`. Add a parallel branch
  for `xp.name == 'jax'` that routes to `tab1_jax`.

**How to measure**
`MEASURE.md::interp_only` micro-benchmark. Target: within 2x of the
numba path.

**Template to copy**
- Numba template: `endf_userpy/primitives/tab1_numba.py:_interp_full_numba`.
- JAX trace-safe log-ratio arithmetic: see the `_safe` variables and
  `jnp.where` masking at `tab1.py:_apply_law_vectorised` lines
  185-258. Preserve this backward-pass safety discipline when
  rewriting.

**Risks**
- JAX cannot cheaply do per-point branching the way numba can. The
  real win here is NOT per-point dispatch; it is **per-panel segment
  dispatch**: for each INT code appearing in `tab.intp`, run the
  corresponding law on exactly the sub-slice of query points that
  fall in that INT's segment. One `lax.switch` per panel is cheap;
  one per point is a full-mesh `lax.cond` which is slow.
- Correctness: the existing `tests/test_tab1_*.py` suite is thorough;
  bit-identical match required.

---

## H3 - `jax.checkpoint` on the chunk scan body [SMALL on throughput, BIG on grad memory] [folded into H1 for RM]

**What**
Wrap the scan body in `_chunked_reconstruct` with `jax.checkpoint`
so `jax.grad(get_reaction_xs)(theta)` has bounded memory even on
large meshes.

**Why**
Forward-only evaluation does not need checkpoint (slight slowdown).
Backward (`jax.grad`) does: without checkpoint, every scan iteration
keeps its forward activations alive for the backward pass, so memory
peaks are proportional to the number of chunks. With checkpoint, the
backward pass recomputes each iteration's activations, memory stays
chunk-sized.

This is the exact pattern `chunked_chi2` uses; `_chunked_reconstruct`
should match.

**Where**
`endf_userpy/mfsec_interpretation/mf2_interpretation_mlbw.py`:
`_chunked_reconstruct`. Decorate `body` with `jax.checkpoint`.

**How to measure**
Autodiff memory benchmark: `jax.grad(sum(reconstruct(data, e,
xp).values()[0]))(data.res_er)` at 100 K points. Peak RSS before
and after. The `chunked_chi2` module docstring measured 6.6 GB ->
0.6 GB on a comparable setup; expect similar order of magnitude.

**Risks**
Forward-only wall time will slightly increase (one extra pass per
iteration during the forward trace). If this exceeds 5%, measure
whether the autodiff users would benefit regardless.

**Template**
`endf_userpy/primitives/fit_helpers.py:_chunked_chi2_jax`. The
`@jax.checkpoint` on `body` is one line.

---

## H4 - Pytree-register `MLBWData`, `RMData`, `RMLData`, `URRData` [ENABLER]

**What**
Register the four preproc dataclasses as JAX pytrees so they can
cross `@jax.jit` boundaries as arguments without `static_argnums`
gymnastics.

**Why**
Not a direct speedup by itself. **Enables** passing the dataclass
through `@jax.jit` cleanly, which many of the ideas above
assume. Without pytree registration, callers either close over the
dataclass (works, but forces one-trace-per-dataclass) or pass each
field separately (ugly).

**Where**
At the bottom of each dataclass file:
`mf2_interpretation_mlbw.py`, `..._reichmoore.py`, `..._rml.py`,
`..._urr.py`.

**Template**
```python
from jax.tree_util import register_pytree_node

def _mlbw_flatten(data):
    arrays = (data.abn, data.spi, data.ki, ..., data.res_er, ...)
    static = (data.group_l.tolist(), ...)  # or stash in aux
    return arrays, static

def _mlbw_unflatten(static, arrays):
    return MLBWData(...)

register_pytree_node(MLBWData, _mlbw_flatten, _mlbw_unflatten)
```

**Risks**
Low. Should not change any numeric output on any backend. Verify
`numba` and `numpy` tests still pass (dataclass construction stays
identical outside the registration call).

---

## H5 - Resonance-chunking for the kernel [RULED OUT during handoff]

Scanning over Ein chunks has been prototyped. See
`FINDINGS.md::prior attempt`. Memory fix, modest speed uplift. Do
not re-explore unless H1 is blocked, in which case consider
`jax.checkpoint` + larger chunks + resonance scan combined.

---

## H6 - `@jax.jit` on `get_reaction_xs` directly [SPECULATIVE]

**What**
Expose a `get_reaction_xs_jit` convenience wrapper that users wrap
their query mesh around for pre-compiled hot loops.

**Why**
Would amortise compile cost on repeat queries at identical shapes.

**Where**
`endf_userpy/quantities.py`.

**Risks**
Admission heuristic + MT iteration are Python-level data-dependent
on the file. Would need to restructure or document that the jit
caches per `(file, mesh_shape)`. Low priority until H1-H2 land.

---

## Explicitly NOT a hypothesis (ruled out)

- **`jax.block_until_ready` after each MT accumulation**: no effect on
  memory or throughput. The OOM is per-MT, not accumulated.
- **`vmap` over chunks**: stacks all chunk outputs in memory, same
  OOM pattern as the pre-chunking state.
- **A `lax.scan` over the per-MT loop in `compute_cumulative_quantity`**:
  the memory bomb is inside one MT's call, not accumulated. scan
  there costs more than it saves.


---

## H7 - Avoid redundant resonance reconstruction across MTs [MEDIUM, small N]

Within one `get_particle_production_xs` / `get_reaction_xs` call the
same MF2 range is reconstructed once per MT that maps to it (U-235:
twice). A per-call cache keyed on (range, energy array identity) in
`resonance_composition` would halve the resonance work. Note: a
reconstruction cache was attempted and abandoned before (#306); read
that history first.

## H8 - Small-N floor: per-MT eager dispatch [MEDIUM]

At 1K-10K points the jax path is ~0.15 s steady state, dominated by
~45 per-MT composition calls of eager ops. Candidates: #350 preproc
memoisation, fewer eager ops in `reconstruct_resonance_xs` for MTs
with no resonance contribution, jitting the per-MT combination.

## H9 - First-call compile cost [MEDIUM]

jax first call is still 4-5 s at 1K-10K points (per-shape op
compilation). Candidates: fewer distinct shapes (pad query meshes to
buckets), persistent compilation cache (`jax.config.update(
'jax_compilation_cache_dir', ...)`) documented for users.

## H10 - 1M-point interpolation transfers [SMALL]

At 1M points ~1.6 s remains in MF3 / MF6-yield interpolation, mostly
host gathers of x1 / x2 / y1 / y2 and their host-to-device transfer.
Candidate: transfer the int32 bracket index and gather inside the
jitted kernel with tables padded to size buckets.
