# Findings: baselines, architecture, prior art

Captured 2026-10-07 to 2026-10-08. Measurements on an 8-core sandbox
with `jax 0.5.x`, `numba 0.68.0`, numpy via standard pip install.

## Standard reproducer

**File**: ENDF/B-VIII.1 U-235, `tests/data_temp/n_092-U-235_9228.dat`
(MAT 9228, Reich-Moore LRU=1 LRF=3, ~3194 resonances across 2
J-groups).

**Queries**:

- `get_reaction_xs(d, '(n,total)', e, options=RunOptions(backend=X))`
- `get_particle_production_xs(d, '(n,total)', 'n', e,
   options=RunOptions(backend=X))`

with `e = np.linspace(1e-5, 1e7, N)` for `N in {10_000, 100_000,
1_000_000}`.

## Baseline measurements (main at `333eb62`)

### `get_particle_production_xs` ('(n,total)', 'n') steady-state wall

| N | numpy | numba | jax (steady) | jax (1st, compile incl.) |
|---|---|---|---|---|
| 100 | 0.025 s | 0.037 s | 1.8 s | 5.9 s |
| 1 000 | 0.025 s | 0.039 s | 3.6 s | 7.8 s |
| 10 000 | 0.093 s | 0.093 s | 9.3 s | 14.4 s |
| 100 000 | 0.72 s | 0.42 s | 65.6 s | 71.0 s |
| 1 000 000 | 83.7 s | 41.0 s | **OOM 51 GB** | n/a |

### `get_reaction_xs` ('(n,total)') on 10M points, numba only

Approximate end-to-end 25 s on main after #349 (numba `tab1.interp`
fast path).

### Post-H1 update (2026-10-08, branch `perf/jax-rm-resonance-scan`)

Re-baselined on a 31 GB / 4C8T sandbox (see `PROGRESS.md`). After the
H1 blocked resonance scan in Reich-Moore, JAX ppxs steady state is
2.6 / 3.0 / 7.0 / 51.6 s at 1 K / 10 K / 100 K / 1 M (1 M no longer
OOMs; 3.6 GB peak). The RM reconstruct gap vs numba is ~3x at 10 K
and ~2x at 100 K. The sections below describe the pre-H1 state.

### State after the 2026-10-08 session (all pending PRs combined, scratch/integration)

U-235 ppxs (`get_particle_production_xs('(n,total)', 'n')`), jax:

| N | main b7d1f6e ss | combined ss | numba |
|---|---|---|---|
| 1K | 2.2-2.6 s | ~0.15-0.3 s | ~0.03 s |
| 10K | 2.8 s | ~0.15-0.3 s | ~0.07 s |
| 100K | 6.7 s | ~0.6 s | ~0.3 s |
| 1M | 47.7 s | ~2.0 s | ~3.3 s |

Remaining at 1M: MF3 tab1 host-lookup path ~1.0 s (host gathers +
transfer of x1/x2/y1/y2 per table), MF6 yields ~0.65 s. At small N:
eager dispatch in the per-MT composition (~45 calls of
reconstruct_resonance_xs, ~1.8 ms each; #350 memoisation helps), plus
per-shape compiles on first call (1st call ~4-5 s at 1K-10K).
Pending PRs and their order: see PROGRESS.md (fix-spi first;
composition-slice stacked on it; tab1-host on H2; yields-host on
tab1-host).

## The 51 GB OOM is in the resonance kernel, not the composition layer

Isolated: `compute_xs(d, mt=1, e, options=jax)` with 1 M points OOMs
by itself at ~51 GB. `compute_xs(d, mt=51, e, ...)` (which has no MF2
contribution) succeeds in ~0.2 s. `compute_xs(d, mt=1, e, options=
include_resonance=False)` also succeeds in ~0.4 s.

51 GB precisely matches `1_000_000 Ein * 3194 resonances * 16 B
complex128` for the Reich-Moore `(ne, nres)` intermediate inside
`_reconstruct_group`.

## The 24x per-chunk kernel gap vs. numba

Isolated at 10 K points on U-235 Reich-Moore (3194 resonances, 2
groups):

| backend | wall | ratio vs. numba |
|---|---|---|
| numba | 0.087 s | 1x |
| numpy | 2.14 s | 25x |
| JAX jit | 2.1 s | 24x |

**JAX is not uniquely slow here. numpy is just as slow.** The 24x gap
is the cost of the fixed-shape `(ne, nres)` matrix the backend-
agnostic reconstruct materialises so every intermediate has a shape
known at trace time (JAX-safe) and every leaf is differentiable.

The numba kernel sidesteps this by using a per-resonance scalar loop
that accumulates into an `(ne,)` buffer. Compact memory, SIMD-
friendly scalar arithmetic. See
`endf_userpy/mfsec_interpretation/mf2_interpretation_reichmoore_numba.py:_reconstruct_kernel`.

## Architectural map

```
user call
  get_particle_production_xs
     |
     +--> _get_particle_production_xs_impl (quantities.py:559)
             |
             +--> compute_cumulative_quantity (quantities_mt_zap/quantities.py:586)
                     |
                     +-- per admitted MT: --
                     +--> compute_prodxs (quantities_mt_zap/quantities.py:310)
                             |
                             +--> compute_yields (MF6 ZAP yield interp)
                             +--> compute_xs (quantities_mt_zap/quantities.py:212)
                                     |
                                     +--> compute_reconstructed_cross_section
                                             |
                                             +--> compute_cross_section_agnostic
                                             |       |
                                             |       +--> tab1.interp
                                             |              |
                                             |              +-- numba fast path: _interp_full_numba
                                             |              +-- else: _apply_law_vectorised (xp-agnostic, 5-branch)
                                             +--> reconstruct_resonance_xs
                                                     |
                                                     +--> _reconstruct_lru1_range  (MLBW / RM / RML dispatch)
                                                     +--> _reconstruct_urr_range   (URR)
                                                             |
                                                             +-- numba path if wants + available
                                                             +-- else: mf2_interpretation_<name>.reconstruct
```

## What is already optimised for JAX in the repo

- **`jax.jit`-safe reconstruct bodies**: MLBW, RM, RML, URR all have
  Option A "fixed-shape indicator mask" architecture. See the
  docstring at
  `endf_userpy/mfsec_interpretation/mf2_interpretation_reichmoore.py:60`
  for the explicit trade-off statement.
- **`chunked_chi2`** at `endf_userpy/primitives/fit_helpers.py`:
  reference pattern for scan + checkpoint under `jax.grad`. The
  module docstring records measured 6.6 GB -> 0.6 GB memory and 4.3 s
  -> 1.5 s per iter on an MLBW NRES=500 NE=200000 fit.
- **`_endf_interp1d_traced_x`** at
  `endf_userpy/primitives/interpolation.py:361`: a different TAB1
  interpolation path with a JAX-friendly trace variant. Used in
  distribution + broadening paths. Not used in MF3 cross-section
  interpolation.
- **Backend adapter**'s `xp.scan` (-> `lax.scan`) and `xp.map` (->
  `lax.map` or Python fallback) at
  `endf_userpy/primitives/array_ns.py:260`.
- **Numpy chunking helper**: `_numpy_chunk_size` and the recursive
  chunking wrapper in `mf2_interpretation_mlbw.py` and
  `mf2_interpretation_reichmoore.py`. 512 MB default cap. Was
  numpy-only until the handoff prototype widened it to JAX.

## Prior attempt on this branch (`perf/jax-ppxs-cum-scan`, local only)

Status at handoff: branch exists locally in the user's working
checkout; **not pushed, not merged**. Changes compared to main:

- `mf2_interpretation_mlbw.py`: new `_chunked_reconstruct` helper.
  Numpy path = plain Python loop (unchanged behaviour). JAX path =
  `jax.lax.scan` over the main chunks + a single padded scan for the
  remainder. Chunking gate widened from `xp.name == 'numpy'` to
  `xp.name in ('numpy', 'jax')`.
- `mf2_interpretation_reichmoore.py`: same chunking widening,
  delegating to the shared `_chunked_reconstruct` helper.

Measured effect on this branch at 100 K points (ppxs):

- Before: 65 s steady state (python-for chunking in eager JAX).
- After: 51 s steady state (`lax.scan` traces body once per shape).
- Correctness: `max_rel_diff <= 1e-14` vs. numba.
- 1 M-point OOM: fixed (runs; latency ~10 min, see M2 for the
  underlying kernel-body cost).

The scan body does **not** currently wrap with `jax.checkpoint`.
`chunked_chi2` does; add it if the first autodiff user complains
about memory blowing up during the backward pass.

## What was ruled out during the handoff work

- **`jax.block_until_ready` after every per-MT accumulation step** in
  `compute_cumulative_quantity`. Did not help on throughput or
  memory. Reason: the OOM is per-MT, not accumulated.
- **`lax.scan` over the per-MT loop** in
  `compute_cumulative_quantity`. Would help if the memory bomb were
  cross-MT. It is not. Prototyped and reverted.
- **vmap over chunks**: would stack all chunk outputs simultaneously
  (same OOM pattern). scan is the right primitive for this axis.
- **`@jax.jit`-wrapping `reconstruct` directly** (without scan over
  resonances): confirmed to work (there is already a test:
  `tests/test_mf2_interpretation_reichmoore.py::
  test_rm_jax_jit_matches_direct_call`). Modest ~2x speed-up vs.
  eager at the body level. The 24x architectural gap persists.

## Numeric sanity constants

- U-235 LRU=1 LRF=3 range: nres = 3194, ngroups = 2, EL ~ 1e-5 eV, EH
  ~ 2250 eV.
- Numba Reich-Moore reconstruct, 10 K points: 87 ms.
- Numpy / JAX Reich-Moore reconstruct, 10 K points: ~2.1 s.
- JAX first-call compile for `reconstruct` at a given shape: ~0.3-1 s
  depending on `nres`.
- Default chunk cap `NUMPY_MAX_INTERMEDIATE_BYTES = 512 MB` =>
  chunk_size at `nres=3194` is ~10 K Ein per chunk.

## Reference numba kernel (template to match)

`endf_userpy/mfsec_interpretation/mf2_interpretation_reichmoore_numba.py`:

- `@njit(cache=True, parallel=True, fastmath=True)`
- Outer `prange(ne)` loop (parallel over Ein).
- Inner scalar loop over resonances. Each iteration updates the
  running `(sct, cap, fis)` accumulators for that Ein.
- Memory: O(ne), not O(ne * nres).
- Reads data arrays once per resonance per Ein.

The challenge for a JAX analogue: replicating the per-resonance scan
with a `(ne,)` accumulator while preserving `jax.grad` support
through resonance parameters.
