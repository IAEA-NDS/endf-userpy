# Progress log

Append one entry per attempt. Keep it short. Format:

```
## YYYY-MM-DD <short tag>

**Hypothesis**: H1 / H2 / ad-hoc / etc.
**Branch**: perf/<slug>
**Change in one sentence**: ...
**Measured**:
  before: ...
  after: ...
**Decision**: merged / dropped / parked - and why.
**Follow-ups**: anything the next attempt should know.
```

---

## 2026-10-07 handoff baseline captured

**Hypothesis**: n/a (baseline).
**Branch**: `main` at `333eb62`.
**Measured**:

  `get_particle_production_xs('(n,total)', 'n', e)` on U-235
  ENDF/B-VIII.1, 8-core sandbox:

  | N | numpy | numba | jax ss |
  |---|---|---|---|
  | 10 K | 0.093 s | 0.093 s | 9.3 s |
  | 100 K | 0.72 s | 0.42 s | 65.6 s |
  | 1 M | 83.7 s | 41.0 s | OOM 51 GB |

**Decision**: no code change. See `FINDINGS.md` for the full
baseline including the kernel-level 24x gap and architectural map.

**Follow-ups**: start with H1 (per-resonance scan in Reich-Moore).

## 2026-10-07 scan-based chunking prototype (not merged)

**Hypothesis**: H5 (ruled out). Attempted on branch
`perf/jax-ppxs-cum-scan` (local only, not pushed).
**Change**: `_chunked_reconstruct` helper in MLBW + Reich-Moore;
chunking gate widened from `xp.name == 'numpy'` to `in ('numpy',
'jax')`; JAX path uses `lax.scan` over main chunks + single padded
scan for remainder.
**Measured**:
  before 100 K ppxs (python-for chunking): 65 s
  after 100 K ppxs (scan chunking): 51 s
  1 M OOM: fixed (runs, slow)
  correctness vs numba: `max rel = 4e-15`
**Decision**: parked pending H1 landing. Would merge as an isolated
OOM-fix if H1 is deferred.
**Follow-ups**: add `jax.checkpoint` on the scan body before merge
(see H3) so `jax.grad` memory stays bounded. Pattern reference:
`primitives/fit_helpers.py::_chunked_chi2_jax`.

## 2026-10-08 environment re-baseline (sandbox session 2)

**Hypothesis**: n/a (baseline).
**Branch**: `main` at `333eb62`, measured from a detached worktree.
**Environment**: i7-1165G7 (4C/8T), 31 GB RAM, no swap; Python 3.11,
jax 0.6.2, numba 0.68.0 (8 threads), numpy 2.2.3. The repo's
`testenv/` venv is Python 3.10 and unusable here; rebuilt a 3.11 venv
with the same pins.
**Measured** (ppxs, s):

  | N | numpy | numba | jax 1st | jax ss |
  |---|---|---|---|---|
  | 1 K | 0.023 | 0.028 | 8.24 | 3.14 |
  | 10 K | 0.081 | 0.066 | 12.3 | 8.02 |
  | 100 K | 0.669 | 0.291 | 68.2 | 61.0 |
  | 1 M | 7.08 | 3.15 | skipped (51 GB > RAM) | - |

  RM reconstruct @10K: numba 0.059, numpy 2.22, jax 2.94.
**Decision**: within 50% of `FINDINGS.md` except the 1 M numpy/numba
cells (83.7 / 41.0 s in FINDINGS vs 7.1 / 3.2 s here). #349 explains
numba 7.5 -> 3.2 s (measured on `f8975c0`); nothing reproduces the
41 / 83.7 s figures, likely memory pressure in the original run. Use
the numbers above as the numpy/numba regression reference.

## 2026-10-08 H1 blocked resonance-axis scan in Reich-Moore (landed)

**Hypothesis**: H1 (+ H3's checkpoint, folded in).
**Branch**: `perf/jax-rm-resonance-scan` (commit `7b9c000`), from `main`.
**Change in one sentence**: on the jax backend the per-group dense
`(ne, nres)` R-matrix build is replaced by
`mf2_interpretation_reichmoore_jax.accumulate_r_matrix`, a jitted,
checkpointed `lax.scan` over *blocks* of resonances whose body is a
real `(ngroups*nch*nch, block) @ (block, ne)` matmul with a one-hot
group scatter; numpy keeps the dense build (moved verbatim into
`_r_matrix_dense` / `_group_gammas`), numba untouched.
**Design notes**:
  - Pure per-resonance scan (the H1 skeleton) was prototyped first:
    R-matrix build 0.21 s @10K, 2.96 s @100K. Blocked matmul variant:
    0.06-0.08 s @10K, 1.18 s @100K (block 256). Chose blocked.
  - Block size = `clip(512 MB // (8 ne), 1, 256)`; at 1 M that is 67
    (R-matrix build ~11 s, ~2 GB).
  - One scan serves all groups (each resonance carries its own group's
    gammas, summed from the masked per-group gammas), removing the
    `ngroups`x redundant work of the mask-based dense path.
  - Without module-level `jax.jit`, eager calls re-traced the scan each
    call and the 1 K ppxs steady state regressed (3.14 -> 3.39 s);
    jitting with static `(ngroups, block)` fixed it (-> 2.59 s).
**Measured** (ppxs on U-235, s; before = `main` re-baseline above):

  | N | jax 1st before -> after | jax ss before -> after | rel vs numba |
  |---|---|---|---|
  | 1 K | 8.24 -> 5.50 | 3.14 -> 2.59 | 2.1e-15 |
  | 10 K | 12.3 -> 7.71 | 8.02 -> 3.03 | 2.1e-15 |
  | 100 K | 68.2 -> 12.4 | 61.0 -> 7.00 | 2.9e-15 |
  | 1 M | OOM -> 58.2 | OOM -> 51.6 (peak RSS 3.6 GB) | 6.0e-15 |

  RM reconstruct only: @10K 2.94 -> 0.23 s (numba 0.072);
  @100K ~30 -> 1.81 s (numba 0.86).
  `jax.grad(sum tot)` wrt `res_er` @20K: 21.1 s / 9.14 GB peak ->
  5.69 s / 1.14 GB (identical |g|); @100K now 10.1 s / 2.19 GB.
  numpy/numba gate (`get_reaction_xs` + ppxs, 100K and 1M, median of
  5 alternating main/branch processes): all within +3.3% / -0.4%;
  main's own spread is wider than that.
**Tests**: new `tests/test_mf2_interpretation_reichmoore_jax.py`
(blocked vs numpy on 3 groups with 0/1/2 fission channels and padded
last block; no `(ne, nres)`-sized aval in the jaxpr; reverse vs
forward mode + FD through the checkpointed scan). Bug-catching check:
with the reichmoore.py change stashed, the jaxpr-size test fails and
the invariant tests pass. MISSION acceptance suite: 119 passed,
4 skipped.
**Decision**: merge candidate (M1 + the RM part of M2). The H5
Ein-chunking prototype (`perf/jax-ppxs-cum-scan`, now committed
locally as a WIP commit) is no longer needed for RM's 1 M OOM.
**Follow-ups**:
  - JAX ppxs at 1-10 K is now dominated by non-resonance per-MT eager
    overhead (~2.5-3 s floor regardless of N): H2 / M3 territory.
  - Mirror the blocked accumulator in MLBW (`mf2_interpretation_mlbw`)
    and RML; MLBW still materialises `(ne, nres)` on jax.
  - Remaining RM jax gap @10K (0.23 s vs numba 0.07 s) is mostly the
    eager post-R-matrix ops (solve, U-row); jitting `_reconstruct_group`
    per group shape is the next small win.
  - Pre-existing (same on `main`): `jax.grad` wrt a fission width that
    is exactly 0 gives NaN (`_signed_sqrt` has infinite slope at 0).
    Affects every resonance in a group without that channel. Needs a
    safe-sqrt with a finite subgradient if anyone differentiates wrt
    `res_gf1/2`.
  - CLAUDE.md's `git ls-files | xargs ruff check` passes excluded
    `raw_aux_scripts/` explicitly and flags it (also on `main`); CI's
    `ruff check .` honours the exclude and is clean.

## 2026-10-08 H1 mirrored to MLBW and RML (landed on branch)

**Hypothesis**: H1 (MLBW / RML mirror).
**Branch**: `perf/jax-mlbw-rml-resonance-scan` (commit `9309c3a`), from
`main` at `88f4a16` (#353 merged).
**Change in one sentence**: MLBW gets its own blocked-scan sibling
(`mf2_interpretation_mlbw_jax.accumulate_channel_sums`: per-L factor
tables gathered per block, five per-channel sums as one-hot matmuls);
RML (KRM=3 = Reich-Moore R-matrix) reuses
`mf2_interpretation_reichmoore_jax.accumulate_r_matrix` for all groups
in one scan. numpy paths moved verbatim into `_channel_sums_dense` /
`_group_gammas` + `_r_matrix_dense`.
**Design notes**:
  - MLBW: evaluating P_L / S_L once per distinct L on `(nL, ne)` is
    bit-identical to the dense per-resonance evaluation (`pnt_shf`
    selects by mask) and removes nres/nL redundant Newton work.
  - MLBW body: five plain `oh @ term` matmuls beat one batched
    `einsum('cb,qbn->qcn')` by ~30% (0.58 -> 0.41 s on the x15 case).
  - Tiling the energy axis too (`lax.map` over E-chunks of 2-16K
    around the resonance scan) gave no gain at 10K or 100K: the MLBW
    body is compute-bound at ~17 ns/(res, E) single-threaded, i.e.
    about numba's per-thread cost; numba wins by its 8 threads. XLA
    CPU runs the scan body's elementwise fusions on one thread. RM is
    less affected because its body is BLAS-matmul dominated.
  - MLBW nres=0 must short-circuit (empty per-L table breaks the
    gather at trace time).
**Measured** (kernel, `reconstruct` @10K, s; before = main):
  MLBW Nb-93 (203 res): jax 0.149 -> 0.068 (numba 0.006)
  MLBW Nb-93 tiled x15 (3045 res): jax 3.66 -> 0.58 (numba 0.05-0.10)
  RML Pu-239 (2085 res): jax 1.62 -> 0.14 (numba 0.03-0.07)
  RML Rh-103 (555 res): jax 0.78 -> 0.17 (numba 0.012)
  End-to-end `get_reaction_xs('(n,total)')`, jax 1st / ss:

  | file, N | main | branch |
  |---|---|---|
  | Nb-93 10K | 6.39 / 0.88 | 6.33 / 0.64 |
  | Nb-93 100K | 10.1 / 5.58 | 6.87 / 2.29 |
  | Pu-239 10K | 10.0 / 5.08 | 5.90 / 0.74 |
  | Pu-239 100K | 51.0 / 46.8 | 9.21 / 4.77 |
  | Rh-103 100K | 26.2 / 24.5 | 17.3 / 15.3 |
  | Nd-143 100K | 23.8 / 21.9 | 22.1 / 20.1 |

  (Nb-93 10K = median of 4 alternating fresh processes; the rest single
  runs.) Peak RSS of the whole e2e script 12.7 -> 5.0 GB. Relative diff
  of jax vs numpy <= 8e-15 everywhere; vs numba unchanged from main
  (numpy itself is 7.5e-11 / 1.9e-9 off numba on Pu-239 / Rh-103 RML,
  pre-existing).
  numpy/numba gate (Nb-93, Pu-239; 100K, 1M; median of 4 alternating):
  -2.6% .. +1.3%.
**Tests**: `tests/test_mf2_interpretation_mlbw_jax.py` (synthetic,
4 channels mixed L, spin-0 target, competitive threshold, padded last
block) and `tests/test_mf2_rml_jax.py` (Pu-239 corpus): jax vs numpy,
no `(ne, nres)` aval, reverse vs forward mode (+ FD). Bug-catching:
reverting each fix fails exactly its jaxpr-size test. Acceptance suite
+ composition tests: 154 passed, 2 skipped.
  Largest MLBW / RML ranges in the ENDF/B-VIII.1 neutron sublibrary
  (library scan: 135 MLBW ranges, max Np-237 761 res; 17 LRF=7, max
  Pu-239 2085, Sr-88 needs KBK=1), single runs @100K, jax
  `reconstruct` main -> branch: Np-237 8.52 -> 1.00 s, Tc-99 6.10 ->
  1.89, Zn-67 6.08 -> 1.17, Ag-109 5.09 -> 0.98, Cu-63 (RML, 6 groups)
  14.7 -> 1.34, Fe-54 (RML) 6.43 -> 0.75 (numba 0.10-0.31). e2e
  `get_reaction_xs` jax ss: Np-237 43.0 -> 19.8, Cu-63 31.2 -> 4.1,
  Fe-54 14.8 -> 2.8 s. The synthetic x15 tiling overstated the MLBW
  gap; real MLBW ranges are <= 761 resonances.
**Decision**: merge candidate.
**Follow-ups**:
  - Nd-143 / Rh-103 e2e are now dominated by the URR range (LRU=2) and
    non-resonance per-MT eager overhead; URR jax path is the next
    resonance-side target, then H2.
  - Remaining MLBW gap is XLA CPU single-threading of the scan body;
    worth a look at XLA CPU flags / sharding across host devices only
    if JAX CPU throughput matters more than GPU.
  - Same `_signed_sqrt` zero-width NaN-gradient caveat in RML
    (`res_gam` zeros for padded / non-member channels): forward-mode
    through `res_gam` is NaN on every backend.

## 2026-10-08 profile finding + jax preproc bulk gather (landed on branch)

**Hypothesis**: ad-hoc, from a cProfile of a steady-state 100K jax ppxs
call on U-235 after #353/#354.
**Branch**: `perf/jax-mf2-preproc-gather` (commit `793c8c8`), from main
`b7d1f6e`.
**Profile finding**: of 9.7 s (cProfile), `rm_data_from_endf_dict` was
4.9 s: ~19.5K eager per-resonance gathers + stacking of 0-d arrays.
U-235's URR range is LSSF=1, so URR is never reconstructed on the
standard reproducer (the URR kernel work below only matters for LSSF=0
files). `factors.pnt_shf` / `phase` were another ~3 s (all six low-L
closed forms evaluated for scalar-L calls).
**Change in one sentence**: RM / MLBW preproc gather / concatenate the
per-resonance fields once instead of indexing one resonance at a time.
**Measured**: jax build RM U-235 0.733 -> 0.005 s, MLBW Np-237 0.281 ->
0.004 s; bit-identical output (8 files x 2 backends). ppxs U-235 jax
ss, median of 3 alternating processes: 1K 3.20 -> 0.51, 10K 3.41 ->
0.89, 100K 7.99 -> 5.32, 1M 56.7 -> 53.3 s; 1st call 8.9/8.7/14.6/68.3
-> 6.6/5.9/11.4/59.4 s.
**Decision**: merge candidate (independent of the URR branch).
**Follow-ups**: scalar-L fast path in `factors.pnt_shf` / `phase`;
#350 memoisation complements this (cached reuse vs cheap first build).
**Methodology note**: single sequential main-then-branch runs on this
shared host gave numpy +50-70% swings at 100K / 1M; only alternating
runs with numpy as a control column are trustworthy.

## 2026-10-08 scalar-L fast path in mf2_interpretation_factors (landed on branch)

**Hypothesis**: ad-hoc, from the same profile (pnt_shf / phase ~3 s of 10 s).
**Branch**: `perf/jax-factors-scalar-l` (commit `0468aa7`), from main `b7d1f6e`.
**Change in one sentence**: concrete scalar L evaluates only its own
closed form (+ needed Newton steps); per-L formulas factored into
`_pnt_shf_low_L` / `_phase_low_L` and shared with the stacked forms.
**Measured**: bit-identical (300 cases). RM reconstruct @10K jax ~0.21 ->
~0.155 s. ppxs U-235 jax ss (3 alternating): 1K -1%, 10K -2%, 100K -3%,
1M -6%; 1st call -6..-9%. numpy unchanged.
**Decision**: merge candidate; small on its own (fixed preproc cost
dominates small N on main), compounds with the preproc gather PR.

## 2026-10-08 URR fluctuation integral on jax (landed on branch)

**Hypothesis**: URR follow-up from the MLBW/RML arc (Nd-143 / Rh-103 e2e
were URR-bound).
**Branch**: `perf/jax-urr-quadrature-scan` (commit `3b2bfba`), from main `b7d1f6e`.
**Variants tried** (Nd-143 synthetic widths, 12 groups, 100K energies):
  - eager (main): 11.1 s; plain jax.jit of the dense block: 3.6 s at 100K
    but 36.9 GB OOM at 1M (XLA does not fuse the (NE,nJ,32) products).
  - lax.scan over the 32 nodes, (NE,nJ) accumulators: 1.9-4.4 s, memory
    bound (~13 full arrays streamed per node; superlinear at 1M).
  - log/exp shared-base factors inside that node scan: no gain at 100K
    (XLA already CSE'd the shared log), 2x at 1M.
  - lax.map over energy chunks with original factors: 4.0-5.1 s
    (transcendental-bound); with shared-base factors: 1.0-1.3 s; chunk
    256 best of 64/256/1024/4096.
  - + integer-nu rsqrt mode: -25% and more accurate than log/exp.
  - + 'zero' mode for identically-zero width tables (241/348 library URR
    ranges have GF and GX both zero): U-235 0.83 -> 0.65 s.
  - + divisions hoisted off the node axis: 0.44 -> 0.30 s moments.
  XLA already uses ~2.6 cores on these fusions (process CPU / wall).
**Library scan**: 351 URR ranges, max nJ 12; AMUG always 0, AMUN in
{1,2}, AMUF in 0..4, AMUX mostly 0/1/2 with ~50 non-integer groups.
176 ranges LSSF=0 (reconstructed), 175 LSSF=1 (U-235 among them).
**Measured**: see commit / PR body: URR kernel 5-18x faster than main,
faster than numba on 3 of 4 files; LSSF=0 e2e ss -46..-82%.
**Decision**: merge candidate.
**Follow-ups**: remaining URR eager work (_interp_per_group: one
jnp.interp per table x group, all groups share ES in practice; tab1 /
pnt_shf / phase) ~1.9 s at 1M. Pre-existing: numpy-preproc URR data +
INT=5 cannot be jitted over E.

## 2026-10-08 MLBW traced-SPI regression fix (fix/mlbw-jax-traced-spi, 6a3b5b4)

**Found by**: composition-level jit-over-energies test for the slice PR.
#354's `_channel_sums_jax` did `np.asarray(_lx_values(..., data.spi, np))`;
`spi` is xp-native (dict-leaf autodiff) and a tracer whenever the
dataclass is built inside a trace -> `jax.jit` over `e` of
`reconstruct_resonance_xs` raised TracerArrayConversionError on every
MLBW file (OK on 88f4a16, broken on b7d1f6e). Fix: tabulate both lx
candidates, select per resonance with xp.where(spi == 0, ...). Lesson:
jax-path tests must also build the dataclass inside the traced function.

## 2026-10-08 composition slicing on jax (perf/jax-composition-slice, 1d18101 + gate)

**Profile finding** (all three earlier branches merged locally): at 1M
U-235 ppxs points, `accumulate_r_matrix` was 33 of 50 s although only
~225 energies lie in the RRR: on jax `_accumulate_range_contrib`
evaluated every range on the full mesh (to keep jit shapes fixed),
even for concrete energies.
**Change**: slice when the in-range mask is concrete and the slice skips
>= 2048 points; scatter back with `.at[idx].add`; traced energies keep
the full-mesh path.
**Gate rationale**: eager jax compiles each op per new array shape; a
1-point slice cost +0.8 s first call at 1K. Gate restores parity there.
**Measured** (3 alternating): ss 10K -12%, 100K -52%, 1M -74%; 1st
call 100K 11.9 -> 8.6 s, 1M 52.9 -> 17.6 s.
**Combined** (preproc + scalar-L + URR + slice, single run): ppxs U-235
jax ss 1K 0.45, 10K 0.47, 100K 1.25, 1M 11.5 s (main: 2.9 / 3.4 / 6.8 /
52); remaining at 1M: MF3 tab1.interp 7.2 s, MF6 yields 4.3 s.
**Env note**: my scratch venv lacked optional mpmath -> Ross-10 URR
contributed 0 and test_urr_quadrature_composition failed (also on
333eb62). Installed mpmath 1.4.1 (matches testenv).

**Next**: H2 -- single-law fast path in `tab1.interp` /
`interpolation._endf_interp1d_traced_x` (both evaluate all 5 laws).

## 2026-10-08 H2: interpolation-law specialisation (perf/jax-tab1-law-specialise)

**Hypothesis**: H2 (adapted). Instead of a jitted per-panel kernel (would
need table-size bucketing to avoid a compile per MT per mesh), evaluate
only the laws a table's concrete INT codes can select, in both
`tab1._apply_law_vectorised` and `interpolation._endf_interp1d_traced_x`.
**Measured**: bit-identical (1464 cases). U-235 ppxs jax ss -2..-8% on
main (resonance-dominated there); numpy (non-numba backend shares the
path) -27..-29% at 100K / 1M.
**Found**: `test_jit_composition::test_get_reaction_xs_under_jit_with_mlbw_composition`
already fails on main since #354 (same traced-SPI regression); it was
not in MISSION's acceptance list. Recommend adding test_jit_composition
to that list.
**Decision**: merge candidate.
**Follow-ups**: jit-fused TAB1 kernel with bucketed table sizes (next
step if MF3 stays dominant after slice + H2).

## 2026-10-08 combined effect of the six pending PRs (scratch/integration, not pushed)

main b7d1f6e + fix-spi + composition-slice + preproc-gather + scalar-L +
URR + H2. U-235 ppxs, median of 2 alternating main / combined:

  | N | jax 1st main -> combined | jax ss main -> combined | numpy (combined) |
  |---|---|---|---|
  | 1K | 6.49 -> 4.88 | 2.15 -> 0.32 | 0.026 |
  | 10K | 6.87 -> 4.38 | 2.88 -> 0.30 | 0.062 |
  | 100K | 11.95 -> 5.56 | 6.74 -> 0.59 | 0.45 |
  | 1M | 52.67 -> 10.58 | 47.68 -> 5.80 | 5.76 |

vs the session-start baseline (333eb62: 1M OOM; 100K ss 61 s): JAX ss
at 100K+ is now on par with the numpy backend; numba remains ~2x faster
at 1M. Small-N JAX is bounded by eager per-op dispatch (~0.3 s floor).

## 2026-10-08 tab1 host lookup + jitted arithmetic (perf/jax-tab1-host-lookup, stacked on H2)

**Variants tried** (95 U-235 MF3 tables): fully jitted kernel with
pow2-bucketed padded tables: 10K 0.25 -> 0.05 s but 1M only 10.6 ->
7.0 s (jnp.searchsorted on CPU = one full pass per binary-search step)
and 12 compiles; host numpy lookup + eager arithmetic: 1M 6.8 s; host
lookup + jitted (N,)-shaped arithmetic kernel: 10K 0.029 s, 1M 3.6 s,
bit-exact, one compile per (N, laws, side). Chose the last.
**Measured** (this + H2 vs main, 3 alternating): jax ss -8..-15%, 1st
call 7.5/7.0/11.9/52.8 -> 5.2/4.4/8.1/42.4 s.
**Next**: same host-lookup split for `interpolation._endf_interp1d_traced_x`
(MF6 yields, ~1.7 s at 1M in the combined profile); remaining small-N
floor is eager per-op dispatch across ~45 MTs.

## 2026-10-08 traced-x interp host lookup (perf/jax-yields-host-lookup, stacked)

Same split for `_endf_interp1d_traced_x` (MF6 yields). Bit-identical.
Stack vs main (3 alternating): jax ss -11..-15%, 1st call 7.6/7.0/11.9/52.7
-> 4.7/4.0/7.6/41.9 s. Broad suite 954 passed.

## 2026-10-09 round 2/3: bug fixes, maintenance, H7 range cache, RM kernel dead ends

**Landed on local branches (see PR_MERGE_PLAN.md round 2 / 3)**: fortran
cwd leak (821de6a), URR INT=5 jit with numpy tables (3c6d7c6), RM/RML
zero-width gradient convention (807e256; user decision: 0 for unused,
infinite for genuine in-group zeros), Ross-10 tables embedded (cb8e2d0),
U-235 ENDF/B-VIII.1 corpus resolver (dfbc4c0), MF6 LAW=1 parity test
asserts (876b4f0, needs Fortran host), H7 per-range per-call
reconstruction cache (7d66569; -38..-73 % on log meshes, all backends).

**RM accumulator variants tried at 100K / 650K RRR points (U-235), all
dead ends on this 4C/8T host** (shared with another agent; load ~10):
  - symmetric (c,c') pairs only (18 -> 12 matmul rows) + one reciprocal
    instead of two divisions: no gain (matmul is not the bottleneck).
  - pmap over 8 virtual host devices (XLA_FLAGS
    --xla_force_host_platform_device_count=8): no gain; the single-device
    path already uses ~5 cores; the 4 physical cores are the limit.
  - per-J-group scans (no one-hot waste) with dot: ~equal to current;
    with a fused broadcast-multiply-reduce instead of dot: 1.3-3.4x
    slower.
  Remaining jax/numba gap (~2x on log meshes at 1M) is mostly the
  memory traffic of materialising (block, ne) intermediates for the dot
  vs numba's register-resident per-point loop; no cheap XLA-level fix.

## 2026-10-09 numba arc: kernel vectorisation, yield interp, composition passes

**Branches** (all from main `fd9e220`, disjoint files): `perf/numba-rm-rml-kernel`
(2fabba2), `perf/numba-mlbw-kernel` (4213fb2), `perf/numba-urr-kernel` (2fa2436,
ed21b7f), `perf/numba-yield-interp` (bde748b), `perf/numba-composition-glue`
(355994a). PR bodies + PR_MERGE_PLAN.md in the repo root.
**Findings that drove it** (asm + isolated timings, U-235 RM inner loop):
  - complex accumulators never vectorise; `range(start, end)` over full arrays
    emits negative-index wraparound (blocks vectorisation); numba's default
    error model puts a zero check on each `/`.
  - per (E, res): current 3.34 ns; real/imag + slices 2.07; + no division
    check 0.54. The check can be avoided WITHOUT changing the error model by
    writing `d ** -1.0` (or guarded `1/max(d, tiny)`); a `min` reduction for
    a zero guard blocks vectorisation again, a float zero-count does not.
  - old numba RM kernel returned 0.0 (fastmath) at E == E_r with Gamma_g = 0;
    numpy returns NaN; new kernel returns NaN (no raise).
**Measured** (combined, scratch/numba-integration vs main, 3 alternating):

  | case (1M pts unless noted) | main | combined |
  |---|---|---|
  | U-235 xs log 100K | 0.87 s | 0.29 s |
  | U-235 xs log | 8.57 s | 3.14 s |
  | U-235 xs lin | 1.32 s | 1.29 s |
  | U-235 ppxs log | 7.25 s | 2.27 s |
  | U-235 ppxs lin | 4.30 s | 0.94 s |
  | Pu-239 xs log / ppxs lin | 4.61 / 4.25 s | 1.42 / 0.96 s |
  | Nb-93 / Nd-143 / Rh-103 xs log | 1.25 / 3.62 / 2.86 s | 0.67 / 1.50 / 1.45 s |
  | Cu-63 ppxs log | 2.96 s | 1.23 s |
  | numpy control U-235 xs lin 100K | 0.62 s | 0.59 s |

  Max rel change vs main <= 2.3e-11 except RM/RML `cap` in its 1 - sum|U|^2
  cancellation region (<= 2.4e-10; main itself is 1.8-3.2e-10 off numpy there).
**Dropped**: tab1 sorted-walk / panel-major (<= 1.4x on the kernel, 0.7x on
unsorted queries); tab1 with numpy error model (no gain); non-inlined R-matrix
accumulator (saves one-time compile, but ~25% slower on Cu-63).
**Not done (user decision)**: lazy jax import in array_ns (~0.2 s import, ~75 MB
RSS for numba-only users) would need JAX_ENABLE_X64 via os.environ instead of
config.update -- touches the x64-before-any-trace guarantee.
**Cold compile** (empty numba cache): RM/RML +2-3 s / +6-7 s, URR -13 s (Nd-143),
interp kernel +1.5 s.

## 2026-10-10 numba arc closed

Added to `perf/numba-composition-glue`: `c38856e` (skip the all-zero
resonance array for MTs without MF2 terms; bit-identical; U-235 xs 1M
lin 1.22-1.27x). Integration branch affected suites: 587 passed.
**Final** (main vs all six PR commits, 3 alternating rounds, warm numba
cache, first call / repeat call, s):

  | case | main | all | repeat speed-up |
  |---|---|---|---|
  | U-235 xs 10K log | 0.27 / 0.076 | 0.20 / 0.035 | 2.2x |
  | U-235 xs 1M log | 9.91 / 8.21 | 2.72 / 2.87 | 2.9x |
  | U-235 xs 1M lin | 1.39 / 1.40 | 1.00 / 1.04 | 1.35x |
  | U-235 ppxs 1M log | 7.84 / 8.46 | 2.17 / 2.13 | 4.0x |
  | U-235 ppxs 1M lin | 4.66 / 4.87 | 0.84 / 0.77 | 6.3x |
  | Pu-239 xs 1M log | 4.52 / 4.41 | 1.30 / 1.27 | 3.5x |
  | Nb-93 xs 1M log | 1.32 / 1.25 | 0.76 / 0.58 | 2.2x |
  | Nd-143 xs 1M log | 3.76 / 3.64 | 1.42 / 1.42 | 2.6x |
  | Cu-63 ppxs 1M log | 3.43 / 3.80 | 1.26 / 1.25 | 3.0x |
  | numpy U-235 xs 100K lin | 0.55 / 0.57 | 0.51 / 0.54 | 1.06x |

**Left open (numba)**: single-pass MF3 interpolation + accumulation over
all MTs in get_reaction_xs (~1.5-2x on linear meshes, structural change
to the shared composition layer); URR per-(E, group) hoisting (~1.3-1.5x
on the URR kernel); cold-compile slimming; lazy jax import (user
decision, x64 guarantee).
**Found, not fixed (jax)**: get_particle_production_xs under jax.jit
with traced energies fails in mf1_interpretation.compute_yields_from_mt456
(np.asarray on the tracer). Full-jit table: see session notes / next entry.

## 2026-10-10 full-jit investigation: fixed-mesh staging (landed on branch), RM kernel (no XLA-level win)

**Branches**: `fix/jax-jit-ppxs-traced-energies` (b0a08c4: MF1 nubar
np.asarray, MF13 NK>1 check on tracers, MF3 denominator of MF13 photon
yield on numpy -> ppxs works under jit over energies; 46/46 producing
cases of a 14-file sweep match numpy), `perf/jax-jit-fixed-mesh`
(e2dc9c3 + d88b6f0, stacked).
**Root causes of full-jit < eager** (branch probe + timings):
  - traced energies: resonance ranges evaluated on the full mesh
    (masking, not slicing); MF3 traced lookup is NOT the problem
    (U-235 1M lin, resonances off: jit 2.26 s vs eager 3.05 s; on: 20.4 s
    vs 2.1 s). Inherent; only a faster kernel helps.
  - closed-over (fixed) mesh: `xp.asarray(numpy)` inside a trace is a
    tracer -> same full-mesh path, AND XLA constant-folds everything
    derived from the constant mesh (327 s compile at 100K once the
    eager host paths were restored; ~15 GB at 1M on main).
**Fix**: stage once at the entry (optimization_barrier + host copy in
_QueryState), host-side range masks, MF3 host panel lookup with only
int32 indices embedded. jit(AP) 1M main -> branch: U-235 log 24.0 ->
13.5 s, lin 22.8 -> 2.1 s, ppxs lin 19.4 -> 1.5 s, Pu-239 14.3 -> 9.0 s,
Nd-143 10.0 -> 4.5 s; peak 9-18 GB -> 3.4-7.5 GB; first call 2-5x faster.
**Dropped**: host-gathered panel values embedded (1.18 s but 17 GB);
`jnp.searchsorted(method=scan_unrolled|compare_all)` (microbench 3-6x,
in-pipeline ~0, compile 3 -> 40 s).
**RM jax kernel (accumulate_r_matrix), U-235, 100K / 617K energies** --
none beats current by > 1.1x: E-tiling (0.82-1.09x), upper-triangle
pairs (1.04x), one reciprocal / stacked / concat / complex operand
(<= 1.03x), 12 fused jnp.sum reductions instead of the dot (5-9x slower,
15.7 GB: XLA materialises each product), block 256 at 617K (1.1x,
+1.4 GB), xla_cpu_use_thunk_runtime=false (~1.08x). Decomposition at
100K: elementwise alone 0.59 s, matmul alone 0.28 s, full body 2.05 s ->
the cost is the materialised (block, ne) dot operands (memory traffic,
~64 GB at 617K x 3194), not flops. Rest of eager reconstruct: ~1.4 s of
149 eager ops at 617K (jitting it needs hashable group statics; not done).
**Open (user decision)**: numba kernel inside jax (pure_callback +
custom_vjp) for CPU parity; GPU would suit the current dot formulation.

## 2026-10-13 H5 re-measured against main after H1 (still parked, but not redundant)

**Hypothesis**: H5, re-examined.
**Branch**: `perf/jax-ppxs-cum-scan`, local only in the `endf-userpy`
checkout, **0 commits**: the edits exist solely as working-tree
changes. The 2026-10-08 entry's parenthetical that they were "now
committed locally as a WIP commit" does not hold; no such commit exists
in any ref of either checkout.
**Context**: the 2026-10-08 entry retired H5 on the grounds that H1
fixed RM's 1 M OOM. That is correct for the OOM. This entry re-measures
against `main` `c6de145` (#388) to see whether anything else remains.
**Change**: unchanged from the 2026-10-07 prototype
(`_chunked_reconstruct`, gate widened from `xp.name == 'numpy'` to
`in ('numpy', 'jax')`, `lax.scan` over the main chunks plus one padded
scan for the remainder). It still applies to `c6de145` under a strict
`git apply`; the six commits that touched these two files since
`333eb62` (#353, #354, #355, #365, #380, #381) do not overlap it.
**Measured** (jax backend, kernel `reconstruct`; U-235 TENDL-2021 and
Nb-93 ENDF/B-VIII.1 RRR; 2 alternating clean/WIP rounds, median of 3
repeats each; peak RSS is `ru_maxrss` of the whole process; "aval" is
the largest intermediate in the jaxpr as a multiple of `ne`):

  | case | peak RSS clean -> WIP | largest aval | t clean -> WIP |
  |---|---|---|---|
  | U-235 RM 200 K | 1808 -> 549 MB | 18.0x -> 1.0x ne | 5.13 -> 7.32 s (+42%) |
  | U-235 RM 1 M | 2796 -> 797 MB | 18.0x -> 1.0x ne | 27.9 -> 30.4 s (+8.7%) |
  | Nb-93 MLBW 200 K | 4709 -> 3566 MB | 48.0x -> 39.7x ne | 2.28 -> 5.33 s (+130%) |
  | Nb-93 MLBW 1 M | 6424 -> 4065 MB | 48.0x -> 7.93x ne | 10.4 -> 17.5 s (+67%) |
  | grad wrt `res_er` 20 K | 855 -> 717 MB | n/a | 10.5 -> 15.1 s (+44%) |
  | grad wrt `res_er` 100 K | 2040 -> 916 MB | n/a | 17.0 -> 23.1 s (+35%) |

  Gradients agree to all printed digits (`|g|_1` = 1.998223e+06 on both
  sides at 100 K) and are finite on both. Forward values agree with the
  unchunked jax and the numpy reference at 3.8e-16 to 3.5e-14 relative,
  over `ne` in {4096, 4097, 1000, 999} on MLBW Nb-93 and RM U-235 /
  Nd-143; the odd sizes exercise the padded tail scan.

**Finding**: H1 did remove the `(ne, nres)` term (at 200 K on U-235 it
would be 6.4e8 elements; the largest aval is 3.6e6). What it left is
O(ne) intermediates with constant factors of 18 (RM) and 48 (MLBW), and
H5 bounds those to O(chunk). So H5 is redundant for the **OOM** only,
not for memory in general. The trade is strongly size- and
formalism-dependent: 3.5x memory for +8.7% time on RM at 1 M is a good
trade, 1.6x for +67% on MLBW is not, and at 200 K neither is, because
the chunking overhead has not amortised.

**Caveat on the trigger**: `_numpy_chunk_size(ne, nres, cap)` still
derives the chunk from the dense `(ne, nres)` intermediate, which no
longer exists on the jax backend, so it fires on a proxy rather than on
the real O(ne) footprint: 95 chunks at 1 M on U-235 RM, 10 at 100 K, 63
on Pu-239 RML at 1 M, 23 on Np-237 MLBW at 1 M. The table above was
measured with that mis-calibrated trigger.

**Decision**: parked, unchanged. Not merged because the trigger is
calibrated to an array that no longer exists, MLBW pays 67 to 130 % for
little, and no measured case came near memory exhaustion (2.8 GB and
6.4 GB at 1 M against 25 GiB free on the 31 GB host). Revisit if memory
becomes the binding constraint: a routine reconstructing several
reactions on one common grid, meshes well beyond 1 M points, or GPU
VRAM rather than host RAM.

**Follow-ups if it is ever taken up**:
  - Re-derive the chunk size from the actual O(ne) footprint (the 18x /
    48x factors above) against a real memory budget, so it stays off at
    200 K and fires at 1 M.
  - Restrict it to RM / RML, or give MLBW a much larger chunk.
  - `jax.checkpoint` on the scan body (the 2026-10-07 follow-up) is
    still absent. It turns out not to be needed for grad memory, which
    H5 reduces anyway because the inner kernels are already
    checkpointed; add it for consistency, not for a measured problem.
  - `import jax` inside `_chunked_reconstruct` is unused (ruff F401) and
    would fail CI as it stands.
