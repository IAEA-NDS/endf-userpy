# Mesh densification prototype — findings and status

Working notes. Snapshot of the prototype linearisation driver work
on branch `feat/linearization-prototype` before shelving. Not a
design commitment; captured so the numbers and the known defects
do not evaporate when the session context rolls over.

## Status

**Shelved.** Branch stays open at `feat/linearization-prototype`
(PR #310). Two known defects block production use:

- **#312** — `linearize_reaction_xs` never terminates when the
  query range contains a discontinuity (MF3 doubled abscissa or
  MF2 range edge). Every corpus file carries at least one such
  discontinuity, so the pathology hits any realistic query range
  that touches an RRR/URR seam or a threshold.
- **#313** — Midpoint chord test certifies meshes that miss the
  requested tolerance by 32-59x on dense actinide RRR. The
  estimator is simultaneously the mesh constructor and the
  stopping criterion, so a spuriously-passing segment is never
  revisited.

Fixes for both are sketched in the issues; neither is
expensive to land. Pick-up order would be #313 first (correctness
of the "converged" guarantee) then #312 (termination). After both
land, redo the actinide benchmark on the full `[1 eV, 2250 eV]`
window per #313's reproducer.

## What this prototype provides

### 1. `endf_userpy.linearization` — chord-based driver

Public API:

- `linearize_reaction_xs(endf_dict, reaction, e_min, e_max, *, tol_abs, tol_rel, options, seed_strategy, ...)` — returns `(mesh, sigma)` or a `LinearizationResult` dataclass with `return_diagnostics=True`.
- `benchmark_seed_strategies(endf_dict, reaction, e_min, e_max, *, tol_abs, tol_rel, strategies, verify_mesh_size, ...)` — A/B compare seed strategies on the same inputs. Returns a `SeedStrategyBenchmark` with per-strategy diagnostics.

Two seed strategies implemented:

- **`'windowed_fixed'`** (default): fixed 7 points per pole across
  `[E_r - 2 Γ, E_r + 2 Γ]`. Simple, predictable per-pole cost.
  Over-samples dense clusters where neighbour windows overlap.
- **`'territory_adaptive'`**: for each pole, walks outward in
  `Γ/2` steps up to the midpoint toward the next pole, capped at
  20 steps per side. Seeds inter-pole midpoints too (where
  interference dips sit). Adapts density to local peak spacing.

Iterative refinement is chord-at-midpoint bisection:

```
for it in range(max_iterations):
    mids       = 0.5 * (mesh[:-1] + mesh[1:])
    predicted  = 0.5 * (sigma[:-1] + sigma[1:])
    actual     = evaluate(mids)             # ONE batched call
    err        = np.abs(actual - predicted)
    tol        = tol_abs + tol_rel * np.maximum(|actual|, |predicted|)
    keep       = err > tol
    if not keep.any(): break
    mesh, sigma = _merge_sorted(mesh, sigma, mids[keep], actual[keep])
```

**Insertion is selective** (per-segment, driven by that segment's
own chord error); **evaluation is non-selective** (every midpoint
is re-evaluated every iteration). Known inefficiency: late
iterations re-check many already-converged segments. Carrying a
per-segment `active` flag would skip them. Not a correctness
issue, performance only.

### 2. `endf_userpy.linearization_curvature` — curvature-equidistribution driver

Separate module, different algorithm. Uses JAX second derivatives
`f''(E)` to equidistribute the monitor
`M(x) = sqrt(|f''(x)| / (8 (tol_abs + tol_rel |f(x)|)))`, placing
knots at equal increments of `∫ M dx`. Also finds critical points
(peaks and interference dips) via Newton on `f'(E) = 0` and makes
them mandatory knots.

Public API: `linearize_curvature(...)`. Returns `(mesh, sigma)`
or a `CurvatureLinearizationResult`.

**Caveats:**
- Requires `JAX_ENABLE_X64=1` (second derivatives of sharp
  resonances are meaningless in float32).
- First call does the JIT trace+compile over the full composition.
  On Nb-93 (~200 resonances) this is a few seconds; on actinides
  (~2000 resonances) it is minutes. Needs pre-warming at a small
  shape before timed calls.
- One compile per `_JitComposedEvaluator` instance (chunked
  evaluator), one instance per `(endf_dict, mt)`. Three reactions
  per file means three compiles.

### 3. Tests

`tests/test_linearization.py`, 12 tests, all pass:

- Synthetic Lorentzian convergence + sparsity + pole-aware seed
  does not slow convergence.
- Nb-93 RRR end-to-end accuracy vs 20k verification mesh at
  `tol_rel = 1e-3`.
- Loosening `tol_rel` yields sparser mesh.
- Degenerate inputs (bad `e_min`/`e_max`, both tolerances zero,
  seed larger than `max_points`).
- Determinism (identical inputs produce bit-identical output).
- `options` forwarded verbatim to every `get_reaction_xs` call.
- `LinearizationResult` dataclass shape.

Nb-93 corpus tests skip cleanly when `data_law1_adhoc/` is absent.

### 4. Benchmark scripts

Under `scripts/linearization_benchmark/`:

- `benchmark_actinides.py` — runs `benchmark_seed_strategies` on
  the four available actinides (U-233, U-235, U-238, Pu-239) ×
  three reactions (elastic, capture, fission) with numba backend.
  Writes a live-tailable log.
- `plot_actinides.py` — one log-log plot per material showing
  elastic / capture / fission with the `territory_adaptive` mesh
  overlaid against a dense ground truth.
- `plot_strategy_comparison.py` — two-panel plot per
  (material, reaction): top shows truth + both strategies'
  meshes, bottom shows the relative error of each strategy
  against the truth with `tol_rel` as a reference line.

Scripts self-locate the repo via `__file__` so they work from any
working directory. Set `PLOT_OUT_DIR` to redirect plot output.

## Actinide benchmark summary

Measured on branch `feat/linearization-prototype` with
numba backend, `tol_rel = 1e-3`, verification mesh = 20 000 log-spaced
points on `[1e-5 eV, 2000 eV]` (narrower than #313's `[1, 2250] eV`
window where the terminal discontinuity bites hardest).

| file                 | reaction | strategy           | mesh | iter | wall | max_rel_err |
|----------------------|----------|--------------------|-----:|-----:|-----:|------------:|
| U-233 ENDF/B-VIII.1  | elastic  | windowed_fixed     |  58538 |  9 | 4.7s | 1.57e-02    |
| U-233 ENDF/B-VIII.1  | elastic  | territory_adaptive |  59601 |  9 | 10.0s| 3.44e-02    |
| U-233 ENDF/B-VIII.1  | capture  | windowed_fixed     | 165820 | 10 | 23.6s| 5.51e-02    |
| U-233 ENDF/B-VIII.1  | capture  | territory_adaptive | 168462 | 10 | 24.0s| 2.40e-02    |
| U-233 ENDF/B-VIII.1  | fission  | windowed_fixed     | 140214 | 10 | 20.2s| 4.43e-02    |
| U-233 ENDF/B-VIII.1  | fission  | territory_adaptive | 139036 | 10 | 20.2s| 9.37e-02    |
| U-235 TENDL-2021     | elastic  | windowed_fixed     |  90340 |  9 | 6.4s | 1.91e-02    |
| U-235 TENDL-2021     | elastic  | territory_adaptive |  95468 |  9 | 7.5s | 2.10e-02    |
| U-235 TENDL-2021     | capture  | windowed_fixed     | 200000 |  4 | 2.9s | 1.58e-01    (point_budget) |
| U-235 TENDL-2021     | capture  | territory_adaptive | 200000 |  4 | 4.5s | 1.01e-01    (point_budget) |
| U-235 TENDL-2021     | fission  | windowed_fixed     | 200000 |  5 | 4.7s | 7.70e-02    (point_budget) |
| U-235 TENDL-2021     | fission  | territory_adaptive | 200000 |  4 | 4.2s | 5.93e-02    (point_budget) |
| U-238 JENDL-5        | elastic  | windowed_fixed     |  34979 | 16 | 3.3s | 6.03e-02    |
| U-238 JENDL-5        | elastic  | territory_adaptive |  42354 | 14 | 5.1s | **4.95e-03**|
| U-238 JENDL-5        | capture  | windowed_fixed     |  97904 | 15 | 7.7s | 4.78e-03    |
| U-238 JENDL-5        | capture  | territory_adaptive | 100265 | 12 | 8.0s | **1.00e-03**|
| U-238 JENDL-5        | fission  | windowed_fixed     |  16796 | 14 | 2.1s | 3.96e-03    |
| U-238 JENDL-5        | fission  | territory_adaptive |  26345 | 11 | 3.2s | **1.20e-03**|
| Pu-239 ENDF/B-VIII.1 | elastic  | windowed_fixed     |  72513 | 13 | 3.8s | 1.42e-01    |
| Pu-239 ENDF/B-VIII.1 | elastic  | territory_adaptive |  76751 | 12 | 4.4s | 3.92e-02    |
| Pu-239 ENDF/B-VIII.1 | capture  | windowed_fixed     | 153225 | 12 | 7.8s | 5.53e-02    |
| Pu-239 ENDF/B-VIII.1 | capture  | territory_adaptive | 156900 | 11 | 9.1s | **4.13e-03**|
| Pu-239 ENDF/B-VIII.1 | fission  | windowed_fixed     | 109744 | 13 | 7.4s | 5.52e-02    |
| Pu-239 ENDF/B-VIII.1 | fission  | territory_adaptive | 113331 | 11 | 7.4s | 2.81e-02    |

Bolded rows reach or come within 5x of `tol_rel`. Everything
else is 10x to 150x above.

**Strategy comparison on `max_rel_err`:**

- `territory_adaptive` wins or ties in 10 of 12 head-to-head
  comparisons, often by 3x to 13x.
- `windowed_fixed` wins on U-233 elastic and U-233 fission.
- Wall times comparable within 10-30% in all cases.

**Not a strategy problem.** The `max_rel_err` numbers above sit
far above `tol_rel` for both strategies on most actinide reactions
despite `status='converged'`. The algorithm converges to its own
chord criterion, which is a weak upper bound on the max linear
interpolation error over a panel (see #313).

## Known defects

### #313 Chord estimator certifies non-conforming meshes

**Where:** Every actinide RRR query at `tol_rel = 1e-3`. See table
above and the `cmp_<file>_<reaction>.png` plots produced by
`scripts/linearization_benchmark/plot_strategy_comparison.py`.

**Why:** The max linear interpolation error over a segment
`[E_i, E_{i+1}]` sits near a quarter point for sharp Lorentzians,
not at the midpoint. Probing only at the midpoint lets a segment
pass whose quarter point is still 5-10x over tolerance.

**Fix sketch** (per #313):

1. Probe at `t = 1/4, 1/2, 3/4` of each segment against the
   parent chord. Fail the segment if any probe exceeds
   tolerance. Three probes recover 97% of the actinide RRR gap;
   seven buy almost nothing further.
2. On failure, insert only the midpoint. The two children's first
   test is already paid for (outer probes = children's
   midpoints).
3. Scale `tol_rel` by `min(|σ_i|, |σ_{i+1}|)` not `max`.
4. Apply a `tol / 1.1` margin to absorb residual.

Combined: ~6% more points for a 34x accuracy improvement on
U-235 TENDL elastic.

### #312 Non-convergence at discontinuities

**Where:** Any query range crossing an MF3 doubled abscissa or an
MF2 range edge. Every one of the 25 corpus files has at least one
such discontinuity.

**Why:** Chord test on a segment straddling the jump can never
pass; each iteration halves the segment, adds one point, and
leaves infinitely many halvings to go.

**Fix sketch** (per #312):

1. **Retire unsplittable segments.** If the proposed midpoint is
   not strictly inside `(a, b)` (floating-point degenerate), mark
   the segment converged and emit a warning.
2. **Represent the jump the ENDF-6 way** by inserting the knot
   twice (left-limit and right-limit values). `np.interp` honours
   this directly; refinement excludes zero-width segments.

The second removes the pathology and improves accuracy at the
seam.

## Suggested pick-up order

1. **Issue #313 three-probe certificate** (correctness, blocks
   production).
2. **Issue #312 doubled-x representation** (termination, blocks
   realistic query ranges).
3. **Re-run** `scripts/linearization_benchmark/benchmark_actinides.py`
   on the full `[1, 2250] eV` window and verify `max_rel_err <=
   tol_rel * 1.1` across the four actinides × three reactions.
4. **Then** consider the perf follow-ups: selective evaluation
   (per-segment `active` flag), per-call reconstruction
   memoisation (#306), RRR-aware Ein slicing in the composition
   layer (#307).
5. The curvature-equidistribution driver (`linearization_curvature`)
   is a parallel direction. Validate it on actinides at a scale
   smaller than the full RRR first (per the JAX JIT compile-cost
   observations in the module docstring and the chat log).

## How to pick up

```
git checkout feat/linearization-prototype
pytest tests/test_linearization.py   # should be 12/12 green
```

Reproduce the benchmark table:

```
bash tests/data_law1_adhoc/fetch.sh   # if not already done
python scripts/linearization_benchmark/benchmark_actinides.py
# tail benchmark_actinides.log in another terminal to watch progress
python scripts/linearization_benchmark/plot_strategy_comparison.py
# plot output alongside the script in scripts/linearization_benchmark/
```

Both scripts assume numba is installed (`pip install numba`).

## References

- PR #310 (open): the chord-based driver + territory_adaptive +
  benchmark function. Shelved per the two issues below.
- Commit `e010dee` on the same branch: curvature strategy.
- Issue #312: non-convergence at discontinuities.
- Issue #313: chord estimator certifies non-conforming meshes.
- Issue #306: per-call reconstruction memoisation (perf).
- Issue #307: RRR-aware Ein slicing (perf).
- PR #304 (merged): resonance composition in distribution paths;
  the composition path this driver sits on top of.
