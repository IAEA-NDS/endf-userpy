# Mission

Close as much of the throughput + memory gap between the **JAX
backend** and the **numba backend** on `endf_userpy`'s resonance
reconstruction + MF3 interpolation hot paths, while preserving the
JAX backend's two reasons for existing:

- `jax.jit`-safety of every public reconstructor.
- `jax.grad` of every file-side leaf (resonance parameters, radii,
  TAB1 y-values, etc.) end-to-end.

## In scope

- `endf_userpy.mfsec_interpretation.mf2_interpretation_reichmoore` -
  Reich-Moore (LRU=1 LRF=3). This is the dominant cost on U-235 and
  the headline target.
- `endf_userpy.mfsec_interpretation.mf2_interpretation_mlbw` - MLBW
  (LRU=1 LRF=2).
- `endf_userpy.mfsec_interpretation.mf2_interpretation_rml` - RML
  (LRU=1 LRF=7 KRM=3).
- `endf_userpy.mfsec_interpretation.mf2_interpretation_urr` - URR
  (LRU=2 Case C).
- `endf_userpy.primitives.tab1.interp` and
  `endf_userpy.primitives.tab1._apply_law_vectorised` - MF3
  interpolation. The numba fast path
  (`endf_userpy.primitives.tab1_numba`) is the model to beat.
- `endf_userpy.quantities_mt_zap.resonance_composition` - composition
  wrapper that already uses slice-and-scatter on numpy and may need
  mirrored thought on JAX.

## Out of scope (do not change)

- Public API in `endf_userpy.quantities`. Signatures, kwarg names,
  default values, return shapes stay fixed.
- The numpy and numba code paths. They may be refactored only to the
  extent that an unavoidable shared helper changes, and even then no
  measurable numpy / numba regression is tolerable.
- The backend adapter interface in `endf_userpy.primitives.array_ns`
  (`xp.scan`, `xp.map`, `wants_accelerator`, `accelerator_available`,
  `name` etc.). Fine to add methods; do not remove or rename.
- Resonance physics. Any output change at machine precision vs. numba
  is a regression and must be investigated.

## Architectural respect (hard constraint)

New code goes in its correct architectural home, matching the
established layer convention and the local module's naming.
Repo-wide invariants (summarised from the root `CLAUDE.md`):

- **`primitives/`** - pure helpers, no MF / MT logic. Scalar
  physical constants, backend adapters, interpolation
  primitives, static-dict wrappers. If your helper needs to read
  `endf_dict[MF][MT][...]`, it does **not** belong here.
- **`mfsec_interpretation/`** - one module per ENDF MF section.
  Pattern: `mf<N>_interpretation.py` for the array-agnostic
  reconstruction, `mf<N>_interpretation_<name>_numba.py` for the
  hand-fused numba sibling, `mf<N>_interpretation_<name>_preproc.py`
  for the ENDF-dict-to-dataclass preproc. A new JAX sibling, if
  warranted, follows the same shape: `mf<N>_interpretation_<name>_jax.py`.
- **`quantities_mt_zap/`** - physics-level combinators that
  take `(endf_dict, mt, zap, ...)` and stitch across MF sections.
  `selectors.py` is for admission predicates `(endf_dict, mt, ...) -> bool`.
  `resonance_composition.py` composes MF2 + MF3.
- **`quantities.py`** - top-level user-facing API only.

**Before placing a new function, run this checklist:**

1. Does an existing module already do the same shape of thing (same
   argument signature, same role in the pipeline)? If yes, put the
   new function next to it in that module, matching its naming.
2. What layer is it? Pure numeric with no MF/MT logic -> `primitives`.
   One-section marshaling / reconstruction -> `mfsec_interpretation`.
   `(endf_dict, mt, ...) -> bool` admission predicate -> `selectors`.
   Multi-MF stitching that returns an array of cross sections ->
   `quantities_mt_zap`. User-facing entry -> `quantities`.
3. Is the name consistent with its neighbours? Public functions in
   `selectors.py` do not carry a leading underscore. Preproc entries
   are named `<name>_data_from_endf_dict`. Numba kernels live under
   `_<name>_kernel` in a `_numba.py` sibling.
4. Does the signature match the neighbours' style? Compare to the
   nearest existing function in the chosen module. For example,
   `selectors.contains_residual_za_and_lfs(endf_dict, mt, residual_za, lfs)`
   resolves the projectile internally from `endf_dict`; a new
   selector should do the same, not take `projectile` as a parameter.

If any of these checks is unclear, prefer re-reading the root
`CLAUDE.md` for the architecture description and the target module
for its naming conventions before writing. Mis-placed code is a
regression to be corrected before merge, not a style preference.

**Concrete example from the recent `include_target` arc**: the
helper `is_target_conserving_mt` was first drafted in
`quantities.py` with a leading underscore and `projectile` /
`target_za` as explicit parameters. Correct placement, after
applying the checklist, was `selectors.py` with no underscore,
`(endf_dict, mt) -> bool` signature, projectile and target_za
resolved internally from `endf_dict` to match
`contains_residual_za_and_lfs`. The function body did not change;
only its home, name, and signature. This is the pattern to follow,
not the exception.

## Acceptance criteria for a merge

A change is ready to propose for merge when **all** are true:

1. **Correctness**: on the standard reproducer (see `MEASURE.md`), the
   JAX output's max relative difference vs. numba is `<= 1e-10`.
2. **No numpy / numba regression**: wall time on
   `get_reaction_xs(d, '(n,total)', e)` with
   `RunOptions(backend='numba')` and `backend='numpy'` is within +/- 5
   percent of pre-change baseline on the standard reproducer.
3. **Autodiff still works**: the following test files all pass:
   - `tests/test_mf2_interpretation_mlbw.py`
   - `tests/test_mf2_interpretation_reichmoore.py`
   - `tests/test_mf2_resonance_autodiff_wrt_E.py`
   - `tests/test_mf2_rml_autodiff.py`
   - `tests/test_mf2_urr_autodiff.py`
   - `tests/test_tab1_numba.py`
   - `tests/test_tab1_interp_discontinuity.py`
   - `tests/test_jit_composition.py` (end-to-end `jax.jit` of the
     composition pipeline; catches concrete-only checks that slip
     into a jax path, e.g. the #354 traced-SPI regression fixed in
     #355)
   - the jax-path suites added in this arc:
     `tests/test_mf2_interpretation_{reichmoore,mlbw,urr}_jax.py`,
     `tests/test_mf2_rml_jax.py`, `tests/test_tab1_jax.py`,
     `tests/test_tab1_law_dispatch.py`,
     `tests/test_resonance_composition.py`
4. **Memory profile**: a 1 M-point JAX call on the standard reproducer
   does not OOM (currently it does without `scan`-based chunking;
   see `FINDINGS.md`).
5. **Baseline direction**: measured JAX wall time strictly less than
   the pre-change JAX baseline on the standard reproducer, for both
   first-call and steady-state at every size in the measurement
   table.
6. **Change is clean**: ruff gate passes (`git ls-files '*.py' | xargs
   ruff check --output-format=github`), a dedicated regression test
   pins the behaviour and the fix is bug-catching-verified
   (temporarily revert the fix, confirm the new test fails,
   restore).

## Milestones

Three staged targets. Each is a merge candidate on its own.

- **M1 (table stakes)**: fix the 1 M-point OOM via chunking. Already
  prototyped on branch `perf/jax-ppxs-cum-scan` with scan-based
  chunking. See `FINDINGS.md` for the branch state. Decide whether to
  merge the prototype as-is, extend with `jax.checkpoint`, or
  reinvent.
- **M2 (kernel body)**: close the ~24x per-chunk kernel gap vs. numba
  (see `FINDINGS.md`). The candidate is a `lax.scan` over the
  resonance axis inside the Reich-Moore (and MLBW) reconstruct body,
  accumulating into an `(ne,)` buffer, replicating the numba kernel's
  memory profile without giving up jit-safety.
- **M3 (MF3 interp)**: write a JAX-friendly `tab1.interp` fast path
  analogous to `tab1_numba._interp_full_numba` (per-point dispatch
  skipping dead branches), to recover some of the per-op wasted work
  for every MT.

M2 is the big-win target. M1 is strictly correctness + memory. M3 is
a smaller multiplier but affects every entry point, not just
resonance.
