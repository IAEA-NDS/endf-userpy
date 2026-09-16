# Contributing to endf-userpy

Rules for adding tests to this repo, distilled from the bug-fix
pull-request pattern that settled during issues #29–#65. Follow
these when adding new tests or extending existing ones.

## Testing conventions

1. **"Does not raise" is not a test.** For any function that
   returns a numeric result, assert on the numeric result — a peak
   position, an integral, a ratio, an equality to a reference —
   not just that the call completed without exception.
2. **Any normalized distribution integrates to 1.** For MF4, MF5
   analytic, MF6, MF14, MF15, add a test that evaluates the
   spectrum on a fine grid at several representative incident
   energies and checks
   `abs(trapezoid(f, eouts) - 1.0) < 1e-3` (import `trapezoid` from
   `endf_userpy.primitives.np_compat` so tests keep running on numpy
   1.x, where `np.trapezoid` is spelled `np.trapz`). Loosen the tolerance
   only when the file's own tabulation forces coarser trapezoid
   error (`< 1e-2` is a reasonable ceiling; wider means the test
   is not really pinning normalization).
3. **New MF/LAW support ships with a data file that actually
   contains that representation.** Two acceptable homes:
   - **Small ENDF excerpts** committed under `tests/data/`. Single-MT
     preferred (the JEFF-3.3 MF4 excerpts already in that
     directory set the pattern).
   - **Larger real files** added to `tests/data_law1_adhoc/fetch.sh`
     with SHA-256 pinning. These are fetched on demand and tests
     that depend on them skip cleanly when the corpus is not
     present, so `pytest tests/` still runs from a fresh checkout.
4. **Physics predicates in `primitives/properties.py` and
   `quantities_mt_zap/selectors.py`** get a unit test on a
   **synthetic dict** — not a real ENDF file. A two-line
   synthetic-dict test catches things like the `has_mf13_mt` typo
   (issue #44) that a corpus-based test would miss because the
   corpus happens not to have MF13.
5. **Bug-catching verification before pushing.** After writing
   tests for a fix, temporarily revert the fix
   (`git stash push -- <file>`) and re-run the new tests. Every
   test that pins the actual bug should fail; tests that pin
   invariants should pass regardless. Restore with `git stash pop`.
   This confirms the tests catch what they claim.
6. **User-visible entry points in `endf_userpy.quantities`** get
   end-to-end tests on real corpus files with concrete expected
   numerics tied to physics: peak positions matching tabulated
   Egs, cumulative-XS matching MF3/MT1, isotropic distributions
   matching xs/(4π), etc. Corpus files that stress a particular
   code path are added to `tests/data_law1_adhoc/` with a README
   entry describing which layout aspect they exercise.

## Pre-push checks

Run before every `git push`:

```
source testenv/bin/activate    # (or your venv equivalent)
pytest tests/                  # full suite
git ls-files '*.py' | xargs ruff check --output-format=github
```

The ruff invocation restricts to tracked files (matching what CI
sees). Running `ruff check .` also flags untracked scripts in
`examples/` and `endf_userpy/translation.py` that CI never sees
and that will look like failures locally when they aren't.

The `pyproject.toml` ruff config uses `select = ["F"]`
(pyflakes-family only): undefined names, unused imports,
unused variables, duplicate definitions, placeholder-less
f-strings. It runs on every PR via `.github/workflows/lint.yml`.
Style rules are deliberately not enabled.

## Coverage

Representations exercised by the current test corpus (as of the
merge that added this file):

| Representation | Corpus coverage | Notes |
| --- | --- | --- |
| MF1 LNU=1 polynomial nubar | none | synthetic-dict tests only (issue #42) |
| MF1 LNU=2 tabulated nubar  | `tendl21_n_U-235.endf` MT 452/455/456 | ad-hoc corpus |
| MF3 cross section | H-1, H-2, Be-9, Al-27, all corpus files | broad |
| MF4 angular distribution | H-1, H-2, Be-9, JEFF-3.3 excerpts | broad; Python/Fortran equivalence pinned |
| MF5 LF=1 tabulated | `tendl21_n_U-235.endf` MT 18 | ad-hoc corpus |
| MF5 LF=5 general evaporation | `tendl21_n_U-235.endf` MT 455 | ad-hoc corpus |
| MF5 LF=7 simple Maxwellian | none | synthetic-dict tests only (issue #41) |
| MF5 LF=9 evaporation | none | synthetic-dict tests only (issue #41) |
| MF5 LF=11/12 | not applicable | not implemented |
| MF6 LAW=1 | Be-9, corpus | broad |
| MF6 LAW=2 | H-1, H-2 | broad |
| MF6 LAW=3/4/5 | not applicable | not implemented |
| MF6 LAW=6 | H-2 MT 16 | present but no dedicated test |
| MF6 LAW=7 | Be-9 MT 16 | present |
| MF12 LO=1 tabulated yields | Be-9, Al-27, all corpus | broad |
| MF12 LO=2 transition probs | Cu-63, Fe-56, U-235 | ad-hoc corpus |
| MF14 LI=0 LTT=1 Legendre | none | untested |
| MF14 LI=0 LTT=2 tabulated | not applicable | not implemented |
| MF14 LI=1 isotropic | Cu-63, Fe-56, corpus | broad |
| MF15 continuous photon spectrum | Al-27 MT 102, U-235 MT 18 | ad-hoc corpus |

Rows without an implementation are documented in `README.md`
under "Known limitations".

## Commit messages and PR bodies

- **Do not add attribution trailers** (`Co-Authored-By: Claude`,
  `Generated with Claude Code`, session URLs). Commits should be
  attributed to the author only.
- **Avoid the U+2014 em-dash character** in commit messages,
  issue text, and PR descriptions. Use `--` or `,` instead.
