# MF6/LAW=1 ND>0 broadening ad-hoc corpus

Candidate ENDF files for widening the coverage of the LAW=1 ND>0
discrete-line broadening (issue #27). Not part of the automated
`pytest tests/` run: the paired script `tests/adhoc_test_law1_
discrete_broadening.py` is named to be excluded from default pytest
collection and lists these files explicitly.

## Fetching the data files

The files are **not committed to the repo** (5 files, ~13 MB total).
Run the fetch script once to populate this directory from
nds.iaea.org:

    bash tests/data_law1_adhoc/fetch.sh

The script downloads each `.zip`, unzips it, renames the payload to
`<library>_n_<isotope>.endf` under this directory, and verifies the
SHA-256 checksum. It's idempotent: files already present with the
expected checksum are skipped.

Requires `bash`, `curl`, `unzip`, `sha256sum`. If SHA-256 checks fail
after a fetch, the upstream file has changed since the fingerprints
were recorded; update the manifest in `fetch.sh` after reviewing what
changed.

## Files

| file | library | origin | LAW=1 ND>0 subsecs | notes |
|---|---|---|---|---|
| `endfb81_n_Be-9.endf` | ENDF-B-VIII.1 | ENDF-B-VIII.1 `n_004-Be-9_0425.zip` | 1 | Baseline: MT 701 subsec 3 is a single 477 keV gamma line (LAW=1 LANG=1 LCT=2 ND=1 NEP=1 NA=0). Same shape as the file in `tests/data/n-004_Be_009.endf` that pins the DDX and 1D real-data tests in the main suite. |
| `endfb81_n_B-11.endf` | ENDF-B-VIII.1 | `n_005-B-11_0528.zip` | 3 | Three MTs with LAW=1 ND>0 gamma content: (n,nα) 22, (n,p) 103, (n,α) 107. Small file; a good "second look" beyond Be-9. |
| `endfb81_n_Al-27.endf` | ENDF-B-VIII.1 | `n_013-Al-27_1325.zip` | 106 | Broadest MT coverage: 51..91 (inelastic level cascades), 601..649 (proton), 701..759 (deuteron), 801..819 (alpha). All ZAP=0 gamma, LANG=1, LCT=2, NA=0, pure discrete (no continuum in the same subsec). 2 panels each. ND ranges from 1 to 41 across MTs. Stress-tests the aggregator + folder over a large MT set in a single file. |
| `tendl21_n_Fe-56.endf` | TENDL-2021 | `n_026-Fe-56_2631.zip` | 26 | Mid-mass MIXED subsecs (both ND>0 AND continuum in the same LAW=1 subsection). LCT=3 (the light-ejectile branch of the frame conversion, which for gammas degenerates to identity). Panel count reaches 42 for MT 16, so panel interpolation is exercised. ND up to 87 lines per subsec. Different evaluator style from ENDF-B: TALYS-based. |
| `tendl21_n_U-235.endf` | TENDL-2021 | `n_092-U-235_9228.zip` | 12 | Actinide MIXED subsecs. LCT=2 and LCT=3 both appear across subsections in the same file. |

## Provenance

Each file was downloaded directly from
`https://nds.iaea.org/public/download-endf/<library>/n/<name>.zip`
and unzipped in place. Only the `.dat` payload was renamed to `.endf`
with a `<library>_n_<isotope>` prefix so the ad-hoc script can find
them via a `glob('*.endf')`.

## Selection rationale

Coverage axes the corpus intentionally exposes:

- **Library**: ENDF-B-VIII.1, TENDL-2021 (JEFF-3.3 is represented in
  the main-suite Be-9 file).
- **Mass range**: light (Be-9, B-11), light-mid (Al-27), mid (Fe-56),
  actinide (U-235).
- **LCT branch**: 2 (all files), 3 (Fe-56, U-235).
- **Pure vs mixed subsecs**: pure (Be-9, B-11, Al-27), mixed
  (Fe-56, U-235). Mixed subsecs stress the interaction between the
  continuous and LAW=1 discrete folders since MF6 attributes their
  yield jointly (compute_yields does not partition ND vs continuum
  for LAW=1) and only the b(k) values inside the discrete-line
  amplitudes split the two contributions.
- **MT coverage**: 22, 51..91, 102..107, 601..819 in various
  combinations across the corpus.
- **ND count and panel count**: 1..87 lines, 2..42 panels.

Coverage axes NOT exposed by this corpus (no known real file with
these traits was found in the surveyed libraries):

- **LANG=2** (Kalbach-Mann) with ND>0. Kalbach-Mann is designed for
  continuum spectra; discrete lines with Kalbach-Mann angular
  parameters do not appear to occur in practice.
- **LANG=11..15** (tabulated angular distribution) with ND>0.
- **NA>0** (non-isotropic gamma angdist) with ND>0.
- **AWP > 0** heavy-ejectile discrete lines. In the surveyed files
  every LAW=1 ND>0 subsection carries ZAP=0 gamma with AWP=0, which
  degenerates the `mf6cm2lab_disc` frame conversion to identity.
  The quadratic-root selection in `mf6cm2lab_disc` is exercised
  only in synthetic hermetic tests today.

## Running

```
pytest tests/adhoc_test_law1_discrete_broadening.py -v
```

Or as a plain summary:

```
python tests/adhoc_test_law1_discrete_broadening.py
```

## Current findings

Running the ad-hoc pytest module against the 5 files at 14 MeV with
per-file kernel widths and tolerances (see `FILE_CFG` in the script):
**all 15 tests pass** after the investigation described below.

The three MTs 106, 111, 112 in the Fe-56 file are skipped from the
integral check via the `expected_short` file-config entry: TENDL's
LAW=1 encoding for those specific subsecs puts every discrete b at
zero (ND is set to a positive count but the entries carry no
weight) and lets the continuum integrate to 0.5 rather than the
ENDF-6-standard 1.0. The folder faithfully reproduces the file, so
its integral undershoots the `xs * yield_from_MF6_Y_table` reference
by exactly the 0.5-vs-1.0 file-normalisation choice for those MTs.
Not a folder bug; a TENDL evaluator choice. Every other admitted
MT in Fe-56 (23 of 26) and every admitted MT in U-235 (12 of 12)
matches xs * yield within tolerance.

### Diagnosis history (four folder-side fixes and two config tweaks)

Findings from the initial exploration and their resolutions:

1. **`eout_min` sensitivity (config)**: the original config had
   `eout_min=100 keV`, which cut off gamma-cascade lines below
   100 keV (found in Al-27 MT 802/808/815/819 and many Fe-56/U-235
   MTs). The config now uses `eout_min=10 eV`; the ad-hoc-test
   findings for those MTs dropped from 11..46% rel err to well
   under 5% purely by widening the window.

2. **Kernel width too wide (config)**: with `sigma=100 keV` the
   Gaussian tail below the smallest peak position leaked out of
   the integration window. The config now uses `sigma=10 keV`
   which handles peaks down to ~15 keV cleanly.

3. **`get_law1_discrete_lines_from_subsec` out-of-range einc
   (folder bug)**: originally raised IndexError from `find_interval`
   when the user's einc fell outside a subsection's panel-mesh
   range (surfaced by Fe-56 MT 11). The wrapper now pre-filters
   einc against the mesh and zero-pads out-of-range rows,
   mirroring the intent of `pad_outside_dist2d_values` on the
   continuum wrapper.

4. **Duplicate discrete-ep values silently dropped (folder bug)**:
   The Fortran `f6law1_dis` uses `imatch(tp, ep, nd)` which returns
   the first index matching `tp`. When a LAW=1 subsec has two or
   more discrete lines at the same ep value (e.g. U-235 MT 103
   with two lines at 10 keV, Fe-56 with similar patterns across
   most gamma MTs), only the first line's b weight was recovered
   and subsequent duplicates were silently dropped. The wrapper
   now pre-sums b rows at coincident ep values in Python before
   handing to the Fortran routine; each unique ep gets the full
   summed weight and no discrete-line mass is lost. This alone
   dropped the observed Fe-56 shortfalls from 30..50% to under
   5% for every MT that isn't a real TENDL Σb=0.5 quirk.

5. **`compute_dxs_dE_broadened` AssertionError on multi-ejectile
   MTs (folder robustness)**: primitives.properties.get_ejectile
   asserts that either a single ejectile is present or the first
   ejectile is a neutron. MTs like Fe-56 (n,pα) 112, (n,pt) 115..
   117 violate this assertion when compute_dexs is invoked with
   zap=gamma. The catch in compute_dxs_dE_broadened now covers
   both IndexError and AssertionError so those MTs' cont-path
   contribution is zeroed cleanly rather than crashing the
   cumulative sum. The LAW=1 discrete folder still handles them.

6. **TENDL Σb=0.5 encoding (file-side, no fix)**: Fe-56 MT 106,
   111, 112 have ND placeholders with b=0 plus a continuum that
   integrates to 0.5. The folder is correct; the reference
   `xs * yield_from_MF6_Y_table` is off by the file-normalisation
   convention. Skipped from the integral check via
   `expected_short` in the file config.

All fixes 3, 4, 5 landed in the PR proper as folder improvements
worth having in the main-suite code.
