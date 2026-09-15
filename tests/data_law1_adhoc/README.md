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
per-file kernel widths and tolerances (see `FILE_CFG` in the script)
gives, as of the initial exploration:

- **Be-9, B-11**: DDX and 1D dxs/dE integral checks pass. Baseline
  case: pure-discrete gamma lines, integral matches xs * yield within
  the file tolerance.
- **Al-27**: DDX + 1D integrals for MTs 51..79 (inelastic-level
  cascades) and MTs 601..748 (charged-particle emission cascades)
  pass tolerance. The higher-MT alpha-emission cascades (MTs 802,
  808..819) come in ~11..27% below xs * yield. Not immediately
  diagnosed. Worth probing whether the eout window / kernel width /
  mu grid picks these up correctly, or whether the aggregator misses
  a subsection.
- **Fe-56**: several MTs (28, 91, 102..107) show rel err 10..50% in
  either the DDX or 1D check, and the cont folder occasionally
  returns tiny negative values (FFT numerical noise beyond the
  1e-10-of-peak tolerance in a few cases). These MTs are all mixed
  cont+disc subsecs. Discrepancies likely reflect the interaction
  between the LAW=1 discrete folder and the continuous folder when
  they share yield in the way LAW=1 stores it -- worth digging in
  to confirm the split is correct.
- **U-235**: MTs 16, 17, 22, 28 come in 30..70% below xs * yield.
  Large mixed subsecs on an actinide; probably related to the Fe-56
  finding.
- **Public API smoke test** (`get_particle_production_ddxs(broadening=)`
  on gamma production) runs to completion on all 5 files.

The intent is that these findings inform whether each file is worth
promoting into the main suite. Files where the tolerances just need
loosening (or the eouts window widening) are easy promotions; files
where a discrepancy points at a real folder bug are more valuable as
regression fixtures once the bug is understood and fixed.
