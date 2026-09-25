# endf-userpy

High-level interpretation of ENDF-6 nuclear data files.

`endf-userpy` is a Python library that answers user-friendly questions
about ENDF-6 nuclear data evaluations: "what is the cross section of
the (n,2n) reaction on Fe-56?", "what is the energy spectrum of
neutrons emitted from U-238 at 14 MeV?", "how much Co-60m is produced
per (n,gamma) reaction on Co-59?".

The library does not parse ENDF-6 itself. It builds on
[endf_parserpy](https://github.com/IAEA-NDS/endf-parserpy), which
turns an ENDF-6 file into a nested Python dict (`endf_dict`).
endf-userpy then walks that dict to reconstruct cross sections,
yields, and differential distributions, hiding which MF section a
piece of data lives in.

> **Status.** This is an early alpha release. The public API is
> stabilising but may still change. See "Known limitations" below.
> Feedback by creating issues is appreciated.

## Installation

> **PyPI releases lag behind `main`.** Development is active and
> new capabilities land several PRs at a time before a release is
> cut. Recent additions that are on `main` but not yet on PyPI
> include end-to-end JAX autodiff through the top-level API across
> all supported MF sections, resolved-resonance reconstruction
> (MLBW and Reich-Moore) with numpy / numba / JAX backends, the
> gamma production pipeline (MF12/13/14/15), and the tabulated
> photon spectra path. If you need those, **install from source**
> (see below). `pip install endf-userpy` gives you the last tagged
> release, which may be missing recent work.

```bash
pip install endf-userpy
```

Prebuilt wheels are available for cpython 3.9–3.13 on:

- Linux x86_64 (`manylinux_2_28`)
- macOS arm64 (>= 14.0)
- macOS x86_64 (>= 15.0)
- Windows AMD64

Other platforms or Python versions install from the sdist
automatically and need a Fortran compiler (`gfortran`) on `PATH`.

Runtime dependencies (`numpy`, `scipy`, `endf_parserpy`) are pulled
in automatically. Some examples additionally use `matplotlib`.

### From source (recommended for latest features)

```bash
git clone https://github.com/IAEA-NDS/endf-userpy
cd endf-userpy
pip install -e .
```

The default install is pure Python and needs no compiler. To also
build the optional Fortran parity-oracle extension used by the
equivalence-test suite, set `ENDF_USERPY_BUILD_FORTRAN=1` before
`pip install` (requires `gfortran` on `PATH`; Windows users can
opt into the Intel Fortran compiler `ifx` with
`ENDF_USERPY_USE_IFX=1`).

## Quick start

```python
import numpy as np
from endf_parserpy import EndfParserFactory
from endf_userpy.quantities import (
    get_available_reactions,
    get_reaction_xs,
)

parser = EndfParserFactory.create()
endf_dict = parser.parsefile("tests/data/n-004_Be_009.endf")

print(get_available_reactions(endf_dict))
# ['(n,total)', '(n,n_0)', '(n,nonelas)', '(n,2n)', '(n,g)', ...]

eincs = np.array([0.0253, 1e3, 1e6, 1.4e7])  # eV
print(get_reaction_xs(endf_dict, "(n,total)", eincs))
# [6.154 6.144 3.341 1.528]   barn
```

## Public API

All user-facing functions live in `endf_userpy.quantities` and take
an `endf_dict` (already parsed) plus user-friendly string identifiers.

| Function | Returns | What it does |
| --- | --- | --- |
| `get_available_reactions(endf_dict)` | list of reaction strings | introspect a file |
| `get_incident_energies(endf_dict, reaction)` | array | tabulated Einc mesh for a channel |
| `get_emission_energies(endf_dict, reaction, particle)` | array | tabulated Eout mesh |
| `get_reaction_xs(endf_dict, reaction, eincs)` | array | cross section of a named channel |
| `get_residual_production_xs(endf_dict, residual, eincs)` | array | production of a specific residual nucleus, isomer-resolved |
| `get_particle_production_xs(endf_dict, reaction, particle, eincs)` | array | ejectile production cross section |
| `get_particle_production_dxs_dE(endf_dict, reaction, particle, eincs, eouts)` | array | dσ/dE energy spectrum of emitted particle |
| `get_particle_production_dxs_dmu(endf_dict, reaction, particle, eincs, mus)` | array | dσ/dΩ angular distribution |
| `get_particle_production_ddxs(endf_dict, reaction, particle, eincs, eouts, mus)` | array | d²σ/dE/dΩ double-differential |

Reaction strings for a neutron projectile follow the ENDF-6 MT
convention:

| String | MT | Meaning |
| --- | --- | --- |
| `"(n,total)"` | MT1 | total cross section |
| `"(n,n_0)"` | MT2 | elastic scattering |
| `"(n,nonelas)"` | MT3 | non-elastic sum |
| `"(n,n)"` | MT4 | inelastic scattering **sum** (MT51..MT91) |
| `"(n,n_i)"` for `i = 1, 2, ...` | MT51 + i-1 | inelastic to i-th discrete level |
| `"(n,2n)"` | MT16 | (n,2n) |
| `"(n,3n)"` | MT17 | (n,3n) |
| `"(n,fission)"` | MT18 | **total fission** (all chances) |
| `"(n,f)"` | MT19 | **first-chance fission only** |
| `"(n,nf)"` | MT20 | second-chance fission |
| `"(n,2nf)"` | MT21 | third-chance fission |
| `"(n,3nf)"` | MT38 | fourth-chance fission |
| `"(n,g)"` | MT102 | radiative capture |
| `"(n,p)"` | MT103 | (n,proton) |
| `"(n,d)"` | MT104 | (n,deuteron) |
| `"(n,t)"` | MT105 | (n,triton) |
| `"(n,h)"` | MT106 | (n,³He) |
| `"(n,a)"` | MT107 | (n,α) |

Two subtleties worth calling out:

- **Fission**: `(n,f)` and `(n,fission)` are **not** the same. `(n,f)`
  is MT19 (first-chance fission only); `(n,fission)` is MT18 (total
  fission summed over all chances). Most modern evaluations put all
  their fission data under MT18 and leave MT19–MT21 empty
  (JENDL-5 U-238, ENDF/B-VIII.1 Pu-239, TENDL-2021 U-235 all do
  this), so `(n,fission)` is the query users almost always want.
  `(n,f)` will return 0 on those files. Some files that split by
  fission chance (older ENDF/B releases, some evaluations of
  higher actinides) do populate MT19 individually; on those,
  `(n,f)` gives you the first-chance piece only.
- **Inelastic scattering**: `(n,n)` is MT4 (**sum** over discrete
  levels + continuum, MT51..MT91). `(n,n_0)` is MT2 (elastic).
  `(n,n_1)` is MT51 (inelastic to the first excited state), and so
  on. See the "Aggregate-reaction sum-MT queries under-count on
  sparse files" note under Known limitations for the admission-
  heuristic caveat.

Full lookup: `endf_userpy/primitives/reactions.py:REACTION_DICT`.
`get_available_reactions(endf_dict)` returns the strings the file
actually populates, so passing that list to your query is the
safest way to avoid a silent zero from a mistyped or file-missing MT.

Particles: `"n"`, `"p"`, `"d"`, `"t"`, `"h"` (helium-3),
`"a"` (alpha), `"g"` (gamma).
Residual nuclei: `"Z-Sym-A"` (e.g. `"27-Co-60"`) or `"Sym-A"`
(e.g. `"Co-60"`), with optional isomer suffix `g`, `m`, `m1`, `m2`,
... (`"Co-60m"` = first metastable).

## Examples

Seven runnable examples in `examples/`:

| File | What it shows |
| --- | --- |
| `01_inspect_file.py` | discover what is in a file |
| `02_reaction_xs.py` | cross sections of named channels with consistency check |
| `03_particle_production_xs.py` | secondary-particle production from Fe-56 |
| `04_residual_production_isomers.py` | Co-58g/m and Co-60g/m via (n,2n) and (n,gamma) |
| `05_emission_spectra_14mev.py` | classic 14 MeV neutron emission spectrum from U-238 |
| `06_ddx_uranium_14mev.py` | double-differential cross section heatmap |
| `07_photonuclear_residuals.py` | (g,Nn) cascade on Au-197 |

Examples 1-2 use a small file shipped under `tests/data/`. Examples 3-7
each include the `wget` command to fetch the JENDL-5 file they need.

## Known limitations

- **No resonance reconstruction.** MF2 (resolved/unresolved resonance
  parameters) is not reconstructed. For evaluations whose MF3 is empty
  in the resonance region, pre-process the file with
  [NJOY RECONR](https://github.com/njoy/NJOY2016) and pass the PENDF
  file in. Evaluations that store MF3 as a **subtractive background**
  in the resolved-resonance region (which ENDF-6 permits, and which
  can produce negative raw MF3 values -- e.g. JENDL-5 Cu-63 MT1/MT2
  tabulate `-0.9 barn` at thermal energies) return the raw background
  by default and emit one summary `UserWarning` per call naming the
  affected MTs and RRR bounds. Configure with the `resonance_range=`
  kwarg on every `get_*` XS API: `'warn'` (default),
  `'warn_nan'`, `'nan'`, `'raise'`.
- **Aggregate-reaction sum-MT queries under-count on sparse files.**
  `get_reaction_xs("(n,n)")` resolves to MT4 (inelastic-scattering
  sum over MT51..90). The admission heuristic drops the parent MT4
  as soon as any child MT51..90 carries detailed MF4/5/6
  distributions -- correct on well-formed evaluations where the
  full discrete-level series is populated, avoiding double-counting.
  On files with **sparse discrete-level enumeration** where some
  child MT has an MF3 cross section but no MF4/5/6/12/13 detail,
  that child's contribution is silently missed by the query (parent
  is dropped in favour of children, and the missing child falls
  through the escape hatch because its ancestor MT4 IS in MF3). Rare
  in modern ENDF/B-VIII, JEFF-4, TENDL-2021, JENDL-5 evaluations
  that populate the full MT51..90 series. When the pathology hits,
  one summary `UserWarning` per (file, parent) is emitted naming
  the missed children; the numeric answer is unchanged. Query the
  parent MT directly (bypasses the heuristic) or ask for the
  missing child MTs individually to recover the missing
  contribution.
- **MT5 catch-all rescue applies only to unique-path MTs.** When an
  evaluation packages a residual channel into MF6/MT5 (the "any other"
  catch-all) instead of the specific MT for that residual,
  `get_reaction_xs` and `get_residual_production_xs` route the MT5
  contribution to the specific MT so the requested reaction is not
  silently zero. The routing only fires when the specific MT is a
  **unique path** to its residual, meaning the residual is unshared
  with any other MT: gamma emission (MT102), single-proton emission
  (MT103), and neutron multiplication with mult >= 2 (MT16 (n,2n),
  MT17 (n,3n), etc.). Deuteron (MT104), triton (MT105), He-3 (MT106)
  and alpha (MT107) emissions are **not** rescued from MT5 because
  their residual ZA is also produced by multi-particle-exit MTs
  (MT28 (n,np) shares MT104's residual; MT32 (n,nd) and MT41
  (n,2np) share MT105's; MT44 (n,n2p) and MT115 (n,pd) share
  MT106's; MT34 (n,n3He), MT116 (n,pt), MT183 (n,npd) and MT190
  (n,2n2p) share MT107's), and lumping the full MT5 catch-all
  onto the single-ejectile MT would over-count the reaction
  whenever the file also folds those multi-particle paths into
  the same catch-all. Also **not** rescued: discrete-level
  scattering MTs (51..90, 601..648, 651..699, 701..748, 751..798,
  801..848, 851..898), continuum-channel MTs (91, 649, 699, 749,
  799, 849, 899), MT4 for a neutron projectile (single-neutron
  ejectile shared with elastic MT2), or the particle-production sum
  MTs (201..207). Files that carry those reactions only through MT5
  return zero for the specific-MT query with no warning -- an MT5
  catch-all cannot be uniquely apportioned across discrete levels,
  a single-neutron ejectile, or a composite-particle ejectile
  whose residual is shared with other exit channels. Ask for MT5
  directly (or use `get_residual_production_xs` for the specific
  residual, or check `get_available_reactions` first) if the file
  evaluation uses an unenumerated catch-all.
- **DDX drops kinematic-delta channels unless broadened.** The
  double-differential API cannot represent elastic and discrete-level
  inelastic channels on a continuous `Eout` grid (their outgoing
  energy is a Dirac delta at `E' = E'_kin(mu, E_in)`). When called
  with `broadening=None` (the default), the API skips those channels
  and emits a `UserWarning` listing the dropped MTs and pointing at
  the `broadening=sigma_eV` argument that folds each delta into a
  finite-width kernel that plots on the grid. They also appear in
  the 1D `dσ/dE` spectrum as sharp peaks.
- **`sigma` above the file's Ein mesh.** By default,
  `get_reaction_xs` and the other `get_*` XS APIs fill above-range
  incident energies with `NaN` and emit one summary `UserWarning`
  per call. Configure with the `above_range=` kwarg (`'warn_nan'`
  default, or `'nan'`, `'warn_zero'`, `'zero'`, `'raise'`).
- **Unimplemented representations** raise `NotImplementedError`:
  MF5 LF=11 (energy-dependent Watt), LF=12 (Madland-Nix); MF6 LAW=3
  (charged-particle elastic isotropic in CM), LAW=4 (recoil), LAW=5
  (charged-particle with phase shift); MF14 LTT=2 (tabulated photon
  angular).
- **Stubs.** `endf_userpy/discrete_quantities.py` and
  `endf_userpy/translation.py` are work-in-progress sketches; do not
  rely on them.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the testing conventions
and per-representation coverage table.

## Filing issues

[github.com/IAEA-NDS/endf-userpy/issues](https://github.com/IAEA-NDS/endf-userpy/issues)

## License and copyright

`endf-userpy` is distributed under the MIT license,
see the `LICENSE` file for details.

Nothing in this license shall be construed as a waiver, either express or implied,
of any of the privileges and immunities accorded to the IAEA by its Member States.

Copyright (c) 2026 International Atomic Energy Agency
