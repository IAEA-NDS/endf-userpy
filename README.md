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

### From source

```bash
git clone https://github.com/IAEA-NDS/endf-userpy
cd endf-userpy
pip install -e .
```

Requires `gfortran` on `PATH`. Windows users who prefer to build
with the Intel Fortran compiler (`ifx`) can opt in by setting
`ENDF_USERPY_USE_IFX=1` in the environment before `pip install`.

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

Reaction strings: `"(n,total)"`, `"(n,n_0)"` (elastic), `"(n,2n)"`,
`"(n,g)"` (capture), `"(n,p)"`, `"(n,a)"`, etc.
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
- **MT5 catch-all rescue applies only to unique-path MTs.** When an
  evaluation packages a residual channel into MF6/MT5 (the "any other"
  catch-all) instead of the specific MT for that residual,
  `get_reaction_xs` and `get_residual_production_xs` route the MT5
  contribution to the specific MT so the requested reaction is not
  silently zero. The routing only fires when the specific MT is a
  **unique path** to its residual, i.e. one light-ion ejectile
  (`g`, `n`, `p`, `d`, `t`, `h`, `a`) whose residual ZA is fixed by
  the projectile, target and ejectile. It does **not** rescue:
  discrete-level scattering MTs (51..90, 601..648, 651..699,
  701..748, 751..798, 801..848, 851..898), continuum-channel MTs
  (91, 649, 699, 749, 799, 849, 899), MT4 for a neutron projectile
  (single-neutron ejectile shared with elastic MT2), or the
  particle-production sum MTs (201..207). Files that carry those
  reactions only through MT5 return zero for the specific-MT query
  with no warning -- an MT5 catch-all cannot be uniquely apportioned
  across discrete levels or a single-neutron ejectile without extra
  information. Ask for MT5 directly (or check `get_available_reactions`
  first) if the file evaluation uses an unenumerated catch-all.
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
