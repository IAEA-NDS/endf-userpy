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

Building requires a Fortran compiler (`gfortran` on Linux/macOS,
`ifx` on Windows) because part of the numerical work is done by a
f2py extension.

```bash
pip install -e .
```

Runtime dependencies: `numpy`, `scipy`, `endf_parserpy`,
`matplotlib` (only for the examples).

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
  file in.
- **DDX drops kinematic-delta channels.** The double-differential API
  silently skips elastic and discrete-level inelastic channels because
  they cannot be represented on a continuous Eout grid. They appear in
  the 1D dσ/dE spectrum as sharp peaks instead.
- **Stubs.** `endf_userpy/discrete_quantities.py` and
  `endf_userpy/translation.py` are work-in-progress sketches; do not
  rely on them.

## Filing issues

[github.com/IAEA-NDS/endf-userpy/issues](https://github.com/IAEA-NDS/endf-userpy/issues)
