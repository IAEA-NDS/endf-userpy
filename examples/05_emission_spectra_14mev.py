"""
05_emission_spectra_14mev.py
============================

Energy spectrum of neutrons emitted from a target at a fixed incident
energy.

``get_particle_production_dxs_dE(endf_dict, reaction, particle, eincs,
eouts)`` returns d-sigma/dE summed over all reaction channels matching
``reaction`` that emit ``particle``.

The example plots the canonical 14 MeV neutron emission spectrum from
n+U-238 (JENDL-5). The shape carries a lot of physics:

  - soft peak around 1 MeV: prompt fission spectrum (MT=18)
  - smooth tail from ~5 to 7 MeV: (n,2n) and (n,3n) continuum (MT=16, 17)
  - plateau ~10-13 MeV: (n,n_c) continuum inelastic (MT=91)
  - sharp bumps just below 14 MeV: discrete-level inelastics (MT=51..90),
    kinematically reconstructed from MF4 angular distributions
  - sharp peak just below the incident energy: elastic scattering (MT=2),
    again kinematically reconstructed

Download the data file once with:

    wget -U "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" \\
         --referer="https://nds.iaea.org/public/download-endf/JENDL-5/n-index.htm" \\
         "https://nds.iaea.org/public/download-endf/JENDL-5/n/n_092-U-238_9237.zip" \\
         -O /tmp/jendl5_n_U238.zip
    unzip -d /tmp /tmp/jendl5_n_U238.zip
"""

import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from endf_parserpy import EndfParserFactory
from endf_userpy.quantities import get_particle_production_dxs_dE

# Silence DeprecationWarning noise that the MF6 integration helpers
# currently emit on numpy >= 1.25.
warnings.filterwarnings("ignore", category=DeprecationWarning)


ENDF_FILE = Path("/tmp/n_092-U-238_9237.dat")

if not ENDF_FILE.exists():
    raise SystemExit(
        f"Data file not found: {ENDF_FILE}\n"
        "See the docstring at the top of this script for the download command."
    )

parser = EndfParserFactory.create()
endf_dict = parser.parsefile(str(ENDF_FILE))


einc = np.array([1.4e7])
eouts = np.linspace(1e3, 1.5e7, 1500)

spec = get_particle_production_dxs_dE(
    endf_dict, "(n,total)", "n", einc, eouts,
)

# Integrating the spectrum gives a rough estimate of the total neutron
# production cross section. The estimate carries a few-percent error
# from the under-resolved discrete-level and elastic kinematic peaks.
integral = np.trapezoid(spec[0], eouts)
print(f"integrated emission spectrum at 14 MeV: {integral:.2f} barn")


fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.semilogy(eouts / 1e6, spec[0])
ax.set_xlabel("emission energy [MeV]")
ax.set_ylabel(r"d$\sigma$/dE [barn/eV]")
ax.set_title(
    f"{ENDF_FILE.name} neutron emission spectrum at $E_n = 14$ MeV"
)
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(0, 14.5)
fig.tight_layout()
plt.show()
