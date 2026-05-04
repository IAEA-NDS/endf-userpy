"""
07_photonuclear_residuals.py
============================

Residual production cross sections from a photonuclear evaluation.

Photonuclear and proton-induced ENDF files often store every channel
in the catch-all MT=5 (in MF6 as ZAP-tagged subsections), with no
per-channel MT in MF3. ``get_residual_production_xs`` handles this
case automatically: when MF8 is absent it falls back to walking
MF6/MT=5 for the requested residual nucleus.

The example uses JENDL-5 g+Au-197 and plots a multi-neutron cascade,
each successive (g,Nn) channel turning on at higher energy:

  - Au-196 from (g,n):  giant dipole resonance, peak ~13 MeV
  - Au-195 from (g,2n): peak ~17 MeV
  - Au-194 from (g,3n): peak ~22 MeV
  - Au-193 from (g,4n): peak ~28 MeV
  - Pt-196 from (g,p):  smaller, charged-particle channel

Download the data file once with:

    wget -U "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" \\
         --referer="https://nds.iaea.org/exfor/endf.htm" \\
         "https://nds.iaea.org/public/download-endf/JENDL-5/g/g_079-Au-197_7925.zip" \\
         -O /tmp/jendl5_g_Au197.zip
    unzip -d /tmp /tmp/jendl5_g_Au197.zip
"""

import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from endf_parserpy import EndfParserFactory
from endf_userpy.quantities import get_residual_production_xs

warnings.filterwarnings("ignore", category=DeprecationWarning)


ENDF_FILE = Path("/tmp/g_079-Au-197_7925.dat")

if not ENDF_FILE.exists():
    raise SystemExit(
        f"Data file not found: {ENDF_FILE}\n"
        "See the docstring at the top of this script for the download command."
    )

parser = EndfParserFactory.create()
endf_dict = parser.parsefile(str(ENDF_FILE))


# Logarithmic mesh from above the (g,n) threshold (~8 MeV) up to
# 100 MeV where the higher-multiplicity channels are open.
eincs = np.logspace(np.log10(7e6), np.log10(1e8), 250)

residuals = ["Au-196", "Au-195", "Au-194", "Au-193", "Pt-196"]
labels = {
    "Au-196": r"$^{197}$Au($\gamma$,n)$^{196}$Au",
    "Au-195": r"$^{197}$Au($\gamma$,2n)$^{195}$Au",
    "Au-194": r"$^{197}$Au($\gamma$,3n)$^{194}$Au",
    "Au-193": r"$^{197}$Au($\gamma$,4n)$^{193}$Au",
    "Pt-196": r"$^{197}$Au($\gamma$,p)$^{196}$Pt",
}
xs = {r: get_residual_production_xs(endf_dict, r, eincs) for r in residuals}


fig, ax = plt.subplots(figsize=(7.5, 4.5))
for r in residuals:
    ax.loglog(eincs / 1e6, xs[r], label=labels[r])
ax.set_xlabel("incident photon energy [MeV]")
ax.set_ylabel("residual production cross section [barn]")
ax.set_title(f"{ENDF_FILE.name} photonuclear residual production")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(1e-4, 1)
fig.tight_layout()
plt.show()
