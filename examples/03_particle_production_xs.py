"""
03_particle_production_xs.py
============================

Cross sections for emission of secondary particles (n, p, alpha, gamma).

``get_particle_production_xs(endf_dict, reaction, particle, eincs)``
returns the cross section for emission of ``particle`` summed over all
reaction channels matching ``reaction``. Pass ``"(n,total)"`` to mean
"any reaction in this file that emits this particle".

The example uses JENDL-5 n+Fe-56. Iron is a structural material in many
applications (reactor pressure vessels, fusion devices), so its
secondary-particle production cross sections are widely studied.

The data file is not shipped with this repository. Download it with:

    wget -U "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" \\
         --referer="https://nds.iaea.org/public/download-endf/JENDL-5/n-index.htm" \\
         "https://nds.iaea.org/public/download-endf/JENDL-5/n/n_026-Fe-56_2631.zip" \\
         -O /tmp/jendl5_n_Fe056.zip
    unzip -d /tmp /tmp/jendl5_n_Fe056.zip
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from endf_parserpy import EndfParserFactory
from endf_userpy.quantities import get_particle_production_xs


ENDF_FILE = Path("/tmp/n_026-Fe-56_2631.dat")

if not ENDF_FILE.exists():
    raise SystemExit(
        f"Data file not found: {ENDF_FILE}\n"
        "See the docstring at the top of this script for the download command."
    )

parser = EndfParserFactory.create()
endf_dict = parser.parsefile(str(ENDF_FILE))


# Logarithmic mesh covering the interesting fast-neutron range. Most
# particle-production thresholds sit between 1 MeV and 25 MeV; above
# that the cross sections plateau or fall off slowly.
eincs = np.logspace(np.log10(1e6), np.log10(6e7), 200)

particles = ["n", "p", "a", "g"]
labels = {"n": "neutron", "p": "proton", "a": "alpha", "g": "gamma"}
xs = {
    p: get_particle_production_xs(endf_dict, "(n,total)", p, eincs)
    for p in particles
}


fig, ax = plt.subplots(figsize=(7, 4.5))
for p in particles:
    ax.loglog(eincs / 1e6, xs[p], label=labels[p])
ax.set_xlabel("incident neutron energy [MeV]")
ax.set_ylabel("particle production cross section [barn]")
ax.set_title(f"{ENDF_FILE.name} secondary-particle production")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
plt.show()
