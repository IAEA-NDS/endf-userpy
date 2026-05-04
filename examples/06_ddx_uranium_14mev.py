"""
06_ddx_uranium_14mev.py
=======================

Double-differential cross section (DDX) of secondary neutron emission.

``get_particle_production_ddxs(endf_dict, reaction, particle, eincs,
eouts, mus)`` returns d^2-sigma/dE/dOmega summed over all reaction
channels matching ``reaction`` that emit ``particle`` with a true
continuous distribution. Channels whose emission is a kinematic delta
(elastic, discrete-level inelastic) are skipped because they cannot be
represented on a continuous Eout grid.

The example plots 14 MeV n+U-238 from JENDL-5 as a 2D heatmap of
emission energy versus emission angle cosine in the LAB frame. The
smooth contributions visible in the heatmap come from:

  - prompt fission spectrum (MT=18): isotropic, dominant at low Eout
  - (n,2n), (n,3n) (MT=16, 17): mildly forward-peaked, mid Eout
  - (n,n_c) continuum inelastic (MT=91): forward-peaked, high Eout

The kinematic features visible in the 1D energy spectrum
(elastic peak ~14 MeV, discrete-level peaks ~13-14 MeV) are
deliberately absent in the DDX, by design.

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
from matplotlib.colors import LogNorm
from endf_parserpy import EndfParserFactory
from endf_userpy.quantities import get_particle_production_ddxs

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
eouts = np.linspace(1e3, 1.4e7, 200)
mus = np.linspace(-1, 1, 100)

ddx = get_particle_production_ddxs(
    endf_dict, "(n,total)", "n", einc, eouts, mus,
)
# shape is (n_einc=1, n_eout, n_mu)


fig, ax = plt.subplots(figsize=(7.5, 5))
mesh = ax.pcolormesh(
    mus, eouts / 1e6, ddx[0],
    norm=LogNorm(vmin=ddx[ddx > 0].min(), vmax=ddx.max()),
    shading="auto", cmap="viridis",
)
fig.colorbar(mesh, ax=ax, label=r"d$^2\sigma$/dE/d$\Omega$ [b/eV/sr]")
ax.set_xlabel(r"emission angle cosine $\mu$ (LAB)")
ax.set_ylabel("emission energy [MeV]")
ax.set_title(
    f"{ENDF_FILE.name} neutron DDX at $E_n = 14$ MeV"
)
fig.tight_layout()
plt.show()
