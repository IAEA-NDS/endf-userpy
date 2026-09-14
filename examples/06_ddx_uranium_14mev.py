"""
06_ddx_uranium_14mev.py
=======================

Double-differential cross section (DDX) of secondary neutron emission.

``get_particle_production_ddxs(endf_dict, reaction, particle, eincs,
eouts, mus)`` returns d^2-sigma/dE/dOmega summed over all reaction
channels matching ``reaction`` that emit ``particle``.

By default only channels with a true continuous distribution contribute:
fission, (n,2n), (n,3n), continuum inelastic and similar smooth sources.
Channels whose secondary emission is a kinematic delta (elastic
scattering MT 2, discrete-level inelastic MT 51..90) live on a 1D curve
in (E_out, mu) space and cannot be represented on a finite heatmap; they
are silently excluded.

Passing ``broadening=sigma`` enables the discrete-channel contribution
to be plotted as well: the kinematic delta is replaced by a Gaussian
of width ``sigma`` (in eV) along E_out, and the continuum is folded
with the same kernel for visual consistency. The result is comparable
to what an experiment with that energy resolution would measure.

This example plots both versions side by side for n+U-238 at 14 MeV
from JENDL-5. In the broadened panel you should see:

  - the elastic ridge tracing E_out_kin(mu) from ~13.77 MeV at mu = -1
    up to ~14.00 MeV at mu = +1, sharply forward-peaked
  - discrete-inelastic bands at the kinematically allowed E_out values
    of MT 51..76 levels
  - the smooth fission spectrum (peaks near 1 MeV) and (n,2n) / (n,n_c)
    contributions, now slightly smoothed by the 200 keV kernel.

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
SIGMA = 200.0e3  # Gaussian kernel width along E_out, in eV

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

ddx_unbroadened = get_particle_production_ddxs(
    endf_dict, "(n,total)", "n", einc, eouts, mus,
)
ddx_broadened = get_particle_production_ddxs(
    endf_dict, "(n,total)", "n", einc, eouts, mus,
    broadening=SIGMA,
)
# shape (n_einc=1, n_eout, n_mu) for both


# Use a shared color scale so the two panels are directly comparable.
positive = np.concatenate([
    ddx_unbroadened[ddx_unbroadened > 0],
    ddx_broadened[ddx_broadened > 0],
])
vmin = positive.min()
vmax = max(ddx_unbroadened.max(), ddx_broadened.max())
norm = LogNorm(vmin=vmin, vmax=vmax)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, ddx, title in (
    (axes[0], ddx_unbroadened, "broadening=None"),
    (axes[1], ddx_broadened, f"broadening={SIGMA/1e3:.0f} keV (Gaussian)"),
):
    mesh = ax.pcolormesh(
        mus, eouts / 1e6, ddx[0],
        norm=norm, shading="auto", cmap="viridis",
    )
    ax.set_xlabel(r"emission angle cosine $\mu$ (LAB)")
    ax.set_title(title)

axes[0].set_ylabel("emission energy [MeV]")
fig.colorbar(mesh, ax=axes, label=r"d$^2\sigma$/dE/d$\Omega$ [b/eV/sr]")
fig.suptitle(f"{ENDF_FILE.name} neutron DDX at $E_n = 14$ MeV")
plt.show()
