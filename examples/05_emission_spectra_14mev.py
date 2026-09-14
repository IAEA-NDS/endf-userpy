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

These last two groups appear as "kinematic boxes" with integrable
1/sqrt singularities at the box edges, produced by the angle
integration of the underlying delta(E_out - E_out_kin(mu)). They're
plot-unfriendly: the divergent edges dominate the y-axis on a log
plot and the underlying line widths are dictated by Jacobian
kinematics rather than physical resolution.

Passing ``broadening=sigma`` to the same call folds a Gaussian of
width ``sigma`` (in eV) along E_out. The 1/sqrt edges become
finite kernel-width-wide bumps; continuum contributions get slightly
smoother but otherwise unchanged. The result is directly comparable
to what a finite-resolution experiment would measure.

This example overlays the unbroadened spectrum with a 200 keV-Gaussian
broadened version so the contrast at the discrete-channel edges is
visible at a glance.

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
SIGMA = 200.0e3

if not ENDF_FILE.exists():
    raise SystemExit(
        f"Data file not found: {ENDF_FILE}\n"
        "See the docstring at the top of this script for the download command."
    )

parser = EndfParserFactory.create()
endf_dict = parser.parsefile(str(ENDF_FILE))


einc = np.array([1.4e7])
eouts = np.linspace(1e3, 1.5e7, 1500)

spec_unbroadened = get_particle_production_dxs_dE(
    endf_dict, "(n,total)", "n", einc, eouts,
)
spec_broadened = get_particle_production_dxs_dE(
    endf_dict, "(n,total)", "n", einc, eouts,
    broadening=SIGMA,
)

# Integrating the spectrum gives an estimate of the total neutron
# production cross section. The unbroadened estimate is sensitive to
# the Jacobian-divergent edges at the kinematic boundaries: the
# trapezoidal rule applied to a 1/sqrt singularity can over-shoot
# significantly on a finite grid. The broadened estimate replaces
# the singular spike with a finite kernel-width bump, which the
# trapezoidal rule handles cleanly; it is usually much closer to
# the reference (n,X-n) production cross section.
int_unbroadened = np.trapezoid(spec_unbroadened[0], eouts)
int_broadened = np.trapezoid(spec_broadened[0], eouts)
print(f"integrated emission spectrum at 14 MeV:")
print(f"  unbroadened:               {int_unbroadened:.2f} barn")
print(f"  broadened (sigma=200 keV): {int_broadened:.2f} barn")


fig, axes = plt.subplots(
    1, 2, figsize=(13, 4.7),
    gridspec_kw={"width_ratios": [2, 1]},
)

# Full-range overlay.
axes[0].semilogy(eouts / 1e6, spec_unbroadened[0],
                 label="broadening=None", color="C0", lw=1.0)
axes[0].semilogy(eouts / 1e6, spec_broadened[0],
                 label=f"broadening={SIGMA/1e3:.0f} keV", color="C3", lw=1.2)
axes[0].set_xlabel("emission energy [MeV]")
axes[0].set_ylabel(r"d$\sigma$/dE [barn/eV]")
axes[0].set_xlim(0, 14.5)
axes[0].set_title("full spectrum")
axes[0].legend(loc="upper right")
axes[0].grid(True, which="both", alpha=0.3)

# Zoom on the elastic + discrete-inelastic region where the kinematic
# boxes live. The unbroadened curve shows sharp Jacobian-divergent
# spikes; the broadened curve replaces them with finite Gaussian bumps.
zoom_mask = (eouts >= 1.20e7) & (eouts <= 1.45e7)
axes[1].semilogy(eouts[zoom_mask] / 1e6, spec_unbroadened[0][zoom_mask],
                 label="broadening=None", color="C0", lw=1.0)
axes[1].semilogy(eouts[zoom_mask] / 1e6, spec_broadened[0][zoom_mask],
                 label=f"broadening={SIGMA/1e3:.0f} keV", color="C3", lw=1.2)
axes[1].set_xlabel("emission energy [MeV]")
axes[1].set_title("zoom: elastic + discrete-inelastic edges")
axes[1].grid(True, which="both", alpha=0.3)

fig.suptitle(
    f"{ENDF_FILE.name} neutron emission spectrum at $E_n = 14$ MeV"
)
fig.tight_layout()
plt.show()
