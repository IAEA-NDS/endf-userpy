"""
04_residual_production_isomers.py
=================================

Isomer-resolved residual production cross sections.

``get_residual_production_xs(endf_dict, residual_nucleus, eincs)``
returns the cross section for production of a specific residual
nucleus, optionally resolved to a particular isomeric state. The
residual is identified by a string of the form ``"Z-Sym-A"`` or
``"Sym-A"``, optionally suffixed with ``g`` (ground state),
``m`` (first metastable, equivalent to ``m1``), or ``mN`` for
higher metastables (``m2``, ``m3``, ...).

The example uses JENDL-5 n+Co-59, which produces Co-58g/m via (n,2n)
(routed through MF8 LMF=10 to MF10) and Co-60g/m via (n,gamma)
(routed through MF8 LMF=9 to MF9). For each isomer pair we plot
total, ground state, and first metastable. The g + m sum equals the
total to floating-point precision at every Einc.

Download the data file once with:

    wget -U "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" \\
         --referer="https://nds.iaea.org/public/download-endf/JENDL-5/n-index.htm" \\
         "https://nds.iaea.org/public/download-endf/JENDL-5/n/n_027-Co-59_2725.zip" \\
         -O /tmp/jendl5_n_Co059.zip
    unzip -d /tmp /tmp/jendl5_n_Co059.zip
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from endf_parserpy import EndfParserFactory
from endf_userpy.quantities import get_residual_production_xs


ENDF_FILE = Path("/tmp/n_027-Co-59_2725.dat")

if not ENDF_FILE.exists():
    raise SystemExit(
        f"Data file not found: {ENDF_FILE}\n"
        "See the docstring at the top of this script for the download command."
    )

parser = EndfParserFactory.create()
endf_dict = parser.parsefile(str(ENDF_FILE))


# Two energy ranges, one tailored to each channel:
#   - Co-58 from (n,2n): threshold ~11 MeV, peak ~14 MeV.
#   - Co-60 from (n,gamma): MF3/MT102 data begins at 100 keV in this
#     file (the resonance region below that lives in MF2 and is not
#     reconstructed by this package).
eincs_co58 = np.linspace(11e6, 50e6, 200)
eincs_co60 = np.logspace(np.log10(1e5), np.log10(3e7), 200)

co58 = {
    label: get_residual_production_xs(endf_dict, label, eincs_co58)
    for label in ["Co-58", "Co-58g", "Co-58m"]
}
co60 = {
    label: get_residual_production_xs(endf_dict, label, eincs_co60)
    for label in ["Co-60", "Co-60g", "Co-60m"]
}


# Internal consistency: g + m should equal total
co58_residual = co58["Co-58"] - co58["Co-58g"] - co58["Co-58m"]
co60_residual = co60["Co-60"] - co60["Co-60g"] - co60["Co-60m"]
print(f"max |Co-58 g+m residual| = {np.max(np.abs(co58_residual)):.2e} barn")
print(f"max |Co-60 g+m residual| = {np.max(np.abs(co60_residual)):.2e} barn")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

for label in ["Co-58", "Co-58g", "Co-58m"]:
    ax1.plot(eincs_co58 / 1e6, co58[label], label=label)
ax1.set_xlabel("incident energy [MeV]")
ax1.set_ylabel("Co-58 production xs [barn]")
ax1.set_title(r"$^{59}$Co(n,2n)$^{58}$Co  (LMF=10 path)")
ax1.legend()
ax1.grid(True, alpha=0.3)

for label in ["Co-60", "Co-60g", "Co-60m"]:
    ax2.loglog(eincs_co60 / 1e6, co60[label], label=label)
ax2.set_xlabel("incident energy [MeV]")
ax2.set_ylabel("Co-60 production xs [barn]")
ax2.set_title(r"$^{59}$Co(n,$\gamma$)$^{60}$Co  (LMF=9 path)")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

fig.tight_layout()
plt.show()
