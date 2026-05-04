"""
02_reaction_xs.py
=================

Cross sections of named reaction channels.

``get_reaction_xs`` returns the cross section of a single channel
evaluated on a user-supplied incident-energy mesh. Reaction channels
are referred to by string (e.g. ``(n,n_0)`` for elastic, ``(n,2n)``,
``(n,g)`` for radiative capture). The list of channels present in a
file comes from ``get_available_reactions`` (see 01_inspect_file.py).

This example plots four cross sections of n+Be-9 from the file shipped
in tests/data/, and verifies that the total decomposes into the
elastic and the nonelastic part to floating-point precision.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from endf_parserpy import EndfParserFactory
from endf_userpy.quantities import (
    get_reaction_xs,
    get_incident_energies,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
endf_file = REPO_ROOT / "tests" / "data" / "n-004_Be_009.endf"

parser = EndfParserFactory.create()
endf_dict = parser.parsefile(str(endf_file))


# Use the file's own incident-energy mesh for (n,total) so every
# channel is evaluated on the same grid.
eincs = get_incident_energies(endf_dict, "(n,total)")

channels = ["(n,total)", "(n,n_0)", "(n,nonelas)", "(n,2n)", "(n,g)"]
xs = {ch: get_reaction_xs(endf_dict, ch, eincs) for ch in channels}


# Internal consistency: total = elastic + nonelastic
diff = xs["(n,total)"] - xs["(n,n_0)"] - xs["(n,nonelas)"]
print(f"max |total - elastic - nonelastic| = {np.max(np.abs(diff)):.2e} barn")


fig, ax = plt.subplots(figsize=(6.5, 4.5))
for ch in channels:
    ax.loglog(eincs / 1e6, xs[ch], label=ch)
ax.set_xlabel("incident energy [MeV]")
ax.set_ylabel("cross section [barn]")
ax.set_title(f"{endf_file.name} reaction cross sections")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
plt.show()
