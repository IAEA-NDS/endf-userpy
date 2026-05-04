"""
01_inspect_file.py
==================

Discover what is inside an ENDF-6 file.

Two functions answer the most basic questions:

- ``get_available_reactions`` lists the reaction channels.
- ``get_incident_energies`` returns the tabulated incident-energy mesh
  for a chosen channel.

The file used here, n-004_Be_009.endf, is shipped in tests/data/ so this
example runs without any additional download.
"""

from pathlib import Path
from endf_parserpy import EndfParserFactory
from endf_userpy.quantities import (
    get_available_reactions,
    get_incident_energies,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
endf_file = REPO_ROOT / "tests" / "data" / "n-004_Be_009.endf"

parser = EndfParserFactory.create()
endf_dict = parser.parsefile(str(endf_file))


# --- list every reaction channel present in the file --------------------
reactions = get_available_reactions(endf_dict)
print(f"{endf_file.name} contains {len(reactions)} reaction channels:")
for r in reactions:
    print(f"  {r}")


# --- for one channel, look at the tabulated incident-energy mesh --------
channel = "(n,n_0)"  # elastic scattering
eincs = get_incident_energies(endf_dict, channel)
print(f"\nincident-energy mesh for {channel}:")
print(f"  {len(eincs)} points")
print(f"  spans {eincs.min():.3e} eV to {eincs.max():.3e} eV")
print(f"  first 5: {eincs[:5]}")
print(f"  last 5:  {eincs[-5:]}")
