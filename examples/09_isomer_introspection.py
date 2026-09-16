"""
09_isomer_introspection.py
==========================

Introspecting the residual-nucleus and isomer-state coverage of an
ENDF-6 evaluation before you query for numbers.

Every ``get_residual_production_xs('X-Ym')`` call resolves the
residual-nucleus string to a ZAP and LFS pair and looks the pair up
in MF8. If the evaluation doesn't declare that (ZAP, LFS) pair via
MF8, the call falls back to MF3 (or MF6 with ZAP-tagged subsections)
and cannot resolve isomers. Three lightweight introspection helpers
let you check what an evaluation actually contains before you rely
on a query:

  - ``get_declared_residuals(endf)`` -- every (ZAP, LFS) pair MF8
    declares, formatted as human-readable residual strings
    ``"X-Y"`` / ``"X-Ym"`` / ``"X-Ym2"`` etc.
  - ``is_residual_declared(endf, residual)`` -- boolean for a
    specific residual string.
  - ``get_declared_isomer_states(endf, residual)`` -- for a bare
    residual (no isomer suffix), the list of isomer-state suffixes
    MF8 declares for it (``''`` for the ground state, ``'m'``,
    ``'m2'``, ...).

This example uses JENDL-5 Cu-63, which has MF8 for a range of
neutron-induced residuals. Of the 21 residuals declared, three
carry both ground and metastable states -- Co-58, Co-60, Co-62 --
and are the ones an isomer-aware caller wants.

Download the data file once with:

    wget -U "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" \\
         --referer="https://nds.iaea.org/public/download-endf/JENDL-5/n-index.htm" \\
         "https://nds.iaea.org/public/download-endf/JENDL-5/n/n_029-Cu-063_2925.zip" \\
         -O /tmp/jendl5_n_Cu063.zip
    unzip -o -d /tmp /tmp/jendl5_n_Cu063.zip
"""
from pathlib import Path
from endf_parserpy import EndfParserFactory
from endf_userpy.quantities import (
    get_declared_residuals,
    is_residual_declared,
    get_declared_isomer_states,
    get_residual_production_xs,
)
import numpy as np
import warnings


ENDF_FILE = Path("/tmp/n_029-Cu-063_2925.dat")

if not ENDF_FILE.exists():
    raise SystemExit(
        f"Data file not found: {ENDF_FILE}\n"
        "See the docstring at the top of this script for the download command."
    )

parser = EndfParserFactory.create()
endf_dict = parser.parsefile(str(ENDF_FILE))


# =====================================================================
# 1) List every residual MF8 declares in this file.
# =====================================================================

residuals = get_declared_residuals(endf_dict)
print(f"MF8 declares {len(residuals)} residual (ZAP, LFS) pairs:")
for r in residuals:
    print(f"  {r}")


# =====================================================================
# 2) Ask about a specific residual before querying it.
# =====================================================================

for probe in ("Co-60", "Co-60m", "Co-60m2", "Cu-64m", "Ni-63"):
    ok = is_residual_declared(endf_dict, probe)
    marker = "declared" if ok else "not in MF8"
    print(f"is_residual_declared({probe!r}) -> {ok}  ({marker})")


# =====================================================================
# 3) Enumerate isomer states MF8 declares for a bare residual.
# =====================================================================

for bare in ("Co-60", "Co-58", "Cu-64", "Fe-56"):
    try:
        states = get_declared_isomer_states(endf_dict, bare)
    except Exception as e:
        print(f"get_declared_isomer_states({bare!r}) -> raised: {e}")
        continue
    if not states:
        print(f"get_declared_isomer_states({bare!r}) -> [] (not in MF8)")
        continue
    # '' is the ground state, 'm' is the first metastable, etc.
    pretty = ', '.join(repr(s) if s == '' else f"'{s}'" for s in states)
    print(f"get_declared_isomer_states({bare!r}) -> [{pretty}]")


# =====================================================================
# 4) Non-declared query pattern: what happens if you ask for a
#    residual MF8 doesn't declare? The call falls back to MF3/MF6
#    and either returns the mass-conserving residual-only answer
#    (if the file has ZAP-tagged subsections) or 0 with a summary
#    UserWarning naming the unrouted MTs.
# =====================================================================

# Sum of Co-60g + Co-60m equals the bare Co-60 residual (which
# resolves to LFS=None, i.e. any isomer state). This works because
# MF8 declares both isomers for the responsible MT.
einc = np.linspace(1e6, 2e7, 10)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    xs_ground = get_residual_production_xs(endf_dict, 'Co-60', einc)
    xs_m1     = get_residual_production_xs(endf_dict, 'Co-60m', einc)
    xs_any    = get_residual_production_xs(endf_dict, 'Co-60', einc)

# xs_any is the ground+m1 sum since MF8 declares both isomers.
print("\nCo-60 production at 5 sample incident energies (barn):")
print(f"  {'Ein [eV]':>12}  {'ground':>12}  {'m1':>12}  {'any LFS':>12}")
for i in (0, 3, 5, 7, 9):
    print(
        f"  {einc[i]:12.4e}  "
        f"{xs_ground[i]:12.6g}  "
        f"{xs_m1[i]:12.6g}  "
        f"{xs_any[i]:12.6g}"
    )
resid = xs_any - xs_ground - xs_m1
print(f"  |bare - ground - m1| max: {float(np.max(np.abs(resid))):.3e}  "
      f"(should be ~0; MF8 declares both isomers so the sums agree)")
