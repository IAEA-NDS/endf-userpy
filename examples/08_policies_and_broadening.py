"""
08_policies_and_broadening.py
=============================

The three runtime policy knobs on the top-level XS / production APIs.

Every ``get_*_xs`` / ``get_particle_production_*`` call accepts three
optional keyword arguments that control what happens at the awkward
boundaries of the ENDF-6 data model:

  - ``above_range``  -- what to return for incident energies above
    the file's tabulated upper Ein bound. ENDF-6 makes no claim about
    the cross section there; this policy says what happens when the
    caller asks anyway.
  - ``resonance_range`` -- what to return in the resolved-resonance
    region when the evaluation stores MF3 as a subtractive background
    (which ENDF-6 permits, and which can produce negative raw MF3
    values -- e.g. JENDL-5 Cu-63 MT1/MT2 tabulate -0.9 barn at
    thermal energies).
  - ``broadening`` -- fold a Gaussian (or arbitrary kernel) of a given
    width along E_out for the double-differential and 1D emission
    spectra, so kinematic-delta channels (elastic, discrete-level
    inelastic, discrete-level (n,x) cascades) become finite-width
    peaks that plot on a continuous grid. When ``broadening=None``,
    the delta channels are dropped from continuous-grid output and
    the API emits a summary warning naming them.

This example demonstrates each in isolation with prints and simple
plots.

Download the three data files once with:

    wget -U "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" \\
         --referer="https://nds.iaea.org/public/download-endf/JENDL-5/n-index.htm" \\
         "https://nds.iaea.org/public/download-endf/JENDL-5/n/n_029-Cu-063_2925.zip" \\
         -O /tmp/jendl5_n_Cu063.zip
    unzip -o -d /tmp /tmp/jendl5_n_Cu063.zip

    wget -U "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" \\
         --referer="https://nds.iaea.org/public/download-endf/JENDL-5/n-index.htm" \\
         "https://nds.iaea.org/public/download-endf/JENDL-5/n/n_026-Fe-056_2631.zip" \\
         -O /tmp/jendl5_n_Fe056.zip
    unzip -o -d /tmp /tmp/jendl5_n_Fe056.zip

    wget -U "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" \\
         --referer="https://www-nds.iaea.org/exfor/endf00.htm" \\
         "https://nds.iaea.org/public/download-endf/ENDF-B-VIII.1/neutron/n_013-Al-027_1325.zip" \\
         -O /tmp/endfb81_n_Al027.zip
    unzip -o -d /tmp /tmp/endfb81_n_Al027.zip
"""
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from endf_parserpy import EndfParserFactory
from endf_userpy.quantities import (
    get_reaction_xs,
    get_particle_production_dxs_dE,
)

CU63 = Path("/tmp/n_029-Cu-063_2925.dat")
FE56 = Path("/tmp/n_026-Fe-056_2631.dat")
AL27 = Path("/tmp/n_013-Al-027_1325.dat")

for f in (CU63, FE56, AL27):
    if not f.exists():
        raise SystemExit(
            f"Data file not found: {f}\n"
            "See the docstring at the top of this script for the "
            "download commands."
        )

parser = EndfParserFactory.create()
cu63 = parser.parsefile(str(CU63))
fe56 = parser.parsefile(str(FE56))
al27 = parser.parsefile(str(AL27))


# =====================================================================
# 1) above_range -- policy at incident energies above the file mesh.
# =====================================================================
#
# Fe-56 MF3/MT1 tabulates up to some max Ein (typically 200 MeV for
# TENDL/JENDL-5). Asking for a point above that is undefined in the
# ENDF-6 data model. `above_range=` chooses what happens.
#
# Policy set:
#   'warn_nan' (default) -- return NaN and emit one summary
#     UserWarning naming the affected MTs.
#   'nan'                -- return NaN, no warning.
#   'warn_zero'          -- return 0 and emit a summary warning.
#   'zero'               -- return 0, no warning.
#   'raise'              -- raise ValueError immediately.
#
# All policies produce the same in-range values; they differ only in
# what they do above the file's mesh.

E_probe = np.array([1e7, 5e7, 3e8])   # 10 MeV, 50 MeV, 300 MeV
print("above_range demo (Fe-56 (n,total) at 10 MeV / 50 MeV / 300 MeV):")
for policy in ('warn_nan', 'nan', 'warn_zero', 'zero'):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        xs = get_reaction_xs(fe56, '(n,total)', E_probe, above_range=policy)
    warn_count = sum(1 for w in caught if 'above_range' in str(w.message))
    print(f"  {policy:11s}: xs = {np.array2string(xs, precision=4)}  "
          f"({warn_count} warning(s))")
try:
    get_reaction_xs(fe56, '(n,total)', E_probe, above_range='raise')
except ValueError as e:
    print(f"  {'raise':11s}: raised ValueError -- {str(e)[:80]}")


# =====================================================================
# 2) resonance_range -- policy in the resolved-resonance region when
#    MF3 stores a subtractive background.
# =====================================================================
#
# JENDL-5 Cu-63 MT1/MT2 tabulate a large NEGATIVE MF3 background in
# the RRR (a few MeV wide). The reconstruction on top of MF2 is
# expected to add the resonance contribution, giving a physical
# positive total. Without reconstructing (this package does not do
# MF2), the raw MF3 return is arithmetically consistent but
# unphysical on its own.
#
# Policy set:
#   'warn'     (default) -- return the raw background, one summary
#      warning per call naming the affected MTs and the RRR bounds.
#   'warn_nan'           -- fill the RRR points with NaN, one warning.
#   'nan'                -- fill with NaN, no warning.
#   'raise'              -- raise immediately.

# Cu-63 MT1 in the RRR (a few keV) is a good demo point.
E_rrr = np.array([1.0, 100.0, 1e3, 1e5])   # 1 eV, 100 eV, 1 keV, 100 keV
print("\nresonance_range demo (Cu-63 (n,total) inside the RRR):")
for policy in ('warn', 'warn_nan', 'nan'):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        xs = get_reaction_xs(
            cu63, '(n,total)', E_rrr, resonance_range=policy,
        )
    warn_count = sum(
        1 for w in caught if 'resonance' in str(w.message).lower()
    )
    print(f"  {policy:11s}: xs = {np.array2string(xs, precision=4)}  "
          f"({warn_count} warning(s))")
try:
    get_reaction_xs(cu63, '(n,total)', E_rrr, resonance_range='raise')
except ValueError as e:
    print(f"  {'raise':11s}: raised ValueError -- {str(e)[:80]}")


# =====================================================================
# 3) broadening -- fold a Gaussian along E_out so kinematic-delta
#    channels appear as finite-width peaks on the E_out grid.
# =====================================================================
#
# Al-27 has MF6/LAW=2 discrete gamma cascades and LAW=1 ND>0 discrete
# line encodings. At 14 MeV incident, querying the gamma emission
# spectrum with `broadening=None` drops those delta channels from the
# continuous E_out grid; a summary UserWarning names them. With
# broadening, each delta is replaced by a Gaussian of the requested
# width and plotted on the grid.

einc = np.array([1.4e7])
eouts = np.linspace(1e5, 1.5e7, 800)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    spec_none = get_particle_production_dxs_dE(
        al27, '(n,total)', 'g', einc, eouts,
    )
# Show the drop warning if it fired.
dropped_msgs = [str(w.message) for w in caught
                if 'discrete' in str(w.message) and 'dropped' in str(w.message)
                or 'LAW=1 ND' in str(w.message)]

spec_broad = get_particle_production_dxs_dE(
    al27, '(n,total)', 'g', einc, eouts, broadening=200e3,
)

print("\nbroadening demo (Al-27 gamma emission spectrum at 14 MeV):")
if dropped_msgs:
    for m in dropped_msgs:
        print(f"  unbroadened: {m[:140]}...")
else:
    print("  unbroadened: no discrete-drop warnings on this query")
print("  broadened (sigma=200 keV): deltas folded into the E_out grid")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.semilogy(eouts / 1e6, spec_none[0],
            label='broadening=None (deltas dropped)', color='C0', lw=1.0)
ax.semilogy(eouts / 1e6, spec_broad[0],
            label='broadening=200 keV', color='C3', lw=1.2)
ax.set_xlabel('gamma emission energy [MeV]')
ax.set_ylabel(r'd$\sigma$/dE [barn/eV]')
ax.set_xlim(0, 12)
ax.set_title('Al-27 gamma emission at $E_n = 14$ MeV')
ax.legend(loc='upper right')
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
plt.show()
