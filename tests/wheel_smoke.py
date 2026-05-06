"""Wheel smoke test exercising the Fortran extension end-to-end.

Used by cibuildwheel after each wheel build to confirm that the
Fortran shared library loads, its bundled runtime libs (libgfortran,
libquadmath, ...) are wired correctly, and the MF6 LAW1
reconstruction path produces sane numerical output.
"""
import os
import sys

import numpy as np
from endf_parserpy import EndfParserCpp
from endf_userpy.quantities import (
    get_particle_production_xs,
    get_particle_production_dxs_dE,
)


DATA_FILE = os.path.join('tests', 'data', 'n-004_Be_009.endf')

ed = EndfParserCpp().parsefile(DATA_FILE)

# Pure-Python path through MF3 + multiplicity table.
einc = np.array([1.4e7])
xs = get_particle_production_xs(ed, '(n,2n)', 'n', einc)
assert xs.shape == (1,), xs.shape
assert xs[0] > 0, f'(n,2n) production xs should be positive, got {xs[0]}'

# This call reaches MF6 LAW1 reconstruction, which delegates to the
# Fortran feep_*_law1con routines. If the .so/.pyd or its bundled
# runtime libs are broken, this either fails to import or returns NaN.
eouts = np.array([1e5, 5e5, 1e6, 2e6, 5e6, 8e6])
dxsde = get_particle_production_dxs_dE(ed, '(n,2n)', 'n', einc, eouts)
assert dxsde.shape == (1, 6), dxsde.shape
assert np.all(np.isfinite(dxsde)), 'expected all finite values'
assert np.any(dxsde > 0), 'expected at least one positive value'

print(f'wheel smoke OK on {sys.platform}: '
      f'(n,2n) prod xs = {xs[0]:.4f} barn; '
      f'dXS/dE peak = {dxsde.max():.3e}')
