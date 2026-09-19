"""Scalar ``@njit`` closed-form factor helpers shared by every MF2
numba kernel.

Numerically identical to the array-agnostic
:mod:`mf2_interpretation_factors` versions, but scalar and
``inline='always'`` so numba fuses each call into the per-energy
kernel body instead of leaving a function-call boundary in the
hot loop.

Currently used by :mod:`mf2_interpretation_mlbw_numba` and
:mod:`mf2_interpretation_reichmoore_numba`; adding SLBW / RML
numba paths later plugs in here rather than re-copying the
tables.
"""
from __future__ import annotations

import math


try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*a, **kw):
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        return lambda f: f


@njit(cache=True, inline='always')
def low_L_pnt_shf(rho, L):
    """Scalar closed-form (P_L, S_L) for L in 0..5.

    Same expressions as
    :func:`mf2_interpretation_factors.low_L_pnt_shf` (array form),
    scalar so numba can fuse them into a per-energy loop.

    L>=6 is rejected at the wrapper boundary in every caller; the
    NaN sentinel here would only reach an already-validated
    call path.
    """
    r2 = rho * rho
    if L == 0:
        return rho, 0.0
    elif L == 1:
        d = 1.0 + r2
        return rho * r2 / d, -1.0 / d
    elif L == 2:
        d = 9.0 + r2 * (3.0 + r2)
        return rho * r2 * r2 / d, -(18.0 + 3.0 * r2) / d
    elif L == 3:
        d = 225.0 + r2 * (45.0 + r2 * (6.0 + r2))
        return (rho * r2 * r2 * r2 / d,
                -(675.0 + r2 * (90.0 + 6.0 * r2)) / d)
    elif L == 4:
        d = 11025.0 + r2 * (1575.0 + r2 * (135.0 + r2 * (10.0 + r2)))
        return (rho * r2 ** 4 / d,
                -(44100.0 + r2 * (4725.0 + r2 * (270.0 + 10.0 * r2))) / d)
    elif L == 5:
        d = 893025.0 + r2 * (
            99225.0 + r2 * (6300.0 + r2 * (315.0 + r2 * (15.0 + r2)))
        )
        p = rho * r2 ** 5 / d
        s = -(4465125.0 + r2 * (
            396900.0 + r2 * (18900.0 + r2 * (630.0 + 15.0 * r2))
        )) / d
        return p, s
    else:
        return float('nan'), float('nan')


@njit(cache=True, inline='always')
def low_L_phase(rho, L):
    """Scalar hard-sphere phase phi_L for L in 0..5. See
    :func:`mf2_interpretation_factors.low_L_phase` for the closed
    forms.
    """
    r2 = rho * rho
    if L == 0:
        return rho
    elif L == 1:
        return rho - math.atan(rho)
    elif L == 2:
        return rho - math.atan2(3.0 * rho, 3.0 - r2)
    elif L == 3:
        return rho - math.atan2(rho * (15.0 - r2), 15.0 - 6.0 * r2)
    elif L == 4:
        return rho - math.atan2(
            rho * (105.0 - 10.0 * r2), 105.0 - r2 * (45.0 - r2),
        )
    elif L == 5:
        return rho - math.atan2(
            rho * (945.0 - r2 * (105.0 - r2)),
            945.0 - r2 * (420.0 - 15.0 * r2),
        )
    else:
        return float('nan')
