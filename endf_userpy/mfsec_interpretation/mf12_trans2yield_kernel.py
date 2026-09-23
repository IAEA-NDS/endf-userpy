"""MF12 LO=2 transition-probability to photon-yield conversion
(``init_trans2yield`` / ``trans2yield`` port).

Ports the Fortran subroutines ``init_trans2yield`` and
``trans2yield`` in ``endf6.f90``. The algorithm walks the nuclear
level cascade from lower to higher excited states, converting
per-transition probabilities into the full set of photon lines
emitted when an inelastic reaction populates a given upper level.

Two-step API mirrors the Fortran:

- :func:`init_trans2yield` builds and returns the level-energy array
  ``ee`` and probability matrices ``r`` (level->level de-excitation)
  and ``a`` (level->level photon branching) for a whole series of
  discrete inelastic reactions. Call once per section (per ejectile
  series).
- :func:`trans2yield` consumes the section's per-level transition
  data plus the running state ``(ee, r, a)`` (which it updates
  in place) and returns the photon lines (``level_energy``,
  ``photon_energy``, ``photon_yield``) originating from populating
  the excited level associated with the given ``mt``.

The state must be updated in level order (lowest excited level
first) because higher-level transitions may cascade through lower
levels whose branching ratios have to be known already. Callers
that use :func:`compute_photon_yields_from_transition_probabilities`
in :mod:`mf12_interpretation` get that ordering automatically.

Numpy-only. The algorithm is small-integer bookkeeping (typical
nuclide has under 60 discrete levels), not amenable to
vectorisation or JAX autodiff.
"""
from __future__ import annotations

import numpy as np


# Series-lookup table: mt0 for each discrete-inelastic MT range.
# Same ranges as the Fortran: strict inequalities in `.gt.` /
# `.lt.` mean the ground-state MT (mt0 itself) and the compound-
# nucleus MT (mt0 + 41, e.g. 91 for (n,n')) are excluded.
_SERIES_TABLE = (
    # (mt_low_exclusive, mt_high_exclusive, mt0)
    (50, 91, 50),      # (z, n')
    (600, 649, 600),   # (z, p')
    (650, 699, 650),   # (z, d')
    (700, 749, 700),   # (z, t')
    (750, 799, 750),   # (z, he-3')
    (800, 849, 800),   # (z, he-4')
    (875, 891, 875),   # (z, 2n') series (rare)
)


def _series_mt0(mt: int) -> int:
    """Return mt0 for the discrete-inelastic series containing
    ``mt``. Raises ``ValueError`` for MTs outside a discrete
    inelastic series (matches Fortran ``Fatal error``)."""
    for lo, hi, mt0 in _SERIES_TABLE:
        if lo < mt < hi:
            return mt0
    raise ValueError(
        f'MT={mt} is not a discrete inelastic reaction (transition '
        'probabilities apply only to MT ranges 51-90, 601-648, '
        '651-698, 701-748, 751-798, 801-848, 876-890)'
    )


def init_trans2yield(elis: float, qm, qi, maxlevel: int):
    """Initialise level-energy array and probability matrices.

    Parameters
    ----------
    elis : float. Excitation energy of the target nucleus relative
        to 0.0 for the ground state (ENDF ``ELIS``).
    qm : (nlevel,) array-like. ``QM`` for each discrete inelastic
        reaction, ordered lowest excited level first.
    qi : (nlevel,) array-like. ``QI`` for each discrete inelastic
        reaction, in the same order as ``qm``.
    maxlevel : int. Row/column dimension of the returned matrices.
        Must satisfy ``maxlevel > nlevel`` (the ground-state row +
        one row per level; the Fortran default is 60).

    Returns
    -------
    (ee, r, a) : ``(maxlevel,)``, ``(maxlevel, maxlevel)``,
        ``(maxlevel, maxlevel)`` float64 arrays. ``ee[0] = 0`` (ground
        state); ``ee[i + 1] = qm[i] + elis - qi[i]`` for
        ``i = 0..nlevel - 1``. ``r`` starts as the identity (no
        cascades yet). ``a`` starts as all zeros.
    """
    qm = np.asarray(qm, dtype=float)
    qi = np.asarray(qi, dtype=float)
    nlevel = int(qm.shape[0])
    if qi.shape[0] != nlevel:
        raise ValueError(
            f'qm ({nlevel}) and qi ({qi.shape[0]}) length mismatch'
        )
    ee = np.zeros(maxlevel, dtype=float)
    a = np.zeros((maxlevel, maxlevel), dtype=float)
    r = np.eye(maxlevel, dtype=float)
    # ee[i + 1] for i = 0..nlevel - 1; the Fortran indexes ee(i+1)
    # from a 1-based i loop, matching a 0-based ee[i + 1] here.
    ee[1:nlevel + 1] = qm + elis - qi
    return ee, r, a


def trans2yield(mt: int, esns: float, esi, tp, gp, ee, r, a,
                maxnk: int = 5000):
    """Convert one section's transition probabilities into photon
    yields, updating the running state ``(ee, r, a)`` in place.

    Parameters
    ----------
    mt : int. MT number of the discrete inelastic reaction (must be
        in one of the series listed in :data:`_SERIES_TABLE`).
    esns : float. Energy of the residual excited-state level
        populated by reaction ``mt`` (ENDF ``ES``).
    esi : (nt,) array-like. Energies of the lower levels this
        section describes direct transitions to. ``esi[i] = 0`` is
        the ground state.
    tp : (nt,) array-like. Direct transition probabilities to each
        ``esi[i]``.
    gp : (nt,) array-like. Conditional probability that the
        transition is photon (rather than internal-conversion). For
        ``LG=1`` sections this is all ones.
    ee : (maxlevel,) mutable float array from
        :func:`init_trans2yield`. Updated in place.
    r : (maxlevel, maxlevel) mutable float array. Updated in place.
    a : (maxlevel, maxlevel) mutable float array. Updated in place.
    maxnk : int. Upper bound on the number of emitted photon lines
        (safety cap; Fortran default 5000).

    Returns
    -------
    dict with keys ``level_energy``, ``photon_energy``,
    ``photon_yield`` (each a 1D float array, ordered by descending
    ``photon_energy`` to match the Fortran).
    """
    esi = np.asarray(esi, dtype=float)
    tp = np.asarray(tp, dtype=float)
    gp = np.asarray(gp, dtype=float)
    nt = int(esi.shape[0])

    mt0 = _series_mt0(int(mt))
    # Fortran: j = mt - mt0 + 1 (1-based level index). In 0-based
    # indexing this is `j0 = mt - mt0` -> the upper level's slot in
    # ee/r/a arrays.
    j0 = int(mt) - mt0
    ee[j0] = float(esns)

    # Match this section's direct transitions to the level-energy
    # array to fill in one column of r/a. The Fortran walks from
    # k = j - 1 downward so that when computing indirect
    # contributions (r[k, j] += a[k, i] * r[i, j] for i > k), the
    # r[i, j] entries used are already up to date.
    for k in range(j0 - 1, -1, -1):
        eek = ee[k]
        matched_i = -1
        for i in range(nt):
            esii = esi[i]
            if esii == 0.0 and eek == 0.0:
                matched_i = i
                break
            if esii != 0.0 and abs(esii - eek) <= 1.0e-4 * esii:
                matched_i = i
                break
        if matched_i == -1:
            r[k, j0] = 0.0
            a[k, j0] = 0.0
        else:
            p = tp[matched_i]
            r[k, j0] = p
            a[k, j0] = p * gp[matched_i]
        if k != j0 - 1:
            # Indirect contributions through intermediate levels.
            # Fortran: r[k,j] += a[k,i] * r[i,j] for i = k+1..j-1.
            for i in range(k + 1, j0):
                r[k, j0] += a[k, i] * r[i, j0]

    # Enumerate photon lines: for each (upper, lower) level pair
    # with a nonzero yield, record (level_energy, photon_energy,
    # yield). Fortran outer loop, 1-based:
    #   do i = 2, j;     j1 = j + 2 - i   ! j1 = j..2 (descending)
    #   do ii = 1, jm1;  j2 = j - ii      ! j2 = jm1..1 (descending)
    # Converting to our 0-based indexing (Fortran j = j0 + 1, so
    # 1-based j1 -> 0-based j1 = j0 + 2 - i and 1-based j2 -> 0-based
    # j2 = j0 - ii):
    es_list = []
    eg_list = []
    y_list = []
    for i in range(2, j0 + 2):
        j1 = j0 + 2 - i
        ej1 = ee[j1]
        for ii in range(1, j0 + 1):
            j2 = j0 - ii
            ej2 = ee[j2]
            yld = a[j2, j1] * r[j1, j0]
            if yld > 0.0:
                if len(y_list) >= maxnk:
                    raise ValueError(
                        f'too many photon lines from transition '
                        f'probabilities (>{maxnk}); increase maxnk.'
                    )
                eg_list.append(ej1 - ej2)
                es_list.append(ej1)
                y_list.append(yld)

    es_arr = np.asarray(es_list, dtype=float)
    eg_arr = np.asarray(eg_list, dtype=float)
    y_arr = np.asarray(y_list, dtype=float)

    # Sort in descending order of photon energy (matches Fortran).
    if eg_arr.size > 1:
        order = np.argsort(-eg_arr, kind='stable')
        eg_arr = eg_arr[order]
        es_arr = es_arr[order]
        y_arr = y_arr[order]

    return {
        'level_energy': es_arr,
        'photon_energy': eg_arr,
        'photon_yield': y_arr,
    }
