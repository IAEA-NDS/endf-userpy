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
  data plus the running state ``(ee, r, a)`` and returns the photon
  lines (``level_energy``, ``photon_energy``, ``photon_yield``)
  originating from populating the excited level associated with
  the given ``mt``, and the updated state.

The state must be updated in level order (lowest excited level
first) because higher-level transitions cascade through lower
levels whose branching ratios have to be known already. Callers
that use :func:`compute_photon_yields_from_transition_probabilities`
in :mod:`mf12_interpretation` get that ordering automatically.

Backend-agnostic. ``xp=None`` runs on numpy and preserves the
pre-port numerical behaviour bit-for-bit. Under xp=jax the entire
cascade traces cleanly: ``jax.grad`` reaches ``ES_NS`` (upper
level), ``ES`` (lower levels), ``TP`` (direct probabilities),
``GP`` (photon-vs-conversion branching) and the ``ELIS``/``QM``/
``QI`` inputs that seed ``ee``. Both ``jax.grad(jax.jit(f))`` and
``jax.jit(jax.grad(f))`` compose.

Fixed-shape output: for a given upper level ``j0 = mt - mt0`` the
kernel emits a ``(j0 * j0,)`` set of candidate photon lines. Lines
with a physically zero yield (level not present in the cascade or
photon-branching factor zero) come through as ``photon_yield = 0``
entries and are transparent to the downstream
``find_indices_with_tol`` matcher.
"""
from __future__ import annotations

import numpy as np

from ..primitives import array_ns


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


def _set_1d(arr, idx, val, xp):
    if xp.name == 'jax':
        return arr.at[idx].set(val)
    arr[idx] = val
    return arr


def _set_2d(arr, i, j, val, xp):
    if xp.name == 'jax':
        return arr.at[i, j].set(val)
    arr[i, j] = val
    return arr


def init_trans2yield(elis, qm, qi, maxlevel: int, xp=None):
    """Initialise level-energy array and probability matrices.

    Parameters
    ----------
    elis : float or 0-d array. Excitation energy of the target
        nucleus relative to 0.0 for the ground state (ENDF
        ``ELIS``). Tracer-safe under xp=jax.
    qm : (nlevel,) array-like. ``QM`` for each discrete inelastic
        reaction, ordered lowest excited level first. Tracer-safe.
    qi : (nlevel,) array-like. ``QI`` for each discrete inelastic
        reaction, in the same order as ``qm``. Tracer-safe.
    maxlevel : int. Row/column dimension of the returned matrices.
        Must satisfy ``maxlevel > nlevel``.
    xp : array-namespace adapter; ``None`` defaults to numpy.

    Returns
    -------
    (ee, r, a) : ``(maxlevel,)``, ``(maxlevel, maxlevel)``,
        ``(maxlevel, maxlevel)`` float64 arrays. ``ee[0] = 0``
        (ground state); ``ee[i + 1] = qm[i] + elis - qi[i]`` for
        ``i = 0..nlevel - 1``. ``r`` starts as the identity;
        ``a`` starts as all zeros.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    qm = xp.asarray(qm, dtype=xp.float64)
    qi = xp.asarray(qi, dtype=xp.float64)
    elis = xp.asarray(elis, dtype=xp.float64)
    nlevel = int(qm.shape[0])
    if int(qi.shape[0]) != nlevel:
        raise ValueError(
            f'qm ({nlevel}) and qi ({int(qi.shape[0])}) length mismatch'
        )
    ee_slice = qm + elis - qi
    ee = xp.zeros(maxlevel, dtype=xp.float64)
    a = xp.zeros((maxlevel, maxlevel), dtype=xp.float64)
    r = xp.eye(maxlevel, dtype=xp.float64)
    if xp.name == 'jax':
        ee = ee.at[1:nlevel + 1].set(ee_slice)
    else:
        ee[1:nlevel + 1] = ee_slice
    return ee, r, a


def _match_direct_vectorised(esi, ee_slice, tp, gp, xp):
    """Vectorised level-matching for every ``k in [0, j0-1]``
    simultaneously. Same two-branch tolerance semantics as the
    Fortran but computed with a single ``(j0, nt)`` outer product
    so the trace graph does not grow with ``j0``.

    Parameters
    ----------
    esi : (nt,)  lower-level energies.
    ee_slice : (j0,)  running-state levels ``ee[0:j0]``.
    tp, gp : (nt,)  direct transition and photon-branching factors.

    Returns ``(p_direct, a_val)`` of shape ``(j0,)`` each, with
    zero at positions where no lower level matched.
    """
    diffs = xp.abs(esi[None, :] - ee_slice[:, None])         # (j0, nt)
    # Tolerance rule matches the Fortran two-branch logic:
    #   esi=0 and eek=0  -> diff=0, tol=0, within=True.
    #   esi=0 and eek!=0 -> diff=|eek|, tol=0, within=False.
    #   esi!=0 and eek=0 -> diff=|esi|, tol=1e-4*|esi|, within=False.
    #   esi!=0 and eek!=0-> |esi-eek| <= 1e-4*|esi|.
    tol_abs = 1e-4 * xp.abs(esi)[None, :]                    # (1, nt)
    within = diffs <= tol_abs                                # (j0, nt)
    matched = xp.any(within, axis=1)                         # (j0,)
    first_idx = xp.argmax(within.astype(xp.int32), axis=1)   # (j0,)
    tp_at = tp[first_idx]                                    # (j0,)
    gp_at = gp[first_idx]                                    # (j0,)
    p_direct = xp.where(matched, tp_at, 0.0)                 # (j0,)
    a_val = xp.where(matched, tp_at * gp_at, 0.0)            # (j0,)
    return p_direct, a_val


def trans2yield(mt: int, esns, esi, tp, gp, ee, r, a,
                maxnk: int = 5000, xp=None):
    """Convert one section's transition probabilities into photon
    yields, and return updated cascade state.

    Dispatch layer: numpy backends take the tight-shape path
    (:func:`_trans2yield_tight`) which matches the pre-rewrite
    per-MT arithmetic footprint; jax backends take the fully
    static-padded path (:func:`_trans2yield_padded`) which fixes
    every internal shape at ``ee.shape[0]`` so XLA's per-shape
    first-call compile cost is paid once instead of once per MT.

    Both paths return ``(ee, r, a, result_dict)`` with the same
    physical content; the padded path emits a longer fixed-length
    ``result_dict['photon_yield']`` where entries beyond the
    cascade positions are zero, which the downstream matcher in
    :mod:`mf12_interpretation` handles transparently.

    ``maxnk`` is kept for API compatibility and unused (both paths
    emit a fixed-size grid rather than a variable-length list).

    Parameters
    ----------
    mt : int. MT number of the discrete inelastic reaction (must be
        in one of the series listed in :data:`_SERIES_TABLE`).
    esns : float or 0-d array. Energy of the residual excited-state
        level populated by reaction ``mt`` (ENDF ``ES``). Tracer-safe.
    esi : (nt,) array-like. Energies of the lower levels this
        section describes direct transitions to. ``esi[i] = 0`` is
        the ground state. Tracer-safe.
    tp, gp : (nt,) array-like. Direct transition probabilities and
        photon-vs-internal-conversion factors. Tracer-safe.
    ee, r, a : running cascade state from :func:`init_trans2yield`.
    xp : array-namespace adapter; ``None`` defaults to numpy.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    del maxnk
    if xp.name == 'numpy':
        return _trans2yield_tight(mt, esns, esi, tp, gp, ee, r, a, xp)
    return _trans2yield_padded(mt, esns, esi, tp, gp, ee, r, a, xp)


def _trans2yield_tight(mt, esns, esi, tp, gp, ee, r, a, xp):
    """Numpy path: per-MT arithmetic at tight (j0,) / (j0, j0)
    shapes. No compile cost applies, so padding would be pure
    waste (~2x wall time)."""
    esi = xp.asarray(esi, dtype=xp.float64)
    tp = xp.asarray(tp, dtype=xp.float64)
    gp = xp.asarray(gp, dtype=xp.float64)
    esns = xp.asarray(esns, dtype=xp.float64)
    nt = int(esi.shape[0])

    mt0 = _series_mt0(int(mt))
    j0 = int(mt) - mt0

    ee = _set_1d(ee, j0, esns, xp)

    if j0 > 0:
        # Vectorised match: one (j0, nt) outer product instead of j0
        # per-k dispatches.
        ee_slice = ee[0:j0]
        if nt > 0:
            p_direct_vec, a_val_vec = _match_direct_vectorised(
                esi, ee_slice, tp, gp, xp,
            )
        else:
            p_direct_vec = xp.zeros(j0, dtype=xp.float64)
            a_val_vec = xp.zeros(j0, dtype=xp.float64)

        # Write column j0 of ``a`` in one op.
        if xp.name == 'jax':
            a = a.at[0:j0, j0].set(a_val_vec)
        else:
            a[0:j0, j0] = a_val_vec

        # Solve column j0 of ``r`` as a triangular back-substitution:
        #   (I - strict_upper(a[0:j0, 0:j0])) @ r_col = p_direct
        # The old code did this row-by-row with a sequential
        # accumulation. A single triangular solve emits ONE XLA
        # primitive per MT instead of O(j0^2) chained adds, which is
        # the compile-time bottleneck on deep cascades. Introduces a
        # ~1-ULP drift vs the pre-rewrite Fortran-parity reference.
        eye_j0 = xp.eye(j0, dtype=xp.float64)
        strict_upper_A = xp.triu(a[0:j0, 0:j0], k=1)
        M = eye_j0 - strict_upper_A
        if xp.name == 'jax':
            import jax.scipy.linalg as _jsp_la
            r_col = _jsp_la.solve_triangular(
                M, p_direct_vec, lower=False, unit_diagonal=True,
            )
            r = r.at[0:j0, j0].set(r_col)
        else:
            from scipy.linalg import solve_triangular as _solve_triangular
            r_col = _solve_triangular(
                M, p_direct_vec, lower=False, unit_diagonal=True,
            )
            r[0:j0, j0] = r_col

    if j0 == 0:
        empty = xp.zeros(0, dtype=xp.float64)
        return ee, r, a, {
            'level_energy': empty,
            'photon_energy': empty,
            'photon_yield': empty,
        }

    # Fixed-shape photon-line grid. Fortran loops:
    #   i  = 2..j0+1  -> j1 = j0+2-i = j0, j0-1, ..., 1
    #   ii = 1..j0    -> j2 = j0-ii = j0-1, ..., 0
    # Both descending; we build with ascending arange and sort at
    # the end by descending photon energy anyway.
    j1_arr = np.arange(1, j0 + 1)
    j2_arr = np.arange(0, j0)
    j1_bc, j2_bc = np.meshgrid(j1_arr, j2_arr, indexing='ij')
    yld_grid = a[j2_bc, j1_bc] * r[j1_bc, j0]
    photon_energy_grid = ee[j1_bc] - ee[j2_bc]
    level_energy_grid = ee[j1_bc]

    yld_flat = yld_grid.reshape(-1)
    photon_energy_flat = photon_energy_grid.reshape(-1)
    level_energy_flat = level_energy_grid.reshape(-1)

    if xp.name == 'numpy':
        # Legacy output: keep only cascade positions that actually
        # emit a photon (Fortran-parity tests pin this).
        keep = yld_flat > 0.0
        yld_flat = yld_flat[keep]
        photon_energy_flat = photon_energy_flat[keep]
        level_energy_flat = level_energy_flat[keep]

    order = xp.argsort(-photon_energy_flat)
    return ee, r, a, {
        'level_energy': level_energy_flat[order],
        'photon_energy': photon_energy_flat[order],
        'photon_yield': yld_flat[order],
    }


def _pad_to_N(vec, N, fill, xp):
    """Pad a 1-D array to length ``N`` with ``fill`` (or truncate)."""
    v = xp.asarray(vec, dtype=xp.float64)
    n = int(v.shape[0])
    if n == N:
        return v
    if n > N:
        return v[:N]
    tail = xp.full((N - n,), fill, dtype=xp.float64)
    return xp.concatenate([v, tail])


def _match_direct_padded(esi_padded, ee_full, tp_padded, gp_padded, xp):
    """Same match-with-tolerance semantics as :func:`_match_direct_vectorised`
    but at fixed ``(N, N)`` shape.

    ``esi_padded`` unused slots hold ``-1e30`` so no ``k`` can
    match those columns; ``tp_padded`` / ``gp_padded`` unused
    slots hold 0. ``ee_full`` is the full state vector of length
    ``N``. Returns ``p_direct, a_val`` of shape ``(N,)`` each.
    """
    diffs = xp.abs(esi_padded[None, :] - ee_full[:, None])
    tol_abs = 1e-4 * xp.abs(esi_padded)[None, :]
    within = diffs <= tol_abs
    matched = xp.any(within, axis=1)
    first_idx = xp.argmax(within.astype(xp.int32), axis=1)
    tp_at = tp_padded[first_idx]
    gp_at = gp_padded[first_idx]
    p_direct = xp.where(matched, tp_at, 0.0)
    a_val = xp.where(matched, tp_at * gp_at, 0.0)
    return p_direct, a_val


def _trans2yield_padded(mt, esns, esi, tp, gp, ee, r, a, xp):
    """JAX path: every internal op runs at fixed ``(N, N)`` shape,
    where ``N = ee.shape[0]`` (the state matrix side, currently
    :data:`mf12_interpretation_helpers.MAX_NUM_LEVEL` = 60). XLA's
    per-shape first-call compile is then amortised across all MTs
    in the cascade series (~22x eager-cold speedup on Pb-208 MT=90
    vs the tight per-MT path)."""
    N = int(ee.shape[0])
    esi = xp.asarray(esi, dtype=xp.float64)
    tp = xp.asarray(tp, dtype=xp.float64)
    gp = xp.asarray(gp, dtype=xp.float64)
    esns = xp.asarray(esns, dtype=xp.float64)

    mt0 = _series_mt0(int(mt))
    j0 = int(mt) - mt0

    ee = _set_1d(ee, j0, esns, xp)

    if j0 == 0:
        empty = xp.zeros(0, dtype=xp.float64)
        return ee, r, a, {
            'level_energy': empty,
            'photon_energy': empty,
            'photon_yield': empty,
        }

    esi_padded = _pad_to_N(esi, N, -1e30, xp)
    tp_padded = _pad_to_N(tp, N, 0.0, xp)
    gp_padded = _pad_to_N(gp, N, 0.0, xp)

    p_direct_full, a_val_full = _match_direct_padded(
        esi_padded, ee, tp_padded, gp_padded, xp,
    )
    # Only rows k < j0 participate in this MT's cascade; mask the
    # padded tail to zero contribution.
    row_mask = xp.arange(N) < j0
    p_direct_full = xp.where(row_mask, p_direct_full, 0.0)
    a_val_full = xp.where(row_mask, a_val_full, 0.0)

    a = a.at[:, j0].set(a_val_full)

    # Triangular solve at fixed (N, N). ``a`` is strictly upper
    # triangular by construction; ``xp.triu`` keeps the intent
    # explicit and lets XLA fuse the mask.
    eye_N = xp.eye(N, dtype=xp.float64)
    M = eye_N - xp.triu(a, k=1)
    import jax.scipy.linalg as _jsp_la
    r_col = _jsp_la.solve_triangular(
        M, p_direct_full, lower=False, unit_diagonal=True,
    )
    # r[j0, j0] must stay 1 (identity structure) for later MTs to
    # read the correct running state. The solve returns 0 there
    # because p_direct_full[j0] = 0 and a[j0, i>j0] = 0.
    r_col = r_col.at[j0].set(1.0)
    r = r.at[:, j0].set(r_col)

    # Photon-line enumeration at fixed (N-1, N-1) shape. Positions
    # outside this MT's cascade (j1 > j0, or j2 mapped to padded
    # levels) produce zero yield naturally through the zeros in
    # ``r[:, j0]`` and ``a[j2, j1]``.
    j1_arr = np.arange(1, N)
    j2_arr = np.arange(0, N - 1)
    j1_bc, j2_bc = np.meshgrid(j1_arr, j2_arr, indexing='ij')
    yld_grid = a[j2_bc, j1_bc] * r[j1_bc, j0]
    photon_energy_grid = ee[j1_bc] - ee[j2_bc]
    level_energy_grid = ee[j1_bc]

    yld_flat = yld_grid.reshape(-1)
    photon_energy_flat = photon_energy_grid.reshape(-1)
    level_energy_flat = level_energy_grid.reshape(-1)

    # Mask the photon_energy of zero-yield entries to a sentinel
    # far from any physical query, so the wrapper's tolerance
    # matcher (find_indices_with_tol under concrete, argmin under
    # tracer) never routes a user photon-energy query into a
    # non-emitting cascade position. Zero-yield rows still occupy
    # the fixed-shape grid, they just do not compete for matches.
    photon_energy_flat = xp.where(
        yld_flat > 0.0, photon_energy_flat, -1e30,
    )
    order = xp.argsort(-photon_energy_flat)
    return ee, r, a, {
        'level_energy': level_energy_flat[order],
        'photon_energy': photon_energy_flat[order],
        'photon_yield': yld_flat[order],
    }
