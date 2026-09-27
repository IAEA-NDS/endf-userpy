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


def _match_direct(esi, ee_slice, tp, gp, xp):
    """Vectorised level-matching for every ``k`` simultaneously.

    Same two-branch tolerance semantics as the Fortran, computed
    with a single outer product of shape ``(len(ee_slice), len(esi))``.
    Under the numpy path ``ee_slice`` has length ``j0`` and ``esi``
    has the section's natural ``nt``; under the jax path all four
    inputs are pre-padded to a fixed ``(N, N)``.

    Returns ``(p_direct, a_val)``, each shaped like ``ee_slice``.
    """
    diffs = xp.abs(esi[None, :] - ee_slice[:, None])
    # Tolerance rule matches the Fortran two-branch logic:
    #   esi=0 and eek=0  -> diff=0, tol=0, within=True.
    #   esi=0 and eek!=0 -> diff=|eek|, tol=0, within=False.
    #   esi!=0 and eek=0 -> diff=|esi|, tol=1e-4*|esi|, within=False.
    #   esi!=0 and eek!=0-> |esi-eek| <= 1e-4*|esi|.
    tol_abs = 1e-4 * xp.abs(esi)[None, :]
    within = diffs <= tol_abs
    matched = xp.any(within, axis=1)
    first_idx = xp.argmax(within.astype(xp.int32), axis=1)
    tp_at = tp[first_idx]
    gp_at = gp[first_idx]
    p_direct = xp.where(matched, tp_at, 0.0)
    a_val = xp.where(matched, tp_at * gp_at, 0.0)
    return p_direct, a_val


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


def trans2yield(mt: int, esns, esi, tp, gp, ee, r, a,
                maxnk: int = 5000, xp=None):
    """Convert one section's transition probabilities into photon
    yields, and return updated cascade state.

    Single algorithm, two backend-driven parametrisations:

    - ``xp.name == 'numpy'``: the local shape ``N`` is ``j0`` so
      every matrix / vector op runs at tight per-MT extent. No
      XLA compile applies, so padding would be pure arithmetic
      waste. The zero-yield photon lines are filtered out to keep
      the Fortran-parity output length contract.
    - jax backends: ``N = ee.shape[0]`` (the state matrix side,
      currently :data:`mf12_interpretation_helpers.MAX_NUM_LEVEL`
      = 60). Every internal op runs at fixed shape so XLA pays a
      first-call compile once per shape and reuses it across all
      MTs in the cascade series (~29x eager-cold speedup on
      Pb-208 MT=90 vs the tight per-MT path). Zero-yield lines
      stay in place with a sentinel ``photon_energy = -1e30`` so
      the wrapper's tolerance matcher ignores them.

    Both paths return ``(ee, r, a, result_dict)`` with the same
    physical content. ``maxnk`` is unused (kept for API parity).

    Parameters
    ----------
    mt : int. MT number of the discrete inelastic reaction.
    esns : float or 0-d array. Excited-state energy ``ES``. Tracer-safe.
    esi : (nt,) lower-level energies. Tracer-safe.
    tp, gp : (nt,) direct-transition and photon-branching factors.
        Tracer-safe.
    ee, r, a : running cascade state from :func:`init_trans2yield`.
    xp : array-namespace adapter; ``None`` defaults to numpy.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    del maxnk

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

    # Local work extent: tight (j0) for numpy, padded (60) for jax.
    is_padded = xp.name != 'numpy'
    N = int(ee.shape[0]) if is_padded else j0

    # State views. Under padded ``N`` equals ``ee.shape[0]``, so a
    # ``[0:N]`` slice is a redundant XLA op that hurts jit steady-
    # state (measured 3x slower before this shortcut). Bypass it.
    ee_view = ee if is_padded else ee[0:N]
    a_view = a if is_padded else a[0:N, 0:N]

    # Under padded, extend esi/tp/gp to length N so the match op
    # can run at fixed shape; sentinels ensure no k matches a
    # padded slot. Under tight (N = j0) the arrays are already the
    # right length (or shorter, when the section has fewer direct
    # transitions than levels), so skip the pad to save the
    # per-MT Python overhead.
    if is_padded:
        esi_v = _pad_to_N(esi, N, -1e30, xp)
        tp_v = _pad_to_N(tp, N, 0.0, xp)
        gp_v = _pad_to_N(gp, N, 0.0, xp)
    else:
        esi_v, tp_v, gp_v = esi, tp, gp

    p_direct, a_val = _match_direct(esi_v, ee_view, tp_v, gp_v, xp)

    if is_padded:
        # Only rows k < j0 belong to this MT's cascade.
        row_mask = xp.arange(N) < j0
        p_direct = xp.where(row_mask, p_direct, 0.0)
        a_val = xp.where(row_mask, a_val, 0.0)

    if xp.name == 'jax':
        a = a.at[:, j0].set(a_val) if is_padded else a.at[0:N, j0].set(a_val)
    else:
        a[0:N, j0] = a_val

    # Triangular solve: (I - strict_upper(a_view)) @ r_col = p_direct.
    # Single XLA primitive per MT (was O(j0^2) chained adds pre
    # Lever 2) -> the compile-time bottleneck on deep cascades.
    # Introduces up to ~1 ULP drift vs the Fortran sequential
    # accumulation. Re-read ``a_view`` after the column write.
    a_view = a if is_padded else a[0:N, 0:N]
    eye_N = xp.eye(N, dtype=xp.float64)
    M = eye_N - xp.triu(a_view, k=1)
    if xp.name == 'jax':
        import jax.scipy.linalg as _jsp_la
        r_col = _jsp_la.solve_triangular(
            M, p_direct, lower=False, unit_diagonal=True,
        )
        if is_padded:
            # r[j0, j0] must stay 1 for later MTs to read the identity;
            # the solve returns 0 there because p_direct[j0] = 0.
            r_col = r_col.at[j0].set(1.0)
            r = r.at[:, j0].set(r_col)
        else:
            r = r.at[0:N, j0].set(r_col)
    else:
        from scipy.linalg import solve_triangular as _solve_triangular
        r_col = _solve_triangular(
            M, p_direct, lower=False, unit_diagonal=True,
        )
        # Under tight N == j0, r[0:N, j0] never touches r[j0, j0],
        # so the identity restore is unnecessary.
        r[0:N, j0] = r_col

    # Photon-line enumeration. Tight uses (j0, j0) so j1 covers
    # 1..j0 inclusive. Padded uses (N-1, N-1) so j1 covers 1..N-1
    # (avoids the ee[N] out-of-range that j1=N would introduce).
    N_enum = j0 if not is_padded else N - 1
    j1_arr = np.arange(1, N_enum + 1)
    j2_arr = np.arange(0, N_enum)
    j1_bc, j2_bc = np.meshgrid(j1_arr, j2_arr, indexing='ij')
    yld_grid = a[j2_bc, j1_bc] * r[j1_bc, j0]
    photon_energy_grid = ee[j1_bc] - ee[j2_bc]
    level_energy_grid = ee[j1_bc]

    yld_flat = yld_grid.reshape(-1)
    photon_energy_flat = photon_energy_grid.reshape(-1)
    level_energy_flat = level_energy_grid.reshape(-1)

    if is_padded:
        # Sentinel-mask zero-yield entries so the wrapper's
        # tolerance matcher never routes queries to non-emitting
        # cascade positions. Fixed shape preserved.
        photon_energy_flat = xp.where(
            yld_flat > 0.0, photon_energy_flat, -1e30,
        )
    else:
        # Fortran-parity contract: filter zero-yield rows out of
        # the output. Variable length is fine here (numpy path).
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
