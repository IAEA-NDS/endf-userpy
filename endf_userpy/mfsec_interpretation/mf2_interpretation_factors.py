"""MF2 resonance-formalism factors: penetration P_L, shift S_L,
hard-sphere phase phi_L.

Closed-form expressions from ENDF-6 Formats Manual D.1.3.3 for
L = 0..5, then a Newton recurrence for L >= 6. All routines are
written against the :mod:`endf_userpy.primitives.array_ns` adapter
so numpy and JAX share one implementation; a numba backend plugs
in the same way.

Shared by every R-matrix formalism (SLBW / MLBW / Reich-Moore /
RML) as auxiliary computations; kept in a dedicated module so the
top-level ``mf2_interpretation_<formalism>.py`` files stay focused
on their own physics.

Design note: the JAX prototype used ``lax.switch`` per element for
L; the numpy backend can't afford per-element Python switching but
also doesn't need it -- we compute all six low-L branches with
vectorised arithmetic and select by L, then take one Newton step
per L above 5. Same code path for both backends via the adapter's
``select`` / ``scan``.
"""
from __future__ import annotations


# The closed-form penetration and shift factors P_L / S_L for L=0..5.
# `r2 = rho * rho`. Formulas (per ENDF-6 D.1.3.3, verified against
# NJOY / SAMMY references):
#
#   L=0:  P = rho                    S = 0
#   L=1:  P = rho^3 / (1 + r2)       S = -1 / (1 + r2)
#   L=2:  P = rho^5 / (9 + 3 r2 + r4)
#         S = -(18 + 3 r2) / (9 + 3 r2 + r4)
#   L=3:  P = rho^7 / D3             S = -(675 + 90 r2 + 6 r4) / D3
#         D3 = 225 + 45 r2 + 6 r4 + r6
#   L=4, 5: analogous, in `_low_L_pnt_shf`.

_HIGH_L_START = 6   # First L whose factors need the Newton recurrence.


def _pnt_shf_low_L(rho, r2, L, xp):
    """Closed-form ``(P_L, S_L)`` for one ``L`` in 0..5 (Python int)."""
    if L == 0:
        return rho, xp.zeros_like(rho)
    if L == 1:
        d1 = 1.0 + r2
        return rho * r2 / d1, -1.0 / d1
    if L == 2:
        d2 = 9.0 + r2 * (3.0 + r2)
        return rho * r2 ** 2 / d2, -(18.0 + 3.0 * r2) / d2
    if L == 3:
        d3 = 225.0 + r2 * (45.0 + r2 * (6.0 + r2))
        return (rho * r2 ** 3 / d3,
                -(675.0 + r2 * (90.0 + 6.0 * r2)) / d3)
    if L == 4:
        d4 = 11025.0 + r2 * (1575.0 + r2 * (135.0 + r2 * (10.0 + r2)))
        return (rho * r2 ** 4 / d4,
                -(44100.0 + r2 * (4725.0 + r2 * (270.0 + 10.0 * r2))) / d4)
    d5 = 893025.0 + r2 * (
        99225.0 + r2 * (6300.0 + r2 * (315.0 + r2 * (15.0 + r2)))
    )
    return rho * r2 ** 5 / d5, -(
        4465125.0
        + r2 * (396900.0 + r2 * (18900.0 + r2 * (630.0 + 15.0 * r2)))
    ) / d5


def low_L_pnt_shf(rho, xp):
    """Compute (P_L, S_L) for L=0..5 elementwise, stacked on axis 0.

    Returns two arrays each of shape (6, *rho.shape). Later code
    picks the row for the desired L per element via `xp.take`
    (numpy) / advanced indexing / `xp.where`.
    """
    r2 = rho * rho
    ps = [_pnt_shf_low_L(rho, r2, L, xp) for L in range(6)]
    p = xp.stack([q[0] for q in ps], axis=0)
    s = xp.stack([q[1] for q in ps], axis=0)
    return p, s


def _phase_low_L(rho, r2, L, xp):
    """Closed-form hard-sphere phase for one ``L`` in 0..5."""
    if L == 0:
        return rho
    if L == 1:
        return rho - xp.arctan(rho)
    if L == 2:
        return rho - xp.arctan2(3.0 * rho, 3.0 - r2)
    if L == 3:
        return rho - xp.arctan2(rho * (15.0 - r2), 15.0 - 6.0 * r2)
    if L == 4:
        return rho - xp.arctan2(rho * (105.0 - 10.0 * r2),
                                105.0 - r2 * (45.0 - r2))
    return rho - xp.arctan2(
        rho * (945.0 - r2 * (105.0 - r2)),
        945.0 - r2 * (420.0 - 15.0 * r2),
    )


def low_L_phase(rho, xp):
    """Hard-sphere phase phi_L for L=0..5, stacked on axis 0.

    Convention: phi_L(rho) = arctan(P_L / (rho - S_L)); the
    equivalent closed forms (per NJOY reconr):

      L=0:  rho
      L=1:  rho - atan(rho)
      L=2:  rho - atan2(3 rho, 3 - r2)
      L=3:  rho - atan2(rho (15 - r2), 15 - 6 r2)
      L=4:  rho - atan2(rho (105 - 10 r2), 105 - 45 r2 + r4)
      L=5:  rho - atan2(rho (945 - 105 r2 + r4), 945 - 420 r2 + 15 r4)
    """
    r2 = rho * rho
    return xp.stack([_phase_low_L(rho, r2, L, xp) for L in range(6)], axis=0)


def _static_l(L):
    """``L`` as a Python int when it is a concrete non-negative scalar
    (Python / numpy int, 0-d numpy or concrete JAX array); ``None``
    for arrays and tracers, which take the vectorised path."""
    if isinstance(L, bool):
        return None
    if getattr(L, 'shape', ()) != ():
        return None
    try:
        v = int(L)
    except Exception:          # JAX tracer, or not integral
        return None
    return v if v >= 0 else None


def newton_step_pnt_shf(p_prev, s_prev, r2, L):
    """One Newton-recurrence step: (P_L, S_L) from (P_{L-1}, S_{L-1}).

    Closed form (per ENDF-6 manual):

        sdif  = L - S_{L-1}
        ratio = r2 / (sdif^2 + P_{L-1}^2)
        P_L   = ratio * P_{L-1}
        S_L   = ratio * sdif - L

    Same expression works for both backends since it's pure arithmetic.
    Callers loop this step externally (numpy: Python loop; JAX: scan)
    for L values above 5.
    """
    sdif = L - s_prev
    ratio = r2 / (sdif * sdif + p_prev * p_prev)
    return ratio * p_prev, ratio * sdif - L


def newton_step_phase(ph_prev, p_prev, s_prev, r2, L, xp):
    """Newton step for hard-sphere phase, running alongside the
    (P, S) recurrence:

        ph_L = ph_{L-1} - atan2(P_{L-1}, L - S_{L-1})

    Returns ``(ph_L, P_L, S_L)`` in the same order the caller
    typically threads through the scan.
    """
    sdif = L - s_prev
    ratio = r2 / (sdif * sdif + p_prev * p_prev)
    ph_new = ph_prev - xp.arctan2(p_prev, sdif)
    return ph_new, ratio * p_prev, ratio * sdif - L


def pnt_shf(rho, L, xp, nl_max: int = 8):
    """Vectorised P_L(rho), S_L(rho) for L an integer array.

    ``rho`` and ``L`` are broadcast-compatible; the return has the
    common broadcast shape. For any element with ``L >= 6``, the
    Newton recurrence runs from L=5 up to that element's L; we run
    the recurrence to ``nl_max - 1`` and pick per-element.
    """
    rho = xp.asarray(rho, dtype=xp.float64)
    L_static = _static_l(L)
    if L_static is not None:
        # Scalar L (every per-group call in RM / RML / URR): evaluate
        # only that L's closed form, then only the Newton steps it
        # needs. Same values as the masked selection below, which
        # adds exact zeros to the selected branch.
        r2 = rho * rho
        p, s = _pnt_shf_low_L(rho, r2, min(L_static, 5), xp)
        for LL in range(_HIGH_L_START, min(L_static, nl_max - 1) + 1):
            p, s = newton_step_pnt_shf(p, s, r2, LL)
        return p, s
    L = xp.asarray(L, dtype=xp.int32)

    p_low, s_low = low_L_pnt_shf(rho, xp)  # (6, *rho.shape)

    # Pick the low-L branch per element (clip L to [0, 5]).
    L_clip = xp.clip(L, 0, 5)
    idx = xp.arange(6).reshape((6,) + (1,) * (rho.ndim if hasattr(rho, 'ndim') else 0))
    mask = idx == L_clip
    p = xp.sum(xp.where(mask, p_low, 0.0), axis=0)
    s = xp.sum(xp.where(mask, s_low, 0.0), axis=0)

    if nl_max <= _HIGH_L_START:
        return p, s

    # Newton recurrence for L=6, 7, ..., nl_max-1. At each step,
    # update only elements whose target L >= step L.
    r2 = rho * rho
    p_prev, s_prev = p_low[5], s_low[5]
    for LL in range(_HIGH_L_START, nl_max):
        p_new, s_new = newton_step_pnt_shf(p_prev, s_prev, r2, LL)
        update = L >= LL
        p = xp.where(update, p_new, p)
        s = xp.where(update, s_new, s)
        p_prev, s_prev = p_new, s_new

    return p, s


def phase(rho, L, xp, nl_max: int = 8):
    """Vectorised phi_L(rho) for L an integer array."""
    rho = xp.asarray(rho, dtype=xp.float64)
    L_static = _static_l(L)
    if L_static is not None:
        # Scalar-L fast path, see :func:`pnt_shf`.
        r2 = rho * rho
        ph = _phase_low_L(rho, r2, min(L_static, 5), xp)
        if L_static >= _HIGH_L_START and nl_max > _HIGH_L_START:
            p, s = _pnt_shf_low_L(rho, r2, 5, xp)
            for LL in range(_HIGH_L_START, min(L_static, nl_max - 1) + 1):
                ph, p, s = newton_step_phase(ph, p, s, r2, LL, xp)
        return ph
    L = xp.asarray(L, dtype=xp.int32)

    ph_low = low_L_phase(rho, xp)  # (6, *rho.shape)
    L_clip = xp.clip(L, 0, 5)
    idx = xp.arange(6).reshape((6,) + (1,) * (rho.ndim if hasattr(rho, 'ndim') else 0))
    mask = idx == L_clip
    ph = xp.sum(xp.where(mask, ph_low, 0.0), axis=0)

    if nl_max <= _HIGH_L_START:
        return ph

    p_prev, s_prev = low_L_pnt_shf(rho, xp)
    p_prev = p_prev[5]
    s_prev = s_prev[5]
    ph_prev = ph_low[5]
    r2 = rho * rho
    for LL in range(_HIGH_L_START, nl_max):
        ph_new, p_new, s_new = newton_step_phase(ph_prev, p_prev, s_prev, r2, LL, xp)
        update = L >= LL
        ph = xp.where(update, ph_new, ph)
        ph_prev, p_prev, s_prev = ph_new, p_new, s_new

    return ph
