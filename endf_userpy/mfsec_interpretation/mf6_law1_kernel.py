"""Backend-agnostic (numpy / JAX) MF6 LAW=1 continuum
reconstruction kernel.

Consumes an :class:`~mf6_law1_preproc.MF6Law1Data` dataclass and
returns ``f_con(E, E', mu)`` on the caller's query grid, with all
arithmetic vectorised over the outer ``(E, E', mu)`` product and
routed through the ``xp`` adapter. JAX tracers stored in
``data.b_panels`` propagate through to the output, so
``jax.grad`` reaches back to file-stored angular parameters
(issue #154 pattern applied to LAW=1 continuum).

The panel loop stays Python-side (small ``n_panels``, data-known
count). Inside a panel pair the ``(nep1 <= nd1)`` /
``(nep2 <= nd2)`` regime branching is static and also Python-side.
Everything else is vectorised.

Scalar helpers still used verbatim from :mod:`mf6_law1_helpers`
for the panel-static Python arithmetic (Kalbach-Mann ``bachaa``
isotope table + semi-empirical constants; the reconstructed value
of ``a = bachaa(...)`` becomes an array via broadcast). The
existing scalar reconstruction path
(:func:`mf6_interpretation_subsecs.get_dist2d_from_subsec_law1`)
is preserved as-is via the composition-based route now going
through :func:`reconstruct` here.
"""
from __future__ import annotations

import numpy as np

from ..primitives import array_ns
from ..primitives.helpers import find_interval


_MF6CM_D2_MIN = 1.0e-38
_MF6CM_C_MIN = 1.0e-19
_LOG_SMALL = 1.0e-38


# ---- Broadcasted core primitives ---------------------------------


def _mf6lab2cm_bc(awr, awi, awp, lct, e, ep, u, xp):
    """Vectorised forward LAB->CM map. Inputs broadcastable arrays
    (or scalars); returns ``(tp, w, dinv)`` arrays of the broadcast
    shape. Identity branch (LCT=1 always; LCT=3 with AWP>=4)
    handled Python-side by the caller since it depends only on
    static ints.
    """
    c0 = float(np.sqrt(awi * awp) / (awi + awr))
    # ep <= 0 signals "no physical outgoing energy". Substitute a
    # tiny positive value for the sqrt/division and then mask via
    # dinv=0 in the where.
    ep_safe = xp.where(ep > 0.0, ep, _MF6CM_D2_MIN)
    c = c0 * xp.sqrt(e / ep_safe)
    d2 = 1.0 + c * c - 2.0 * c * u
    d2_safe = xp.where(d2 < _MF6CM_D2_MIN, _MF6CM_D2_MIN, d2)
    c_safe = xp.where(d2 < _MF6CM_D2_MIN, u - _MF6CM_C_MIN, c)
    tp = ep_safe * d2_safe
    dinv = 1.0 / xp.sqrt(d2_safe)
    w = dinv * (u - c_safe)
    w = xp.where(w > 1.0, 1.0, w)
    w = xp.where(w < -1.0, -1.0, w)
    dinv = xp.where(ep > 0.0, dinv, 0.0)
    return tp, w, dinv


def _yintp_bc(law, x1, y1, x2, y2, x, xp):
    """Vectorised 2-point interpolation matching Fortran ``yintp``
    with log-INT small-value guards. All args broadcastable.
    ``law`` is a static Python int.
    """
    if law == 1:
        # Histogram / constant: y1 broadcast to x's shape.
        return xp.asarray(y1) + xp.zeros_like(x)
    if law == 2:
        # Handle x==x1 or x==x2 to match Fortran identity branch
        num = (x - x1) * (y2 - y1)
        den = (x2 - x1)
        # den can be zero if x1 == x2 (degenerate); return y1
        den_safe = xp.where(den == 0.0, 1.0, den)
        result = y1 + num / den_safe
        result = xp.where(den == 0.0, y1, result)
        return result
    if law == 3:
        x1_safe = xp.where(x1 == 0.0, _LOG_SMALL, x1)
        log_ratio_den = xp.log(x2 / x1_safe)
        den_safe = xp.where(log_ratio_den == 0.0, 1.0, log_ratio_den)
        result = y1 + xp.log(x / x1_safe) * (y2 - y1) / den_safe
        return xp.where(log_ratio_den == 0.0, y1, result)
    if law == 4:
        y1_safe = xp.where(y1 == 0.0, _LOG_SMALL, y1)
        den = (x2 - x1)
        den_safe = xp.where(den == 0.0, 1.0, den)
        result = y1_safe * xp.exp((x - x1) * xp.log(y2 / y1_safe) / den_safe)
        return xp.where(den == 0.0, y1, result)
    if law == 5:
        x1_safe = xp.where(x1 == 0.0, _LOG_SMALL, x1)
        y1_safe = xp.where(y1 == 0.0, _LOG_SMALL, y1)
        log_x_ratio = xp.log(x2 / x1_safe)
        den_safe = xp.where(log_x_ratio == 0.0, 1.0, log_x_ratio)
        result = y1_safe * xp.exp(
            xp.log(x / x1_safe) * xp.log(y2 / y1_safe) / den_safe
        )
        return xp.where(log_x_ratio == 0.0, y1, result)
    raise TypeError(f'interpolation scheme (INT={law}) not implemented')


# ---- Kalbach-Mann vectorised ------------------------------------


def _bachaa_bc(zai, zap, zat, e_arr, ep_arr, xp):
    """Vectorised Kalbach-Mann ``a`` (endf6.f90:1501). ``zai``,
    ``zap``, ``zat`` are static ints so all the isotope-table and
    special-case corrections are precomputed as Python constants;
    the (e, ep) inputs vectorise.
    """
    # Static ints from the ZA arguments
    from .mf6_law1_helpers import _BACHAA_ISOTOPE_TABLE, _BACHAA_EPS, _BACHAA_EMC2, _BACHAA_EMEV
    iza1i = int(zai + _BACHAA_EPS)
    iza2 = int(zap + _BACHAA_EPS)
    izat = int(zat + _BACHAA_EPS)
    iza1 = iza1i if iza1i != 0 else 1
    iza = _BACHAA_ISOTOPE_TABLE.get(izat, izat)
    aa = iza % 1000
    if aa == 0:
        raise ValueError(f'bachaa: dominant isotope not known for ZA={iza}')
    za_i = iza // 1000
    ac = aa + (iza1 % 1000)
    zc = za_i + (iza1 // 1000)
    ab = ac - (iza2 % 1000)
    zb = zc - (iza2 // 1000)
    na_i = aa - za_i
    nb = ab - zb
    nc = ac - zc

    third = 1.0 / 3.0
    twoth = 2.0 / 3.0
    fourth = 4.0 / 3.0
    c1, c2, c3 = 15.68, -28.07, -18.56
    c4, c5, c6 = 33.22, -0.717, 1.211
    s2, s3, s4, s5 = 2.22, 8.48, 7.72, 28.3
    b1, b2, b3 = 0.04, 1.8e-6, 6.7e-7
    d1 = 9.3
    ea1, ea2 = 41.0, 130.0

    sa = (c1 * (ac - aa)
          + c2 * ((nc - zc) ** 2 / ac - (na_i - za_i) ** 2 / aa)
          + c3 * (ac ** twoth - aa ** twoth)
          + c4 * ((nc - zc) ** 2 / ac ** fourth
                  - (na_i - za_i) ** 2 / aa ** fourth)
          + c5 * (zc ** 2 / ac ** third - za_i ** 2 / aa ** third)
          + c6 * (zc ** 2 / ac - za_i ** 2 / aa))
    if iza1 == 1002:
        sa -= s2
    elif iza1 == 1003:
        sa -= s3
    elif iza1 == 2003:
        sa -= s4
    elif iza1 == 2004:
        sa -= s5

    sb = (c1 * (ac - ab)
          + c2 * ((nc - zc) ** 2 / ac - (nb - zb) ** 2 / ab)
          + c3 * (ac ** twoth - ab ** twoth)
          + c4 * ((nc - zc) ** 2 / ac ** fourth
                  - (nb - zb) ** 2 / ab ** fourth)
          + c5 * (zc ** 2 / ac ** third - zb ** 2 / ab ** third)
          + c6 * (zc ** 2 / ac - zb ** 2 / ab))
    if iza2 == 1002:
        sb -= s2
    elif iza2 == 1003:
        sb -= s3
    elif iza2 == 2003:
        sb -= s4
    elif iza2 == 2004:
        sb -= s5

    fa = 0.0 if iza1 == 2004 else 1.0
    if iza2 == 1:
        fb = 0.5
    elif iza2 == 2004:
        fb = 2.0
    else:
        fb = 1.0

    # Now vectorise on (e, ep). e and ep in eV; convert to MeV.
    e_mev = e_arr / _BACHAA_EMEV
    ep_mev = ep_arr / _BACHAA_EMEV
    ecm = aa * e_mev / ac
    ea = ecm + sa
    eb = ep_mev * ac / ab + sb
    x1_bc = xp.where(ea <= ea2, eb, ea2 * eb / ea)
    x3_bc = xp.where(ea <= ea1, eb, ea1 * eb / ea)
    bb = b1 * x1_bc + b2 * x1_bc ** 3 + b3 * fa * fb * x3_bc ** 4
    if iza1i == 0:
        # Incident photon extra factor
        fact_raw = xp.where(ep_mev == 0.0, d1, d1 / xp.sqrt(ep_mev))
        fact_lo = xp.where(fact_raw < 1.0, 1.0, fact_raw)
        fact = xp.where(fact_lo > 4.0, 4.0, fact_lo)
        bb = bb * xp.sqrt(e_mev / (2.0 * _BACHAA_EMC2)) * fact
    return bb


_KALBACH_AMIN = 1.0e-38


def _ykalbach_bc(zai, zap, zat, e_arr, ep_arr, u, a_row, na, xp):
    """Vectorised Kalbach-Mann angular distribution. ``a_row`` is
    the interpolated ``(na+1,)`` parameter vector broadcast over
    the query axes as an extra trailing coefficient dim: shape
    ``(..., na+1)``. ``na`` is a static Python int.
    """
    f0 = a_row[..., 0]
    if na == 0:
        return 0.5 * f0
    r = a_row[..., 1]
    if na == 1:
        a = _bachaa_bc(zai, zap, zat, e_arr, ep_arr, xp)
    else:
        # na == 2: a from evaluator
        a = a_row[..., 2]

    # Guard |a| ~ 0 to fall back to isotropic distribution.
    a_safe = xp.where(xp.abs(a) < _KALBACH_AMIN, 1.0, a)
    # Numerically stable form of
    #   f = 0.5 * a * f0 * (cosh(au) + r sinh(au)) / sinh(a)
    # Rewrite as:
    #   f = 0.5 * a * f0 * ((1+r) e^(a(u-1)) + (1-r) e^(-a(u+1)))
    #                     / (1 - e^(-2 a))
    # Valid for both signs of a; ratio stays finite even when
    # |a| is large enough to overflow cosh(a)/sinh(a) directly
    # (typical for high-energy Kalbach evaluations).
    exp_pos = xp.exp(a_safe * (u - 1.0))
    exp_neg = xp.exp(-a_safe * (u + 1.0))
    denom = 1.0 - xp.exp(-2.0 * xp.abs(a_safe))
    denom_safe = xp.where(denom == 0.0, 1.0, denom)
    numer = (1.0 + r) * exp_pos + (1.0 - r) * exp_neg
    # The (1 - e^{-2|a|}) denominator normalisation applies for
    # a > 0. For a < 0 the same algebra gives a sign flip that
    # cancels in the overall 0.5 * a * f0 prefactor when the
    # formula is written symmetrically as above.
    sign = xp.where(a_safe >= 0.0, 1.0, -1.0)
    result_reg = 0.5 * xp.abs(a_safe) * f0 * sign * numer / denom_safe
    return xp.where(xp.abs(a) < _KALBACH_AMIN, 0.5 * f0, result_reg)


# ---- Legendre eval (delegates to primitives) ---------------------


def _yleg_bc(a_row, mu, na, xp):
    """Vectorised Fortran ``yleg`` = ``sum_{L=0..na} (L+0.5) a_L
    P_L(mu)``. ``a_row`` has shape ``(..., na+1)`` and ``mu`` has
    shape ``(...)`` matching the leading axes of ``a_row``.

    Direct Bonnet recurrence; matches the scalar
    :func:`mf6_law1_helpers._legendre_eval_at_mu` and the
    Fortran ``yleg`` reference (endf6.f90:1394).
    """
    n = na + 1
    if n == 0:
        return xp.zeros_like(mu)
    # (L + 0.5) weights per ENDF Legendre convention
    L_weights = xp.arange(n, dtype=a_row.dtype) + 0.5
    weighted = a_row * L_weights
    P_prev = xp.ones_like(mu)                       # P_0
    total = weighted[..., 0] * P_prev
    if n == 1:
        return total
    P_curr = mu                                     # P_1
    total = total + weighted[..., 1] * P_curr
    for L in range(1, n - 1):
        P_next = ((2 * L + 1) * mu * P_curr - L * P_prev) / (L + 1)
        total = total + weighted[..., L + 1] * P_next
        P_prev = P_curr
        P_curr = P_next
    return total


# ---- Continuum single-panel evaluation ---------------------------


def _f6law1_con_panel_bc(data, p, e_scalar, tp, w, xp):
    """Single-panel continuum amplitude ``f6law1_con`` broadcast
    over the query axes (endf6.f90:874). ``tp`` and ``w`` have
    shape ``(...)`` (broadcastable). ``p`` is Python int; scalar
    panel constants come from ``data`` fields.

    Returns an ``(...,)`` array; zero where ``tp`` lies outside
    the panel's continuum Ep range.
    """
    nep = int(data.nep_arr[p])
    nd = int(data.nd_arr[p])
    na = int(data.na_arr[p])
    lang = int(data.lang)
    lep = int(data.lep)

    ep_panel_full = data.ep_panels[p]           # (max_nep,) numpy
    b_panel_full = data.b_panels[p]             # (max_nep, max_na+1) xp

    # Continuum portion: ep_panel[nd:nep] (0-indexed)
    if nep <= nd:
        # No continuum data on this panel
        return xp.zeros(tp.shape, dtype=b_panel_full.dtype)

    ep_cont = ep_panel_full[nd:nep]             # numpy
    ep_cont_xp = xp.asarray(ep_cont)

    # searchsorted for the bracket: idx_rel in [0, nep-nd] with
    # idx_rel = 0 meaning tp < ep_cont[0], nep-nd meaning tp > ep_cont[-1]
    idx_rel = xp.searchsorted(ep_cont_xp, tp, side='right')
    # Valid bracket exists iff 1 <= idx_rel <= nep-nd-1 (i.e. tp
    # falls strictly inside [ep_cont[0], ep_cont[-1]])
    # Fortran ihigh returns >0 only when x0 in (x[i0], x[n]].
    valid = (idx_rel > 0) & (idx_rel < (nep - nd))
    # Clamp so we can advance-index safely on both valid and
    # invalid positions; invalid ones are masked out at the end.
    idx_rel_clamped = xp.where(valid, idx_rel, 1)
    i2 = idx_rel_clamped + nd                   # abs index in full ep_panel
    i1 = i2 - 1

    ep1_val = xp.asarray(ep_panel_full)[i1]     # (...,)
    ep2_val = xp.asarray(ep_panel_full)[i2]

    if lang in (1, 2):
        nt = na + 1
        # Advanced-index the b_panel: b_panel_full[i1] has shape
        # (..., max_na+1). Slice to nt.
        a1 = b_panel_full[i1][..., :nt]         # (..., nt)
        a2 = b_panel_full[i2][..., :nt]
        # Per-coefficient outer Ep interp
        # Broadcast tp, ep1_val, ep2_val against the nt axis by
        # adding a trailing dim
        tp_bc = tp[..., None]
        ep1_bc = ep1_val[..., None]
        ep2_bc = ep2_val[..., None]
        a_interp = _yintp_bc(lep, ep1_bc, a1, ep2_bc, a2, tp_bc, xp)
        # (..., nt)
        if lang == 1:
            f = _yleg_bc(a_interp, w, na, xp)
        else:  # lang == 2 (Kalbach-Mann)
            f = _ykalbach_bc(
                data.zai, data.zap, data.za,
                xp.broadcast_to(xp.asarray(e_scalar, dtype=tp.dtype), tp.shape),
                tp, w, a_interp, na, xp,
            )
    elif 11 <= lang <= 15:
        # Tabulated (u, p(u)) pairs per Ep row.
        f01 = b_panel_full[i1, 0]                # (...,)
        f02 = b_panel_full[i2, 0]
        if na > 0:
            nmu_ = na // 2
            # Extract mu and y arrays from b_panel; layout per row:
            # [f0, u_1, y_1, u_2, y_2, ..., u_nmu, y_nmu]
            # Ep is indexed via i1 / i2.
            # Build (..., nmu) arrays
            u_idx = 1 + 2 * np.arange(nmu_)
            y_idx = 2 + 2 * np.arange(nmu_)
            u1 = b_panel_full[i1][..., u_idx]     # (..., nmu)
            u2 = b_panel_full[i2][..., u_idx]
            y1 = 2.0 * f01[..., None] * b_panel_full[i1][..., y_idx]
            y2 = 2.0 * f02[..., None] * b_panel_full[i2][..., y_idx]
            lmu = lang - 10
            # Unit-base interp over the two Ep panels + tabulated mu
            f = _unit_base_intp_two_panel_bc(
                ep1_val, u1, y1, nmu_, lmu,
                ep2_val, u2, y2, nmu_, lmu,
                lep, tp, w, xp,
            )
        else:
            # Isotropic: interp f0 over Ep then /2
            f0 = _yintp_bc(lep, ep1_val, f01, ep2_val, f02, tp, xp)
            f = 0.5 * f0
    else:
        raise NotImplementedError(f'MF6 LAW=1 LANG={lang} not supported')

    return xp.where(valid, f, 0.0)


def _tab1_scalar_lin_bc(x_arr, y_arr, n, law, x0, xp):
    """Vectorised single-panel TAB1 interp with one INT law. ``x_arr``
    has shape ``(..., n)`` (n static), ``y_arr`` matching, ``x0``
    broadcastable. Returns ``(...,)`` array clamped at the ends.
    """
    # searchsorted along the last axis: broadcast row against x0.
    # We handle out-of-bound clamping by masking i to [1, n-1] and
    # comparing to the boundary values afterwards.
    # For simplicity here, iterate in Python since x_arr is only
    # (..., n) with small n; each row search reduces to one
    # `searchsorted` call. Because x_arr can differ per broadcast
    # slot (unit-base transformed), we do a per-element loop that
    # scipy-style broadcasting doesn't cleanly express. However,
    # for the current use case x_arr has shape (nmu,) shared across
    # all query points (Fortran unit_base_intp: x is scalar per
    # panel), so we can reduce to searchsorted on a 1D x_arr.
    # Simplify: assume x_arr is 1D (nmu,).
    idx = xp.searchsorted(x_arr, x0, side='right')
    idx = xp.where(idx < 1, 1, idx)
    idx = xp.where(idx > n - 1, n - 1, idx)
    x1 = x_arr[idx - 1]
    x2 = x_arr[idx]
    y1 = y_arr[idx - 1]
    y2 = y_arr[idx]
    # Clamp at ends: x0 <= x_arr[0] returns y_arr[0], x0 >= x_arr[-1] returns y_arr[-1]
    result = _yintp_bc(law, x1, y1, x2, y2, x0, xp)
    result = xp.where(x0 <= x_arr[0], y_arr[0], result)
    result = xp.where(x0 >= x_arr[n - 1], y_arr[n - 1], result)
    return result


def _unit_base_intp_two_panel_bc(y1, x1_arr, f1_arr, np1, lmu1,
                                   y2, x2_arr, f2_arr, np2, lmu2,
                                   inty, y0, x0, xp):
    """Vectorised two-panel unit-base interp (endf6.f90:2173).
    ``y1, y2`` are scalar-per-query (bracketing Ep values); ``x1_arr,
    f1_arr`` are (nmu,); ``x2_arr, f2_arr`` are (nmu,); ``y0, x0``
    broadcastable query axes.

    In the LAW=1 LANG=11..15 call site, ``y1, y2`` may be arrays of
    the (..., ) query shape (bracketing Ep for each query point).
    """
    law = inty % 10
    # Compute unit-base bounds and slope. `y1, y2` are broadcastable.
    x1low = x1_arr[0]      # scalar
    x1high = x1_arr[np1 - 1]
    x1range = x1high - x1low
    x2low = x2_arr[0]
    x2high = x2_arr[np2 - 1]
    x2range = x2high - x2low
    y2_minus_y1 = y2 - y1
    y2_minus_y1_safe = xp.where(y2_minus_y1 == 0.0, 1.0, y2_minus_y1)
    yslope = (y0 - y1) / y2_minus_y1_safe
    xlow = x1low + yslope * (x2low - x1low)
    xhigh = x1high + yslope * (x2high - x1high)
    xrange = xhigh - xlow
    xrange_safe = xp.where(xrange == 0.0, 1.0, xrange)
    xslope = (x0 - xlow) / xrange_safe
    x_at_p1 = x1low + xslope * x1range
    x_at_p2 = x2low + xslope * x2range
    f1x = _tab1_scalar_lin_bc(x1_arr, f1_arr, np1, lmu1, x_at_p1, xp) * (x1range / xrange_safe)
    f2x = _tab1_scalar_lin_bc(x2_arr, f2_arr, np2, lmu2, x_at_p2, xp) * (x2range / xrange_safe)
    result = _yintp_bc(law, y1, f1x, y2, f2x, y0, xp)
    # Fortran: return 0 outside [y1, y2]
    return xp.where((y0 < y1) | (y0 > y2), 0.0, result)


# ---- Continuum two-panel outer combination -----------------------


def _f6law1con_panel_pair_bc(data, panel_idx, lei, e_bc, tp_bc, w_bc, xp):
    """Two-panel continuum contribution for one E-panel pair
    (endf6.f90:793). All (e, tp, w) are broadcast arrays of the
    query shape. Returns the same-shape amplitude array; zero when
    neither panel has continuum data.
    """
    p1 = panel_idx
    p2 = panel_idx + 1
    e1 = float(data.ei_mesh[p1])
    e2 = float(data.ei_mesh[p2])
    nep1 = int(data.nep_arr[p1])
    nd1 = int(data.nd_arr[p1])
    nep2 = int(data.nep_arr[p2])
    nd2 = int(data.nd_arr[p2])
    law = int(lei) % 10

    p1_has_cont = (nep1 > nd1)
    p2_has_cont = (nep2 > nd2)
    if not p1_has_cont and not p2_has_cont:
        return xp.zeros(tp_bc.shape, dtype=data.b_panels.dtype)

    if not p1_has_cont:
        f1 = xp.zeros_like(tp_bc)
        f2 = _f6law1_con_panel_bc(data, p2, e2, tp_bc, w_bc, xp)
    elif not p2_has_cont:
        f1 = _f6law1_con_panel_bc(data, p1, e1, tp_bc, w_bc, xp)
        f2 = xp.zeros_like(tp_bc)
    else:
        # Both have continuum: unit-base transform on tp
        x1low = float(data.ep_panels[p1, nd1])
        x1high = float(data.ep_panels[p1, nep1 - 1])
        x1range = x1high - x1low
        x2low = float(data.ep_panels[p2, nd2])
        x2high = float(data.ep_panels[p2, nep2 - 1])
        x2range = x2high - x2low
        e2_minus_e1 = e2 - e1
        yslope = (e_bc - e1) / e2_minus_e1
        xlow = x1low + yslope * (x2low - x1low)
        xhigh = x1high + yslope * (x2high - x1high)
        xrange = xhigh - xlow
        xrange_safe = xp.where(xrange == 0.0, 1.0, xrange)
        xslope = (tp_bc - xlow) / xrange_safe
        tp_at_p1 = x1low + xslope * x1range
        tp_at_p2 = x2low + xslope * x2range
        f1 = _f6law1_con_panel_bc(data, p1, e1, tp_at_p1, w_bc, xp) * (x1range / xrange_safe)
        f2 = _f6law1_con_panel_bc(data, p2, e2, tp_at_p2, w_bc, xp) * (x2range / xrange_safe)

    # Outer E interp between the panel results
    e1_arr = xp.asarray(e1, dtype=e_bc.dtype)
    e2_arr = xp.asarray(e2, dtype=e_bc.dtype)
    return _yintp_bc(law, e1_arr, f1, e2_arr, f2, e_bc, xp)


# ---- Top-level reconstruction ------------------------------------


def reconstruct(data, energies_in, energies_out, angle_cosines_out,
                 to_lab, xp=None, panel_idx=None):
    """Backend-agnostic MF6 LAW=1 continuum reconstruction.

    Consumes an :class:`MF6Law1Data` dataclass (built via
    :func:`~mf6_law1_preproc.mf6_law1_data_from_endf_dict` from an
    ENDF-6 dict, or hand-built with tracer-replaced fields for
    autodiff), returns the continuum contribution
    ``f_con(E, E', mu)`` in the LAB frame on the caller's query
    grid.

    ``xp=None`` (default) resolves to numpy. Passing
    ``xp=array_ns.get_backend('jax')`` makes the arithmetic
    JAX-native; ``jax.grad`` propagates through the file-stored
    angular parameters in ``data.b_panels``, and also through
    ``angle_cosines_out`` and ``energies_out`` (analytic paths).

    ``panel_idx`` (Python int) is the autodiff entry point for
    ``jax.grad`` wrt ``energies_in``: with a static panel choice
    the section-wide ``find_interval`` / panel loop is skipped and
    ``energies_in`` flows through the kernel as a tracer. The
    caller must keep ``energies_in`` inside
    ``[ei_mesh[panel_idx], ei_mesh[panel_idx + 1]]`` -- the
    amplitude has physical C0 kinks at each panel knot.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    lct = data.lct if to_lab else 1
    if lct in (1, 2):
        eff_lct = lct
    elif lct == 3:
        eff_lct = 1 if data.awp > 4 else 2
    else:
        raise NotImplementedError(f'LCT={lct} not implemented')

    e_in = xp.asarray(energies_in, dtype=xp.float64)
    ep_out = xp.asarray(energies_out, dtype=xp.float64)
    mu = xp.asarray(angle_cosines_out, dtype=xp.float64)
    n_e = e_in.shape[0]
    n_ep = ep_out.shape[0]
    n_mu = mu.shape[0]

    if panel_idx is not None:
        # Single-panel autodiff path: skip find_interval and the
        # inside-mask branch. Everything is xp-native, so tracers
        # flow through energies_in, energies_out, angle_cosines_out
        # end-to-end.
        from ..primitives.helpers import convert_interp_repr as _cvt_p
        ei_interp_full = _cvt_p(
            np.asarray(data.int_arr), np.asarray(data.nbt_arr),
        )
        p = int(panel_idx)
        lei = int(ei_interp_full[p])
        e_bc = e_in[:, None, None]
        ep_bc = ep_out[None, :, None]
        mu_bc = mu[None, None, :]
        if eff_lct == 1 or (eff_lct == 3 and data.awp >= 4.0):
            tp_bc = xp.broadcast_to(ep_bc, (n_e, n_ep, n_mu))
            w_bc = xp.broadcast_to(mu_bc, (n_e, n_ep, n_mu))
            dinv_bc = xp.ones((n_e, n_ep, n_mu), dtype=tp_bc.dtype)
        else:
            e_full = xp.broadcast_to(e_bc, (n_e, n_ep, n_mu))
            ep_full = xp.broadcast_to(ep_bc, (n_e, n_ep, n_mu))
            mu_full = xp.broadcast_to(mu_bc, (n_e, n_ep, n_mu))
            tp_bc, w_bc, dinv_bc = _mf6lab2cm_bc(
                data.awr, data.awi, data.awp, eff_lct,
                e_full, ep_full, mu_full, xp,
            )
        e_bc_full = xp.broadcast_to(e_bc, (n_e, n_ep, n_mu))
        f = _f6law1con_panel_pair_bc(
            data, p, lei, e_bc_full, tp_bc, w_bc, xp,
        )
        return f * dinv_bc

    # Broadcast to (n_e, n_ep, n_mu)
    e_bc = e_in[:, None, None]
    ep_bc = ep_out[None, :, None]
    mu_bc = mu[None, None, :]

    # Vectorised LAB->CM. Identity branch handled Python-side.
    if eff_lct == 1 or (eff_lct == 3 and data.awp >= 4.0):
        # Identity map (LCT=1 or LCT=3-heavy)
        tp_bc = xp.broadcast_to(ep_bc, (n_e, n_ep, n_mu))
        w_bc = xp.broadcast_to(mu_bc, (n_e, n_ep, n_mu))
        dinv_bc = xp.ones((n_e, n_ep, n_mu), dtype=xp.float64)
    else:
        # LCT=2 or LCT=3-light: CM shift
        e_full = xp.broadcast_to(e_bc, (n_e, n_ep, n_mu))
        ep_full = xp.broadcast_to(ep_bc, (n_e, n_ep, n_mu))
        mu_full = xp.broadcast_to(mu_bc, (n_e, n_ep, n_mu))
        tp_bc, w_bc, dinv_bc = _mf6lab2cm_bc(
            data.awr, data.awi, data.awp, eff_lct,
            e_full, ep_full, mu_full, xp,
        )

    # Panel-by-panel continuum evaluation. Group query E by panel.
    ei_mesh_np = np.asarray(data.ei_mesh)
    # Build the full per-panel INT lookup (mirror the outer
    # int_arr / nbt_arr conversion).
    from ..primitives.helpers import convert_interp_repr as _cvt
    ei_interp_full = _cvt(np.asarray(data.int_arr), np.asarray(data.nbt_arr))
    e_in_np = np.asarray(e_in)
    # Only compute for E's that are inside the ei_mesh range.
    e_min = float(ei_mesh_np[0])
    e_max = float(ei_mesh_np[-1])
    inside_mask_np = (e_in_np >= e_min) & (e_in_np <= e_max)

    result = xp.zeros((n_e, n_ep, n_mu), dtype=tp_bc.dtype)
    if not inside_mask_np.any():
        return result * dinv_bc

    e_inside_np = e_in_np[inside_mask_np]
    idcs = find_interval(ei_mesh_np, e_inside_np)
    inside_positions = np.where(inside_mask_np)[0]

    for panel_idx in np.unique(idcs):
        row_mask = (idcs == panel_idx)
        rows_np = inside_positions[row_mask]        # positions in full n_e
        # Slice the broadcast arrays to the E-rows for this panel
        rows_xp = xp.asarray(rows_np)
        e_sub = xp.take(e_bc[:, 0, 0], rows_xp, axis=0)[:, None, None]
        tp_sub = xp.take(tp_bc, rows_xp, axis=0)
        w_sub = xp.take(w_bc, rows_xp, axis=0)
        e_sub_bc = xp.broadcast_to(
            e_sub, (rows_np.shape[0], n_ep, n_mu),
        )
        lei = int(ei_interp_full[panel_idx])
        f_sub = _f6law1con_panel_pair_bc(
            data, int(panel_idx), lei, e_sub_bc, tp_sub, w_sub, xp,
        )
        # Scatter back into result at these rows
        result = _scatter_rows(result, rows_np, f_sub, xp)

    return result * dinv_bc


def _scatter_rows(dest, rows_np, src, xp):
    """Write ``src`` into ``dest[rows]`` on the outer axis (0)."""
    if xp.name == 'jax':
        return dest.at[np.asarray(rows_np)].set(src)
    dest_np = np.asarray(dest).copy()
    dest_np[rows_np] = np.asarray(src)
    return xp.asarray(dest_np)
