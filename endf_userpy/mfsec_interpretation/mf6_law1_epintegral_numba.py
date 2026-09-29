"""Numba fast path for :func:`mf6_law1_epintegral._law1_spectrum_panel_pair`
specialised to the gamma-ejectile isotropic LEP-piecewise case.

Applicability gate (all must hold on the panel pair):

- ``xp`` is numpy,
- ``data.lang == 1`` and ``data.na_arr[p1] == data.na_arr[p2] == 0``
  (single-coefficient isotropic Legendre),
- ``data.lep in (1, 2)`` (histogram or lin-lin Ep interpolation),
- ``lei % 10 in (1, 2)`` (histogram or lin-lin outer Ein interpolation),
- both panels carry continuum (``nep > nd``),
- ``c0 == 0`` where ``c0 = sqrt(awi * awp) / (awi + awr)`` (i.e. the
  ejectile is massless: photons, ``awp == 0``),
- ``eff_lct in (1, 2)``.

Mathematical shortcut. When ``c0 == 0`` the LAB<->CM map is identity
(``tp == ep``, ``dinv == 1``), so the amplitude built by
:func:`mf6_law1_kernel._f6law1con_panel_pair_bc` does not depend on
``mu``. Combined with ``na == 0`` (which reduces the ``_yleg_bc``
polar expansion to a constant), the kink-aware polar-angle
Gauss-Legendre integration reduces analytically to

    spectrum(e, ep) = f_amp(e, ep) * int_0^pi sin(z) dz
                    ~= 2 * f_amp(e, ep)

exact to machine precision at ``n_gl=10`` (Gauss-Legendre of ``sin``
over ``[0, pi]``). One multiply per ``(e, ep)`` replaces an
``(n_sub, n_gl)`` accumulation that materialises tens of MB of
mostly-zero-contribution intermediates.

Fall-through is silent: when the gate does not match,
:func:`try_panel_pair` returns None so the numpy caller can use its
own path. Bit-identical to :func:`_law1_spectrum_panel_pair` modulo
floating-point round-off (max rel diff ~1e-17 on the target case).
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    prange = range


@njit(cache=True, parallel=True, fastmath=False)
def _numba_law1_gamma_na0_panel_pair(
    ep_cont1, b_cont1,
    ep_cont2, b_cont2,
    e1, e2,
    e_sub, ep_out,
    gl_x, gl_w,
    lep,
    lei_law,
):
    n_e = e_sub.shape[0]
    n_ep = ep_out.shape[0]
    n_gl = gl_x.shape[0]
    n1 = ep_cont1.shape[0]
    n2 = ep_cont2.shape[0]

    half_pi = np.pi / 2.0
    sin_integral = 0.0
    for k in range(n_gl):
        z = half_pi * (1.0 + gl_x[k])
        sin_integral += gl_w[k] * np.sin(z)
    sin_integral *= half_pi

    x1low = ep_cont1[0]
    x1high = ep_cont1[n1 - 1]
    x1range = x1high - x1low
    x2low = ep_cont2[0]
    x2high = ep_cont2[n2 - 1]
    x2range = x2high - x2low
    e2_minus_e1 = e2 - e1

    result = np.zeros((n_e, n_ep), dtype=np.float64)

    for ie in prange(n_e):
        e = e_sub[ie]
        yslope = (e - e1) / e2_minus_e1
        xlow = x1low + yslope * (x2low - x1low)
        xhigh = x1high + yslope * (x2high - x1high)
        xrange = xhigh - xlow
        xrange_safe = 1.0 if xrange == 0.0 else xrange
        r1 = x1range / xrange_safe
        r2 = x2range / xrange_safe

        for iep in range(n_ep):
            ep = ep_out[iep]
            xslope = (ep - xlow) / xrange_safe
            tp_at_p1 = x1low + xslope * x1range
            tp_at_p2 = x2low + xslope * x2range

            lo = 0
            hi = n1
            while lo < hi:
                mid = (lo + hi) >> 1
                if ep_cont1[mid] <= tp_at_p1:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo
            if idx <= 0 or idx >= n1:
                b1_at = 0.0
            else:
                x1a = ep_cont1[idx - 1]
                x1b = ep_cont1[idx]
                y1a = b_cont1[idx - 1]
                y1b = b_cont1[idx]
                if lep == 1:
                    b1_at = y1a
                else:
                    d = x1b - x1a
                    b1_at = y1a if d == 0.0 else y1a + (tp_at_p1 - x1a) * (y1b - y1a) / d

            lo = 0
            hi = n2
            while lo < hi:
                mid = (lo + hi) >> 1
                if ep_cont2[mid] <= tp_at_p2:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo
            if idx <= 0 or idx >= n2:
                b2_at = 0.0
            else:
                x2a = ep_cont2[idx - 1]
                x2b = ep_cont2[idx]
                y2a = b_cont2[idx - 1]
                y2b = b_cont2[idx]
                if lep == 1:
                    b2_at = y2a
                else:
                    d = x2b - x2a
                    b2_at = y2a if d == 0.0 else y2a + (tp_at_p2 - x2a) * (y2b - y2a) / d

            f1 = 0.5 * b1_at * r1
            f2 = 0.5 * b2_at * r2

            if lei_law == 1:
                f_amp = f1
            else:
                f_amp = f1 + (e - e1) * (f2 - f1) / e2_minus_e1

            result[ie, iep] = f_amp * sin_integral

    return result


def try_panel_pair(data, panel_idx, lei, e_sub, ep_out,
                   eff_lct, gl_x, gl_w, xp):
    """Return the fused numba-computed panel-pair spectrum, or None
    if the fast-path gate does not apply. See module docstring for
    the gate. None means the caller should use its own numpy path.
    """
    if not HAS_NUMBA:
        return None
    if xp.name != 'numpy':
        return None

    p1 = int(panel_idx)
    p2 = p1 + 1
    nep_np = np.asarray(data.nep_arr)
    nd_np = np.asarray(data.nd_arr)
    na_np = np.asarray(data.na_arr)
    nep1 = int(nep_np[p1]); nep2 = int(nep_np[p2])
    nd1 = int(nd_np[p1]);   nd2 = int(nd_np[p2])
    na1 = int(na_np[p1]);   na2 = int(na_np[p2])

    if int(data.lang) != 1 or na1 != 0 or na2 != 0:
        return None
    if int(data.lep) not in (1, 2):
        return None
    if nep1 <= nd1 or nep2 <= nd2:
        return None
    if eff_lct not in (1, 2):
        return None

    c0 = float(np.sqrt(data.awi * data.awp) / (data.awi + data.awr))
    if c0 != 0.0:
        return None

    lei_law = int(lei) % 10
    if lei_law not in (1, 2):
        return None

    ep_panels_np = np.asarray(data.ep_panels)
    b_panels_np = np.asarray(data.b_panels)

    ep_cont1 = np.ascontiguousarray(ep_panels_np[p1, nd1:nep1], dtype=np.float64)
    ep_cont2 = np.ascontiguousarray(ep_panels_np[p2, nd2:nep2], dtype=np.float64)
    b_cont1 = np.ascontiguousarray(b_panels_np[p1, nd1:nep1, 0], dtype=np.float64)
    b_cont2 = np.ascontiguousarray(b_panels_np[p2, nd2:nep2, 0], dtype=np.float64)
    e1 = float(np.asarray(data.ei_mesh)[p1])
    e2 = float(np.asarray(data.ei_mesh)[p2])

    e_sub_np = np.ascontiguousarray(np.asarray(e_sub, dtype=np.float64))
    ep_out_np = np.ascontiguousarray(np.asarray(ep_out, dtype=np.float64))
    gl_x_np = np.ascontiguousarray(np.asarray(gl_x, dtype=np.float64))
    gl_w_np = np.ascontiguousarray(np.asarray(gl_w, dtype=np.float64))

    return _numba_law1_gamma_na0_panel_pair(
        ep_cont1, b_cont1, ep_cont2, b_cont2,
        e1, e2, e_sub_np, ep_out_np, gl_x_np, gl_w_np,
        int(data.lep), lei_law,
    )
