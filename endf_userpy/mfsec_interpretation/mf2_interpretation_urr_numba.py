"""Numba-compiled kernel for :mod:`mf2_interpretation_urr`.

Same three-backend pattern as MLBW and R-M: numpy / JAX share
the array-namespace implementation in
:mod:`mf2_interpretation_urr`; this module holds a hand-fused
``@njit(parallel=True, fastmath=True)`` per-energy kernel that
implements the same physics for the ``'numba'`` backend.

The physics is exactly the numpy path (see the module docstring
of :mod:`mf2_interpretation_urr`): chi-squared width-fluctuation
integrals per spin group, reduced to a 1D Gauss-Legendre
quadrature on the compactified ``u = t / (1 + t)`` variable.

Inputs (:class:`~mf2_interpretation_urr.URRData`) and outputs
(``dict`` of average partial cross sections) are the same.

Scope
-----

- LRU=2 LRF=2 (Case C, energy-dependent widths).
- INT=2 (lin-lin) energy-table interpolation only. Other INT
  codes are rejected at the wrapper.
- Common NE across all J-groups (enforced at preproc).

The scalar penetration / shift / phase helpers
(``pnt_shf_any_L``, ``phase_any_L``) are the same ones used by
the MLBW and R-M numba kernels, imported from
:mod:`mf2_interpretation_factors_numba`.
"""
from __future__ import annotations

import math

import numpy as np

from ..primitives import array_ns
from ..primitives import tab1 as tab1_mod
from .mf2_interpretation_factors_numba import (
    pnt_shf_any_L as _pnt_shf,
    phase_any_L as _phase,
)

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*a, **kw):
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        return lambda f: f

    prange = range


_EPS = 1e-38


@njit(cache=True, inline='always')
def _interp_lin_lin_scalar(es_row, y_row, e):
    """Scalar lin-lin interpolation with clamp-at-endpoint outside.

    ``es_row`` and ``y_row`` are 1D arrays of the same length; ``e``
    is a scalar query energy. Assumes ``es_row`` is monotone
    increasing. Below ``es_row[0]`` returns ``y_row[0]``; above
    ``es_row[-1]`` returns ``y_row[-1]``.
    """
    n = es_row.shape[0]
    if e <= es_row[0]:
        return y_row[0]
    if e >= es_row[n - 1]:
        return y_row[n - 1]
    # Linear search is fine for typical NE ~ 20 (URR ranges have
    # few knots). Bisection would be faster asymptotically but
    # adds branching in the hot loop.
    for j in range(1, n):
        if e < es_row[j]:
            e0 = es_row[j - 1]
            e1 = es_row[j]
            y0 = y_row[j - 1]
            y1 = y_row[j]
            return y0 + (e - e0) / (e1 - e0) * (y1 - y0)
    return y_row[n - 1]   # unreachable given the clamp above


@njit(cache=True, inline='always')
def _interp_dispatch_scalar(es_row, y_row, e, int_code):
    """Dispatch INT=2 (lin-lin) and INT=5 (log-log) per group. Rows
    with any non-positive y fall back to lin-lin for numerical
    safety (typical case: GF row all zero on a non-fissile group)."""
    if int_code != 5:
        return _interp_lin_lin_scalar(es_row, y_row, e)
    n = y_row.shape[0]
    all_pos = True
    for j in range(n):
        if y_row[j] <= 0.0 or es_row[j] <= 0.0:
            all_pos = False
            break
    if not all_pos:
        return _interp_lin_lin_scalar(es_row, y_row, e)
    # log-log with clamp-at-endpoint.
    if e <= es_row[0]:
        return y_row[0]
    if e >= es_row[n - 1]:
        return y_row[n - 1]
    for j in range(1, n):
        if e < es_row[j]:
            le0 = math.log(es_row[j - 1])
            le1 = math.log(es_row[j])
            ly0 = math.log(y_row[j - 1])
            ly1 = math.log(y_row[j])
            frac = (math.log(e) - le0) / (le1 - le0)
            return math.exp(ly0 + frac * (ly1 - ly0))
    return y_row[n - 1]


@njit(cache=True, inline='always')
def _channel_factor_scalar(alpha, nu, t, order):
    """Scalar version of :func:`mf2_interpretation_urr._channel_factor`.

    ``alpha`` and ``nu`` are per-channel scalars at a fixed (E,
    group); ``t`` is a scalar quadrature node. ``order`` selects
    which Gamma-power is inside the expectation (0 / 1 / 2).

    nu == 0 (deterministic width) returns ``exp(-t * alpha)`` at
    all orders.
    """
    if nu <= _EPS:
        return math.exp(-t * alpha)
    base = 1.0 + 2.0 * t * alpha / nu
    if order == 0:
        return base ** (-nu / 2.0)
    if order == 1:
        return base ** (-nu / 2.0 - 1.0)
    # order == 2
    return (1.0 + 2.0 / nu) * base ** (-nu / 2.0 - 2.0)


@njit(cache=True, parallel=True, fastmath=True)
def _reconstruct_kernel(
    e,               # (ne,) query energies
    r_a_e,           # (ne,) channel radius interpolated at e
    r_ap_e,          # (ne,) scattering radius interpolated at e
    abn, ki,
    group_l,         # (nJ,)
    group_g,         # (nJ,)
    group_amun,      # (nJ,)
    group_amug,
    group_amuf,
    group_amux,
    table_es,        # (nJ, NE_tab)
    table_d,
    table_gn0,
    table_gg,
    table_gf,
    table_gx,
    group_int,       # (nJ,) ENDF INT law per group (2 or 5)
    t_nodes,         # (Nq,)
    w_t,             # (Nq,)
    group_pot_weight,  # (nJ,) 2L+1 at the first J-group of
                       # each unique L; 0 elsewhere.
):
    ne = e.shape[0]
    nJ = group_l.shape[0]
    Nq = t_nodes.shape[0]

    sct = np.zeros(ne)
    cap = np.zeros(ne)
    fis = np.zeros(ne)
    rxx = np.zeros(ne)
    pot = np.zeros(ne)

    for i in prange(ne):
        E = e[i]
        E_safe = E if E > 0.0 else 0.0
        if E > 0.0:
            k_e = ki * math.sqrt(E_safe)
            inv_k2 = math.pi / (k_e * k_e)
        else:
            inv_k2 = 0.0
        r_a_i = r_a_e[i]
        r_ap_i = r_ap_e[i]
        rho_a = ki * math.sqrt(E_safe) * r_a_i
        rho_ap = ki * math.sqrt(E_safe) * r_ap_i

        sct_i = 0.0
        cap_i = 0.0
        fis_i = 0.0
        rxx_i = 0.0
        pot_i = 0.0

        for g in range(nJ):
            L = group_l[g]
            gJ = group_g[g]
            nu_n = group_amun[g]
            nu_g = group_amug[g]
            nu_f = group_amuf[g]
            nu_x = group_amux[g]

            # ---- Interpolated widths + level spacing at E.
            int_code = group_int[g]
            gn0 = _interp_dispatch_scalar(table_es[g], table_gn0[g], E, int_code)
            gg  = _interp_dispatch_scalar(table_es[g], table_gg[g],  E, int_code)
            gf  = _interp_dispatch_scalar(table_es[g], table_gf[g],  E, int_code)
            gx  = _interp_dispatch_scalar(table_es[g], table_gx[g],  E, int_code)
            D   = _interp_dispatch_scalar(table_es[g], table_d[g],   E, int_code)

            # ---- Penetration / phase; v_L = P_L / rho (with L=0
            # branch pinned to 1 near rho=0).
            p_L, _s_L = _pnt_shf(rho_a, L)
            phi_L = _phase(rho_ap, L)
            if L == 0:
                v_L = 1.0
            else:
                v_L = p_L / rho_a if rho_a > _EPS else 0.0

            # ---- Physical average neutron width.
            alpha_n = gn0 * math.sqrt(E_safe) * v_L
            alpha_g = gg
            alpha_f = gf
            alpha_x = gx

            # ---- Fluctuation integrals: sum over Nq quadrature
            # nodes, four channels.
            R_ncap_val = 0.0
            R_nfis_val = 0.0
            R_ncomp_val = 0.0
            R_nn_val = 0.0
            for q in range(Nq):
                t = t_nodes[q]
                w = w_t[q]

                # Neutron order-0 factor is not used: every R
                # integral has neutron as c1, so we always want
                # g1_n or g2_n on the neutron side (same reason
                # the numpy path drops g0_n).
                g0_g = _channel_factor_scalar(alpha_g, nu_g, t, 0)
                g0_f = _channel_factor_scalar(alpha_f, nu_f, t, 0)
                g0_x = _channel_factor_scalar(alpha_x, nu_x, t, 0)

                g1_n = _channel_factor_scalar(alpha_n, nu_n, t, 1)
                g1_g = _channel_factor_scalar(alpha_g, nu_g, t, 1)
                g1_f = _channel_factor_scalar(alpha_f, nu_f, t, 1)
                g1_x = _channel_factor_scalar(alpha_x, nu_x, t, 1)

                g2_n = _channel_factor_scalar(alpha_n, nu_n, t, 2)

                R_ncap_val  += w * g1_n * g1_g * g0_f * g0_x
                R_nfis_val  += w * g1_n * g0_g * g1_f * g0_x
                R_ncomp_val += w * g1_n * g0_g * g0_f * g1_x
                R_nn_val    += w * g2_n * g0_g * g0_f * g0_x

            R_ncap_val  *= alpha_n * alpha_g
            R_nfis_val  *= alpha_n * alpha_f
            R_ncomp_val *= alpha_n * alpha_x
            R_nn_val    *= alpha_n * alpha_n

            # ---- Assemble per-group contribution.
            D_safe = D if D > _EPS else 1.0
            factor = (2.0 * math.pi / D_safe) * gJ

            # Resonance elastic + interference correction. The
            # interference formula matches NJOY unresr exactly:
            #   Δσ_int = -(4π²/k²) · g_J · <Γ_n> · sin²(φ_L) / <D>
            # Split as: sct_i (per-group internal units) gets
            # R_nn * factor for the resonance piece, then the
            # interference correction with sin²(φ_L) (not
            # sin(2·)) and <Γ_n> (not <Γ_n²/Γ>).
            sin_phi = math.sin(phi_L)
            sct_i += R_nn_val * factor
            sct_i += -4.0 * math.pi * gJ * alpha_n * sin_phi * sin_phi / D_safe
            cap_i += R_ncap_val * factor
            fis_i += R_nfis_val * factor
            rxx_i += R_ncomp_val * factor

            # Potential elastic: (2L+1) · sin²(φ_L) at the first
            # J-group of each unique L (accumulator initialised
            # by the wrapper via ``group_pot_weight``, which is
            # zero for every non-first group). Matches NJOY
            # unresr line 1072 (``if j.eq.1: spot += (2ll+1)
            # sin²(ps)``).
            pot_i += group_pot_weight[g] * sin_phi * sin_phi

        sct[i] = inv_k2 * sct_i + 4.0 * inv_k2 * pot_i
        cap[i] = inv_k2 * cap_i
        fis[i] = inv_k2 * fis_i
        rxx[i] = inv_k2 * rxx_i
        pot[i] = 4.0 * inv_k2 * pot_i

    return sct, cap, fis, rxx, pot


def reconstruct(data, energies_in):
    """URR average cross sections via the numba-compiled kernel.

    Signature-compatible with
    :func:`mf2_interpretation_urr.reconstruct` minus the backend
    argument (implicit here). The wrapper pre-interpolates the
    radii on the query grid (numba does not call the TAB1 helper
    directly), guards INT=2, and dispatches. First call incurs
    the numba compile cost (typically a few seconds); subsequent
    calls hit the cached compilation.
    """
    if not HAS_NUMBA:
        raise RuntimeError(
            'URR numba backend requested but `numba` is not installed. '
            "`pip install numba` or use `array_ns.get_backend('numpy')`."
        )

    # ---- INT-code guard: same as the numpy path (INT=2 lin-lin and
    # INT=5 log-log; other INT codes still unsupported).
    group_int = np.asarray(data.group_int, dtype=np.int64)
    bad = [int(v) for v in group_int if int(v) not in (2, 5)]
    if bad:
        raise NotImplementedError(
            f'URR numba kernel supports INT=2 (lin-lin) and INT=5 '
            f'(log-log) energy-table interpolation only; got INT '
            f'values {sorted(set(bad))} in the URR spin groups.'
        )

    e = np.asarray(energies_in, dtype=np.float64)
    e_safe = np.maximum(e, 0.0)

    # ---- Radii on the query grid.
    xp = array_ns.get_backend('numpy')
    r_a_e = np.asarray(tab1_mod.interp(data.r_a, e_safe, xp))
    r_ap_e = np.asarray(tab1_mod.interp(data.r_ap, e_safe, xp))

    # ---- Gauss-Legendre nodes / weights: use the module-level
    # constants from the numpy path for exact numerical
    # agreement.
    from .mf2_interpretation_urr import _T_NODES, _T_WEIGHTS
    t_nodes = np.asarray(_T_NODES, dtype=np.float64)
    w_t = np.asarray(_T_WEIGHTS, dtype=np.float64)

    # Per-group potential-elastic weight: (2L+1) at the FIRST
    # J-group of each unique L, 0 elsewhere. Distributes the
    # per-L potential sum across the kernel's per-group loop
    # so we don't need a second pass.
    group_l_arr = np.asarray(data.group_l, dtype=np.int64)
    group_pot_weight = np.zeros(group_l_arr.shape[0], dtype=np.float64)
    seen_L: set = set()
    for g_idx, L_val in enumerate(group_l_arr):
        if int(L_val) in seen_L:
            continue
        seen_L.add(int(L_val))
        group_pot_weight[g_idx] = 2.0 * int(L_val) + 1.0

    sct, cap, fis, rxx, pot = _reconstruct_kernel(
        e, r_a_e, r_ap_e,
        float(data.abn), float(data.ki),
        group_l_arr,
        np.asarray(data.group_g, dtype=np.float64),
        np.asarray(data.group_amun, dtype=np.float64),
        np.asarray(data.group_amug, dtype=np.float64),
        np.asarray(data.group_amuf, dtype=np.float64),
        np.asarray(data.group_amux, dtype=np.float64),
        np.asarray(data.table_es, dtype=np.float64),
        np.asarray(data.table_d, dtype=np.float64),
        np.asarray(data.table_gn0, dtype=np.float64),
        np.asarray(data.table_gg, dtype=np.float64),
        np.asarray(data.table_gf, dtype=np.float64),
        np.asarray(data.table_gx, dtype=np.float64),
        group_int,
        t_nodes, w_t,
        group_pot_weight,
    )
    tot = sct + cap + fis + rxx
    return {'sct': sct, 'cap': cap, 'fis': fis,
            'rxx': rxx, 'pot': pot, 'tot': tot}
