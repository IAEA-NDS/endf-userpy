"""Unresolved-resonance region (LRU=2) reconstruction: dataclass +
kernel.

ENDF-6 MF2/MT151 LRU=2 (URR) parameterises the average cross
section over the region where individual resonances can no longer
be resolved. Instead of resonance-by-resonance parameters, the
file lists, per spin group (L, J), the average widths
(``GN0``, ``GG``, ``GF``, ``GX``) and level spacing (``D``) as
functions of incident energy, along with degrees-of-freedom
(``AMUN``, ``AMUG``, ``AMUF``, ``AMUX``) that specify how each
width fluctuates around its average via chi-squared distributions.

Layout mirrors :class:`~mf2_interpretation_mlbw.MLBWData` /
:class:`~mf2_interpretation_reichmoore.RMData`: scalars for the
range, per-group arrays, tabulated widths. See
:mod:`mf2_interpretation_urr_preproc` for the ENDF-dict lift.

Physics
-------

For each spin group ``(L, J)`` at incident energy ``E``, the
average partial cross section is (Fröhner 1978; NJOY unresr
manual, chapter 4)::

    <σ_{n,c}(E)> = (π / k(E)²) g_J · (2π / <D(E)>) · <Γ_n Γ_c / Γ>

where the four-dimensional chi-squared expectation on the right
is reduced to a **one-dimensional integral** using the Laplace
identity ``1/Γ = ∫₀^∞ exp(-tΓ) dt``. Each channel width ``Γ_i``
follows a chi-squared distribution with DoF ``ν_i`` and mean
``⟨Γ_i(E)⟩``, so its moment-generating factors are analytic::

    <exp(-tΓ_i)>       = (1 + 2t ⟨Γ_i⟩ / ν_i)^(-ν_i/2)
    <Γ_i exp(-tΓ_i)>   = ⟨Γ_i⟩ (1 + 2t ⟨Γ_i⟩ / ν_i)^(-ν_i/2 - 1)
    <Γ_i² exp(-tΓ_i)>  = ⟨Γ_i⟩² (1 + 2/ν_i)
                                 (1 + 2t ⟨Γ_i⟩ / ν_i)^(-ν_i/2 - 2)

Assembling::

    <Γ_{c1} Γ_{c2} / Γ> = α_{c1} α_{c2}
                          ∫₀^∞ [∏_i g_i^{(0)}(t)] · g_{c1}^{(1)}(t) · g_{c2}^{(1)}(t) dt
                          (c1 ≠ c2)

    <Γ_n² / Γ>          = α_n²
                          ∫₀^∞ [∏_{i≠n} g_i^{(0)}(t)] · g_n^{(2)}(t) dt

where ``g_i^{(0/1/2)}`` are the per-channel factors above (the
``⟨Γ⟩^k`` prefactors have been pulled outside).

The channel with ``ν_i = 0`` (a common ENDF convention meaning
"no fluctuation") is handled by taking the ``ν → 0`` limit of the
chi-squared distribution, which is a delta at ``⟨Γ_i⟩``::

    <exp(-tΓ_i)>       = exp(-t ⟨Γ_i⟩)             (all orders)

The one-dimensional integral over ``t ∈ (0, ∞)`` is done by
Gauss-Legendre quadrature on the compactified variable
``u = t / (1 + t) ∈ [0, 1]`` with Jacobian ``dt = du / (1-u)²``.
32 nodes give sub-permille accuracy on the typical DoF ranges
found in ENDF-6 URR files. All operations are pure arithmetic on
arrays through the :mod:`~endf_userpy.primitives.array_ns`
adapter, so numpy / JAX / numba share one implementation and
``jax.grad`` flows through without special-casing.

Potential elastic is added on top per group as
``4π/k² · g_J sin²(φ_L)``. The resonance-potential interference
term is not yet included (typically small in the URR because
resonance widths are much smaller than level spacing); track
against NJOY unresr to decide whether to add it later.

Scope
-----

- **LRF=2** (Case C): energy-dependent widths tabulated at knot
  energies per J-group. Covers essentially every modern actinide
  URR range.
- **INT=2** (lin-lin) for the energy tables. Real URR files
  overwhelmingly use lin-lin; other INT codes are rejected at
  the wrapper with a clear message.
- **Backend-agnostic**: numpy / JAX via
  :mod:`~endf_userpy.primitives.array_ns`; numba path lives in
  :mod:`mf2_interpretation_urr_numba` (planned follow-up commit).

Not covered:

- Interference between resonance and potential elastic (small in
  URR; add if the NJOY-unresr comparison shows systematic bias).
- Case A / B (LRF=1 constant widths). Special case of Case C
  with NE=1; add when a real file surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..primitives import tab1
from . import mf2_interpretation_factors as factors


_EPS = 1e-38


# Gauss-Legendre quadrature on the compactified [0, 1] variable
# u = t / (1 + t). 32 nodes on [-1, 1] mapped to [0, 1], then
# Jacobian factor for the substitution back to t ∈ (0, ∞).
_QUAD_ORDER = 32
_gl_x, _gl_w = np.polynomial.legendre.leggauss(_QUAD_ORDER)
_U_NODES = 0.5 * (_gl_x + 1.0)                       # (Nq,) in [0, 1]
_U_WEIGHTS = 0.5 * _gl_w
_T_NODES = _U_NODES / (1.0 - _U_NODES)               # (Nq,) in (0, ∞)
_T_WEIGHTS = _U_WEIGHTS / (1.0 - _U_NODES) ** 2      # includes Jacobian


@dataclass
class URRData:
    """Natural-size URR (LRU=2 LRF=2) input for one isotope / one range.

    See module docstring for the fluctuation-integral physics
    that consumes these fields. See
    :mod:`mf2_interpretation_urr_preproc` for the ENDF-dict lift.
    """

    abn: np.ndarray
    spi: np.ndarray
    ap: np.ndarray
    awri: np.ndarray
    ki: np.ndarray
    naps: int

    group_l: np.ndarray
    group_j2: np.ndarray
    group_g: np.ndarray
    group_amun: np.ndarray
    group_amug: np.ndarray
    group_amuf: np.ndarray
    group_amux: np.ndarray
    group_int: np.ndarray

    table_es: np.ndarray
    table_d: np.ndarray
    table_gn0: np.ndarray
    table_gg: np.ndarray
    table_gf: np.ndarray
    table_gx: np.ndarray

    r_a: tab1.TAB1
    r_ap: tab1.TAB1


def _interp_per_group(table, es, e_query, xp):
    """Lin-lin batched interp: for a (nJ, NE_tab) table with per-group
    ES rows and an ``(NE,)`` query grid, return ``(NE, nJ)``.

    Uses a Python-side loop over the ``nJ`` groups (typically ~10)
    with ``xp.interp`` per column. JAX unrolls at trace time;
    fine at these sizes.
    """
    cols = []
    for g in range(int(table.shape[0])):
        cols.append(xp.interp(e_query, es[g], table[g]))
    return xp.stack(cols, axis=-1)   # (NE, nJ)


def _channel_factor(alpha, nu, t_nodes, order, xp):
    """Per-channel integrand factor at each quadrature node.

    ``alpha`` and ``nu`` broadcast to shape ``(..., 1)``; ``t_nodes``
    has shape ``(Nq,)``. Returns ``(..., Nq)``.

    ``order`` selects which power of ``Γ_i`` sits inside the
    expectation:

    - ``0``: ``<exp(-tΓ_i)>`` (channel appears only in the total).
    - ``1``: ``<Γ_i exp(-tΓ_i)> / <Γ_i>`` (channel is one of the
      partials in the numerator; the ``α_i`` prefactor is pulled
      out in the caller).
    - ``2``: ``<Γ_i² exp(-tΓ_i)> / <Γ_i>²`` (elastic case, ``c1 =
      c2 = n``; the ``α_n²`` prefactor is pulled out in the
      caller).

    Handles ``ν = 0`` (deterministic width, "no fluctuation")
    with ``exp(-t α)`` at all orders. Double-``where`` guards the
    ``(1 + 2/ν)`` factor at ``ν = 0`` so autodiff does not see a
    ``0/0`` on the unused branch.
    """
    alpha_bc = alpha[..., None]                        # (..., 1)
    nu_bc = nu[..., None]                              # (..., 1)
    nu_safe = xp.where(nu_bc > _EPS, nu_bc, 1.0)
    base = 1.0 + 2.0 * t_nodes * alpha_bc / nu_safe    # (..., Nq)
    exp_form = xp.exp(-t_nodes * alpha_bc)             # (..., Nq)
    is_zero = nu_bc <= _EPS

    if order == 0:
        pow_form = base ** (-nu_bc / 2.0)
        return xp.where(is_zero, exp_form, pow_form)
    if order == 1:
        pow_form = base ** (-nu_bc / 2.0 - 1.0)
        return xp.where(is_zero, exp_form, pow_form)
    if order == 2:
        factor2 = xp.where(is_zero, 1.0, 1.0 + 2.0 / nu_safe)
        pow_form = factor2 * base ** (-nu_bc / 2.0 - 2.0)
        return xp.where(is_zero, exp_form, pow_form)
    raise ValueError(f'order must be 0, 1, or 2; got {order!r}')


def reconstruct(data: URRData, energies_in, xp) -> dict:
    """URR average cross sections at ``energies_in``.

    Returns a dict with keys ``'sct'`` (elastic including
    potential), ``'cap'``, ``'fis'``, ``'rxx'`` (competitive),
    ``'pot'`` (potential elastic, diagnostic), and ``'tot'``,
    each shape ``(len(energies_in),)``, in barn.

    Callers who want only a subset can slice the returned dict.
    Query energies outside the URR range ``[EL, EH]`` are computed
    with the same formulas (the energy-table interpolation clamps
    at the endpoints), so it is the caller's job to mask them if
    needed; downstream composition in
    :mod:`~endf_userpy.quantities_mt_zap.resonance_composition`
    does that clipping.

    Parameters
    ----------
    data : URRData
        Preprocessed URR input for the range (one isotope, one
        range).
    energies_in : array_like
        Incident-neutron energies, in eV.
    xp : backend
        As returned by :func:`~endf_userpy.primitives.array_ns.get_backend`.
    """
    if getattr(xp, 'name', None) == 'numba':
        from . import mf2_interpretation_urr_numba as _numba
        return _numba.reconstruct(data, energies_in)

    # ---- INT-code guard. LRF=2 URR files overwhelmingly use
    # lin-lin (INT=2). Fail loud on anything else so a caller
    # doesn't quietly get wrong average widths from a mis-applied
    # linear interpolation.
    if not bool(xp.all(data.group_int == 2)):
        offending = [
            int(v) for v in data.group_int if int(v) != 2
        ]
        raise NotImplementedError(
            f'URR reconstruction currently supports INT=2 (lin-lin) '
            f'energy-table interpolation only; got INT values '
            f'{sorted(set(offending))} in the URR spin groups. Add '
            f'log-lin / lin-log / log-log if a real file needs it.'
        )

    e = xp.asarray(energies_in, dtype=xp.float64)
    e_safe = xp.where(e > 0.0, e, 1.0)                 # avoid sqrt(0)

    # ---- Radii + wavenumber at the query grid.
    r_a_e = tab1.interp(data.r_a, e_safe, xp)          # (NE,)
    r_ap_e = tab1.interp(data.r_ap, e_safe, xp)
    ki = xp.asarray(data.ki, dtype=xp.float64)
    k_e = ki * xp.sqrt(e_safe)                          # (NE,)
    rho_a_e = k_e * r_a_e                               # (NE,)
    rho_ap_e = k_e * r_ap_e
    inv_k2 = xp.pi / (k_e * k_e)                        # (NE,)

    # ---- Per-group L, penetration + shift + phase.
    L_by_g = xp.asarray(data.group_l, dtype=xp.int32)
    rho_a_bc = rho_a_e[:, None]                         # (NE, 1)
    rho_ap_bc = rho_ap_e[:, None]
    L_bc = L_by_g[None, :]                              # (1, nJ)

    p_by_g, _s_by_g = factors.pnt_shf(rho_a_bc, L_bc, xp)  # (NE, nJ)
    phi_by_g = factors.phase(rho_ap_bc, L_bc, xp)        # (NE, nJ)

    # v_L(E) = P_L(ρ) / ρ (with the L=0 branch pinned to 1 for
    # numerical safety near ρ = 0).
    rho_a_bc_safe = xp.where(rho_a_bc > _EPS, rho_a_bc, 1.0)
    v_L = xp.where(L_bc == 0, 1.0, p_by_g / rho_a_bc_safe)  # (NE, nJ)

    # ---- Per-group interpolated widths + level spacing.
    es = data.table_es                                   # (nJ, NE_tab)
    alpha_n0 = _interp_per_group(data.table_gn0, es, e_safe, xp)  # (NE, nJ)
    alpha_gg = _interp_per_group(data.table_gg, es, e_safe, xp)
    alpha_gf = _interp_per_group(data.table_gf, es, e_safe, xp)
    alpha_gx = _interp_per_group(data.table_gx, es, e_safe, xp)
    d_avg = _interp_per_group(data.table_d, es, e_safe, xp)

    # Physical neutron width from ENDF-reduced GN0(E):
    # <Γ_n(E)> = <GN0(E)> · √E · v_L(E)
    alpha_n_phys = alpha_n0 * xp.sqrt(e_safe[:, None]) * v_L  # (NE, nJ)

    # ---- Fluctuation-integral machinery.
    nu_n = xp.asarray(data.group_amun, dtype=xp.float64)[None, :]  # (1, nJ)
    nu_g = xp.asarray(data.group_amug, dtype=xp.float64)[None, :]
    nu_f = xp.asarray(data.group_amuf, dtype=xp.float64)[None, :]
    nu_x = xp.asarray(data.group_amux, dtype=xp.float64)[None, :]

    t_nodes = xp.asarray(_T_NODES)                         # (Nq,)
    w_t = xp.asarray(_T_WEIGHTS)                           # (Nq,)

    # (NE, nJ, Nq) per-channel factors. The neutron order-0
    # factor is not needed: every R integral has neutron as c1,
    # so we always want g1_n or g2_n on the neutron side.
    g0_g = _channel_factor(alpha_gg, nu_g, t_nodes, 0, xp)
    g0_f = _channel_factor(alpha_gf, nu_f, t_nodes, 0, xp)
    g0_x = _channel_factor(alpha_gx, nu_x, t_nodes, 0, xp)

    g1_n = _channel_factor(alpha_n_phys, nu_n, t_nodes, 1, xp)
    g1_g = _channel_factor(alpha_gg, nu_g, t_nodes, 1, xp)
    g1_f = _channel_factor(alpha_gf, nu_f, t_nodes, 1, xp)
    g1_x = _channel_factor(alpha_gx, nu_x, t_nodes, 1, xp)

    g2_n = _channel_factor(alpha_n_phys, nu_n, t_nodes, 2, xp)

    # Integrand for each fluctuation-averaged partial, then
    # quadrature over t.
    R_ncap = alpha_n_phys * alpha_gg * xp.sum(
        g1_n * g1_g * g0_f * g0_x * w_t, axis=-1,
    )
    R_nfis = alpha_n_phys * alpha_gf * xp.sum(
        g1_n * g0_g * g1_f * g0_x * w_t, axis=-1,
    )
    R_ncomp = alpha_n_phys * alpha_gx * xp.sum(
        g1_n * g0_g * g0_f * g1_x * w_t, axis=-1,
    )
    R_nn = alpha_n_phys * alpha_n_phys * xp.sum(
        g2_n * g0_g * g0_f * g0_x * w_t, axis=-1,
    )

    # ---- Assemble average partial XS.
    # <σ_{n,c}(E)> = (π/k²) g_J · (2π/D) · R_c
    d_safe = xp.where(d_avg > _EPS, d_avg, 1.0)
    g_by_g = xp.asarray(data.group_g, dtype=xp.float64)[None, :]
    two_pi_over_d = 2.0 * xp.pi / d_safe                  # (NE, nJ)

    sct_res_per_g = R_nn * two_pi_over_d * g_by_g
    cap_per_g = R_ncap * two_pi_over_d * g_by_g
    fis_per_g = R_nfis * two_pi_over_d * g_by_g
    rxx_per_g = R_ncomp * two_pi_over_d * g_by_g

    sct_res = inv_k2 * xp.sum(sct_res_per_g, axis=-1)
    cap = inv_k2 * xp.sum(cap_per_g, axis=-1)
    fis = inv_k2 * xp.sum(fis_per_g, axis=-1)
    rxx = inv_k2 * xp.sum(rxx_per_g, axis=-1)

    # ---- Potential elastic: 4π/k² Σ g_J sin²(φ_L). Per-group
    # so that missing J-groups contribute nothing (they have no
    # g_J entry). Interference between resonance and potential
    # elastic is not yet included.
    sin2_phi = xp.sin(phi_by_g) ** 2                        # (NE, nJ)
    pot = 4.0 * inv_k2 * xp.sum(g_by_g * sin2_phi, axis=-1)

    sct = sct_res + pot
    tot = sct + cap + fis + rxx

    return {
        'sct': sct,
        'cap': cap,
        'fis': fis,
        'rxx': rxx,
        'pot': pot,
        'tot': tot,
    }
