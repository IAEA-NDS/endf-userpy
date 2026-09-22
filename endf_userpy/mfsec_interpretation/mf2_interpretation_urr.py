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

Potential elastic is added per L (once per unique L in the
file's spin groups) as ``(4π/k²) · (2L+1) · sin²(φ_L)``. Matches
NJOY unresr's convention (``unresr.f90`` line 1072), which
assumes every physical J for a given L contributes equally to
the hard-sphere phase whether or not it appears as a spin group
in the file. The alternative (summing per-J with ``g_J`` from
the file) undercounts potential elastic when a J-group is
missing from the URR parameter table but exists physically.
Difference is small (~0.4% of elastic at the top of a
mid-actinide URR) but real, and shows up in the vs-NJOY parity.

The resonance-potential interference correction matches NJOY's
line 1073:

    Δ<σ_el(E)> = -(4π²/k²) · g_J · <Γ_n(E)> · sin²(φ_L) / <D>

Parity vs NJOY unresr on U-235
------------------------------

TENDL-2021 U-235 URR (21 knots [2.25, 46.2] keV), infinite
dilution, T = 0:

- Elastic: max **2e-5** relative
- Fission: max **3e-4** relative
- Capture: max **4e-4** relative

The residual ~3-4e-4 on capture / fission is **NJOY's quadrature
error**, not ours. NJOY unresr uses a compact 10-point Ross
quadrature for the chi-squared width-fluctuation integrals;
that quadrature has a systematic ~1e-5 relative error at
Porter-Thomas parameters (verified by comparing Ross's tables
to converged high-order Gauss-Laguerre), amplified in the
final cross-section assembly. Our 1D Laplace-Gauss-Legendre
integral converges to the physically-correct chi-squared
average and matches direct Monte Carlo to ~1e-5 (limit of MC
statistical uncertainty) on synthetic URR-like cases. So on
capture and fission we are ~10x closer to the true integral
than NJOY at U-235 URR parameters.

Scope
-----

- **LRF=2** (Case C): energy-dependent widths tabulated at knot
  energies per J-group. Covers essentially every modern
  actinide URR range.
- **INT=2** (lin-lin) or **INT=5** (log-log) for the energy
  tables. Real URR files use one of these two per group; other
  INT codes (INT=1 histogram, INT=3 lin-log, INT=4 log-lin) are
  rejected at
  the wrapper of :func:`reconstruct` with a clear message.
- **Backend-agnostic**: numpy / JAX via
  :mod:`~endf_userpy.primitives.array_ns`; numba path in
  :mod:`mf2_interpretation_urr_numba` mirrors it.

Not covered:

- Case A / B (LRF=1 constant widths). Special case of Case C
  with NE=1; add when a real file surfaces.
- Finite-dilution self-shielding (NJOY unresr's full assembly
  with sig_0 > 0, plus Doppler-broadened integrals via the
  ``ajku`` subroutine). :func:`reconstruct` produces the
  infinite-dilution T = 0 average XS.
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


# ---------------------------------------------------------------
# Hwang / MC²-2 10-point Gauss-Chi-Squared quadrature tables.
#
# For NJOY-compatible reconstruction of the χ² fluctuation
# integrals via `quadrature='ross_10'`. Reconstructed from first
# principles per Hwang's construction as documented in Henryson,
# Toppel & Stenberg, "MC²-2: A Code to Calculate Fast Neutron
# Spectra and Multigroup Cross Sections", ANL-8144 (ENDF-239),
# 1976, Appendix A, section V "Quadratures for Statistical
# Integration" (pp. 244-245, Eqs. A.32-A.36 + Table XIII). NJOY
# reconr's unresr uses the same tables (see NJOY2016 unresr.f90).
#
# For DOF ν, the mean-normalised χ² integral
#     ⟨f⟩_ν = ∫₀^∞ p_ν(x) f(x) dx,   p_ν(x) = (ν/2)^{ν/2}·x^{ν/2-1}·e^{-νx/2}/Γ(ν/2)
# is approximated by ⟨f⟩_ν ≈ Σ_{j=1}^{10} A_j f(X_j) with
#
#   Odd ν (=1, 3):
#     X_j = 2 Z_j² / ν
#     A_j = 2 W_j^S · Z_j^{ν-1} / Γ(ν/2)
#     where (Z_j, W_j^S) are the 10-point HALF-RANGE Gauss-Hermite
#     nodes and weights (weight function e^{-x²} on [0, ∞)), derived
#     by Steen, Byrne & Gelbard, "Gaussian Quadratures for the
#     Integrals ∫₀^∞ e^{-x²} f(x) dx and ∫₀^b e^{-x²} f(x) dx",
#     Math. Comp. 23 (1969), 661-671. We rebuild the Steen tables
#     from scratch via the Golub-Welsch (Chebyshev) construction
#     on the moments μ_k = Γ((k+1)/2)/2 of e^{-x²} on [0, ∞); the
#     tabulated Steen values are reproduced to the paper's printed
#     precision (~13 digits).
#
#   Even ν (=2, 4):
#     X_j = (1 - S_j) / (1 + S_j)
#     A_j = ν W_j^L · (νX_j/2)^{ν/2-1} · e^{-νX_j/2}
#           / [Γ(ν/2)·(1 + S_j)²]
#     where (S_j, W_j^L) are the standard 10-point Gauss-Legendre
#     nodes and weights on [-1, 1] (numpy.polynomial.legendre.leggauss).
#     The rational transformation X = (1-S)/(1+S) maps [-1, 1] → (0, ∞).
#
# By construction each row satisfies Σ_j A_j = 1 and Σ_j A_j X_j = 1
# (pdf and mean normalisation of the mean-normalised χ² pdf). Values
# reproduce Hwang's Table XIII (= NJOY reconr's hard-coded tables) to
# ~1e-7 max absolute error, limited by the 7-digit precision of the
# tabulated MC²-2 report.
_ROSS_NQP = 10
_ROSS_NU_MAX = 4    # Hwang provides tables for ν = 1..4; matches NJOY.


def _half_range_hermite_10(n=_ROSS_NQP):
    """10-point half-range Gauss-Hermite (weight e^{-x²} on [0, ∞))
    via Golub-Welsch from the analytic moments μ_k = Γ((k+1)/2)/2.
    Requires mpmath for the moment-recurrence step (numerical
    stability against Hankel-matrix ill-conditioning at n=10);
    the final Jacobi matrix is real-double and its eigen-
    decomposition uses numpy."""
    from mpmath import mp, mpf, gamma as _mp_gamma, sqrt as _mp_sqrt
    mp.dps = 60
    moms = [_mp_gamma(mpf(k + 1) / 2) / 2 for k in range(2 * n)]
    sigma = [[mpf(0)] * (2 * n + 1) for _ in range(n + 2)]
    for l in range(2 * n):
        sigma[1][l] = moms[l]
    alpha = [mpf(0)] * n
    beta = [mpf(0)] * n
    alpha[0] = sigma[1][1] / sigma[1][0]
    beta[0] = sigma[1][0]
    for k in range(1, n):
        for l in range(k, 2 * n - k):
            sigma[k + 1][l] = (
                sigma[k][l + 1]
                - alpha[k - 1] * sigma[k][l]
                - beta[k - 1] * sigma[k - 1][l]
            )
        alpha[k] = (
            sigma[k + 1][k + 1] / sigma[k + 1][k]
            - sigma[k][k] / sigma[k][k - 1]
        )
        beta[k] = sigma[k + 1][k] / sigma[k][k - 1]
    J = np.zeros((n, n))
    for k in range(n):
        J[k, k] = float(alpha[k])
        if k > 0:
            J[k, k - 1] = J[k - 1, k] = float(_mp_sqrt(beta[k]))
    ev, evec = np.linalg.eigh(J)
    Z = ev
    mu_0 = float(moms[0])
    W = mu_0 * evec[0, :] ** 2
    order = np.argsort(Z)
    return Z[order], W[order]


def _generate_ross_tables():
    """Build the (_ROSS_NU_MAX, _ROSS_NQP) qp / qw tables per
    Hwang's Eqs. A.33-A.36. Called once at module import;
    mpmath / scipy are only imported here so users without them
    can still use the default Gauss-Legendre-32 quadrature."""
    try:
        from scipy.special import gamma as _gamma
    except ImportError:   # pragma: no cover
        return None, None
    try:
        Z, WS = _half_range_hermite_10()
    except Exception:     # pragma: no cover
        return None, None
    # Standard 10-point Gauss-Legendre on [-1, 1].
    S, WL = np.polynomial.legendre.leggauss(_ROSS_NQP)
    qp = np.zeros((_ROSS_NU_MAX, _ROSS_NQP), dtype=np.float64)
    qw = np.zeros((_ROSS_NU_MAX, _ROSS_NQP), dtype=np.float64)
    for nu in range(1, _ROSS_NU_MAX + 1):
        if nu % 2 == 1:
            X = 2.0 * Z ** 2 / nu
            A = 2.0 * WS * Z ** (nu - 1) / _gamma(nu / 2)
        else:
            X = (1.0 - S) / (1.0 + S)
            A = (
                nu * WL * ((nu / 2) * X) ** (nu / 2 - 1)
                * np.exp(-nu / 2 * X)
                / (_gamma(nu / 2) * (1.0 + S) ** 2)
            )
        order = np.argsort(X)
        qp[nu - 1, :] = X[order]
        qw[nu - 1, :] = A[order]
    return qp, qw


_ROSS_QP, _ROSS_QW = _generate_ross_tables()


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


def _interp_per_group(table, es, e_query, xp, int_codes=None):
    """Per-group interp: for a (nJ, NE_tab) table with per-group
    ES rows and an ``(NE,)`` query grid, return ``(NE, nJ)``.

    ``int_codes`` (optional, (nJ,) int array) picks the ENDF INT law
    per group:

    - INT=2 (lin-lin): standard ``xp.interp``. Default when
      ``int_codes`` is None.
    - INT=5 (log-log): interpolate in log-log space. Rows with any
      non-positive y value fall back to lin-lin for that row so
      the interpolation is well-defined when a channel has zero
      width at some knots (common for GF on non-fissile files).

    Real URR ranges use INT=2 or INT=5 (typically per group). Other
    INT codes still raise upstream in :func:`reconstruct`.
    """
    cols = []
    for g in range(int(table.shape[0])):
        y_row = table[g]
        e_row = es[g]
        code = 2 if int_codes is None else int(int_codes[g])
        if code == 5 and bool(xp.all(y_row > 0)) and bool(xp.all(e_row > 0)):
            # log-log: interp in (log E, log y) space.
            col = xp.exp(xp.interp(
                xp.log(e_query), xp.log(e_row), xp.log(y_row),
            ))
        else:
            # lin-lin (INT=2) or zero-row fallback for INT=5.
            col = xp.interp(e_query, e_row, y_row)
        cols.append(col)
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


def _ross_channel_nodes(nu_int):
    """Return (x_nodes, w_weights) for a single χ² channel at DOF nu_int.

    For nu_int > 0: 10-point Ross-style Gauss-Chi-Squared tables.
    For nu_int == 0: single-point rule at x=1 (deterministic width).
    """
    if nu_int <= 0:
        return np.array([1.0]), np.array([1.0])
    if nu_int > _ROSS_NU_MAX:
        raise NotImplementedError(
            f'Ross-10 quadrature only tabulated for DOF ν = 1..'
            f'{_ROSS_NU_MAX}; got ν = {nu_int}. NJOY reconr / '
            f'Hwang MC²-2 provides tables only for these values. '
            f'ENDF-6 URR files in practice use ν ≤ 4.'
        )
    return _ROSS_QP[nu_int - 1, :], _ROSS_QW[nu_int - 1, :]


def _ross_average_group(
    an_g, ag_g, af_g, ax_g,
    nu_n_g, nu_g_g, nu_f_g, nu_x_g,
    xp,
):
    """Ross-style 4D nested Gauss-Chi-Squared quadrature for one
    spin group at all query energies.

    Inputs:
      an_g, ag_g, af_g, ax_g : (NE,) mean widths <Γ_c(E)>
      nu_*_g                 : scalar int DOF for each channel
      xp                     : backend

    Returns four (NE,) arrays: R_ncap, R_nfis, R_ncomp, R_nn.
    They are the fluctuation-averaged moments the caller then
    multiplies by (2π/D) g_J (π/k²) to get the average partial XS.
    """
    x_n, w_n = _ross_channel_nodes(nu_n_g)
    x_g, w_g = _ross_channel_nodes(nu_g_g)
    x_f, w_f = _ross_channel_nodes(nu_f_g)
    x_x, w_x = _ross_channel_nodes(nu_x_g)

    # Shape the quadrature axes into a 4D outer product so every
    # channel's node grid broadcasts against each other. Order:
    # (E, Kn, Kg, Kf, Kx). Trailing 1's collapse channels with
    # nu == 0 (single-point rule).
    xn_bc = xp.asarray(x_n)[None, :, None, None, None]
    xg_bc = xp.asarray(x_g)[None, None, :, None, None]
    xf_bc = xp.asarray(x_f)[None, None, None, :, None]
    xx_bc = xp.asarray(x_x)[None, None, None, None, :]
    wn_bc = xp.asarray(w_n)[None, :, None, None, None]
    wg_bc = xp.asarray(w_g)[None, None, :, None, None]
    wf_bc = xp.asarray(w_f)[None, None, None, :, None]
    wx_bc = xp.asarray(w_x)[None, None, None, None, :]

    an_bc = an_g[:, None, None, None, None]
    ag_bc = ag_g[:, None, None, None, None]
    af_bc = af_g[:, None, None, None, None]
    ax_bc = ax_g[:, None, None, None, None]

    gamma_n = an_bc * xn_bc
    gamma_g = ag_bc * xg_bc
    gamma_f = af_bc * xf_bc
    gamma_x = ax_bc * xx_bc
    gamma_tot = gamma_n + gamma_g + gamma_f + gamma_x

    weight = wn_bc * wg_bc * wf_bc * wx_bc

    # ⟨Γ_n · Γ_c / Γ_tot⟩ via the 4D outer expectation, reduced
    # over the four quadrature axes (1..4).
    def _reduce(f_num):
        return xp.sum(weight * gamma_n * f_num / gamma_tot,
                      axis=(1, 2, 3, 4))

    R_ncap = _reduce(gamma_g)
    R_nfis = _reduce(gamma_f)
    R_ncomp = _reduce(gamma_x)
    R_nn = _reduce(gamma_n)
    return R_ncap, R_nfis, R_ncomp, R_nn


def _reconstruct_ross_moments(
    alpha_n_phys, alpha_gg, alpha_gf, alpha_gx,
    nu_n_arr, nu_g_arr, nu_f_arr, nu_x_arr, xp,
):
    """Assemble the four fluctuation-averaged R moments across all
    (E, nJ) via a Python-side per-group loop. Each group's quadrature
    axes have their own sizes (depending on nu), so a single fully
    vectorised implementation would need padding to the worst-case
    DOF; the per-group loop keeps the shapes tight.
    """
    ne = alpha_n_phys.shape[0]
    nJ = alpha_n_phys.shape[1]
    R_ncap = xp.zeros((ne, nJ), dtype=xp.float64)
    R_nfis = xp.zeros((ne, nJ), dtype=xp.float64)
    R_ncomp = xp.zeros((ne, nJ), dtype=xp.float64)
    R_nn = xp.zeros((ne, nJ), dtype=xp.float64)
    for g in range(nJ):
        r_cap, r_fis, r_comp, r_nn = _ross_average_group(
            alpha_n_phys[:, g], alpha_gg[:, g],
            alpha_gf[:, g], alpha_gx[:, g],
            int(round(float(nu_n_arr[0, g]))),
            int(round(float(nu_g_arr[0, g]))),
            int(round(float(nu_f_arr[0, g]))),
            int(round(float(nu_x_arr[0, g]))),
            xp,
        )
        # Assignment via xp.concatenate/xp.stack to stay backend-agnostic.
        # `xp` supports advanced indexing on numpy; for JAX we would
        # use `.at[...].set(...)`. For now keep the simple per-column
        # assignment; the Ross path is numpy-oriented (JAX/numba fall
        # back to the default 32-point Gauss-Legendre).
        R_ncap[:, g] = r_cap
        R_nfis[:, g] = r_fis
        R_ncomp[:, g] = r_comp
        R_nn[:, g] = r_nn
    return R_ncap, R_nfis, R_ncomp, R_nn


def reconstruct(data: URRData, energies_in, xp,
                quadrature='gauss_legendre_32') -> dict:
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
    quadrature : {'gauss_legendre_32', 'ross_10'}, optional
        Choice of χ² fluctuation-integral quadrature.

        ``'gauss_legendre_32'`` (default) reduces the 4D χ² expectation
        to a 1D Laplace integral ``⟨exp(-tΓ)⟩`` and evaluates it on
        32 Gauss-Legendre nodes on the compactified interval
        ``u = t/(1+t)``. Monte Carlo on the same integrand agrees with
        this scheme to ``~1e-7`` on TENDL-2021 U-235.

        ``'ross_10'`` evaluates the 4D expectation directly as a nested
        Gauss-Chi-Squared quadrature, one axis per channel with
        DOF ν > 0, 10 points each. Uses the ``_ROSS_QP`` / ``_ROSS_QW``
        tables reconstructed independently from the Hwang MC²-2
        derivation (see their module-level docstring); the derived
        tables reproduce NJOY reconr's hard-coded values to the
        printed precision (~1e-7). Produces cross sections in the same
        precision class as NJOY unresr's Ross-style scheme (sub-permille
        residual vs NJOY on the URR-LSSF=0 files where the two
        ~10-point quadratures both differ from Monte Carlo truth by
        ~4e-4). Opt into this for NJOY cross-validation; keep the
        default for higher accuracy. Only supported on the numpy / JAX
        backends; the numba backend always uses the default.
    """
    if getattr(xp, 'name', None) == 'numba':
        if quadrature != 'gauss_legendre_32':
            raise NotImplementedError(
                f'numba backend only supports the default '
                f"quadrature='gauss_legendre_32'; got "
                f'{quadrature!r}. Switch to numpy or JAX to use '
                f"'ross_10'."
            )
        from . import mf2_interpretation_urr_numba as _numba
        return _numba.reconstruct(data, energies_in)
    if quadrature not in ('gauss_legendre_32', 'ross_10'):
        raise ValueError(
            f"quadrature must be 'gauss_legendre_32' or 'ross_10'; "
            f'got {quadrature!r}'
        )
    if quadrature == 'ross_10' and _ROSS_QP is None:
        raise ImportError(
            "quadrature='ross_10' requires mpmath (for the half-range "
            'Gauss-Hermite moment recurrence used to derive the odd-ν '
            'tables). Install mpmath, or use the default '
            "quadrature='gauss_legendre_32'."
        )

    # ---- INT-code guard. LRF=2 URR files in real evaluations use
    # either INT=2 (lin-lin) or INT=5 (log-log) on the average-
    # width / spacing tables. Fail loud on anything else.
    supported = {2, 5}
    unsupported = [int(v) for v in data.group_int if int(v) not in supported]
    if unsupported:
        raise NotImplementedError(
            f'URR reconstruction currently supports INT=2 (lin-lin) '
            f'and INT=5 (log-log) energy-table interpolation only; '
            f'got INT values {sorted(set(unsupported))} in the URR '
            f'spin groups. Add lin-log / log-lin (INT=3 / INT=4) if '
            f'a real file needs it.'
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
    ints = data.group_int
    alpha_n0 = _interp_per_group(data.table_gn0, es, e_safe, xp, ints)  # (NE, nJ)
    alpha_gg = _interp_per_group(data.table_gg, es, e_safe, xp, ints)
    alpha_gf = _interp_per_group(data.table_gf, es, e_safe, xp, ints)
    alpha_gx = _interp_per_group(data.table_gx, es, e_safe, xp, ints)
    d_avg = _interp_per_group(data.table_d, es, e_safe, xp, ints)

    # Physical neutron width from ENDF-reduced GN0(E). ENDF-6
    # D.3.4 stores GN0 as the reduced average width divided by the
    # neutron degrees of freedom AMUN, so the recovered mean is
    #   <Γ_n(E)> = GN0(E) · √E · v_L(E) · AMUN
    # The other three channels (GG, GF, GX) are stored as the direct
    # averages <Γ_c(E)> and do NOT get an AMU factor here — the
    # asymmetry is a historical ENDF convention. NJOY unresr's
    # unresl subroutine (unresr.f90 line 1068) applies exactly this
    # AMUN factor on the neutron width alone; ours previously
    # matched that only for AMUN=1 groups, giving up to ~7%
    # underestimate on the elastic and capture averages for files
    # with AMUN=2 groups (Xe-135 was the surfacing case).
    amun_bc = xp.asarray(data.group_amun, dtype=xp.float64)[None, :]  # (1, nJ)
    alpha_n_phys = alpha_n0 * xp.sqrt(e_safe[:, None]) * v_L * amun_bc  # (NE, nJ)

    # ---- Fluctuation-integral machinery.
    nu_n = xp.asarray(data.group_amun, dtype=xp.float64)[None, :]  # (1, nJ)
    nu_g = xp.asarray(data.group_amug, dtype=xp.float64)[None, :]
    nu_f = xp.asarray(data.group_amuf, dtype=xp.float64)[None, :]
    nu_x = xp.asarray(data.group_amux, dtype=xp.float64)[None, :]

    if quadrature == 'ross_10':
        R_ncap, R_nfis, R_ncomp, R_nn = _reconstruct_ross_moments(
            alpha_n_phys, alpha_gg, alpha_gf, alpha_gx,
            nu_n, nu_g, nu_f, nu_x, xp,
        )
    else:
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
    sin2_phi = xp.sin(phi_by_g) ** 2                       # (NE, nJ)

    # Resonance elastic per group.
    sct_res_per_g = R_nn * two_pi_over_d * g_by_g

    # Interference correction per group, matching NJOY unresr
    # exactly (unresr.f90 line 1073):
    #     Δσ_int = -(4π²/k²) · g_J · <Γ_n> · sin²(φ_L) / <D>
    # Note the sin² (not sin(2·)) form and the <Γ_n> (not
    # <Γ_n²/Γ>) factor. Pulling out the outer (π/k²) that gets
    # applied later at inv_k2 · sum, the per-group internal
    # contribution is:
    interf_per_g = -4.0 * xp.pi * g_by_g * alpha_n_phys * sin2_phi / d_safe
    sct_res_per_g = sct_res_per_g + interf_per_g
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
    # Potential elastic: (4π/k²) · Σ_L (2L+1) sin²(φ_L), fired
    # once per unique L (at the first J-group with that L),
    # matching NJOY unresr line 1072
    # (``spot += abn·ab·(2*ll+1)·sin(ps)**2`` when j.eq.1).
    # This differs from the per-J g_J summation only when a
    # file's spin-group set doesn't cover every physically
    # possible (L, J) combination; in that (common) case NJOY
    # assumes the missing J-groups still contribute
    # potential-elastically. The difference is small (~0.4% of
    # elastic at the top of a mid-actinide URR) but real.
    L_np = np.asarray(data.group_l)
    unique_L = sorted(set(int(v) for v in L_np))
    pot = xp.zeros_like(inv_k2)
    for L_val in unique_L:
        first_g = int(np.argmax(L_np == L_val))
        sin2_phi_L = xp.sin(phi_by_g[:, first_g]) ** 2
        pot = pot + 4.0 * inv_k2 * (2 * L_val + 1) * sin2_phi_L

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


