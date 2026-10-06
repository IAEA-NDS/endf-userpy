"""MF6 LAW=5 charged-particle elastic scattering reconstruction.

ENDF-6 Formats Manual Section 6.2.7. The angular distribution is
written as

    dsigma/dOmega(mu, E) = sigma_c(mu, E) + sigma_i(mu, E)
                        + sigma_n(mu, E)

where ``sigma_c`` is the pure Coulomb (Rutherford) term,
``sigma_i`` the Coulomb-nuclear interference term, and ``sigma_n``
the pure nuclear term. The file stores the latter two through
``LTP`` dependent representations; the Coulomb term is always
computed from the point-wise formula using the projectile and
target charges, projectile mass, and incident laboratory energy.

**Scope of this module.** Only ``LTP=1`` with ``LIDP=0`` (nuclear
amplitude expansion, distinguishable particles) is reconstructed.
``LIDP=1`` (identical particles, e.g. p+p) and the tabulated
``LTP in {2, 12, 14, 15}`` forms raise ``NotImplementedError``
with a pointer to the follow-up issue. Covers 6 of 49 MF6 LAW=5
subsections in the ENDF/B-VIII.0 incident-proton sublibrary; the
tabulated forms cover the remaining 42.

Return convention: ``dsigma/dmu`` in barns, i.e. the angular
distribution of ejectile cosines rather than solid angles, so
that the result composes with the per-mu interpolation grid used
elsewhere in the DDX pipeline. The MF3/MT=2 cross section is
forced to 1.0 for ``LTP=1``/``LTP=2`` by the ENDF-6 convention
(manual Section 6.2.7, paragraph following eq. 6.13), so the
pipeline composition ``MF3 * angdist`` is already in barns.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..primitives import array_ns
from ..primitives.conversion import (
    compute_r2,
    convert_angcos_to_cmsys,
    convert_angdist_to_labsys,
)
from ..primitives.interpolation import endf_interp1d
from ..primitives.physical_constants import (
    AMU_TO_MEV,
    FINE_STRUCTURE_ALPHA,
    HBARC_MEV_FM,
    PARTICLE_MASSES_AMU,
)

# Neutron mass in amu. ENDF-6 stores AWR, AWP in "neutron mass
# units"; the Rutherford / Sommerfeld formulas take masses in amu,
# so AWP must be multiplied by this constant before being passed
# into the k and eta helpers. Matches NJOY2016 acefc.f90::coul's
# ``ai = awp * amassn`` conversion.
_NEUTRON_MASS_AMU = PARTICLE_MASSES_AMU['n']


# ---- Kinematic helpers.


def _target_charge_from_za(za):
    """Return the integer target charge Z from the ENDF ZA number
    (ZA = 1000 * Z + A).
    """
    return int(float(za)) // 1000


def _projectile_charge_from_zap(zap):
    """Return the integer projectile charge Z from the ENDF ZAP
    number of the outgoing particle (for MF6 LAW=5 elastic the
    outgoing particle is the same species as the incident, so
    ZAP = ZAI).
    """
    return int(float(zap)) // 1000


def _sommerfeld_eta(z1, z2, m1_amu, e_lab_ev, xp):
    """Dimensionless Sommerfeld parameter.

    eta = z1 * z2 * alpha * sqrt(m1 c^2 / (2 E_cm))
        = z1 * z2 * alpha * sqrt(m1 * u / (2 E_lab))   [eq. 6.12]

    The second equality uses E_cm = E_lab * A/(1+A) and
    m_reduced = m1 * A/(1+A), so the (A/(1+A)) factor cancels and
    the manual expresses eta purely in terms of the incident lab
    energy. The result is backend-agnostic through ``xp``.
    """
    e_mev = e_lab_ev * 1e-6
    return z1 * z2 * FINE_STRUCTURE_ALPHA * xp.sqrt(
        m1_amu * AMU_TO_MEV / (2.0 * e_mev)
    )


def _cm_wavenumber_per_sqrt_barn(a_ratio, m1_amu, e_lab_ev, xp):
    """CM-frame wave number in units of inverse sqrt(barn).

    k = (A / (1 + A)) * sqrt(2 * m1 * c^2 * E_lab) / (hbar c)   [eq. 6.11]

    where A = target/projectile mass ratio. The ENDF manual quotes
    k in barn^(-1/2) by applying a 10^-14 conversion from inverse
    cm; we equivalently convert fm^-1 -> barn^(-1/2) by the exact
    factor 10 (1 fm^-1 = 10 barn^(-1/2) because 1 fm^2 = 10^-2
    barn).
    """
    e_mev = e_lab_ev * 1e-6
    k_per_fm = (
        (a_ratio / (1.0 + a_ratio))
        * xp.sqrt(2.0 * m1_amu * AMU_TO_MEV * e_mev)
        / HBARC_MEV_FM
    )
    return k_per_fm * 10.0


# ---- Pure Coulomb (Rutherford) differential cross sections.


def _sigma_coulomb_distinguishable(mu, eta, k):
    """Pointwise Rutherford dsigma/dOmega(mu, E) for distinguishable
    particles [manual eq. 6.9]:

        sigma_cd = eta^2 / (k^2 * (1 - mu)^2)       (barns/sr)

    Diverges at ``mu == 1``. The caller is responsible for keeping
    the query grid away from ``mu == 1`` or for masking the
    singularity.
    """
    return eta ** 2 / (k ** 2 * (1.0 - mu) ** 2)


def _sigma_coulomb_identical(mu, eta, k, spi, xp):
    """Pointwise Rutherford dsigma/dOmega(mu, E) for IDENTICAL
    particles [manual eq. 6.10]:

        sigma_ci = (2 eta^2 / (k^2 (1 - mu^2))) *
                   [ (1 + mu^2) / (1 - mu^2)
                     + ((-1)^(2 s) / (2 s + 1))
                       * cos(eta * ln((1 + mu) / (1 - mu))) ]
                                                        (barns/sr)

    ``spi`` is the particle spin ``s`` (e.g. 1/2 for p+p elastic);
    the ``(-1)^(2 s)`` sign flips between even- and odd-integer-
    multiplicity spin cases. Diverges at ``mu == +/- 1`` through
    the ``(1 - mu^2)`` denominators, and the ``cos`` term oscillates
    rapidly near those limits because of the ``log((1+mu)/(1-mu))``
    singularity. The caller is responsible for keeping the query
    grid away from ``mu == +/- 1``.

    Backend-agnostic through ``xp``.
    """
    one_minus_mu2 = 1.0 - mu ** 2
    two_s_int = int(round(2.0 * spi))
    sign = -1.0 if (two_s_int & 1) == 1 else 1.0
    cos_term = xp.cos(eta * xp.log((1.0 + mu) / (1.0 - mu)))
    bracket = (
        (1.0 + mu ** 2) / one_minus_mu2
        + sign / (2.0 * spi + 1.0) * cos_term
    )
    return (2.0 * eta ** 2 / (k ** 2 * one_minus_mu2)) * bracket


# ---- LTP=1 nuclear amplitude expansion.


def _unpack_ltp1_distinguishable(a_arr, nl):
    """Split the LTP=1 LIDP=0 LIST record into (b_coeffs, a_coeffs).

    Layout per manual Section 6.2.7:

        A = [b_0, b_1, ..., b_{2NL}, Re(a_0), Im(a_0),
             Re(a_1), Im(a_1), ..., Re(a_NL), Im(a_NL)]

    with ``NW = 4 * NL + 3`` total entries. Returns a real-valued
    ``b`` array of length ``2*NL + 1`` and a complex-valued ``a``
    array of length ``NL + 1``.
    """
    nb = 2 * nl + 1
    na = nl + 1
    expected = nb + 2 * na
    if a_arr.size != expected:
        raise ValueError(
            f'LTP=1 LIDP=0 record has {a_arr.size} entries; '
            f'expected 4*NL+3 = {expected} for NL={nl}.'
        )
    b = a_arr[:nb]
    a_flat = a_arr[nb:]
    a_re = a_flat[0::2]
    a_im = a_flat[1::2]
    a_complex = a_re + 1j * a_im
    return b, a_complex


def _unpack_ltp1_identical(a_arr, nl):
    """Split the LTP=1 LIDP=1 LIST record into (b_coeffs, a_coeffs).

    Layout per manual Section 6.2.7:

        A = [b_0, b_1, ..., b_NL, Re(a_0), Im(a_0),
             Re(a_1), Im(a_1), ..., Re(a_NL), Im(a_NL)]

    with ``NW = 3 * NL + 3`` total entries. The ``b_l`` are
    coefficients of ``P_{2l}(mu)`` (even Legendre only; manual
    eq. 6.14 pure-nuclear sum), so there are ``NL + 1`` of them.
    The ``a_l`` are complex, same as the LIDP=0 case.
    """
    nb = nl + 1
    na = nl + 1
    expected = nb + 2 * na
    if a_arr.size != expected:
        raise ValueError(
            f'LTP=1 LIDP=1 record has {a_arr.size} entries; '
            f'expected 3*NL+3 = {expected} for NL={nl}.'
        )
    b = a_arr[:nb]
    a_flat = a_arr[nb:]
    a_re = a_flat[0::2]
    a_im = a_flat[1::2]
    a_complex = a_re + 1j * a_im
    return b, a_complex


def _legendre_poly_table(mu, lmax, xp):
    """Return the Legendre polynomials ``P_0(mu) ... P_lmax(mu)``
    stacked along a leading axis of size ``lmax + 1``.

    ``mu`` may have any rank; the output has shape
    ``(lmax + 1, *mu.shape)``. 1-D ``mu`` is the CM-frame
    reconstruction shape; 2-D ``mu`` of shape ``(n_e, n_mu)`` is
    the LAB-frame shape after the per-Ein ``mu_LAB -> mu_CM``
    kinematic map (one CM cosine per (E, mu) query point).

    Bonnet recurrence, built as a Python list of rows and stacked
    once at the end; tracer-safe under ``xp.name == 'jax'`` end-to-end.
    """
    n = lmax + 1
    if n == 0:
        return xp.zeros((0,) + tuple(mu.shape), dtype=mu.dtype)
    p0 = xp.ones_like(mu)
    if n == 1:
        return xp.stack([p0], axis=0)
    rows = [p0, mu]
    for ll in range(1, lmax):
        p_next = (
            (2 * ll + 1) * mu * rows[-1] - ll * rows[-2]
        ) / (ll + 1)
        rows.append(p_next)
    return xp.stack(rows, axis=0)


def _reconstruct_ltp1_lidp0_single_ein(b, a_complex, mu, eta, k, xp):
    """Reconstruct dsigma/dOmega(mu, E) at one incident energy for
    the LTP=1 LIDP=0 nuclear amplitude expansion [manual eq. 6.13]:

        sigma_e(mu) = sigma_c(mu)
                    - (2 eta / (1 - mu))
                      * Re{ exp(i eta ln((1 - mu) / 2))
                            * sum_{l=0}^{NL} (2l+1)/2 * a_l * P_l(mu) }
                    + sum_{l=0}^{2 NL} (2l+1)/2 * b_l * P_l(mu)

    ``b`` is a real array of length ``2*NL + 1``; ``a_complex`` is a
    complex array of length ``NL + 1``. ``mu`` has shape ``(nmu,)``.
    ``eta`` and ``k`` are scalar at this Ein. Returns the b/sr
    differential cross section at every ``mu`` as a real array of
    shape ``(nmu,)``.

    Singularity: ``sigma_c`` diverges at ``mu == 1``. The expansion
    terms also contain ``1/(1-mu)`` through the interference. The
    caller must keep ``mu`` away from unity.
    """
    nl = a_complex.shape[0] - 1
    lmax = max(nl, 2 * nl)
    p_table = _legendre_poly_table(mu, lmax, xp)
    # Interference term: Re{ exp(i eta ln((1-mu)/2)) * sum_l (2l+1)/2 a_l P_l(mu) }
    #                   * (-2 eta / (1 - mu))
    l_range_a = xp.arange(nl + 1, dtype=mu.dtype)
    weights_a = (2.0 * l_range_a + 1.0) / 2.0
    # Sum_l (2l+1)/2 * a_l * P_l(mu)   shape (nmu,) complex
    a_weighted = (weights_a * a_complex)[:, None] * p_table[: nl + 1, :]
    sum_a = a_weighted.sum(axis=0)
    # Coulomb phase factor exp(i eta ln((1-mu)/2))
    one_minus_mu_half = (1.0 - mu) / 2.0
    phase = xp.exp(1j * eta * xp.log(one_minus_mu_half))
    interference = -(2.0 * eta / (1.0 - mu)) * (phase * sum_a).real
    # Pure nuclear term: sum_l (2l+1)/2 * b_l * P_l(mu), l up to 2*NL
    l_range_b = xp.arange(2 * nl + 1, dtype=mu.dtype)
    weights_b = (2.0 * l_range_b + 1.0) / 2.0
    nuclear = ((weights_b * b)[:, None] * p_table[: 2 * nl + 1, :]).sum(axis=0)
    # Coulomb
    coulomb = _sigma_coulomb_distinguishable(mu, eta, k)
    return coulomb + interference + nuclear


def _reconstruct_ltp1_lidp1_vectorized(
    coef_at_e, mu_bc, eta_e, k_e, nl, spi, xp,
):
    """Vectorised reconstruction of eq. 6.14 (nuclear amplitude
    Legendre expansion for IDENTICAL particles, LIDP=1).

    ``coef_at_e`` has shape ``(n_e, 3*NL+3)`` and carries the
    coefficient array ``[b_0, ..., b_NL, Re(a_0), Im(a_0), ...,
    Re(a_NL), Im(a_NL)]`` already section-level interpolated.
    ``mu_bc`` has shape ``(n_e, n_mu)``: CM-frame cosines where the
    reconstruction is evaluated. ``eta_e``, ``k_e`` are per-E
    scalars; ``spi`` is the particle spin ``s`` (file-level, Python
    static float). Returns ``dsigma/dOmega`` of shape
    ``(n_e, n_mu)`` in CM, b/sr.

    The formula per manual eq. 6.14:

        sigma_ei(mu) = sigma_ci(mu)
                     - (2 eta / (1 - mu^2))
                       * Re{ sum_l [(1 + mu) exp(i eta ln((1-mu)/2))
                             + (-1)^l (1 - mu) exp(i eta ln((1+mu)/2))]
                             * (2 l + 1) / 2 * a_l(E) * P_l(mu) }
                     + sum_l (4 l + 1) / 2 * b_l(E) * P_{2l}(mu)

    Matches NJOY2016 ``acefc.f90::coul`` LIDP=1 branch (lines
    8366-8386) line-for-line: ``cs1`` is the ``sum_l (2l+1)/2 a_l
    P_l``, ``cs2`` the alternating-sign ``sum_l (-1)^l (2l+1)/2
    a_l P_l``; the pure-nuclear ``sigr`` uses even-only Legendre
    with weight ``(4l+1)/2``.

    Backend-agnostic: every operation routes through ``xp``.
    """
    nb = nl + 1
    b = coef_at_e[:, :nb]                       # (n_e, NL+1)
    a_re = coef_at_e[:, nb::2]                   # (n_e, NL+1)
    a_im = coef_at_e[:, nb + 1::2]              # (n_e, NL+1)
    a_complex = a_re + 1j * a_im                 # (n_e, NL+1)
    lmax_pure = 2 * nl                            # highest P_{2l} order
    p_table = _legendre_poly_table(mu_bc, lmax_pure, xp)  # (lmax_pure+1, n_e, n_mu)

    # Pure-nuclear sum sigr = sum_l (4l+1)/2 b_l P_{2l}(mu)  (even Legendre only).
    # Extract rows [0, 2, 4, ..., 2*NL] from the P_l table.
    even_rows = p_table[0::2]                     # (NL+1, n_e, n_mu)
    l_vec = xp.arange(nb, dtype=mu_bc.dtype)
    weights_b = (4.0 * l_vec + 1.0) / 2.0         # (NL+1,)
    nuclear = xp.einsum('el,len->en', b * weights_b, even_rows)

    # Interference sums.
    l_vec_a = xp.arange(nl + 1, dtype=mu_bc.dtype)
    weights_a = (2.0 * l_vec_a + 1.0) / 2.0
    # cs1 = sum_l (2l+1)/2 a_l P_l(mu)
    cs1 = xp.einsum('el,len->en', a_complex * weights_a,
                    p_table[: nl + 1])
    # cs2 = sum_l (-1)^l (2l+1)/2 a_l P_l(mu)
    alt_sign = (-1.0) ** l_vec_a                  # (NL+1,)
    cs2 = xp.einsum('el,len->en', a_complex * weights_a * alt_sign,
                    p_table[: nl + 1])

    # Coulomb phases exp(i eta ln((1 +/- mu) / 2)).
    eta_bc = eta_e[:, None]
    k_bc = k_e[:, None]
    one_minus_mu_half = (1.0 - mu_bc) / 2.0
    one_plus_mu_half = (1.0 + mu_bc) / 2.0
    phase1 = xp.exp(1j * eta_bc * xp.log(one_minus_mu_half))
    phase2 = xp.exp(1j * eta_bc * xp.log(one_plus_mu_half))

    one_minus_mu2 = 1.0 - mu_bc ** 2
    interference_sum = (
        (1.0 + mu_bc) * phase1 * cs1
        + (1.0 - mu_bc) * phase2 * cs2
    )
    interference = -(2.0 * eta_bc / one_minus_mu2) * interference_sum.real

    # Identical-particle Rutherford.
    coulomb = _sigma_coulomb_identical(mu_bc, eta_bc, k_bc, spi, xp)
    return coulomb + interference + nuclear


def _reconstruct_ltp_tabulated_vectorized(
    coef_at_e, mu_bc, eta_e, k_e, sigma_ni_e, nl, ltp, xp,
):
    """Vectorised reconstruction of eq. 6.19/6.20 (tabulated
    nuclear-plus-interference form) for LAW=5 LTP in {12, 14, 15},
    LIDP=0.

    ``coef_at_e`` has shape ``(n_e, 2*NL)`` and carries the per-Ein
    coefficient array ``[mu_1, p_NI(mu_1), mu_2, p_NI(mu_2), ...,
    mu_NL, p_NI(mu_NL)]`` already section-level interpolated.
    ``mu_bc`` has shape ``(n_e, n_mu)``: CM-frame cosines where the
    reconstruction is evaluated. ``sigma_ni_e`` has shape
    ``(n_e,)`` and carries MF3/MT=2 interpolated at each Ein (the
    file-side ``sigma_NI`` per manual eq. 6.19). ``eta_e``,
    ``k_e`` same as the LTP=1 path.

    Returns the CM-frame ``dsigma/dOmega`` of shape ``(n_e, n_mu)``:

        sigma_e(mu, E) = sigma_c(mu, E) + sigma_NI(E) * p_NI(mu, E)

    where ``p_NI`` is interpolated per the stored LTP code:
    LTP=12 or 15 use linear-in-mu interpolation of ``p_NI``;
    LTP=14 uses linear-in-mu interpolation of ``ln(p_NI)`` with the
    exponential re-applied. ``p_NI`` evaluates to zero outside the
    tabulated ``[mu_min, mu_max]`` range (manual eq. 6.20), leaving
    only the pointwise Coulomb term there.
    """
    # Unpack (mu_k, p_NI_k) pairs. coef_at_e[:, 0::2] = mu values,
    # coef_at_e[:, 1::2] = p_NI values. For a given Ein we expect
    # the mu grid to be the same across the NE grid (checked by the
    # caller via a NL-uniform guard), so collapse along the Ein axis.
    mu_knots = coef_at_e[:, 0::2]        # (n_e, NL)
    p_knots = coef_at_e[:, 1::2]         # (n_e, NL)
    if int(ltp) == 14:
        # Clamp strictly-positive floor so log is finite; a zero
        # p_NI entry would otherwise feed log(0) = -inf into the
        # interpolation. Downstream the result is masked outside the
        # tabulated mu range anyway, so a floor-replaced sample does
        # not leak into physical regions.
        p_floor = xp.where(p_knots > 0.0, p_knots, 1e-300)
        p_interp_src = xp.log(p_floor)
    else:
        p_interp_src = p_knots

    # Linear-in-mu interpolation of ``p_interp_src`` evaluated at
    # ``mu_bc[e, m]``. For each Ein row this is a 1D interpolation
    # on the (mu_knots[e], p_interp_src[e]) pair; vectorise with
    # xp.searchsorted on the NL knot grid plus per-element indexing.
    # The ENDF convention stores mu_knots in ascending order from
    # mu_min to mu_max for the tabulated LTP forms.
    n_e = mu_bc.shape[0]
    # Per-row searchsorted into the mu_knots grid.
    # Use a padded index to make xp.take_along_axis safe at the edges;
    # mu samples outside [mu_min, mu_max] are masked back to zero below.
    # searchsorted expects a sorted 1D array; apply row-wise via a
    # Python list comp (NE is small, typically O(100)).
    pdf_bc = xp.stack([
        _row_linear_interp_1d(
            mu_knots[i], p_interp_src[i], mu_bc[i], xp,
        )
        for i in range(n_e)
    ])
    if int(ltp) == 14:
        pdf_bc = xp.exp(pdf_bc)

    # Mask outside the tabulated mu range (manual eq. 6.20: p_NI
    # = 0 outside [mu_min, mu_max]). Use the Ein-independent knot
    # extremes along axis -1.
    mu_min = mu_knots[:, :1]              # (n_e, 1)
    mu_max = mu_knots[:, -1:]             # (n_e, 1)
    in_range = (mu_bc >= mu_min) & (mu_bc <= mu_max)
    pdf_bc = xp.where(in_range, pdf_bc, xp.zeros_like(pdf_bc))

    # Compose: sigma_e = sigma_c + sigma_NI(E) * p_NI(mu, E).
    coulomb = eta_e[:, None] ** 2 / (
        k_e[:, None] ** 2 * (1.0 - mu_bc) ** 2
    )
    return coulomb + sigma_ni_e[:, None] * pdf_bc


def _row_linear_interp_1d(x_knots, y_knots, x_query, xp):
    """1D linear interpolation along ``x_knots`` evaluated at
    ``x_query``, clamped at the knot extremes (no extrapolation).

    All inputs are 1D xp-native arrays. Backend-agnostic.
    Operations are tracer-safe: ``xp.searchsorted`` + per-element
    arithmetic without Python-side branches.
    """
    idx = xp.searchsorted(x_knots, x_query, side='right')
    n = x_knots.shape[0]
    idx = xp.where(idx < 1, 1, idx)
    idx = xp.where(idx > n - 1, n - 1, idx)
    i1 = (idx - 1).astype(xp.int32)
    i2 = idx.astype(xp.int32)
    x1 = xp.take(x_knots, i1, axis=0)
    x2 = xp.take(x_knots, i2, axis=0)
    y1 = xp.take(y_knots, i1, axis=0)
    y2 = xp.take(y_knots, i2, axis=0)
    denom = xp.where(x2 == x1, 1.0, x2 - x1)
    slope = (y2 - y1) / denom
    return y1 + slope * (x_query - x1)


def _reconstruct_ltp1_lidp0_vectorized(
    coef_at_e, mu_bc, eta_e, k_e, nl, xp,
):
    """Vectorised sibling of :func:`_reconstruct_ltp1_lidp0_single_ein`
    broadcasting over both an incident-energy axis and a mu axis.

    ``coef_at_e`` has shape ``(n_e, 4*NL+3)`` and carries the
    coefficient array ``[b_0, ..., b_{2NL}, Re(a_0), Im(a_0), ...,
    Re(a_NL), Im(a_NL)]`` at each requested Ein (already
    section-level interpolated). ``mu_bc`` carries the CM-frame
    cosine for every (E, query) point and has shape
    ``(n_e, n_mu)``: either the user query broadcast against the
    E axis (CM-frame caller) or the per-E ``mu_CM(mu_LAB)`` map
    produced by :func:`convert_angcos_to_cmsys` (LAB-frame caller).
    ``eta_e`` and ``k_e`` have shape ``(n_e,)``. Returns
    ``dsigma/dOmega`` in CM at every ``mu_bc[i, j]`` as an array
    of shape ``(n_e, n_mu)``.

    Backend-agnostic: every operation routes through ``xp``. Under
    ``xp.name == 'jax'`` the function is both ``jax.jit``-safe and
    ``jax.grad``-transparent wrt ``mu_bc``, ``eta_e``, ``k_e``, and
    the ``coef_at_e`` leaves.
    """
    nb = 2 * nl + 1
    b = coef_at_e[:, :nb]                       # (n_e, 2*NL+1)
    a_re = coef_at_e[:, nb::2]                   # (n_e, NL+1)
    a_im = coef_at_e[:, nb + 1::2]              # (n_e, NL+1)
    a_complex = a_re + 1j * a_im                 # (n_e, NL+1)
    lmax = max(nl, 2 * nl)
    p_table = _legendre_poly_table(mu_bc, lmax, xp)  # (lmax+1, n_e, n_mu)
    # Pure nuclear sum: sum_l (2l+1)/2 b_l(E) P_l(mu_bc[e, m])
    # Shape of einsum: (n_e, 2*NL+1) · (2*NL+1, n_e, n_mu) -> (n_e, n_mu)
    # Contract over the L axis only; the Ein axis of b and p_table
    # tracks together ("el,len->en").
    l_range_b = xp.arange(nb, dtype=mu_bc.dtype)
    weights_b = (2.0 * l_range_b + 1.0) / 2.0
    nuclear = xp.einsum(
        'el,len->en', b * weights_b, p_table[:nb],
    )
    # Interference sum: complex (n_e, n_mu)
    l_range_a = xp.arange(nl + 1, dtype=mu_bc.dtype)
    weights_a = (2.0 * l_range_a + 1.0) / 2.0
    cs = xp.einsum(
        'el,len->en', a_complex * weights_a, p_table[: nl + 1],
    )
    # Coulomb phase factor exp(i eta(E) log((1-mu)/2))
    eta_bc = eta_e[:, None]
    k_bc = k_e[:, None]
    one_minus_mu_half = (1.0 - mu_bc) / 2.0
    phase = xp.exp(1j * eta_bc * xp.log(one_minus_mu_half))
    interference = -(2.0 * eta_bc / (1.0 - mu_bc)) * (phase * cs).real
    # Rutherford Coulomb cross section
    coulomb = eta_bc ** 2 / (k_bc ** 2 * (1.0 - mu_bc) ** 2)
    return coulomb + interference + nuclear


# ---- Public API.


def get_angdist_from_subsec_law5(
    endf_dict, mt, subsec_num, energies_in, angle_cosines_out, to_lab,
    xp=None,
):
    """Angular distribution for MF6 LAW=5 charged-particle elastic
    scattering (manual Section 6.2.7).

    Returns ``dsigma/dmu(mu, E)`` in barns, i.e. ``2 pi *
    dsigma/dOmega``, at every requested incident energy and
    cosine. The ENDF-6 convention forces MF3/MT=2 to 1.0 for LTP=1
    (manual paragraph following eq. 6.13), so the DDX pipeline's
    ``MF3 * angdist`` composition is already the physical
    differential cross section.

    Current scope (issue #264 first increment plus #334): ``LTP == 1``
    and ``LIDP == 0``. Covers 6 of 49 MF6 LAW=5 subsections in the
    ENDF/B-VIII.0 incident-proton sublibrary. ``LIDP == 1``
    (identical particles, p+p only) and the tabulated LTP forms
    (12, 14, 15) raise ``NotImplementedError`` for the caller to
    handle. The stored frame is CM for every LAW=5 subsection in
    the public neutron-adjacent corpora we checked (LCT=2 on all
    49); ``to_lab=True`` applies the two-body elastic Jacobian
    (``conversion.compute_r2`` + ``convert_angcos_to_cmsys`` +
    ``convert_angdist_to_labsys``) to return a LAB-frame
    ``dsigma/dmu``. ``to_lab=False`` returns the stored CM-frame
    distribution directly. LCT=1 LAW=5 (file already in LAB)
    would need a different branch and is rejected explicitly.

    Backend-agnostic through ``xp``; the Legendre recurrence,
    Coulomb-phase complex exponential, and interference sum all
    run on the chosen backend. The entire pipeline is
    ``@jax.jit``-safe and ``jax.grad``-transparent wrt
    ``energies_in``, ``angle_cosines_out``, and the file-stored
    ``a_l`` / ``b_l`` coefficients (when the caller injects
    tracers into ``subsec['A']`` for file-leaf autodiff). No
    Python-side conversion of query or coefficient values to
    concrete floats, no Python-side control flow on tracers; the
    below-threshold mask uses ``xp.where``.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')

    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    law = int(subsec['LAW'])
    if law != 5:
        raise ValueError(
            f'MF6/MT{mt} subsection {subsec_num} has LAW={law}, '
            f'not 5; call the LAW={law} handler instead.'
        )
    lct = int(sec['LCT'])
    if lct != 2:
        raise NotImplementedError(
            f'MF6/MT{mt} LAW=5 with LCT={lct} is not implemented. '
            f'Every incident-proton evaluation in the ENDF/B-VIII.0 '
            f'sublibrary uses LCT=2 (CM frame); LCT=1 or LCT=3 '
            f'variants would need a frame transform (issue #264).'
        )

    lidp = int(subsec['LIDP'])
    if lidp == 1 and (set(int(v) for v in subsec['LTP'].values()) - {1}):
        # Identical-particle tabulated LTP forms (hypothetical: no
        # file in the ENDF/B-VIII.0 proton sublibrary uses this
        # combination). Deferred: eq. 6.14 is LTP=1 only, and the
        # tabulated eq. 6.19-6.20 forms are defined for LIDP=0 in
        # the manual's current working examples.
        raise NotImplementedError(
            f'MF6/MT{mt} LAW=5 LIDP=1 with tabulated LTP (12/14/15) '
            f'is not implemented; no file in the known neutron-'
            f'adjacent corpora uses this combination.'
        )

    # Supported LTP forms: 1 (nuclear amplitude Legendre expansion,
    # manual eq. 6.13) and 12 / 14 / 15 (tabulated p_NI vs mu,
    # manual eq. 6.19-6.20). LTP=2 (residual-XS Legendre expansion
    # of eq. 6.15-6.18) still raises; no corpus file in the
    # neutron-adjacent sublibraries exercises it.
    ltps = {int(v) for v in subsec['LTP'].values()}
    _SUPPORTED_LTPS = {1, 12, 14, 15}
    unsupported = ltps - _SUPPORTED_LTPS
    if unsupported:
        raise NotImplementedError(
            f'MF6/MT{mt} LAW=5 has LTP values {sorted(unsupported)} '
            f'that are not implemented. Supported: '
            f'{sorted(_SUPPORTED_LTPS)}.'
        )
    if len(ltps) != 1:
        raise NotImplementedError(
            f'MF6/MT{mt} LAW=5 mixes LTP values {sorted(ltps)} '
            f'across the NE grid; the reconstruction assumes a '
            f'single LTP code per subsection. Deferred.'
        )
    ltp = next(iter(ltps))

    # Interpolate per-Ein coefficient arrays onto the requested
    # incident-energy grid. Each per-Ein record carries NL, so the
    # arrays live in different shapes and the simplest approach is
    # a linear interpolation on a NL-static section. All public
    # proton-sublibrary LAW=5 subsections keep NL fixed across the
    # NE grid; reject the variable-NL case explicitly.
    nl_set = {int(v) for v in subsec['NL'].values()}
    if len(nl_set) != 1:
        raise NotImplementedError(
            f'MF6/MT{mt} LAW=5 has NL varying across the NE grid '
            f'({sorted(nl_set)}); interpolating coefficient arrays '
            f'of different lengths is not implemented. Deferred.'
        )
    nl = nl_set.pop()
    if ltp == 1:
        # Per manual Section 6.2.7:
        # LIDP=0 nuclear amplitude expansion stores (2*NL+1) b
        # coefficients + (NL+1) complex a coefficients -> NW=4*NL+3.
        # LIDP=1 (identical particles) stores (NL+1) b coefficients
        # of the EVEN-only Legendre expansion + (NL+1) complex a
        # coefficients -> NW=3*NL+3.
        nw = 4 * nl + 3 if lidp == 0 else 3 * nl + 3
    else:
        # LTP in {12, 14, 15}: tabulated form stores (mu, p_NI) pairs,
        # NW = 2 * NL per manual Section 6.2.7.
        nw = 2 * nl

    # Build the (NE, NW) coefficient matrix xp-natively. The dict
    # values stored under ``subsec['E'][i]`` and ``subsec['A'][i][k]``
    # may be plain Python floats, numpy scalars, or JAX tracers
    # (mesh-knot / file-leaf autodiff pattern); ``xp.asarray`` on a
    # tracer is a no-op that preserves it, and ``xp.stack`` of a
    # list of scalars produces one array that keeps every tracer
    # dependency alive. This replaces the earlier per-element
    # assignment into a numpy array, which collapsed any tracer
    # into a concrete float and silently broke leaf autodiff.
    e_keys = sorted(subsec['E'].keys())
    a_keys = sorted(subsec['A'].keys())
    e_mesh = xp.stack([xp.asarray(subsec['E'][i], dtype=xp.float64)
                       for i in e_keys])
    coef_matrix = xp.stack([
        xp.stack([xp.asarray(subsec['A'][i][k + 1], dtype=xp.float64)
                  for k in range(nw)])
        for i in a_keys
    ])  # (NE, NW)

    # Section-level incident-energy interpolation (TAB2 across E).
    int_arr = np.asarray(subsec.get('INT', [2]), dtype=int)
    nbt_arr = np.asarray(subsec.get('NBT', [len(e_keys)]), dtype=int)

    e_in = xp.asarray(energies_in, dtype=xp.float64)
    mu = xp.asarray(angle_cosines_out, dtype=xp.float64)

    # Target / projectile charges and masses. Z and AWR are
    # file-static integers / floats; AWP is a per-subsection float.
    za_target = int(float(endf_dict[1][451]['ZA']))
    z_target = za_target // 1000
    zap = int(float(subsec['ZAP']))
    z_proj = zap // 1000
    awp = float(subsec['AWP'])
    awr = float(sec['AWR'])
    a_ratio = awr / awp  # amu / neutron-unit factor cancels in the ratio
    m1_amu = awp * _NEUTRON_MASS_AMU  # NJOY-matched conversion (eq. 6.11-6.12)

    # Interpolate coefficients at each requested Ein, column by
    # column. ``endf_interp1d`` under ``xp.name == 'jax'`` routes
    # through the traced-x path, so both ``e_in`` tracers (query
    # autodiff) and ``coef_matrix`` tracers (file-leaf autodiff)
    # propagate; outputs are xp-native.
    coef_at_queries = xp.stack([
        endf_interp1d(
            e_in, e_mesh, coef_matrix[:, col],
            int_arr, nbt_arr, xp=xp,
        )
        for col in range(nw)
    ], axis=-1)  # (n_e, nw)

    # eta, k vectorised over e_in. Below-threshold / zero-energy
    # queries blow up eta and k; clamp e_in with xp.where to a
    # positive stand-in so the arithmetic stays finite, then mask
    # the output to zero on those rows (issue #45 pattern).
    e_safe = xp.where(e_in > 0.0, e_in, xp.ones_like(e_in))
    eta_e = _sommerfeld_eta(z_proj, z_target, m1_amu, e_safe, xp)
    k_e = _cm_wavenumber_per_sqrt_barn(a_ratio, m1_amu, e_safe, xp)

    # Build the broadcast CM-frame cosines mu_bc of shape
    # (n_e, n_mu) that the reconstruction evaluates P_l and the
    # Rutherford formula at.
    #
    # - to_lab=False: user mu is already a CM cosine (LCT=2 file,
    #   caller asked for the stored frame). Broadcast once against
    #   the E axis so every (E, mu) pair maps to the same mu_CM.
    # - to_lab=True: user mu is a LAB cosine; map it to the CM
    #   cosine via the two-body elastic kinematics (eq. 6.2 /
    #   section 6.2 of the ENDF-6 manual). For charged-particle
    #   elastic Q=0 and awi=awp (projectile == ejectile), so
    #   compute_r2 reduces to the pure mass-ratio factor
    #   r^2 = (awr/awp)^2 independent of E. r > 1 for every
    #   neutron-adjacent target heavier than the proton, so the
    #   LAB->CM map is single-valued and the forbidden-angle
    #   mask is physically empty for this scope.
    if to_lab:
        r2 = compute_r2(e_in, awi=awp, awr=awr, awp=awp, q=0.0, xp=xp)
        if lidp == 1:
            # Identical-particle elastic (p+p): both outgoing
            # particles are the same species, so the "ejectile" is
            # inherently ambiguous and the LAB angular range is the
            # forward hemisphere mu_LAB in [0, 1] only. The mass
            # ratio awr/awp is close to unity (not exactly 1 due to
            # the H-1 vs proton AWR/AWP distinction in the file), so
            # the LAB->CM mapping is still single-valued and the
            # primary-branch primitive produces a finite result.
            # However, the LAB density for r ~ 1 picks up contributions
            # from BOTH CM branches that map to the same mu_LAB and
            # our primary-branch primitive returns only one; the
            # LAB-frame output is therefore the single-branch result,
            # not the full identical-particle LAB density. Flag it
            # explicitly so a user plotting LAB-frame p+p elastic
            # knows the stored CM-frame data is more authoritative.
            warnings.warn(
                f'MF6/MT{mt} LAW=5 LIDP=1 (identical particles) with '
                f'to_lab=True returns only the primary CM-branch '
                f'contribution to the LAB-frame angular distribution. '
                f'For identical-particle elastic the physical LAB '
                f'density at each mu_LAB sums contributions from two '
                f'CM branches that map to the same LAB cosine; this '
                f'handler currently returns only one. Query '
                f'to_lab=False for the stored CM-frame distribution, '
                f'which the ENDF manual (Section 6.2.7) uses as the '
                f'authoritative reference for identical-particle '
                f'cases.',
                UserWarning, stacklevel=2,
            )
        mu_cm_bc = convert_angcos_to_cmsys(mu, r2, xp=xp)
    else:
        mu_cm_bc = xp.broadcast_to(mu[None, :], (e_in.shape[0], mu.shape[0]))

    if ltp == 1:
        if lidp == 0:
            sigma_omega_cm = _reconstruct_ltp1_lidp0_vectorized(
                coef_at_queries, mu_cm_bc, eta_e, k_e, nl, xp,
            )  # (n_e, n_mu), dsigma/dOmega in CM frame, b/sr
        else:
            # LIDP=1 identical particles (eq. 6.14). ``SPI`` is the
            # subsection-level particle spin stored by ENDF, needed
            # for the identical-particle Rutherford branch
            # (eq. 6.10).
            spi = float(subsec['SPI'])
            sigma_omega_cm = _reconstruct_ltp1_lidp1_vectorized(
                coef_at_queries, mu_cm_bc, eta_e, k_e, nl, spi, xp,
            )
    else:
        # Tabulated form: fetch MF3/MT=2 ``sigma_NI(E)`` which the
        # manual (eq. 6.19) defines as the integrated NI piece and
        # the file stores point-wise. For LTP=1/2 the file forces
        # MF3 to 1.0; for LTP=12/14/15 it is the real sigma_NI,
        # possibly negative at destructive-interference energies.
        sigma_ni_e = _interp_mf3(endf_dict, mt, e_in, xp)
        sigma_omega_cm = _reconstruct_ltp_tabulated_vectorized(
            coef_at_queries, mu_cm_bc, eta_e, k_e, sigma_ni_e,
            nl, ltp, xp,
        )  # (n_e, n_mu), dsigma/dOmega in CM frame, b/sr

    if to_lab:
        # Apply the CM -> LAB angular Jacobian so the output is a
        # LAB-frame dsigma/dOmega at the user's mu_LAB.
        sigma_omega = convert_angdist_to_labsys(
            mu_cm_bc, sigma_omega_cm, r2, xp=xp,
        )
        # Forbidden LAB angles (NaN from the mu_LAB->mu_CM map) go
        # to zero: the pattern matches the LAW=2 / LAW=4 paths and
        # issue #45 for below-threshold / unreachable-cosine cases.
        sigma_omega = xp.where(
            xp.isnan(sigma_omega), xp.zeros_like(sigma_omega),
            sigma_omega,
        )
    else:
        sigma_omega = sigma_omega_cm

    # Convert from barns/sr to barns/mu (azimuthal symmetry).
    out = sigma_omega * (2.0 * np.pi)
    # Mask below-threshold rows (where the clamp above hid the issue).
    e_pos_mask = (e_in > 0.0)[:, None]
    out = xp.where(e_pos_mask, out, xp.zeros_like(out))

    # Pipeline composition convention: downstream DDX multiplies
    # ``MF3(E) * angdist(mu, E)``. For LTP=1/2 the file forces
    # MF3=1.0 (manual Section 6.2.7) so returning ``sigma_e``
    # directly already produces the right composite. For LTP in
    # {12, 14, 15} the file stores ``sigma_NI`` in MF3, so divide
    # by it here to keep the invariant ``MF3 * angdist = sigma_e``.
    # ``sigma_NI`` can be small or sign-change at destructive-
    # interference energies (p + C-12 crosses zero twice over the
    # file's E grid); clamp |sigma_NI| at a tiny floor so the
    # ratio stays finite, and emit a one-shot warning when any
    # queried E hits the clamp. The composite then goes to zero
    # at that E rather than NaN; the dsigma/dmu reconstruction
    # loses the Coulomb pole in a narrow neighbourhood of the
    # crossing but stays well-behaved everywhere else.
    if ltp != 1:
        _TOL = 1e-30
        abs_sigma_ni = xp.abs(sigma_ni_e)
        sigma_ni_safe = xp.where(
            abs_sigma_ni > _TOL,
            sigma_ni_e,
            xp.where(sigma_ni_e >= 0.0,
                      xp.full_like(sigma_ni_e, _TOL),
                      xp.full_like(sigma_ni_e, -_TOL)),
        )
        out = out / sigma_ni_safe[:, None]
        try:
            degenerate_count = int(xp.sum(abs_sigma_ni <= _TOL))
            if degenerate_count > 0:
                warnings.warn(
                    f'MF6/MT{mt} LAW=5 LTP={ltp}: MF3/MT=2 sigma_NI '
                    f'is near zero (|value| <= {_TOL}) at '
                    f'{degenerate_count} of {int(e_in.shape[0])} '
                    f'queried incident energies. The composite '
                    f'MF3 * angdist goes to zero at those points '
                    f'to avoid NaN; the point-wise Coulomb pole is '
                    f'lost in a narrow neighbourhood of each '
                    f'destructive-interference zero crossing of '
                    f'sigma_NI(E).',
                    UserWarning, stacklevel=2,
                )
        except Exception:
            # Tracer path (jax jit): ``int()`` on a tracer scalar
            # fails. Skip the warning rather than raise; the clamp
            # is still applied.
            pass
    return out


def _interp_mf3(endf_dict, mt, e_in, xp):
    """Interpolate MF3/MT (tabulated cross section) at ``e_in``.

    For LAW=5 LTP in {12, 14, 15}, MF3 carries the file-stored
    ``sigma_NI(E)`` (manual eq. 6.19) needed to compose the
    tabulated nuclear-plus-interference distribution. Backend-
    agnostic through ``xp``; ``e_in`` tracers propagate through
    ``endf_interp1d``'s traced-x path.
    """
    xst = endf_dict[3][mt]['xstable']
    e_mesh_np = np.asarray(xst['E'], dtype=float)
    xs_np = np.asarray(xst['xs'], dtype=float)
    int_arr = np.asarray(xst.get('INT', [2]), dtype=int)
    nbt_arr = np.asarray(xst.get('NBT', [e_mesh_np.size]), dtype=int)
    return endf_interp1d(
        e_in, xp.asarray(e_mesh_np), xp.asarray(xs_np),
        int_arr, nbt_arr, xp=xp,
    )
