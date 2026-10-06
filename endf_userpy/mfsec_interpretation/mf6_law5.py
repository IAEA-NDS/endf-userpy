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


def _legendre_poly_table(mu, lmax, xp):
    """Return the Legendre polynomials ``P_0(mu) ... P_lmax(mu)``
    stacked along a leading axis of size ``lmax + 1``. ``mu`` is a
    1-D array; the output has shape ``(lmax + 1, mu.size)``.

    Bonnet recurrence. Backend-agnostic through ``xp``.
    """
    n = lmax + 1
    out = xp.zeros((n, mu.shape[0]), dtype=mu.dtype)
    if n == 0:
        return out
    out = out.at[0].set(xp.ones_like(mu)) if hasattr(out, 'at') \
        else _np_set_row(out, 0, xp.ones_like(mu))
    if n == 1:
        return out
    out = out.at[1].set(mu) if hasattr(out, 'at') \
        else _np_set_row(out, 1, mu)
    for ll in range(1, lmax):
        p_prev = out[ll - 1]
        p_curr = out[ll]
        p_next = ((2 * ll + 1) * mu * p_curr - ll * p_prev) / (ll + 1)
        out = out.at[ll + 1].set(p_next) if hasattr(out, 'at') \
            else _np_set_row(out, ll + 1, p_next)
    return out


def _np_set_row(arr, i, row):
    arr = np.asarray(arr).copy()
    arr[i] = np.asarray(row)
    return arr


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

    Current scope (issue #264 first increment): ``LTP == 1`` and
    ``LIDP == 0``. Covers 6 of 49 MF6 LAW=5 subsections in the
    ENDF/B-VIII.0 incident-proton sublibrary. ``LIDP == 1``
    (identical particles, p+p only) and the tabulated LTP forms
    (12, 14, 15) raise ``NotImplementedError`` for the caller to
    handle. The frame is CM for every LAW=5 subsection in the
    public neutron-adjacent corpora we checked (LCT=2 on all 49),
    so to_lab=True without a CM-to-LAB conversion matches the file
    convention; LCT=1 LAW=5 would be rejected explicitly.

    Backend-agnostic through ``xp``; the Legendre recurrence,
    Coulomb-phase complex exponential, and interference sum all
    run on the chosen backend. ``jax.grad`` wrt the stored ``a_l``
    and ``b_l`` coefficients propagates through the reconstruction.
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
    if lidp == 1:
        raise NotImplementedError(
            f'MF6/MT{mt} LAW=5 with LIDP=1 (identical particles, '
            f'e.g. p+p) is not implemented in the first #264 '
            f'increment. Requires the eq. 6.14 reconstruction '
            f'(two Coulomb-phase terms, even-only Legendre pure-'
            f'nuclear expansion). Deferred as #264 follow-up.'
        )

    # Only LTP=1 supported in this increment.
    ltps = {int(v) for v in subsec['LTP'].values()}
    unsupported = ltps - {1}
    if unsupported:
        raise NotImplementedError(
            f'MF6/MT{mt} LAW=5 has LTP values {sorted(unsupported)} '
            f'(tabulated nuclear-plus-interference forms per '
            f'manual eq. 6.19/6.20). The first #264 increment '
            f'implements only LTP=1 (nuclear amplitude Legendre '
            f'expansion). Tabulated LTP=12/14/15 cover 42 of 49 '
            f'proton-sublibrary files and are a #264 follow-up.'
        )

    # Interpolate per-Ein coefficient arrays onto the requested
    # incident-energy grid. Each per-Ein record carries NL, so the
    # arrays live in different shapes and the simplest approach is
    # a linear interpolation on a NL-static section. All public
    # LTP=1 LIDP=0 proton files we checked keep NL fixed across
    # the NE grid; reject the variable-NL case explicitly.
    nl_set = {int(v) for v in subsec['NL'].values()}
    if len(nl_set) != 1:
        raise NotImplementedError(
            f'MF6/MT{mt} LAW=5 has NL varying across the NE grid '
            f'({sorted(nl_set)}); interpolating coefficient arrays '
            f'of different lengths is not implemented. Deferred.'
        )
    nl = nl_set.pop()

    e_mesh = np.array(
        [subsec['E'][i] for i in sorted(subsec['E'].keys())],
        dtype=float,
    )
    # (NE, NW) coefficient matrix
    nw = 4 * nl + 3
    coef_matrix = np.zeros((e_mesh.size, nw), dtype=float)
    for row_i, ei in enumerate(sorted(subsec['A'].keys())):
        a_row = subsec['A'][ei]
        for k_idx in range(nw):
            coef_matrix[row_i, k_idx] = a_row[k_idx + 1]  # 1-based keys

    # Section-level incident-energy interpolation (TAB2 across E).
    int_arr = np.asarray(subsec.get('INT', [2]), dtype=int)
    nbt_arr = np.asarray(subsec.get('NBT', [e_mesh.size]), dtype=int)

    e_in = xp.asarray(energies_in, dtype=xp.float64)
    mu = xp.asarray(angle_cosines_out, dtype=xp.float64)
    n_e = e_in.shape[0]
    n_mu = mu.shape[0]

    # Target / projectile charges and masses.
    za_target = int(float(endf_dict[1][451]['ZA']))
    z_target = za_target // 1000
    zap = int(float(subsec['ZAP']))
    z_proj = zap // 1000
    awp = float(subsec['AWP'])         # projectile mass in neutron units
    awr = float(sec['AWR'])            # target mass in neutron units
    # Target/projectile mass ratio A = AWR / AWP; the AMU/neutron
    # unit factor cancels so the ratio is the same in either system.
    a_ratio = awr / awp
    # Convert projectile mass to amu for the Sommerfeld / wave-number
    # formulas (manual eq 6.11-6.12 give m1 in amu). Matches NJOY's
    # ``ai = awp * amassn`` in acefc.f90::coul.
    m1_amu = awp * _NEUTRON_MASS_AMU

    # Interpolate coefficients at each requested Ein; keep this on
    # numpy for the array-length-varying case even when xp is jax.
    # (The static-NL branch above has fixed coef shape, so the
    # interpolation is a plain per-column tab1 lookup.)
    e_in_np = np.asarray(e_in, dtype=float)
    coef_at_queries = np.zeros((n_e, nw), dtype=float)
    for col in range(nw):
        coef_at_queries[:, col] = np.asarray(endf_interp1d(
            e_in_np, e_mesh, coef_matrix[:, col],
            int_arr, nbt_arr,
        ))

    out = xp.zeros((n_e, n_mu), dtype=xp.float64)
    for i in range(n_e):
        ei_val = float(e_in_np[i])
        if ei_val <= 0.0:
            # Below-threshold / zero-energy query: eta and k blow up;
            # return zero like the MF4/MF6 LAW=2 paths do for
            # kinematic failures (issue #45 pattern).
            continue
        eta_i = float(_sommerfeld_eta(z_proj, z_target, m1_amu,
                                       ei_val, np))
        k_i = float(_cm_wavenumber_per_sqrt_barn(a_ratio, m1_amu,
                                                   ei_val, np))
        b_i, a_i = _unpack_ltp1_distinguishable(
            coef_at_queries[i], nl,
        )
        b_xp = xp.asarray(b_i)
        a_xp = xp.asarray(a_i)
        row_sigma_omega = _reconstruct_ltp1_lidp0_single_ein(
            b_xp, a_xp, mu, eta_i, k_i, xp,
        )
        # Convert from barns/sr to barns/mu by multiplying with 2 pi
        # (azimuthal symmetry: dOmega = 2 pi d mu).
        row_sigma_mu = row_sigma_omega * (2.0 * np.pi)
        if hasattr(out, 'at'):
            out = out.at[i].set(row_sigma_mu)
        else:
            out_np = np.asarray(out).copy()
            out_np[i] = np.asarray(row_sigma_mu)
            out = xp.asarray(out_np)

    # to_lab=True: for LCT=2 the stored distribution is in CM. A
    # proper CM->LAB conversion for charged-particle elastic
    # requires the two-body kinematics module. We flag the gap
    # explicitly rather than return a mislabelled distribution.
    if to_lab:
        warnings.warn(
            f'MF6/MT{mt} LAW=5 (charged-particle elastic): the '
            f'reconstruction returns the CM-frame dsigma/dmu even '
            f'when to_lab=True. The CM-to-LAB Jacobian for '
            f'charged-particle two-body kinematics is not applied '
            f'in the first #264 increment (deferred follow-up). '
            f'For heavy targets the CM and LAB frames differ by '
            f'at most awp/awr ~ 1/awr, which is small; for light '
            f'targets this is a noticeable approximation.',
            UserWarning, stacklevel=2,
        )
    return out
