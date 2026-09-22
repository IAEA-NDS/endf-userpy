import warnings
import numpy as np
from ..primitives.interpolation import interp_tab1
from ..fortran.endf6 import (
    mf6_get_law1,
    mf6_get_law1_disc_lines,
)
from ..primitives import array_ns
from ..primitives.conversion import (
    compute_r2,
    convert_angcos_to_cmsys,
    convert_angdist_to_labsys,
)
from ..primitives.interpolation import (
    evaluate_interp_legendre_polynomials,
    interp_tab2,
    _interp_two_point_columns,
)
from ..primitives.helpers import (
    dict2array,
    convert_interp_repr,
    find_interval,
)
from ..primitives.properties import (
    get_AWR,
    get_AWI,
    get_ZA,
    get_ZAI,
    get_QI,
)
from .mf6_interpretation_helpers import (
    pad_outside_dist2d_values,
    pad_outside_angdist_values,
)
import logging


module_logger = logging.getLogger(__name__)


@pad_outside_dist2d_values
def get_dist2d_from_subsec_law1(
    endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out, to_lab
):
    sec = endf_dict[6][mt]
    if sec['subsection'][subsec_num]['LAW'] != 1:
        raise ValueError(
            f'MT={mt} subsec_num={subsec_num} is '
            f'LAW={sec["subsection"][subsec_num]["LAW"]}, not LAW=1'
        )
    eu = energies_in
    neu = len(eu)
    epu = energies_out
    nepu = len(epu)
    uu = angle_cosines_out
    nuu = len(uu)
    awr = get_AWR(endf_dict)
    awi = get_AWI(endf_dict)
    za = get_ZA(endf_dict)
    zai = get_ZAI(endf_dict)
    lct = sec['LCT'] if to_lab else 1
    # subsection variables
    subsec = sec['subsection'][subsec_num]
    zap = subsec['ZAP']
    awp = subsec['AWP']
    lang = subsec['LANG']
    lep = subsec['LEP'] 
    ei_mesh = dict2array(subsec['E'], dtype=float)
    int_arr = np.array(subsec['INT'], dtype=int)
    nbt_arr = np.array(subsec['NBT'], dtype=int)
    ei_interp = convert_interp_repr(int_arr, nbt_arr)
    nd_arr = dict2array(subsec['ND'], dtype=int)
    na_arr = dict2array(subsec['NA'], dtype=int)

    # determine effective LCT based on emitted particle (CM or LAB)
    if lct in (1, 2):
        eff_lct = lct
    elif lct == 3:
        eff_lct = 1 if awp > 4 else 2
    else:
        raise NotImplementedError(f'LCT={lct} not implemented')

    # find enclosing energy intervals
    idcs = find_interval(ei_mesh, energies_in)

    result_dim = (neu, nepu, nuu)
    disc_result_arr = np.zeros(result_dim, dtype=float)
    cont_result_arr = np.zeros(result_dim, dtype=float)

    for i in range(cont_result_arr.shape[0]):
        curidx = idcs[i]
        cur_eu = np.array([eu[i]], order='F')
        lei = ei_interp[curidx].item()

        e1 = ei_mesh[curidx].item()
        nd1 = nd_arr[curidx].item()
        na1 = na_arr[curidx].item()
        ep1 = dict2array(subsec['Ep'][curidx+1], dtype=float, order='F')
        b1 = dict2array(subsec['b'][curidx+1], dtype=float, order='F')

        e2 = ei_mesh[curidx+1].item()
        nd2 = nd_arr[curidx+1].item()
        na2 = na_arr[curidx+1].item()
        ep2 = dict2array(subsec['Ep'][curidx+2], dtype=float, order='F')
        b2 = dict2array(subsec['b'][curidx+2], dtype=float, order='F')

        cur_disc_res = np.zeros((1, nepu, nuu), dtype=float, order='F')
        cur_cont_res = np.zeros((1, nepu, nuu), dtype=float, order='F')

        # neu, nepu, nep1, nep2 are automatically inferred
        # hence dropped from the argument list
        mf6_get_law1(
            cur_eu, epu, uu, nuu,
            awr, awi, awp, za, zai, zap, eff_lct, lang, lep, lei,
            e1, nd1, na1, ep1, b1, e2, nd2, na2, ep2, b2,
            cur_disc_res, cur_cont_res
        )

        disc_result_arr[i:i+1,:,:] = cur_disc_res
        cont_result_arr[i:i+1,:,:] = cur_cont_res

    return cont_result_arr


def get_law1_discrete_lines_from_subsec(
    endf_dict, mt, subsec_num, energies_in, angle_cosines_out, to_lab=True,
):
    """Discrete-line positions and amplitudes for one MF6/LAW=1
    subsection.

    Returns (ep_disc_lab, amp_disc), both of shape (n_einc, n_mus,
    nd_common) where nd_common = min(ND across all panels bracketed
    by any einc). Entries where the CM->LAB inverse map has no
    physical solution (below-threshold cases) are zero; callers
    should treat them as "no contribution at this cell".

    to_lab must be True; setting it False raises. The underlying
    Fortran uses the section's LCT, and this wrapper does not model
    an "evaluation frame" the caller can pick.
    """
    if to_lab is not True:
        raise ValueError(
            'get_law1_discrete_lines_from_subsec requires `to_lab=True`'
        )
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    if subsec['LAW'] != 1:
        raise ValueError(
            f'MT={mt} subsec_num={subsec_num} is LAW={subsec["LAW"]}, '
            'not LAW=1'
        )
    eu_full = np.asfortranarray(np.asarray(energies_in, dtype=float))
    uu = np.asfortranarray(np.asarray(angle_cosines_out, dtype=float))
    neu = len(eu_full)
    nuu = len(uu)

    awr = get_AWR(endf_dict)
    awi = get_AWI(endf_dict)
    za = get_ZA(endf_dict)
    zai = get_ZAI(endf_dict)
    lct = sec['LCT']
    zap = subsec['ZAP']
    awp = subsec['AWP']
    lang = subsec['LANG']
    lep = subsec['LEP']
    ei_mesh = dict2array(subsec['E'], dtype=float)
    int_arr = np.array(subsec['INT'], dtype=int)
    nbt_arr = np.array(subsec['NBT'], dtype=int)
    ei_interp = convert_interp_repr(int_arr, nbt_arr)
    nd_arr = dict2array(subsec['ND'], dtype=int)
    na_arr = dict2array(subsec['NA'], dtype=int)

    nd_max = int(nd_arr.max()) if nd_arr.size else 0
    ep_disc_lab = np.zeros((neu, nuu, nd_max), dtype=float, order='F')
    amp_disc = np.zeros((neu, nuu, nd_max), dtype=float, order='F')
    if nd_max == 0:
        return ep_disc_lab, amp_disc

    # Zero-pad einc entries that fall outside this subsection's
    # panel-mesh range: below-threshold or above-max einc get no
    # contribution from this subsection (mirroring what
    # pad_outside_dist2d_values does for the continuum wrapper).
    inside = (eu_full >= ei_mesh.min()) & (eu_full <= ei_mesh.max())
    if not np.any(inside):
        return ep_disc_lab, amp_disc
    eu_inside = eu_full[inside]
    idcs = find_interval(ei_mesh, eu_inside)
    inside_pos = np.flatnonzero(inside)

    # The Fortran routine processes one incident-energy panel bracket
    # (e1, e2) at a time. Group user einc by the panel they land in
    # so we make one Fortran call per bracket.
    for panel_idx in np.unique(idcs):
        mask_inside = (idcs == panel_idx)
        if not np.any(mask_inside):
            continue
        cur_eu = np.asfortranarray(eu_inside[mask_inside])
        # Map back to positions in the full einc array so we can
        # write results into the right slots.
        dst_rows = inside_pos[mask_inside]
        e1 = ei_mesh[panel_idx].item()
        e2 = ei_mesh[panel_idx + 1].item()
        nd1 = nd_arr[panel_idx].item()
        na1 = na_arr[panel_idx].item()
        ep1_full = dict2array(subsec['Ep'][panel_idx + 1], dtype=float)
        b1_full = dict2array(subsec['b'][panel_idx + 1], dtype=float)
        nd2 = nd_arr[panel_idx + 1].item()
        na2 = na_arr[panel_idx + 1].item()
        ep2_full = dict2array(subsec['Ep'][panel_idx + 2], dtype=float)
        b2_full = dict2array(subsec['b'][panel_idx + 2], dtype=float)
        lei = ei_interp[panel_idx].item()

        # Deduplicate coincident discrete Ep values in each panel:
        # the downstream Fortran f6law1_dis uses imatch which returns
        # only the first index of a repeated ep, so any additional
        # rows with the same ep would be silently dropped and their
        # b weight lost. Physically identical to summing them (both
        # sit at the same LAB position after broadening), so pre-sum
        # here and pass unique-ep arrays to the Fortran routine.
        ep1_disc, b1_disc, nd1_ded = _dedup_discrete_lines(ep1_full, b1_full, nd1)
        ep2_disc, b2_disc, nd2_ded = _dedup_discrete_lines(ep2_full, b2_full, nd2)
        ep1 = np.asfortranarray(np.concatenate([ep1_disc, ep1_full[nd1:]]))
        b1 = np.asfortranarray(np.concatenate([b1_disc, b1_full[nd1:]], axis=0))
        ep2 = np.asfortranarray(np.concatenate([ep2_disc, ep2_full[nd2:]]))
        b2 = np.asfortranarray(np.concatenate([b2_disc, b2_full[nd2:]], axis=0))

        cur_ep = np.zeros(
            (cur_eu.size, nuu, nd_max), dtype=float, order='F',
        )
        cur_amp = np.zeros(
            (cur_eu.size, nuu, nd_max), dtype=float, order='F',
        )
        mf6_get_law1_disc_lines(
            cur_eu, uu,
            awr, awi, awp, za, zai, zap, lct, lang, lep, lei,
            e1, nd1_ded, na1, ep1, b1,
            e2, nd2_ded, na2, ep2, b2,
            nd_max, cur_ep, cur_amp,
        )
        ep_disc_lab[dst_rows, :, :] = cur_ep
        amp_disc[dst_rows, :, :] = cur_amp

    return ep_disc_lab, amp_disc


def _dedup_discrete_lines(ep, b, nd):
    """Sum b rows at coincident ep values in the first ND entries of
    the panel arrays. Returns (ep_ded, b_ded, nd_ded); shapes are
    reduced to ND_unique rows, still row-major over (row, angular).
    """
    if nd <= 0:
        return ep[:0], b[:0], 0
    ep_disc = ep[:nd]
    b_disc = b[:nd, :]
    # Group by ep value; preserve order of first occurrence.
    seen = {}
    for i, val in enumerate(ep_disc):
        key = float(val)
        if key in seen:
            seen[key] = seen[key] + [i]
        else:
            seen[key] = [i]
    if len(seen) == nd:
        # No duplicates; keep the original arrays unchanged.
        return ep_disc, b_disc, nd
    ep_ded = np.empty(len(seen), dtype=float)
    b_ded = np.empty((len(seen), b.shape[1]), dtype=float)
    for out_i, (key, rows) in enumerate(seen.items()):
        ep_ded[out_i] = key
        b_ded[out_i, :] = b_disc[rows, :].sum(axis=0)
    return ep_ded, b_ded, len(seen)


def _law2_legendre_coeffs_array(subsec):
    """Build the full Legendre coefficient array
    ``(n_panels, max_L+1)`` from the ENDF-dict `subsec['A']`.

    ENDF-6 MF6 LAW=2 LANG=0 stores per-panel Legendre coefficients
    ``a_1, a_2, ..., a_{NL}`` (a_0 = 1 is implied by normalisation
    and not written). We prepend ``a_0 = 1`` and apply the
    ``(L + 0.5)`` factor (ENDF convention: ``f(mu) = sum_L (L+0.5)
    a_L P_L(mu)``) so the caller can feed the result directly into
    :func:`evaluate_interp_legendre_polynomials`, which evaluates
    ``sum_L coeffs[..., L] P_L(mu)`` without further weighting.

    Padding ragged rows to the max NL with zeros is safe: a zero
    higher-degree coefficient contributes nothing to the Legendre
    sum.
    """
    a_arr = dict2array(subsec['A'], dtype=float, order='C', fill_value=0.0)
    nl_arr = np.array(list(subsec['NL'].values()), dtype=int)
    n_panels = a_arr.shape[0]
    max_L = a_arr.shape[1]                 # a_1 .. a_{max_L}
    # Full coefficients including a_0.
    coeffs = np.zeros((n_panels, max_L + 1), dtype=float)
    coeffs[:, 0] = 1.0                     # a_0 = 1
    for p in range(n_panels):
        nl = int(nl_arr[p])
        coeffs[p, 1:nl + 1] = a_arr[p, :nl]
    # (L + 0.5) prefactor per ENDF convention.
    L_indices = np.arange(coeffs.shape[1]).reshape(1, -1)
    coeffs = coeffs * (L_indices + 0.5)
    return coeffs


def _law2_tab1_records(subsec, lang):
    """Build a list of TAB1-shaped record dicts from the LANG=12/14
    tabulated data in ``subsec['A']``.

    ENDF-6 stores each panel as ``[u_1, p_1, u_2, p_2, ...]``. We
    unpack to separate ``mu`` and ``f`` arrays and mark the
    per-panel INT (``lang - 10``: LANG=12 -> lin-lin INT=2,
    LANG=14 -> log-lin INT=4). Format matches what
    :func:`interp_tab2` consumes.
    """
    a_arr = dict2array(subsec['A'], dtype=float, order='C', fill_value=0.0)
    nl_arr = np.array(list(subsec['NL'].values()), dtype=int)
    inner_int = lang - 10
    records = []
    for p in range(a_arr.shape[0]):
        nl = int(nl_arr[p])
        pairs = a_arr[p, :2 * nl]
        mu_p = pairs[0::2]
        f_p = pairs[1::2]
        records.append({
            'mu': mu_p, 'f': f_p,
            'INT': np.array([inner_int], dtype=int),
            'NBT': np.array([nl], dtype=int),
        })
    return records


@pad_outside_angdist_values
def get_angdist_from_subsec_law2(
    endf_dict, mt, subsec_num, energies_in, angle_cosines_out, to_lab,
    xp=None,
):
    """Backend-agnostic MF6 LAW=2 (discrete two-body reaction)
    angular distribution.

    Follows the ENDF-6 recipe (manual sec. 6.2.4): the evaluated
    ``f(E, mu)`` at CM (or LAB, per ``LCT``) is either a Legendre
    expansion (``LANG=0``) or a tabulated (u, p(u)) table
    (``LANG=12`` lin-lin, ``LANG=14`` log-lin in mu). When the
    evaluation is CM (``LCT=2``, or ``LCT=3`` with the ejectile
    mass ``AWP <= 4``), the query LAB cosine is mapped to the CM
    cosine via :func:`convert_angcos_to_cmsys` and the CM result
    is multiplied by the CM -> LAB Jacobian via
    :func:`convert_angdist_to_labsys`. Forbidden LAB angles
    (``mu_lab < mu_lab_min`` for equal-mass / light-ejectile
    back-scatter) produce NaN through the CM-cos mapping; we clip
    NaN and negatives to zero to match the Fortran reference and
    the MF4 Python path (issue #45).

    Backend-agnostic: ``xp=None`` (the default) resolves to numpy
    and gives the bit-identical numpy path used before the port.
    Passing ``xp=array_ns.get_backend('jax')`` runs the arithmetic
    on JAX, with the same file-side / query-side autodiff boundary
    as the underlying primitives (see issue #154 for the full
    dict-source autodiff scope). The Fortran-backed version used
    to be provided by ``mf6_get_law2`` here; it now lives in
    :mod:`mf6_interpretation_subsecs_fort` as the equivalence
    oracle.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    # Short-circuit for gamma ZAP (issue #78): the LAW=2 CM<->LAB
    # conversion assumes a massive ejectile; photons need
    # massless-ejectile kinematics which we do not implement yet.
    if subsec.get('ZAP') == 0.0:
        warnings.warn(
            f'MF6/MT{mt} subsection {subsec_num} stores gamma '
            f'(ZAP=0) with LAW=2. The 2-body angular distribution '
            f'conversion currently assumes a massive ejectile and '
            f'returns NaN for photons; skipping this contribution '
            f'(returning zeros). Files that need gamma production '
            f'from radiative-capture channels should carry the '
            f'photon yield and spectrum in MF12+MF14 or MF13+MF14; '
            f'the MF6/LAW=2 photon-kinematics path is not yet '
            f'implemented (issue #78).',
            UserWarning, stacklevel=3,
        )
        return xp.zeros(
            (len(energies_in), len(angle_cosines_out)),
            dtype=xp.float64,
        )
    awr = get_AWR(endf_dict)
    awi = get_AWI(endf_dict)
    awp = subsec['AWP']
    q = get_QI(endf_dict, mt)
    lct = sec['LCT'] if to_lab else 1
    lang = int(subsec['LANG'])
    ei_mesh = dict2array(subsec['E'], dtype=float)
    int_arr = np.array(subsec['INT'], dtype=int)
    nbt_arr = np.array(subsec['NBT'], dtype=int)

    # Determine effective LCT for this ejectile.
    if lct in (1, 2):
        eff_lct = lct
    elif lct == 3:
        eff_lct = 1 if awp > 4 else 2
    else:
        raise NotImplementedError(f'LCT={lct} not implemented')

    # Convert query LAB cosine to CM cosine if the evaluated data
    # is in CM. `convert_angcos_to_cmsys` returns shape
    # `(nE, nmu)`; when LAB (eff_lct == 1) we broadcast `mu` to the
    # same shape by hand so the downstream evaluators see a
    # consistent 2D query.
    e_in = xp.asarray(energies_in, dtype=xp.float64)
    mu_lab = xp.asarray(angle_cosines_out, dtype=xp.float64)
    if eff_lct == 2:
        r2 = compute_r2(e_in, awi, awr, awp, q, xp=xp)
        mu_eff = convert_angcos_to_cmsys(mu_lab, r2, xp=xp)
    else:
        mu_eff = xp.broadcast_to(
            mu_lab.reshape(1, -1),
            (e_in.shape[0], mu_lab.shape[0]),
        )

    # Evaluate the CM (or LAB, if eff_lct == 1) f(E, mu_eff).
    if lang == 0:
        coeffs = _law2_legendre_coeffs_array(subsec)
        f_eff = evaluate_interp_legendre_polynomials(
            np.asarray(energies_in, dtype=float), np.asarray(mu_eff),
            ei_mesh, coeffs, int_arr, nbt_arr,
            xp=xp,
        )
    elif lang in (12, 14):
        records = _law2_tab1_records(subsec, lang)
        f_eff = interp_tab2(
            np.asarray(energies_in, dtype=float), np.asarray(mu_eff),
            ei_mesh, int_arr, nbt_arr, records, 'mu', 'f',
            xp=xp,
        )
    else:
        raise NotImplementedError(
            f'MF6 LAW=2 LANG={lang} not supported (only 0, 12, 14).'
        )

    if eff_lct == 2:
        # CM -> LAB Jacobian, then clip forbidden-region NaN /
        # negative artefacts (matches MF4's post-conversion clip;
        # see mf4_interpretation.compute_angdist_values for the
        # rationale on issue #45).
        f_lab = convert_angdist_to_labsys(mu_eff, f_eff, r2, xp=xp)
        f_lab = xp.where(xp.isnan(f_lab), 0.0, f_lab)
        f_lab = xp.where(f_lab < 0.0, 0.0, f_lab)
        return f_lab
    return f_eff


def _law6_kernel(awr, awi, awp, q, apsx, npsx, eu, epu, uu, xp):
    """Backend-agnostic MF6 LAW=6 (N-body phase-space) kernel.

    Per ENDF-6 manual sec. 6.2.7 / Kalbach LA-13166, the LAB-frame
    double-differential distribution for one of ``npsx`` phase-space
    ejectiles is

        f(E, E', mu) = C_n * sqrt(E') * (E_i^max - E'_c)^(1.5*npsx - 4)

    where ``E'_c = E_s + E' - 2 mu sqrt(E_s E')`` is the CM
    outgoing energy (with LAB kinematic shift
    ``E_s = (AWI*AWP/(AWI+AWR)^2) * E``), ``E_i^max =
    ((APSX - AWP)/APSX) * ((AWR/(AWI+AWR)) * E + Q)`` is the
    maximum available CM energy per ejectile, and

        C_3 = 4/(pi E_i^max^2)
        C_4 = 105/(32 E_i^max^{7/2})
        C_5 = 256/(14 pi E_i^max^5)

    Zero outside the kinematically allowed region (``E'_c <
    E_i^max``) or for ``npsx`` outside {3, 4, 5} (per Fortran
    reference; the manual's general formula collapses on itself
    for other N and the endf6.f90 hard-codes only these three).

    Vectorised over the ``(nE, nE', nmu)`` outer product; every mu
    column feels the mu-dependence through E'_c. Backend-agnostic:
    all arithmetic goes through ``xp`` (numpy or jax.numpy).
    """
    eu = xp.asarray(eu, dtype=xp.float64)
    epu = xp.asarray(epu, dtype=xp.float64)
    uu = xp.asarray(uu, dtype=xp.float64)

    awc = awi + awr
    ea = (awr / awc) * eu + q                             # (nE,)
    eimax = ((apsx - awp) / apsx) * ea                    # (nE,)
    es = (awi * awp / (awc * awc)) * eu                   # (nE,)

    # 3D outer product: broadcast (nE, nE', nmu).
    eimax_bc = eimax[:, None, None]
    es_bc = es[:, None, None]
    ep_bc = epu[None, :, None]
    mu_bc = uu[None, None, :]

    epc = es_bc + ep_bc - 2.0 * mu_bc * xp.sqrt(es_bc * ep_bc)
    delta = eimax_bc - epc                                # (nE, nE', nmu)

    npsx_i = int(npsx)
    if npsx_i == 3:
        c_n = 4.0 / (xp.pi * eimax * eimax)
    elif npsx_i == 4:
        c_n = 105.0 / (32.0 * eimax ** 3.5)
    elif npsx_i == 5:
        c_n = 256.0 / (14.0 * xp.pi * eimax ** 5.0)
    else:
        # Fortran returns zero for npsx outside {3, 4, 5}.
        return xp.zeros((eu.shape[0], epu.shape[0], uu.shape[0]),
                        dtype=xp.float64)

    c_bc = c_n[:, None, None]
    power = 1.5 * npsx_i - 4.0
    # Guard the power against negative bases (JAX / numpy would emit
    # NaN); zero those out via `where` explicitly.
    delta_safe = xp.where(delta > 0.0, delta, 1.0)
    f = c_bc * xp.sqrt(ep_bc) * delta_safe ** power
    # Also require Eimax > 0 (below reaction threshold in LAB).
    valid = (delta > 0.0) & (eimax_bc > 0.0)
    return xp.where(valid, f, 0.0)


def get_dist2d_from_subsec_law6(
    endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out,
    to_lab, xp=None,
):
    # NOTE: to_lab parameter ignored for LAW=6; the ENDF-6 formula is
    # defined directly in the LAB frame per the manual.
    if xp is None:
        xp = array_ns.get_backend('numpy')
    sec = endf_dict[6][mt]
    awr = get_AWR(endf_dict)
    awi = get_AWI(endf_dict)
    q = get_QI(endf_dict, mt)
    subsec = sec['subsection'][subsec_num]
    awp = subsec['AWP']
    apsx = subsec['APSX']
    npsx = subsec['NPSX']
    return _law6_kernel(awr, awi, awp, q, apsx, npsx,
                        energies_in, energies_out, angle_cosines_out, xp)


def _law7_tab1_records_for_e_panel(subsec, e_panel_key):
    """Build the list of per-mu-knot TAB1 records for one incident-
    energy panel of an MF6 LAW=7 subsection. Each record maps an
    outgoing-energy mesh ``Ep`` to the conditional distribution
    ``f(Ep | mu_i, E_panel)`` with its own INT/NBT for the inner
    (Ep) axis interpolation.

    ``e_panel_key`` is the 1-indexed key into ``subsec['table']``
    (matches the ``curidx + 1`` / ``curidx + 2`` convention used by
    the pre-port Fortran wrapper).
    """
    per_mu_tables = subsec['table'][e_panel_key]
    n_mu = len(per_mu_tables)
    records = []
    for mu_i in range(n_mu):
        curtab = per_mu_tables[mu_i + 1]
        records.append({
            'Ep': np.asarray(curtab['Ep'], dtype=float),
            'f': np.asarray(curtab['f'], dtype=float),
            'INT': np.asarray(curtab['INT'], dtype=int),
            'NBT': np.asarray(curtab['NBT'], dtype=int),
        })
    return records


def _law7_eval_mu_panel(subsec, e_panel_key, mu_out, ep_out, xp):
    """Evaluate the ``(mu, Ep)`` grid for one incident-energy panel
    of MF6 LAW=7. Returns shape ``(n_mu_out, n_ep_out)``.

    LAW=7 always uses unit-base interpolation between adjacent mu
    subpanels (ENDF-6 manual sec. 6.2.8). The per-subpanel base
    interpolation law comes from ``subsec['mu_interpol'][panel][
    'INT']`` (typically INT=2 lin-lin); we force unit-base by
    normalising the code to the 21..25 range (``20 + (raw mod
    10)``), which triggers the unit-base branch of
    :func:`interp_tab2`.
    """
    mu_mesh = dict2array(subsec['mu'][e_panel_key], dtype=float)
    mu_interpol = subsec['mu_interpol'][e_panel_key]
    mu_int_raw = np.asarray(mu_interpol['INT'], dtype=int)
    mu_nbt = np.asarray(mu_interpol['NBT'], dtype=int)
    # Force unit-base (LAW=7 semantic requirement).
    mu_int = 20 + (mu_int_raw % 10)
    records = _law7_tab1_records_for_e_panel(subsec, e_panel_key)
    return interp_tab2(
        np.asarray(mu_out, dtype=float),
        np.asarray(ep_out, dtype=float),
        mu_mesh, mu_int, mu_nbt, records, 'Ep', 'f',
        outside_value=0.0, xp=xp,
    )


@pad_outside_dist2d_values
def get_dist2d_from_subsec_law7(
    endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out,
    to_lab, xp=None,
):
    """Backend-agnostic MF6 LAW=7 (tabulated E'/mu double-
    differential) reconstruction.

    Per ENDF-6 manual sec. 6.2.8: for each incident-energy panel,
    the distribution is a TAB2 in mu whose entries are TAB1 records
    of ``f(Ep | mu)``; unit-base interpolation is used both
    between adjacent mu subpanels within an E panel and between
    the two E panels bracketing the query. The outer E interpolation
    picks the base law from ``subsec['E_interpol']['INT']`` (only
    the mod-10 base law matters here; the ``+ 20`` unit-base
    variant applies to the inner mu interp).

    LAW=7 is always defined in the LAB system, so ``to_lab`` is
    ignored (matches the pre-port Fortran behaviour).

    Backend-agnostic: ``xp=None`` defaults to the numpy backend and
    is bit-identical to the pre-port path; passing
    ``xp=array_ns.get_backend('jax')`` runs the arithmetic on JAX,
    with the same file-side / query-side autodiff boundary as
    documented in :func:`interp_tab2` (see issue #154 for the
    dict-source autodiff scope).
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]

    ei_mesh = dict2array(subsec['E'], dtype=float)
    int_arr = np.array(subsec['E_interpol']['INT'], dtype=int)
    nbt_arr = np.array(subsec['E_interpol']['NBT'], dtype=int)
    ei_interp = convert_interp_repr(int_arr, nbt_arr)

    ep_out = np.asarray(energies_out, dtype=float)
    mu_out = np.asarray(angle_cosines_out, dtype=float)
    e_in = np.asarray(energies_in, dtype=float)

    n_e = e_in.shape[0]
    n_ep = ep_out.shape[0]
    n_mu = mu_out.shape[0]

    idcs = find_interval(ei_mesh, e_in)
    rows = []
    for i, curidx in enumerate(idcs):
        e = float(e_in[i])
        e1 = float(ei_mesh[curidx])
        e2 = float(ei_mesh[curidx + 1])
        lei_law = int(ei_interp[curidx]) % 10

        # Per-E-panel unit-base mu evaluation: (n_mu, n_ep)
        f1 = _law7_eval_mu_panel(subsec, curidx + 1, mu_out, ep_out, xp)
        f2 = _law7_eval_mu_panel(subsec, curidx + 2, mu_out, ep_out, xp)

        # Outer E-interp between f1 and f2 at query e.
        # `_interp_two_point_columns` broadcasts elementwise over
        # the shared 2D shape of (f1, f2).
        f_e = _interp_two_point_columns(e, e1, e2, f1, f2, lei_law, xp)
        # f_e shape: (n_mu, n_ep). Result axis order is
        # (E_in, Ep_out, mu_out), so transpose to (n_ep, n_mu).
        rows.append(f_e.T)

    if len(rows) == 0:
        return xp.zeros((n_e, n_ep, n_mu), dtype=xp.float64)
    return xp.stack(rows, axis=0)


def get_incident_energies_from_subsec(endf_dict, mt, subsec_num):
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    yield_tab = subsec['yields']
    return np.array(yield_tab['Eint'], copy=True)


def get_emission_energies_from_subsec(endf_dict, mt, subsec_num, nofail=False):
    """Tabulated outgoing-energy mesh of one MF6 subsection.

    LAW=1 and LAW=7 have per-Ein / per-(Ein, mu) Ep tabulations
    that this function unions and returns. LAW=2 (angular
    distribution only), LAW=3 (isotropic charged-particle),
    LAW=4 (recoil), LAW=5 (charged-particle with phase shift),
    LAW=6 (n-body phase-space) do not tabulate a per-Ein Ep
    mesh; their emission energies are either kinematically
    determined (LAW=2, 3, 4) or analytic (LAW=6). Return an
    empty ndarray for those (issue #87 mode A: the pre-fix
    behaviour raised NotImplementedError even with the default
    ``nofail=False`` so any dispatcher walking a mixed-LAW
    file crashed at the first LAW=2 subsection encountered).

    The `nofail` parameter is now vestigial: an empty return is
    always produced for LAWs without a tabulated Ep mesh, and
    the switch has no effect. Kept for signature compatibility
    with pre-#107 callers.
    """
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    law = subsec['LAW']
    if law == 1:
        energies = [v for u in subsec['Ep'].values() for v in u.values()]
    elif law == 7:
        energies = [
            e
            for u in subsec['table'].values()
            for v in u.values()
            for e in v['Ep']
        ]
    else:
        energies = []
    return np.unique(energies)


def compute_yields_from_subsec(endf_dict, mt, subsec_num, energies_in):
    module_logger.debug(f'compute yield from subsec with number {subsec_num}')
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    yield_tab = subsec['yields']
    interp_yields = interp_tab1(
        energies_in, yield_tab, 'Eint', 'yi', outside_value=0.0
    )
    module_logger.debug(interp_yields)
    return interp_yields


def compute_dist2d_from_subsec(
    endf_dict, mt, subsec_num,
    energies_in, energies_out, angle_cosines_out, to_lab=True
):
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    law = subsec['LAW']
    if law == 1:
        return get_dist2d_from_subsec_law1(
            endf_dict, mt, subsec_num,
            energies_in, energies_out, angle_cosines_out, to_lab
        )
    elif law == 6:
        return get_dist2d_from_subsec_law6(
            endf_dict, mt, subsec_num,
            energies_in, energies_out, angle_cosines_out, to_lab
        )
    elif law == 7:
        return get_dist2d_from_subsec_law7(
            endf_dict, mt, subsec_num,
            energies_in, energies_out, angle_cosines_out, to_lab
        )
    else:
        raise NotImplementedError(
            f'DDX interpretation for LAW={law} not implemented.'
        )


def compute_angdist_from_subsec(
    endf_dict, mt, subsec_num,
    energies_in, angle_cosines_out, to_lab=True
):
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    law = subsec['LAW']
    if law == 2:
        return get_angdist_from_subsec_law2(
            endf_dict, mt, subsec_num,
            energies_in, angle_cosines_out, to_lab
        )
    else:
        raise NotImplementedError(
            f'Angular distribution interpretation for LAW={law} '
            'not implemented.'
        )
