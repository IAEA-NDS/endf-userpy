import warnings
import numpy as np
from ..primitives.interpolation import interp_tab1
from . import mf6_law1_helpers
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
from . import mf6_law2_preproc
from .mf6_interpretation_helpers import (
    pad_outside_dist2d_values,
    pad_outside_angdist_values,
)
import logging


module_logger = logging.getLogger(__name__)


@pad_outside_dist2d_values
def get_dist2d_from_subsec_law1(
    endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out, to_lab,
):
    """MF6 LAW=1 continuum-part angle-energy distribution.

    Returns only the continuum contribution ``f6con(E, E', mu)``.
    The discrete-line positions and amplitudes are handled by the
    separate :func:`get_law1_discrete_lines_from_subsec` because a
    delta at ``ep_disc_lab`` is not usefully sampled on an
    arbitrary user ``E_out`` grid.

    Pure-Python replacement for the previous Fortran-backed path
    (``mf6_get_law1``, endf6.f90 line 110). Uses the LAB->CM
    forward map from :mod:`mf6_law1_helpers.mf6lab2cm` and the
    per-panel continuum amplitude
    :func:`mf6_law1_helpers.f6law1con_amplitude`.
    """
    sec = endf_dict[6][mt]
    if sec['subsection'][subsec_num]['LAW'] != 1:
        raise ValueError(
            f'MT={mt} subsec_num={subsec_num} is '
            f'LAW={sec["subsection"][subsec_num]["LAW"]}, not LAW=1'
        )
    eu = np.asarray(energies_in, dtype=float)
    neu = eu.shape[0]
    epu = np.asarray(energies_out, dtype=float)
    nepu = epu.shape[0]
    uu = np.asarray(angle_cosines_out, dtype=float)
    nuu = uu.shape[0]
    awr = get_AWR(endf_dict)
    awi = get_AWI(endf_dict)
    za = get_ZA(endf_dict)
    zai = get_ZAI(endf_dict)
    lct = int(sec['LCT']) if to_lab else 1
    subsec = sec['subsection'][subsec_num]
    zap = subsec['ZAP']
    awp = subsec['AWP']
    lang = int(subsec['LANG'])
    lep = int(subsec['LEP'])
    ei_mesh = dict2array(subsec['E'], dtype=float)
    int_arr = np.array(subsec['INT'], dtype=int)
    nbt_arr = np.array(subsec['NBT'], dtype=int)
    ei_interp = convert_interp_repr(int_arr, nbt_arr)
    nd_arr = dict2array(subsec['ND'], dtype=int)
    na_arr = dict2array(subsec['NA'], dtype=int)

    if lct in (1, 2):
        eff_lct = lct
    elif lct == 3:
        eff_lct = 1 if awp > 4 else 2
    else:
        raise NotImplementedError(f'LCT={lct} not implemented')

    idcs = find_interval(ei_mesh, eu)

    cont_result_arr = np.zeros((neu, nepu, nuu), dtype=float)

    # Process one E panel bracket at a time; group einc by panel so
    # we do one marshaling pass per bracket.
    for panel_idx in np.unique(idcs):
        mask = (idcs == panel_idx)
        if not np.any(mask):
            continue
        rows = np.where(mask)[0]
        e1 = float(ei_mesh[panel_idx])
        e2 = float(ei_mesh[panel_idx + 1])
        nd1 = int(nd_arr[panel_idx])
        na1 = int(na_arr[panel_idx])
        ep1 = dict2array(subsec['Ep'][panel_idx + 1], dtype=float)
        b1 = dict2array(subsec['b'][panel_idx + 1], dtype=float)
        nd2 = int(nd_arr[panel_idx + 1])
        na2 = int(na_arr[panel_idx + 1])
        ep2 = dict2array(subsec['Ep'][panel_idx + 2], dtype=float)
        b2 = dict2array(subsec['b'][panel_idx + 2], dtype=float)
        lei = int(ei_interp[panel_idx])
        for row in rows:
            e = float(eu[row])
            for je in range(nepu):
                ep = float(epu[je])
                for ju in range(nuu):
                    u = float(uu[ju])
                    tp, w, dinv = mf6_law1_helpers.mf6lab2cm(
                        awr, awi, awp, eff_lct, e, ep, u,
                    )
                    fcon = mf6_law1_helpers.f6law1con_amplitude(
                        e, tp, w, za, zai, zap, lang, lep, lei,
                        e1, nd1, na1, ep1, b1,
                        e2, nd2, na2, ep2, b2,
                    )
                    cont_result_arr[row, je, ju] = fcon * dinv

    return cont_result_arr


def _law1_disc_lines_two_point_interp(law, x1, y1, x2, y2, x):
    """Scalar equivalent of the Fortran ``yintp`` (endf6.f90 line
    2011) used by the discrete-lines port. Guards match Fortran:

    - x == x1 or x1 == x2: return y1
    - x == x2: return y2
    - INT == 1 or y1 == y2: constant (returns y1)
    - INT == 3/5 with x1 == 0: clamp x1 to 1e-38
    - INT == 4/5 with y1 == 0: clamp y1 to 1e-38
    """
    small = 1.0e-38
    if x2 == x1 or x == x1:
        return y1
    if x == x2:
        return y2
    if law == 1 or y2 == y1:
        return y1
    if law == 2:
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    if law == 3:
        x1 = small if x1 == 0.0 else x1
        return y1 + np.log(x / x1) * (y2 - y1) / np.log(x2 / x1)
    if law == 4:
        y1 = small if y1 == 0.0 else y1
        return y1 * np.exp((x - x1) * np.log(y2 / y1) / (x2 - x1))
    if law == 5:
        x1 = small if x1 == 0.0 else x1
        y1 = small if y1 == 0.0 else y1
        return y1 * np.exp(np.log(x / x1) * np.log(y2 / y1) / np.log(x2 / x1))
    raise TypeError(f'interpolation scheme (INT={law}) not implemented')


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
    logic uses the section's LCT; this wrapper does not model an
    "evaluation frame" the caller can pick.

    Pure-Python replacement for the previous Fortran-backed path
    (``mf6_get_law1_disc_lines``, endf6.f90 line 190). Physics
    helpers live in :mod:`mf6_law1_helpers` and are shared with
    the (upcoming) LAW=1 continuum port. The Fortran-backed
    version is preserved as an equivalence oracle in
    :mod:`mf6_interpretation_subsecs_fort`.
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
    eu_full = np.asarray(energies_in, dtype=float)
    uu = np.asarray(angle_cosines_out, dtype=float)
    neu = len(eu_full)
    nuu = len(uu)

    awr = get_AWR(endf_dict)
    awi = get_AWI(endf_dict)
    za = get_ZA(endf_dict)
    zai = get_ZAI(endf_dict)
    lct = int(sec['LCT'])
    zap = subsec['ZAP']
    awp = subsec['AWP']
    lang = int(subsec['LANG'])
    ei_mesh = dict2array(subsec['E'], dtype=float)
    int_arr = np.array(subsec['INT'], dtype=int)
    nbt_arr = np.array(subsec['NBT'], dtype=int)
    ei_interp = convert_interp_repr(int_arr, nbt_arr)
    nd_arr = dict2array(subsec['ND'], dtype=int)
    na_arr = dict2array(subsec['NA'], dtype=int)

    nd_max = int(nd_arr.max()) if nd_arr.size else 0
    ep_disc_lab = np.zeros((neu, nuu, nd_max), dtype=float)
    amp_disc = np.zeros((neu, nuu, nd_max), dtype=float)
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

    # Process one incident-energy panel bracket (e1, e2) at a time.
    # Group user einc by the panel they land in so we do one panel
    # setup per bracket.
    for panel_idx in np.unique(idcs):
        mask_inside = (idcs == panel_idx)
        if not np.any(mask_inside):
            continue
        dst_rows = inside_pos[mask_inside]
        cur_eu = eu_inside[mask_inside]
        e1 = float(ei_mesh[panel_idx])
        e2 = float(ei_mesh[panel_idx + 1])
        nd1 = int(nd_arr[panel_idx])
        na1 = int(na_arr[panel_idx])
        ep1_full = dict2array(subsec['Ep'][panel_idx + 1], dtype=float)
        b1_full = dict2array(subsec['b'][panel_idx + 1], dtype=float)
        nd2 = int(nd_arr[panel_idx + 1])
        na2 = int(na_arr[panel_idx + 1])
        ep2_full = dict2array(subsec['Ep'][panel_idx + 2], dtype=float)
        b2_full = dict2array(subsec['b'][panel_idx + 2], dtype=float)
        lei = int(ei_interp[panel_idx])
        law = lei % 10       # base interp law for the outer E axis

        # Deduplicate coincident discrete Ep values in each panel.
        # The Fortran imatch returns only the first index of a
        # repeated ep, so any additional rows with the same ep
        # would be silently dropped and their b weight lost.
        # Physically identical to summing them (both sit at the
        # same LAB position after broadening), so pre-sum here.
        ep1_disc, b1_disc, nd1_ded = _dedup_discrete_lines(ep1_full, b1_full, nd1)
        ep2_disc, b2_disc, nd2_ded = _dedup_discrete_lines(ep2_full, b2_full, nd2)
        ep1_used = np.concatenate([ep1_disc, ep1_full[nd1:]])
        b1_used = np.concatenate([b1_disc, b1_full[nd1:]], axis=0)
        ep2_used = np.concatenate([ep2_disc, ep2_full[nd2:]])
        b2_used = np.concatenate([b2_disc, b2_full[nd2:]], axis=0)
        nd_used = min(nd1_ded, nd2_ded, nd_max)

        for i_local in range(cur_eu.size):
            e = float(cur_eu[i_local])
            if e < e1 or e > e2:
                continue
            dst_row = int(dst_rows[i_local])
            for ju in range(nuu):
                u = float(uu[ju])
                for k in range(nd_used):
                    # Panel-interpolated discrete eval-frame energy
                    # at user's E_in. For genuine level-decay lines
                    # ep1[k] == ep2[k], so this is usually a no-op.
                    ep1k = float(ep1_used[k])
                    ep2k = float(ep2_used[k])
                    tp = _law1_disc_lines_two_point_interp(
                        law, e1, ep1k, e2, ep2k, e,
                    )
                    ep_lab, w, dinv = mf6_law1_helpers.mf6cm2lab_disc(
                        awr, awi, awp, lct, e, tp, u,
                    )
                    if dinv <= 0.0:
                        continue
                    f1 = mf6_law1_helpers.f6law1_dis_amplitude(
                        e1, ep1k, w, za, zai, zap, lang,
                        nd1_ded, na1, ep1_used, b1_used,
                    )
                    f2 = mf6_law1_helpers.f6law1_dis_amplitude(
                        e2, ep2k, w, za, zai, zap, lang,
                        nd2_ded, na2, ep2_used, b2_used,
                    )
                    amp_interp = _law1_disc_lines_two_point_interp(
                        law, e1, f1, e2, f2, e,
                    )
                    ep_disc_lab[dst_row, ju, k] = ep_lab
                    amp_disc[dst_row, ju, k] = amp_interp * dinv

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


def _law2_reconstruct_from_data(
    data, energies_in, angle_cosines_out, to_lab, xp,
):
    """Backend-agnostic MF6 LAW=2 kernel operating on the
    :class:`MF6Law2Data` dataclass produced by
    :mod:`mf6_law2_preproc`.

    All arithmetic runs through ``xp``. JAX tracers stored in
    ``data.coeffs`` or the ``'f'`` field of a ``data.records`` entry
    propagate through to the return value; autodiff-driven
    parameter fitting (issue #154) works by calling this function
    with a dataclass whose relevant field has been replaced with a
    tracer via :func:`dataclasses.replace`.
    """
    lct = data.lct if to_lab else 1
    lang = int(data.lang)
    # Effective LCT for this ejectile.
    if lct in (1, 2):
        eff_lct = lct
    elif lct == 3:
        eff_lct = 1 if data.awp > 4 else 2
    else:
        raise NotImplementedError(f'LCT={lct} not implemented')

    e_in = xp.asarray(energies_in, dtype=xp.float64)
    mu_lab = xp.asarray(angle_cosines_out, dtype=xp.float64)
    if eff_lct == 2:
        r2 = compute_r2(e_in, data.awi, data.awr, data.awp, data.q, xp=xp)
        mu_eff = convert_angcos_to_cmsys(mu_lab, r2, xp=xp)
    else:
        mu_eff = xp.broadcast_to(
            mu_lab.reshape(1, -1),
            (e_in.shape[0], mu_lab.shape[0]),
        )

    if lang == 0:
        f_eff = evaluate_interp_legendre_polynomials(
            np.asarray(energies_in, dtype=float), np.asarray(mu_eff),
            data.ei_mesh, data.coeffs, data.int_arr, data.nbt_arr,
            xp=xp,
        )
    elif lang in (12, 14):
        f_eff = interp_tab2(
            np.asarray(energies_in, dtype=float), np.asarray(mu_eff),
            data.ei_mesh, data.int_arr, data.nbt_arr,
            data.records, 'mu', 'f',
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
    on JAX. This dict-facing entry composes
    :func:`~mf6_law2_preproc.mf6_law2_data_from_endf_dict` with
    :func:`_law2_reconstruct_from_data` internally; callers who
    want ``jax.grad`` back to file-stored Legendre coefficients
    (issue #154) should call those two functions themselves and
    replace the coefficient array with a tracer before
    reconstruction.
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
    data = mf6_law2_preproc.mf6_law2_data_from_endf_dict(
        endf_dict, mt, subsec_num, xp=xp,
    )
    return _law2_reconstruct_from_data(
        data, energies_in, angle_cosines_out, to_lab, xp,
    )


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
