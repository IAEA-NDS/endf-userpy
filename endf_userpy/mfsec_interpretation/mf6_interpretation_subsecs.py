import warnings
import numpy as np
from ..primitives.interpolation import interp_tab1
from ..fortran.endf6 import (
    mf6_get_law1,
    mf6_get_law1_disc_lines,
    mf6_get_law2,
    mf6_get_law6,
    mf6_get_law7,
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


@pad_outside_angdist_values
def get_angdist_from_subsec_law2(
    endf_dict, mt, subsec_num, energies_in, angle_cosines_out, to_lab
):
    # TODO: how to signal to the user that this is a
    #       discrete distribution with a dirac delta in the
    #       the emission energy.
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    # Short-circuit for gamma ZAP (issue #78): the LAW=2 CM<->LAB
    # conversion in `mf6_get_law2` uses massive-particle 2-body
    # kinematics with the subsection's `AWP` as the ejectile mass.
    # For photons the correct treatment is massless-ejectile
    # kinematics (energy and direction related by |p_gamma| = E_gamma
    # rather than E = p^2 / 2m), which the Fortran evaluator does not
    # implement. Previously the massive-formula path silently
    # produced all NaN in the returned angular distribution and the
    # caller propagated the NaN into `get_particle_production_dxs_dmu`.
    # Return zeros so the sum-over-MTs stays well-defined; emit one
    # summary warning per file so users see a clear signal.
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
        return np.zeros(
            (len(energies_in), len(angle_cosines_out)), dtype=float,
        )
    awr = get_AWR(endf_dict)
    awi = get_AWI(endf_dict)
    awp = subsec['AWP']
    q = get_QI(endf_dict, mt)
    lct = sec['LCT'] if to_lab else 1
    lang = subsec['LANG']
    ei_mesh = dict2array(subsec['E'], dtype=float)
    int_arr = np.array(subsec['INT'], dtype=int)
    nbt_arr = np.array(subsec['NBT'], dtype=int)
    ei_interp = convert_interp_repr(int_arr, nbt_arr)

    a_arr = dict2array(subsec['A'], dtype=float, order='F', fill_value=0.0)
    nl_arr = dict2array(subsec['NL'], dtype=float, order='F')

    # determine effective LCT based on emitted particle (CM or LAB)
    if lct in (1, 2):
        eff_lct = lct
    elif lct == 3:
        eff_lct = 1 if awp > 4 else 2
    else:
        raise NotImplementedError(f'LCT={lct} not implemented')

    # find enclosing energy intervals
    idcs = find_interval(ei_mesh, energies_in)

    eu = energies_in
    neu = len(eu)
    nmu = len(angle_cosines_out)
    xmu = angle_cosines_out
    nmu = len(xmu)

    result_dim = (neu, nmu)
    result_arr = np.zeros(result_dim, dtype=float)

    for i in range(neu):
        curidx = idcs[i]
        cur_eu = np.array([eu[i]], dtype=float, order='F')
        ilaw = ei_interp[curidx].item()

        e1 = ei_mesh[curidx].item()
        a1 = a_arr[curidx]
        nl1 = nl_arr[curidx].item()

        e2 = ei_mesh[curidx+1].item()
        a2 = a_arr[curidx+1]
        nl2 = nl_arr[curidx+1].item()

        cur_result = np.zeros((1, nmu), dtype=float, order='F')

        # remove `ne` (=1) because automatically inferred
        mf6_get_law2(
            awr, awi, awp, q, eff_lct, lang,
            e1, a1, nl1, e2, a2, nl2,
            ilaw,cur_eu, xmu,nmu, cur_result
        )

        result_arr[i:i+1,:] = cur_result

    return result_arr


def get_dist2d_from_subsec_law6(
    endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out, to_lab
):
    # NOTE: to_lab parameter ignored for LAW=6
    sec = endf_dict[6][mt]
    awr = get_AWR(endf_dict)
    awi = get_AWI(endf_dict)
    q = get_QI(endf_dict, mt)
    subsec = sec['subsection'][subsec_num]
    awp = subsec['AWP']
    apsx = subsec['APSX']
    npsx = subsec['NPSX']

    eu = energies_in
    neu = len(eu)
    epu = energies_out
    nepu = len(epu)
    uu = angle_cosines_out
    nuu = len(uu)

    result_dim = (neu, nepu, nuu)
    result_arr = np.zeros(result_dim, dtype=float, order='F')

    mf6_get_law6(
        awr, awi, awp, q, apsx, npsx,
        eu, epu, uu, nuu, result_arr
    )
    return result_arr


@pad_outside_dist2d_values
def get_dist2d_from_subsec_law7(
    endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out, to_lab
):
    # NOTE: to_lab parameter ignored because LAW=7 always in lab system
    mu = angle_cosines_out
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]

    ei_mesh = dict2array(subsec['E'], dtype=float)
    int_arr = np.array(subsec['E_interpol']['INT'])
    nbt_arr = np.array(subsec['E_interpol']['NBT'])
    ei_interp = convert_interp_repr(int_arr, nbt_arr)

    result_arr = np.zeros(
        (len(energies_in), len(energies_out), len(angle_cosines_out)), dtype=float
    )

    idcs = find_interval(ei_mesh, energies_in)
    for i, curidx in enumerate(idcs): 
        cur_en =  energies_in[i:i+1]
        en1 = ei_mesh[curidx]
        en2 = ei_mesh[curidx+1]
        interp_law = ei_interp[curidx]
        mu_mesh1 = dict2array(subsec['mu'][curidx+1], dtype=float)
        mu_mesh2 = dict2array(subsec['mu'][curidx+2], dtype=float)
        mu_interpol1 = subsec['mu_interpol'][curidx+1] 
        mu_interpol_arr1 = convert_interp_repr(
            np.array(mu_interpol1['INT']), np.array(mu_interpol1['NBT'])
        )
        mu_interpol2 = subsec['mu_interpol'][curidx+2] 
        mu_interpol_arr2 = convert_interp_repr(
            np.array(mu_interpol2['INT']), np.array(mu_interpol2['NBT'])
        )
        idcs21 = find_interval(mu_mesh1, mu) 
        idcs22 = find_interval(mu_mesh2, mu)
        for j, (idx21, idx22) in enumerate(zip(idcs21, idcs22)): 
            cur_mu = mu[j:j+1]
            mu11 = mu_mesh1[idx21]
            mu12 = mu_mesh1[idx21+1]
            mu21 = mu_mesh2[idx22]
            mu22 = mu_mesh2[idx22+1]
            interp_mu_law1 = mu_interpol_arr1[idx21]
            interp_mu_law2 = mu_interpol_arr2[idx22]
            curtable11 = subsec['table'][curidx+1][idx21+1]
            curtable12 = subsec['table'][curidx+1][idx21+2]
            curtable21 = subsec['table'][curidx+2][idx22+1]
            curtable22 = subsec['table'][curidx+2][idx22+2]
            ep11 = np.array(curtable11['Ep'], dtype=float, order='F')
            ep12 = np.array(curtable12['Ep'], dtype=float, order='F')
            ep21 = np.array(curtable21['Ep'], dtype=float, order='F')
            ep22 = np.array(curtable22['Ep'], dtype=float, order='F')
            f11 = np.array(curtable11['f'], dtype=float, order='F')
            f12 = np.array(curtable12['f'], dtype=float, order='F')
            f21 = np.array(curtable21['f'], dtype=float, order='F')
            f22 = np.array(curtable22['f'], dtype=float, order='F')
            np11 = len(ep11)
            np12 = len(ep12)
            np21 = len(ep21)
            np22 = len(ep22)
            ibt11 = np.array(curtable11['INT'], dtype=float, order='F')
            ibt12 = np.array(curtable12['INT'], dtype=float, order='F')
            ibt21 = np.array(curtable21['INT'], dtype=float, order='F')
            ibt22 = np.array(curtable22['INT'], dtype=float, order='F')
            nbt11 = np.array(curtable11['NBT'], dtype=float, order='F')
            nbt12 = np.array(curtable12['NBT'], dtype=float, order='F')
            nbt21 = np.array(curtable21['NBT'], dtype=float, order='F')
            nbt22 = np.array(curtable22['NBT'], dtype=float, order='F')
            nr11 = len(ibt11)
            nr12 = len(ibt12)
            nr21 = len(ibt21)
            nr22 = len(ibt22)

            cur_result_arr = np.zeros((1, len(energies_out), 1), dtype=float, order='F')

            mf6_get_law7(
                cur_en, energies_out, cur_mu, 1, interp_law,
                en1, interp_mu_law1,
                mu11, ep11, f11, np11, nbt11, ibt11, nr11,
                mu12, ep12, f12, np12, nbt12, ibt12, nr12,
                en2, interp_mu_law2,
                mu21, ep21, f21, np21, nbt21, ibt21, nr21,
                mu22, ep22, f22, np22, nbt22, ibt22, nr22,
                cur_result_arr
            )

            result_arr[i:i+1,:,j:j+1] = cur_result_arr
    return result_arr


def get_incident_energies_from_subsec(endf_dict, mt, subsec_num):
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    yield_tab = subsec['yields']
    return np.array(yield_tab['Eint'], copy=True)


def get_emission_energies_from_subsec(endf_dict, mt, subsec_num, nofail=False):
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
        if nofail:
            energies = []
        else:
            raise NotImplementedError(f'Unable to obtain emission energies for LAW={law}')
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
