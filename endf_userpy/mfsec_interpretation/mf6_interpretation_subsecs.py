import numpy as np
from ..fortran.endf6 import (
    mf6_get_law1,
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


def pad_outside_values(func):

    def wrapfunc(
        endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out, *args, **kwargs
    ):
        eincs = energies_in
        eouts = energies_out
        subsec = endf_dict[6][mt]['subsection'][subsec_num]
        ei_mesh = dict2array(subsec['E'], dtype=float)
        is_inside_x = (eincs >= np.min(ei_mesh)) & (eincs <= np.max(ei_mesh))
        is_inside_y = (eouts >= 1e-20)
        eincs_inside = eincs[is_inside_x]
        eouts_inside = eouts[is_inside_y]
        # special casing regular case with all points within mesh for small speedup
        if len(eincs_inside) == len(eincs) and len(eouts_inside) == len(eouts):
            return func(
                endf_dict, mt, subsec_num,
                eincs, eouts, angle_cosines_out,
                *args, **kwargs
            )
        # special treatment if some points outside mesh
        res_dim = (len(eincs), len(eouts), len(angle_cosines_out))
        res_arr = np.full(res_dim, 0.0, dtype=float)
        if len(eincs_inside) == 0 or len(eouts_inside) == 0:
            return res_arr
        arr_inside = func(
            endf_dict, mt, subsec_num,
            eincs_inside, eouts_inside, angle_cosines_out,
            *args, **kwargs
        )
        res_arr[np.ix_(is_inside_x, is_inside_y, np.array([True]))] =  arr_inside
        return res_arr

    return wrapfunc


@pad_outside_values
def get_dist2d_from_subsec_law1(
    endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out, to_lab
):
    sec = endf_dict[6][mt]
    sec['subsection'][subsec_num]['LAW'] == 1
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
    nep_arr = dict2array(subsec['NEP'], dtype=int)

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
        lei = ei_interp[curidx]

        e1 = ei_mesh[curidx]
        nd1 = nd_arr[curidx]
        na1 = na_arr[curidx]
        ep1 = dict2array(subsec['Ep'][curidx+1], dtype=float, order='F')
        nep1 = len(ep1)  # also nep_arr[curidx]
        b1 = dict2array(subsec['b'][curidx+1], dtype=float, order='F')

        e2 = ei_mesh[curidx+1]
        nd2 = nd_arr[curidx+1]
        na2 = na_arr[curidx+1]
        ep2 = dict2array(subsec['Ep'][curidx+2], dtype=float, order='F')
        nep2 = len(ep2)
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


@pad_outside_values
def get_dist1d_from_subsec_law2(
    endf_dict, mt, subsec_num, energies_in, angle_cosines_out, to_lab
):
    # TODO: how to signal to the user that this is a
    #       discrete distribution with a dirac delta in the
    #       the emission energy.
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
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
        ilaw = ei_interp[curidx]

        e1 = ei_mesh[curidx]
        a1 = a_arr[curidx]
        nl1 = nl_arr[curidx]

        e2 = ei_mesh[curidx+1]
        a2 = a_arr[curidx+1]
        nl2 = nl_arr[curidx+1]

        cur_result = np.zeros((1, nmu), dtype=float, order='F')

        # remove `ne` (=1) because automatically inferred
        mf6_get_law2(
            awr, awi, awp, q, lct, lang,
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


@pad_outside_values
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
