import numpy as np
from .fortran.endf6 import (
    mf6_get_law1,
    mf6_get_law2,
    mf6_get_law6,
)
from .helpers import (
    dict2array,
    convert_interp_repr,
    find_interval,
)
from .properties import (
    get_AWR,
    get_AWI,
    get_AWP,
    get_ZA,
    get_ZAI,
    get_QI,
)


def get_ddx_from_subsec_law1(
    endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out
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
    lct = sec['LCT']

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

    return disc_result_arr, cont_result_arr


def get_ddx_from_subsec_law2(
    endf_dict, mt, subsec_num, energies_in, angle_cosines_out
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
    lct = sec['LCT']
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


def get_ddx_from_subsec_law6(
    endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out
):
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


def compute_ddx_from_subsec(
    endf_dict, mt, subsec_num,
    energies_in, energies_out, angle_cosines_out
):
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    law = subsec['LAW']
    if law == 1:
        return get_ddx_from_subsec_law1(
            endf_dict, mt, subsec_num,
            energies_in, energies_out, angle_cosines_out
        )
    elif law == 2:
        return get_ddx_from_subsec_law2(
            endf_dict, mt, subsec_num,
            energies_in, angle_cosines_out
        )
    elif law == 6:
        return get_ddx_from_subsec_law6(
            endf_dict, mt, subsec_num,
            energies_in, energies_out, angle_cosines_out
        )
    else:
        raise NotImplementedError(
            f'DDX interpretation for LAW={law} not implemented.'
        )
