import numpy as np
from .fortran.endf6 import mf6_get_law1 
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
    else:
        raise NotImplementedError(
            f'DDX interpretation for LAW={law} not implemented.'
        )
