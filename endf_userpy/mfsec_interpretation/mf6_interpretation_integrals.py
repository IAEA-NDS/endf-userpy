import numpy as np
from ..fortran.endf6 import (
    feep_full_law1con,
)
from .mf6_interpretation_helpers import (
    pad_outside_dist2d_values,
    pad_outside_energydist_values,
)
from ..primitives.helpers import (
    dict2array,
    convert_interp_repr,
    find_interval,
    find_indices_with_tol,
)
from ..primitives.properties import (
    get_AWR,
    get_AWI,
    get_ZA,
    get_ZAI,
)


@pad_outside_energydist_values
def get_energydist_from_subsec_law1(
    endf_dict, mt, subsec_num, energies_in, energies_out, to_lab
):
    sec = endf_dict[6][mt]
    sec['subsection'][subsec_num]['LAW'] == 1
    eu = energies_in
    neu = len(eu)
    epu = energies_out
    nepu = len(epu)
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

    result_dim = (neu, nepu)
    result_arr = np.zeros(result_dim, dtype=np.float64)

    tol=1e-3
    epu = np.array(energies_out, dtype=np.float64, order='F', copy=True)
    nepu = len(epu)
    nepmax = int(max(1e5, nepu + 3e4))

    for i in range(result_arr.shape[0]):
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

        ep = np.zeros(nepmax, dtype=np.float64)
        feep = np.zeros(nepmax, dtype=np.float64, order='F')
        fdev = np.zeros(nepmax, dtype=np.float64, order='F')

        nep_arr = np.zeros(1, dtype=np.int64)

        # NOTE: nep1 and nep2 automatically inferred
        #       from ep1 and ep2, respectively and therefore
        #       not passed.
        feep_full_law1con(
            cur_eu,
            awr, awi, awp, za, zai, zap, eff_lct, lang, lep, lei,
            e1, nd1, na1, ep1, b1, e2, nd2, na2, ep2, b2,
            tol, nepu, epu, nepmax, ep, feep, fdev, nep_arr
        )

        nep = nep_arr.item()

        eouts = ep[:nep]
        idcs = find_indices_with_tol(eouts, epu, rtol=1e-8, atol=1e-10)
        result_arr[i,:] = feep[idcs]

    return result_arr
