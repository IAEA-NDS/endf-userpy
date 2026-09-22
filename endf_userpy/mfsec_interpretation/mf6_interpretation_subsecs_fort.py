"""Fortran-backed MF6 sub-section reconstructors, kept as an
equivalence oracle for the pure-Python implementations in
:mod:`mf6_interpretation_subsecs`.

Not imported by any production code path (see
:mod:`endf_userpy.quantities_mt_zap.distribution2d`); the pytest
suite in ``tests/test_mf6_law*_equivalence.py`` imports this module
and asserts bit-for-bit agreement between each function here and
its Python sibling on a representative corpus. Mirrors the
existing MF4 split: ``mf4_interpretation.py`` (production Python)
vs :mod:`mf4_interpretation_fort` (Fortran oracle).

As each LAW routine gets ported (issue #47), its Fortran-backed
version moves from ``mf6_interpretation_subsecs.py`` into this
module. Once every LAW has a Python implementation, ``endf6.f90``
can move to ``tests_fortran/`` and drop out of the wheel entirely.
"""
import numpy as np
from ..fortran.endf6 import mf6_get_law2, mf6_get_law6, mf6_get_law7
from ..primitives.properties import get_AWI, get_AWR, get_QI
from ..primitives.helpers import (
    dict2array,
    convert_interp_repr,
    find_interval,
)


def get_dist2d_from_subsec_law6(
    endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out,
    to_lab,
):
    # NOTE: to_lab parameter ignored for LAW=6.
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
        eu, epu, uu, nuu, result_arr,
    )
    return result_arr


def get_angdist_from_subsec_law2(
    endf_dict, mt, subsec_num, energies_in, angle_cosines_out, to_lab,
):
    """Fortran-backed MF6 LAW=2 angular distribution. Kept here as
    the equivalence oracle for the pure-Python
    :func:`mf6_interpretation_subsecs.get_angdist_from_subsec_law2`.

    Copy of the pre-port implementation from
    ``mf6_interpretation_subsecs.py``: per query incident energy,
    find the enclosing panel, call the Fortran ``mf6_get_law2``
    with the two panels' coefficients and the requested cosines.
    """
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

    if lct in (1, 2):
        eff_lct = lct
    elif lct == 3:
        eff_lct = 1 if awp > 4 else 2
    else:
        raise NotImplementedError(f'LCT={lct} not implemented')

    idcs = find_interval(ei_mesh, energies_in)

    eu = energies_in
    neu = len(eu)
    xmu = angle_cosines_out
    nmu = len(xmu)

    result_arr = np.zeros((neu, nmu), dtype=float)

    for i in range(neu):
        curidx = idcs[i]
        cur_eu = np.array([eu[i]], dtype=float, order='F')
        ilaw = ei_interp[curidx].item()

        e1 = ei_mesh[curidx].item()
        a1 = a_arr[curidx]
        nl1 = nl_arr[curidx].item()

        e2 = ei_mesh[curidx + 1].item()
        a2 = a_arr[curidx + 1]
        nl2 = nl_arr[curidx + 1].item()

        cur_result = np.zeros((1, nmu), dtype=float, order='F')

        mf6_get_law2(
            awr, awi, awp, q, eff_lct, lang,
            e1, a1, nl1, e2, a2, nl2,
            ilaw, cur_eu, xmu, nmu, cur_result,
        )
        result_arr[i:i + 1, :] = cur_result

    return result_arr


def get_dist2d_from_subsec_law7(
    endf_dict, mt, subsec_num, energies_in, energies_out, angle_cosines_out,
    to_lab,
):
    """Fortran-backed MF6 LAW=7 (tabulated E'/mu double-differential)
    reconstruction. Copy of the pre-port implementation from
    ``mf6_interpretation_subsecs.py`` for equivalence testing.

    LAW=7 is always LAB; ``to_lab`` is ignored.
    """
    mu = angle_cosines_out
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]

    ei_mesh = dict2array(subsec['E'], dtype=float)
    int_arr = np.array(subsec['E_interpol']['INT'])
    nbt_arr = np.array(subsec['E_interpol']['NBT'])
    ei_interp = convert_interp_repr(int_arr, nbt_arr)

    result_arr = np.zeros(
        (len(energies_in), len(energies_out), len(angle_cosines_out)),
        dtype=float,
    )

    idcs = find_interval(ei_mesh, energies_in)
    for i, curidx in enumerate(idcs):
        cur_en = energies_in[i:i + 1]
        en1 = ei_mesh[curidx]
        en2 = ei_mesh[curidx + 1]
        interp_law = ei_interp[curidx]
        mu_mesh1 = dict2array(subsec['mu'][curidx + 1], dtype=float)
        mu_mesh2 = dict2array(subsec['mu'][curidx + 2], dtype=float)
        mu_interpol1 = subsec['mu_interpol'][curidx + 1]
        mu_interpol_arr1 = convert_interp_repr(
            np.array(mu_interpol1['INT']), np.array(mu_interpol1['NBT']),
        )
        mu_interpol2 = subsec['mu_interpol'][curidx + 2]
        mu_interpol_arr2 = convert_interp_repr(
            np.array(mu_interpol2['INT']), np.array(mu_interpol2['NBT']),
        )
        idcs21 = find_interval(mu_mesh1, mu)
        idcs22 = find_interval(mu_mesh2, mu)
        for j, (idx21, idx22) in enumerate(zip(idcs21, idcs22)):
            cur_mu = mu[j:j + 1]
            mu11 = mu_mesh1[idx21]
            mu12 = mu_mesh1[idx21 + 1]
            mu21 = mu_mesh2[idx22]
            mu22 = mu_mesh2[idx22 + 1]
            interp_mu_law1 = mu_interpol_arr1[idx21]
            interp_mu_law2 = mu_interpol_arr2[idx22]
            curtable11 = subsec['table'][curidx + 1][idx21 + 1]
            curtable12 = subsec['table'][curidx + 1][idx21 + 2]
            curtable21 = subsec['table'][curidx + 2][idx22 + 1]
            curtable22 = subsec['table'][curidx + 2][idx22 + 2]
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

            cur_result_arr = np.zeros(
                (1, len(energies_out), 1), dtype=float, order='F',
            )

            mf6_get_law7(
                cur_en, energies_out, cur_mu, 1, interp_law,
                en1, interp_mu_law1,
                mu11, ep11, f11, np11, nbt11, ibt11, nr11,
                mu12, ep12, f12, np12, nbt12, ibt12, nr12,
                en2, interp_mu_law2,
                mu21, ep21, f21, np21, nbt21, ibt21, nr21,
                mu22, ep22, f22, np22, nbt22, ibt22, nr22,
                cur_result_arr,
            )
            result_arr[i:i + 1, :, j:j + 1] = cur_result_arr
    return result_arr
