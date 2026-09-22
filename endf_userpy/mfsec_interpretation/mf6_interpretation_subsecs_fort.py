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
from ..fortran.endf6 import mf6_get_law2, mf6_get_law6
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
