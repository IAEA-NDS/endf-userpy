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
from ..fortran.endf6 import mf6_get_law6
from ..primitives.properties import get_AWI, get_AWR, get_QI


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
