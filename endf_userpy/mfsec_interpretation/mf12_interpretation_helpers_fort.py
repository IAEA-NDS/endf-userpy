"""Fortran-backed MF12 LO=2 photon-yield helpers.

Preserved as the equivalence oracle for the Python port in
:mod:`mf12_trans2yield_kernel`. Nothing in the runtime path uses
these; parity tests use them to pin the Python port.
"""
import numpy as np
from ..fortran.endf6 import (
    trans2yield as trans2yield_fort,
    init_trans2yield as init_trans2yield_fort,
)
from ..primitives import properties as prop
from ..primitives.helpers import dict2array
from .mf12_interpretation_helpers import (
    MAX_NUM_LEVEL,
    MAX_NK,
    get_available_series_mts,
)


def init_trans2yield_fort_wrapper(endf_dict, mt):
    maxlevel = MAX_NUM_LEVEL
    elis = prop.get_ELIS(endf_dict)
    avail_series_mts = get_available_series_mts(endf_dict, mt, False)

    nlevel = len(avail_series_mts)
    qms = [prop.get_QM(endf_dict, mt) for mt in avail_series_mts]
    qis = [prop.get_QI(endf_dict, mt) for mt in avail_series_mts]

    ee = np.empty(maxlevel, dtype=float, order='F')
    a = np.empty((maxlevel, maxlevel), dtype=float, order='F')
    r = np.empty((maxlevel, maxlevel), dtype=float, order='F')

    init_trans2yield_fort(elis, nlevel, qms, qis, ee, r, a)
    state_cache = {'ee': ee, 'r': r, 'a': a}
    return avail_series_mts, state_cache


def trans2yield_fort_wrapper(endf_dict, mt, state_cache):
    maxnk = MAX_NK

    ee = state_cache['ee']
    r = state_cache['r']
    a = state_cache['a']

    mtsec = endf_dict[12][mt]
    esns = mtsec['ES_NS']
    nt = len(mtsec['ES'])
    esi = dict2array(mtsec['ES'], dtype=float, order='F')
    tp = dict2array(mtsec['TP'], dtype=float, order='F')

    if mtsec['LO'] != 2:
        raise ValueError(
            'MT does not contain transition probability tables (LO != 2)'
        )

    if mtsec['LG'] == 1:
        gp = np.ones(nt, dtype=float)
    elif mtsec['LG'] == 2:
        gp = dict2array(mtsec['GP'], dtype=float, order='F')
    else:
        raise ValueError(
            f'invalid value for LG encountered (LG={mtsec["LG"]})'
        )

    es = np.zeros(maxnk, dtype=float, order='F')
    eg = np.zeros(maxnk, dtype=float, order='F')
    y = np.zeros(maxnk, dtype=float, order='F')
    nko = np.empty(1, dtype=np.int64)

    trans2yield_fort(mt, esns, nt, esi, tp, gp, ee, r, a, maxnk, nko, es, eg, y)
    nk = int(nko.item())
    return {
        'level_energy': es[:nk],
        'photon_energy': eg[:nk],
        'photon_yield': y[:nk],
    }
