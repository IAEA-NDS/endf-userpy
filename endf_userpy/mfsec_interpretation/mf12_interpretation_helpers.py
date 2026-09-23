import numpy as np
from ..primitives import properties as prop
from ..primitives.helpers import dict2array
from ..mfsec_interpretation import mf3_interpretation as mf3_interp
from . import mf12_trans2yield_kernel as _kernel


DISCRETE_MT_SERIES = {
    'n': [mt for mt in range(50, 91)],
    'p': [mt for mt in range(600, 649)],
    'd': [mt for mt in range(650, 699)],
    't': [mt for mt in range(700, 749)],
    'h': [mt for mt in range(750, 799)],
    'a': [mt for mt in range(800, 849)],
}

INV_DISCRETE_SERIES_MAP = {mt: k for k, v in DISCRETE_MT_SERIES.items() for mt in v}

MAX_NUM_LEVEL = 60
MAX_NK = 5000


def get_discrete_series_mts(endf_dict, mt, include_ground_state=False):
    ejectile = INV_DISCRETE_SERIES_MAP[mt]
    series_mts = DISCRETE_MT_SERIES[ejectile]
    if not include_ground_state:
        series_mts = series_mts[1:]
    return series_mts


def get_available_series_mts(endf_dict, mt, include_ground_state=False):
    series_mts = get_discrete_series_mts(
        endf_dict, mt, include_ground_state=False
    )
    avail_mts = mf3_interp.get_reaction_mts(endf_dict)
    avail_series_mts = [mt for mt in series_mts if mt in avail_mts]
    if not np.all(np.diff(avail_series_mts) == 1):
        raise IndexError('discrete MT number missing')
    return avail_series_mts


def init_trans2yield(endf_dict, mt):
    """Return ``(available_mts, state_cache)`` for the discrete
    inelastic series containing ``mt``. Pure-Python port of the
    Fortran ``init_trans2yield``; the Fortran-backed variant is
    preserved as :func:`mf12_interpretation_helpers_fort.init_trans2yield_fort_wrapper`.
    """
    elis = prop.get_ELIS(endf_dict)
    avail_series_mts = get_available_series_mts(endf_dict, mt, False)
    qms = [prop.get_QM(endf_dict, mt) for mt in avail_series_mts]
    qis = [prop.get_QI(endf_dict, mt) for mt in avail_series_mts]

    ee, r, a = _kernel.init_trans2yield(elis, qms, qis, MAX_NUM_LEVEL)
    state_cache = {'ee': ee, 'r': r, 'a': a}
    return avail_series_mts, state_cache


def trans2yield(endf_dict, mt, state_cache):
    """Convert the ``mt`` MF12 LO=2 section into photon lines,
    updating ``state_cache`` in place. Pure-Python port; call in
    ascending-MT order per :func:`init_trans2yield`'s
    ``avail_series_mts`` so cascades resolve correctly.
    """
    mtsec = endf_dict[12][mt]
    esns = mtsec['ES_NS']
    nt = len(mtsec['ES'])
    esi = dict2array(mtsec['ES'], dtype=float)
    tp = dict2array(mtsec['TP'], dtype=float)

    if mtsec['LO'] != 2:
        raise ValueError(
            'MT does not contain transition probability tables (LO != 2)'
        )

    if mtsec['LG'] == 1:
        gp = np.ones(nt, dtype=float)
    elif mtsec['LG'] == 2:
        gp = dict2array(mtsec['GP'], dtype=float)
    else:
        raise ValueError(
            f'invalid value for LG encountered (LG={mtsec["LG"]})'
        )

    return _kernel.trans2yield(
        mt, esns, esi, tp, gp,
        state_cache['ee'], state_cache['r'], state_cache['a'],
        maxnk=MAX_NK,
    )
