import numpy as np
from ..primitives.helpers import dict2array
from ..mfsec_interpretation import mf3_interpretation as mf3_interp
from ..primitives import properties as prop
from ..fortran.endf6 import (
    trans2yield as trans2yield_fort,
    init_trans2yield as init_trans2yield_fort,
)


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


def _get_discrete_series_mts(endf_dict, mt, include_ground_state=False):
    ejectile = INV_DISCRETE_SERIES_MAP[mt]
    series_mts = DISCRETE_MT_SERIES[ejectile]
    if not include_ground_state:
        series_mts = series_mts[1:]
    return series_mts


def init_trans2yield(endf_dict, mt): 
    maxlevel = MAX_NUM_LEVEL 
    elis = prop.get_ELIS(endf_dict)
    series_mts = _get_discrete_series_mts(
        endf_dict, mt, include_ground_state=False
    )

    avail_mts = mf3_interp.get_reaction_mts(endf_dict) 
    avail_series_mts = [mt for mt in series_mts if mt in avail_mts] 
    if not np.all(np.diff(avail_series_mts) == 1):
        raise IndexError('discrete MT number missing')

    nlevel = len(avail_series_mts)
    
    qms = [prop.get_QM(endf_dict, mt) for mt in avail_series_mts]   
    qis = [prop.get_QI(endf_dict, mt) for mt in avail_series_mts]

    ee = np.empty(maxlevel, dtype=float, order='F')  
    a = np.empty((maxlevel, maxlevel), dtype=float, order='F')
    r = np.empty((maxlevel, maxlevel), dtype=float, order='F')

    init_trans2yield_fort(elis, nlevel, qms, qis, ee, r, a)
    state_cache = {
        'ee': ee, 'r': r, 'a': a
    }
    return avail_series_mts, state_cache


# subroutine trans2yield(mt,esns,nt,esi,tp,gp,maxlevel,ee,r,a,maxnk,nk,es,eg,y) ! in :endf6:endf6.f90
def trans2yield(endf_dict, mt, state_cache):
    maxlevel = MAX_NUM_LEVEL
    maxnk = MAX_NK 

    ee = state_cache['ee']
    r = state_cache['r']
    a = state_cache['a']

    mtsec = endf_dict[12][mt]
    esns = mtsec['ES_NS'] 
    nt = len(mtsec['ES'])
    esi = dict2array(mtsec['ES'], dtype=float, order='F')
    tp = dict2array(mtsec['TP'], dtype=float, order='F')

    if mtsec['LG'] == 1:
        gp = np.ones(nt, dtype=float)
    elif mtsec['LG'] == 2:
        gp = dict2array(mtsec['GP'], dtype=float, order='F')
    else:
        raise ValueError(f'invalid value for LG encountered (LG={LG})')

    # allocate variables for output vars from fortran subroutine

    # energy level from which photon originates
    es = np.zeros(maxnk, dtype=float, order='F') 
    # photon energy 
    eg = np.zeros(maxnk, dtype=float, order='F')
    # photon yield
    y = np.zeros(maxnk, dtype=float, order='F')
    # number of transitions
    nko = np.empty(1, dtype=np.int64)

    # call fortran subroutine
    trans2yield_fort(mt, esns, nt, esi, tp, gp, ee, r, a, maxnk, nko, es, eg, y)
    nk = int(nko.item())
    return {
        'level_energy': es[:nk],
        'photon_energy': eg[:nk],
        'photon_yield': y[:nk],
    }


def compute_cascade_photon_yields(endf_dict, mts):
    series_mts = _get_discrete_series_mts(
        endf_dict, mts[0], include_ground_state=False
    )
    if not np.all(np.isin(mts, series_mts)):
        raise ValueError(
            'All MT numbers must belong to the same discrete '
            ' MT series (e.g. fall in the range from 51 to 90 '
            'for inelastic neutron scattering.'
        )

    disc_mts, state_cache = init_trans2yield(endf_dict, mts[0]) 

    user_mts = set(mts)
    user_max_mt = max(user_mts)
    results = {} 
    for mt in disc_mts: 
        cur_result = (
            trans2yield(endf_dict, mt, state_cache)
        )
        if mt in user_mts:
            results[mt] = cur_result
            if mt == user_max_mt:
                break
    return results
