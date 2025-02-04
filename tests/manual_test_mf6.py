import os
import numpy as np
from scipy.integrate import dblquad
from endf_parserpy import EndfParserCpp
from endf_userpy.mf6_interpretation_helpers import (
    get_zaps_for_all_mts,
    find_subsec_num,
    get_subsec,
    contains_subsec_dist2d,
    has_disc_part,
    has_cont_part,
)
from endf_userpy.mf6_interpretation import (
    get_incident_energies,
    compute_dist2d_values,
)
from endf_userpy.properties import get_QI


def create_dist2d_fun(endf_dict, mt, zap, energy):
    def dist2d(x, y):
        ens = np.array([energy], dtype=float)
        ens_out = np.array([y], dtype=float)
        mus_out = np.array([x], dtype=float)
        return compute_dist2d_values(endf_dict, mt, zap, ens, ens_out, mus_out)
    
    return dist2d


endf_file = os.path.join('data', 'n-004_Be_009.endf') 
parser = EndfParserCpp()
endf_dict = parser.parsefile(endf_file)


def logprint(logmsg):
    print(logmsg, end='')
    with open('log_mf6_n_0425_4-Be-9.txt', 'a') as file:
        file.writelines(logmsg)


mt_zaps = get_zaps_for_all_mts(endf_dict)
for mt, zaps in mt_zaps.items():
    qval = get_QI(endf_dict, mt)
    for zap in zaps: 
        eincs = get_incident_energies(endf_dict, mt, zap)
        subsec_num = find_subsec_num(endf_dict, mt, zap)
        subsec = endf_dict[6][mt]['subsection'][subsec_num]
        # prepare some info for printing
        law = subsec['LAW']
        if not contains_subsec_dist2d(endf_dict, mt, subsec_num):
            logprint(f'skipping MF6/MT{mt} (ZAP={zap}) because not a 2d distribution (LAW={law})\n') 
            continue
        lawstr = f'LAW={law}'
        if has_disc_part(endf_dict, mt, zap):
            logprint(f'skipping MF6/MF:{mt}/ZAP:{zap} because discrete part present ({lawstr})')
            continue
        if law == 1:
            lang = subsec['LANG'] 
            lawstr += f' LANG={lang}'
        # remove the 1e-5 point for MT5
        if mt == 5 and eincs[0] == 1e-5:
            eincs = eincs[1:]
        eincs = np.array(min(eincs) + (max(eincs)-min(eincs)) * np.array([0, 0.21234, 0.79381, 1.0]))
        # check normalization for a couple of energies
        logprint("\n")
        for einc in eincs:
            dist2d = create_dist2d_fun(endf_dict, mt, zap, einc)
            intres = dblquad(dist2d, 1e-10, einc+qval, -1, 1, epsabs=0.01, epsrel=1e-2)
            logmsg = f'MT: {mt} ZAP: {zap} EINC: {einc} INT: {intres[0]:.2f}+-{intres[1]:.2f} ({lawstr})\n'
            if intres[0] >= 0.95 and intres[0] <= 1.05:
                logmsg = "ok   " + logmsg
            else:
                logmsg = "fail " + logmsg
            logprint(logmsg)

