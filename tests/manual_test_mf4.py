import os
import numpy as np
from scipy.integrate import quad
from endf_parserpy import EndfParserCpp
from endf_userpy.mf3_interpretation import get_incident_energies
from endf_userpy.mf4_interpretation_fort import compute_angdist_values


def create_angdist_fun(endf_dict, mt, energy):
    def angdist(x):
        ens = np.array([energy], dtype=float)
        mus_out = np.array([x], dtype=float)
        return compute_angdist_values(endf_dict, mt, ens, mus_out)
    
    return angdist


endf_file = os.path.join('data', 'n-004_Be_009.endf')
parser = EndfParserCpp()
endf_dict = parser.parsefile(endf_file)


mf4 = endf_dict[4]
for mt in mf4.keys():
    eincs = get_incident_energies(endf_dict, mt)
    eincs = min(eincs) + (max(eincs)-min(eincs)) * np.array([0, 0.2412, 0.7213, 1.0] )
    ltt = mf4[mt]['LTT'] 
    li = mf4[mt]['LI'] 
    for einc in eincs:
        angdist_fun = create_angdist_fun(endf_dict, 2, eincs[0]) 
        intres = quad(angdist_fun, -1, 1)
        logmsg = f'MT:{mt} - EINC: {einc} - INTRES: {intres[0]:.2f}+-{intres[1]:.2f} (LTT={ltt}, LI={li})'
        if intres[0] >= 0.95 and intres[1] <= 1.05:
            logmsg = "ok   " + logmsg
        else:
            logmsg = "fail " + logmsg
        print(logmsg)

