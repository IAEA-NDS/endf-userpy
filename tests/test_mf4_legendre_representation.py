import numpy as np
from endf_parserpy import EndfParser
from endf_userpy.endf_interpretation import (
    get_angdist_from_legendre,
    deg2rad,
)
import endf_userpy
import os
import matplotlib.pyplot as plt


parser = EndfParser(ignore_missing_tpid=True)
endf_dict = parser.parsefile(os.path.join('data', 'jeff33_1-H-2g_mf4_mt2.endf'))


# 1 MeV and 50 angles
energies = np.full(100, 1e6)
angcos = np.cos(deg2rad(np.linspace(0.0, 180.0, 50)))



values = get_angdist_from_legendre(endf_dict, 2, energies,  angcos)
values.shape

plt.plot(angcos, values.flatten())
plt.show()

