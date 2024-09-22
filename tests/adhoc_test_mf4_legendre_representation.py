import numpy as np
from endf_parserpy import EndfParser
from endf_userpy.mf4_interpretation import (
    compute_angdist_from_legrepr,
    compute_angdist,
)
from endf_userpy.mf4_interpretation_fort import (
    compute_angdist as compute_angdist_fort,
)
from endf_userpy.helpers import deg2rad
import endf_userpy
import os
import matplotlib.pyplot as plt


parser = EndfParser(ignore_missing_tpid=True)
endf_dict = parser.parsefile(os.path.join('data', 'jeff33_1-H-2g_mf4_mt2.endf'))


# 1 MeV and 50 angles
energies = np.array([1e6, 2e6, 3e6]) 
angcos = np.cos(deg2rad(np.linspace(0.0, 180.0, 5)))


res = compute_angdist_from_legrepr(endf_dict[4][2], energies, angcos)
res = compute_angdist(endf_dict, 2, energies, angcos)


values = compute_angdist_fort(endf_dict, 2, energies,  angcos)
values
values.shape

plt.plot(angcos, values.flatten())
plt.show()



from endf_userpy.fortran.endf6 import yleg


x = np.cos(np.linspace(10, 150, 5))
a = np.array([0, 1, 2], dtype=float)

apy = a * (np.arange(len(a)) + 0.5)

yleg(x[2], a, len(a))


from numpy.polynomial.legendre import Legendre
Legendre(apy)(x)






