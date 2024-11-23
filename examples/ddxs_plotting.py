import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from endf_parserpy import EndfParserCpp

from endf_userpy.helpers import deg2rad
from endf_userpy.reactions import get_reaction_string_for_mt
from endf_userpy.properties import get_QI
from endf_userpy.mf3_interpretation import (
    get_reactions,
    get_reaction_mts,
    get_incident_energy_range,
)
from endf_userpy.mf6_interpretation_helpers import (
    get_zaps_for_mt,
)
from endf_userpy.quantities import (
    compute_xs,
    compute_dxs,
    compute_ddxs,
)
from endf_userpy.mf6_interpretation import compute_dist2d_values


endf_file = os.path.join('..', 'tests', 'data', 'n-004_Be_009.endf')
parser = EndfParserCpp()
endf_dict = parser.parsefile(endf_file)

reacs = get_reactions(endf_dict)
reac_mts = get_reaction_mts(endf_dict)
reac_dt = pd.DataFrame({'reaction': reacs, 'mt': reac_mts})
reac_dt

mt = 16
energy_range = get_incident_energy_range(endf_dict, 16)
energies = np.linspace(energy_range[0], energy_range[1], 1000) 
xs = compute_xs(endf_dict, 16, energies)

plt.plot(energies, xs)
plt.xlabel('energy [eV]')
plt.ylabel('cross section [barn]')
plt.show()

zaps = get_zaps_for_mt(endf_dict, mt)
print(zaps)

cur_energy = 5e6
qval = get_QI(endf_dict, mt) 
energies_in = np.array([cur_energy])
energies_out = np.linspace(1e-10, cur_energy+qval, 1000)  
angle_cosines_out = np.linspace(-1, 1, 1000)

ddx_values = compute_ddxs(endf_dict, mt, zaps[0], energies_in, energies_out, angle_cosines_out)

# make a beautiful 3d plot
Z_values = ddx_values[0,:,:]
X, Y = np.meshgrid(angle_cosines_out, energies_out) 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z_values, cmap='coolwarm', edgecolor='none')
ax.set_zscale('log')
fig.colorbar(surf)
plt.title(f'incident energy: {cur_energy/1e6:.2f} MeV')
plt.show()


# dist = ddx_values[0,:,:]
# dist1d = np.sum(dist, axis=0) * (energies_out[1] - energies_out[0])
# np.sum(dist1d) * (angle_cosines_out[1] - angle_cosines_out[0])
energies_out = np.linspace(1e-5, 3.5e6, int(1e4))
ddx_values = compute_ddxs(
    endf_dict, mt, zaps[0], np.array([5e6]), energies_out,
    np.cos([deg2rad(60)])
).flatten()

plt.plot(energies_out, ddx_values)
plt.yscale('log')
plt.show()

