import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from endf_parserpy import EndfParserCpp, EndfDict
from endf_parserpy.utils.user_tools import show_content

from endf_userpy.helpers import deg2rad
from endf_userpy.reactions import get_reaction_string_for_mt
from endf_userpy.properties import get_zap_for_particle
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
)
from endf_userpy.mf6_interpretation import compute_dist2d_values


endf_file = os.path.join('..', 'tests', 'data', 'n-004_Be_009.endf')
parser = EndfParserCpp()
endf_dict = parser.parsefile(endf_file)

reacs = get_reactions(endf_dict)
reac_mts = get_reaction_mts(endf_dict)
reac_dt = pd.DataFrame({'reaction': reacs, 'mt': reac_mts}).set_index('mt')
reac_dt

mt = 2
energy_range = get_incident_energy_range(endf_dict, mt)
energies = np.linspace(energy_range[0], energy_range[1], 1000) 
xs = compute_xs(endf_dict, mt, energies)

plt.plot(energies / 1e6, xs)
plt.xlabel('energy [MeV]')
plt.ylabel('cross section [barn]')
plt.title(reac_dt.at[mt, "reaction"])
plt.show()


endf_dict[4].keys()

zap = get_zap_for_particle('n')
energies_in = np.linspace(0.1e6, 1e6, 1000)
angle_cosines_out = np.linspace(-1, 1, 1000)
dx_values = compute_dxs(endf_dict, mt, zap, energies_in, angle_cosines_out)

# make a beautiful 3d plot
Z_values = dx_values[:,:]
X, Y = np.meshgrid(angle_cosines_out, energies_in / 1e6) 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z_values, cmap='coolwarm', edgecolor='none')
ax.set_zscale('log')
fig.colorbar(surf)
plt.title(f'angle-differential xs for {reac_dt.at[mt, "reaction"]}')
plt.ylabel('incident energy [MeV]')
plt.xlabel('angle cosine')
plt.show()

# make a reduced plot showing only one incident energy

cur_einc = 6.5e5
angle_cosines_out = np.linspace(-1, 1, 1000)
dxvals = compute_dxs(
    endf_dict, mt, zap, np.array([cur_einc]), angle_cosines_out
).flatten()

plt.plot(angle_cosines_out, dxvals)
plt.title(f'angle-differential xs for {reac_dt.at[mt, "reaction"]} at Einc={cur_einc/1e6} MeV')
plt.xlabel('angle cosine')
plt.ylabel('angle-differential xs [barn/sr]')
plt.show()

