import numpy as np
from .interpolation import interp_tab1
from .properties import get_projectile
from .reactions import get_reaction_string_for_mt


def get_incident_energies(endf_dict, mt):
    sec = endf_dict[3][mt]
    xstab = sec['xstable']
    return np.array(xstab['E'])


def get_reaction_mts(endf_dict):
    return list(endf_dict[3].keys())


def get_reactions(endf_dict):
    mts = get_reaction_mts(endf_dict) 
    proj = get_projectile(endf_dict)
    reacs = [get_reaction_string_for_mt(r) for r in mts] 
    reacs = [r.replace('(z,', f'({proj},') for r in reacs]
    reacs = [r.replace('(y,', f'({proj},') for r in reacs]
    reacs = [r.replace(',z', f',{proj}') for r in reacs]
    return reacs


def get_incident_energy_range(endf_dict, mt):
    eincs = get_incident_energies(endf_dict, mt)
    return (min(eincs), max(eincs))


def compute_cross_section(endf_dict, mt, energies_in):
    sec = endf_dict[3][mt]
    xstab = sec['xstable']
    return interp_tab1(energies_in, xstab, 'E', 'xs')
