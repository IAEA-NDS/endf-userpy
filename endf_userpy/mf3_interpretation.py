import numpy as np
from .interpolation import interp_tab1
from .properties import (
    get_projectile,
    get_reaction_string_for_mt,
)


def get_incident_energies(endf_dict, mt):
    sec = endf_dict[3][mt]
    xstab = sec['xstable']
    return np.array(xstab['E'])


def get_reaction_mts(endf_dict):
    return list(endf_dict[3].keys())


def get_reactions(endf_dict):
    mts = get_reaction_mts(endf_dict)
    reacs = [get_reaction_string_for_mt(endf_dict, m) for m in mts] 
    return reacs


def get_incident_energy_range(endf_dict, mt):
    eincs = get_incident_energies(endf_dict, mt)
    return (min(eincs), max(eincs))


def compute_cross_section(endf_dict, mt, energies_in):
    sec = endf_dict[3][mt]
    xstab = sec['xstable']
    return interp_tab1(energies_in, xstab, 'E', 'xs')
