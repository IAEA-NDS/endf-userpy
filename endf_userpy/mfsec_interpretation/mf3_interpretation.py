import numpy as np
from ..primitives.helpers import treat_duplicates
from ..primitives.interpolation import interp_tab1
from ..primitives.properties import (
    get_projectile,
    get_reaction_string_for_mt,
)


def get_nominal_incident_energies(endf_dict, mt):
    sec = endf_dict[3][mt]
    xstab = sec['xstable']
    return np.array(xstab['E'], copy=True)


def get_nominal_incident_energy_range(endf_dict, mt):
    energies = get_nominal_incident_energies(endf_dict, mt)
    return (min(energies), max(energies))


def get_incident_energies(endf_dict, mt):
    energies = get_nominal_incident_energies(endf_dict, mt)
    treat_duplicates(energies, inplace=True)
    xs = np.array(endf_dict[3][mt]['xstable']['xs'], dtype=float)
    idcs = np.nonzero(xs)[0]
    first_idx = idcs[0]-1 if idcs[0] > 0 else 0
    last_idx = idcs[-1]+1 if idcs[-1]+1 < len(idcs) else idcs[-1]
    return energies[first_idx:last_idx+1].copy()


def get_incident_energy_range(endf_dict, mt):
    eincs = get_incident_energies(endf_dict, mt)
    return (min(eincs), max(eincs))


def get_reaction_mts(endf_dict):
    return list(endf_dict[3].keys())


def get_reactions(endf_dict):
    mts = get_reaction_mts(endf_dict)
    reacs = [get_reaction_string_for_mt(endf_dict, m) for m in mts] 
    return reacs


def compute_cross_section(endf_dict, mt, energies_in):
    sec = endf_dict[3][mt]
    xstab = sec['xstable']
    return interp_tab1(energies_in, xstab, 'E', 'xs', outside_value=0.0)
