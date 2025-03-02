import numpy as np
from .primitives.physical_constants import get_zap_for_particle
from .quantities_mt_zap.quantities import (
    compute_yields,
    compute_xs,
    get_reaction_mt_numbers,
)


# TODO: check and complete


def get_particle_production_xs(endf_dict, particle, energies_in):
    zap = get_zap_for_particle(particle)
    prodxs = np.zeros_like(energies_in, dtype=float)
    mt_list = get_reaction_mt_numbers(endf_dict)
    for mt in mt_list:
        try:
            particle_yield = compute_yields(endf_dict, mt, zap, energies_in)
        except Exception:
            continue
        curxs = compute_xs(endf_dict, mt, energies_in)
        prodxs += curxs * particle_yield
    return prodxs


def get_particle_production_dxs_dmu(endf_dict, particle, energies_in, angle_cosines_out):
    zap = get_zap_for_particle(particle)
    res = np.zeros((len(energies_in), len(angle_cosines_out)), dtype=float)
    mt_list = get_reaction_mt_numbers(endf_dict)
    for mt in mt_list:
        try:
            particle_yield = compute_yields(endf_dict, mt, zap, energies_in)
        except Exception:
            continue
        curres = compute_daxs(endf_dict, mt, zap, energies_in, angle_cosines_out)  
        res += curres 
    return res
