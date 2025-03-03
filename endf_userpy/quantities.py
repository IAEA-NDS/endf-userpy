import numpy as np
from .primitives.physical_constants import get_zap_for_particle
from .quantities_mt_zap.quantities import (
    compute_yields,
    compute_xs,
    get_reaction_mt_numbers,
    compute_cumulative_quantity,
)
from .quantities_mt_zap.selectors import (
    contains_zap
)


# TODO: check and complete


def get_particle_production_xs(endf_dict, particle, energies_in):
    zap = get_zap_for_particle(particle)
    return compute_cumulative_quantity(
        lambda endf_dict, mt, zap, energies_in: (
            compute_yields(endf_dict, mt, zap, energies_in) *
            compute_xs(endf_dict, mt, energies_in)
        ),
        lambda endf_dict, mt, zap, energies_in: contains_zap(endf_dict, mt, zap),
        endf_dict, zap, energies_in
    )


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
