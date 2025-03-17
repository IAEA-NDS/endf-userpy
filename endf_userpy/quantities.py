import numpy as np
from .primitives import physical_constants as physconst
from .primitives import properties as prop
from .primitives import reactions as reac
from .quantities_mt_zap.quantities import (
    compute_yields,
    compute_xs,
    compute_dexs,
    compute_daxs,
    get_reaction_mt_numbers,
    compute_cumulative_quantity,
)
from .quantities_mt_zap import distribution1d as dist1d
from .quantities_mt_zap.selectors import (
    contains_zap,
    satisfies_select_heuristic,
)


# TODO: check and complete


def get_available_reactions(endf_dict):
    mts = get_reaction_mt_numbers(endf_dict)
    reacs = [
        prop.get_reaction_string_for_mt(endf_dict, mt)
        for mt in mts
    ]
    return reacs


def get_reaction_xs(endf_dict, reaction, energies_in, mt5_contrib=True):
    mt = reac.translate_reaction_string_to_mt(reaction)
    return compute_xs(endf_dict, mt, energies_in, mt5_contrib)


def get_particle_production_xs(endf_dict, particle, energies_in):
    zap = physconst.get_zap_for_particle(particle)
    return compute_cumulative_quantity(
        lambda endf_dict, mt, zap, energies_in: (
            compute_yields(
                endf_dict, mt, zap, energies_in, include_discrete=True
            ) * compute_xs(endf_dict, mt, energies_in)
        ),
        lambda endf_dict, mt, zap, energies_in: (
            contains_zap(endf_dict, mt, zap) and
            satisfies_select_heuristic(endf_dict, mt)
        ),
        endf_dict, zap, energies_in
    )


def get_particle_production_dxs_dE(endf_dict, particle, energies_in, energies_out):
    zap = physconst.get_zap_for_particle(particle)
    return compute_cumulative_quantity(
        lambda endf_dict, mt, zap, energies_in, energies_out: (
            compute_dexs(endf_dict, mt, zap, energies_in, energies_out)
        ),
        lambda endf_dict, mt, zap, energies_in, energies_out: (
            contains_zap(endf_dict, mt, zap) and
            satisfies_select_heuristic(endf_dict, mt)
        ),
        endf_dict, zap, energies_in, energies_out
    )


def get_particle_production_dxs_dmu(endf_dict, particle, energies_in, angle_cosines_out):
    zap = physconst.get_zap_for_particle(particle)
    return compute_cumulative_quantity(
        lambda endf_dict, mt, zap, energies_in, angle_cosines_out: (
            compute_daxs(endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True)
        ),
        lambda endf_dict, mt, zap, energies_in, energies_out: (
            contains_zap(endf_dict, mt, zap) and
            satisfies_select_heuristic(endf_dict, mt)
        ),
        endf_dict, zap, energies_in, angle_cosines_out
    )
