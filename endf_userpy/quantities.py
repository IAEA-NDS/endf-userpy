import numpy as np
from .primitives import physical_constants as physconst
from .primitives import properties as prop
from .primitives import reactions as reac
from .quantities_mt_zap import quantities as quant_mt_zap
from .quantities_mt_zap import distribution1d as dist1d
from .quantities_mt_zap import selectors


# TODO: check and complete


def get_available_reactions(endf_dict):
    mts = quant_mt_zap.get_reaction_mt_numbers(endf_dict)
    reacs = [
        prop.get_reaction_string_for_mt(endf_dict, mt)
        for mt in mts
    ]
    return reacs


def get_reaction_xs(endf_dict, reaction, energies_in, mt5_contrib=True):
    mt = reac.translate_reaction_string_to_mt(reaction)
    return quant_mt_zap.compute_xs(endf_dict, mt, energies_in, mt5_contrib)


def get_particle_production_xs(endf_dict, particle, energies_in):
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        lambda endf_dict, mt, zap, energies_in: (
            quant_mt_zap.compute_yields(
                endf_dict, mt, zap, energies_in, include_discrete=True
            ) * quant_mt_zap.compute_xs(endf_dict, mt, energies_in)
        ),
        lambda endf_dict, mt, zap, energies_in: (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt)
        ),
        endf_dict, zap, energies_in
    )


def get_particle_production_dxs_dE(endf_dict, particle, energies_in, energies_out):
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        lambda endf_dict, mt, zap, energies_in, energies_out: (
            quant_mt_zap.compute_dexs(endf_dict, mt, zap, energies_in, energies_out)
        ),
        lambda endf_dict, mt, zap, energies_in, energies_out: (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt)
        ),
        endf_dict, zap, energies_in, energies_out
    )


def get_particle_production_dxs_dmu(endf_dict, particle, energies_in, angle_cosines_out):
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        lambda endf_dict, mt, zap, energies_in, angle_cosines_out: (
            quant_mt_zap.compute_daxs(endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True)
        ),
        lambda endf_dict, mt, zap, energies_in, energies_out: (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt)
        ),
        endf_dict, zap, energies_in, angle_cosines_out
    )
