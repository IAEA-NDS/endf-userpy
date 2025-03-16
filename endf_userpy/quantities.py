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


def get_reaction_xs(endf_dict, reaction, energies_in):
    proj = prop.get_projectile(endf_dict)
    mt = reac.translate_reaction_string_to_mt(reaction)
    simple_xs = compute_xs(endf_dict, mt, energies_in)

    # determine whether combination with MT5 is possible
    ejectiles = reac.get_ejectiles(proj, mt)
    is_simple_xs = ejectiles is None
    is_simple_xs |= reac.is_discrete_level_scattering(mt)
    if is_simple_xs:
        return simple_xs
    is_simple_xs |= len(ejectiles) > 1
    if is_simple_xs:
        return simple_xs
    ejectile = ejectiles[0][1]
    is_simple_xs |= ejectile not in ('n', 'p')
    if is_simple_xs:
        return simple_xs

    # determine corresponding residual product
    za_projectile = prop.get_ZAI(endf_dict)
    za_target = prop.get_ZA(endf_dict)
    za_ejectile = physconst.get_zap_for_particle(ejectile)
    m = reac.get_multiplicity_for_zap(endf_dict, mt, za_ejectile)
    za_residual = za_target + za_projectile - m * za_ejectile

    # reconstruct cross section component from mf6/mt5
    einc_sel = (simple_xs == 0.0)
    einc_reduced = energies_in[einc_sel]
    yield_mt5 = compute_yields(
        endf_dict, 5, za_residual, einc_reduced, include_discrete=True
    )
    xs_mt5 = compute_xs(endf_dict, 5, einc_reduced)
    xs_add = xs_mt5 * yield_mt5
    simple_xs[einc_sel] += xs_add
    return simple_xs


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
