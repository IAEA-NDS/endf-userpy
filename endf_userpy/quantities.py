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
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    avail_mts = set(quant_mt_zap.get_reaction_mt_numbers(endf_dict))
    iter_mts = avail_mts.copy()
    iter_mts.update(user_mts)
    xs = np.zeros_like(energies_in, dtype=float)
    proj = prop.get_projectile(endf_dict)
    for mt in sorted(iter_mts):
        print(f'considering MT: {mt}')
        should_select = selectors.satisfies_select_heuristic(
            endf_dict, mt, user_mts
        )
        mt_available = mt in avail_mts
        if should_select and mt_available:
            print(f'selecting MT: {mt}')
            cur_xs = quant_mt_zap.compute_xs(endf_dict, mt, energies_in)
            xs += cur_xs

        # add associated MT5 component if available and permissible
        if (mt in user_mts
                and mt5_contrib
                and 5 not in user_mts
                and not reac.any_ancestor_in_mts(5, user_mts)
                and reac.is_unique_path_to_residual(proj, mt)):
            if not mt_available or not should_select:
                cur_xs = np.zeros_like(energies_in, dtype=float)
            eincs_sel = (~np.bool(mt_available)) | (cur_xs == 0.0)
            mt5_xs = quant_mt_zap.compute_xs_mt5_contrib(
                endf_dict, mt, energies_in[eincs_sel]
            )
            xs[eincs_sel] += mt5_xs
            if np.any(mt5_xs != 0.0):
                print('including MT5 component')
    return xs


def get_particle_production_xs(endf_dict, reaction, particle, energies_in):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        quant_mt_zap.compute_prodxs,
        lambda endf_dict, mt, zap, energies_in: (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
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
