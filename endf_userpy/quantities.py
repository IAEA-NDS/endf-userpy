import numpy as np
from .primitives import physical_constants as physconst
from .primitives import properties as prop
from .primitives import reactions as reac
from .primitives.helpers import unpack_za
from .quantities_mt_zap import quantities as quant_mt_zap
from .quantities_mt_zap import distribution1d as dist1d
from .quantities_mt_zap import selectors
import logging
# TODO: Remove direct use of mf6_interpretation module in this module
from .mfsec_interpretation import mf6_interpretation as mf6interp
from .mfsec_interpretation import mf6_interpretation_helpers as mf6help


module_logger = logging.getLogger(__name__)


# TODO: check and complete


def get_available_reactions(endf_dict):
    mts = quant_mt_zap.get_reaction_mt_numbers(endf_dict)
    reacs = [
        prop.get_reaction_string_for_mt(endf_dict, mt)
        for mt in mts
    ]
    return reacs


def get_incident_energies(endf_dict, reaction):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    mts = quant_mt_zap.get_reaction_mt_numbers(endf_dict)
    select_mts = [
        mt for mt in mts
        if selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
    ]
    module_logger.debug('selected ' + ','.join(str(mt) for mt in select_mts))
    energy_meshes = [
        quant_mt_zap.get_incident_energies(endf_dict, mt)
        for mt in select_mts
    ]
    return np.unique(np.concatenate(energy_meshes))


def get_emission_energies(endf_dict, reaction, particle, nofail=False):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    mts = quant_mt_zap.get_reaction_mt_numbers(endf_dict)
    select_mts = [
        mt for mt in mts
        if selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        and selectors.contains_zap(endf_dict, mt, zap)
    ]
    module_logger.debug('selected ' + ','.join(str(mt) for mt in select_mts))
    energy_meshes = [
        mf6interp.get_emission_energies(endf_dict, mt, zap, nofail)
        for mt in select_mts
    ]
    return np.unique(np.concatenate(energy_meshes))


def get_reaction_xs(endf_dict, reaction, energies_in, mt5_contrib=True):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    avail_mts = set(quant_mt_zap.get_reaction_mt_numbers(endf_dict))
    iter_mts = avail_mts.copy()
    iter_mts.update(user_mts)
    xs = np.zeros_like(energies_in, dtype=float)
    proj = prop.get_projectile(endf_dict)
    for mt in sorted(iter_mts):
        module_logger.debug(f'consider MT={mt} for reaction xs')
        should_select = selectors.satisfies_select_heuristic(
            endf_dict, mt, user_mts
        )
        mt_available = mt in avail_mts
        if should_select and mt_available:
            module_logger.debug(f'select MT={mt} for reaction xs')
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
                module_logger.debug('include MF6/MT5 component for MT={mt}')
    return xs


def get_residual_production_xs(endf_dict, residual_nucleus, energies_in, mt5_contrib=True):
    module_logger.debug(f'ahahahahah')
    za_residual, level = physconst.get_za_for_residual_nucleus(residual_nucleus)
    if level is not None:
        module_logger.debug(f'user requested isomeric state {level}')
    xs = quant_mt_zap.compute_cumulative_quantity(
        lambda endf_dict, mt: (
            quant_mt_zap.compute_xs(endf_dict, mt, energies_in)
        ),
        lambda endf_dict, mt: (
            selectors.contains_residual_za(endf_dict, mt, za_residual) and
            selectors.satisfies_select_heuristic(endf_dict, mt)
        ),
        endf_dict
    )
    # add mt5 contribution if demanded
    if not mt5_contrib:
        module_logger.debug(f'inclusion of MF6/MT5 disabled by user')
    elif (prop.has_mf6_mt(endf_dict, 5)
            and mf6help.contains_zap(endf_dict, 5, za_residual)):
        module_logger.debug(f'include MF6/MT5 component for residual ZA={za_residual}')
        mt5_totxs = quant_mt_zap.compute_xs(endf_dict, 5, energies_in)
        mt5_yield = quant_mt_zap.compute_yields(
            endf_dict, 5, za_residual, energies_in, level=level
        )
        module_logger.debug(f'mt5_yield: {mt5_yield}')
        mt5_xs = mt5_totxs * mt5_yield
        module_logger.debug(f'mt5_xs: {mt5_xs}')
        assert np.all(xs[mt5_xs != 0.0] == 0.0)
        assert np.all(mt5_xs[xs != 0.0] == 0.0)
        xs += mt5_xs
    else:
        module_logger.debug(f'no MF6/MT5 component available for residual ZA={za_residual}')
    return xs


def get_particle_production_xs(endf_dict, reaction, particle, energies_in):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        quant_mt_zap.compute_prodxs,
        lambda endf_dict, mt, zap, energies_in: (
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
            and selectors.contains_zap(endf_dict, mt, zap)
        ),
        endf_dict, zap, energies_in
    )


def get_particle_production_dxs_dE(
    endf_dict, reaction, particle, energies_in, energies_out
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        quant_mt_zap.compute_dexs,
        lambda endf_dict, mt, zap, energies_in, energies_out: (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        ),
        endf_dict, zap, energies_in, energies_out
    )


def get_particle_production_dxs_dmu(
    endf_dict, reaction, particle, energies_in, angle_cosines_out
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        quant_mt_zap.compute_daxs,
        lambda endf_dict, mt, zap, energies_in, angle_cosines_out: (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        ),
        endf_dict, zap, energies_in, angle_cosines_out
    )


def get_particle_production_ddxs(
    endf_dict, reaction, particle, energies_in, energies_out, angle_cosines_out
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        quant_mt_zap.compute_ddxs,
        lambda endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out: (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        ),
        endf_dict, zap, energies_in, energies_out, angle_cosines_out
    )
