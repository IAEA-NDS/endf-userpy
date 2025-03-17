import numpy as np
from scipy.integrate import quad
from ..primitives import properties
from ..primitives import reactions as reaction
from ..primitives.physical_constants import get_zap_for_particle
from ..primitives.properties import (
    is_zap_consistent,
    has_mf6_mt,
)
from ..mfsec_interpretation import mf1_interpretation as mf1_interp
from ..mfsec_interpretation import mf3_interpretation as mf3_interp
from ..mfsec_interpretation import mf6_interpretation as mf6_interp
from .distribution1d import (
    compute_angdist_values,
    compute_energydist_values,
)
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from .distribution2d import compute_dist2d_values

# functions borrowed as is
from ..mfsec_interpretation.mf3_interpretation import (
    get_incident_energy_range,
    get_incident_energies,
    get_reaction_mts as get_reaction_mt_numbers,
)


def compute_yields(endf_dict, mt, zap, energies_in, include_discrete=True):
    if mt == 18:
        neutron_zap = get_zap_for_particle('n')
        if zap != neutron_zap:
            raise ValueError(
                f'For fission, only yield of emitted neutrons can be computed '
                f'zap={neutron_zap} but obtained zap={zap}'
            )
        # if MT=18 (n,f), we assume user wants to know prompt neutron yields
        yields = mf1_interp.compute_yields(endf_dict, 456, energies_in)
    elif has_mf6_mt(endf_dict, mt):
        yields = mf6_interp.compute_yields(
            endf_dict, mt, zap, energies_in, include_discrete
        )
    else:
        proj = properties.get_projectile(endf_dict)
        mult = reaction.get_multiplicity_for_zap(proj, mt, zap)
        yields = np.full(len(energies_in), mult, dtype=float)
    return yields


def compute_xs_mt5_contrib(endf_dict, mt, energies_in):
    if not has_mf6_mt(endf_dict, 5):
        return np.zeros_like(energies_in, dtype=float)
    proj = properties.get_projectile(endf_dict)
    ejectiles = reaction.get_ejectiles(proj, mt)
    if ejectiles is None or len(ejectiles) > 1:
        return np.zeros_like(energies_in, dtype=float)
    ejectile = ejectiles[0][1]
    if ejectile not in ('n', 'p'):
        return np.zeros_like(energies_in, dtype=float)

    za_projectile = properties.get_ZAI(endf_dict)
    za_target = properties.get_ZA(endf_dict)
    za_ejectile = get_zap_for_particle(ejectile)
    m = reaction.get_multiplicity_for_zap(endf_dict, mt, za_ejectile)
    za_residual = za_target + za_projectile - m * za_ejectile
    if not mf6_help.has_subsecs_for_mt_zap(endf_dict, mt, za_residual):
        return np.zeros_like(energies_in, dtype=float)

    yield_mt5 = compute_yields(
        endf_dict, 5, za_residual, energies_in, include_discrete=True
    )
    xs_mt5 = mf3_interp.compute_cross_section(endf_dict, 5, energies_in)
    return xs_mt5 * yield_mt5


def compute_xs(endf_dict, mt, energies_in, mt5_contrib=False):
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
    if mt5_contrib:
        xs_mt5 = compute_xs_mt5_contrib(endf_dict, mt, energies_in)
        xs += xs_mt5
    return xs


def compute_daxs(endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True):
    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=True
    ).reshape(-1, 1)
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in).reshape(-1, 1)
    angdist = compute_angdist_values(
        endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab
    )
    return angdist * yields * xs / (2*np.pi)


def compute_dexs(endf_dict, mt, zap, energies_in, energies_out, to_lab=True):
    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=True
    ).reshape(-1, 1)
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in).reshape(-1, 1)
    energydist = compute_energydist_values(
        endf_dict, mt, zap, energies_in, energies_out, to_lab
    )
    return energydist * yields * xs


def compute_ddxs(endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab=True):
    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=False
    ).reshape(-1, 1, 1)
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in).reshape(-1, 1, 1)
    f = compute_dist2d_values(
        endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab
    )
    return f * yields * xs / (2*np.pi)


def compute_cumulative_quantity(func, select, endf_dict, *args, **kwargs):
    mt_list = get_reaction_mt_numbers(endf_dict)
    is_first = True
    cum_res = None
    for mt in mt_list:

        if select is not None:
            if not select(endf_dict, mt, *args, **kwargs):
                continue

        cur_res = func(endf_dict, mt, *args, **kwargs)
        if is_first:
            cum_res = cur_res
            is_first = False
        else:
            cum_res += cur_res

    return cum_res
