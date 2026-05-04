import numpy as np
from scipy.integrate import quad
from ..primitives import properties
from ..primitives import reactions as reaction
from ..primitives.physical_constants import get_zap_for_particle
from ..mfsec_interpretation import mf1_interpretation as mf1_interp
from ..mfsec_interpretation import mf3_interpretation as mf3_interp
from ..mfsec_interpretation import mf6_interpretation as mf6_interp
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from ..mfsec_interpretation import mf8_interpretation as mf8_interp
from ..mfsec_interpretation import mf9_interpretation as mf9_interp
from ..mfsec_interpretation import mf10_interpretation as mf10_interp
from .distribution1d import (
    compute_angdist_values,
    compute_energydist_values,
)
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from .distribution2d import compute_dist2d_values
import logging


module_logger = logging.getLogger(__name__)


# functions borrowed as is
from ..mfsec_interpretation.mf3_interpretation import (
    get_incident_energy_range,
    get_incident_energies,
    get_reaction_mts as get_reaction_mt_numbers,
)


def compute_yields(endf_dict, mt, zap, energies_in, include_discrete=True, level=None):
    module_logger.debug(f'compute yields for MT={mt} and ZAP={zap} and level={level}')
    if mt == 18:
        neutron_zap = get_zap_for_particle('n')
        if zap != neutron_zap:
            raise ValueError(
                f'For fission, only yield of emitted neutrons can be computed '
                f'zap={neutron_zap} but obtained zap={zap}'
            )
        if level is not None:
            raise ValueError(
                f'For fission, `level` argument must be `None`'
            )
        # if MT=18 (n,f), we assume user wants to know prompt neutron yields
        module_logger.debug(f'--> getting yields for MT={mt} and ZAP={zap} from MF1/MT456')
        yields = mf1_interp.compute_yields(endf_dict, 456, energies_in)
    elif properties.has_mf6_mt(endf_dict, mt):
        module_logger.debug(f'--> getting yields for MT={mt} and ZAP={zap} from MF6/MT{mt}')
        yields = mf6_interp.compute_yields(
            endf_dict, mt, zap, energies_in, include_discrete, level
        )
    else:
        if level is not None:
            raise ValueError(
                f'Yield directly derived from MT number (MT={mt}, '
                '`level` argument must be `None`'
            )
        proj = properties.get_projectile(endf_dict)
        mult = reaction.get_multiplicity_for_zap(proj, mt, zap)
        module_logger.debug(
            f'--> getting yields for MT={mt} and ZAP={zap} from reaction string (yield={mult})'
        )
        yields = np.full(len(energies_in), mult, dtype=float)
    return yields


def compute_xs_mt5_contrib(endf_dict, mt, energies_in):
    zero_xs_result = np.zeros_like(energies_in, dtype=float)
    if not properties.has_mf6_mt(endf_dict, 5):
        return zero_xs_result

    proj = properties.get_projectile(endf_dict)
    if not reaction.is_unique_path_to_residual(proj, mt):
        return zero_xs_result
    ejectile = reaction.get_unique_ejectile(proj, mt)

    mt5 = 5
    za_projectile = properties.get_ZAI(endf_dict)
    za_target = properties.get_ZA(endf_dict)
    za_ejectile = get_zap_for_particle(ejectile)
    m = reaction.get_multiplicity_for_zap(proj, mt, za_ejectile)
    if m is None:
        return zero_xs_result
    za_residual = za_target + za_projectile - m * za_ejectile

    if not mf6_help.has_subsecs_for_mt_zap(endf_dict, mt5, za_residual):
        return zero_xs_result
    yield_mt5 = compute_yields(
        endf_dict, mt5, za_residual, energies_in, include_discrete=True
    )
    xs_mt5 = mf3_interp.compute_cross_section(endf_dict, mt5, energies_in)
    return xs_mt5 * yield_mt5


def compute_xs(endf_dict, mt, energies_in):
    return mf3_interp.compute_cross_section(endf_dict, mt, energies_in)


def compute_prodxs(endf_dict, mt, zap, energies_in):
    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=True
    )
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
    return xs * yields


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
    module_logger.debug(f'compute dexs for MT={mt} and ZAP={zap}')
    yields = compute_yields(
        endf_dict, mt, zap, energies_in, include_discrete=True
    ).reshape(-1, 1)
    xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in).reshape(-1, 1)
    energydist = compute_energydist_values(
        endf_dict, mt, zap, energies_in, energies_out, to_lab
    )
    module_logger.debug(f'average yield is {np.mean(yields)} for MT={mt} and ZAP={zap}')
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

        module_logger.debug(f'consider MT={mt} for inclusion in cumulative quantity')
        if select is not None:
            if not select(endf_dict, mt, *args, **kwargs):
                continue
        module_logger.debug(f'select MT={mt} for inclusion in cumulative quantity')
        cur_res = func(endf_dict, mt, *args, **kwargs)
        if is_first:
            cum_res = cur_res
            is_first = False
        else:
            cum_res += cur_res

    return cum_res


def _compute_residual_xs_for_lfs(endf_dict, mt, za_residual, lfs, energies_in):
    lmf = mf8_interp.get_mf_switch(endf_dict, mt, za_residual, lfs)
    if lmf == 3:
        return mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
    if lmf == 6:
        xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
        y = mf6_interp.compute_yields(
            endf_dict, mt, za_residual, energies_in,
            include_discrete=True, level=lfs,
        )
        return xs * y
    if lmf == 9:
        xs = mf3_interp.compute_cross_section(endf_dict, mt, energies_in)
        y = mf9_interp.compute_yields(
            endf_dict, mt, za_residual, energies_in, level=lfs
        )
        return xs * y
    if lmf == 10:
        return mf10_interp.compute_cross_section(
            endf_dict, mt, za_residual, energies_in, level=lfs
        )
    raise ValueError(
        f'unsupported LMF={lmf} in MF8/MT={mt} for ZAP={za_residual}, LFS={lfs}'
    )


def compute_residual_xs(endf_dict, mt, za_residual, lfs, energies_in):
    """Cross section for producing (za_residual, lfs) via reaction MT.

    Dispatches via MF8 LMF: 3 (MF3 only, no isomer split), 6
    (MF3/MT * MF6/MT yield resolved by LFS), 9 (MF3/MT * MF9/MT
    yield), 10 (MF10/MT directly). When MF8/MT is absent, falls back
    to MF3/MT (and refuses to resolve a non-zero LFS).

    If `lfs` is None, sums contributions from all LFS values present
    in MF8/MT for this ZAP.
    """
    if not (8 in endf_dict and mt in endf_dict[8]):
        if lfs not in (None, 0):
            raise ValueError(
                f'no MF8 information for MT={mt}, cannot resolve isomer level '
                f'(requested LFS={lfs})'
            )
        return mf3_interp.compute_cross_section(endf_dict, mt, energies_in)

    available_lfs = sorted({
        sub['LFS'] for sub in endf_dict[8][mt]['subsection'].values()
        if sub['ZAP'] == za_residual
    })
    if not available_lfs:
        return np.zeros_like(energies_in, dtype=float)

    if lfs is not None:
        if lfs not in available_lfs:
            return np.zeros_like(energies_in, dtype=float)
        return _compute_residual_xs_for_lfs(
            endf_dict, mt, za_residual, lfs, energies_in
        )

    total = np.zeros_like(energies_in, dtype=float)
    for cur_lfs in available_lfs:
        total = total + _compute_residual_xs_for_lfs(
            endf_dict, mt, za_residual, cur_lfs, energies_in
        )
    return total
