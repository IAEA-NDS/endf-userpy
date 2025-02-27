import numpy as np
from . import properties
from . import reactions as reaction
from .physical_constants import get_zap_for_particle
from .properties import (
    is_zap_consistent,
    has_mf4_mt,
    has_mf5_mt,
    has_mf6_mt,
)
from . import mf1_interpretation as mf1_interp
from . import mf3_interpretation as mf3_interp
from . import mf4_interpretation_fort as mf4_interp
from . import mf6_interpretation as mf6_interp
from .distribution2d import compute_dist2d_values


compute_xs = mf3_interp.compute_cross_section


def compute_daxs(endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True):
    if not is_zap_consistent(endf_dict, mt, zap):
        raise ValueError(f'MT={mt} and ZAP={mt} are not consistent')
    angdist = mf4_interp.compute_angdist_values(
        endf_dict, mt, energies_in, angle_cosines_out, to_lab
    )
    xs = compute_xs(endf_dict, mt, energies_in).reshape(-1, 1)
    return angdist * xs


def compute_yields(endf_dict, mt, zap, energies_in):
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
        yields = mf6_interp.compute_yields(endf_dict, mt, zap, energies_in)
    else:
        proj = properties.get_projectile(endf_dict)
        mult = reaction.get_multiplicity_for_zap(proj, mt, zap)
        yields = np.full(len(energies_in), mult, dtype=float)
    return yields


def compute_ddxs(endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab=True):
    f = compute_dist2d_values(
        endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab
    )
    yields = compute_yields(endf_dict, mt, zap, energies_in).reshape(-1, 1, 1)
    xs = compute_xs(endf_dict, mt, energies_in).reshape(-1, 1, 1)
    return f * yields * xs / (2*np.pi)
