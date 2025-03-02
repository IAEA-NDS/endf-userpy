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
from ..mfsec_interpretation import mf6_interpretation as mf6_interp
from .distribution1d import (
    compute_angdist_values,
    compute_energydist_values,
)
from .distribution2d import compute_dist2d_values

# functions borrowed as is
from ..mfsec_interpretation.mf3_interpretation import (
    compute_cross_section as compute_xs,
    get_incident_energy_range,
    get_incident_energies,
    get_reaction_mts as get_reaction_mt_numbers,
)


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


def compute_daxs(endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True):
    yields = compute_yields(endf_dict, mt, zap, energies_in).reshape(-1, 1)
    xs = compute_xs(endf_dict, mt, energies_in).reshape(-1, 1)
    angdist = compute_angdist_values(
        endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab
    )
    return angdist * yields * xs / (2*np.pi)


def compute_dexs(endf_dict, mt, zap, energies_in, energies_out, to_lab=True):
    yields = compute_yields(endf_dict, mt, zap, energies_in).reshape(-1, 1)
    xs = compute_xs(endf_dict, mt, energies_in).reshape(-1, 1)
    energydist = compute_energydist_values(
        endf_dict, mt, zap, energies_in, energies_out, to_lab
    )
    return energydist * yields * xs / (2*np.pi)


def compute_ddxs(endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab=True):
    yields = compute_yields(endf_dict, mt, zap, energies_in).reshape(-1, 1, 1)
    xs = compute_xs(endf_dict, mt, energies_in).reshape(-1, 1, 1)
    f = compute_dist2d_values(
        endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab
    )
    return f * yields * xs / (2*np.pi)
