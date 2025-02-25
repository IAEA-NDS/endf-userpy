import numpy as np
from .properties import (
    is_zap_consistent,
)
from .mf3_interpretation import (
    compute_cross_section,
)
from .mf4_interpretation_fort import (
    compute_angdist_values,
)
from .mf6_interpretation import (
    compute_dist2d_values,
    compute_yields,
)


compute_xs = compute_cross_section


def compute_dxs(endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True):
    if not is_zap_consistent(endf_dict, mt, zap):
        raise ValueError(f'MT={mt} and ZAP={mt} are not consistent')
    angdist = compute_angdist_values(endf_dict, mt, energies_in, angle_cosines_out, to_lab)
    xs = compute_cross_section(endf_dict, mt, energies_in).reshape(-1, 1)
    return angdist * xs


def compute_ddxs(endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab=True):
    f = compute_dist2d_values(
        endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out, to_lab
    )
    yields = compute_yields(endf_dict, mt, zap, energies_in).reshape(-1, 1, 1)
    xs = compute_cross_section(endf_dict, mt, energies_in).reshape(-1, 1, 1)
    return f * yields * xs / (2*np.pi)
