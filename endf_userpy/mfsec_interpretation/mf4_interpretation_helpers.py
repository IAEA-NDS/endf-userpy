import numpy as np
from ..primitives.helpers import (
    dict2array,
    pad_outside_values,
    no_filter,
)


def _filter_energies_in(
    energies_in, endf_dict, mt, *args, **kwargs
):
    ei_mesh = dict2array(endf_dict[4][mt]['E'], dtype=float)
    eincs = energies_in
    return (eincs >= np.min(ei_mesh)) & (eincs <= np.max(ei_mesh))


def pad_outside_angdist_values(func):
    return pad_outside_values(
        ['energies', 'angle_cosines'],
        [_filter_energies_in, no_filter]
    )(func)
