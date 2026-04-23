import numpy as np
from ..primitives.helpers import (
    dict2array,
    pad_outside_values,
    no_filter,
)


def _filter_energies_in(
    energies_in, endf_dict, mt, *args, **kwargs
):
    eincs = energies_in
    orig_en_dict = endf_dict[4][mt].get('E')
    if orig_en_dict is None:
        # Purely isotropic angular distributions (LTT=0, LI=1)
        # don't come with an incident energy mesh
        return (eincs > 0)
    ei_mesh = dict2array(endf_dict[4][mt]['E'], dtype=float)
    return (eincs >= np.min(ei_mesh)) & (eincs <= np.max(ei_mesh))


def pad_outside_angdist_values(func):
    return pad_outside_values(
        ['energies', 'angle_cosines'],
        [_filter_energies_in, no_filter]
    )(func)
