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
    try:
        ei_mesh = dict2array(orig_en_dict, dtype=float)
    except Exception:
        # Under mesh-knot autodiff a JAX tracer sits at some
        # ``ei_mesh`` entry; ``dict2array`` without ``xp`` then
        # raises ``TracerArrayConversionError``. The pad_outside
        # fast-path needs a concrete boolean mask, so assume all
        # query energies are inside the mesh and let the traced
        # reconstruction kernels handle out-of-range via their
        # own ``outside_value`` fill.
        return np.ones(np.asarray(eincs).shape, dtype=bool)
    return (eincs >= np.min(ei_mesh)) & (eincs <= np.max(ei_mesh))


def pad_outside_angdist_values(func):
    return pad_outside_values(
        ['energies', 'angle_cosines'],
        [_filter_energies_in, no_filter]
    )(func)
