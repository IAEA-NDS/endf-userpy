import numpy as np
from ..primitives import array_ns
from ..primitives.interpolation import endf_interp1d


def find_subsec_nums(endf_dict, mt, zap, level=None):
    sec = endf_dict[9][mt]
    nums = []
    for idx, subsec in sec['subsection'].items():
        if subsec['IZAP'] != zap:
            continue
        if level is not None and subsec['LFS'] != level:
            continue
        nums.append(idx)
    return nums


def get_subsecs(endf_dict, mt, zap, level=None):
    subsec_nums = find_subsec_nums(endf_dict, mt, zap, level)
    subsecs = endf_dict[9][mt]['subsection']
    return tuple(subsecs[idx] for idx in subsec_nums)


def compute_yields(endf_dict, mt, zap, energies_in, level=None, xp=None):
    """MF9 multiplicity of residual ``zap`` (isomeric ``level``) for
    ``mt`` at ``energies_in``.

    ``xp=None`` (default) is numpy. With the jax adapter the query
    energies may be tracers (``jax.jit`` over the energies):
    ``endf_interp1d`` takes its traced-x path.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    subsecs = get_subsecs(endf_dict, mt, zap, level)
    if len(subsecs) == 0:
        levelstr = f', level={level}' if level is not None else ''
        raise IndexError(
            f'No subsection associated with MF=10, MT={mt}, ZAP={zap}{levelstr}'
        )
    if len(subsecs) > 1:
        levelstr = f', level={level}' if level is not None else ''
        raise IndexError(
            f'Multiple subsections associated with MF=10, MT={mt}, ZAP={zap}{levelstr}'
        )
    subsec = subsecs[0]
    intarr = subsec['INT']
    nbtarr = subsec['NBT']
    en_mesh = np.array(subsec['E'])
    yield_mesh = np.array(subsec['Y'])
    en_out = xp.asarray(energies_in)
    yields = endf_interp1d(
        en_out, en_mesh, yield_mesh, intarr, nbtarr, outside_value=0.0, xp=xp,
    )
    return yields
