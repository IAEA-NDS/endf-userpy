import numpy as np
from ..primitives.interpolation import endf_interp1d


def find_subsec_nums(endf_dict, mt, zap, level=None):
    sec = endf_dict[10][mt]
    idcs = tuple()
    nums = []
    final_states = []
    for idx, subsec in sec['subsection'].items():
        if subsec['IZAP'] != zap:
            continue
        if level is not None and subsec['LFS'] != level:
            continue
        nums.append(idx)
    return nums


def get_subsecs(endf_dict, mt, zap, level=None):
    subsec_nums = find_subsec_nums(endf_dict, mt, zap, level)
    subsecs = endf_dict[10][mt]['subsection']
    return tuple(subsecs[idx] for idx in subsec_nums)


def compute_cross_section(endf_dict, mt, zap, energies_in, level=None):
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
    xs_mesh = np.array(subsec['sigma'])
    en_out = np.array(energies_in)
    xs = endf_interp1d(en_out, en_mesh, xs_mesh, intarr, nbtarr, outside_value=0.0)
    return xs
