import numpy as np
from ..primitives.interpolation import endf_interp1d
from .mf3_interpretation import _handle_above_range, _above_range_policy


def find_subsec_nums(endf_dict, mt, zap, level=None):
    sec = endf_dict[10][mt]
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
    subsecs = endf_dict[10][mt]['subsection']
    return tuple(subsecs[idx] for idx in subsec_nums)


def compute_cross_section(
    endf_dict, mt, zap, energies_in, level=None, above_range=None,
):
    """Residual-production cross section from MF10.

    ``above_range=None`` (default) reads the active policy from the
    private ``_above_range_policy`` contextvar populated by
    :func:`endf_userpy.mfsec_interpretation.mf3_interpretation._warning_summary_ctx`
    for the duration of a top-level ``endf_userpy.quantities`` call;
    falls back to ``'warn_nan'`` when the leaf is called outside
    that context.
    """
    if above_range is None:
        above_range = _above_range_policy.get()
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
    en_out = np.asarray(energies_in, dtype=float)
    e_max = float(np.asarray(en_mesh, dtype=float).max())
    above_mask = en_out > e_max
    fill_value = _handle_above_range(
        above_range, mt, e_max, above_mask, en_out,
    )
    xs = endf_interp1d(en_out, en_mesh, xs_mesh, intarr, nbtarr, outside_value=0.0)
    if above_mask.any() and fill_value != 0.0:
        xs = np.where(above_mask, fill_value, xs)
    return xs
