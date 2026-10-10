import numpy as np
from ..primitives import array_ns
from ..primitives.interpolation import endf_interp1d
from .mf3_interpretation import _handle_above_range


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
    endf_dict, mt, zap, energies_in, level=None,
    above_range='warn_nan', _query_state=None, xp=None,
):
    """Residual-production cross section from MF10.

    ``above_range`` (default ``'warn_nan'``) follows the same
    convention as :func:`mf3_interpretation.compute_cross_section`.
    The top-level ``endf_userpy.quantities`` entry points thread
    the effective policy from ``options.above_range`` explicitly
    (issue #143); direct callers of this leaf get the
    ``'warn_nan'`` default.

    ``xp=None`` (default) is numpy; with the jax adapter the query
    energies may be tracers (``jax.jit`` over the energies), in which
    case the above-range fill is applied without the host-side
    warning / raise.

    ``_query_state`` is the private :class:`_QueryState` accumulator
    the top-level entry points thread through internal callers so
    ONE summary UserWarning fires per top-level query instead of
    one per (MT, call). ``_query_state=None`` (the leaf's default)
    falls back to a per-call warning for direct leaf usage.
    """
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
    if xp is None:
        xp = array_ns.get_backend('numpy')
    e_max = float(np.asarray(en_mesh, dtype=float).max())
    try:
        en_out = np.asarray(energies_in, dtype=float)
    except Exception:          # jax tracer (jax.jit over the energies)
        en_out = None
    if en_out is None:
        # Traced energies: no host-side warning / raise is possible;
        # apply the policy's fill on the traced mask, as
        # ``mf3_interpretation.compute_cross_section`` does.
        e_tr = xp.asarray(energies_in, dtype=xp.float64)
        xs = endf_interp1d(
            e_tr, en_mesh, xs_mesh, intarr, nbtarr, outside_value=0.0, xp=xp,
        )
        if above_range in ('warn_nan', 'nan'):
            xs = xp.where(e_tr > e_max, float('nan'), xs)
        return xs
    above_mask = en_out > e_max
    fill_value = _handle_above_range(
        above_range, mt, e_max, above_mask, en_out, query_state=_query_state,
    )
    xs = endf_interp1d(
        en_out, en_mesh, xs_mesh, intarr, nbtarr, outside_value=0.0, xp=xp,
    )
    if above_mask.any() and fill_value != 0.0:
        xs = xp.where(xp.asarray(above_mask), fill_value, xs)
    return xs
