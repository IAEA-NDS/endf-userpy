"""JAX TAB1 interpolation fast path for concrete query meshes.

Companion to :mod:`endf_userpy.primitives.tab1`, in the same spirit as
:mod:`tab1_numba`: the panel lookup (``searchsorted`` over the table
abscissae, the interpolation-region index, ``x1`` / ``x2``) is integer
bookkeeping on concrete data and stays on numpy, whose C
``searchsorted`` is far cheaper than ``jnp.searchsorted`` on CPU (one
full pass over the queries per binary-search step). Only the value
arithmetic runs on JAX, in one ``jax.jit``-compiled kernel
(:func:`interp_from_lookup`) instead of ~35 eager per-op passes over
the query mesh.

Every kernel input has the query shape, so the kernel compiles once
per (mesh size, law set, side), independent of the table size: one
compile serves every MF3 table of a file at a given mesh.

Used by :func:`tab1.interp` on the ``'jax'`` backend when the query
``x`` and the table abscissae are concrete. The ordinates may be
traced: they are gathered with the host-computed indices on JAX, so
``jax.grad`` wrt tabulated values flows through. Traced queries or a
traced mesh keep the fully traceable vectorised path.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import array_ns


def host_lookup(tab_x, tab_nbt, tab_intp, x, side):
    """Panel lookup on numpy: returns ``(i, law)`` with ``i`` the upper
    panel index (``tab_x[i - 1] <= x <= tab_x[i]`` inside the mesh) and
    ``law`` the ENDF law code of that panel (``intp % 10``), both
    shaped like ``x``. Same convention as :func:`tab1.interp`."""
    i = np.clip(np.searchsorted(tab_x, x, side=side), 1, len(tab_x) - 1)
    k = np.searchsorted(tab_nbt, i, side='left')
    return i, (tab_intp[k] % 10).astype(np.int32)


@partial(jax.jit, static_argnames=('laws', 'side'))
def interp_from_lookup(law, x, x1, x2, y1, y2, outside_value, laws, side):
    """TAB1 values from a host lookup: degenerate-panel and
    out-of-range handling plus the law arithmetic of
    :func:`tab1._apply_law_vectorised`, fused. ``outside_value`` is a
    traced scalar so a NaN fill does not defeat the compile cache."""
    from .tab1 import _apply_law_vectorised
    degenerate = (x1 == x2) | (y1 == y2)
    law = jnp.where(degenerate, 1 if side == 'left' else 6, law)
    law = jnp.where((x < x1) | (x > x2), 0, law)
    return _apply_law_vectorised(
        law, x, x1, x2, y1, y2, array_ns.get_backend('jax'),
        outside_value=outside_value, laws=laws,
    )


@partial(jax.jit, static_argnames=('laws', 'mask_outside'))
def traced_x_from_lookup(x, x1, x2, y1, y2, interp_type, is_inside,
                         outside_value, laws, mask_outside):
    """Jitted arithmetic of
    :func:`interpolation._endf_interp1d_traced_x` from a host bracket
    lookup. ``y1`` / ``y2`` may be batched ``(..., N)``; every other
    input has the query shape ``(N,)``. ``mask_outside=False`` keeps
    the ``outside_value=None`` semantics (no off-mesh masking)."""
    from .interpolation import _traced_x_arith
    return _traced_x_arith(
        x, x1, x2, y1, y2, interp_type, is_inside,
        outside_value if mask_outside else None, laws,
        array_ns.get_backend('jax'),
    )
