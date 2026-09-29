"""Tracer-safe / jit-friendly MF6 LAW=1 discrete-line kernel
(issue #290 Phase 2 sub-task 5).

Analogue of :func:`mf6_interpretation_subsecs.get_law1_discrete_lines_from_subsec`
that operates on the :class:`~mf6_law1_preproc.MF6Law1Data`
dataclass produced by the shared preproc. Vectorises the Ein and mu
axes and broadcasts arithmetic through the ``xp`` adapter so
``@jax.jit`` and ``jax.grad`` can flow ``energies_in`` and
``angle_cosines_out`` through as symbolic axes.

MVP scope
---------
The traced kernel implements **LANG=1 with NA=0** on both panels of
the bracket (isotropic Legendre, single ``b`` coefficient) -- the
common gamma-production case, which is what U-233 (n,g) uses. For
this specialisation the panel-pair amplitude at discrete slot ``k``
reduces to

    f_p(e_p, ep_p_k, w) = 0.5 * b_disc_ded[p, k, 0]

(no ``mu`` dependence, no Kalbach-Mann). Outer E-axis interpolation
between the two panels reuses the shared :func:`_yintp_bc`.

For other ``(LANG, NA)`` combinations the dispatcher in
:func:`compute_law1_discrete_lines` continues to use the numpy
reference; that non-traced path stays correct but fails under jit
until Phase 2b.

Padding invariant relied upon
-----------------------------
``data.ep_disc_ded[p, k]`` and ``data.b_disc_ded[p, k, ...]`` for
``k >= data.nd_ded_arr[p]`` are zero (from the preproc pad). The
kernel iterates ``k`` in ``range(max_nd_ded)`` at Python-side and
masks out padded slots via ``xp.where(k < nd_ded_at_panel, ...,
0.0)``.
"""
from __future__ import annotations

import numpy as np

from . import mf6_law1_kernel as _kernel
from . import mf6_law1_helpers as _helpers


def _require_supported(data):
    """Enforce the MVP restriction: LANG=1 uniform NA=0."""
    lang = int(data.lang)
    na = np.asarray(data.na_arr)
    if lang != 1 or not np.all(na == 0):
        raise NotImplementedError(
            'get_law1_discrete_lines_from_subsec_traced covers only '
            'LANG=1 NA=0 (issue #290 Phase 2 MVP); got '
            f'LANG={lang}, NA={list(na[:5])}...'
        )


def get_law1_discrete_lines_from_subsec_traced(
    data, energies_in, angle_cosines_out, to_lab, xp,
):
    """Return ``(ep_disc_lab, amp_disc)`` for one MF6 LAW=1
    subsection with ``xp``-native arithmetic.

    Parameters mirror the numpy reference in
    :func:`mf6_interpretation_subsecs.get_law1_discrete_lines_from_subsec`:

    - ``data``: :class:`~mf6_law1_preproc.MF6Law1Data` built via
      ``mf6_law1_data_from_endf_dict(endf_dict, mt, subsec_num, xp)``.
    - ``energies_in``: xp array of incident energies (shape ``(n_ein,)``).
    - ``angle_cosines_out``: xp array of LAB cosines (shape ``(n_mus,)``).
    - ``to_lab``: must be ``True``; the frame choice is fixed by
      ``data.lct``.
    - ``xp``: backend adapter (``array_ns.get_backend('numpy'|'jax')``).

    Returns ``(ep_disc_lab, amp_disc)`` both of shape
    ``(n_ein, n_mus, max_nd_ded)``. Cells where the CM->LAB inverse
    has no physical solution (``dinv=0`` from
    :func:`mf6_law1_helpers.mf6cm2lab_disc_bc`) or where the Ein
    falls outside the section's ``ei_mesh`` range are zero.
    """
    if to_lab is not True:
        raise ValueError(
            'get_law1_discrete_lines_from_subsec_traced requires to_lab=True'
        )
    _require_supported(data)

    e_in = xp.asarray(energies_in)
    u_in = xp.asarray(angle_cosines_out)
    ei_mesh = xp.asarray(data.ei_mesh)
    ep_disc_ded = xp.asarray(data.ep_disc_ded)
    b_disc_ded = data.b_disc_ded   # xp-native from preproc
    nd_ded_arr = np.asarray(data.nd_ded_arr)   # numpy int per panel

    n_panels = ei_mesh.shape[0]
    n_pairs = n_panels - 1
    max_nd_ded = int(ep_disc_ded.shape[1])
    n_ein = int(e_in.shape[0])
    n_mus = int(u_in.shape[0])

    if max_nd_ded == 0:
        # Section has no discrete lines; the caller expects a
        # zero-slot last axis and downstream broadening treats
        # that as "no contribution".
        zeros = xp.zeros((n_ein, n_mus, 0), dtype=e_in.dtype)
        return zeros, zeros

    # convert_interp_repr on integer descriptors is Python-side (numpy).
    from ..primitives.helpers import convert_interp_repr
    ei_interp_full = convert_interp_repr(
        np.asarray(data.int_arr), np.asarray(data.nbt_arr),
    )
    # Uniform lei across pairs, following the multipanel-traced
    # kernel's precondition.
    unique_lei = np.unique(np.asarray(ei_interp_full))
    if unique_lei.size == 1:
        static_lei = int(unique_lei[0])
    else:
        raise NotImplementedError(
            'get_law1_discrete_lines_from_subsec_traced requires '
            'uniform lei across panel pairs.'
        )
    outer_law = static_lei % 10

    awr = float(data.awr)
    awi = float(data.awi)
    awp = float(data.awp)
    lct = int(data.lct) if to_lab else 1

    # Uniform nd_ded across bracketed panels: preproc padding
    # guarantees k < nd_ded_arr[p] entries are physical. Broadcast
    # the per-panel valid-slot count to an (n_ein,) tensor via take.
    nd_ded_arr_xp = xp.asarray(nd_ded_arr)

    # Panel index per Ein; clip so the +1 for the outer bracket is
    # safe on either edge.
    idx_raw = xp.searchsorted(ei_mesh, e_in, side='right') - 1
    idx = xp.clip(idx_raw, 0, n_pairs - 1)
    idx_next = idx + 1

    # Per-Ein panel scalars.
    e1_ein = xp.take(ei_mesh, idx, axis=0)                # (n_ein,)
    e2_ein = xp.take(ei_mesh, idx_next, axis=0)           # (n_ein,)
    nd_ded_min = xp.minimum(
        xp.take(nd_ded_arr_xp, idx, axis=0),
        xp.take(nd_ded_arr_xp, idx_next, axis=0),
    )                                                     # (n_ein,)

    # Per-Ein per-k panel data. ``xp.take`` on the panel axis with a
    # tracer index; ``k`` axis stays static.
    ep1_all = xp.take(ep_disc_ded, idx, axis=0)           # (n_ein, max_nd_ded)
    ep2_all = xp.take(ep_disc_ded, idx_next, axis=0)      # (n_ein, max_nd_ded)
    b1_all = xp.take(b_disc_ded, idx, axis=0)             # (n_ein, max_nd_ded, max_na+1)
    b2_all = xp.take(b_disc_ded, idx_next, axis=0)        # (n_ein, max_nd_ded, max_na+1)

    # In-range mask along the Ein axis.
    e_min = ei_mesh[0]
    e_max = ei_mesh[-1]
    ein_in_range = (e_in >= e_min) & (e_in <= e_max)      # (n_ein,)

    # k-axis validity mask: (n_ein, max_nd_ded), True where slot is
    # physical (k < nd_ded_min[ein]).
    k_range = xp.arange(max_nd_ded)                       # (max_nd_ded,)
    k_valid = k_range[None, :] < nd_ded_min[:, None]      # (n_ein, max_nd_ded)

    # Two-point interpolation of tp (the eval-frame discrete energy)
    # between panels: for genuine level-decay lines ep1_k == ep2_k so
    # tp = ep1_k regardless of e; kept general because the preproc
    # ordering may pair panel-1 slot k with panel-2 slot k under
    # dedup and the two values may differ if the discrete-line
    # configuration is different at the two panel knots.
    e_bc = e_in[:, None]                                  # (n_ein, 1)
    e1_bc = e1_ein[:, None]
    e2_bc = e2_ein[:, None]
    tp_at_e = _kernel._yintp_bc(
        outer_law, e1_bc, ep1_all, e2_bc, ep2_all, e_bc, xp,
    )                                                     # (n_ein, max_nd_ded)

    # Amplitude per panel at slot k (LANG=1, NA=0): 0.5 * b[..., 0].
    f1 = 0.5 * b1_all[..., 0]                              # (n_ein, max_nd_ded)
    f2 = 0.5 * b2_all[..., 0]
    amp_at_e = _kernel._yintp_bc(
        outer_law, e1_bc, f1, e2_bc, f2, e_bc, xp,
    )                                                     # (n_ein, max_nd_ded)

    # LAB<->CM inverse map broadcast over (n_ein, n_mus, max_nd_ded).
    e_bcast = e_in[:, None, None]                          # (n_ein, 1, 1)
    tp_bcast = tp_at_e[:, None, :]                         # (n_ein, 1, max_nd_ded)
    u_bcast = u_in[None, :, None]                          # (1, n_mus, 1)
    ep_lab, _w, dinv = _helpers.mf6cm2lab_disc_bc(
        awr, awi, awp, lct, e_bcast, tp_bcast, u_bcast, xp,
    )                                                     # each (n_ein, n_mus, max_nd_ded)

    amp_bcast = amp_at_e[:, None, :]                       # (n_ein, 1, max_nd_ded)
    amp_final = amp_bcast * dinv

    # Compose the k-slot mask with the Ein-in-range mask so padded
    # slots and out-of-range Eins are zero. dinv=0 already zeroes
    # below-threshold cells inside mf6cm2lab_disc_bc.
    ein_mask = ein_in_range[:, None, None]
    k_mask = k_valid[:, None, :]
    combined = ein_mask & k_mask                           # (n_ein, 1, max_nd_ded)
    zero = xp.zeros_like(ep_lab)
    ep_disc_lab = xp.where(combined, ep_lab, zero)
    amp_disc = xp.where(combined, amp_final, zero)
    return ep_disc_lab, amp_disc
