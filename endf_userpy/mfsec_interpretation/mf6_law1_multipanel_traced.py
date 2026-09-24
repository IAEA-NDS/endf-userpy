"""MF6 LAW=1 tracer-panel-index kernels (issue #166).

Provides tracer-panel-index versions of ``_law1_spectrum_panel_pair``
(from ``mf6_law1_epintegral``) and of the DDX per-panel-pair
amplitude (from ``mf6_law1_kernel``). These are single-graph
routines suitable for ``jax.grad`` wrt incident energy ``E`` across
panels: the caller does not need to compute ``panel_idx`` ahead of
time; the kernel gathers panel data through ``xp.take`` on tracer
indices instead.

Requires the ``MF6Law1Data`` produced under ``xp=jax`` (or numpy
with the same padding invariant), where per-panel scalar arrays
(``ei_mesh``, ``nep_arr``, ``nd_arr``, ``na_arr``) and ``ep_panels``
are xp-native. The ``mf6_law1_preproc.py`` change in this PR
guarantees this under ``xp=jax``.

Padding invariant relied upon
-----------------------------
``ep_panels[p, k]`` for ``k >= nep_arr[p]`` replicates the last
valid Ep value (``ep_panels[p, nep-1]``). Kinks at those slots
collapse to the panel's continuum upper endpoint and produce
zero-width subpanels under sort. The single-panel numpy kernel
never dereferences those slots (bounded by ``nep_arr[p]``), so this
padding is invisible to it.

Uniform-NA assumption
---------------------
The traced amplitude uses ``max(na_arr) + 1`` columns for every
panel; slots beyond ``na_arr[p] + 1`` in ``b_panels`` are zero (from
the standard preproc padding). For Legendre (LANG=1) that adds
zero-contribution higher-degree terms — harmless. For Kalbach-Mann
(LANG=2) the semantics differ between ``na=1`` (r only, ``a`` from
``bachaa``) and ``na=2`` (r and ``a`` both from evaluator); to avoid
a per-panel branch on ``na``, we require all panels of a subsection
to share the same NA. The K-39 MT649 / MT849 corner (NA varies per
panel) is not handled here; callers with such data can fall back to
the single-panel entry via explicit ``panel_idx=``.
"""
from __future__ import annotations

import numpy as np

from ..primitives.helpers import convert_interp_repr
from . import mf6_law1_kernel as _kernel
from . import mf6_law1_epintegral as _epi


def _requires_jax(xp):
    if xp.name != 'jax':
        raise ValueError(
            'multipanel-traced kernels require xp=jax; got '
            f'xp.name={xp.name!r}. On numpy the section-wide '
            'find_interval + panel loop in the non-traced kernel is '
            'strictly cheaper.'
        )


def _uniform_na(data):
    """Validate that all panels in ``data`` share the same NA."""
    nas = np.asarray(data.na_arr)
    return int(nas.min()) == int(nas.max())


def _gather_panel_scalar(arr_xp, p, xp):
    """xp.take on a scalar index, returning a scalar-shape array."""
    return xp.take(arr_xp, p, axis=0)


def _f6law1_con_panel_traced(
    data, p, e_scalar, tp, w, xp,
    max_nep, max_na_plus_one, lang, lep,
):
    """Traced single-panel continuum amplitude. Analogue of
    ``mf6_law1_kernel._f6law1_con_panel_bc`` with ``p`` as a tracer.

    Assumes LANG in {1, 2} and uniform NA across panels of the
    subsection (both enforced by the caller).
    """
    nep = _gather_panel_scalar(data.nep_arr, p, xp)     # tracer scalar
    nd = _gather_panel_scalar(data.nd_arr, p, xp)
    # na (angular parameter count) is uniform across panels; we use
    # max_na_plus_one for the trailing dim of the b row and let the
    # Legendre / Kalbach evaluator zero-fill the trailing L terms
    # implicitly via the padded b values.

    ep_row = xp.take(data.ep_panels, p, axis=0)         # (max_nep,)
    b_row = xp.take(data.b_panels, p, axis=0)           # (max_nep, max_na+1)

    # Searchsorted over the full padded row. Padding replicates
    # ep_row[nep-1], so tp exactly at ep_row[nep-1] returns nep;
    # tp beyond returns > nep. valid = strictly interior bracket.
    idx_full = xp.searchsorted(ep_row, tp, side='right')
    # Panel has continuum only if nep > nd; add mask.
    has_cont = nep > nd
    # Valid iff bracket falls in (nd, nep-1] — matches the pre-port
    # (idx_rel > 0) & (idx_rel < nep-nd) check.
    valid = (idx_full > nd) & (idx_full < nep) & has_cont

    # Clamp so advanced indexing is safe on invalid positions.
    # nd + 1 is always within [1, max_nep]; may be > max_nep-1 if
    # nd = max_nep-1 (degenerate panel), but has_cont guard rejects.
    idx_clamped = xp.where(valid, idx_full, nd + 1)
    i2 = xp.clip(idx_clamped, 1, max_nep - 1)
    i1 = i2 - 1

    ep1_val = xp.take(ep_row, i1, axis=0)               # (...,)
    ep2_val = xp.take(ep_row, i2, axis=0)

    if lang not in (1, 2):
        raise NotImplementedError(
            f'multipanel-traced kernel supports LANG in (1, 2); '
            f'got LANG={lang}'
        )

    # Fixed-size na: use max_na_plus_one columns always. Trailing
    # zeros in b_row contribute nothing to the Legendre sum
    # (na_true+1..max_na_plus_one all zero) and are harmless for
    # Kalbach with uniform NA.
    nt = max_na_plus_one
    # b_row[i1] has shape (..., max_na+1) via advanced indexing on
    # the leading axis. We use jnp.take_along_axis or plain fancy
    # indexing since i1 is a broadcast integer index over the (nE, ..)
    # axes; b_row shape is (max_nep, max_na+1). Straightforward
    # xp gather.
    a1 = xp.take(b_row, i1, axis=0)                     # (..., nt)
    a2 = xp.take(b_row, i2, axis=0)                     # (..., nt)

    tp_bc = tp[..., None]
    ep1_bc = ep1_val[..., None]
    ep2_bc = ep2_val[..., None]
    a_interp = _kernel._yintp_bc(lep, ep1_bc, a1, ep2_bc, a2, tp_bc, xp)

    if lang == 1:
        # Legendre: sum over max_na_plus_one L. Padded L slots have
        # zero coefficients (from b_panels preproc padding) so they
        # contribute nothing to the sum.
        f = _kernel._yleg_bc(a_interp, w, nt - 1, xp)
    else:  # lang == 2
        # Kalbach-Mann. na_true is uniform across panels (enforced
        # by caller); use the Python-side max as the static na to
        # feed _ykalbach_bc. na=1 vs na=2 semantics differ; require
        # uniform NA so we know at trace time which branch to use.
        na_static = int(np.asarray(data.na_arr).max())
        e_full = xp.broadcast_to(
            xp.asarray(e_scalar, dtype=tp.dtype), tp.shape,
        )
        f = _kernel._ykalbach_bc(
            data.zai, data.zap, data.za,
            e_full, tp, w, a_interp, na_static, xp,
        )

    return xp.where(valid, f, xp.zeros_like(f))


def _f6law1con_panel_pair_traced(
    data, panel_idx, lei, e_bc, tp_bc, w_bc, xp,
    max_nep, max_na_plus_one, lang, lep,
):
    """Traced two-panel continuum contribution. Analogue of
    ``mf6_law1_kernel._f6law1con_panel_pair_bc`` with ``panel_idx``
    as a tracer.

    Both branches (unit-base transform with continuum on both panels,
    and the single-panel-only branches when one panel lacks continuum)
    are computed and selected via ``xp.where`` on the tracer
    per-panel ``has_cont`` flags.
    """
    p1 = panel_idx
    p2 = panel_idx + 1
    e1 = _gather_panel_scalar(data.ei_mesh, p1, xp)
    e2 = _gather_panel_scalar(data.ei_mesh, p2, xp)
    nep1 = _gather_panel_scalar(data.nep_arr, p1, xp)
    nd1 = _gather_panel_scalar(data.nd_arr, p1, xp)
    nep2 = _gather_panel_scalar(data.nep_arr, p2, xp)
    nd2 = _gather_panel_scalar(data.nd_arr, p2, xp)

    p1_has_cont = nep1 > nd1
    p2_has_cont = nep2 > nd2

    # Panel-pair unit-base bounds. Padding invariant on ep_panels
    # (replicate last valid) makes ep_panels[p, -1] == ep_panels[
    # p, nep-1] == the continuum upper endpoint. Same trick lets
    # us gather ep_panels[p, nd] as the continuum lower endpoint
    # via xp.take on tracer nd.
    ep_row_p1 = xp.take(data.ep_panels, p1, axis=0)     # (max_nep,)
    ep_row_p2 = xp.take(data.ep_panels, p2, axis=0)
    x1low = xp.take(ep_row_p1, nd1, axis=0)
    x1high = xp.take(ep_row_p1, nep1 - 1, axis=0)
    x2low = xp.take(ep_row_p2, nd2, axis=0)
    x2high = xp.take(ep_row_p2, nep2 - 1, axis=0)

    x1range = x1high - x1low
    x2range = x2high - x2low
    e2_minus_e1 = e2 - e1
    e2_minus_e1_safe = xp.where(e2_minus_e1 == 0.0, 1.0, e2_minus_e1)
    yslope = (e_bc - e1) / e2_minus_e1_safe

    # Both-panel branch: unit-base transform of tp.
    xlow = x1low + yslope * (x2low - x1low)
    xhigh = x1high + yslope * (x2high - x1high)
    xrange = xhigh - xlow
    xrange_safe = xp.where(xrange == 0.0, 1.0, xrange)
    xslope = (tp_bc - xlow) / xrange_safe
    tp_at_p1 = x1low + xslope * x1range
    tp_at_p2 = x2low + xslope * x2range

    # Always compute both single-panel amplitudes at their own tp
    # (used when only one panel has continuum).
    f1_single = _f6law1_con_panel_traced(
        data, p1, e1, tp_bc, w_bc, xp,
        max_nep, max_na_plus_one, lang, lep,
    )
    f2_single = _f6law1_con_panel_traced(
        data, p2, e2, tp_bc, w_bc, xp,
        max_nep, max_na_plus_one, lang, lep,
    )
    # Both-panel path: evaluate each panel at the unit-base
    # transformed tp, scaled by x{1,2}range / xrange.
    x1range_over_xrange = x1range / xrange_safe
    x2range_over_xrange = x2range / xrange_safe
    f1_both = _f6law1_con_panel_traced(
        data, p1, e1, tp_at_p1, w_bc, xp,
        max_nep, max_na_plus_one, lang, lep,
    ) * x1range_over_xrange
    f2_both = _f6law1_con_panel_traced(
        data, p2, e2, tp_at_p2, w_bc, xp,
        max_nep, max_na_plus_one, lang, lep,
    ) * x2range_over_xrange

    both_cont = p1_has_cont & p2_has_cont
    f1 = xp.where(both_cont, f1_both,
                  xp.where(p1_has_cont, f1_single, xp.zeros_like(f1_single)))
    f2 = xp.where(both_cont, f2_both,
                  xp.where(p2_has_cont, f2_single, xp.zeros_like(f2_single)))

    # Outer E interp between panel results
    law = int(lei) % 10
    return _kernel._yintp_bc(law, e1, f1, e2, f2, e_bc, xp)


def _law1_spectrum_panel_pair_traced(
    data, panel_idx, lei, e_sub, ep_out_xp, eff_lct, gl_x, gl_w, xp,
    max_nep, max_na_plus_one, lang, lep,
):
    """Traced panel-pair spectrum kernel. Analogue of
    ``mf6_law1_epintegral._law1_spectrum_panel_pair`` with
    ``panel_idx`` as a tracer.

    Kink array has fixed shape ``(nE, nEp, 2 * max_nep)``. Padded-
    row slots collapse to the panel's continuum upper endpoint,
    yielding zero-width subpanels after sort — no contribution.
    """
    p1 = panel_idx
    p2 = panel_idx + 1
    e1 = _gather_panel_scalar(data.ei_mesh, p1, xp)
    e2 = _gather_panel_scalar(data.ei_mesh, p2, xp)
    nep1 = _gather_panel_scalar(data.nep_arr, p1, xp)
    nd1 = _gather_panel_scalar(data.nd_arr, p1, xp)
    nep2 = _gather_panel_scalar(data.nep_arr, p2, xp)
    nd2 = _gather_panel_scalar(data.nd_arr, p2, xp)

    n_e_sub = int(e_sub.shape[0])
    n_ep = int(ep_out_xp.shape[0])

    # Panel-pair Ep upper bound at each incident e. Padding invariant
    # makes ep_panels[p, -1] the continuum upper endpoint.
    ep_row_p1 = xp.take(data.ep_panels, p1, axis=0)
    ep_row_p2 = xp.take(data.ep_panels, p2, axis=0)
    ep1max = xp.take(ep_row_p1, nep1 - 1, axis=0)
    ep2max = xp.take(ep_row_p2, nep2 - 1, axis=0)
    e_bc = e_sub[:, None]                                    # (nE, 1)
    e2_minus_e1 = e2 - e1
    e2_minus_e1_safe = xp.where(e2_minus_e1 == 0.0, 1.0, e2_minus_e1)
    yslope = (e_bc - e1) / e2_minus_e1_safe                  # (nE, 1)
    epmax_eff = ep1max + yslope * (ep2max - ep1max)          # (nE, 1)

    ep_bc = ep_out_xp[None, :]                               # (1, nEp)
    umin = _epi._mu_min_bc(data, eff_lct, e_bc, ep_bc, epmax_eff, xp)
    z_max = _epi._safe_arccos(umin, xp)                      # (nE, nEp)

    # Kink locations. Under CM-frame mapping, each LEP knot Ep'_k of
    # a panel maps under unit-base to Ep'_k * (epmax_eff / epmax_p),
    # which is a kink of the amplitude in Ep' space, then to mu via
    # the CM<->LAB map. Traced version uses the FULL padded ep_row
    # (constant shape max_nep) instead of a slice by tracer nep.
    is_cm = (eff_lct == 2) or (eff_lct == 3 and data.awp < 4.0)
    c0 = float(np.sqrt(data.awi * data.awp) / (data.awi + data.awr))
    if is_cm and c0 > 0.0:
        ep1max_safe = xp.where(ep1max == 0.0, 1.0, ep1max)
        ep2max_safe = xp.where(ep2max == 0.0, 1.0, ep2max)
        # (max_nep,) -> broadcast to (nE, nEp, max_nep)
        knots1_img = (
            ep_row_p1[None, None, :]
            * (epmax_eff[..., None] / ep1max_safe)
        )
        knots2_img = (
            ep_row_p2[None, None, :]
            * (epmax_eff[..., None] / ep2max_safe)
        )
        all_knots = xp.concatenate([knots1_img, knots2_img], axis=-1)
        ep_safe = xp.where(ep_bc > 0.0, ep_bc, 1.0)
        denom = 2.0 * c0 * xp.sqrt(ep_safe * e_bc)
        c0_sq_e = (c0 * c0) * e_bc
        numer = ep_bc[..., None] + c0_sq_e[..., None] - all_knots
        mu_kinks = numer / denom[..., None]
    else:
        mu_kinks = xp.broadcast_to(
            umin[..., None], (n_e_sub, n_ep, 2 * max_nep),
        )

    umin_bc = umin[..., None]
    mu_kinks_c = xp.minimum(xp.maximum(mu_kinks, umin_bc), 1.0)
    z_kinks = _epi._safe_arccos(mu_kinks_c, xp)
    z_kinks = xp.sort(z_kinks, axis=-1)

    zeros_lead = xp.zeros((n_e_sub, n_ep, 1), dtype=z_max.dtype)
    zmax_trail = z_max[..., None]
    z_bounds = xp.concatenate([zeros_lead, z_kinks, zmax_trail], axis=-1)
    z_left = z_bounds[..., :-1]
    z_right = z_bounds[..., 1:]
    dz = z_right - z_left

    z_all = z_left[..., None] + (dz[..., None] / 2.0) * (1.0 + gl_x)
    mu_all = xp.cos(z_all)
    sin_z_all = xp.sin(z_all)

    e_full = xp.broadcast_to(e_bc[..., None, None], mu_all.shape)
    ep_full = xp.broadcast_to(ep_bc[..., None, None], mu_all.shape)
    tp_full, w_full, dinv_full = _kernel._mf6lab2cm_bc(
        data.awr, data.awi, data.awp, eff_lct,
        e_full, ep_full, mu_all, xp,
    )
    f_amp = _f6law1con_panel_pair_traced(
        data, panel_idx, lei, e_full, tp_full, w_full, xp,
        max_nep, max_na_plus_one, lang, lep,
    )
    integrand = f_amp * dinv_full * sin_z_all
    weighted = integrand * gl_w * (dz[..., None] / 2.0)
    # Panel-pair mask: at least one panel must have continuum.
    p1_has_cont = nep1 > nd1
    p2_has_cont = nep2 > nd2
    any_cont = p1_has_cont | p2_has_cont
    result = xp.sum(weighted, axis=(-2, -1))
    return xp.where(any_cont, result, xp.zeros_like(result))


def integrate_law1_spectrum_multipanel_traced(
    data, energies_in, energies_out, eff_lct, gl_x, gl_w, xp,
):
    """Multi-panel LAW=1 continuum spectrum via tracer-panel-index
    kernel. Auto-dispatch entry from
    ``mf6_law1_epintegral.integrate_law1_spectrum`` when ``xp=jax``
    and no ``panel_idx=`` is explicitly given.

    Each ``E`` in ``energies_in`` picks its panel via ``searchsorted``
    on ``ei_mesh`` (as a tracer int), then evaluates the single-graph
    per-panel-pair kernel. JAX tracers on ``E`` propagate to
    ``jax.grad`` inside each panel-pair interior; at exactly the
    ei_mesh knots the gradient is one-sided (physical C0 kink of the
    amplitude at every panel boundary).
    """
    _requires_jax(xp)
    if not _uniform_na(data):
        raise NotImplementedError(
            'multipanel-traced kernel requires uniform NA across '
            'panels of the subsection. This file has varying NA; '
            'use explicit panel_idx= entry per panel instead.'
        )

    max_nep = int(data.ep_panels.shape[1])
    max_na_plus_one = int(data.b_panels.shape[2])
    lang = int(data.lang)
    lep = int(data.lep)

    e_in_xp = xp.asarray(energies_in)
    ep_out_xp = xp.asarray(energies_out)
    n_pairs = int(np.asarray(data.ei_mesh).shape[0]) - 1

    # Concrete numpy view for outer-region interp lookup (data-side
    # only; the panel-pair kernel gathers with a tracer index).
    ei_interp_full = convert_interp_repr(
        np.asarray(data.int_arr), np.asarray(data.nbt_arr),
    )
    # If all panel-pairs share the same lei (typical for corpus), we
    # can pass it as a static Python int. If they differ, fall back
    # to always-INT=2 (LAW=1 continuum interpolation matches lin-lin
    # in every corpus subsection surveyed to date).
    unique_lei = np.unique(np.asarray(ei_interp_full))
    if unique_lei.size == 1:
        static_lei = int(unique_lei[0])
    else:
        raise NotImplementedError(
            'multipanel-traced kernel requires uniform lei across '
            'panel pairs. This file has mixed lei; use explicit '
            'panel_idx= entries.'
        )

    ei_mesh_xp = xp.asarray(data.ei_mesh)
    e_min = float(np.asarray(data.ei_mesh)[0])
    e_max = float(np.asarray(data.ei_mesh)[-1])

    def _spectrum_one_e(e_scalar):
        # panel_idx = clamp(searchsorted(ei_mesh, e, side='right') - 1,
        #                   0, n_pairs - 1)
        idx = xp.searchsorted(ei_mesh_xp, e_scalar, side='right') - 1
        idx = xp.clip(idx, 0, n_pairs - 1)
        row = _law1_spectrum_panel_pair_traced(
            data, idx, static_lei,
            e_scalar[None], ep_out_xp, eff_lct, gl_x, gl_w, xp,
            max_nep, max_na_plus_one, lang, lep,
        )[0]
        # Mask out-of-range E (below section min or above section max)
        in_range = (e_scalar >= e_min) & (e_scalar <= e_max)
        return xp.where(in_range, row, xp.zeros_like(row))

    # vmap over the E axis. Under jax, vmap traces once with a scalar
    # tracer -> single-graph compile.
    import jax
    result = jax.vmap(_spectrum_one_e)(e_in_xp)
    return result
