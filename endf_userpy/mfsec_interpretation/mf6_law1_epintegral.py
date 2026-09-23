"""Backend-agnostic MF6 LAW=1 continuum spectrum ``f(E, E')``
via ``mu``-integration of the double-differential distribution.

Fixed-grid emission spectrum: for each caller-supplied ``(E, E')``,
integrate

    f_spec(E, E') = int_{mu_min}^{+1} f_con(E, E', mu) dmu

where ``f_con`` is the LAW=1 continuum reconstruction (see
:mod:`mf6_law1_kernel`) and ``mu_min = mu_min(E, E')`` comes from
the LAB-frame kinematic cutoff (Fortran ``feep_law1con`` line 3020).

**Kink-aware Gauss-Legendre.** The amplitude has a C0 kink at every
LEP-piecewise ``E'`` knot of the two bracketing panels; under the
CM<->LAB map ``E'(mu) = Ep + c0^2 E - 2 c0 sqrt(Ep E) mu`` each knot
maps to a specific mu location. This module partitions the polar
angle ``z = acos(mu)`` at exactly those kink locations, then applies
fixed-order Gauss-Legendre per subpanel. Within a subpanel the
amplitude is smooth (LEP interpolation is analytic per piece) and GL
converges exponentially; between subpanels the sum stitches the
pieces together without penalty.

Compared to the earlier equal-``dz`` polar subdivision, kink-aware
subdivision gives orders-of-magnitude tighter accuracy at
comparable node count on realistic ENDF tabulations (kinks come
from physics, not a knob).

Backend-agnostic and JAX-safe: kink count per panel-pair is a
static Python-side ``nep1 + nep2``, so shapes are known at trace
time. Kink locations that fall outside ``(mu_min, +1)`` are clamped
to the endpoint and yield zero-width subpanels (no branching, no
``lax.while_loop``).

Design: input is an :class:`~mf6_law1_preproc.MF6Law1Data`
dataclass (produced by
:func:`~mf6_law1_preproc.mf6_law1_data_from_endf_dict`); output is
a ``(n_E, n_E')`` array. JAX tracers stored in ``data.b_panels``
propagate through the integrator so ``jax.grad`` reaches back to
file-stored angular parameters end-to-end.
"""
from __future__ import annotations

import numpy as np

from ..primitives import array_ns
from ..primitives.helpers import convert_interp_repr, find_interval
from . import mf6_law1_kernel as _kernel


def integrate_law1_spectrum(data, energies_in, energies_out, to_lab,
                              xp=None, n_gl=10, panel_idx=None):
    """MF6 LAW=1 continuum emission spectrum ``f(E, E')`` via
    kink-aware polar-angle Gauss-Legendre.

    Parameters
    ----------
    data : MF6Law1Data
    energies_in : (n_E,) query incident energies (LAB, eV).
    energies_out : (n_Ep,) query outgoing energies (LAB, eV).
    to_lab : bool. LAW=1 is always LAB; this argument gates
        whether the section's LCT is honoured. False forces LAB
        (no CM->LAB shift), True uses the section's LCT.
    xp : optional backend adapter (``array_ns.get_backend(name)``).
        Defaults to numpy.
    n_gl : Gauss-Legendre order per kink-aligned subpanel
        (default 10). Each subpanel spans one LEP-piecewise region
        of the amplitude, so this order controls exponential
        convergence within each smooth piece.
    panel_idx : optional Python int. If given, skip the
        section-wide find_interval / panel loop and evaluate the
        single-panel-pair kernel ``(panel_idx, panel_idx + 1)``
        directly on ``energies_in`` and ``energies_out`` (kept
        xp-native, no numpy conversion). This is the entry point
        for ``jax.grad`` wrt ``energies_in`` / ``energies_out``:
        with a static panel choice, both arrays flow through the
        integrator as tracers. The caller is responsible for
        keeping ``energies_in`` inside ``[ei_mesh[panel_idx],
        ei_mesh[panel_idx + 1]]`` -- the amplitude has physical
        C0 kinks at each panel knot, so grad across a knot is
        undefined.

    Returns
    -------
    (n_E, n_Ep) array. Same dtype as ``data.b_panels`` (float on
    numpy, backend-native on JAX).
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    lct = data.lct if to_lab else 1
    if lct in (1, 2):
        eff_lct = lct
    elif lct == 3:
        eff_lct = 1 if data.awp > 4 else 2
    else:
        raise NotImplementedError(f'LCT={lct} not implemented')

    # Gauss-Legendre nodes / weights (static, precomputed once)
    gl_x_np, gl_w_np = np.polynomial.legendre.leggauss(int(n_gl))
    gl_x = xp.asarray(gl_x_np)
    gl_w = xp.asarray(gl_w_np)

    if panel_idx is not None:
        # Single-panel autodiff path: no numpy conversion, no
        # find_interval, no panel loop. Tracers flow through
        # energies_in / energies_out end-to-end.
        p = int(panel_idx)
        ei_interp_full = convert_interp_repr(
            np.asarray(data.int_arr), np.asarray(data.nbt_arr),
        )
        lei = int(ei_interp_full[p])
        e_sub = xp.asarray(energies_in)
        ep_out_xp = xp.asarray(energies_out)
        return _law1_spectrum_panel_pair(
            data, p, lei, e_sub, ep_out_xp, eff_lct,
            gl_x, gl_w, xp,
        )

    e_in_np = np.asarray(energies_in, dtype=float)
    ep_out_np = np.asarray(energies_out, dtype=float)
    n_e = e_in_np.shape[0]
    n_ep = ep_out_np.shape[0]
    ei_mesh_np = np.asarray(data.ei_mesh)

    ei_interp_full = convert_interp_repr(
        np.asarray(data.int_arr), np.asarray(data.nbt_arr),
    )

    # Mask to keep only E's inside the section's E-mesh range
    e_min = float(ei_mesh_np[0])
    e_max = float(ei_mesh_np[-1])
    inside_mask_np = (e_in_np >= e_min) & (e_in_np <= e_max)

    ep_out_xp = xp.asarray(ep_out_np)
    result = xp.zeros((n_e, n_ep), dtype=data.b_panels.dtype)
    if not inside_mask_np.any():
        return result

    e_inside_np = e_in_np[inside_mask_np]
    idcs = find_interval(ei_mesh_np, e_inside_np)
    inside_positions = np.where(inside_mask_np)[0]

    for panel_idx_iter in np.unique(idcs):
        row_mask = (idcs == panel_idx_iter)
        rows_np = inside_positions[row_mask]
        e_sub_np = e_inside_np[row_mask]
        e_sub = xp.asarray(e_sub_np)
        lei = int(ei_interp_full[panel_idx_iter])
        f_sub = _law1_spectrum_panel_pair(
            data, int(panel_idx_iter), lei, e_sub, ep_out_xp, eff_lct,
            gl_x, gl_w, xp,
        )
        result = _kernel._scatter_rows(result, rows_np, f_sub, xp)

    return result


def _law1_spectrum_panel_pair(data, panel_idx, lei, e_sub, ep_out_xp,
                                eff_lct, gl_x, gl_w, xp):
    """Per-panel-pair spectrum on ``(e_sub, ep_out)`` via kink-aware
    polar-angle Gauss-Legendre. See module docstring for the kink
    story.

    ``nep1 + nep2`` sets the (fixed) kink-count per panel pair; the
    total subpanel count is ``nep1 + nep2 + 1``. Kinks that map
    outside ``(mu_min, +1)`` clamp to the endpoint and give
    zero-width subpanels.
    """
    p1 = panel_idx
    p2 = panel_idx + 1
    e1 = float(data.ei_mesh[p1])
    e2 = float(data.ei_mesh[p2])
    nep1 = int(data.nep_arr[p1])
    nd1 = int(data.nd_arr[p1])
    nep2 = int(data.nep_arr[p2])
    nd2 = int(data.nd_arr[p2])

    n_e_sub = int(e_sub.shape[0])
    n_ep = int(ep_out_xp.shape[0])

    if nep1 <= nd1 or nep2 <= nd2:
        # No continuum data on at least one panel; spectrum is 0.
        return xp.zeros((n_e_sub, n_ep), dtype=data.b_panels.dtype)

    # Panel-pair Ep upper bound at each incident e (unit-base
    # transformed from panel-1 and panel-2 continuum tail).
    ep1max = float(data.ep_panels[p1, nep1 - 1])
    ep2max = float(data.ep_panels[p2, nep2 - 1])
    e_bc = e_sub[:, None]                                        # (nE, 1)
    yslope = (e_bc - e1) / (e2 - e1)                             # (nE, 1)
    epmax_eff = ep1max + yslope * (ep2max - ep1max)              # (nE, 1)

    ep_bc = ep_out_xp[None, :]                                   # (1, nEp)
    umin = _mu_min_bc(data, eff_lct, e_bc, ep_bc, epmax_eff, xp)  # (nE, nEp)
    z_max = _safe_arccos(umin, xp)                                # (nE, nEp)

    # Kink locations. In CM-frame mode, each LEP knot Ep'_k of a
    # panel maps under unit-base to Ep'_k * (epmax_eff / epmax_p),
    # and that image is a kink of the amplitude in Ep' space. Under
    # the CM<->LAB map (fixed E, Ep) this Ep' image is at
    # mu_kink = (Ep + c0^2 E - Ep'_image) / (2 c0 sqrt(Ep E)). In
    # LAB frame we short-circuit to "no kinks": every kink clamps to
    # umin, so the whole [0, z_max] is one wide subpanel and GL of
    # the smooth (Legendre / Kalbach) amplitude covers it.
    is_cm = (eff_lct == 2) or (eff_lct == 3 and data.awp < 4.0)
    c0 = float(np.sqrt(data.awi * data.awp) / (data.awi + data.awr))
    if is_cm and c0 > 0.0:
        ep1_arr = xp.asarray(data.ep_panels[p1, :nep1])
        ep2_arr = xp.asarray(data.ep_panels[p2, :nep2])
        knots1_img = ep1_arr[None, None, :] * (epmax_eff[..., None] / ep1max)
        knots2_img = ep2_arr[None, None, :] * (epmax_eff[..., None] / ep2max)
        # (nE, 1, K) then broadcast to (nE, nEp, K).
        all_knots = xp.concatenate([knots1_img, knots2_img], axis=-1)
        all_knots = xp.broadcast_to(
            all_knots, (n_e_sub, n_ep, nep1 + nep2),
        )
        ep_safe = xp.where(ep_bc > 0.0, ep_bc, 1.0)
        denom = 2.0 * c0 * xp.sqrt(ep_safe * e_bc)                # (nE, nEp)
        c0_sq_e = (c0 * c0) * e_bc                                # (nE, 1)
        numer = ep_bc[..., None] + c0_sq_e[..., None] - all_knots  # (nE, nEp, K)
        mu_kinks = numer / denom[..., None]
    else:
        # LAB frame or massless ejectile: no CM<->LAB-map kinks.
        mu_kinks = xp.broadcast_to(
            umin[..., None], (n_e_sub, n_ep, nep1 + nep2),
        )

    # Clamp to [umin, +1] so out-of-range kinks land at endpoints
    # (yielding zero-width subpanels). Then sort ascending in z.
    umin_bc = umin[..., None]                                    # (nE, nEp, 1)
    mu_kinks_c = xp.minimum(xp.maximum(mu_kinks, umin_bc), 1.0)
    z_kinks = _safe_arccos(mu_kinks_c, xp)                       # (nE, nEp, K)
    z_kinks = xp.sort(z_kinks, axis=-1)

    # Boundaries: [0, sorted z_kinks..., z_max]. Shape (nE, nEp, K+2).
    zeros_lead = xp.zeros((n_e_sub, n_ep, 1), dtype=z_max.dtype)
    zmax_trail = z_max[..., None]
    z_bounds = xp.concatenate([zeros_lead, z_kinks, zmax_trail], axis=-1)
    z_left = z_bounds[..., :-1]                                  # (nE, nEp, K+1)
    z_right = z_bounds[..., 1:]                                  # (nE, nEp, K+1)
    dz = z_right - z_left                                        # (nE, nEp, n_sub)

    # GL nodes at (nE, nEp, n_sub, n_gl):
    z_all = z_left[..., None] + (dz[..., None] / 2.0) * (1.0 + gl_x)
    mu_all = xp.cos(z_all)
    sin_z_all = xp.sin(z_all)

    e_full = xp.broadcast_to(e_bc[..., None, None], mu_all.shape)
    ep_full = xp.broadcast_to(ep_bc[..., None, None], mu_all.shape)
    tp_full, w_full, dinv_full = _kernel._mf6lab2cm_bc(
        data.awr, data.awi, data.awp, eff_lct,
        e_full, ep_full, mu_all, xp,
    )
    f_amp = _kernel._f6law1con_panel_pair_bc(
        data, panel_idx, lei, e_full, tp_full, w_full, xp,
    )
    integrand = f_amp * dinv_full * sin_z_all
    weighted = integrand * gl_w * (dz[..., None] / 2.0)
    return xp.sum(weighted, axis=(-2, -1))


def _safe_arccos(x, xp):
    """``arccos(x)`` that stays JAX-grad-safe at the ``|x| = 1``
    boundaries. Naive ``arccos(clip(x, -1, 1))`` has an infinite
    derivative when the input is clipped to +/-1 (arccos'(1) = -1 /
    sqrt(1 - 1**2)); the double-``where`` pattern ensures the
    traced computation only ever evaluates ``arccos`` at a strictly
    interior point, and the final ``where`` selects the true value
    without propagating the infinite derivative.

    Values ``x >= 1 - eps`` map to ``0`` and ``x <= -1 + eps`` to
    ``pi``. Both branches carry zero derivative (constants), so
    outside the valid range the gradient is exactly zero -- which
    is what we want for zero-width subpanels.
    """
    eps = 1e-15
    x_safe = xp.where((x > -1.0 + eps) & (x < 1.0 - eps), x, 0.0)
    z_from_arccos = xp.arccos(x_safe)
    z_hi = xp.zeros_like(z_from_arccos)
    z_lo = xp.asarray(np.pi, dtype=z_from_arccos.dtype)
    z = xp.where(
        x >= 1.0 - eps, z_hi,
        xp.where(x <= -1.0 + eps, z_lo, z_from_arccos),
    )
    return z


def _mu_min_bc(data, eff_lct, e_bc, ep_bc, epmax_eff, xp):
    """Broadcast LAB-frame lower-cosine cutoff ``mu_min(E, E')``
    for MF6 LAW=1 continuum ``mu`` integration (matches Fortran
    ``feep_law1con`` lines 3020-3034).

    Returns ``mu_min`` clamped to ``[-1, 1]``; ``ep <= 0`` returns
    +1 so the integration domain is empty (zero contribution).
    """
    if not (eff_lct == 2 or (eff_lct == 3 and data.awp < 4.0)):
        # LAB frame: mu_min = -1 (full range).
        return xp.broadcast_to(
            xp.asarray(-1.0, dtype=e_bc.dtype),
            xp.broadcast_shapes(e_bc.shape, ep_bc.shape),
        )
    c0 = float(np.sqrt(data.awi * data.awp) / (data.awi + data.awr))
    if c0 == 0.0:
        # Photon/massless ejectile: no LAB<->CM shift, LAB cutoff
        # collapses to mu_min = -1 (full range).
        return xp.broadcast_to(
            xp.asarray(-1.0, dtype=e_bc.dtype),
            xp.broadcast_shapes(e_bc.shape, ep_bc.shape),
        )
    c0_sq_e = (c0 * c0) * e_bc
    # Guard ep <= 0 so sqrt(ep * e) doesn't NaN out or divide by
    # zero; the where at the end masks those cells back to umin = 1
    # (empty domain). e_bc is always > 0 in the inside-mask path.
    ep_safe = xp.where(ep_bc > 0.0, ep_bc, 1.0)
    denom_safe = 2.0 * c0 * xp.sqrt(ep_safe * e_bc)
    umin_raw = (ep_bc + c0_sq_e - epmax_eff) / denom_safe
    umin_clamped = xp.clip(umin_raw, -1.0, 1.0)
    return xp.where(ep_bc > 0.0, umin_clamped, 1.0)
