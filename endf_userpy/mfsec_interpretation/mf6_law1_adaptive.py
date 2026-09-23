"""MF6 LAW=1 continuum spectrum adaptive linearization
(``feep_full_law1con`` port).

Given a single incident ``E`` and (optionally) a hint list of
outgoing energies of interest, returns an ``(ep_mesh, f_mesh)`` pair
where linear interpolation of ``f_mesh`` on ``ep_mesh`` is accurate
to ``tol`` relative for any ``E'`` in ``[ep_mesh[0], ep_mesh[-1]]``.
This is the primitive downstream data-preparation code
(multi-group group-averaging, MC transport preprocessors, sanity
plots) consumes to avoid preselecting an ``E'`` grid.

Algorithm mirrors the Fortran ``feep_full_law1con`` (endf6.f90 line
2754):

1. **Initial grid** (``_initial_epgrid``): seed with every LEP-
   piecewise ``E'`` knot of both bracketing panels (unit-base
   transformed to the query ``E``), the CM<->LAB kinematic special
   points (``c0**2 E`` and ``(sqrt(Ep_max) + c0 sqrt(E))**2`` where
   the LAB f(E, E') slope kinks), plus caller-supplied ``E'``
   hints. Sort and dedupe.
2. **Adaptive bisection**: for each initial interval, evaluate the
   midpoint. Accept if linear interpolation between the endpoints
   matches the midpoint value within ``tol`` and both sub-slopes
   have the same sign. Otherwise recurse on the left half, pushing
   the right endpoint onto a small stack (max depth 20). Same
   accept criteria as the Fortran.

Numpy-only. Adaptive bisection has data-dependent iteration count
which is not JAX-friendly; a fixed-max-depth uniform-bisection
variant can be added later if JAX is wanted here.

Boundary behaviour at ``Ep -> 0``
---------------------------------

The LAB-frame emission spectrum rises like ``f(Ep) ~ C * sqrt(Ep)``
near ``Ep = 0`` (the CM<->LAB Jacobian ``dinv = sqrt(Ep_CM/Ep_LAB)``
diverges and the mu-integration domain shrinks simultaneously),
which has infinite derivative at the boundary. For an interval
``[eps1, eps2]``, the linear-interp error at the midpoint is
``1 - ((sqrt(eps1) + sqrt(eps2))/2) / sqrt((eps1 + eps2)/2)``,
which for ``eps1 > 0`` shrinks quickly under bisection but for
``eps1 = 0`` evaluates to a fixed ``1 - 1/sqrt(2) ~= 0.293``,
independent of ``eps2``. Bisecting from ``[0, w]`` all the way to
``[0, w / 2**20]`` does not move that number: the leftmost interval
always has ``eps1 = 0``, so the leftmost accepted mesh point (near
the max-depth safety cap) inherits a ~29% relative interp error.

**Both the Fortran** ``feep_full_law1con`` **and this port hit the
same boundary error at the same midpoint** -- the parity test
against Fortran confirms this (see
``tests/test_mf6_law1_adaptive.py``). This is not a bug in either
implementation; it is a limitation of bisecting a linear-interp
representation against a non-analytic boundary.

**Practical impact** on downstream use is negligible: the absolute
values of ``f`` in the affected region (typically the first few
tens of eV) are 4-5 orders of magnitude below the spectrum peak,
so any integral over a physically meaningful outgoing-energy group
picks up essentially nothing from there. Multi-group generators and
Monte Carlo CDF builders that consume the linearized mesh are
unaffected.

**If the fine-structure of the boundary matters** (unusually
fine-grained downstream grids sensitive down to sub-eV): call
:func:`mf6_law1_epintegral.integrate_law1_spectrum` directly and
use adaptive quadrature (e.g. ``scipy.integrate.quad`` with
``points=`` set to the mesh knots) instead of relying on the
linearized mesh. Options to change the port itself (a log-scale
bisection near ``ep=0``, a leading boundary segment ``(0, 0) ->
(ep_min, f_min)``, or non-linear interp laws per interval) are
tracked in the issue backlog but not implemented; they add API
complexity for a scenario nobody has yet needed.
"""
from __future__ import annotations

import numpy as np

from ..primitives import array_ns
from ..primitives.helpers import convert_interp_repr, find_interval
from . import mf6_law1_epintegral as _epint


def linearize_law1_spectrum(
    data, e_scalar, energies_out_hint=None, to_lab=True,
    n_gl=10, tol=1.0e-3, max_depth=20, max_points=100_000,
    tol_x=1.0e-6,
):
    """Adaptive linearization of ``f(E, E')`` for one incident ``E``.

    Parameters
    ----------
    data : MF6Law1Data
    e_scalar : float. One incident energy (LAB, eV), inside
        ``[ei_mesh[0], ei_mesh[-1]]``.
    energies_out_hint : optional (n,) array. ``E'`` values the
        caller wants included in the output mesh (regardless of the
        adaptive step). Merged with the internal seed grid.
    to_lab : bool. LAW=1 is always LAB; this argument gates whether
        the section's LCT is honoured (False forces LAB frame).
    n_gl : Gauss-Legendre order for the underlying point-wise
        integrator (see :mod:`mf6_law1_epintegral`).
    tol : relative tolerance for the linearization accept
        criterion. Matches the Fortran default of ``1e-3``.
    max_depth : bisection stack max depth (safety cap; matches the
        Fortran ``ksmax=20``).
    max_points : soft cap on the returned mesh size. Raises
        ``ValueError`` if the linearization exceeds it.
    tol_x : relative interval-width floor. Matches the Fortran
        ``tolx=1e-6``.

    Returns
    -------
    (ep_mesh, f_mesh, dev_mesh) : three (n_out,) arrays.
        ``ep_mesh``: monotonically increasing E' values.
        ``f_mesh``: emission spectrum at ep_mesh.
        ``dev_mesh``: absolute deviation estimate (zeros here,
        placeholder for parity with the Fortran signature; the
        Fortran ``dy`` came from Romberg's per-level extrapolation,
        which the GL-based port does not produce).
    """
    xp = array_ns.get_backend('numpy')
    lct = data.lct if to_lab else 1
    if lct in (1, 2):
        eff_lct = lct
    elif lct == 3:
        eff_lct = 1 if data.awp > 4 else 2
    else:
        raise NotImplementedError(f'LCT={lct} not implemented')

    e_val = float(e_scalar)
    ei_mesh = np.asarray(data.ei_mesh)
    if e_val < float(ei_mesh[0]) or e_val > float(ei_mesh[-1]):
        raise ValueError(
            f'e_scalar={e_val} outside section E-mesh '
            f'[{float(ei_mesh[0])}, {float(ei_mesh[-1])}]'
        )
    panel_idx = int(find_interval(ei_mesh, np.array([e_val]))[0])
    ei_interp_full = convert_interp_repr(
        np.asarray(data.int_arr), np.asarray(data.nbt_arr),
    )
    lei = int(ei_interp_full[panel_idx])

    ep0 = _initial_epgrid(
        data, e_val, panel_idx, eff_lct,
        None if energies_out_hint is None else np.asarray(energies_out_hint, dtype=float),
    )

    # Static GL nodes (shared across all midpoint evaluations)
    gl_x_np, gl_w_np = np.polynomial.legendre.leggauss(int(n_gl))
    gl_x = xp.asarray(gl_x_np)
    gl_w = xp.asarray(gl_w_np)

    def eval_f(ep_scalar: float) -> float:
        result = _epint._law1_spectrum_panel_pair(
            data, panel_idx, lei,
            xp.asarray([e_val]), xp.asarray([ep_scalar]),
            eff_lct, gl_x, gl_w, xp,
        )
        return float(result[0, 0])

    return _adaptive_bisect(
        ep0, eval_f, tol=tol, tol_x=tol_x,
        max_depth=max_depth, max_points=max_points,
    )


def _initial_epgrid(data, e_val, panel_idx, eff_lct, epu):
    """Build the initial E' seed grid (mirror of Fortran
    ``initial_epgrid``)."""
    p1 = panel_idx
    p2 = panel_idx + 1
    e1 = float(data.ei_mesh[p1])
    e2 = float(data.ei_mesh[p2])
    nep1 = int(data.nep_arr[p1])
    nd1 = int(data.nd_arr[p1])
    nep2 = int(data.nep_arr[p2])
    nd2 = int(data.nd_arr[p2])

    # get_griddata: continuum-only Ep min/max per panel
    if nep1 > nd1 + 1:
        ep1min = float(data.ep_panels[p1, nd1])
        ep1max = float(data.ep_panels[p1, nep1 - 1])
    else:
        ep1min = 0.0
        ep1max = 0.0
    if nep2 > nd2 + 1:
        ep2min = float(data.ep_panels[p2, nd2])
        ep2max = float(data.ep_panels[p2, nep2 - 1])
    else:
        ep2min = 0.0
        ep2max = 0.0
    ep1range = ep1max - ep1min
    ep2range = ep2max - ep2min
    slope = (e_val - e1) / (e2 - e1)
    epmin_eff = ep1min + (ep2min - ep1min) * slope
    epmax_eff = ep1max + (ep2max - ep1max) * slope
    eprange = epmax_eff - epmin_eff

    grid = [0.0]
    is_cm = (eff_lct == 2) or (eff_lct == 3 and data.awp < 4.0)
    if is_cm:
        c0 = float(np.sqrt(data.awi * data.awp) / (data.awr + data.awi))
        c = c0 * c0 * e_val
        grid.append(c)
        el = np.sqrt(epmax_eff) + c0 * np.sqrt(e_val)
        grid.append(el * el)
        if nep1 > nd1 + 1 and e_val != e2:
            for i in range(nd1, nep1):
                ep_val = data.ep_panels[p1, i]
                if ep1range > 0:
                    ep_i = epmin_eff + (float(ep_val) - ep1min) * eprange / ep1range
                else:
                    ep_i = epmin_eff
                grid.append(float(ep_i) + c)
            el1 = np.sqrt(ep1min + ep1range) + c0 * np.sqrt(e1)
            grid.append(el1 * el1)
        if nep2 > nd2 + 1 and e_val != e1:
            for i in range(nd2, nep2):
                ep_val = data.ep_panels[p2, i]
                if ep2range > 0:
                    ep_i = epmin_eff + (float(ep_val) - ep2min) * eprange / ep2range
                else:
                    ep_i = epmin_eff
                grid.append(float(ep_i) + c)
            el2 = np.sqrt(ep2min + ep2range) + c0 * np.sqrt(e2)
            grid.append(el2 * el2)
    else:
        if nep1 > nd1 + 1 and e_val != e2:
            for i in range(nd1, nep1):
                ep_val = float(data.ep_panels[p1, i])
                if ep1range > 0:
                    grid.append(epmin_eff + (ep_val - ep1min) * eprange / ep1range)
                else:
                    grid.append(epmin_eff)
            grid.append(float(data.ep_panels[p1, nep1 - 1]))
        if nep2 > nd2 + 1 and e_val != e1:
            for i in range(nd2, nep2):
                ep_val = float(data.ep_panels[p2, i])
                if ep2range > 0:
                    grid.append(epmin_eff + (ep_val - ep2min) * eprange / ep2range)
                else:
                    grid.append(epmin_eff)
        grid.append(float(data.ep_panels[p2, nep2 - 1]))

    if epu is not None:
        grid.extend([float(x) for x in epu])

    arr = np.asarray(grid, dtype=float)
    arr = np.sort(arr)
    # Remove entries within tol0=1e-6 relative of their predecessor.
    tol0 = 1.0e-6
    keep = np.ones(arr.shape[0], dtype=bool)
    for i in range(1, arr.shape[0]):
        ref = abs(arr[i])
        if ref == 0.0:
            if arr[i] == arr[i - 1]:
                keep[i] = False
        elif abs(arr[i] - arr[i - 1]) <= tol0 * ref:
            keep[i] = False
    return arr[keep]


def _adaptive_bisect(ep0, eval_f, tol, tol_x, max_depth, max_points):
    """Adaptive-bisection driver (mirror of the Fortran outer + inner
    loop in ``feep_full_law1con``). ``eval_f(ep)`` is the scalar
    ``f(E, ep)`` evaluator (E is closed over)."""
    n0 = int(ep0.shape[0])
    if n0 < 2:
        # Trivial: single seed point, just evaluate and return.
        y = np.array([eval_f(float(ep0[0]))]) if n0 == 1 else np.array([])
        return ep0, y, np.zeros_like(y)

    ep_out = np.empty(max_points, dtype=float)
    f_out = np.empty(max_points, dtype=float)
    dev_out = np.empty(max_points, dtype=float)
    stack_x = np.empty(max_depth, dtype=float)
    stack_y = np.empty(max_depth, dtype=float)
    stack_dy = np.empty(max_depth, dtype=float)

    x1 = float(ep0[0])
    y1 = eval_f(x1)
    dy1 = 0.0
    ep_out[0] = x1
    f_out[0] = y1
    dev_out[0] = dy1
    j = 1

    for i in range(1, n0):
        x2 = float(ep0[i])
        y2 = eval_f(x2)
        dy2 = 0.0
        k = 0  # stack depth
        istop = False
        while not istop:
            xm = 0.5 * (x1 + x2)
            hm = xm - x1
            ym = eval_f(xm)
            dym = 0.0
            yl = 0.5 * (y1 + y2)
            if hm == 0.0:
                slope1 = 0.0
                slope2 = 0.0
            else:
                slope1 = (ym - y1) / hm
                slope2 = (y2 - ym) / hm
            accept_lin = (
                abs(ym - yl) <= abs(tol * ym)
                and slope1 * slope2 > 0.0
            )
            accept_flat = (y2 == ym and y1 == ym)
            accept_narrow = abs(x2 - x1) <= abs(tol_x * x2)
            accept_stack_full = (k == max_depth)
            if accept_lin or accept_flat or accept_narrow or accept_stack_full:
                if j >= max_points:
                    raise ValueError(
                        f'linearization exceeded max_points={max_points}; '
                        'try loosening tol or raising max_points.'
                    )
                ep_out[j] = x2
                f_out[j] = y2
                dev_out[j] = dy2
                j += 1
                if k == 0:
                    istop = True
                else:
                    x1, y1, dy1 = x2, y2, dy2
                    k -= 1
                    x2, y2, dy2 = stack_x[k], stack_y[k], stack_dy[k]
            else:
                stack_x[k] = x2
                stack_y[k] = y2
                stack_dy[k] = dy2
                k += 1
                x2, y2, dy2 = xm, ym, dym
        x1, y1, dy1 = x2, y2, dy2

    return ep_out[:j].copy(), f_out[:j].copy(), dev_out[:j].copy()
