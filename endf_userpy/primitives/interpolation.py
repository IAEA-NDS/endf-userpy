"""ENDF-6 interpolation primitives (dict-of-arrays layout).

The older / broader of the two interpolation modules in
``endf_userpy.primitives``. Works on the ``endf_parserpy`` nested
dict-of-arrays representation (records reached by name, e.g.
``tab1['E']`` / ``tab1['xs']``, with parallel ``NBT`` / ``INT``
arrays) and is used by every MF3 / MF5 / MF13 / MF15 code path.

Backend support: the leaf arithmetic (per-scheme interp,
two-point column interp, unit-base transform,
:func:`evaluate_interp_legendre_polynomials`,
:func:`interp_tab2`) accepts an optional ``xp=None`` adapter that
resolves to numpy by default. Passing
``xp=array_ns.get_backend('jax')`` makes the arithmetic run on
JAX -- values in query axes (``x``, ``mu``, ``y``) propagate
gradients cleanly. File-side setup that has dynamic shapes
(``treat_duplicates`` on a mesh with runtime-many duplicated
values, dict-of-records access) stays on numpy per the same
architectural line used by :mod:`endf_userpy.primitives.tab1`;
the values it produces are converted at the boundary with
``xp.asarray``.
"""
import numpy as np
from . import array_ns
from .helpers import (
    check_int_nbt,
    find_interval,
    get_enclosing_points,
    convert_interp_repr,
    treat_duplicates,
)


def _resolve_xp(xp):
    return xp if xp is not None else array_ns.get_backend('numpy')


def interp_const(x, xp_mesh, fp, xp=None):
    """Constant interpolation. Backend-agnostic; ``xp`` unused since
    the operation is pure indexing."""
    x1, y1, x2, y2 = get_enclosing_points(x, xp_mesh, fp)
    return y1


def interp_lin_lin(x, xp_mesh, fp, xp=None):
    """Linear-Linear interpolation. Backend-agnostic; ``xp`` unused
    since the operation is pure arithmetic."""
    x1, y1, x2, y2 = get_enclosing_points(x, xp_mesh, fp)
    return y1 + (x-x1)*(y2-y1)/(x2-x1)


_INTERP_LOG_SMALL = 1.0e-38


def interp_lin_log(x, xp_mesh, fp, xp=None):
    """Linear-Logarithmic interpolation.

    Clamps ``x1 == 0`` to a small positive value before taking
    logs, matching the Fortran ``yintp`` reference behaviour
    (endf6.f90 line 2044). This surfaces on ENDF files whose x-mesh
    starts at zero (e.g. MF6 LAW=7 outgoing-energy tabulations at
    Ep=0) when INT=3/5 is applied.

    Backend-agnostic: ``xp=None`` (default) resolves to numpy.
    """
    xp = _resolve_xp(xp)
    x1, y1, x2, y2 = get_enclosing_points(x, xp_mesh, fp)
    x1 = xp.where(x1 == 0.0, _INTERP_LOG_SMALL, x1)
    return y1 + xp.log(x/x1)*(y2-y1)/xp.log(x2/x1)


def interp_log_lin(x, xp_mesh, fp, xp=None):
    """Logarithmic-Linear interpolation.

    Clamps ``y1 == 0`` to a small positive value before taking
    logs, matching Fortran ``yintp`` (endf6.f90 line 2049).

    Backend-agnostic: ``xp=None`` (default) resolves to numpy.
    """
    xp = _resolve_xp(xp)
    x1, y1, x2, y2 = get_enclosing_points(x, xp_mesh, fp)
    y1 = xp.where(y1 == 0.0, _INTERP_LOG_SMALL, y1)
    return y1*xp.exp((x-x1)*xp.log(y2/y1)/(x2-x1))


def interp_log_log(x, xp_mesh, fp, xp=None):
    """Logarithmic-Logarithmic interpolation.

    Clamps both ``x1 == 0`` and ``y1 == 0`` to a small positive
    value before taking logs, matching Fortran ``yintp``
    (endf6.f90 lines 2054-2055).

    Backend-agnostic: ``xp=None`` (default) resolves to numpy.
    """
    xp = _resolve_xp(xp)
    x1, y1, x2, y2 = get_enclosing_points(x, xp_mesh, fp)
    x1 = xp.where(x1 == 0.0, _INTERP_LOG_SMALL, x1)
    y1 = xp.where(y1 == 0.0, _INTERP_LOG_SMALL, y1)
    return y1*xp.exp(xp.log(x/x1)*xp.log(y2/y1)/xp.log(x2/x1))


def _interp_two_point_columns(x, x1, x2, y1, y2, interp_type, xp=None):
    """Closed-form 2-point interpolation applied column-wise.

    `x, x1, x2` are scalars; `y1, y2` are 1D arrays of the same
    length K holding the two bracketing rows of a matrix whose K
    columns we interpolate along. Returns the interpolated row,
    shape `(K,)`. Used by `interp_tab2` in place of a Python loop
    over K that called `interp` once per column (issue #46).

    The five interpolation types collapse for two points to the
    same closed form each scheme uses in `interp_*`, so this
    function is exact-equivalent numerically, not just
    approximately.

    For the log-based schemes (INT=3/4/5) the closed form would
    NaN out at ``x1 == 0`` (log-in-x) or where ``y1 == 0``
    element-wise (log-in-y). Matching the Fortran ``yintp``
    reference (endf6.f90 line 2011), we clamp those degenerate
    values to ``1e-38`` before taking the log; on the (rare)
    columns where this bites, the interpolated value is
    numerically dominated by ``y1`` / a small perturbation of it,
    which is the conventional evaluator behaviour.

    Backend-agnostic: pass ``xp=array_ns.get_backend(name)`` to
    dispatch. ``xp=None`` (the default) resolves to numpy.
    """
    xp = _resolve_xp(xp)
    _small = 1.0e-38
    if interp_type == 1:      # histogram / constant
        # `broadcast_to` gives a read-only view on numpy; we return
        # it as-is since callers only read the result. JAX
        # `broadcast_to` returns a regular jnp array.
        return xp.broadcast_to(y1, y1.shape)
    if interp_type == 2:      # lin-lin
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    if interp_type == 3:      # lin-log (log in x)
        x1_safe = x1 if x1 != 0.0 else _small
        return y1 + xp.log(x / x1_safe) * (y2 - y1) / xp.log(x2 / x1_safe)
    if interp_type == 4:      # log-lin (log in y)
        y1_safe = xp.where(y1 == 0.0, _small, y1)
        return y1_safe * xp.exp((x - x1) * xp.log(y2 / y1_safe) / (x2 - x1))
    if interp_type == 5:      # log-log
        x1_safe = x1 if x1 != 0.0 else _small
        y1_safe = xp.where(y1 == 0.0, _small, y1)
        return y1_safe * xp.exp(
            xp.log(x / x1_safe) * xp.log(y2 / y1_safe) / xp.log(x2 / x1_safe)
        )
    raise TypeError(
        f'interpolation scheme (INT={interp_type}) not implemented'
    )


_INTERP_DISPATCH = {
    1: interp_const,
    2: interp_lin_lin,
    3: interp_lin_log,
    4: interp_log_lin,
    5: interp_log_log,
}


def interp(x, xp_mesh, fp, interp_type, outside_value=None, xp=None):
    """Interpolation using various schemes.

    Backend-agnostic: ``xp=None`` (the default) resolves to numpy;
    passing ``xp=array_ns.get_backend('jax')`` keeps ``fp`` tracers
    alive through the arithmetic (issue #154). Query ``x`` and
    mesh ``xp_mesh`` are treated as numpy arrays for the panel
    lookup (index-selection is inherently nondifferentiable).
    """
    xp = _resolve_xp(xp)
    # TODO: Here we provisionally let NaN values pass through the
    #       program logic for comparison with the Fortran routines.
    #       However, eventually no NaN values should appear in x.
    x_np = np.asarray(x)
    xp_mesh_np = np.asarray(xp_mesh)
    # Normalise fp so downstream advanced indexing works regardless
    # of whether the caller passed a Python list, numpy array, or
    # JAX tracer (see `endf_interp1d` for the same rationale).
    fp = xp.asarray(fp)
    is_inside = (
        (x_np >= np.min(xp_mesh_np)) & (x_np <= np.max(xp_mesh_np))
    ) | np.isnan(x_np)
    if not np.all(is_inside) and outside_value is None:
        raise ValueError('some `x` value outside mesh given by `xp`')
    xi = x_np[is_inside]
    scheme = _INTERP_DISPATCH.get(int(interp_type))
    if scheme is None:
        raise TypeError(
            f'interpolation scheme (INT={interp_type}) not implemented'
        )
    fi = scheme(xi, xp_mesh_np, fp, xp=xp)
    if np.all(is_inside):
        return fi
    return _scatter_inside(fi, is_inside, x_np.shape, outside_value, xp)


def _scatter_inside(fi, is_inside, full_shape, outside_value, xp):
    """Scatter the inside-mesh values `fi` into a full-shape array
    filled with `outside_value` at the outside positions. Numpy
    uses in-place assignment; JAX uses ``.at[].set()`` since jnp
    arrays are immutable."""
    is_inside_np = np.asarray(is_inside)
    if xp.name == 'jax':
        # Build the full array on jax with the outside fill, then
        # scatter fi into the inside slots.
        full = xp.full(full_shape, outside_value, dtype=fi.dtype)
        return full.at[is_inside_np].set(fi)
    full = np.empty(full_shape, dtype=float)
    full[~is_inside_np] = outside_value
    full[is_inside_np] = np.asarray(fi)
    return xp.asarray(full)


def _endf_interp1d_traced_x(
    x, xp_mesh, fp, int_arr, nbt_arr, outside_value, xp,
):
    """JAX-friendly ``endf_interp1d`` variant that supports tracer
    ``x`` (issue #190). Computes all 5 INT schemes element-wise on
    the full ``x`` array and selects the per-point result based on
    which INT region each point's bracket falls into. Slower than
    the numpy per-region loop (5x scheme evaluations per point) but
    JAX-native and single-graph.

    ``xp_mesh``, ``int_arr``, ``nbt_arr`` are file-side data (small,
    concrete) and stay numpy for the panel-index precomputation;
    only the arithmetic on ``x`` and ``fp`` runs through ``xp``.
    """
    _small = 1.0e-38
    x = xp.asarray(x)
    fp = xp.asarray(fp)
    # No ``treat_duplicates`` here: the ``dx_safe`` line below already
    # handles zero-width brackets from duplicated mesh values (ENDF-6
    # encodes a step discontinuity as two adjacent equal-x mesh points).
    # Skipping dedup also lets a JAX-tracer mesh flow through for
    # mesh-knot autodiff, since ``np.asarray(tracer)`` would raise
    # inside the numpy-only dedup step.
    xp_mesh_xp = xp.asarray(xp_mesh)
    n_mesh = int(xp_mesh_xp.shape[0])
    if n_mesh < 2:
        # Degenerate: too few mesh points for any bracket. Return
        # outside_value everywhere (or zeros if outside_value is None).
        fill = 0.0 if outside_value is None else float(outside_value)
        return xp.full(x.shape, fill, dtype=x.dtype)

    # Per-mesh-point INT law (length n_mesh). ``convert_interp_repr``
    # assigns the shared boundary mesh point to the LOWER region;
    # the ENDF-6 convention (see endf_interp1d numpy loop) is that
    # the bracket starting at a shared boundary belongs to the
    # UPPER region, so we look up bracket k's INT via the upper
    # endpoint's mesh-point INT: ``int_per_mesh_point[k + 1]``.
    int_per_mesh_point_np = convert_interp_repr(
        np.asarray(int_arr), np.asarray(nbt_arr),
    )
    int_per_mesh_point_xp = xp.asarray(int_per_mesh_point_np)

    # Bracket each x point in the mesh. side='right' means idx = i
    # where xp_mesh[i-1] <= x < xp_mesh[i]; we shift to i-1 so idx is
    # the lower bracket, then clip to [0, n_mesh - 2] for safe gather.
    idx = xp.searchsorted(xp_mesh_xp, x, side='right') - 1
    idx = xp.clip(idx, 0, n_mesh - 2)

    x1 = xp.take(xp_mesh_xp, idx)
    x2 = xp.take(xp_mesh_xp, idx + 1)
    y1 = xp.take(fp, idx)
    y2 = xp.take(fp, idx + 1)

    # Safe denominators and log arguments so the branches we don't
    # select don't propagate NaN.
    dx = x2 - x1
    dx_safe = xp.where(dx == 0.0, 1.0, dx)
    x1_pos = xp.where(x1 > 0.0, x1, _small)
    x2_pos = xp.where(x2 > 0.0, x2, _small)
    y1_pos = xp.where(y1 > 0.0, y1, _small)
    y2_pos = xp.where(y2 > 0.0, y2, _small)
    x_pos = xp.where(x > 0.0, x, _small)

    r1 = y1                                          # INT=1 histogram
    r2 = y1 + (x - x1) * (y2 - y1) / dx_safe         # INT=2 lin-lin
    # INT=3 lin-log (log in x)
    log_x_ratio = xp.log(x_pos / x1_pos)
    log_x2_ratio = xp.log(x2_pos / x1_pos)
    log_x2_ratio_safe = xp.where(log_x2_ratio == 0.0, 1.0, log_x2_ratio)
    r3 = y1 + log_x_ratio * (y2 - y1) / log_x2_ratio_safe
    # INT=4 log-lin (log in y)
    log_y_ratio = xp.log(y2_pos / y1_pos)
    r4 = y1_pos * xp.exp((x - x1) * log_y_ratio / dx_safe)
    # INT=5 log-log
    r5 = y1_pos * xp.exp(log_x_ratio * log_y_ratio / log_x2_ratio_safe)

    # Per-point INT law: read from the bracket's UPPER endpoint
    # (matches endf_interp1d's "boundary belongs to upper region"
    # convention). idx is the LOWER bracket, so lookup at idx + 1.
    interp_type = xp.take(int_per_mesh_point_xp, idx + 1)
    result = xp.where(
        interp_type == 1, r1,
        xp.where(
            interp_type == 2, r2,
            xp.where(
                interp_type == 3, r3,
                xp.where(interp_type == 4, r4, r5),
            ),
        ),
    )

    # Out-of-mesh handling. Under trace we cannot raise on missing
    # outside_value with off-mesh x; the caller must pass one if
    # traced x might leave the mesh. When outside_value=None we
    # silently pass the clamped-bracket result (matches jax semantics
    # of "no error inside a trace").
    is_inside = (x >= xp_mesh_xp[0]) & (x <= xp_mesh_xp[-1])
    if outside_value is not None:
        result = xp.where(is_inside, result, outside_value)
    return result


def endf_interp1d(x, xp_mesh, fp, int_arr, nbt_arr, outside_value=None, xp=None):
    """Piecewise ENDF-6 TAB1 interpolation across INT regions.

    Backend-agnostic: ``xp=None`` (the default) resolves to numpy;
    passing ``xp=array_ns.get_backend('jax')`` keeps ``fp`` tracers
    alive through the arithmetic. ``treat_duplicates`` only operates
    on the mesh (concrete file-side data), so leaving it on numpy is
    safe for the gradient chain; ``find_interval`` for panel
    indexing is inherently non-differentiable and stays numpy.
    """
    xp = _resolve_xp(xp)
    check_int_nbt(int_arr, nbt_arr)
    # Under xp=jax, route through the traced-x path so query-axis
    # tracers propagate to jax.grad (issue #190). The traced path
    # is 5x slower per point than the numpy per-region loop below
    # (evaluates every INT scheme and selects) but is the only
    # form that handles a tracer x. Concrete jax arrays go through
    # the same path -- fine for autodiff, mildly wasteful for
    # non-autodiff jax use; callers who need raw jax throughput
    # without autodiff can pass xp=numpy.
    if xp.name == 'jax':
        return _endf_interp1d_traced_x(
            x, xp_mesh, fp, int_arr, nbt_arr, outside_value, xp,
        )
    x = np.asarray(x)
    # Normalise fp to an xp-native array so downstream advanced
    # indexing (`fp[idcs]` inside `get_enclosing_points`) works
    # regardless of whether the caller passed a Python list, a
    # numpy array, or a JAX tracer. `xp.asarray` on a JAX tracer
    # is a no-op that preserves the tracer.
    fp = xp.asarray(fp)
    # Rebind `xp_mesh` to a deduplicated copy rather than mutating
    # the caller's array in place. `treat_duplicates` perturbs
    # repeated mesh values by a relative epsilon so `searchsorted`
    # can distinguish them; if we did that in place, any caller
    # that passes a long-lived array (e.g. cached from the ENDF
    # dict) would have its mesh silently modified, and a second
    # call on the same array would perturb it again (issue #49).
    xp_mesh = treat_duplicates(np.asarray(xp_mesh))
    is_inside = (
        (x >= np.min(xp_mesh)) & (x <= np.max(xp_mesh))
    ) | np.isnan(x)
    if not np.all(is_inside) and outside_value is None:
        raise ValueError('some `x` value outside mesh given by `xp`')
    xi = x[is_inside]
    idcs = find_interval(xp_mesh, xi)
    # Per ENDF-6 TAB1 semantics, consecutive interpolation regions
    # share their boundary row: region n covers rows NBT(n-1)..NBT(n).
    # In Python 0-indexing, region 0 is xp_mesh[0:NBT[0]] and region
    # k>0 is xp_mesh[NBT[k-1]-1:NBT[k]] (overlapping the boundary).
    # The interval starting at the shared row belongs to the upper
    # region.
    fi_pieces = []
    piece_positions = []
    first_idx = 0
    nregions = len(int_arr)
    for i in range(nregions):
        last_idx = int(nbt_arr[i])
        is_last = (i == nregions - 1)
        interp_type = int(int_arr[i])
        cur_xp = xp_mesh[first_idx:last_idx]
        cur_fp = fp[first_idx:last_idx]  # slice preserves jax tracers
        upper = last_idx if is_last else last_idx - 1
        is_in_range = (idcs >= first_idx) & (idcs < upper)
        cur_x = xi[is_in_range]
        if cur_x.size == 0:
            first_idx = last_idx - 1
            continue
        cur_fi = interp(
            cur_x, cur_xp, cur_fp, interp_type, outside_value, xp=xp,
        )
        fi_pieces.append(cur_fi)
        piece_positions.append(np.where(is_in_range)[0])
        first_idx = last_idx - 1

    # Assemble in the original xi order.
    if fi_pieces:
        all_positions = np.concatenate(piece_positions)
        fi_concat = xp.concatenate(
            [xp.asarray(p).reshape(-1) for p in fi_pieces], axis=0,
        )
        order = np.argsort(all_positions)
        fi = fi_concat[np.asarray(order)]
    else:
        fi = xp.zeros(xi.shape, dtype=xp.float64)

    if np.all(is_inside):
        return fi
    return _scatter_inside(fi, is_inside, x.shape, outside_value, xp)


def interp_legendre_coeffs(x, xp_mesh, coeffs, int_arr, nbt_arr,
                            outside_value=None, xp=None):
    """Per-column interpolation of a `(NE_mesh, L+1)` Legendre
    coefficient array onto a query energy vector.

    Backend-agnostic: ``xp=None`` (default) is numpy; JAX tracers
    in ``coeffs`` propagate through the column-wise
    :func:`endf_interp1d` calls (see issue #154). ``x`` may also
    be a tracer under ``xp=jax``: `endf_interp1d` routes tracer
    queries through its ``_endf_interp1d_traced_x`` fast path
    (PR #192). Issue #201.
    """
    xp = _resolve_xp(xp)
    # Do NOT materialise `x` via ``np.asarray`` here: under
    # ``xp=jax`` this call is on a tracer that must survive into
    # ``endf_interp1d``'s traced-x path (issue #201).
    ncols = int(coeffs.shape[1])
    cols = []
    for i in range(ncols):
        cols.append(endf_interp1d(
            x, xp_mesh, coeffs[:, i], int_arr, nbt_arr, outside_value, xp=xp,
        ))
    return xp.stack(cols, axis=-1)


def _eval_legendre_series(coeffs, mu, xp):
    """Evaluate a Legendre series `sum_L coeffs[..., L] * P_L(mu)`
    via the Bonnet recurrence.

    ``coeffs`` has shape ``(..., L+1)`` (coefficient axis last;
    ``L+1`` = number of coefficients including the constant term
    ``P_0``). ``mu`` has shape ``(..., M)`` broadcastable with
    ``coeffs[..., 0]``. Returns shape
    ``broadcast(coeffs[..., 0], mu)``, one Legendre value per (row,
    mu) pair.

    Recurrence: ``P_0(mu) = 1``, ``P_1(mu) = mu``,
    ``(n+1) P_{n+1} = (2n+1) mu P_n - n P_{n-1}``.

    Backend-agnostic: replaces the numpy-only
    ``numpy.polynomial.legendre.Legendre(coeffs)(mu)`` call so the
    same routine runs on JAX; ``jax.grad`` flows through both the
    coefficient axis and the ``mu`` argument.
    """
    n_coeffs = int(coeffs.shape[-1])
    if n_coeffs == 0:
        return xp.zeros(mu.shape, dtype=mu.dtype)
    # Broadcast coefficients against mu on the "M" (query) axis
    # by adding a trailing axis to each coeff row, then removing
    # after the sum.
    coeffs_bc = coeffs[..., None]                # (..., L+1, 1) so we
    # can pull each c_L broadcast against mu (..., M) after we add a
    # leading axis to mu below.
    # We accumulate the series in `total`. To avoid conditionals on
    # array shapes, we materialise P_prev and P_curr with the same
    # shape as `mu` from the start.
    ones = xp.ones_like(mu)
    P_prev = ones                                # P_0(mu) = 1
    total = coeffs_bc[..., 0, :] * P_prev
    if n_coeffs == 1:
        return total
    P_curr = mu                                  # P_1(mu) = mu
    total = total + coeffs_bc[..., 1, :] * P_curr
    for L in range(1, n_coeffs - 1):
        P_next = ((2 * L + 1) * mu * P_curr - L * P_prev) / (L + 1)
        total = total + coeffs_bc[..., L + 1, :] * P_next
        P_prev = P_curr
        P_curr = P_next
    return total


def evaluate_interp_legendre_polynomials(
    x, mu, xp_mesh, coeffs, int_arr, nbt_arr, outside_value=None, xp=None,
):
    """Evaluate Legendre polynomials of degree `coeffs.shape[1]-1`
    at each `mu` after interpolating the per-degree coefficient
    across the incident-energy axis `xp_mesh`.

    `outside_value=None` (default) raises when any `x` value is
    outside `xp_mesh` -- callers that pre-filter `x` to the
    tabulated range (via the `pad_outside_*` decorators used by
    the MF4 evaluators) rely on this to catch mistakes. Callers
    that walk per-photon-line tables whose own `xp_mesh` is
    narrower than the caller's `x` (MF14 discrete lines with
    per-line `E` mesh -- issue #81) pass `outside_value=0.0` so
    the out-of-mesh interpolated coefficients zero out and the
    Legendre evaluation gives 0 at those `x` (a photon line only
    contributes at Ein values it was tabulated at).

    Backend-agnostic (issue #154): pass
    ``xp=array_ns.get_backend(name)`` to dispatch the whole
    reconstruction onto that backend. JAX tracers in ``coeffs``
    now propagate through the per-degree coefficient interpolation
    (:func:`interp_legendre_coeffs`) and the Legendre evaluation
    (:func:`_eval_legendre_series`) so ``jax.grad`` reaches all
    the way back to the file's Legendre coefficients.
    ``xp=None`` (the default) resolves to numpy, keeping every
    pre-port caller on the same bit-identical path.

    Note: the mesh parameter is named ``xp_mesh`` (was ``xp``
    pre-port) to avoid the collision with the backend adapter
    also conventionally named ``xp``. Positional callers are
    unaffected since the argument position is unchanged.

    ``x`` and ``mu`` may be tracers under ``xp=jax`` (issue #201):
    the per-column coefficient interpolation routes through
    ``endf_interp1d``'s traced-x path (PR #192), and the Legendre
    series evaluation is pure arithmetic on ``mu``.
    """
    xp = _resolve_xp(xp)
    # Keep ``x`` and ``mu`` xp-native so JAX tracers survive into
    # the downstream interp and Legendre evaluation (issue #201).
    # Shape inspection (``.ndim``, ``.shape``) is safe on tracers
    # because shapes are concrete under trace; only value reads
    # would fail.
    x = xp.asarray(x)
    mu = xp.asarray(mu)
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    if mu.shape[0] == 1:
        mu = xp.broadcast_to(mu, (int(x.shape[0]), mu.shape[1]))
    # Backend-agnostic per-degree coefficient interpolation over the
    # file's energy mesh. Tracers in `coeffs` and in `x` propagate
    # through.
    interp_coeffs = interp_legendre_coeffs(
        x, xp_mesh, coeffs, int_arr, nbt_arr, outside_value, xp=xp,
    )
    # Caller passes the FULL coefficient array including the ``a_0``
    # constant term with any (2L+1)/2 normalisation factor already
    # applied (see e.g.
    # :func:`mf4_interpretation._convert_legendre_to_numpy_array`),
    # so ``_eval_legendre_series`` evaluates
    # ``sum_L coeffs[..., L] * P_L(mu)`` directly.
    return _eval_legendre_series(interp_coeffs, mu, xp)


def interp_tab1(x, tab1, xp_name, fp_name, outside_value=None, xp=None):
    """Interpolate a TAB1 record (`{xp_name: mesh, fp_name: fp,
    'INT': ..., 'NBT': ...}` dict-of-arrays layout) at query `x`.

    Backend-agnostic: ``xp=None`` (default) is numpy; passing a
    JAX backend preserves tracers stored at either ``tab1[fp_name]``
    or ``tab1[xp_name]`` (fp-side or mesh-side autodiff). ``xp.asarray``
    on a list containing a JAX tracer scalar preserves the tracer's
    functional dependency. INT / NBT stay numpy because they are
    region-descriptor integers with no autodiff meaning.
    """
    xp = _resolve_xp(xp)
    x_mesh = xp.asarray(tab1[xp_name], dtype=xp.float64)
    f_mesh = xp.asarray(tab1[fp_name], dtype=xp.float64)
    int_arr = np.asarray(tab1['INT'], dtype=int)
    nbt_arr = np.asarray(tab1['NBT'], dtype=int)
    return endf_interp1d(
        x, x_mesh, f_mesh, int_arr, nbt_arr, outside_value, xp=xp,
    )


def determine_unit_base_coordinates(
    x, y, x1, x2, y1_min, y1_max, y2_min, y2_max
):
    """Map a target ``(x, y)`` into unit-base coordinates spanning
    two panels ``(x1, y1_min..y1_max)`` and ``(x2, y2_min..y2_max)``,
    returning both the projected ``cur_y1`` / ``cur_y2`` on each
    panel's native y-axis and the Jacobians for the amplitude
    rescaling. Pure arithmetic; works on any array backend as long
    as the operators (``+``, ``-``, ``*``, ``/``) are backend-native
    -- no explicit ``xp`` needed since no ``np.<fn>`` calls appear.
    """
    y1_delta = y1_max - y1_min
    y2_delta = y2_max - y2_min
    rx = (x - x1) / (x2 - x1)
    y_lo = y1_min + rx * (y2_min - y1_min)
    y_hi = y1_max + rx * (y2_max - y1_max)
    y_delta = y_hi - y_lo
    ry = (y - y_lo ) / y_delta
    cur_y1 = y1_min + ry * y1_delta
    cur_y2 = y2_min + ry * y2_delta
    y_delta = y_hi - y_lo
    jac1 = y1_delta / y_delta
    jac2 = y2_delta / y_delta
    return cur_y1, cur_y2, jac1, jac2


def _interp_tab2_traced_x(
    x, y, xp_mesh, int_arr, nbt_arr, tab1_records, yp_name, fp_name,
    outside_value, xp,
):
    """JAX-friendly ``interp_tab2`` variant that supports tracer
    ``x`` (issue #201). Loops over the concrete panel index (small,
    numpy-side), computes each panel's per-y inner-interp
    contribution once, evaluates the outer 2-point interp on the
    full tracer ``x`` array via broadcasting, and selects the
    correct panel's contribution per query x with ``xp.where``.

    Only the non-unit-base outer interpolation types (INT=1..5) are
    supported. Unit-base (INT=21..25) is used mostly in
    Ep-differential contexts (MF6 LAW=7) where the y range varies
    per panel; those callers still hit the numpy per-x loop.

    Handles both shared inner y (``y.shape[0] == 1``, e.g. a fixed
    mu grid across all incident energies) and per-x inner y
    (``y.shape == (n_x, n_y)``, e.g. LAW=2 LCT=2 where the CM mu
    depends on Ein via ``convert_angcos_to_cmsys``).
    """
    x = xp.asarray(x)
    y_xp = xp.asarray(y)
    if y_xp.ndim == 1:
        y_xp = y_xp.reshape(1, -1)

    xp_mesh_np = np.asarray(xp_mesh)
    n_panels = int(xp_mesh_np.shape[0]) - 1
    if n_panels < 1:
        raise ValueError('interp_tab2 requires >= 2 mesh points')
    interp_arr_np = convert_interp_repr(int_arr, nbt_arr)
    n_x = int(x.shape[0])
    n_y = int(y_xp.shape[1])
    per_x_y = int(y_xp.shape[0]) != 1                           # True if y varies per x
    x_col = x.reshape(-1, 1)                                    # (n_x, 1)

    _small = 1.0e-38

    def _outer_2pt(interp_type, x_col, x1, x2, y1, y2):
        """Return an (n_x, n_y) block for one panel, given the
        panel's inner-interp values ``y1``, ``y2`` (both (n_y,))
        and scalars ``x1``, ``x2``."""
        y1_row = y1.reshape(1, -1)                              # (1, n_y)
        y2_row = y2.reshape(1, -1)                              # (1, n_y)
        if interp_type == 1:
            return xp.broadcast_to(y1_row, (n_x, n_y))
        if interp_type == 2:
            return y1_row + (x_col - x1) * (y2_row - y1_row) / (x2 - x1)
        if interp_type == 3:
            x1s = x1 if x1 != 0.0 else _small
            x2s = x2 if x2 != 0.0 else _small
            # x may be zero or negative under trace; clamp for log
            # safety on the branches we do not select. Selection is
            # by is_in_panel; only the correct panel's values reach
            # the output.
            x_safe = xp.where(x_col > 0.0, x_col, _small)
            return y1_row + xp.log(x_safe / x1s) * (
                y2_row - y1_row
            ) / xp.log(x2s / x1s)
        if interp_type == 4:
            y1_safe = xp.where(y1_row == 0.0, _small, y1_row)
            return y1_safe * xp.exp(
                (x_col - x1) * xp.log(y2_row / y1_safe) / (x2 - x1)
            )
        if interp_type == 5:
            x1s = x1 if x1 != 0.0 else _small
            x2s = x2 if x2 != 0.0 else _small
            x_safe = xp.where(x_col > 0.0, x_col, _small)
            y1_safe = xp.where(y1_row == 0.0, _small, y1_row)
            return y1_safe * xp.exp(
                xp.log(x_safe / x1s) * xp.log(y2_row / y1_safe)
                / xp.log(x2s / x1s)
            )
        raise NotImplementedError(
            f'interp_tab2 traced-x supports INT=1..5 only; got '
            f'INT={interp_type}. Unit-base (21..25) or unknown '
            f'schemes need the numpy path.'
        )

    def _outer_2pt_row(interp_type, x_col, x1, x2, y1_row, y2_row):
        """Per-x-y outer 2-point interp when y1/y2 already have
        shape (n_x, n_y) (per-x y case)."""
        if interp_type == 1:
            return y1_row
        if interp_type == 2:
            return y1_row + (x_col - x1) * (y2_row - y1_row) / (x2 - x1)
        if interp_type == 3:
            x1s = x1 if x1 != 0.0 else _small
            x2s = x2 if x2 != 0.0 else _small
            x_safe = xp.where(x_col > 0.0, x_col, _small)
            return y1_row + xp.log(x_safe / x1s) * (
                y2_row - y1_row
            ) / xp.log(x2s / x1s)
        if interp_type == 4:
            y1_safe = xp.where(y1_row == 0.0, _small, y1_row)
            return y1_safe * xp.exp(
                (x_col - x1) * xp.log(y2_row / y1_safe) / (x2 - x1)
            )
        if interp_type == 5:
            x1s = x1 if x1 != 0.0 else _small
            x2s = x2 if x2 != 0.0 else _small
            x_safe = xp.where(x_col > 0.0, x_col, _small)
            y1_safe = xp.where(y1_row == 0.0, _small, y1_row)
            return y1_safe * xp.exp(
                xp.log(x_safe / x1s) * xp.log(y2_row / y1_safe)
                / xp.log(x2s / x1s)
            )
        raise NotImplementedError(
            f'interp_tab2 traced-x supports INT=1..5 only; got '
            f'INT={interp_type}.'
        )

    # Assemble by masking each panel's block into the result.
    result = xp.zeros((n_x, n_y), dtype=xp.float64)
    mesh_lo = float(xp_mesh_np[0])
    mesh_hi = float(xp_mesh_np[-1])
    for p in range(n_panels):
        x1 = float(xp_mesh_np[p])
        x2 = float(xp_mesh_np[p + 1])
        interp_type = int(interp_arr_np[p])
        curtab1 = tab1_records[p]
        curtab2 = tab1_records[p + 1]
        if 21 <= interp_type <= 25:
            # Unit-base traced-x path (roadmap #198 / issue #220
            # PR 4): match the numpy branch's
            # ``determine_unit_base_coordinates`` transform with
            # per-x scalar broadcasting, then reuse the per-x-y
            # outer 2pt interp with the derived non-unit-base code
            # ``interp_type - 20``. Per-panel ``y_min`` / ``y_max``
            # are static file constants (materialised to Python
            # floats) so the unit-base arithmetic stays inside the
            # xp graph on x, y and the inner ``interp_tab1``
            # samples. LAW=7 is the primary use.
            y1_np = np.asarray(curtab1[yp_name], dtype=float)
            y2_np = np.asarray(curtab2[yp_name], dtype=float)
            y1_min = float(y1_np.min())
            y1_max = float(y1_np.max())
            y2_min = float(y2_np.min())
            y2_max = float(y2_np.max())
            y1_delta = y1_max - y1_min
            y2_delta = y2_max - y2_min
            rx = (x_col - x1) / (x2 - x1)                       # (n_x, 1)
            y_lo = y1_min + rx * (y2_min - y1_min)              # (n_x, 1)
            y_hi = y1_max + rx * (y2_max - y1_max)              # (n_x, 1)
            y_delta_col = y_hi - y_lo                           # (n_x, 1)
            # Safe divide guard for degenerate panels where
            # ``y_delta_col`` collapses. In-panel mask below will
            # discard these rows anyway; the guard just keeps the
            # arithmetic finite everywhere for jax.
            _safe = xp.where(y_delta_col != 0.0, y_delta_col, 1.0)
            if per_x_y:
                y_arr = y_xp                                    # (n_x, n_y)
            else:
                y_arr = xp.broadcast_to(
                    y_xp[0].reshape(1, -1), (n_x, n_y),
                )
            ry = (y_arr - y_lo) / _safe                         # (n_x, n_y)
            cur_y1 = y1_min + ry * y1_delta                     # (n_x, n_y)
            cur_y2 = y2_min + ry * y2_delta                     # (n_x, n_y)
            jac1 = y1_delta / _safe                             # (n_x, 1)
            jac2 = y2_delta / _safe                             # (n_x, 1)
            f1_flat = interp_tab1(
                cur_y1.reshape(-1), curtab1, yp_name, fp_name,
                outside_value, xp=xp,
            )
            f2_flat = interp_tab1(
                cur_y2.reshape(-1), curtab2, yp_name, fp_name,
                outside_value, xp=xp,
            )
            f1_row = f1_flat.reshape(n_x, n_y) * jac1           # (n_x, n_y)
            f2_row = f2_flat.reshape(n_x, n_y) * jac2
            eff_interp = interp_type - 20
            block = _outer_2pt_row(
                eff_interp, x_col, x1, x2, f1_row, f2_row,
            )
        elif per_x_y:
            # y varies per x (e.g. LAW=2 LCT=2 with per-Ein CM mu).
            # Flatten to a single 1-D interp call, then reshape.
            y_flat = y_xp.reshape(-1)                          # (n_x * n_y,)
            f1_flat = interp_tab1(
                y_flat, curtab1, yp_name, fp_name, outside_value,
                xp=xp,
            )
            f2_flat = interp_tab1(
                y_flat, curtab2, yp_name, fp_name, outside_value,
                xp=xp,
            )
            f1_row = f1_flat.reshape(n_x, n_y)
            f2_row = f2_flat.reshape(n_x, n_y)
            block = _outer_2pt_row(interp_type, x_col, x1, x2, f1_row, f2_row)
        else:
            # Shared y row across all x.
            y_row = y_xp[0]                                    # (n_y,)
            f1 = interp_tab1(
                y_row, curtab1, yp_name, fp_name, outside_value, xp=xp,
            )
            f2 = interp_tab1(
                y_row, curtab2, yp_name, fp_name, outside_value, xp=xp,
            )
            block = _outer_2pt(interp_type, x_col, x1, x2, f1, f2)
        # ENDF convention: the upper endpoint of a panel belongs to
        # the upper region, except the final panel includes its top.
        if p < n_panels - 1:
            in_panel = (x >= x1) & (x < x2)
        else:
            in_panel = (x >= x1) & (x <= x2)
        result = xp.where(in_panel.reshape(-1, 1), block, result)

    if outside_value is not None:
        is_inside = (x >= mesh_lo) & (x <= mesh_hi)
        fill = xp.full((n_x, n_y), outside_value, dtype=xp.float64)
        result = xp.where(is_inside.reshape(-1, 1), result, fill)
    return result


def interp_tab2(
    x, y, xp_mesh, int_arr, nbt_arr, tab1_records, yp_name, fp_name,
    outside_value=None, xp=None,
):
    """Perform 2d interpolation using TAB2/TAB1 record sequence.

    Parameters
    ----------
    x : numpy.ndarray
        Target x value
    y : numpy.ndarray
        Target y value
    xp_mesh : numpy.ndarray
        Mesh of x-values (renamed from ``xp`` pre-port to avoid the
        collision with the backend adapter also called ``xp``).
    int_arr : numpy.ndarray
        Interpolation types for x-segments
    nbt_arr : numpy.ndarray
        Definition of x-segments
    tab1_records : list
        List of ENDF TAB1 records
    yp_name : str
        Key name for y-mesh in all TAB1 records
    fp_name : str
        Key name for mesh of associated function values
        in TAB1 record
    outside_value : bool
        Returned value for point with x-value outside x-mesh limits.
        If `None`, a `ValueError` is raised if outside points encountered.
    xp : optional
        Array-namespace adapter from
        :func:`endf_userpy.primitives.array_ns.get_backend`. Defaults
        to numpy. The per-panel outer interpolation, the unit-base
        Jacobian arithmetic, and the per-panel inner
        :func:`interp_tab1` evaluation all run through this backend,
        so tracers stored in the per-panel ``tab1_records[i][fp_name]``
        list survive into the outer two-point interp.

    Returns
    -------
    array
        A two-dimensional array (numpy or backend-native depending
        on ``xp``) with the interpolated function values. The value
        in the i-th row and j-th column corresponds to the function
        value for ``x[i]`` and ``y[j]``.
    """
    xp = _resolve_xp(xp)
    # Under xp=jax, route tracer queries through the traced-x fast
    # path (issue #201 PR-B) so ``jax.grad`` wrt ``x`` propagates.
    # The traced variant supports the standard non-unit-base outer
    # interp types (INT=1..5) with a shared y grid, which covers
    # MF4 LTT=2/3 and MF6 LAW=2 LANG=12/14. Unit-base (21..25) and
    # per-x y still take the numpy path.
    if xp.name == 'jax':
        try:
            return _interp_tab2_traced_x(
                x, y, xp_mesh, int_arr, nbt_arr, tab1_records,
                yp_name, fp_name, outside_value, xp,
            )
        except NotImplementedError:
            # Fall through to numpy path for unit-base / per-x y.
            pass
    # Convert mesh + y to numpy for the file-side loop (dict-of-
    # records access and per-panel unit-base bounds computation stay
    # on numpy). The per-row row-vector of outer-interp results is
    # collected xp-native and stacked at the end.
    y_np = np.asarray(y)
    if y_np.ndim == 1:
        y_np = y_np.reshape(1, -1)
    xp_mesh_np = np.asarray(xp_mesh)
    x_np = np.asarray(x)

    if outside_value is not None:
        is_inside = (x_np >= np.min(xp_mesh_np)) & (x_np <= np.max(xp_mesh_np))
        any_outside = not np.all(is_inside)
        x_orig = x_np
        x_np = x_np[is_inside]

    idcs = find_interval(xp_mesh_np, x_np)
    interp_arr = convert_interp_repr(int_arr, nbt_arr)
    rows = []
    for i, idx in enumerate(idcs):
        cur_x = float(x_np[i])
        x1 = float(xp_mesh_np[idx])
        x2 = float(xp_mesh_np[idx + 1])
        cur_y = y_np[0, :] if y_np.shape[0] == 1 else y_np[i, :]
        interp_type = int(interp_arr[idx])
        curtab1 = tab1_records[idx]
        curtab2 = tab1_records[idx + 1]

        if 1 <= interp_type <= 5:
            cur_y1 = cur_y
            cur_y2 = cur_y
            eff_interp_type = interp_type
            jac1 = jac2 = None
        elif 21 <= interp_type <= 25:
            # unit-base interpolation
            y1_min = np.min(curtab1[yp_name])
            y1_max = np.max(curtab1[yp_name])
            y2_min = np.min(curtab2[yp_name])
            y2_max = np.max(curtab2[yp_name])
            cur_y1, cur_y2, jac1, jac2 = determine_unit_base_coordinates(
                cur_x, cur_y, x1, x2, y1_min, y1_max, y2_min, y2_max,
            )
            eff_interp_type = interp_type - 20
        else:
            raise ValueError(
                f'Unsupported interpolation type INT={interp_type}.'
            )

        # Inner interp threads xp through so tracers stored in the
        # per-panel ``curtab*[fp_name]`` list survive into the outer
        # two-point interp (issue #169). The ENDF-region loop inside
        # ``interp_tab1`` runs on numpy-side indexing (mesh, INT/NBT)
        # and only the values arithmetic is xp-native, matching
        # ``interp_tab1``'s own dispatch.
        f1 = interp_tab1(
            cur_y1, curtab1, yp_name, fp_name, outside_value, xp=xp,
        )
        f2 = interp_tab1(
            cur_y2, curtab2, yp_name, fp_name, outside_value, xp=xp,
        )

        if jac1 is not None:
            f1 = f1 * xp.asarray(jac1)
            f2 = f2 * xp.asarray(jac2)

        rows.append(_interp_two_point_columns(
            cur_x, x1, x2, f1, f2, eff_interp_type, xp,
        ))

    if len(rows) == 0:
        result_arr = xp.zeros((0, y_np.shape[1]), dtype=xp.float64)
    else:
        result_arr = xp.stack(rows, axis=0)

    if outside_value is not None and any_outside:
        full_shape = (len(x_orig), y_np.shape[1])
        # Build the full-size result on xp; scatter the computed
        # rows into the inside positions. Numpy-native path is
        # bit-identical to the old in-place assignment; on JAX we
        # use `.at[].set()` for the same effect (jnp arrays are
        # immutable).
        if xp.name == 'jax':
            full = xp.full(full_shape, outside_value, dtype=xp.float64)
            full = full.at[is_inside, :].set(result_arr)
            result_arr = full
        else:
            full = np.full(full_shape, outside_value, dtype=float)
            full[is_inside, :] = np.asarray(result_arr)
            result_arr = xp.asarray(full)

    return result_arr
