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


def interp_const(x, xp, fp):
    """Constant interpolation"""
    x1, y1, x2, y2 = get_enclosing_points(x, xp, fp)
    return y1


def interp_lin_lin(x, xp, fp):
    """Linear-Linear interpolation"""
    x1, y1, x2, y2 = get_enclosing_points(x, xp, fp)
    return y1 + (x-x1)*(y2-y1)/(x2-x1)


_INTERP_LOG_SMALL = 1.0e-38


def interp_lin_log(x, xp, fp):
    """Linear-Logarithmic interpolation.

    Clamps ``x1 == 0`` to a small positive value before taking
    logs, matching the Fortran ``yintp`` reference behaviour
    (endf6.f90 line 2044). This surfaces on ENDF files whose x-mesh
    starts at zero (e.g. MF6 LAW=7 outgoing-energy tabulations at
    Ep=0) when INT=3/5 is applied.
    """
    x1, y1, x2, y2 = get_enclosing_points(x, xp, fp)
    x1 = np.where(x1 == 0.0, _INTERP_LOG_SMALL, x1)
    return y1 + np.log(x/x1)*(y2-y1)/np.log(x2/x1)


def interp_log_lin(x, xp, fp):
    """Logarithmic-Linear interpolation.

    Clamps ``y1 == 0`` to a small positive value before taking
    logs, matching Fortran ``yintp`` (endf6.f90 line 2049).
    """
    x1, y1, x2, y2 = get_enclosing_points(x, xp, fp)
    y1 = np.where(y1 == 0.0, _INTERP_LOG_SMALL, y1)
    return y1*np.exp((x-x1)*np.log(y2/y1)/(x2-x1))


def interp_log_log(x, xp, fp):
    """Logarithmic-Logarithmic interpolation.

    Clamps both ``x1 == 0`` and ``y1 == 0`` to a small positive
    value before taking logs, matching Fortran ``yintp``
    (endf6.f90 lines 2054-2055).
    """
    x1, y1, x2, y2 = get_enclosing_points(x, xp, fp)
    x1 = np.where(x1 == 0.0, _INTERP_LOG_SMALL, x1)
    y1 = np.where(y1 == 0.0, _INTERP_LOG_SMALL, y1)
    return y1*np.exp(np.log(x/x1)*np.log(y2/y1)/np.log(x2/x1))


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


def interp(x, xp, fp, interp_type, outside_value=None):
    """Interpolation using various schemes"""
    # TODO: Here we provisionally let NaN values pass through the
    #       program logic for comparison with the Fortran routines.
    #       However, eventually no NaN values should appear in x.
    is_inside = ((x >= np.min(xp)) & (x <= np.max(xp))) | np.isnan(x)
    if not np.all(is_inside) and outside_value is None:
        raise ValueError('some `x` value outside mesh given by `xp`')
    xi = x[is_inside]
    if interp_type == 1:
        fi = interp_const(xi, xp, fp)
    elif interp_type == 2:
        fi = interp_lin_lin(xi, xp, fp)
    elif interp_type == 3:
        fi = interp_lin_log(xi, xp, fp)
    elif interp_type == 4:
        fi = interp_log_lin(xi, xp, fp)
    elif interp_type == 5:
        fi = interp_log_log(xi, xp, fp)
    else:
        raise TypeError(f"interpolation scheme (INT={interp_type}) not implemented")
    f = np.full(x.shape, outside_value, dtype=float)
    f[is_inside] = fi
    return f


def endf_interp1d(x, xp, fp, int_arr, nbt_arr, outside_value=None):
    check_int_nbt(int_arr, nbt_arr)
    x = np.asarray(x)
    # Rebind `xp` to a deduplicated copy rather than mutating the
    # caller's array in place. `treat_duplicates` perturbs repeated
    # mesh values by a relative epsilon so `searchsorted` can
    # distinguish them; if we did that in place, any caller that
    # passes a long-lived array (e.g. cached from the ENDF dict)
    # would have its mesh silently modified, and a second call on
    # the same array would perturb it again (issue #49).
    xp = treat_duplicates(xp)
    # TODO: Here we provisionally let NaN values pass through the
    #       program logic for comparison with the Fortran routines.
    #       However, eventually no NaN values should appear in x.
    is_inside = ((x >= np.min(xp)) & (x <= np.max(xp))) | np.isnan(x)
    if not np.all(is_inside) and outside_value is None:
        raise ValueError('some `x` value outside mesh given by `xp`')
    xi = x[is_inside]
    fi = np.zeros(xi.shape, dtype=float)
    idcs = find_interval(xp, xi)
    # Per ENDF-6 TAB1 semantics, consecutive interpolation regions
    # share their boundary row: region n covers rows NBT(n-1)..NBT(n).
    # In Python 0-indexing, region 0 is xp[0:NBT[0]] and region k>0 is
    # xp[NBT[k-1]-1:NBT[k]] (overlapping the boundary). The interval
    # starting at the shared row belongs to the upper region.
    first_idx = 0
    nregions = len(int_arr)
    for i in range(nregions):
        last_idx = nbt_arr[i]
        is_last = (i == nregions - 1)
        interp_type = int_arr[i]
        cur_xp = xp[first_idx:last_idx]
        cur_fp = fp[first_idx:last_idx]
        upper = last_idx if is_last else last_idx - 1
        is_in_range = (idcs >= first_idx) & (idcs < upper)
        cur_x = xi[is_in_range]
        fi[is_in_range] = interp(
            cur_x, cur_xp, cur_fp, interp_type, outside_value
        )
        # Share the boundary row with the next region.
        first_idx = last_idx - 1

    f = np.empty(x.shape, dtype=float)
    f[~is_inside] = outside_value
    f[is_inside] = fi
    return f


def interp_legendre_coeffs(x, xp, coeffs, int_arr, nbt_arr, outside_value=None):
    x = np.asarray(x)
    interp_coeffs = np.zeros((x.shape[0], coeffs.shape[1]), dtype=float)
    for i in range(interp_coeffs.shape[1]):
        interp_coeffs[:, i] = endf_interp1d(
            x, xp, coeffs[:, i], int_arr, nbt_arr, outside_value,
        )
    return interp_coeffs


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

    Backend-agnostic: pass ``xp=array_ns.get_backend(name)`` to
    dispatch the Legendre evaluation onto that backend. The
    per-degree coefficient interpolation over the file's energy
    mesh (via :func:`interp_legendre_coeffs`) stays on numpy since
    it operates on file-provided arrays whose shape and duplicate
    structure are only known at Python time; the interpolated
    coefficients are converted at the boundary via ``xp.asarray``.
    ``xp=None`` (the default) resolves to numpy, keeping every
    pre-port caller on the same bit-identical path.

    Note: the mesh parameter is named ``xp_mesh`` (was ``xp``
    pre-port) to avoid the collision with the backend adapter
    also conventionally named ``xp``. Positional callers are
    unaffected since the argument position is unchanged.
    """
    xp = _resolve_xp(xp)
    x = np.asarray(x)
    mu = np.asarray(mu)
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    if mu.shape[0] == 1:
        mu = np.broadcast_to(mu, (x.size, mu.shape[1]))
    # Coefficient interpolation over the file's energy mesh stays
    # on numpy: `endf_interp1d` walks per-region loops with
    # dynamically-many ENDF interpolation zones and calls
    # `treat_duplicates` (np.unique-based) on the mesh.
    interp_coeffs = interp_legendre_coeffs(
        x, xp_mesh, coeffs, int_arr, nbt_arr, outside_value,
    )
    # Move to the requested backend for the Legendre evaluation.
    # Caller passes the FULL coefficient array including the ``a_0``
    # constant term with any (2L+1)/2 normalisation factor already
    # applied (see e.g.
    # :func:`mf4_interpretation._convert_legendre_to_numpy_array`),
    # so ``_eval_legendre_series`` evaluates
    # ``sum_L coeffs[..., L] * P_L(mu)`` directly.
    interp_coeffs_xp = xp.asarray(interp_coeffs)
    mu_xp = xp.asarray(mu)
    return _eval_legendre_series(interp_coeffs_xp, mu_xp, xp)


def interp_tab1(x, tab1, xp_name, fp_name, outside_value=None):
    x_mesh = np.array(tab1[xp_name], dtype=float)
    f_mesh = np.array(tab1[fp_name], dtype=float)
    int_arr = np.array(tab1['INT'], dtype=int)
    nbt_arr = np.array(tab1['NBT'], dtype=int)
    return endf_interp1d(
        x, x_mesh, f_mesh, int_arr, nbt_arr, outside_value
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
        to numpy. The per-panel outer interpolation and the unit-base
        Jacobian arithmetic run through this backend; the per-panel
        inner :func:`interp_tab1` evaluation stays on numpy since it
        loops over ENDF-interpolation regions of dynamically-many
        zones (its result is converted at the boundary via
        ``xp.asarray`` so the outer arithmetic is backend-native).

    Returns
    -------
    array
        A two-dimensional array (numpy or backend-native depending
        on ``xp``) with the interpolated function values. The value
        in the i-th row and j-th column corresponds to the function
        value for ``x[i]`` and ``y[j]``.
    """
    xp = _resolve_xp(xp)
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

        # Inner interp stays on numpy (dict-of-records access +
        # ENDF-region loop). Boundary-convert to xp for the outer
        # two-point interp so the outer arithmetic is backend-native.
        f1 = xp.asarray(interp_tab1(
            cur_y1, curtab1, yp_name, fp_name, outside_value,
        ))
        f2 = xp.asarray(interp_tab1(
            cur_y2, curtab2, yp_name, fp_name, outside_value,
        ))

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

    return result_arr
