import logging
import numpy as np
from numpy.polynomial.legendre import Legendre
from .helpers import (
    check_int_nbt,
    find_interval,
    get_enclosing_points,
    convert_interp_repr,
)


def interp_const(x, xp, fp):
    """Constant interpolation"""
    x1, y1, x2, y2 = get_enclosing_points(x, xp, fp)
    return y1


def interp_lin_lin(x, xp, fp):
    """Linear-Linear interpolation"""
    x1, y1, x2, y2 = get_enclosing_points(x, xp, fp)
    return y1 + (x-x1)*(y2-y1)/(x2-x1)


def interp_lin_log(x, xp, fp):
    """Linear-Logarithmic interpolation"""
    x1, y1, x2, y2 = get_enclosing_points(x, xp, fp)
    return y1 + np.log(x/x1)*(y2-y1)/np.log(x2/x1)


def interp_log_lin(x, xp, fp):
    """Logarithmic-Linear interpolation"""
    x1, y1, x2, y2 = get_enclosing_points(x, xp, fp)
    return y1*np.exp((x-x1)*np.log(y2/y1)/(x2-x1)) 


def interp_log_log(x, xp, fp):
    """Logarithmic-Logarithmic interpolation"""
    x1, y1, x2, y2 = get_enclosing_points(x, xp, fp)
    return y1*np.exp(np.log(x/x1)*np.log(y2/y1)/np.log(x2/x1))


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
    x = np.array(x, copy=None)
    # TODO: Here we provisionally let NaN values pass through the
    #       program logic for comparison with the Fortran routines.
    #       However, eventually no NaN values should appear in x.
    is_inside = ((x >= np.min(xp)) & (x <= np.max(xp))) | np.isnan(x)
    if not np.all(is_inside) and outside_value is None:
        raise ValueError('some `x` value outside mesh given by `xp`')
    xi = x[is_inside]
    fi = np.zeros(xi.shape, dtype=float)
    idcs = find_interval(xp, xi)
    first_idx = 0
    for i in range(len(int_arr)):
        last_idx = nbt_arr[i]
        interp_type = int_arr[i]
        cur_xp = xp[first_idx:last_idx]
        cur_fp = fp[first_idx:last_idx]
        is_in_range = (idcs >= first_idx) & (idcs < last_idx)
        cur_idcs = idcs[is_in_range]
        cur_x = xi[is_in_range]
        fi[is_in_range] = interp(
            cur_x, cur_xp, cur_fp, interp_type, outside_value
        )
        first_idx = last_idx

    f = np.empty(x.shape, dtype=float)
    f[~is_inside] = outside_value
    f[is_inside] = fi
    return f


def interp_legendre_coeffs(x, xp, coeffs, int_arr, nbt_arr):
    x = np.array(x, copy=None)
    interp_coeffs = np.zeros((x.shape[0], coeffs.shape[1]), dtype=float)
    for i in range(interp_coeffs.shape[1]):
        interp_coeffs[:, i] = endf_interp1d(x, xp, coeffs[:,i], int_arr, nbt_arr)
    return interp_coeffs


def evaluate_interp_legendre_polynomials(x, mu, xp, coeffs, int_arr, nbt_arr):
    x = np.array(x, copy=None)
    mu = np.array(mu, copy=None)
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    if mu.shape[0] == 1:
        mu = np.tile(mu, (x.size, 1))
    result = np.zeros((x.size, mu.shape[1]), dtype=float)
    interp_coeffs = interp_legendre_coeffs(x, xp, coeffs, int_arr, nbt_arr)
    for i in range(interp_coeffs.shape[0]):
        cur_res = Legendre(interp_coeffs[i,:])(mu[i,:])
        result[i,:] = cur_res
    return result


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
    x, y, xp, int_arr, nbt_arr, tab1_records, yp_name, fp_name,
    outside_value=None
):
    """Perform 2d interpolation using TAB2/TAB1 record sequence

    Parameters
    ----------
    x : numpy.ndarray
        Target x value
    y: numpy.ndarray
        Target y value
    xp : numpy.ndarray
        Mesh of x-values
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

    Returns
    -------
    numpy.ndarray
        A two-dimensional array with the interpolated function values.
        The value in the i-th row and j-th column corresponds is the
        function values for `x[i]` and `y[j]`.
    """
    if y.ndim == 1:
        y = y.reshape(1, -1)

    if outside_value is not None:
        is_inside = (x >= np.min(xp)) & (x <= np.max(xp))
        any_outside = not np.all(is_inside)
        x_orig = x
        x = x_orig[is_inside]

    idcs = find_interval(xp, x)
    interp_arr = convert_interp_repr(int_arr, nbt_arr)
    # interpolation between angles
    result_dim = (len(x), y.shape[1])
    result_arr = np.zeros(result_dim, dtype=float)
    for i, idx in enumerate(idcs):
        cur_x = x[i]
        x1 = xp[idx]
        x2 = xp[idx+1]
        cur_y = y[0,:] if y.shape[0] == 1 else y[i,:]
        interp_type = interp_arr[idx]
        curtab1 = tab1_records[idx]
        curtab2 = tab1_records[idx+1]

        if interp_type >= 1 and interp_type <= 5:
            cur_y1 = cur_y
            cur_y2 = cur_y
            eff_interp_type = interp_type
        elif interp_type >= 21 and interp_type <= 25:
            # unit-base interpolation
            y1_min = np.min(curtab1[yp_name])
            y1_max = np.max(curtab1[yp_name])
            y2_min = np.min(curtab2[yp_name])
            y2_max = np.max(curtab2[yp_name])
            cur_y1, cur_y2, jac1, jac2 = \
                determine_unit_base_coordinates(
                    cur_x, cur_y, x1, x2, y1_min, y1_max, y2_min, y2_max
                )
            eff_interp_type = interp_type - 20
        else:
            raise ValueError(f'Unsupported interpolation type INT={interp_type}.')

        f1 = interp_tab1(cur_y1, curtab1, yp_name, fp_name, outside_value)
        f2 = interp_tab1(cur_y2, curtab2, yp_name, fp_name, outside_value)

        if interp_type >= 21 and interp_type <= 25:
            # apply Jacobian in the case of unit-base interpolation
            f1 *= jac1
            f2 *= jac2

        red_xp = np.array([x1, x2], dtype=float)
        red_f = np.vstack([f1, f2])
        curres = np.zeros(y.shape[1], dtype=float)
        for j in range(y.shape[1]):
            curres[j] = \
                interp(cur_x, red_xp , red_f[:,j], eff_interp_type)

        result_arr[i,:] = curres

    if outside_value is not None and any_outside:
            full_result_dim = (len(x_orig), y.shape[1])
            full_result_arr = np.full(full_result_dim, outside_value, dtype=float)
            full_result_arr[is_inside, :] = result_arr
            result_arr = full_result_arr

    return result_arr
