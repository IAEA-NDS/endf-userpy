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
    is_inside = (x >= np.min(xp)) & (x <= np.max(xp))
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
    is_inside = (x >= np.min(xp)) & (x <= np.max(xp))
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


def interp_tab2(
    x, y, xp, int_arr, nbt_arr, tab1_records, yp_name, fp_name,
    outside_value=None
):
    if y.ndim == 1:
        y = y.reshape(1, -1)
    idcs = find_interval(xp, x)
    interp_arr = convert_interp_repr(int_arr, nbt_arr)
    # interpolation between angles
    result_dim = (len(x), y.shape[1])
    result_arr = np.zeros(result_dim, dtype=float)
    for i, idx in enumerate(idcs):
        cur_y = y[0,:] if y.shape[0] == 1 else y[i,:]
        curtab1 = tab1_records[idx]
        curtab2 = tab1_records[idx+1]
        f1 = interp_tab1(cur_y, curtab1, yp_name, fp_name, outside_value)
        f2 = interp_tab1(cur_y, curtab2, yp_name, fp_name, outside_value)
        interp_type = interp_arr[idx]
        red_xp = xp[idx:idx+2]
        red_f = np.vstack([f1, f2])
        cur_x = x[i]
        curres = np.zeros(y.shape[1], dtype=float)
        for j in range(y.shape[1]):
            curres[j] = \
                interp(cur_x, red_xp , red_f[:,j], interp_type)

        result_arr[i,:] = curres

    return result_arr
