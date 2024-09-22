import numpy as np
from numpy.polynomial.legendre import Legendre
from .helpers import (
    check_int_nbt,
    find_interval,
    get_enclosing_points,
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


def interp(x, xp, fp, interp_type):
    """Interpolation using various schemes"""
    if interp_type == 1:
        return interp_const(x, xp, fp)
    elif interp_type == 2:
        return interp_lin_lin(x, xp, fp)
    elif interp_type == 3:
        return interp_lin_log(x, xp, fp)
    elif interp_type == 4:
        return interp_log_lin(x, xp, fp)
    elif interp_type == 5:
        return interp_log_log(x, xp, fp)
    else:
        raise TypeError(f"interpolation scheme (INT={interp_type}) not implemented")


def endf_interp1d(x, xp, fp, int_arr, nbt_arr):
    check_int_nbt(int_arr, nbt_arr)
    x = np.array(x, copy=None)
    f = np.zeros(x.shape, dtype=float)
    idcs = find_interval(xp, x)
    first_idx = 0
    for i in range(len(int_arr)):
        last_idx = nbt_arr[i]
        interp_type = int_arr[i]
        cur_xp = xp[first_idx:last_idx]
        cur_fp = fp[first_idx:last_idx]
        is_in_range = (idcs >= first_idx) & (idcs < last_idx)
        cur_idcs = idcs[is_in_range]
        cur_x = x[is_in_range]
        f[is_in_range] = interp(cur_x, cur_xp, cur_fp, interp_type)  
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
