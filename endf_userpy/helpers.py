import numpy as np


def deg2rad(values):
    return np.pi / 180.0 * np.array(values, copy=None)


def dict2array(obj, dtype=None):
    return np.array(list(v for v in obj.values()), dtype=dtype)


def check_int_nbt(int_arr, nbt_arr):
    int_arr = np.array(int_arr, copy=None)
    nbt_arr = np.array(nbt_arr, copy=None)
    if int_arr.ndim != 1 or nbt_arr.ndim != 1:
        raise IndexError('`int_arr` and `nbt_arr` must be 1d arrays')
    if int_arr.size != nbt_arr.size:
        raise IndexError('`int_arr` and `nbt_arr` must be of same size')


def convert_interp_repr(int_arr, nbt_arr):
    num_elements = nbt_arr[-1]
    interp_arr = np.zeros(num_elements, dtype=int)
    first_idx = 0
    for i in range(len(nbt_arr)):
        upper_idx = nbt_arr[i]
        interp_arr[first_idx:upper_idx] = int_arr[i]
        first_idx = upper_idx
    return interp_arr


def find_interval(a, v):
    """Find indices where elements should be inserted to maintain order."""
    a = np.array(a, copy=None)
    v = np.array(v, copy=None)
    # range checks
    if np.any((np.min(a) > v) | (np.max(a) < v)):
        raise IndexError(
            "Some values in `v` are not in the range of mesh spanned by `a`."
        )
    # interval finding
    idcs = np.searchsorted(a, v, side='right')
    if v.ndim == 0:
        if idcs == a.size:
            idcs -= 1
    else:
        idcs[idcs == a.size] -= 1
    idcs -= 1
    return idcs


def get_enclosing_points(x, xp, fp):
    x = np.array(x, copy=None)
    xp = np.array(xp, copy=None)
    fp = np.array(fp, copy=None)
    idcs1 = find_interval(xp, x)
    idcs2 = idcs1 + 1
    x1 = xp[idcs1]
    x2 = xp[idcs2]
    y1 = fp[idcs1]
    y2 = fp[idcs2]
    return x1, y1, x2, y2
