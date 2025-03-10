import numpy as np


def deg2rad(values):
    return np.pi / 180.0 * np.array(values, copy=None)


def _determine_dims(obj):
    """Determine dimensions to convert ragged nested list to numpy array."""
    if isinstance(next(iter(obj)), list):
        dims_tuples = tuple(_determine_dims(v) for v in obj)
        cur_size = len(dims_tuples)
        num_dims = len(dims_tuples[0])
        if not all(len(t) == num_dims for t in dims_tuples):
            raise IndexError("different levels of nesting encountered")
        max_dims_tuple = tuple(max(v[i] for v in dims_tuples) for i in range(num_dims))
        ext_dims = (cur_size,) + max_dims_tuple 
        return ext_dims
    else:
        return (len(obj),)


def dict2list(obj):
    """Convert dict-style array into list-style array."""
    if isinstance(next(iter(obj.values())), dict):
        lst = list(dict2list(v) for v in obj.values())
    else:
        lst = list(obj.values())
    return lst


def pad_nested_ragged_lists(obj, fill_value=0.0, dims=None):
    """Pad a nested list with trailing fill values inplace to obtain regular shape."""
    if dims is None:
        dims = _determine_dims(obj)
    if not isinstance(obj, list):
        return
    eff_fill_value = fill_value if len(dims) == 1 else []
    for i in range(dims[0]):
        if i == len(obj):
            obj.append(fill_value)
        if len(dims) > 1:
            pad_nested_ragged_lists(obj[i], fill_value, dims[1:])


def dict2array(obj, dtype=None, order='K', fill_value=None):
    """Construct (multi-dim) array from nested dictionaries"""
    arr_list = dict2list(obj)
    if fill_value is not None:
        pad_nested_ragged_lists(arr_list, fill_value)
    return np.array(arr_list, dtype=dtype, order=order)


def check_int_nbt(int_arr, nbt_arr):
    int_arr = np.array(int_arr, copy=None)
    nbt_arr = np.array(nbt_arr, copy=None)
    if int_arr.ndim != 1 or nbt_arr.ndim != 1:
        raise IndexError('`int_arr` and `nbt_arr` must be 1d arrays')
    if int_arr.size != nbt_arr.size:
        raise IndexError('`int_arr` and `nbt_arr` must be of same size')


def is_sorted(arr):
    return np.all(arr[:-1] <= arr[1:])


def treat_duplicates(arr, releps=1e-8, inplace=False):
    """Deduplicate values in an array."""
    if not inplace:
        arr = arr.copy()
    if not is_sorted(arr):
        raise ValueError('array not sorted')
    elems, counts = np.unique(arr, return_counts=True)
    elem_idx = 0
    for curelem, curcount in zip(elems, counts):
        if curcount > 1:
            arr[elem_idx:elem_idx+curcount] *= (1 - releps * np.arange(curcount-1, -1, -1))
        elem_idx += curcount
    return arr


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


def find_indices_with_tol(a, v, atol, rtol):
    a = np.array(a)
    v = np.array(v)
    # bring into order
    ordidcs = np.argsort(a)
    a = a[ordidcs]
    # match elements
    idcs = np.minimum(np.searchsorted(a, v), len(a)-1)
    s = np.isclose(a[idcs], v, atol=atol, rtol=rtol)
    rem_idcs = np.maximum(idcs[~s]-1, 0)
    idcs[~s] = rem_idcs
    s[~s] = np.isclose(a[rem_idcs], v[~s], atol=atol, rtol=rtol)
    idcs[~s] = -1  # indication for not found
    # map to original order
    found_sel = idcs != -1
    idcs[found_sel] = ordidcs[idcs[found_sel]]
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


# Taken from https://stackoverflow.com/a/457805
# with small adjustments for array compatibility
def erf(x):
    # save the sign of x
    sign = np.ones_like(x, dtype=float)
    sign[x < 0] = -1.0
    x = np.abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)
