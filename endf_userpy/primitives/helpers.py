import inspect
import numpy as np
from typing import List, Callable, Union


def unpack_za(za):
    return int(za // 1000), int(za % 1000)


def deg2rad(values):
    return np.pi / 180.0 * np.asarray(values)


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
    for i in range(dims[0]):
        if i == len(obj):
            obj.append(fill_value)
        if len(dims) > 1:
            pad_nested_ragged_lists(obj[i], fill_value, dims[1:])


def dict2array(obj, dtype=None, order='K', fill_value=None, xp=None):
    """Construct (multi-dim) array from nested dictionaries.

    Backend-agnostic: ``xp=None`` (default) preserves the pre-port
    ``np.array(...)`` behaviour bit-for-bit. Passing a backend
    adapter routes through ``xp.asarray`` so JAX tracers stored in
    the source dict propagate through (issue #154). The ``order``
    argument is only honoured on numpy; JAX doesn't expose a
    memory-order flag and silently ignores it.
    """
    arr_list = dict2list(obj)
    if fill_value is not None:
        pad_nested_ragged_lists(arr_list, fill_value)
    if xp is None:
        return np.array(arr_list, dtype=dtype, order=order)
    if dtype is None:
        return xp.asarray(arr_list)
    return xp.asarray(arr_list, dtype=dtype)


def check_int_nbt(int_arr, nbt_arr):
    int_arr = np.asarray(int_arr)
    nbt_arr = np.asarray(nbt_arr)
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
    a = np.asarray(a)
    v = np.asarray(v)
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


def get_enclosing_points(x, xp_mesh, fp):
    """Return (x1, y1, x2, y2) for the mesh intervals enclosing each
    query x.

    Mesh (xp_mesh) and query (x) are always converted to numpy for
    the ``find_interval`` index lookup (panel-finding is inherently
    nondifferentiable). ``fp`` (the tabulated function values) is
    NOT force-converted, so a JAX tracer ``fp`` propagates through
    the advanced-index lookup ``fp[idcs]`` -- essential for the
    dict-source autodiff path (issue #154).
    """
    x = np.asarray(x)
    xp_mesh = np.asarray(xp_mesh)
    # NB: no `fp = np.asarray(fp)` -- keep tracers intact.
    idcs1 = find_interval(xp_mesh, x)
    idcs2 = idcs1 + 1
    x1 = xp_mesh[idcs1]
    x2 = xp_mesh[idcs2]
    y1 = fp[idcs1]
    y2 = fp[idcs2]
    return x1, y1, x2, y2


# Taken from https://stackoverflow.com/a/457805
# with small adjustments for array compatibility
def erf(x, xp=None):
    """Backend-dispatched error function.

    Uses ``scipy.special.erf`` on the numpy path (full double
    precision, vectorised, ``scipy`` is already a hard runtime
    dependency) and ``jax.scipy.special.erf`` on the JAX path
    (autodiff-safe, full precision). Replaces an earlier hand-
    coded A&S 7.1.26 rational approximation whose max error was
    ~1.5e-7 -- well below the tolerances the MF5 fission-spectrum
    tests operate at, but not worth the maintenance burden or the
    precision loss on the modern JAX autodiff use case.
    """
    if xp is None or getattr(xp, 'name', None) == 'numpy':
        # Import lazily so importing this module still works if
        # scipy is somehow unavailable at build time (it is not,
        # per setup.py's install_requires, but the lazy import
        # keeps the failure mode local to callers of erf).
        from scipy.special import erf as _scipy_erf
        return _scipy_erf(x)
    if getattr(xp, 'name', None) == 'jax':
        import jax.scipy.special as _jsp
        return _jsp.erf(x)
    # Unknown adapter: try xp.erf, else fall back to scipy.
    if hasattr(xp, 'erf'):
        return xp.erf(x)
    from scipy.special import erf as _scipy_erf
    return _scipy_erf(x)


def pad_outside_values(argnames: List[str], selectors: Union[List[Callable], Callable]):
    """Decorator factory to zero-pad results for invalid inputs."""
    def decorator(func):
        def get_arrays(args):
            all_argnames = list(inspect.signature(func).parameters.keys())
            return [
                args[all_argnames.index(p)] for p in argnames
            ]

        def replace_arrays(args, new_arrays):
            all_argnames = list(inspect.signature(func).parameters.keys())
            new_args = list(args)
            for i, an in enumerate(argnames):
                idx = all_argnames.index(an)
                new_args[idx] = new_arrays[i]
            return new_args

        def wrapfunc(*args, **kwargs):
            arr_list = get_arrays(args)
            inside_list = [
                s(a, *args, **kwargs) for s, a in zip(selectors, arr_list)
            ]
            if all(np.all(v) for v in inside_list):
                return func(*args, **kwargs)
            filtered_arrays = [
                a[f] for a, f in zip(arr_list, inside_list)
            ]
            new_args = replace_arrays(args, filtered_arrays)
            res_dim = [a.shape[-1] for a in inside_list]
            res_arr = np.zeros(res_dim, dtype=np.float64)
            squeezed_inside_list = [a if a.ndim == 1 else np.squeeze(a) for a in inside_list]
            res_arr[np.ix_(*squeezed_inside_list)] = func(*new_args, **kwargs)
            return res_arr
        return wrapfunc
    return decorator


def no_filter(arr, *args, **kwargs):
    dims = [1]*(len(arr.shape)-1) + [arr.shape[-1]]
    return np.ones(dims, dtype=bool)
