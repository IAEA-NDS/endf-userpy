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


_exp1_jax_customjvp = None
_gammainc_jax_customjvp = None


# Abramowitz & Stegun 5.1.53 (0 <= x <= 1): E1(x) + ln(x) = sum a_k x^k.
# Accuracy ~2e-7, adequate for the MF5 LF=12 use case.
_AS_5_1_53 = (
    -0.57721566, 0.99999193, -0.24991055,
    0.05519968, -0.00976004, 0.00107857,
)
# A&S 5.1.56 (1 <= x < inf):
#   x exp(x) E1(x) = (x^4 + a1 x^3 + a2 x^2 + a3 x + a4)
#                    / (x^4 + b1 x^3 + b2 x^2 + b3 x + b4)
# Accuracy ~2e-8.
_AS_5_1_56_NUM = (8.5733287401, 18.0590169730, 8.6347608925, 0.2677737343)
_AS_5_1_56_DEN = (9.5733223454, 25.6329561486, 21.0996530827, 3.9584969228)


def _exp1_hand_coded_jax(x):
    """E1(x) for x > 0 via A&S rational approximations, jit-friendly.

    Two branches blended with ``jnp.where`` so the graph stays flat
    (no ``jnp.piecewise`` -> ``lax.switch`` expansion). Both
    branches are pure polynomial arithmetic on the query x, so the
    grad graph is small too.

    Accuracy: ~2e-7 on x in [0, 1], ~2e-8 on x >= 1. Comfortably
    below any physical tolerance the MF5 LF=12 kernel demands.
    """
    import jax.numpy as jnp
    x = jnp.asarray(x)
    # Safe x for the log branch (x <= 1) so log(x) does not NaN on
    # x <= 0 elements we mask out.
    x_lo_safe = jnp.where(x > 0.0, x, 1.0)
    # Horner-style polynomial for the low branch: sum a_k * x^k.
    a = _AS_5_1_53
    poly_lo = a[0] + x_lo_safe * (
        a[1] + x_lo_safe * (
            a[2] + x_lo_safe * (
                a[3] + x_lo_safe * (
                    a[4] + x_lo_safe * a[5]
                )
            )
        )
    )
    e1_lo = poly_lo - jnp.log(x_lo_safe)

    # High branch (x >= 1): (x^4 + a1 x^3 + a2 x^2 + a3 x + a4) /
    #                      (x^4 + b1 x^3 + b2 x^2 + b3 x + b4)
    x_hi_safe = jnp.where(x >= 1.0, x, 2.0)  # keep hi branch well-defined
    an = _AS_5_1_56_NUM
    bn = _AS_5_1_56_DEN
    x2 = x_hi_safe * x_hi_safe
    x3 = x2 * x_hi_safe
    x4 = x2 * x2
    num = x4 + an[0] * x3 + an[1] * x2 + an[2] * x_hi_safe + an[3]
    den = x4 + bn[0] * x3 + bn[1] * x2 + bn[2] * x_hi_safe + bn[3]
    e1_hi = jnp.exp(-x_hi_safe) * num / (x_hi_safe * den)

    return jnp.where(x <= 1.0, e1_lo, e1_hi)


def _get_exp1_jax_customjvp():
    """Return a cached ``custom_jvp``-wrapped E1(x) implementation
    with an analytic gradient rule (issue #207).

    Forward: hand-coded A&S rational approximation (jit-friendly,
    no ``jnp.piecewise``).
    JVP: analytic ``-exp(-x) / x``.

    Bypasses the ``jnp.piecewise`` -> ``lax.switch`` expansion in
    ``jax.scipy.special.exp1`` that made the LF=12 loss grad take
    ~1 minute per call under naked ``jax.grad``.
    """
    global _exp1_jax_customjvp
    if _exp1_jax_customjvp is None:
        import jax
        import jax.numpy as jnp

        @jax.custom_jvp
        def _fn(x):
            return _exp1_hand_coded_jax(x)

        @_fn.defjvp
        def _fn_jvp(primals, tangents):
            (x,) = primals
            (dx,) = tangents
            return _fn(x), -jnp.exp(-x) / x * dx

        _exp1_jax_customjvp = _fn
    return _exp1_jax_customjvp


def _gammainc_hand_coded_jax(a, x):
    """Regularised lower incomplete gamma P(a, x) via a series /
    asymptotic form, jit-friendly (no ``jnp.piecewise``).

    For ``x < a + 1``: series
      P(a, x) = x^a exp(-x) / Gamma(a) * sum_{k=0}^inf x^k / (a+k)! ...
    In practice we use:
      P(a, x) = x^a exp(-x) / Gamma(a+1) * (1 + x/(a+1) + x^2/((a+1)(a+2)) + ...)

    For ``x >= a + 1``: 1 - Q(a, x) via a Lentz continued fraction.

    We support only fixed ``a = 1.5`` for the MF5 LF=12 use case;
    a general implementation would blend both branches with
    ``jnp.where``. Since the LF=12 kernel calls this at a static
    scalar ``a``, we can specialise.
    """
    import jax.numpy as jnp
    a = jnp.asarray(a, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    x_safe = jnp.where(x > 0.0, x, 1e-38)

    # Series form for the "small x" branch: use enough terms to
    # cover x up to ~2*a comfortably.
    def _series(a, x):
        # P(a, x) = x^a exp(-x) / Gamma(a+1) * S(a, x)
        # where S = sum_{n=0}^N x^n / Pochhammer(a+1, n)
        n_terms = 40
        term = jnp.ones_like(x)
        total = term
        for k in range(1, n_terms):
            term = term * x / (a + k)
            total = total + term
        # Gamma(a+1) via lgamma
        import jax.scipy.special as _jsp
        return (
            jnp.exp(a * jnp.log(x_safe) - x - _jsp.gammaln(a + 1.0))
            * total
        )

    # Continued fraction form (Lentz) for large x.
    def _cf(a, x):
        # Q(a, x) = x^a exp(-x) / Gamma(a) * CF(a, x)
        # CF(a, x) = 1/(x+1-a - 1*(1-a)/(x+3-a - 2*(2-a)/(x+5-a - ...)))
        n_terms = 60
        # Modified Lentz's method
        tiny = 1e-30
        b = x + 1.0 - a
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in range(1, n_terms):
            an = -i * (i - a)
            b = b + 2.0
            d = an * d + b
            d = jnp.where(jnp.abs(d) < tiny, tiny, d)
            c = b + an / c
            c = jnp.where(jnp.abs(c) < tiny, tiny, c)
            d = 1.0 / d
            delta = d * c
            h = h * delta
        import jax.scipy.special as _jsp
        q = jnp.exp(a * jnp.log(x_safe) - x - _jsp.gammaln(a)) * h
        return 1.0 - q

    series_val = _series(a, x_safe)
    cf_val = _cf(a, x_safe)
    # Blend at x = a + 1
    return jnp.where(x < a + 1.0, series_val, cf_val)


def _get_gammainc_jax_customjvp():
    """Return a cached ``custom_jvp``-wrapped P(a, x) with analytic
    grad wrt x (issue #207).

    Forward: hand-coded series / continued-fraction (jit-friendly).
    JVP wrt x: ``x^(a-1) exp(-x) / Gamma(a)``.
    Grad wrt ``a`` returns zero (not used in the physics).
    """
    global _gammainc_jax_customjvp
    if _gammainc_jax_customjvp is None:
        import jax
        import jax.numpy as jnp
        import jax.scipy.special as _jsp

        @jax.custom_jvp
        def _fn(a, x):
            return _gammainc_hand_coded_jax(a, x)

        @_fn.defjvp
        def _fn_jvp(primals, tangents):
            a, x = primals
            _, dx = tangents
            val = _fn(a, x)
            gamma_of_a = jnp.exp(_jsp.gammaln(a))
            dP_dx = x ** (a - 1.0) * jnp.exp(-x) / gamma_of_a
            return val, dP_dx * dx

        _gammainc_jax_customjvp = _fn
    return _gammainc_jax_customjvp


def exp1(x, xp=None):
    """Backend-dispatched exponential integral E_1(x) for x > 0.

    Uses ``scipy.special.exp1`` on the numpy path and, on the JAX
    path, a ``custom_jvp``-wrapped ``jax.scipy.special.exp1`` whose
    gradient rule is the analytic ``-exp(-x) / x`` (issue #207).
    Forward is still ``jsp.exp1`` (which fails on 0-D scalar inputs
    per an upstream JAX bug, so callers pass at least 1-D arrays);
    the JVP override keeps the grad graph small and compile-fast.
    Used by the MF5 LF=12 (Madland-Nix) fission-spectrum kernel.
    """
    if xp is None or getattr(xp, 'name', None) == 'numpy':
        from scipy.special import exp1 as _scipy_exp1
        return _scipy_exp1(x)
    if getattr(xp, 'name', None) == 'jax':
        return _get_exp1_jax_customjvp()(x)
    if hasattr(xp, 'exp1'):
        return xp.exp1(x)
    from scipy.special import exp1 as _scipy_exp1
    return _scipy_exp1(x)


def gammainc(a, x, xp=None):
    """Backend-dispatched regularised lower incomplete gamma
    P(a, x) = gamma(a, x) / Gamma(a). ``scipy.special.gammainc``
    on numpy; on JAX a ``custom_jvp``-wrapped
    ``jax.scipy.special.gammainc`` whose gradient rule wrt x is the
    analytic ``x^(a-1) exp(-x) / Gamma(a)`` (issue #207). Grad wrt
    a is left at zero (not used in the physics).
    """
    if xp is None or getattr(xp, 'name', None) == 'numpy':
        from scipy.special import gammainc as _scipy_gi
        return _scipy_gi(a, x)
    if getattr(xp, 'name', None) == 'jax':
        import jax.numpy as jnp
        a_arr = jnp.asarray(a, dtype=jnp.float64)
        x_arr = jnp.asarray(x)
        return _get_gammainc_jax_customjvp()(a_arr, x_arr)
    if hasattr(xp, 'gammainc'):
        return xp.gammainc(a, x)
    from scipy.special import gammainc as _scipy_gi
    return _scipy_gi(a, x)


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
