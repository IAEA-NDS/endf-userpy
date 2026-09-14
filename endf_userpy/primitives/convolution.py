"""Generic 1D adaptive convolution.

`adaptive_convolve(f, kernel, eval_points, kernel_width=...)` returns
the convolution `(f * kernel)` evaluated at `eval_points`, using a
uniform internal mesh that is doubled until the values at `eval_points`
stop changing. Each doubling reuses every previously evaluated `f`
value at the original mesh nodes; only the inserted midpoints are
evaluated.

The primitive has no awareness of MF/MT or ENDF data structures.
"""
import numpy as np
from scipy.signal import fftconvolve


class ConvergenceWarning(UserWarning):
    pass


def adaptive_convolve(
    f,
    kernel,
    eval_points,
    *,
    kernel_width,
    h0=None,
    rtol=1e-3,
    atol=0.0,
    max_iter=6,
    min_iter=2,
    n_kernel_widths=5.0,
    richardson=True,
):
    """Compute `(f * kernel)(E)` at `eval_points` via adaptive FFT
    convolution on a doubling uniform internal mesh.

    Parameters
    ----------
    f : callable
        Function to be convolved. `f(E)` accepts a 1D array of length N
        and returns an ndarray of shape `(..., N)`. Leading axes are
        treated as independent slices and broadcast through the
        convolution; the same internal mesh is used for all slices.
    kernel : callable
        Convolution kernel. `kernel(delta_E)` accepts a 1D array and
        returns a 1D array of the same length. Should integrate to
        approximately 1 over its support so that a constant `f` maps
        to itself in the interior.
    eval_points : 1D array_like
        Output abscissae where the convolution is requested.
    kernel_width : float
        Characteristic kernel width (e.g. sigma for a Gaussian). Sets
        the initial mesh resolution and the kernel truncation range.
    h0 : float, optional
        Initial internal-mesh spacing. Default: `kernel_width / 2`.
    rtol, atol : float
        Convergence tolerance, applied as
        `max|R_k - R_{k-1}| <= rtol * max|R_k| + atol`.
    max_iter : int
        Maximum number of mesh doublings.
    min_iter : int
        Minimum number of doublings before convergence may be declared.
        Guards against false convergence at the initial resolution.
    n_kernel_widths : float
        Kernel is truncated at +/- `n_kernel_widths * kernel_width`;
        the internal mesh extends `eval_points` by the same margin on
        each side so the boundary does not pollute eval_points.
    richardson : bool
        If True, return `(4 R_k - R_{k-1}) / 3` after convergence
        (Richardson extrapolation for second-order linear-in-h error).
        If False, return `R_k`.

    Returns
    -------
    result : ndarray
        Convolved values at `eval_points`, shape `(..., len(eval_points))`.

    Raises
    ------
    ValueError
        If `eval_points` is not 1D or `kernel_width` is non-positive.

    Warns
    -----
    ConvergenceWarning
        If `max_iter` doublings are reached without satisfying the
        tolerance.
    """
    import warnings

    eval_points = np.asarray(eval_points, dtype=float)
    if eval_points.ndim != 1:
        raise ValueError("eval_points must be 1D")
    if kernel_width <= 0:
        raise ValueError("kernel_width must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    if h0 is None:
        h0 = kernel_width / 2.0

    margin = n_kernel_widths * kernel_width
    emin = float(eval_points.min()) - margin
    emax = float(eval_points.max()) + margin

    n_intervals = max(1, int(np.ceil((emax - emin) / h0)))
    n_intervals = 1 << int(np.ceil(np.log2(n_intervals)))  # round up to power of 2
    mesh = np.linspace(emin, emax, n_intervals + 1)
    h = (emax - emin) / n_intervals

    values = np.asarray(f(mesh))
    if values.shape[-1] != mesh.shape[0]:
        raise ValueError(
            f"f(mesh) must return shape (..., {mesh.shape[0]}); "
            f"got {values.shape}"
        )

    def _convolve_and_sample(values, h):
        n_half = int(np.ceil(n_kernel_widths * kernel_width / h))
        delta = np.arange(-n_half, n_half + 1) * h
        k_vals = np.asarray(kernel(delta))
        # fftconvolve requires matching ndim. Broadcast the kernel along
        # the leading axes of values by adding singleton dims.
        k_vals_b = k_vals.reshape((1,) * (values.ndim - 1) + k_vals.shape)
        conv = fftconvolve(values, k_vals_b, mode='same', axes=-1) * h
        return _interp_last_axis(conv, mesh, eval_points)

    R_prev_prev = None
    R_prev = _convolve_and_sample(values, h)
    converged = False

    for it in range(1, max_iter + 1):
        new_mesh = np.empty(2 * mesh.shape[0] - 1)
        new_mesh[0::2] = mesh
        new_mesh[1::2] = 0.5 * (mesh[:-1] + mesh[1:])

        new_values = np.empty(values.shape[:-1] + (new_mesh.shape[0],))
        new_values[..., 0::2] = values
        new_values[..., 1::2] = f(new_mesh[1::2])

        mesh = new_mesh
        values = new_values
        h = h / 2.0

        R_cur = _convolve_and_sample(values, h)

        diff = float(np.max(np.abs(R_cur - R_prev)))
        scale = float(np.max(np.abs(R_cur)))
        tol = rtol * scale + atol

        recent_ok = diff <= tol
        prior_ok = (
            R_prev_prev is None
            or np.max(np.abs(R_prev - R_prev_prev)) <= 4 * tol
        )
        if recent_ok and prior_ok and it >= min_iter:
            converged = True

        R_prev_prev = R_prev
        R_prev = R_cur

        if converged:
            break

    if not converged:
        warnings.warn(
            f"adaptive_convolve did not converge in {max_iter} doublings "
            f"(last diff={diff:.3e}, tol={tol:.3e})",
            ConvergenceWarning,
            stacklevel=2,
        )

    if richardson and R_prev_prev is not None:
        return (4.0 * R_prev - R_prev_prev) / 3.0
    return R_prev


def _interp_last_axis(arr, x_in, x_out):
    """Vectorised linear interpolation along the last axis.

    `arr` shape `(..., len(x_in))`, `x_in` strictly increasing,
    returns shape `(..., len(x_out))`.
    """
    idx = np.searchsorted(x_in, x_out, side='right') - 1
    idx = np.clip(idx, 0, x_in.shape[0] - 2)
    x_left = x_in[idx]
    x_right = x_in[idx + 1]
    t = (x_out - x_left) / (x_right - x_left)
    left = arr[..., idx]
    right = arr[..., idx + 1]
    return left * (1.0 - t) + right * t
