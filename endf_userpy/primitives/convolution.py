"""Generic 1D adaptive convolution.

`adaptive_convolve(f, kernel, eval_points, kernel_width=...)` returns
the convolution `(f * kernel)` evaluated at `eval_points`, using a
uniform internal mesh that is doubled until the values at `eval_points`
stop changing. Each doubling reuses every previously evaluated `f`
value at the original mesh nodes; only the inserted midpoints are
evaluated.

The primitive has no awareness of MF/MT or ENDF data structures.

Backend-agnostic (issue #169): pass ``xp=array_ns.get_backend('jax')``
to route the values arrays and the FFT convolution through JAX so
tracers in ``f`` outputs or in ``kernel`` closures reach the
convergence loop's output. ``eval_points`` and ``kernel_width`` are
treated as concrete (they drive mesh construction and truncation
bounds); the convergence-tolerance check is evaluated on a
materialised host copy so the Python loop control flow does not
depend on tracers.
"""
import numpy as np
from scipy.signal import fftconvolve

from . import array_ns


class ConvergenceWarning(UserWarning):
    pass


def _fftconvolve(values, kernel_vals, xp):
    """Backend-dispatched fftconvolve(mode='same', axes=-1).

    scipy.signal.fftconvolve broadcasts the leading (non-axis)
    dimensions; jax.scipy.signal.fftconvolve requires them to match
    exactly. Broadcast the kernel explicitly before the JAX call.
    """
    if xp.name == 'numpy':
        return fftconvolve(values, kernel_vals, mode='same', axes=-1)
    if xp.name == 'jax':
        from jax.scipy.signal import fftconvolve as jax_fftconvolve
        target_shape = values.shape[:-1] + kernel_vals.shape[-1:]
        kernel_bcast = xp.broadcast_to(kernel_vals, target_shape)
        return jax_fftconvolve(values, kernel_bcast, mode='same', axes=-1)
    raise ValueError(f'unsupported xp backend: {xp.name!r}')


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
    xp=None,
    mesh_bounds=None,
    chunk_size=1024,
    window_kernel_widths=None,
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
    xp : optional
        Array-namespace adapter from
        :func:`endf_userpy.primitives.array_ns.get_backend`. ``xp=None``
        (default) is numpy. Passing a JAX adapter dispatches the FFT
        through ``jax.scipy.signal.fftconvolve`` and keeps the values
        arrays xp-native, so tracers in ``f`` outputs or in ``kernel``
        closures propagate through to the returned array. The mesh
        itself always stays numpy (its shape must be static at trace
        time); only the final interpolation onto ``eval_points`` runs
        through ``xp`` and therefore supports tracer eval_points.
    mesh_bounds : (float, float), optional
        ``(emin, emax)`` bounds for the internal convolution mesh.
        Required when ``eval_points`` is a jax tracer (under
        ``@jax.jit`` or ``jax.grad`` wrt eval_points), since deriving
        the bounds from ``eval_points.min() / .max()`` would need to
        materialise the tracer. Also useful for concrete eval_points
        when the caller wants a fixed mesh shape across calls (e.g.
        to reuse a jit-compiled graph). When None (default) and
        eval_points is concrete, bounds are derived as before via
        ``eval_points.min() - margin`` / ``.max() + margin``. When
        passed, the ``n_kernel_widths * kernel_width`` margin is
        assumed to be already included; do not double-count.
    chunk_size : int, default 1024
        Memory-cap chunk size applied at two independent stages:

        1. **f-eval chunking** -- the internal mesh is split into
           chunks of this size, ``f`` is evaluated on each chunk,
           and the results are concatenated along the last axis.
           Caps the peak size of the (..., n_mesh) tensor that
           ``f`` materialises (the MF6 LAW=1 kernel's
           ``(n_Ein, n_mesh, n_mu)`` reconstruction grows to GB of
           intermediates on a resonance-dense file after a few mesh
           doublings).
        2. **fftconvolve chunking** -- the flattened leading axes
           of ``values`` are processed in blocks of this size
           before the FFT convolution, with results reassembled by
           reshape. Each FFT invocation allocates a complex128
           workspace of shape ``(block, n_mesh)``, so chunking the
           leading axes caps that too.

        Both splits are numerically no-ops; they only trade peak
        memory for a small Python loop. Set to ``None`` to disable
        both.
    window_kernel_widths : float or None, default None
        Target span of each windowed sub-convolution, in units of
        ``kernel_width``. ``None`` auto-selects per backend state:

        * **concrete eval_points** (eager numpy / eager jax):
          auto value is ``50.0``; function partitions the query
          via point-centric interval merge (each query point
          sprouts a ``[E - margin, E + margin]`` segment; overlapping
          segments are merged). Cost is proportional to the number
          of merged disjoint segments, not the span of ``eval_points``.
        * **tracer eval_points** (``@jax.jit`` / ``jax.grad``):
          auto value is ``inf`` -> single-window. This preserves
          the pre-windowing XLA trace graph; windowing under jit
          unrolls ``W x max_iter`` adaptive doublings at trace
          time, which can push jit compile time from seconds to
          many minutes for scattered ``eval_points``. Pass an
          explicit float to opt into windowing under jit (memory
          reduction at the cost of compile time).

        When set to an explicit float, both branches use fixed-grid
        partition: ``W = ceil((eval_hi - eval_lo) / (
        window_kernel_widths * kernel_width))`` static windows,
        with ``eval_points`` split by index into contiguous slices
        under the "sorted eval_points" precondition. XLA sees a
        static graph.

        Pass ``inf`` or a very large value (e.g. ``1e9``) to force
        single-window behaviour explicitly.

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

    if xp is None:
        xp = array_ns.get_backend('numpy')

    from ..mfsec_interpretation.mf6_law1_kernel import _is_jax_tracer
    is_tracer = _is_jax_tracer(eval_points)
    if not is_tracer:
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

    def _single_window(win_eval_points, win_emin, win_emax):
        """One adaptive_convolve pass over a single [win_emin,
        win_emax] range at the eval points ``win_eval_points``.
        Closes over all tuning parameters of the outer call."""
        n_intervals = max(1, int(np.ceil((win_emax - win_emin) / h0)))
        n_intervals = 1 << int(np.ceil(np.log2(n_intervals)))
        mesh = np.linspace(win_emin, win_emax, n_intervals + 1)
        h = (win_emax - win_emin) / n_intervals

        def _f_chunked(points_1d):
            if chunk_size is None or points_1d.shape[0] <= chunk_size:
                return f(points_1d)
            parts = []
            for start in range(0, points_1d.shape[0], chunk_size):
                parts.append(f(points_1d[start:start + chunk_size]))
            return xp.concatenate(parts, axis=-1)

        values = _f_chunked(mesh)
        if xp.name == 'numpy':
            values = np.asarray(values)
        if values.shape[-1] != mesh.shape[0]:
            raise ValueError(
                f"f(mesh) must return shape (..., {mesh.shape[0]}); "
                f"got {values.shape}"
            )

        def _convolve_and_sample(values, h, mesh):
            n_half = int(np.ceil(n_kernel_widths * kernel_width / h))
            delta = np.arange(-n_half, n_half + 1) * h
            k_vals = kernel(delta)
            if xp.name == 'jax':
                k_vals = xp.asarray(k_vals)
            else:
                k_vals = np.asarray(k_vals)

            def _conv_block(block):
                k_vals_b = k_vals.reshape(
                    (1,) * (block.ndim - 1) + k_vals.shape,
                )
                conv = _fftconvolve(block, k_vals_b, xp) * h
                return _interp_last_axis(
                    conv, mesh, win_eval_points, xp,
                )

            lead_shape = values.shape[:-1]
            n_lead = 1
            for s in lead_shape:
                n_lead *= int(s)
            if (chunk_size is None or values.ndim <= 1
                    or n_lead <= chunk_size):
                return _conv_block(values)
            flat = values.reshape(n_lead, values.shape[-1])
            parts = []
            for start in range(0, n_lead, chunk_size):
                parts.append(_conv_block(flat[start:start + chunk_size]))
            stacked = xp.concatenate(parts, axis=0)
            return stacked.reshape(
                lead_shape + (int(stacked.shape[-1]),),
            )

        def _to_host_max_abs(arr):
            try:
                return float(np.max(np.abs(np.asarray(arr))))
            except Exception:
                return float('nan')

        R_prev_prev = None
        R_prev = _convolve_and_sample(values, h, mesh)
        converged = False
        is_traced_local = False

        for it in range(1, max_iter + 1):
            new_mesh = np.empty(2 * mesh.shape[0] - 1)
            new_mesh[0::2] = mesh
            new_mesh[1::2] = 0.5 * (mesh[:-1] + mesh[1:])

            mid_values = _f_chunked(new_mesh[1::2])
            if xp.name == 'jax':
                new_values = xp.zeros(
                    values.shape[:-1] + (new_mesh.shape[0],),
                    dtype=values.dtype,
                )
                new_values = new_values.at[..., 0::2].set(values)
                new_values = new_values.at[..., 1::2].set(
                    xp.asarray(mid_values),
                )
            else:
                new_values = np.empty(
                    values.shape[:-1] + (new_mesh.shape[0],),
                )
                new_values[..., 0::2] = values
                new_values[..., 1::2] = mid_values

            mesh = new_mesh
            values = new_values
            h = h / 2.0

            R_cur = _convolve_and_sample(values, h, mesh)

            diff = _to_host_max_abs(R_cur - R_prev)
            scale = _to_host_max_abs(R_cur)
            tol = rtol * scale + atol

            if np.isnan(diff) or np.isnan(scale):
                is_traced_local = True
            else:
                recent_ok = diff <= tol
                prior_ok = (
                    R_prev_prev is None
                    or _to_host_max_abs(R_prev - R_prev_prev) <= 4 * tol
                )
                if recent_ok and prior_ok and it >= min_iter:
                    converged = True

            R_prev_prev = R_prev
            R_prev = R_cur

            if converged:
                break

        if not converged and not is_traced_local:
            warnings.warn(
                f"adaptive_convolve did not converge in {max_iter} "
                f"doublings (last diff={diff:.3e}, tol={tol:.3e})",
                ConvergenceWarning,
                stacklevel=3,
            )

        if richardson and R_prev_prev is not None:
            return (4.0 * R_prev - R_prev_prev) / 3.0
        return R_prev

    # ---- Segment partition ------------------------------------------
    # Each segment is (slice_start, slice_stop, win_emin, win_emax).
    # eval_points[slice_start:slice_stop] is the segment's query
    # subset; [win_emin, win_emax] is its internal mesh range
    # (already including the kernel margin on both sides).
    if window_kernel_widths is None:
        # Auto: 50 kernel widths per window for concrete, infinity
        # (= single window) for tracer. See docstring for the trace
        # time rationale.
        effective_wkw = float('inf') if is_tracer else 50.0
    else:
        effective_wkw = float(window_kernel_widths)
    window_span = effective_wkw * kernel_width
    def _fixed_grid_segments(n_eval, emin, emax):
        """Partition sorted eval_points by index into slots that
        uniformly subdivide the eval_points' *own* range [emin+margin,
        emax-margin] (``mesh_bounds`` is the mesh extent, so
        ``emin + margin`` is the lower eval_points bound by
        convention). Returns a list of
        (slice_start, slice_stop, win_emin, win_emax)."""
        eval_lo = emin + margin
        eval_hi = emax - margin
        if eval_hi <= eval_lo:
            raise ValueError(
                f"mesh_bounds=({emin}, {emax}) does not leave room "
                f"for the kernel margin on both sides "
                f"(2 * n_kernel_widths * kernel_width = {2 * margin})"
            )
        W = max(1, int(np.ceil((eval_hi - eval_lo) / window_span)))
        per_window = max(1, int(np.ceil(n_eval / W)))
        n_segments = int(np.ceil(n_eval / per_window))
        out = []
        for i in range(n_segments):
            start = i * per_window
            stop = min(start + per_window, n_eval)
            slot_lo = eval_lo + (eval_hi - eval_lo) * i / n_segments
            slot_hi = eval_lo + (eval_hi - eval_lo) * (i + 1) / n_segments
            out.append((start, stop, slot_lo - margin, slot_hi + margin))
        return out

    if is_tracer:
        # Fixed-grid partition from static mesh_bounds. Required under
        # @jax.jit / jax.grad wrt eval_points because segment count
        # and bounds must be static Python ints/floats at trace time.
        if mesh_bounds is None:
            raise TypeError(
                "adaptive_convolve was called with a jax tracer for "
                "eval_points but no mesh_bounds override. Under "
                "@jax.jit / jax.grad wrt eval_points the internal "
                "mesh bounds must be provided as static concrete "
                "floats via mesh_bounds=(emin, emax); the "
                "n_kernel_widths * kernel_width margin should be "
                "included on both sides."
            )
        emin, emax = float(mesh_bounds[0]), float(mesh_bounds[1])
        if emax <= emin:
            raise ValueError(
                f"mesh_bounds must satisfy emax > emin; got "
                f"({emin}, {emax})"
            )
        n_eval = int(eval_points.shape[0])
        if n_eval == 0:
            return xp.zeros((0,), dtype=xp.float64)
        segments = _fixed_grid_segments(n_eval, emin, emax)
    else:
        n_eval = int(eval_points.shape[0])
        if n_eval == 0:
            return np.zeros((0,), dtype=float)
        # Point-centric interval merge; requires eval_points sorted
        # so each merged segment covers a contiguous index range.
        if n_eval > 1 and np.any(np.diff(eval_points) < 0):
            raise ValueError(
                "adaptive_convolve requires eval_points to be sorted "
                "ascending when the point-centric windowing path is "
                "used (concrete eval_points, no mesh_bounds override)."
            )
        if mesh_bounds is not None:
            # User gave explicit bounds: fixed-grid partition (same
            # path as tracer, so caller-tuned mesh_bounds is honoured
            # as the global mesh range).
            emin, emax = float(mesh_bounds[0]), float(mesh_bounds[1])
            if emax <= emin:
                raise ValueError(
                    f"mesh_bounds must satisfy emax > emin; got "
                    f"({emin}, {emax})"
                )
            segments = _fixed_grid_segments(n_eval, emin, emax)
        else:
            # Point-centric merge: each eval point sprouts a
            # [p - margin, p + margin] interval; overlapping
            # intervals coalesce into disjoint segments.
            segments = []
            seg_start = 0
            seg_lo = float(eval_points[0]) - margin
            seg_hi = float(eval_points[0]) + margin
            for i in range(1, n_eval):
                p_lo = float(eval_points[i]) - margin
                p_hi = float(eval_points[i]) + margin
                if p_lo <= seg_hi:
                    if p_hi > seg_hi:
                        seg_hi = p_hi
                else:
                    segments.append((seg_start, i, seg_lo, seg_hi))
                    seg_start = i
                    seg_lo = p_lo
                    seg_hi = p_hi
            segments.append((seg_start, n_eval, seg_lo, seg_hi))

    # ---- Run segments and concatenate -------------------------------
    if len(segments) == 1:
        start, stop, win_lo, win_hi = segments[0]
        return _single_window(eval_points[start:stop], win_lo, win_hi)
    parts = []
    for start, stop, win_lo, win_hi in segments:
        parts.append(
            _single_window(eval_points[start:stop], win_lo, win_hi),
        )
    return xp.concatenate(parts, axis=-1)


def _interp_last_axis(arr, x_in, x_out, xp=None):
    """Vectorised linear interpolation along the last axis.

    ``arr`` shape ``(..., len(x_in))``, ``x_in`` strictly increasing
    (and always the numpy internal mesh built by
    :func:`adaptive_convolve`), returns shape
    ``(..., len(x_out))``.

    ``x_out`` may be numpy or a jax tracer. Because the source mesh
    ``x_in`` is a UNIFORM grid (``np.linspace``), we can locate the
    bracketing interval for each query in O(1) via
    ``idx = floor((x_out - x_in[0]) / h)`` without a searchsorted,
    which keeps the interp jit-safe and grad-safe under jax.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    from ..mfsec_interpretation.mf6_law1_kernel import _is_jax_tracer
    x_in_np = np.asarray(x_in)
    n_in = x_in_np.shape[0]
    x0 = float(x_in_np[0])
    h = float(x_in_np[-1] - x_in_np[0]) / (n_in - 1)

    if _is_jax_tracer(x_out):
        # Fully xp-native path: idx, t and the two gathers all flow
        # as tracers so jax.jit / jax.grad can differentiate through
        # x_out.
        idx_raw = xp.floor((x_out - x0) / h).astype(xp.int32)
        idx = xp.clip(idx_raw, 0, n_in - 2)
        x_in_xp = xp.asarray(x_in_np)
        x_left = xp.take(x_in_xp, idx, axis=0)
        x_right = xp.take(x_in_xp, idx + 1, axis=0)
        t = (x_out - x_left) / (x_right - x_left)
        left = xp.take(arr, idx, axis=-1)
        right = xp.take(arr, idx + 1, axis=-1)
        return left * (1.0 - t) + right * t

    # Concrete x_out: keep the numpy fast path (advanced-indexing on
    # arr's last axis, no take call). Behaviour bit-identical to the
    # pre-refactor implementation.
    x_out_np = np.asarray(x_out)
    idx = np.searchsorted(x_in_np, x_out_np, side='right') - 1
    idx = np.clip(idx, 0, n_in - 2)
    x_left = x_in_np[idx]
    x_right = x_in_np[idx + 1]
    t = xp.asarray((x_out_np - x_left) / (x_right - x_left))
    left = arr[..., idx]
    right = arr[..., idx + 1]
    return left * (1.0 - t) + right * t
