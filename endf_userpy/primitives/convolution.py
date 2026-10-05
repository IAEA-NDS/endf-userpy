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

    ``'numba'`` falls back to the numpy path: numba has no native
    FFT and ``NumbaBackend`` inherits from ``NumpyBackend`` for the
    non-accelerated array ops, so scipy.signal.fftconvolve on the
    underlying numpy arrays is the correct behaviour.
    """
    if xp.name in ('numpy', 'numba'):
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
    window_kernel_widths : float or None, default None
        Target span of each windowed sub-convolution, in units of
        ``kernel_width``. ``None`` (default) resolves to ``inf``
        -> **single window** on both the concrete-eval_points and
        tracer-eval_points paths. The single-window default keeps
        jit compile time bounded regardless of eval_points layout:
        windowing under jit unrolls ``W x max_iter`` adaptive
        doublings into the XLA graph (one unroll per Python-level
        window iteration), which on wide, scattered eval_points
        pushes compile time from seconds to many minutes.

        Pass an explicit finite float to opt into windowing. Two
        partition strategies:

        * **concrete eval_points with no mesh_bounds**: point-centric
          interval merge. Each query point sprouts a
          ``[E - margin, E + margin]`` segment; overlapping segments
          are merged. Cost scales with the number of merged segments.
          Best eager memory savings on wide, scattered queries
          (e.g. actinide broadening); not jit-friendly because the
          Python-level merge loop unrolls at trace time.
        * **tracer eval_points, or concrete with ``mesh_bounds``
          given**: fixed-grid partition.
          ``W = ceil((eval_hi - eval_lo) / (
          window_kernel_widths * kernel_width))`` static windows,
          with ``eval_points`` split by index into contiguous
          slices under the "sorted eval_points" precondition. XLA
          sees a static graph; routed through ``xp.scan`` so the
          body traces once.

        Rules of thumb:

        * Leave as ``None`` for predictable jit compile times and
          simple eager workflows.
        * Pass ``50.0`` for maximum eager-memory savings on actinide
          broadening (point-centric merge; eager only).
        * Pass ``50.0`` + ``mesh_bounds=`` for memory savings under
          jit (fixed-grid, scan-based; opt-in jit memory reduction
          at the cost of a larger XLA trace graph).

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

    from ..mfsec_interpretation.mf6_law1_kernel import (
        _is_jax_tracer as _is_tracer,
    )

    def _build_mesh(win_emin, win_emax, n_intervals):
        """Numpy mesh when bounds are concrete (embeds as a jaxpr
        constant under @jax.jit), xp-native mesh only when bounds
        are jax tracers (``xp.scan`` body on jax). Keeps the trace
        graph small on the single-window path."""
        if _is_tracer(win_emin) or _is_tracer(win_emax):
            return xp.linspace(win_emin, win_emax, n_intervals + 1)
        return np.linspace(win_emin, win_emax, n_intervals + 1)

    def _single_window(win_eval_points, win_emin, win_emax, win_span):
        """One adaptive_convolve pass over a single [win_emin,
        win_emax] range at the eval points ``win_eval_points``.
        Closes over all tuning parameters of the outer call.

        ``win_span`` is the Python-float window width (always a
        static value even when ``win_emin`` / ``win_emax`` are jax
        tracers under ``xp.scan``). ``n_intervals`` and ``h`` are
        derived from ``win_span`` so the mesh *shape* stays static
        for the XLA trace; the mesh *values* fall back to
        ``np.linspace`` whenever the bounds are concrete to keep
        the jaxpr small (see ``_build_mesh``).
        """
        n_intervals = max(1, int(np.ceil(win_span / h0)))
        n_intervals = 1 << int(np.ceil(np.log2(n_intervals)))
        mesh = _build_mesh(win_emin, win_emax, n_intervals)
        h = win_span / n_intervals

        values = xp.asarray(f(mesh))
        if values.shape[-1] != mesh.shape[0]:
            raise ValueError(
                f"f(mesh) must return shape (..., {mesh.shape[0]}); "
                f"got {values.shape}"
            )

        def _convolve_and_sample(values, h, mesh):
            n_half = int(np.ceil(n_kernel_widths * kernel_width / h))
            delta = np.arange(-n_half, n_half + 1) * h
            k_vals = xp.asarray(kernel(delta))
            k_vals_b = k_vals.reshape(
                (1,) * (values.ndim - 1) + k_vals.shape,
            )
            conv = _fftconvolve(values, k_vals_b, xp) * h
            return _interp_last_axis(
                conv, mesh, win_eval_points, xp,
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

        n_intervals_cur = n_intervals
        for it in range(1, max_iter + 1):
            # Mesh doubling: build the new uniform mesh (numpy when
            # bounds are concrete; xp-native only when tracers, see
            # ``_build_mesh``). Compute f on the inserted midpoints
            # only. Shape stays static.
            n_intervals_cur = 2 * n_intervals_cur
            new_mesh = _build_mesh(win_emin, win_emax, n_intervals_cur)
            mid_values = f(new_mesh[1::2])
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
            h = win_span / n_intervals_cur

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
        # Auto: single-window default (``inf``). Opt in to windowing
        # by passing a finite value (e.g. ``50``) for memory
        # reduction on wide, resonance-dense queries. Defaulting to
        # single-window keeps jit-compile fast when ``eval_points``
        # is a concrete closure argument of a jitted function, where
        # auto-windowing would otherwise unroll ``W`` adaptive-
        # convolve bodies into the XLA graph.
        effective_wkw = float('inf')
    else:
        effective_wkw = float(window_kernel_widths)
    window_span = effective_wkw * kernel_width
    def _fixed_grid_segments(n_eval, emin, emax):
        """Partition sorted eval_points by index into slots that
        uniformly subdivide the eval_points' *own* range [emin+margin,
        emax-margin] (``mesh_bounds`` is the mesh extent, so
        ``emin + margin`` is the lower eval_points bound by
        convention). Returns ``(segments, uniform_window_span,
        per_window)``.

        - ``segments``: list of ``(slice_start, slice_stop,
          win_emin, win_emax)``.
        - ``uniform_window_span``: a static Python float, equal
          to ``win_emax - win_emin`` for every slot. Fed to
          :func:`_run_scan_windows` so the mesh shape stays static
          across scan iterations.
        - ``per_window``: static integer slice length so scan
          iterations have uniform tensor shape (the last slice is
          padded to this length before scan runs).
        """
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
        slot_span = (eval_hi - eval_lo) / n_segments
        uniform_window_span = slot_span + 2.0 * margin
        out = []
        for i in range(n_segments):
            start = i * per_window
            stop = min(start + per_window, n_eval)
            slot_lo = eval_lo + slot_span * i
            slot_hi = slot_lo + slot_span
            out.append((start, stop, slot_lo - margin, slot_hi + margin))
        return out, uniform_window_span, per_window

    _fixed_grid_window_span = None
    _fixed_grid_per_window = None
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
        segments, _fixed_grid_window_span, _fixed_grid_per_window = (
            _fixed_grid_segments(n_eval, emin, emax)
        )
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
            segments, _fixed_grid_window_span, _fixed_grid_per_window = (
                _fixed_grid_segments(n_eval, emin, emax)
            )
        elif effective_wkw == float('inf'):
            # Single-window shortcut (default): skip the point-centric
            # merge, build one segment covering every eval point. This
            # is the predictable fast path for jit workflows that
            # capture ``eval_points`` as a concrete Python closure
            # (and for eager callers who have not opted into windowing).
            segments = [(
                0, n_eval,
                float(eval_points[0]) - margin,
                float(eval_points[-1]) + margin,
            )]
        else:
            # Point-centric merge (opt-in via explicit finite
            # ``window_kernel_widths``): each eval point sprouts a
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
    # Fixed-grid partitions (same width across windows) can be routed
    # through ``xp.scan`` so the body is traced once; under jax that
    # keeps the XLA trace size O(1) in the window count rather than
    # unrolling W copies of the mesh-doubling loop. Point-centric
    # segments have varying widths (merged intervals) and stay on
    # the Python for-loop.
    use_scan = (
        _fixed_grid_window_span is not None
        and len(segments) > 1
    )
    if use_scan:
        return _run_scan_windows(
            segments, _fixed_grid_window_span,
            _single_window, eval_points, n_eval, xp,
        )
    if len(segments) == 1:
        start, stop, win_lo, win_hi = segments[0]
        win_span = float(win_hi) - float(win_lo)
        return _single_window(
            eval_points[start:stop], win_lo, win_hi, win_span,
        )

    # Multi-segment point-centric path. If all segments share the
    # same shape (``per_window`` and ``win_span``), batch them via
    # ``xp.map`` so the dispatcher produces one body for the whole
    # batch. The per-backend map strategy lives in ``xp.map`` on
    # each backend: numpy iterates with ``np.stack``, jax jit uses
    # ``jax.lax.map`` (one body in the XLA graph), jax eager falls
    # back to a jnp-stack loop. Non-uniform segments stay on the
    # Python for-loop below.
    if len(segments) > 1:
        first_pw = segments[0][1] - segments[0][0]
        first_span = segments[0][3] - segments[0][2]
        uniform = all(
            (s[1] - s[0]) == first_pw and
            abs((s[3] - s[2]) - first_span)
            <= 1e-9 * max(1.0, abs(first_span))
            for s in segments[1:]
        )
        if uniform:
            return _run_map_point_centric(
                segments, first_span, first_pw,
                _single_window, eval_points, xp,
            )
    parts = []
    for start, stop, win_lo, win_hi in segments:
        win_span = float(win_hi) - float(win_lo)
        parts.append(
            _single_window(
                eval_points[start:stop], win_lo, win_hi, win_span,
            ),
        )
    return xp.concatenate(parts, axis=-1)


def _run_map_point_centric(segments, win_span, per_window,
                           single_window, eval_points, xp):
    """Batch uniform-shape point-centric segments through
    ``xp.map`` so the ``_single_window`` body traces once under
    jax jit (and falls back to a Python for-loop under jax eager
    where ``jax.lax.map``'s per-invocation compile cost would
    exceed the dispatch savings).

    All segments share ``per_window`` points and ``win_span``
    width; only ``win_emin`` / ``win_emax`` vary per segment.
    """
    n_segs = len(segments)
    eval_batch = xp.stack(
        [eval_points[s[0]:s[1]] for s in segments], axis=0,
    )
    win_lo_arr = xp.asarray(
        [float(s[2]) for s in segments], dtype=xp.float64,
    )
    win_hi_arr = xp.asarray(
        [float(s[3]) for s in segments], dtype=xp.float64,
    )

    def body(args):
        win_eval, win_lo, win_hi = args
        return single_window(win_eval, win_lo, win_hi, win_span)

    stacked = xp.map(body, (eval_batch, win_lo_arr, win_hi_arr))
    moved = xp.moveaxis(stacked, 0, -2)
    flat_shape = moved.shape[:-2] + (n_segs * per_window,)
    return moved.reshape(flat_shape)


def _run_scan_windows(segments, window_span, single_window,
                      eval_points, n_eval, xp):
    """Run uniform-span windows via ``xp.scan`` and reassemble.

    All segments share the same ``window_span`` (a static Python
    float, equal to ``win_emax - win_emin`` for every slot). Each
    scan iteration consumes one static-size slice of a padded
    ``eval_points`` plus a per-iteration ``(win_lo, win_hi)`` pair,
    produces a static-shape output, and the stack is reshaped and
    sliced back to ``n_eval``.
    """
    W = len(segments)
    per_window = segments[0][1] - segments[0][0]
    # Pad eval_points to length W * per_window with trailing copies
    # of the last real value so the dropped tail doesn't contaminate
    # the kept output after slicing.
    n_padded = W * per_window
    if n_padded == n_eval:
        eval_pts_padded = eval_points
    else:
        pad = xp.full(
            (n_padded - n_eval,),
            eval_points[-1],
            dtype=eval_points.dtype,
        )
        eval_pts_padded = xp.concatenate([eval_points, pad], axis=0)

    # Pre-compute per-window (win_lo, win_hi) as static xp arrays
    # of shape (W,). Pass to scan as xs so the body sees one row
    # per iteration.
    win_lo_arr = xp.asarray(
        [float(s[2]) for s in segments], dtype=xp.float64,
    )
    win_hi_arr = xp.asarray(
        [float(s[3]) for s in segments], dtype=xp.float64,
    )
    starts_arr = xp.asarray(
        [s[0] for s in segments], dtype=xp.int64,
    )

    # Dynamic gather: ``xp.take`` with a static-shape index array
    # whose values depend on ``start`` works on both backends.
    # NumpyBackend: numpy integer indexing. JaxBackend: compiles to
    # ``jax.lax.gather`` with tracer-safe index arithmetic.
    offsets = xp.arange(per_window)

    def body(carry, idx_tuple):
        start, win_lo, win_hi = idx_tuple
        win_eval = xp.take(
            eval_pts_padded, start + offsets, axis=0,
        )
        y = single_window(win_eval, win_lo, win_hi, window_span)
        return carry, y

    # ``xp.scan`` on NumpyBackend expects xs as an iterable of
    # elements; on jax it uses lax.scan which takes a pytree with
    # a leading axis. Build xs as a list of per-iteration tuples so
    # both backends iterate the same way.
    if xp.name == 'jax':
        xs = (starts_arr, win_lo_arr, win_hi_arr)
        _, stacked = xp.scan(body, None, xs)
    else:
        xs = list(zip(
            [int(x) for x in starts_arr],
            [float(x) for x in win_lo_arr],
            [float(x) for x in win_hi_arr],
        ))
        _, stacked = xp.scan(body, None, xs)

    # stacked shape: (W, leading..., per_window). Move axis 0 to
    # second-to-last, flatten the two last axes, strip the padding.
    moved = xp.moveaxis(stacked, 0, -2)
    flat_shape = moved.shape[:-2] + (W * per_window,)
    flat = moved.reshape(flat_shape)
    return flat[..., :n_eval]


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

    if _is_jax_tracer(x_in):
        # Tracer mesh (e.g. ``xp.linspace(tracer_lo, tracer_hi, n)``
        # under ``xp.scan`` in adaptive_convolve's window loop):
        # keep every arithmetic step xp-native so x_in flows through
        # jit/grad. Mesh shape is still static (``n`` is a Python
        # int at trace time).
        n_in = int(x_in.shape[0])
        x0 = x_in[0]
        h = (x_in[-1] - x_in[0]) / (n_in - 1)
        idx_raw = xp.floor((x_out - x0) / h).astype(xp.int32)
        idx = xp.clip(idx_raw, 0, n_in - 2)
        x_left = xp.take(x_in, idx, axis=0)
        x_right = xp.take(x_in, idx + 1, axis=0)
        t = (x_out - x_left) / (x_right - x_left)
        left = xp.take(arr, idx, axis=-1)
        right = xp.take(arr, idx + 1, axis=-1)
        return left * (1.0 - t) + right * t

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
