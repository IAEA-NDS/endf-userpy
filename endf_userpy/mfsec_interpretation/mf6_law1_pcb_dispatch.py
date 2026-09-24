"""MF6 LAW=1: multi-panel ``jax.grad`` wrt incident energy E via
``jax.custom_vjp`` + ``jax.pure_callback`` dispatch to JIT-compiled
per-panel batched kernels (issue #190 follow-up).

The "hand-tuned fast lane" for the multi-panel autodiff use case:

- One ``jax.pure_callback`` per call receives the whole ``(n_E,)``
  incident-energy array.
- Inside the callback, the panel assignment for each E is found by
  ``numpy.searchsorted`` (concrete E, µs).
- Es sharing a panel are grouped and dispatched to a JIT-compiled
  per-panel kernel. The per-panel input shape is picked from a fixed
  small set of powers of two (see ``_CHUNK_SIZES``) so the JIT cache
  stays bounded: with 4 panels and 5 sizes, at most 20 forward and
  20 backward compiles across the lifetime of a session.
- Partial chunks are padded with an in-range dummy E and the padded
  outputs are discarded on the return leg.
- ``jax.custom_vjp`` routes ``jax.grad`` back through the same
  numpy-lookup + jitted-panel pattern on the backward pass.

Why adaptive chunk sizes: the per-panel kernel's cost is ~linear in
the input length (kink-aware GL is per-E work, minimal vectorisation
between Es), so padding to a fixed large chunk wastes work
proportional to the padding count. Choosing the smallest chunk >=
panel Es count amortises callback overhead without wasting slots.

Trade-offs vs the tracer-panel-index
``integrate_law1_spectrum_multipanel_traced``:

- ``pure_callback`` breaks the JAX trace, so surrounding kernels
  cannot fuse across the callback boundary.
- First visit to a new (panel, chunk-size) compiles a fresh kernel
  (~3-5 s cold). Subsequent chunks of the same shape hit the cache.
- Outer ``jax.vmap`` over this entry point runs sequentially
  (``vmap_method='sequential'``); if this ever becomes important,
  upgrade the callback to accept an extra batch axis.
"""
from __future__ import annotations

import numpy as np

from . import mf6_law1_epintegral as _epi


# Fixed set of chunk sizes tried per panel. Powers of two keep the
# cardinality small (log2 up to max) while giving reasonable coverage.
# Ordered ascending; the dispatcher picks the smallest size >= panel
# Es count, and for count > _CHUNK_SIZES[-1] splits into repeated
# max-size chunks.
_CHUNK_SIZES = (4, 8, 16, 32, 64, 128)


# Module-level cache: (id(data), n_ep, to_lab, n_gl) -> cache.
# Keeping this at module scope means jitted per-panel kernels survive
# across calls (across grad steps in an optimizer). Without this,
# every call would rebuild the cache and recompile every panel visited.
_CACHE_REGISTRY = {}


def _get_cache(data, energies_out, to_lab, n_gl, xp):
    n_ep = int(xp.asarray(energies_out).shape[0])
    key = (id(data), n_ep, bool(to_lab), int(n_gl))
    if key not in _CACHE_REGISTRY:
        _CACHE_REGISTRY[key] = _PerPanelKernelCache(
            data, energies_out, to_lab, n_gl, xp,
        )
    return _CACHE_REGISTRY[key]


def _pick_chunk_size(n):
    """Smallest entry of ``_CHUNK_SIZES`` >= n, or the max entry if
    n exceeds it (in which case the caller splits into repeated
    max-size chunks)."""
    for s in _CHUNK_SIZES:
        if s >= n:
            return s
    return _CHUNK_SIZES[-1]


class _PerPanelKernelCache:
    """Cache of JIT-compiled batched per-panel kernels.

    Each entry maps ``(kind, panel, chunk_size)`` to a jitted
    function. One compile per (panel, chunk_size, kind) on first
    access; every subsequent chunk of the same shape reuses it.
    """

    def __init__(self, data, energies_out, to_lab, n_gl, xp):
        import jax
        self._data = data
        self._to_lab = to_lab
        self._n_gl = n_gl
        self._xp = xp
        self._energies_out = xp.asarray(energies_out)
        self._n_ep = int(self._energies_out.shape[0])
        self._cache = {}
        self._jax = jax

    def _make_forward_kernel(self, panel):
        xp = self._xp
        eout = self._energies_out
        data = self._data
        to_lab = self._to_lab
        n_gl = self._n_gl

        def fn(E_vec):
            return _epi.integrate_law1_spectrum(
                data, E_vec, eout, to_lab,
                xp=xp, n_gl=n_gl, panel_idx=panel,
            )

        return self._jax.jit(fn)

    def _make_grad_kernel(self, panel, chunk_size):
        fwd = self.forward(panel, chunk_size)

        def loss(E_vec, g_mat):
            return (fwd(E_vec) * g_mat).sum()

        return self._jax.jit(self._jax.grad(loss, argnums=0))

    def forward(self, panel, chunk_size):
        key = ('fwd', panel, chunk_size)
        if key not in self._cache:
            # jit is shape-polymorphic in the closure; a new JIT
            # instance per shape gives one XLA compile per shape.
            self._cache[key] = self._make_forward_kernel(panel)
        return self._cache[key]

    def grad_of_forward(self, panel, chunk_size):
        key = ('grad', panel, chunk_size)
        if key not in self._cache:
            self._cache[key] = self._make_grad_kernel(panel, chunk_size)
        return self._cache[key]


def _assign_panels(E_np, ei_mesh_np, n_pairs, e_min, e_max):
    panels = np.searchsorted(ei_mesh_np, E_np, side='right') - 1
    panels = np.clip(panels, 0, n_pairs - 1)
    in_range = (E_np >= e_min) & (E_np <= e_max)
    return panels, in_range


def _split_indices_into_chunks(idxs, max_size):
    """Given panel-member indices ``idxs``, split into a list of
    ``(sub_idxs, chunk_size)`` pairs where each ``sub_idxs`` fits
    into a chunk of that size. Uses adaptive sizing: single chunk
    of smallest fitting size when ``len(idxs) <= _CHUNK_SIZES[-1]``,
    else repeated max-size chunks (last one padded)."""
    n = len(idxs)
    if n == 0:
        return []
    max_size = _CHUNK_SIZES[-1]
    if n <= max_size:
        return [(idxs, _pick_chunk_size(n))]
    out = []
    for start in range(0, n, max_size):
        sub = idxs[start:start + max_size]
        out.append((sub, _pick_chunk_size(len(sub))))
    return out


def integrate_law1_spectrum_pcb(
    data, energies_in, energies_out, to_lab=True, xp=None, n_gl=10,
):
    """Multi-panel LAW=1 spectrum with ``jax.grad`` wrt E via
    numpy panel lookup + JIT-compiled per-panel batched kernels.

    Requires ``xp = array_ns.get_backend('jax')``. See module
    docstring for the dispatch pattern.

    Parameters
    ----------
    data : MF6Law1Data
    energies_in : (n_E,) jax array of incident energies (eV).
    energies_out : (n_Ep,) jax or numpy array of outgoing energies.
    to_lab : bool
    xp : jax adapter (from ``array_ns.get_backend('jax')``).
    n_gl : Gauss-Legendre order per subpanel (default 10).

    Returns
    -------
    (n_E, n_Ep) jax array. Same units as ``integrate_law1_spectrum``.
    """
    if xp is None or xp.name != 'jax':
        raise ValueError(
            'integrate_law1_spectrum_pcb requires xp=jax; got '
            f'xp={xp!r}. On numpy the single-panel per-E loop is '
            'strictly faster and does not need callbacks.'
        )

    import jax
    import jax.numpy as jnp

    ei_mesh_np = np.asarray(data.ei_mesh)
    e_min = float(ei_mesh_np[0])
    e_max = float(ei_mesh_np[-1])
    n_pairs = int(ei_mesh_np.shape[0]) - 1

    e_in_xp = xp.asarray(energies_in)
    n_E = int(e_in_xp.shape[0])

    if n_pairs < 1 or n_E == 0:
        return xp.zeros(
            (n_E, int(xp.asarray(energies_out).shape[0])),
            dtype=jnp.float64,
        )

    cache = _get_cache(data, energies_out, to_lab, n_gl, xp)
    n_ep = cache._n_ep

    def _numpy_forward_batch(E_arr):
        E_np = np.asarray(E_arr, dtype=np.float64)
        n = E_np.shape[0]
        result = np.zeros((n, n_ep), dtype=np.float64)
        panels, in_range = _assign_panels(
            E_np, ei_mesh_np, n_pairs, e_min, e_max,
        )
        if not in_range.any():
            return result
        for p in np.unique(panels[in_range]):
            idxs = np.where(in_range & (panels == p))[0]
            for sub, chunk in _split_indices_into_chunks(
                idxs, _CHUNK_SIZES[-1],
            ):
                actual = len(sub)
                Es_slot = np.empty(chunk, dtype=np.float64)
                Es_slot[:actual] = E_np[sub]
                if actual < chunk:
                    Es_slot[actual:] = E_np[sub[0]]
                kernel = cache.forward(int(p), chunk)
                rows = np.asarray(kernel(jnp.asarray(Es_slot)))
                result[sub] = rows[:actual]
        return result

    def _numpy_backward_batch(E_arr, g_arr):
        E_np = np.asarray(E_arr, dtype=np.float64)
        g_np = np.asarray(g_arr, dtype=np.float64)
        n = E_np.shape[0]
        result = np.zeros(n, dtype=np.float64)
        panels, in_range = _assign_panels(
            E_np, ei_mesh_np, n_pairs, e_min, e_max,
        )
        if not in_range.any():
            return result
        for p in np.unique(panels[in_range]):
            idxs = np.where(in_range & (panels == p))[0]
            for sub, chunk in _split_indices_into_chunks(
                idxs, _CHUNK_SIZES[-1],
            ):
                actual = len(sub)
                Es_slot = np.empty(chunk, dtype=np.float64)
                Gs_slot = np.zeros((chunk, n_ep), dtype=np.float64)
                Es_slot[:actual] = E_np[sub]
                Gs_slot[:actual] = g_np[sub]
                if actual < chunk:
                    Es_slot[actual:] = E_np[sub[0]]
                kernel = cache.grad_of_forward(int(p), chunk)
                dEs = np.asarray(
                    kernel(jnp.asarray(Es_slot), jnp.asarray(Gs_slot))
                )
                result[sub] = dEs[:actual]
        return result

    fwd_shape = jax.ShapeDtypeStruct((n_E, n_ep), jnp.float64)
    bwd_shape = jax.ShapeDtypeStruct((n_E,), jnp.float64)

    @jax.custom_vjp
    def _batched_kernel(E_arr):
        return jax.pure_callback(
            _numpy_forward_batch, fwd_shape, E_arr,
            vmap_method='sequential',
        )

    def _fwd(E_arr):
        rows = jax.pure_callback(
            _numpy_forward_batch, fwd_shape, E_arr,
            vmap_method='sequential',
        )
        return rows, (E_arr,)

    def _bwd(res, g):
        (E_arr,) = res
        dE = jax.pure_callback(
            _numpy_backward_batch, bwd_shape, E_arr, g,
            vmap_method='sequential',
        )
        return (dE,)

    _batched_kernel.defvjp(_fwd, _bwd)

    return _batched_kernel(e_in_xp)
