"""Array-namespace adapter for the resonance module.

Small (~150 LoC) abstraction so the same physics core runs with
numpy, JAX, or (planned) numba as the linear-algebra backend. The
goal is that a MLBW / Reich-Moore / RML reconstruction is written
ONCE against this adapter -- no `if backend == 'jax': lax.switch(...)
else: np.select(...)` sprinkled through the physics.

Design principles:

- **Namespace-passing over global state.** Every physics function
  takes an `xp` parameter (the returned adapter). No thread-local
  or global mutation -- makes it easy to mix backends in tests
  ("run the same case on numpy AND jax; compare bit-for-bit").
- **Delegate the common bits.** Universal ufuncs and array-creation
  routines (`sqrt`, `exp`, `where`, `zeros`, `asarray`, ...) are
  forwarded to the underlying `numpy` / `jnp` module via
  `__getattr__`. The adapter only NAMES the operations that differ
  between backends.
- **The differences that matter.** Explicit adapter methods are only
  provided for operations where numpy and JAX diverge in interface
  or semantics: `segment_sum`, `switch`, `scan`, and (for RML later)
  a per-block `solve`. That surface is small enough that a new
  backend (numba, PyTorch, ...) needs only ~50 lines of glue.
- **No fixed sizes.** The adapter does NOT propagate the JAX-only
  padding constraints; when JAX is the backend, callers can pad
  input arrays at their boundary (typically at the preprocessing
  stage) so the physics core still sees natural sizes.

Currently: numpy, JAX, and numba. The numba backend is a **signal**
backend: its adapter surface is numpy's (numba doesn't expose ops
individually), but its ``.name`` triggers physics modules to route
to their dedicated ``@njit`` kernels (see e.g.
:mod:`endf_userpy.mfsec_interpretation.mf2_interpretation_mlbw_numba`).
"""
from __future__ import annotations


class NumpyBackend:
    """Backend that dispatches through ``numpy``.

    Preferred default: no compile step, natural sizes, plays with
    the rest of the numpy scientific ecosystem. Used as the
    reference implementation against which the JAX backend is
    validated.
    """

    name = 'numpy'

    def __init__(self):
        import numpy as np  # deferred so pure-import cost stays low
        self._np = np

    # ---- Operations where backends genuinely differ. ----

    def segment_sum(self, data, segment_ids, num_segments):
        """Sum `data` over segments identified by `segment_ids`.

        Equivalent to `jax.ops.segment_sum`. Numpy uses `np.add.at`
        which is a scatter-add; O(n) and correct for repeated indices.
        """
        out = self._np.zeros(num_segments, dtype=data.dtype)
        self._np.add.at(out, segment_ids, data)
        return out

    def switch(self, index, funcs, *args):
        """Call `funcs[index](*args)`.

        Numpy version accepts a Python int index; branchless
        dispatch is caller's problem (usually not needed because
        Python loops are cheap here). JAX version uses
        `jax.lax.switch` for tracing compatibility.
        """
        return funcs[int(index)](*args)

    def scan(self, fn, init, xs):
        """Sequential scan: `carry_i+1, y_i = fn(carry_i, x_i)`.

        Returns `(final_carry, stacked_ys)`. If `fn` returns
        `(carry, None)` on every step, `stacked_ys` is None.
        """
        carry = init
        ys = []
        for x in xs:
            carry, y = fn(carry, x)
            ys.append(y)
        if all(y is None for y in ys):
            return carry, None
        return carry, self._np.stack(ys)

    def solve(self, a, b):
        """Solve `a @ x == b`. Delegates to `np.linalg.solve`."""
        return self._np.linalg.solve(a, b)

    # ---- Forwarders (universal to all array backends). ----

    def __getattr__(self, name):
        # `sqrt`, `exp`, `where`, `zeros`, `asarray`, `pi`, `float64`, ...
        return getattr(self._np, name)


class JaxBackend:
    """Backend dispatching to ``jax.numpy`` + ``jax.lax``.

    Enables trace-and-compile, autodiff, and (in principle) GPU
    execution -- at the cost of the fixed-shape / no-Python-loop
    constraints. When this backend is active, callers that construct
    the resonance data structures are expected to pad variable-size
    arrays at input; the physics core inside `xp.scan` / `xp.switch`
    stays trace-clean.
    """

    name = 'jax'

    def __init__(self):
        import jax
        import jax.numpy as jnp
        from jax import lax
        jax.config.update("jax_enable_x64", True)
        self._jax = jax
        self._jnp = jnp
        self._lax = lax

    def segment_sum(self, data, segment_ids, num_segments):
        return self._jax.ops.segment_sum(data, segment_ids, num_segments)

    def switch(self, index, funcs, *args):
        # `funcs` must be a fixed tuple; `index` may be a traced int.
        return self._lax.switch(index, funcs, *args)

    def scan(self, fn, init, xs):
        return self._lax.scan(fn, init, xs)

    def solve(self, a, b):
        return self._jnp.linalg.solve(a, b)

    def __getattr__(self, name):
        return getattr(self._jnp, name)


class NumbaBackend(NumpyBackend):
    """Signal backend: MLBW / (planned) SLBW / RM dispatch here go to
    dedicated ``@njit``-compiled kernels instead of the vectorised
    adapter path.

    Numba doesn't expose ufuncs / array creation via a namespace object
    the way ``numpy`` or ``jax.numpy`` do, so trying to translate every
    ``xp.sqrt`` / ``xp.where`` call individually would just wrap numpy
    with dispatch overhead. Instead, the physics modules keep the
    adapter path for numpy and JAX (one implementation, two backends),
    and provide a hand-written ``@njit`` fused kernel for numba
    -- see :mod:`endf_userpy.mfsec_interpretation.mf2_interpretation_mlbw_numba`.
    This backend's ``.name = 'numba'`` is the signal that triggers the
    kernel dispatch.

    The adapter surface itself inherits from :class:`NumpyBackend` so
    that any code that does hit ``xp.sqrt`` (etc.) still works
    correctly, just without a numba speedup.
    """

    name = 'numba'

    def __init__(self):
        super().__init__()
        try:
            import numba  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "numba backend requested but `numba` is not installed; "
                "install with `pip install numba`."
            )


_BACKENDS: dict[str, type] = {'numpy': NumpyBackend}
try:
    import jax  # noqa: F401
    _BACKENDS['jax'] = JaxBackend
except ImportError:
    pass  # JAX optional; numpy backend always available
try:
    import numba  # noqa: F401
    _BACKENDS['numba'] = NumbaBackend
except ImportError:
    pass  # numba optional; numpy backend always available


def get_backend(name: str = 'numpy'):
    """Instantiate the named backend.

    Prefer this over `set_backend` + module-level lookup: the
    returned adapter is cheap (~one lazy import), safe to pass to
    every physics function, and encourages tests that exercise
    both backends in the same session::

        for xp in (get_backend('numpy'), get_backend('jax')):
            xs = reconstruct_mlbw(data, energies, xp=xp)
            ...
    """
    if name not in _BACKENDS:
        raise ValueError(
            f'unknown backend {name!r}; known: {list(_BACKENDS)}. '
            f'JAX support requires the `jax` package to be installed.'
        )
    return _BACKENDS[name]()


def available_backends() -> list[str]:
    """List backends registered at import time."""
    return list(_BACKENDS)
