"""Array-namespace adapter for the resonance module.

Small (~200 LoC) abstraction so the same physics core runs with
numpy, JAX, or (opportunistically) numba as the linear-algebra backend.
The goal is that a MLBW / Reich-Moore / RML reconstruction is written
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

**Algebra vs accelerator.** Numpy and JAX are *algebras* (array
semantics). Numba is an *accelerator* (JIT of functions already
written against numpy semantics). The adapter classes encode both
axes jointly in their class identity so inconsistent combinations
("jax algebra + numba required") are not spellable.

Each backend class exposes three accelerator-policy methods
(see each method's docstring for semantics):

- ``wants_accelerator(name)``: does this backend take the
  accelerator branch when the accelerator is available?
- ``accelerator_available(name)``: pure runtime probe for whether
  the accelerator is importable.
- ``raise_if_needed_but_missing(name)``: no-op or raise, depending
  on whether this backend's contract requires the accelerator.

Resonance-reconstruction dispatch sites use the three-method API
rather than branching on ``xp.name``; see
:mod:`endf_userpy.quantities_mt_zap.resonance_composition`.
"""
from __future__ import annotations


_SUPPORTED_ACCELERATORS = ('numba',)


def _validate_accelerator(name: str) -> None:
    if name not in _SUPPORTED_ACCELERATORS:
        raise ValueError(
            f'unknown accelerator {name!r}; '
            f'supported: {_SUPPORTED_ACCELERATORS}'
        )


def _inside_jit_trace() -> bool:
    """True iff the caller sits inside an active ``@jax.jit`` /
    ``jax.grad`` trace. Falls back to False when jax is unavailable
    or the detection helper disappears in a future release; the
    eager fallback path is always correct, just not jit-optimal.
    Uses the private ``jax._src.core.trace_state_clean`` helper
    because jax has no public ``is_tracing()`` as of jax 0.5.x.
    """
    try:
        from jax._src.core import trace_state_clean
    except Exception:
        return False
    try:
        return not trace_state_clean()
    except Exception:
        return False


def _numba_importable() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


class NumpyBackend:
    """Backend that dispatches through ``numpy``.

    Explicit "numpy, no accelerators" when selected directly via
    ``RunOptions(backend='numpy')``. Shares its array-op surface with
    :class:`NumbaBackend` and :class:`AutoBackend` (both subclass it);
    the three differ only in their accelerator-policy methods.
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

    def map(self, fn, xs):
        """Independent-iteration map: ``y_i = fn(x_i)``, with ``xs``
        carrying iteration along its leading axis; returns the
        ``y_i`` stacked along that axis.

        ``xs`` may be a single array (iterated as ``xs[0], xs[1],
        ...``) or a tuple/list of arrays sharing a common leading
        axis (iterated zip-style, with ``fn`` receiving the same
        tuple shape as ``xs``). NumpyBackend runs this as a
        Python for-loop with ``np.stack`` on the outputs; this is
        the behavioural counterpart of ``jax.lax.map`` on
        :class:`JaxBackend`.
        """
        if isinstance(xs, (tuple, list)):
            n = int(xs[0].shape[0])
            ys = [fn(tuple(x[i] for x in xs)) for i in range(n)]
        else:
            n = int(xs.shape[0])
            ys = [fn(xs[i]) for i in range(n)]
        return self._np.stack(ys)

    def solve(self, a, b):
        """Solve `a @ x == b`. Delegates to `np.linalg.solve`."""
        return self._np.linalg.solve(a, b)

    # ---- Accelerator policy (base: no acceleration). ----

    def wants_accelerator(self, name: str) -> bool:
        """False for explicit numpy: the user said no acceleration."""
        _validate_accelerator(name)
        return False

    def accelerator_available(self, name: str) -> bool:
        """Pure runtime probe: is the accelerator importable right now?

        Shared across all subclasses; AutoBackend wraps this to fire
        a one-shot missing-numba warning.
        """
        _validate_accelerator(name)
        if name == 'numba':
            return _numba_importable()
        return False

    def raise_if_needed_but_missing(self, name: str) -> None:
        """No-op: this backend has no accelerator requirement."""
        _validate_accelerator(name)

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

    JAX cannot interoperate with numba (traced arrays vs numpy
    arrays), so every accelerator-policy method here answers "no":
    ``wants_accelerator`` is False (jax never takes the numba
    branch), ``accelerator_available`` reports jax's own view
    (always False for numba) regardless of whether numba is
    installed system-wide, and ``raise_if_needed_but_missing`` is a
    no-op (jax never requires numba).
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

    def map(self, fn, xs):
        """Independent-iteration map; see :meth:`NumpyBackend.map`.

        Under an active ``@jax.jit`` / ``jax.grad`` trace this
        dispatches to ``jax.lax.map`` so XLA sees one body
        regardless of iteration count. In eager mode (no active
        trace) ``jax.lax.map`` has to compile per invocation,
        which on complex bodies can exhaust the LLVM memory
        allocator -- there we fall back to a Python for-loop with
        ``jnp.stack``, letting jax's per-op cache amortise the
        compile across elements with matching shape. The
        detection uses the private ``jax._src.core.trace_state_clean``
        helper (there is no public ``is_tracing()`` as of jax
        0.5.x); if that symbol disappears, we fall back to the
        eager path, which is always correct.
        """
        if _inside_jit_trace():
            return self._lax.map(fn, xs)
        if isinstance(xs, (tuple, list)):
            n = int(xs[0].shape[0])
            ys = [fn(tuple(x[i] for x in xs)) for i in range(n)]
        else:
            n = int(xs.shape[0])
            ys = [fn(xs[i]) for i in range(n)]
        return self._jnp.stack(ys)

    def solve(self, a, b):
        return self._jnp.linalg.solve(a, b)

    def wants_accelerator(self, name: str) -> bool:
        _validate_accelerator(name)
        return False

    def accelerator_available(self, name: str) -> bool:
        _validate_accelerator(name)
        # jax algebra never dispatches through numba kernels; report
        # unavailable even if numba is installed, since from this
        # backend's point of view the accelerator cannot be used.
        return False

    def raise_if_needed_but_missing(self, name: str) -> None:
        _validate_accelerator(name)

    def __getattr__(self, name):
        return getattr(self._jnp, name)


class NumbaBackend(NumpyBackend):
    """Numpy algebra with numba acceleration as a hard requirement.

    ``RunOptions(backend='numba')`` resolves here. The constructor
    verifies that ``numba`` is importable; if not, it raises so the
    error surfaces at options construction rather than deep in a
    resonance dispatch site.

    Shares its array-op surface with :class:`NumpyBackend` so any
    ``xp.sqrt`` etc. call that bypasses the numba kernel still works
    (just without a numba speedup). The three accelerator-policy
    methods assert "numba is wanted, is present, and is required".
    """

    name = 'numba'

    def __init__(self):
        super().__init__()
        if not _numba_importable():
            raise RuntimeError(
                "numba backend requested but `numba` is not installed; "
                "install with `pip install numba`."
            )

    def wants_accelerator(self, name: str) -> bool:
        _validate_accelerator(name)
        return name == 'numba'

    # accelerator_available inherited (import probe): since __init__
    # verified numba is present, this returns True in normal use.
    # Kept inherited rather than overridden to True so that if numba
    # becomes unavailable mid-process the probe reflects reality.

    def raise_if_needed_but_missing(self, name: str) -> None:
        """Belt-and-suspenders: construction already verified numba
        presence, but if the module disappeared mid-process (cache
        blown, uninstalled) re-raise the same error the constructor
        would have raised.
        """
        _validate_accelerator(name)
        if name == 'numba' and not self.accelerator_available('numba'):
            raise RuntimeError(
                "numba backend was constructed but `numba` is no "
                "longer importable at dispatch time."
            )


class AutoBackend(NumpyBackend):
    """Numpy algebra with opportunistic numba acceleration.

    ``RunOptions(backend='auto')`` (the default) resolves here. The
    accelerator-policy methods encode "soft-use" semantics:

    - ``wants_accelerator('numba')`` is True, so resonance dispatch
      sites consider the numba branch.
    - ``accelerator_available('numba')`` probes the import and, on
      the first miss per Python session, emits a one-shot
      UserWarning naming the fallback. Dispatch sites then fall
      through to the numpy branch without raising.
    - ``raise_if_needed_but_missing`` is a no-op: AutoBackend never
      *requires* numba, only prefers it.

    ``name = 'numpy'`` deliberately: AutoBackend's visible array
    semantics ARE numpy's. Accelerator preference is queried via
    the three methods above, not via ``xp.name``. This keeps the
    pre-existing ``xp.name == 'numpy'`` branches (numpy-chunking,
    materialisation, jax-vs-numpy dispatch) working for
    ``backend='auto'`` without touching every call site. Use
    ``isinstance(xp, AutoBackend)`` if a test specifically needs
    to tell AutoBackend apart from NumpyBackend.
    """

    name = 'numpy'

    _warned_missing_numba = False  # class-level one-shot

    def wants_accelerator(self, name: str) -> bool:
        _validate_accelerator(name)
        return name == 'numba'

    def accelerator_available(self, name: str) -> bool:
        _validate_accelerator(name)
        available = super().accelerator_available(name)
        if (
            name == 'numba'
            and not available
            and not AutoBackend._warned_missing_numba
        ):
            AutoBackend._warned_missing_numba = True
            import warnings
            warnings.warn(
                "backend='auto' fell back to numpy for MF2 resonance "
                "reconstruction because numba is not installed. Install "
                "with `pip install numba` for the ~30x speedup on real "
                "actinide files. Pass RunOptions(backend='numpy') to "
                "silence this warning.",
                UserWarning, stacklevel=2,
            )
        return available


_BACKENDS: dict[str, type] = {
    'numpy': NumpyBackend,
    'auto': AutoBackend,
}
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
