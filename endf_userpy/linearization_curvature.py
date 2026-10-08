"""Curvature-equidistribution linearisation of reconstructed cross
sections, driven by JAX second derivatives wrt incident energy.

This is a different strategy from the chord-error bisection in
:mod:`endf_userpy.linearization`. Instead of guessing a mesh and
subdividing until a midpoint probe stops complaining, it *computes*
the mesh from the second derivative of the cross section.

Why curvature is the right quantity
-----------------------------------
For linear interpolation on a segment of width ``h``, the classical
error identity is

.. math::

    \\max_{[x_i,\\,x_{i+1}]} |f - L| \\;\\le\\; \\frac{h^2}{8}
    \\max |f''|

so a target accuracy ``tol`` inverts directly into a local spacing
``h = sqrt(8 tol / |f''|)`` -- no iteration. Better: the mesh that
*minimises the point count* for a prescribed tolerance is the one
that equidistributes the monitor function

.. math::

    M(x) = \\sqrt{\\frac{|f''(x)|}
                       {8\\,(\\mathrm{tol_{abs}}
                        + \\mathrm{tol_{rel}}|f(x)|)}}

placing knots at equal increments of ``\\int M dx``, with the total
point count ``N \\approx \\int M dx`` (de Boor's equidistribution
principle for piecewise-linear approximation). Chord bisection
cannot reach this mesh even in principle: its spacings are forced
to binary subdivisions of whatever seed it started from.

Why JAX specifically
--------------------
``f''`` by finite differences is not usable here. Resonance widths
run to meV on top of Ein of hundreds of eV, and the second-difference
error scales as ``eps^-2`` against roundoff, so no single step size
works across the resolved-resonance region and the smooth background
at once. Measured on Nb-93 ``(n,total)`` at E=378 eV, a central
second difference with ``eps=0.378`` reports ``7038`` where the true
value (confirmed by an ``eps``-refinement study) is ``2227``. The AD
value is exact to roundoff at any energy, with no step to choose.

Autodiff additionally gives the *critical points*: Newton on
``f'(E) = 0`` using ``f'/f''`` locates true peak positions (shifted
off ``E_r`` by the shift factor and by interference) and the
interference dips, which sit at no pole energy. Measured, this turns
out **not** to improve the mesh -- see ``include_critical_points``
in :func:`linearize_curvature` for why that is the expected result
and what the option is still good for.

What this strategy actually buys
--------------------------------
Measured on Nb-93 over ``[1, 1000]`` eV at ``tol_rel=1e-3``, against
the chord-bisection driver in :mod:`endf_userpy.linearization`, with
accuracy checked by interpolating each mesh onto a 2e6-point log grid
and comparing to :func:`~endf_userpy.quantities.get_reaction_xs`:

=============  =====================  ====================
reaction       bisection              curvature
=============  =====================  ====================
``(n,g)``      8958 pts, err 1.0e-3   15342 pts, err 1.0e-3
``(n,total)``  4129 pts, err 6.1e-2   7142 pts, err 1.0e-3
=============  =====================  ====================

So this is **not** a uniform improvement. On ``(n,g)`` bisection wins
outright: capture is a sum of near-pure Lorentzian peaks, the
midpoint chord probe is a tight estimator there, and it reaches the
same accuracy with 40 percent fewer points. On ``(n,total)`` the
resonance-potential interference term puts sharp asymmetric dips
*between* poles, exactly where a midpoint probe is blindest; the
bisection driver reports ``status='converged'`` while sitting 61x
outside its own tolerance, and no amount of verification-mesh
refinement changes that verdict (50k / 400k / 2e6 agree).

The property worth having is therefore **calibration**, not raw
sparsity: ask for ``tol_rel`` and get it. Across two decades the
point count tracks the theoretical ``N ~ tol^(-1/2)`` to 0.3 percent
(2795 -> 8853 -> 27910 points for 1e-2 -> 1e-3 -> 1e-4, achieving
1.04e-2 / 9.95e-4 / 1.00e-4), whereas bisection on ``(n,total)``
returned 6.1e-2 for a 1e-3 request and 2.0e-4 for a 1e-4 request --
wrong in both directions.

Performance notes (measured, Nb-93 ``(n,total)``, 202 resonances)
-----------------------------------------------------------------
The composed ``get_reaction_xs`` is **not** ``jax.jit``-able today:
tracing halts in the MLBW preproc on a Python branch over a traced
value (``if gj_dif > 1e-30``). Eager JAX is ~440x slower than the
numba path, which makes AD look hopeless.

The fix is structural rather than numerical: the preproc depends only
on file data, never on Ein, so it is hoisted out of the traced region
and run once eagerly. What remains -- ``reconstruct(data, E, xp)``
plus the MF3 TAB1 background -- jits cleanly. Measured, 2000 points:

==========================  ==========
jit ``reconstruct``           0.026 s
jit ``(f, f', f'')``          0.190 s
numba composed (baseline)     0.022 s
==========================  ==========

So the jitted primal reaches numba parity, and the full derivative
trio costs about 7x a primal -- which buys a one-shot near-optimal
mesh instead of ~15 bisection passes.

Three consequences shape this module's design:

* **One compiled shape.** XLA recompiles per input shape, so a driver
  whose mesh size changes every pass would spend all its time
  compiling. Every evaluation here is chunked to a *fixed*
  ``chunk_size`` with the tail padded, so exactly one kernel is
  compiled per entry point regardless of mesh size.
* **Compile cost dominates, so kernels are cached and built
  lazily.** Compiling the twice-differentiated graph costs ~13 s
  against ~1.7 s for the primal, which is now the driver's largest
  single cost. Evaluators (and their compiled kernels) are therefore
  cached per ``(file, MT, chunk_size)`` in
  :data:`_EVALUATOR_CACHE`, the first-derivative kernel is compiled
  only if critical-point location is requested, and ``nl_max`` is
  capped at the file's own maximum L so the penetration-factor
  Newton recurrence is not unrolled further than the physics needs.
  Use :func:`clear_evaluator_cache` after mutating an
  ``endf_dict`` in place.
* **Bounded memory.** ``reconstruct`` materialises an
  ``(NE, nres)`` complex128 intermediate and only chunks itself on
  the numpy backend (646 MB at NE=200k here). The same fixed
  chunking bounds that to ``chunk_size * nres * 16`` bytes.

Scope
-----
Supports LRU=1 LRF=2 (MLBW), LRF=3 (Reich-Moore) and LRF=7 (RML
KRM=3) ranges composed additively onto the MF3 background, which is
the regime where linearisation is hard. LRU=2 URR ranges are **not**
reconstructed by this module's evaluator; a query range overlapping
an LSSF=0 URR range gets a loud warning and the returned cross
section omits the URR contribution there. Use
:func:`validate_against_quantities` to check this module's evaluator
against :func:`endf_userpy.quantities.get_reaction_xs` on any file
before trusting a mesh built from it.

Requires ``jax`` with 64-bit mode enabled; ``f''`` of a sharp
resonance in float32 is meaningless, so the module refuses to run
without it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .primitives import array_ns, reactions as reac, tab1 as tab1_mod
from .run_options import RunOptions


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------


@dataclass
class CurvatureLinearizationResult:
    """Bundle returned by :func:`linearize_curvature` when called
    with ``return_diagnostics=True``."""

    mesh: np.ndarray
    """Sorted incident energies (eV); endpoints are ``e_min`` /
    ``e_max``."""

    sigma: np.ndarray
    """Composed cross section (barn) at each mesh point."""

    monitor_integral: float
    """``\\int M dx`` over the pilot mesh. This is the theoretical
    segment count the equidistribution asks for, before mandatory
    knots are unioned in; compare against ``n_equidistributed`` to
    see whether ``max_points`` clipped the request."""

    n_pilot: int
    """Pilot-mesh size used to sample the monitor function."""

    n_equidistributed: int
    """Knots produced by equidistributing the monitor."""

    n_critical: int
    """Critical points (peaks / interference dips) added as
    mandatory knots, found by Newton on ``f' = 0``."""

    n_mandatory: int
    """Mandatory knots added for representation reasons: MF3 TAB1
    abscissae (slope kinks), MF2 range edges ``EL`` / ``EH``
    (genuine jumps, bracketed on both sides), and the endpoints."""

    max_chord_error: float
    """Largest ``|sigma_mid - (sigma_i + sigma_{i+1})/2|`` over the
    final mesh, in barn. The same estimator the bisection driver
    uses, reported for comparability."""

    max_bound_error: float
    """Largest ``h^2/8 * |f''|`` over the final mesh, in barn --
    the curvature error *bound* rather than a midpoint sample.
    ``f''`` is evaluated at segment midpoints, so this is a
    midpoint-sampled bound, not a certified one (see
    ``Caveats`` in :func:`linearize_curvature`)."""

    n_primal_evaluations: int
    """Count of cross-section values computed (pilot + Newton +
    final mesh + verification), as a cost proxy comparable across
    strategies."""

    status: str
    """``'converged'`` (every segment within tolerance by the
    chord test), ``'point_budget'`` (``max_points`` clipped the
    mesh), or ``'residual_error'`` (refinement passes exhausted
    with segments still over tolerance)."""

    history: list[dict] = field(default_factory=list)
    """Per-refinement-pass bookkeeping:
    ``{pass, n_mesh, n_add, max_chord_err, max_bound_err}``."""

    n_unrefinable: int = 0
    """Segments retired because they could not be subdivided: the
    proposed knot did not land strictly inside them. Non-zero means a
    genuine discontinuity (typically an MF2 range edge) sits inside the
    query range, and the mesh does not meet the tolerance across that
    one-ULP-wide interval."""

    urr_overlap: tuple[float, float] | None = None
    """``(EL, EH)`` of an LSSF=0 URR range overlapping the query
    window, if any. When set, ``sigma`` omits the URR contribution
    in that interval and the mesh there is driven by the MF3
    background alone."""


# ----------------------------------------------------------------------
# JIT-compiled evaluator with the preproc hoisted out of the trace
# ----------------------------------------------------------------------


def _require_x64():
    """JAX must be in 64-bit mode. ``f''`` of a resonance whose width
    is ~1e-4 of its centre energy is dominated by roundoff in
    float32, so a mesh built from it would be noise.

    Instantiating the JAX backend normally flips ``jax_enable_x64`` on
    (see :mod:`endf_userpy.primitives.array_ns`), so callers do not
    have to set it themselves; this is a guard against a caller who
    forced it back off, not a configuration step.
    """
    import jax
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError(
            'linearization_curvature requires 64-bit JAX, but '
            'jax_enable_x64 is off. Re-enable it with\n'
            "    import jax; jax.config.update('jax_enable_x64', True)\n"
            'Second derivatives of sharp resonances are meaningless in '
            'float32.'
        )


class _JitComposedEvaluator:
    """``(f, f', f'')`` of the composed cross section wrt Ein, with
    every ``endf_dict``-dependent step hoisted out of the JAX trace.

    Construction runs the per-range MF2 preprocessors eagerly (they
    read the parsed dict and depend only on file data), so the traced
    region contains nothing but array arithmetic:
    ``reconstruct(data, E, xp)`` summed over ranges with the
    half-open ``[EL, EH)`` mask, plus the MF3 TAB1 background.

    All evaluation is chunked to a fixed ``chunk_size`` with the tail
    zero-padded, so XLA compiles exactly one kernel per entry point
    no matter how the mesh size evolves.
    """

    def __init__(self, endf_dict, mt: int, *, chunk_size: int = 4096):
        # Resolve the backend first: constructing it is what turns
        # jax_enable_x64 on, so the check below has to follow it.
        self.xp = array_ns.get_backend('jax')
        _require_x64()
        import jax
        import jax.numpy as jnp
        from .quantities_mt_zap import resonance_composition as _rc
        from .mfsec_interpretation import (
            mf2_interpretation_mlbw as _mlbw,
            mf2_interpretation_mlbw_preproc as _mlbw_pre,
            mf2_interpretation_reichmoore as _rm,
            mf2_interpretation_reichmoore_preproc as _rm_pre,
            mf2_interpretation_rml as _rml,
            mf2_interpretation_rml_preproc as _rml_pre,
            mf3_interpretation,
        )

        self._jax = jax
        self._jnp = jnp
        self.endf_dict = endf_dict
        self.mt = int(mt)
        self.chunk_size = int(chunk_size)
        self.n_primal_evaluations = 0

        # --- hoisted preprocessing: one (reconstruct_fn, data, keys,
        #     EL, EH) record per supported LRU=1 range.
        self._ranges: list[tuple[Callable, Any, tuple[str, ...], float, float]] = []
        self.pole_table: list[tuple[float, float]] = []   # (E_r, Gamma_tot)
        self.range_edges: list[float] = []

        for iso_i, rng_i, rng in _rc._iter_lru1_ranges(endf_dict):
            lrf = int(rng.get('LRF', 0))
            if lrf == 2:
                keys = _rc._MLBW_MT_TO_KEYS.get(self.mt)
                pre, recon_mod = _mlbw_pre.mlbw_data_from_endf_dict, _mlbw
            elif lrf == 3:
                keys = _rc._RM_MT_TO_KEYS.get(self.mt)
                pre, recon_mod = _rm_pre.rm_data_from_endf_dict, _rm
            elif lrf == 7:
                keys = _rc._RML_MT_TO_KEYS.get(self.mt)
                pre, recon_mod = _rml_pre.rml_data_from_endf_dict, _rml
            else:
                warnings.warn(
                    f'linearization_curvature: LRU=1 iso={iso_i} rng={rng_i} '
                    f'LRF={lrf} is not reconstructed by this package; its '
                    f'resonance contribution is omitted from the mesh '
                    f'driver and from the returned cross section.',
                    UserWarning, stacklevel=3,
                )
                continue
            if not keys:
                # This MT receives no MF2 contribution (e.g. MT=4 under
                # MLBW); the background alone governs. Still record the
                # edges: the mask seam is a feature of the composed
                # function even when this range contributes nothing.
                self.range_edges.extend((float(rng['EL']), float(rng['EH'])))
                continue
            try:
                data = pre(
                    endf_dict, isotope_idx=iso_i, range_idx=rng_i, xp=self.xp,
                )
            except NotImplementedError as exc:
                warnings.warn(
                    f'linearization_curvature: skipping LRU=1 iso={iso_i} '
                    f'rng={rng_i} LRF={lrf} ({exc}). Its resonance '
                    f'contribution is omitted.',
                    UserWarning, stacklevel=3,
                )
                continue
            el, eh = float(rng['EL']), float(rng['EH'])
            # ``pnt_shf`` / ``phase`` unroll a Newton recurrence from
            # L=6 up to ``nl_max - 1``, and double-differentiating
            # triples that graph. The recurrence is skipped entirely
            # for ``nl_max <= 6`` (closed forms cover L<=5), so
            # capping it at the file's own maximum L is exactly
            # physics-preserving and trims the compile. Only MLBW
            # exposes the parameter.
            kwargs: dict[str, Any] = {}
            if lrf == 2:
                max_l = int(max(
                    np.max(np.asarray(data.res_l)),
                    np.max(np.asarray(data.ch_l)),
                ))
                kwargs['nl_max'] = max(6, max_l + 1)
            self._ranges.append(
                (recon_mod.reconstruct, data, keys, el, eh, kwargs)
            )
            self.range_edges.extend((el, eh))
            self.pole_table.extend(_poles_from_data(data, lrf))

        # --- URR overlap detection (not reconstructed here).
        self.urr_ranges: list[tuple[float, float]] = [
            (float(rng['EL']), float(rng['EH']))
            for _, _, rng in _rc._iter_lru2_lssf0_ranges(endf_dict)
        ]

        # --- MF3 background TAB1, built once.
        self._mf3_tab1 = None
        self.mf3_knots = np.asarray([], dtype=float)
        if 3 in endf_dict and self.mt in endf_dict[3]:
            self._mf3_tab1 = tab1_mod.from_endf_dict(endf_dict[3][self.mt])
            self.mf3_knots = np.asarray(self._mf3_tab1.x, dtype=float)
        else:
            warnings.warn(
                f'linearization_curvature: no MF3/MT={self.mt} in the file; '
                f'the composed cross section is the resonance '
                f'reconstruction alone.',
                UserWarning, stacklevel=3,
            )
        self._mf3_interp = mf3_interpretation

        # --- traced kernel: array arithmetic only.
        def _raw(E):
            xp = self.xp
            out = jnp.zeros_like(E)
            for reconstruct, data, keys, el, eh, kwargs in self._ranges:
                recon = reconstruct(data, E, xp, **kwargs)
                contrib = jnp.zeros_like(E)
                for k in keys:
                    if k in recon:
                        contrib = contrib + recon[k]
                # Half-open [EL, EH) matches resonance_composition's
                # seam convention (issue #149).
                in_range = (E >= el) & (E < eh)
                out = out + jnp.where(in_range, contrib, 0.0)
            if self._mf3_tab1 is not None:
                out = out + tab1_mod.interp(self._mf3_tab1, E, xp)
            return out

        self._raw = _raw

        def _fp(E):
            # f is elementwise in E, so a jvp with a ones-tangent
            # returns the diagonal of the Jacobian, i.e. f'(E_i) at
            # every point -- 1.4x the primal, against 6x for a vmap
            # over a scalar jacfwd.
            _, t = jax.jvp(_raw, (E,), (jnp.ones_like(E),))
            return t

        def _fpp(E):
            _, t = jax.jvp(_fp, (E,), (jnp.ones_like(E),))
            return t

        # Kernels are compiled lazily. XLA's compile of the twice-
        # differentiated graph costs ~13 s against ~1.7 s for the
        # primal (measured, Nb-93, 202 resonances, chunk 4096), so a
        # caller that never asks for f' must not pay to compile it.
        def _fppp(E):
            _, t = jax.jvp(_fpp, (E,), (jnp.ones_like(E),))
            return t

        self._kernels: dict[str, Any] = {}
        self._kernel_builders = {
            'value': lambda: jax.jit(_raw),
            'pair': lambda: jax.jit(lambda E: (_raw(E), _fpp(E))),
            'grad': lambda: jax.jit(lambda E: (_raw(E), _fp(E))),
            'trio': lambda: jax.jit(lambda E: (_raw(E), _fp(E), _fpp(E))),
            'quad': lambda: jax.jit(
                lambda E: (_raw(E), _fp(E), _fpp(E), _fppp(E))),
        }

    def _kernel(self, name: str):
        if name not in self._kernels:
            self._kernels[name] = self._kernel_builders[name]()
        return self._kernels[name]

    # -- chunked drivers: one compiled shape each -----------------

    def _chunked(self, fn, E, n_out: int):
        """Apply ``fn`` over ``E`` in fixed-size chunks, zero-padding
        the tail, and return ``n_out`` concatenated arrays."""
        jnp = self._jnp
        E = np.asarray(E, dtype=float).ravel()
        n = E.size
        if n == 0:
            return tuple(np.asarray([], dtype=float) for _ in range(n_out))
        cs = self.chunk_size
        outs = [[] for _ in range(n_out)]
        for start in range(0, n, cs):
            block = E[start:start + cs]
            pad = cs - block.size
            if pad:
                # Pad by repeating the last real value: keeps the
                # padded lanes inside the file's energy range so no
                # spurious NaN or out-of-range branch is taken, and
                # the results are sliced off below.
                block = np.concatenate([block, np.full(pad, block[-1])])
            res = fn(jnp.asarray(block))
            if n_out == 1:
                res = (res,)
            keep = cs - pad
            for i in range(n_out):
                outs[i].append(np.asarray(res[i])[:keep])
        self.n_primal_evaluations += n
        return tuple(np.concatenate(o) for o in outs)

    def value(self, E) -> np.ndarray:
        """Composed cross section at ``E``."""
        return self._chunked(self._kernel('value'), E, 1)[0]

    def pair(self, E):
        """``(f, f'')`` at ``E`` -- the monitor's inputs, without
        paying to compile the first-derivative kernel."""
        return self._chunked(self._kernel('pair'), E, 2)

    def grad(self, E):
        """``(f, f')`` at ``E`` -- the Hermite estimator's inputs."""
        return self._chunked(self._kernel('grad'), E, 2)

    def trio(self, E):
        """``(f, f', f'')`` at ``E``. Only needed for critical-point
        location; the monitor path uses :meth:`pair`."""
        return self._chunked(self._kernel('trio'), E, 3)

    def quad(self, E):
        """``(f, f', f'', f''')`` at ``E``. ``f'''`` bounds how much
        ``f''`` can vary inside a segment, which is what turns the
        curvature estimate into an actual upper bound."""
        return self._chunked(self._kernel('quad'), E, 4)


# Compiled XLA kernels and the eagerly preprocessed MF2 data are
# expensive to build (~0.4 s preproc + ~13 s to compile the twice-
# differentiated graph) and depend only on (file, MT, chunk size), so
# they are reused across calls. The parsed dict is held alongside the
# evaluator so its ``id`` cannot be recycled by the garbage collector
# while the entry is live.
_EVALUATOR_CACHE: dict[tuple, tuple[dict, '_JitComposedEvaluator']] = {}


def _get_evaluator(endf_dict, mt: int, chunk_size: int):
    key = (id(endf_dict), int(mt), int(chunk_size))
    hit = _EVALUATOR_CACHE.get(key)
    if hit is not None:
        return hit[1]
    ev = _JitComposedEvaluator(endf_dict, mt, chunk_size=chunk_size)
    _EVALUATOR_CACHE[key] = (endf_dict, ev)
    return ev


def clear_evaluator_cache() -> None:
    """Drop every cached evaluator and its compiled kernels.

    Call this after mutating an ``endf_dict`` in place, since the
    cache keys on the dict's identity, not its contents.
    """
    _EVALUATOR_CACHE.clear()


def _poles_from_data(data, lrf: int):
    """``(E_r, Gamma_total)`` pairs from a preprocessed range, used to
    place the pilot mesh densely enough to resolve the monitor."""
    er = np.asarray(data.res_er, dtype=float)
    if lrf == 2:
        gt = (np.abs(np.asarray(data.res_gn, dtype=float))
              + np.abs(np.asarray(data.res_gg, dtype=float))
              + np.abs(np.asarray(data.res_gf, dtype=float))
              + np.abs(np.asarray(data.res_gx, dtype=float)))
    elif lrf == 3:
        gt = (np.abs(np.asarray(data.res_gn, dtype=float))
              + np.abs(np.asarray(data.res_gg, dtype=float))
              + np.abs(np.asarray(data.res_gf1, dtype=float))
              + np.abs(np.asarray(data.res_gf2, dtype=float)))
    else:
        gt = np.sum(np.abs(np.asarray(data.res_gam, dtype=float)), axis=1)
    return [
        (float(a), float(b))
        for a, b in zip(er, gt)
        if np.isfinite(a) and np.isfinite(b) and b > 0.0
    ]


# ----------------------------------------------------------------------
# Mesh construction
# ----------------------------------------------------------------------


def _build_pilot_mesh(
    ev: _JitComposedEvaluator, e_min: float, e_max: float,
    *, per_width: int, width_span: float, background_per_decade: int,
) -> np.ndarray:
    """Pilot mesh on which the monitor function is sampled.

    The monitor only has to be *resolved*, not accurate: it varies on
    the ``Gamma`` scale, so a handful of points per width across a few
    widths around each pole is enough to integrate ``sqrt(|f''|)``
    faithfully. This is deliberately much cheaper than the final mesh.
    """
    pts = [float(e_min), float(e_max)]
    offsets = np.linspace(-width_span, width_span, 2 * per_width + 1)
    for e_r, gamma in ev.pole_table:
        if not (e_min - width_span * gamma <= e_r <= e_max + width_span * gamma):
            continue
        pts.extend((e_r + offsets * gamma).tolist())
    pts.extend(_mandatory_knots(ev, e_min, e_max).tolist())
    if background_per_decade > 0 and e_min > 0:
        n_bg = max(2, int(np.log10(e_max / e_min) * background_per_decade))
        pts.extend(np.geomspace(e_min, e_max, n_bg).tolist())
    arr = np.asarray(pts, dtype=float)
    arr = arr[(arr >= e_min) & (arr <= e_max)]
    return np.unique(arr)


def _build_pilot_mesh_territory(
    ev: _JitComposedEvaluator, e_min: float, e_max: float,
    *, core_pts: int, wing_pts_per_octave: int, max_wing_octaves: int,
    background_per_decade: int,
) -> np.ndarray:
    """Territory-tiled pilot mesh with monitor-aware wing spacing.

    Combines two ideas, each addressing a defect of
    :func:`_build_pilot_mesh`:

    * **Territory tiling** (from ``territory_adaptive`` in
      :mod:`endf_userpy.linearization`): each pole owns the interval
      between the midpoints to its neighbours, so sampling adapts to
      the *local* ratio of inter-pole spacing to ``Gamma`` instead of
      using one fixed window everywhere. A fixed ``+-4 Gamma`` window
      over-samples dense clusters (neighbouring windows overlap) and
      abandons the gaps between well-separated poles to the log
      background.
    * **Logarithmic wing spacing**, which is what the monitor itself
      asks for. Under a relative tolerance, a Lorentzian's monitor in
      the wings behaves as

      .. math::

          M \\propto \\frac{1}{\\Gamma_{1/2}\\,|u|}, \\qquad
          u = \\frac{E - E_r}{\\Gamma_{1/2}}

      because ``|f''| ~ u^-4`` and ``tol ~ |f| ~ u^-2`` shrink
      together. So ``\\int M du ~ ln|u|``: the wings contribute
      *logarithmically* to the required knot count, and resolving
      them needs points uniform in ``ln|u|``, not the linear
      ``Gamma/2`` stepping ``territory_adaptive`` uses for its own
      seed. Linear stepping out to a territory boundary hundreds of
      widths away costs O(u_max) points where O(log u_max) suffice.

    The core ``|u| <= 1`` is sampled uniformly, where ``M`` is
    flat-ish and the peak shape lives.
    """
    pts = [float(e_min), float(e_max)]

    poles = sorted(
        (er, gam) for er, gam in ev.pole_table
        if np.isfinite(er) and gam > 0
    )
    n = len(poles)
    for i, (er, gamma) in enumerate(poles):
        hw = 0.5 * gamma
        if not (hw > 0):
            continue
        left_bnd = e_min if i == 0 else 0.5 * (poles[i - 1][0] + er)
        right_bnd = e_max if i == n - 1 else 0.5 * (er + poles[i + 1][0])
        if right_bnd < e_min or left_bnd > e_max:
            continue

        # Core: uniform in u over [-1, 1].
        pts.extend((er + np.linspace(-1.0, 1.0, core_pts) * hw).tolist())

        # Wings: uniform in ln|u| from |u|=1 out to the territory
        # boundary, capped at max_wing_octaves doublings.
        for sign, bnd in ((-1.0, left_bnd), (1.0, right_bnd)):
            u_max = abs(bnd - er) / hw
            if not (u_max > 1.0):
                continue
            u_max = min(u_max, 2.0 ** max_wing_octaves)
            n_oct = max(1.0, np.log2(u_max))
            n_w = max(2, int(np.ceil(n_oct * wing_pts_per_octave)))
            us = np.geomspace(1.0, u_max, n_w)
            pts.extend((er + sign * us * hw).tolist())

    # Territory boundaries themselves: interference dips sit near the
    # inter-pole midpoints, and a dip missed in the *pilot*
    # underestimates the monitor there, which is the chord test's
    # blind spot one level up.
    pts.extend(0.5 * (poles[i][0] + poles[i + 1][0]) for i in range(n - 1))

    pts.extend(_mandatory_knots(ev, e_min, e_max).tolist())
    if background_per_decade > 0 and e_min > 0:
        n_bg = max(2, int(np.log10(e_max / e_min) * background_per_decade))
        pts.extend(np.geomspace(e_min, e_max, n_bg).tolist())

    arr = np.asarray(pts, dtype=float)
    arr = arr[(arr >= e_min) & (arr <= e_max)]
    return np.unique(arr)


def _discontinuity_energies(ev, e_min: float, e_max: float) -> np.ndarray:
    """Energies where the composed cross section genuinely jumps.

    Two sources, both common in real files:

    * **Doubled abscissae in the MF3 TAB1.** ENDF-6 encodes a
      discontinuity as a repeated ``x`` carrying two different ``y``.
      This is by far the most frequent case: across the 25-file corpus
      every resonance evaluation has some, from 3 (Pb-208) to 311
      (JENDL-5 Cu-63). They mark thresholds and, importantly, the
      RRR/URR seam, where MF3 is zero through the resolved range and
      takes over above it.
    * **MF2 range edges.** ``resonance_composition`` masks each range
      with ``(E >= EL) & (E < EH)``, so the reconstruction switches on
      at ``EL`` and off at ``EH``.

    The two are designed to cancel at a shared boundary (MF3 rises
    exactly where MF2 is masked off), but they do not cancel exactly:
    on U-235 at ``EH = 2250`` the composed value steps from 11.6254 to
    11.9489 barn, a 2.8 percent jump, because the URR average does not
    meet the RRR endpoint. That residual is physical, not a defect, and
    no mesh can interpolate across it.

    Interpolation laws that are themselves discontinuous (``INT=1``,
    histogram) would belong here too, but no corpus file uses one for
    a cross section, which is unsurprising: cross sections are
    physically continuous, so evaluators tabulate them lin-lin.
    """
    pts = []
    knots = ev.mf3_knots
    if knots.size > 1:
        dup = np.nonzero(np.diff(knots) == 0.0)[0]
        pts.extend(knots[dup].tolist())
    pts.extend(float(e) for e in ev.range_edges)
    arr = np.asarray(pts, dtype=float)
    arr = arr[(arr >= e_min) & (arr <= e_max)]
    return np.unique(arr)


def _apply_doubled_points(ev, mesh, sigma, disc, e_min, e_max):
    """Represent each discontinuity the way ENDF-6 does: the energy
    appears **twice**, carrying the left-hand and right-hand limits.

    ``np.interp`` honours this directly - a query below the point
    interpolates toward the first ordinate, one above starts from the
    second - so the jump is reproduced exactly instead of being
    smeared across the neighbouring segment.

    This replaces bracketing the edge with ``np.nextafter``, which
    merely approximated the same idea and left a one-ULP segment
    straddling the jump. Such a segment can never satisfy a chord test
    (every interior point is still on one side, so the error stays at
    half the jump for any ``h``) and cannot even be subdivided, since
    its midpoint is not representable strictly inside it.

    The function's own value at a jump energy is already the
    right-hand limit under both conventions in play: the MF2 mask is
    half-open ``[EL, EH)``, and :func:`primitives.tab1.interp` uses
    ``side='right'`` so a doubled abscissa selects the upper panel. So
    only the left limit needs a separate evaluation, one ULP below.
    """
    if disc.size == 0:
        return mesh, sigma
    idx = np.searchsorted(mesh, disc)
    safe = np.minimum(idx, mesh.size - 1)
    present = mesh[safe] == disc
    disc, idx = disc[present], idx[present]
    if disc.size == 0:
        return mesh, sigma

    sigma = sigma.copy()
    sigma[idx] = ev.value(disc)                       # right-hand limit
    # A jump sitting exactly at e_min needs no left-hand twin: that
    # limit lies outside the queried interval.
    take = disc > e_min
    if not np.any(take):
        return mesh, sigma
    left = ev.value(np.nextafter(disc[take], -np.inf))
    return (np.insert(mesh, idx[take], disc[take]),
            np.insert(sigma, idx[take], left))


def _mandatory_knots(
    ev: _JitComposedEvaluator, e_min: float, e_max: float,
) -> np.ndarray:
    """Knots the mesh must contain for *representation* reasons,
    independent of any accuracy target.

    Two kinds, both invisible to a curvature monitor:

    * **MF3 TAB1 abscissae.** A lin-lin background panel has
      ``f'' = 0`` inside and a slope discontinuity at each knot. AD
      faithfully reports zero curvature and would place no point
      there, yet interpolating across a kink is exactly what breaks
      the error bound. Putting the abscissa in the mesh makes the
      kink exactly representable instead.
    * **MF2 range edges.** ``resonance_composition`` masks each range
      with ``(E >= EL) & (E < EH)`` and a hard zero outside, so the
      composed function genuinely *jumps* at the seam unless the
      reconstruction vanishes there. A single knot cannot represent a
      jump, so both edges are bracketed with their floating-point
      neighbours.
    """
    pts = [float(e_min), float(e_max)]
    knots = ev.mf3_knots
    pts.extend(knots[(knots >= e_min) & (knots <= e_max)].tolist())
    # Range edges enter as single points here; the jump they carry is
    # represented by _apply_doubled_points, which gives each one a
    # second ordinate rather than bracketing it with nextafter.
    pts.extend(float(e) for e in ev.range_edges)
    arr = np.asarray(pts, dtype=float)
    return np.unique(arr[(arr >= e_min) & (arr <= e_max)])


_HERMITE_CRITERIA = ('hermite', 'hermite_fh')

_HERMITE_T = np.linspace(0.0, 1.0, 65)[1:-1]
"""Sample points in ``t`` for maximising the Hermite-minus-chord cubic.

``R`` is a cubic in ``t``, so its extremum could be had from a
quadratic root, but that needs per-segment branching on degenerate
coefficients. Sampling a fixed 63-point grid is pure vectorised numpy
over all segments at once, has no edge cases, and locates the
maximum of a cubic to well under a percent, which is far finer than
the estimator's own accuracy.
"""


def _hermite_residual_seg(lo, hi, f_lo, f_hi, fp_lo, fp_hi):
    """Max ``|H - L|`` per segment and where it occurs, for an
    arbitrary subset of segments given by their endpoints.

    ``H`` is the cubic Hermite interpolant through ``(f, f')`` at both
    knots; ``L`` is the linear chord. With ``t = (x - lo)/h``,
    ``s = (f_hi - f_lo)/h``, ``A = fp_lo - s`` and ``B = fp_hi - s``:

    .. math::

        H(t) - L(t) = h\\,t(1-t)\\,[\\,A - (A+B)\\,t\\,]

    At ``t = 1/2`` this is ``h(fp_lo - fp_hi)/8``, i.e. the classical
    ``-h^2/8 f''``: the same leading term, with the same constant, as
    the midpoint chord. The two therefore agree to leading order and
    differ only at ``O(h^3)``, which is why they measure as
    equivalent on any mesh fine enough to be near tolerance.

    Taking the subset by endpoint arrays (rather than slicing a whole
    mesh) is what lets the refinement skip converged segments.
    """
    h = hi - lo
    with np.errstate(invalid='ignore', divide='ignore'):
        s_ = (f_hi - f_lo) / h
        a_coef = fp_lo - s_
        b_coef = fp_hi - s_
        t = _HERMITE_T[:, None]
        r = h[None, :] * t * (1.0 - t) * (
            a_coef[None, :] - (a_coef + b_coef)[None, :] * t)
        absr = np.abs(r)
        absr = np.where(np.isfinite(absr), absr, 0.0)
        k = np.argmax(absr, axis=0)
    cols = np.arange(h.size)
    return absr[k, cols], lo + _HERMITE_T[k] * h


def _monitor(f: np.ndarray, fpp: np.ndarray, tol_abs: float, tol_rel: float,
             safety: float = 1.0):
    """``M = sqrt(|f''| / (8 tol(x)))`` with ``tol(x) = tol_abs +
    tol_rel |f(x)|``.

    Equidistributing ``M`` minimises the knot count subject to the
    local error bound ``h^2/8 |f''| <= tol(x)``, which is why the
    square root and the ``1/8`` belong inside the monitor rather than
    being applied afterwards.
    """
    tol = (tol_abs + tol_rel * np.abs(f)) / max(safety, 1e-30)
    tol = np.where(tol > 0, tol, np.inf)      # inf tol -> zero density
    m = np.sqrt(np.abs(fpp) / (8.0 * tol))
    return np.where(np.isfinite(m), m, 0.0)


def _equidistribute(x: np.ndarray, m: np.ndarray, max_points: int):
    """Place knots at equal increments of ``Phi(x) = \\int_a^x M dt``.

    Returns ``(knots, phi_total, clipped)``. ``phi_total`` is the
    number of segments the monitor asks for; ``clipped`` says whether
    ``max_points`` cut that request short.
    """
    dx = np.diff(x)
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (m[:-1] + m[1:]) * dx)])
    phi_total = float(phi[-1])
    if not np.isfinite(phi_total) or phi_total <= 0:
        # No curvature anywhere (pure linear background): the
        # mandatory knots already represent the function exactly.
        return np.asarray([x[0], x[-1]], dtype=float), 0.0, False
    n_seg = int(np.ceil(phi_total))
    clipped = False
    if n_seg + 1 > max_points:
        n_seg, clipped = max(1, max_points - 1), True
    # np.interp needs a strictly increasing table; flat stretches of
    # Phi (zero curvature) would otherwise collapse to one abscissa.
    phi_mono = np.maximum.accumulate(phi)
    phi_mono = phi_mono + np.linspace(
        0.0, 1e-12 * max(phi_total, 1.0), phi_mono.size,
    )
    targets = np.linspace(0.0, phi_mono[-1], n_seg + 1)
    return np.interp(targets, phi_mono, x), phi_total, clipped


def _find_critical_points(
    ev: _JitComposedEvaluator, x: np.ndarray, fp: np.ndarray, fpp: np.ndarray,
    *, newton_iters: int, e_min: float, e_max: float,
) -> np.ndarray:
    """Locate extrema by Newton on ``f'(E) = 0``.

    Brackets come from sign changes of ``f'`` on the pilot mesh; the
    initial guess is the secant root of ``f'`` inside each bracket.
    Newton steps use the AD ``f'/f''`` and are clipped back into their
    bracket so a near-zero ``f''`` at an inflection cannot throw the
    iterate across the mesh.

    This is the part a pole-seeded mesh cannot do: peak maxima sit off
    ``E_r`` (shift factor, interference) and the dips between
    overlapping resonances sit at no pole energy at all.
    """
    sign_change = np.signbit(fp[:-1]) != np.signbit(fp[1:])
    idx = np.nonzero(sign_change)[0]
    if idx.size == 0:
        return np.asarray([], dtype=float)
    lo, hi = x[idx], x[idx + 1]
    f_lo, f_hi = fp[idx], fp[idx + 1]
    denom = f_hi - f_lo
    guess = np.where(
        np.abs(denom) > 0, lo - f_lo * (hi - lo) / np.where(denom != 0, denom, 1.0),
        0.5 * (lo + hi),
    )
    cur = np.clip(guess, lo, hi)
    for _ in range(max(0, newton_iters)):
        _, d1, d2 = ev.trio(cur)
        step = np.where(np.abs(d2) > 0, d1 / np.where(d2 != 0, d2, 1.0), 0.0)
        cur = np.clip(cur - step, lo, hi)
    cur = cur[(cur >= e_min) & (cur <= e_max) & np.isfinite(cur)]
    return np.unique(cur)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def linearize_curvature(
    endf_dict: dict,
    reaction: str,
    e_min: float,
    e_max: float,
    *,
    tol_abs: float = 0.0,
    tol_rel: float = 1e-3,
    pilot_strategy: str = 'territory_log',
    pilot_per_width: int = 32,
    pilot_width_span: float = 4.0,
    pilot_background_per_decade: int = 160,
    pilot_core_pts: int = 9,
    pilot_wing_pts_per_octave: int = 2,
    pilot_max_wing_octaves: int = 12,
    monitor_safety: float = 2.0,
    criterion: str = 'chord',
    ref_rule: str = 'min',
    split: str = 'midpoint',
    include_critical_points: bool = False,
    newton_iters: int = 3,
    max_points: int = 200_000,
    refine_passes: int = 2,
    chunk_size: int = 4096,
    return_diagnostics: bool = False,
) -> Any:
    """Linearise ``reaction``'s composed cross section on
    ``[e_min, e_max]`` by equidistributing a JAX-derived curvature
    monitor.

    Parameters
    ----------
    endf_dict : dict
        Parsed ENDF-6 dict (``endf_parserpy`` layout).
    reaction : str
        Reaction string as accepted by
        :func:`endf_userpy.quantities.get_reaction_xs`.
    e_min, e_max : float
        Query range in eV, ``0 < e_min < e_max``.
    tol_abs : float, optional
        Absolute accuracy floor in barn. Default 0.
    tol_rel : float, optional
        Relative accuracy. Default ``1e-3``.
    pilot_strategy : {'uniform_gamma', 'territory_log'}, optional
        How to place the pilot samples.

        - ``'uniform_gamma'`` (default): ``2*pilot_per_width + 1``
          points uniform in ``Gamma`` units over
          ``[E_r +- pilot_width_span*Gamma]`` per pole, plus a log
          background.
        - ``'territory_log'``: each pole owns the interval between
          the midpoints to its neighbours (territory tiling, as in
          ``territory_adaptive``), its core ``|u| <= 1`` sampled
          uniformly and its wings uniformly in ``ln|u|`` out to the
          territory boundary. See
          :func:`_build_pilot_mesh_territory` for why log spacing is
          what the monitor asks for in the wings.

        Governed by ``pilot_core_pts``,
        ``pilot_wing_pts_per_octave`` and
        ``pilot_max_wing_octaves`` instead of ``pilot_per_width`` /
        ``pilot_width_span``.
    pilot_core_pts : int, optional
        ``'territory_log'`` only. Points uniform in ``u`` across the
        core ``|u| <= 1`` of each pole. Default 9.
    pilot_wing_pts_per_octave : int, optional
        ``'territory_log'`` only. Pilot points per doubling of
        ``|u|`` in the wings. Default 4.
    pilot_max_wing_octaves : int, optional
        ``'territory_log'`` only. Cap on wing doublings per side,
        bounding the cost on an isolated pole whose territory
        boundary is very far away. Default 12 (``|u|`` up to 4096).
    pilot_per_width, pilot_width_span : int, float, optional
        Pilot-mesh density: ``2 * pilot_per_width + 1`` points spread
        over ``[E_r - span*Gamma, E_r + span*Gamma]`` per pole. The
        pilot only has to *resolve* the monitor, not meet the
        tolerance, and it is evaluated once, so it is cheap relative
        to the final mesh.

        Density here pays for itself: a coarsely sampled
        ``sqrt(|f''|)`` overestimates its own integral, so the mesh
        comes out needlessly dense. Measured on Nb-93 ``(n,total)``
        at ``tol_rel=1e-3``, raising ``pilot_per_width`` 4 -> 16 ->
        32 -> 64 took the final mesh 8853 -> 5819 -> 4964 -> 4571
        points at an unchanged ~1.0e-3 achieved error, with flat wall
        time. Default 32 sits just past the knee.
    pilot_background_per_decade : int, optional
        Log-spaced pilot density away from poles. Default 160.
    monitor_safety : float, optional
        Divide the tolerance by this factor when building the monitor,
        making the equidistributed mesh denser by ``sqrt(safety)``.
        Default 2.0.

        This is the knob that makes the driver actually meet its
        stated tolerance, and it is not cosmetic. Plain
        equidistribution (``safety=1``) targets
        ``h^2/8 |f''| == tol`` using ``f''`` sampled at one point per
        segment, but the error bound needs ``max|f''|`` *over* the
        segment. Where the sampled value understates that maximum,
        the mesh comes out marginally too coarse and the achieved
        error lands just above target. Measured on Nb-93 at a 1e-3
        request:

        =========  ===================  ===================
        safety     ``(n,g)``            ``(n,total)``
        =========  ===================  ===================
        1.0        10538 pts, 1.001e-3  4933 pts, 1.003e-3
        1.5        10382 pts, 1.001e-3  4670 pts, 9.964e-4
        2.0        11698 pts, 9.993e-4  5292 pts, 9.960e-4
        3.0        14122 pts, 9.882e-4  6409 pts, 9.928e-4
        =========  ===================  ===================

        ``safety=2`` is the smallest value that clears the target on
        both, at 7-11 percent more points. It also does most of the
        chord refinement's job in advance: first-pass insertions on
        ``(n,g)`` fall 2393 -> 407 -> 180 -> 16 across the rows above,
        and at ``safety=4`` the refinement never fires at all.

        A properly certified alternative would sample ``f''`` several
        times per segment or use a closed-form per-pole envelope;
        this factor is the pragmatic stand-in.
    include_critical_points : bool, optional
        Add Newton-located extrema (``f' = 0``) as mandatory knots.
        Default False.

        This measured **neutral** on Nb-93 ``(n,total)``: 56 extrema
        located, identical achieved error with and without
        (1.004e-3 either way). That is the expected outcome rather
        than a disappointment -- equidistribution already places knot
        density proportional to ``sqrt(|f''|)``, which is maximal
        exactly at extrema, so once the mesh has thousands of knots
        the nearest one is already a small fraction of ``Gamma`` from
        each true extremum. Kept as a knob because it may matter at
        loose tolerances (where knots are sparse) and because the
        located extrema are independently useful as reported peak
        positions. Enabling it also forces the first-derivative
        kernel to be compiled.
    newton_iters : int, optional
        Newton steps per critical point. Default 3.
    max_points : int, optional
        Cap on the final mesh.
    refine_passes : int, optional
        Chord-test passes run after equidistribution, to absorb
        places where the midpoint-sampled bound understated the true
        curvature. Default 2. Zero trusts the monitor alone.
    chunk_size : int, optional
        Fixed evaluation block size. Governs both the single compiled
        XLA shape and the ``(chunk_size, nres)`` intermediate's
        memory. Default 4096.
    return_diagnostics : bool, optional
        Return a :class:`CurvatureLinearizationResult` instead of
        ``(mesh, sigma)``.

    Returns
    -------
    (mesh, sigma) : tuple of ndarray, by default
    result : :class:`CurvatureLinearizationResult` if
        ``return_diagnostics=True``.

    Caveats
    -------
    The bound ``h^2/8 max|f''|`` needs the maximum of ``|f''|`` over
    each segment; this driver samples ``f''`` at segment midpoints
    instead. That is far stronger than a chord probe (it sees
    curvature rather than inferring it), but it is not a certificate:
    a feature narrower than a segment can still be missed. The
    ``refine_passes`` chord sweep exists to catch that case, and
    ``max_bound_error`` is reported separately from
    ``max_chord_error`` so the two can be compared. A genuinely
    certified mesh needs either a third-derivative bound on ``f''``'s
    variation or a closed-form per-pole envelope; neither is
    implemented here.

    LRU=2 URR ranges are not reconstructed by this module. If
    ``[e_min, e_max]`` overlaps an LSSF=0 URR range, a warning fires
    and the returned cross section omits the URR contribution there.
    """
    if not (e_min > 0):
        raise ValueError(f'e_min must be positive, got {e_min!r}')
    if not (e_min < e_max):
        raise ValueError(
            f'e_min ({e_min!r}) must be strictly less than e_max ({e_max!r})'
        )
    if max_points < 2:
        raise ValueError(f'max_points must be at least 2, got {max_points!r}')
    if tol_abs <= 0 and tol_rel <= 0:
        raise ValueError(
            'at least one of tol_abs / tol_rel must be positive; with both '
            'zero the monitor function is unbounded and no finite mesh '
            'satisfies the tolerance.'
        )

    _CRITERIA = ('chord', 'hermite', 'hermite_fh', 'f3bound')
    if criterion not in _CRITERIA:
        raise ValueError(
            f'criterion must be one of {_CRITERIA}; got {criterion!r}')
    if split not in ('midpoint', 'argmax'):
        raise ValueError(
            f"split must be 'midpoint' or 'argmax'; got {split!r}")
    if ref_rule not in ('min', 'max'):
        raise ValueError(f"ref_rule must be 'min' or 'max'; got {ref_rule!r}")
    # Validated here rather than at the point of use, so an invalid
    # value is rejected before the evaluator is built: construction
    # runs the MF2 preprocessors and compiles XLA kernels, which a
    # mistyped knob should never pay for.
    if pilot_strategy not in ('uniform_gamma', 'territory_log'):
        raise ValueError(
            "pilot_strategy must be 'uniform_gamma' or 'territory_log'; "
            f'got {pilot_strategy!r}')

    mt = reac.translate_reaction_string_to_mt(reaction)
    ev = _get_evaluator(endf_dict, mt, chunk_size)

    urr_overlap = None
    for el, eh in ev.urr_ranges:
        if eh > e_min and el < e_max:
            urr_overlap = (el, eh)
            warnings.warn(
                f'linearization_curvature: query range [{e_min!r}, {e_max!r}] '
                f'overlaps an LSSF=0 URR range [{el!r}, {eh!r}), which this '
                f'module does not reconstruct. The returned cross section '
                f'omits the URR contribution in that interval and the mesh '
                f'there follows the MF3 background alone. Restrict the query '
                f'range, or use endf_userpy.linearization for URR coverage.',
                UserWarning, stacklevel=2,
            )
            break

    # --- phase 1: sample the monitor on a cheap pilot mesh.
    if pilot_strategy == 'uniform_gamma':
        pilot = _build_pilot_mesh(
            ev, e_min, e_max,
            per_width=pilot_per_width,
            width_span=pilot_width_span,
            background_per_decade=pilot_background_per_decade,
        )
    elif pilot_strategy == 'territory_log':
        pilot = _build_pilot_mesh_territory(
            ev, e_min, e_max,
            core_pts=pilot_core_pts,
            wing_pts_per_octave=pilot_wing_pts_per_octave,
            max_wing_octaves=pilot_max_wing_octaves,
            background_per_decade=pilot_background_per_decade,
        )
    if include_critical_points:
        f_p, fp_p, fpp_p = ev.trio(pilot)
    else:
        f_p, fpp_p = ev.pair(pilot)
        fp_p = None
    m = _monitor(f_p, fpp_p, tol_abs, tol_rel, safety=monitor_safety)

    # --- phase 2: equidistribute it.
    equi, phi_total, clipped = _equidistribute(pilot, m, max_points)

    # --- phase 3: union in the knots the monitor cannot see.
    mandatory = _mandatory_knots(ev, e_min, e_max)
    critical = (
        _find_critical_points(
            ev, pilot, fp_p, fpp_p,
            newton_iters=newton_iters, e_min=e_min, e_max=e_max,
        )
        if include_critical_points else np.asarray([], dtype=float)
    )
    mesh = np.unique(np.concatenate([
        equi, mandatory, critical, [e_min, e_max],
    ]))
    mesh = mesh[(mesh >= e_min) & (mesh <= e_max)]
    status = 'point_budget' if clipped else 'converged'
    if mesh.size > max_points:
        # Mandatory knots alone can exceed the budget on a file with a
        # very dense MF3 grid. Keep them (they are representation, not
        # accuracy) and say so.
        warnings.warn(
            f'linearization_curvature: mesh ({mesh.size}) exceeds '
            f'max_points ({max_points}) after unioning mandatory knots '
            f'(MF3 abscissae + range edges + critical points), which are '
            f'kept because dropping them breaks representability. Raise '
            f'max_points to silence this.',
            UserWarning, stacklevel=2,
        )
        status = 'point_budget'

    sigma = ev.value(mesh)
    disc = _discontinuity_energies(ev, e_min, e_max)
    mesh, sigma = _apply_doubled_points(ev, mesh, sigma, disc, e_min, e_max)

    # --- phase 4: verification and refinement.
    #
    # Only *pending* segments are evaluated. Inserting a knot inside
    # segment i replaces that segment with two children and leaves
    # every other segment with identical endpoints AND identical
    # endpoint values, so their estimates are unchanged and a
    # converged verdict stays valid permanently. Convergence is
    # therefore monotone per segment, and re-testing a segment that
    # already passed is pure waste.
    #
    # It is a large waste in practice. U-235 capture at safety=2
    # refines 226 segments on the first pass and exactly one on each
    # of the next three, yet the previous revision evaluated all
    # ~290 000 midpoints on every one of five passes: ~1.45 M
    # evaluations where ~290 k plus a few hundred suffice.
    #
    # Criteria also take only the derivatives they need. 'chord' needs
    # f at the midpoint and nothing else (`value`, 0.0157 s / 4096
    # pts), where the previous revision called `pair` (0.1888 s) purely
    # to report a diagnostic -- a 12x overcharge on the phase that
    # dominates the run.
    history: list[dict] = []
    n_unrefinable = 0
    n_seg = mesh.size - 1
    pending = np.ones(n_seg, dtype=bool)
    est_all = np.full(n_seg, np.nan)
    bound_all = np.full(n_seg, np.nan)
    fp_mesh = None
    if criterion in _HERMITE_CRITERIA:
        sigma, fp_mesh = ev.grad(mesh)

    for p in range(max(0, refine_passes) + 1):
        # Zero-width segments are the doubled points that encode a
        # discontinuity. They span no interval, so they carry no
        # interpolation error and must never be refined.
        pending &= np.diff(mesh) > 0.0
        idx = np.nonzero(pending)[0]
        if idx.size == 0:
            status = 'converged'
            break

        lo, hi = mesh[idx], mesh[idx + 1]
        f_lo, f_hi = sigma[idx], sigma[idx + 1]
        h = hi - lo
        mids = 0.5 * (lo + hi)
        f_mid = None

        if criterion in _HERMITE_CRITERIA:
            est, x_at_max = _hermite_residual_seg(
                lo, hi, f_lo, f_hi, fp_mesh[idx], fp_mesh[idx + 1])
            if criterion == 'hermite_fh':
                h_mid = (0.5 * (f_lo + f_hi)
                         + h * (fp_mesh[idx] - fp_mesh[idx + 1]) / 8.0)
                f_mid = ev.value(mids)
                est = est + np.abs(f_mid - h_mid)
            bound = np.full(idx.size, np.nan)
        elif criterion == 'f3bound':
            f_mid, _, fpp_mid, fppp_mid = ev.quad(mids)
            bound = h ** 2 / 8.0 * np.abs(fpp_mid)
            est = h ** 2 / 8.0 * (
                np.abs(fpp_mid) + 0.5 * h * np.abs(fppp_mid))
            x_at_max = mids
        else:   # 'chord' -- f at the midpoint only
            f_mid = ev.value(mids)
            est = np.abs(f_mid - 0.5 * (f_lo + f_hi))
            bound = np.full(idx.size, np.nan)
            x_at_max = mids

        est_all[idx] = est
        bound_all[idx] = bound

        if ref_rule == 'min':
            ref_f = np.minimum(np.abs(f_lo), np.abs(f_hi))
        else:
            ref_f = np.maximum(np.abs(f_lo), np.abs(f_hi))
        tol = tol_abs + tol_rel * ref_f

        # A NaN estimate (above-range sigma) compares False, so such a
        # segment converges and is dropped, matching the bisection
        # driver's warn_nan semantics.
        need = est > tol
        n_add = int(need.sum())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            max_chord = float(np.nanmax(est_all)) if n_seg else 0.0
            max_bound = (float(np.nanmax(bound_all))
                         if np.any(np.isfinite(bound_all)) else float('nan'))
        history.append({
            'pass': p, 'n_mesh': int(mesh.size), 'n_add': n_add,
            'n_checked': int(idx.size),
            'max_chord_err': max_chord, 'max_bound_err': max_bound,
        })

        # Segments that passed are done for good.
        pending[idx[~need]] = False
        if n_add == 0:
            status = 'converged'
            break
        if p == refine_passes:
            status = 'residual_error'
            break
        if mesh.size + n_add > max_points:
            status = 'point_budget'
            break

        # A segment can only be refined if the proposed knot lands
        # strictly inside it. Once a segment is one ULP wide its
        # midpoint is not representable between the endpoints and
        # rounds onto one of them, so "splitting" inserts a DUPLICATE
        # mesh point and leaves the segment untouched: it fails again
        # next pass, forever, emitting one duplicate per pass.
        #
        # That is reachable in practice, not a theoretical worry.
        # resonance_composition masks each MF2 range with
        # (E >= EL) & (E < EH), so the composed cross section genuinely
        # jumps at EH. The mandatory knots bracket EH with
        # np.nextafter, and the resulting one-ULP segment straddles the
        # jump: every interior point is still < EH, so the chord error
        # stays at half the jump no matter how small h gets. On U-235
        # elastic over [1, 2250] (EH = 2250 = e_max) this produced nine
        # zero-width segments and a permanent 'residual_error'.
        #
        # Such a segment is genuinely unrefinable, so retire it and
        # report it rather than looping.
        cand_x = x_at_max[need] if split == 'argmax' else mids[need]
        lo_n, hi_n = lo[need], hi[need]
        splittable = (cand_x > lo_n) & (cand_x < hi_n)
        n_floor = int((~splittable).sum())
        if n_floor:
            stuck = np.nonzero(need)[0][~splittable]
            pending[idx[stuck]] = False      # retire: cannot subdivide
            n_unrefinable += n_floor
            need = need.copy()
            need[stuck] = False
            n_add = int(need.sum())
            history[-1]['n_add'] = n_add
            history[-1]['n_unrefinable'] = n_floor
            if n_add == 0:
                status = 'converged'
                break

        split_seg = idx[need]
        new_x = x_at_max[need] if split == 'argmax' else mids[need]
        if f_mid is not None and split != 'argmax':
            new_f = f_mid[need]
        else:
            new_f = ev.value(new_x)

        comb = np.concatenate([mesh, new_x])
        order = np.argsort(comb, kind='mergesort')
        mesh = comb[order]
        sigma = np.concatenate([sigma, new_f])[order]
        if criterion in _HERMITE_CRITERIA:
            _, new_fp = ev.grad(new_x)
            fp_mesh = np.concatenate([fp_mesh, new_fp])[order]

        # Rebuild the per-segment bookkeeping: a split segment becomes
        # two pending children, every other segment keeps its verdict.
        is_split = np.zeros(n_seg, dtype=bool)
        is_split[split_seg] = True
        counts = np.where(is_split, 2, 1)
        pending = np.repeat(np.where(is_split, True, pending), counts)
        est_all = np.repeat(np.where(is_split, np.nan, est_all), counts)
        bound_all = np.repeat(np.where(is_split, np.nan, bound_all), counts)
        n_seg = mesh.size - 1

    if n_unrefinable:
        warnings.warn(
            f'linearize_curvature: {n_unrefinable} segment(s) could not be '
            f'subdivided further (the proposed knot was not representable '
            f'strictly inside them) and were accepted as-is. This normally '
            f'means a genuine discontinuity in the composed cross section, '
            f'typically an MF2 range edge EH where the resonance '
            f'contribution is masked off; the affected interval is about '
            f'one ULP wide, so the effect on an interpolated value is '
            f'negligible, but the mesh does not meet the stated tolerance '
            f'there.',
            UserWarning, stacklevel=2,
        )

    # Drop duplicate abscissae that carry no information, while keeping
    # the ones that deliberately encode a jump: a repeated energy whose
    # two ordinates differ is the ENDF-6 discontinuity convention and
    # np.interp reproduces it exactly.
    if mesh.size > 1:
        same_x = np.diff(mesh) == 0.0
        same_y = np.diff(sigma) == 0.0
        drop = np.concatenate([[False], same_x & same_y])
        if np.any(drop):
            mesh, sigma = mesh[~drop], sigma[~drop]

    if return_diagnostics:
        return CurvatureLinearizationResult(
            mesh=mesh, sigma=sigma,
            monitor_integral=phi_total,
            n_pilot=int(pilot.size),
            n_equidistributed=int(equi.size),
            n_critical=int(critical.size),
            n_mandatory=int(mandatory.size),
            max_chord_error=max_chord,
            max_bound_error=max_bound,
            n_primal_evaluations=int(ev.n_primal_evaluations),
            status=status,
            history=history,
            n_unrefinable=int(n_unrefinable),
            urr_overlap=urr_overlap,
        )
    return mesh, sigma


# ----------------------------------------------------------------------
# Validation against the user-facing API
# ----------------------------------------------------------------------


def validate_against_quantities(
    endf_dict: dict,
    reaction: str,
    e_min: float,
    e_max: float,
    *,
    n_sample: int = 2000,
    options: RunOptions | None = None,
    chunk_size: int = 4096,
) -> dict:
    """Check this module's hoisted-preproc evaluator against
    :func:`endf_userpy.quantities.get_reaction_xs`.

    The evaluator re-composes MF3 + MF2 itself so it can be jitted,
    which means it can silently drift from the user-facing API (URR
    ranges, MT selection heuristics, MT5 redistribution). Run this on
    any new file before trusting a mesh built from it.

    Returns a dict with ``max_abs_diff``, ``max_rel_diff``, the
    energy at which the worst relative difference occurs, and the
    sample size.
    """
    from .quantities import get_reaction_xs

    if options is None:
        options = RunOptions(
            include_resonance=True, above_range='zero', resonance_range='nan',
        )
    mt = reac.translate_reaction_string_to_mt(reaction)
    ev = _get_evaluator(endf_dict, mt, chunk_size)
    grid = np.geomspace(e_min, e_max, n_sample)
    mine = ev.value(grid)
    theirs = np.asarray(
        get_reaction_xs(endf_dict, reaction, grid, options=options),
        dtype=float,
    )
    diff = np.abs(mine - theirs)
    denom = np.maximum(np.abs(theirs), 1e-30)
    rel = diff / denom
    k = int(np.nanargmax(rel)) if rel.size else 0
    return {
        'max_abs_diff': float(np.nanmax(diff)) if diff.size else 0.0,
        'max_rel_diff': float(np.nanmax(rel)) if rel.size else 0.0,
        'worst_energy': float(grid[k]) if rel.size else float('nan'),
        'n_sample': int(n_sample),
    }
