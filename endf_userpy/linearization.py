"""Adaptive mesh-refinement driver for linearising reconstructed
cross sections.

The user-facing :func:`endf_userpy.quantities.get_reaction_xs`
returns the composed cross section on a caller-supplied grid.
This module provides :func:`linearize_reaction_xs`, a driver that
constructs the grid *adaptively*: dense around resonance peaks and
interference features, sparse where the function is smooth, with a
pointwise linear-interpolation accuracy bound set by the caller.

Algorithm, in two phases:

1. **Resonance-aware seed.** Peek into MF2 to pull pole energies
   and total widths (sum of partial widths), add a few points per
   resonance in a window of a few total widths around each pole,
   add the URR energy-table knots, and fill gaps with a weak
   log-spaced background. On a file without MF2 the seed falls
   back to the log-spaced background only.
2. **Iterative chord-error bisection.** For every adjacent segment
   of the current mesh, evaluate the cross section at the segment
   midpoint in one batched call to ``get_reaction_xs``. Compare
   against the linear interpolate; insert midpoints whose error
   exceeds the user tolerance (``tol_abs + tol_rel * |sigma|``).
   Repeat until no segment needs refinement, the point budget is
   hit, or ``max_iterations`` is reached.

The approach follows the shape of NJOY RECONR's linearisation step
(see the RECONR module for prior art) but keeps the implementation
small, uses the existing composed ``get_reaction_xs`` as the point
evaluator, and relies on Python-level batched array ops rather
than inlined Fortran.

Scope: one reaction string per call (``'(n,g)'``, ``'(n,total)'``,
...). For multiple reactions, call once per reaction and union the
meshes externally if a shared grid is needed.

See issue #304 for the composed-cross-section pipeline this driver
sits on top of; see follow-ups #306 (memoisation) and #307 (Ein
slicing) for performance work that benefits this driver linearly.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .quantities import get_reaction_xs
from .run_options import RunOptions


@dataclass
class LinearizationResult:
    """Diagnostics bundle returned when
    :func:`linearize_reaction_xs` is called with
    ``return_diagnostics=True``.
    """

    mesh: np.ndarray
    """Sorted 1D ndarray of incident energies (eV). First and last
    entries equal the user's ``e_min`` and ``e_max``."""

    sigma: np.ndarray
    """Cross section (barn) at each mesh point, as returned by
    :func:`endf_userpy.quantities.get_reaction_xs` under the
    caller's :class:`~endf_userpy.run_options.RunOptions`."""

    iterations: int
    """Number of refinement iterations run (0 means the seed mesh
    already met the tolerance)."""

    max_error: float
    """Maximum chord error (``|sigma_mid - (sigma_i + sigma_{i+1})/2|``)
    across all segments at the last iteration. Zero if no
    refinement candidates were evaluated."""

    status: str
    """One of ``'converged'`` (all segments below tolerance),
    ``'point_budget'`` (``max_points`` reached), or
    ``'max_iterations'`` (``max_iterations`` reached with pending
    refinements)."""

    history: list[dict] = field(default_factory=list)
    """Per-iteration bookkeeping: ``{iter, n_mesh, n_add, max_err}``.
    Useful for diagnosing convergence behaviour on hard files."""


@dataclass
class SeedStrategyComparison:
    """One strategy's result in a
    :func:`benchmark_seed_strategies` run."""

    strategy: str
    """Seed strategy name."""

    seed_size: int
    """Seed mesh size before iterative refinement."""

    final_size: int
    """Final mesh size after convergence (or at
    status='point_budget' / 'max_iterations' exit)."""

    iterations: int
    """Number of refinement iterations run."""

    wall_time_s: float
    """Wall clock time for the whole run in seconds (seed +
    iterative refinement + verification)."""

    verify_max_rel_err: float
    """Maximum relative error of linear interpolation on the
    verification mesh, measured against the ground-truth
    :func:`endf_userpy.quantities.get_reaction_xs` on that same
    mesh. Should be at or below the user tolerance for a
    'converged' run."""

    status: str
    """``'converged'`` / ``'point_budget'`` / ``'max_iterations'``,
    forwarded from the underlying :class:`LinearizationResult`."""


@dataclass
class SeedStrategyBenchmark:
    """Bundle returned by :func:`benchmark_seed_strategies`."""

    verify_mesh_size: int
    """Number of points in the dense verification mesh used to
    estimate linear-interpolation accuracy."""

    results: list[SeedStrategyComparison]
    """One :class:`SeedStrategyComparison` per strategy, in the
    order they were requested (default: windowed_fixed then
    territory_adaptive)."""


_SEED_STRATEGIES = ('windowed_fixed', 'territory_adaptive')


def linearize_reaction_xs(
    endf_dict: dict,
    reaction: str,
    e_min: float,
    e_max: float,
    *,
    tol_abs: float = 0.0,
    tol_rel: float = 1e-3,
    options: RunOptions | None = None,
    max_points: int = 200_000,
    max_iterations: int = 25,
    seed_strategy: str = 'windowed_fixed',
    seed_poles_window_widths: float = 2.0,
    seed_poles_window_npts: int = 7,
    seed_territory_max_wing_steps: int = 20,
    seed_background_per_decade: int = 5,
    return_diagnostics: bool = False,
) -> Any:
    """Linearise ``reaction``'s reconstructed cross section on
    ``[e_min, e_max]`` to a prescribed pointwise accuracy.

    The returned mesh is sparse where the cross section is smooth
    and dense around resonance peaks and interference features.
    Linear interpolation of the returned ``sigma`` on the returned
    ``mesh`` is guaranteed (modulo the chord-error estimator's
    tightness) to be accurate to ``tol_abs + tol_rel * |sigma|`` at
    every energy in ``[e_min, e_max]``.

    Parameters
    ----------
    endf_dict : dict
        Parsed ENDF-6 dict (``endf_parserpy`` layout).
    reaction : str
        Reaction string, e.g. ``'(n,g)'``, ``'(n,total)'``, as
        accepted by
        :func:`endf_userpy.quantities.get_reaction_xs`.
    e_min, e_max : float
        Query range in eV. Must satisfy ``0 < e_min < e_max``.
    tol_abs : float, optional
        Absolute accuracy floor in barn. Default 0 (pure relative).
    tol_rel : float, optional
        Relative accuracy. Default ``1e-3`` (0.1 percent).
    options : RunOptions, optional
        Runtime policies forwarded to each ``get_reaction_xs``
        call. ``None`` resolves to the physics-first default
        ``RunOptions()`` (``include_resonance=True``).
    max_points : int, optional
        Hard cap on the final mesh size. On overflow, the driver
        inserts the worst-error midpoints up to the budget and
        returns with ``status='point_budget'``.
    max_iterations : int, optional
        Hard cap on refinement iterations. On overflow the driver
        returns with ``status='max_iterations'``.
    seed_strategy : {'windowed_fixed', 'territory_adaptive'}, optional
        How to seed per-pole samples.

        - ``'windowed_fixed'`` (default): fixed
          ``seed_poles_window_npts`` points across
          ``[E_r - K Γ, E_r + K Γ]`` for every pole, where
          ``K = seed_poles_window_widths``. Simple, predictable
          per-pole cost; over-samples dense clusters where
          neighbour windows overlap.
        - ``'territory_adaptive'``: for each pole, walks outward
          in ``Γ/2`` steps up to the midpoint toward the
          neighbouring pole (or the ``[e_min, e_max]`` boundary),
          capped at ``seed_territory_max_wing_steps`` steps per
          side. Also seeds the inter-pole midpoints themselves,
          where interference dips tend to sit. Adapts density to
          local peak spacing: sparser in dense clusters, denser
          in isolated-peak regions (hence the wing cap).
    seed_poles_window_widths : float, optional
        ``'windowed_fixed'`` only. Half-width of the per-pole
        window in units of ``Γ_total``. Default 2.0.
    seed_poles_window_npts : int, optional
        ``'windowed_fixed'`` only. Number of seed points per pole,
        uniformly spaced across the window. Default 7.
    seed_territory_max_wing_steps : int, optional
        ``'territory_adaptive'`` only. Maximum ``Γ/2`` steps per
        side on isolated peaks (bounds the wing-sample cost when
        the next-pole midpoint is many widths away). Default 20
        (covers ``±10 Γ``, where a Lorentzian has decayed to
        ~1 percent of the peak amplitude).
    seed_background_per_decade : int, optional
        Log-spaced background seed density, both strategies.
        Default 5 points per decade of ``[e_min, e_max]``.
    return_diagnostics : bool, optional
        If True, return a :class:`LinearizationResult` with
        ``mesh``, ``sigma``, and iteration bookkeeping. Default
        False returns the ``(mesh, sigma)`` tuple only.

    Returns
    -------
    (mesh, sigma) : tuple of ndarray, by default
        Linearised mesh and cross section.
    result : LinearizationResult, if ``return_diagnostics=True``.
    """
    if not (e_min > 0):
        raise ValueError(f'e_min must be positive, got {e_min!r}')
    if not (e_min < e_max):
        raise ValueError(
            f'e_min ({e_min!r}) must be strictly less than e_max ({e_max!r})'
        )
    if max_points < 2:
        raise ValueError(f'max_points must be at least 2, got {max_points!r}')
    if seed_strategy not in _SEED_STRATEGIES:
        raise ValueError(
            f'seed_strategy must be one of {_SEED_STRATEGIES}; '
            f'got {seed_strategy!r}'
        )
    if options is None:
        options = RunOptions()

    if seed_strategy == 'windowed_fixed':
        seed = _build_seed_mesh(
            endf_dict, e_min, e_max,
            poles_window_widths=seed_poles_window_widths,
            poles_window_npts=seed_poles_window_npts,
            background_per_decade=seed_background_per_decade,
        )
    else:
        seed = _build_seed_mesh_territory_adaptive(
            endf_dict, e_min, e_max,
            max_wing_steps=seed_territory_max_wing_steps,
            background_per_decade=seed_background_per_decade,
        )
    if seed.size > max_points:
        warnings.warn(
            f'seed mesh ({seed.size} pts) already exceeds max_points '
            f'({max_points}); returning truncated seed without refinement. '
            f'Raise max_points or widen the (e_min, e_max) range to run '
            f'the iterative refinement.',
            UserWarning, stacklevel=2,
        )
        seed = seed[:max_points]

    def evaluate(E):
        return np.asarray(
            get_reaction_xs(endf_dict, reaction, E, options=options),
            dtype=float,
        )

    sigma_seed = evaluate(seed)

    if tol_abs <= 0 and tol_rel <= 0:
        warnings.warn(
            'both tol_abs and tol_rel are zero; iterative refinement '
            'would never converge. Returning the seed mesh unchanged.',
            UserWarning, stacklevel=2,
        )
        if return_diagnostics:
            return LinearizationResult(
                mesh=seed, sigma=sigma_seed, iterations=0,
                max_error=float('nan'), status='converged',
            )
        return seed, sigma_seed

    mesh, sigma, history, status = _iterative_refine(
        seed, sigma_seed, evaluate,
        tol_abs=tol_abs, tol_rel=tol_rel,
        max_points=max_points, max_iterations=max_iterations,
    )
    max_err = history[-1]['max_err'] if history else 0.0
    iters = len(history)

    if return_diagnostics:
        return LinearizationResult(
            mesh=mesh, sigma=sigma, iterations=iters,
            max_error=float(max_err), status=status, history=history,
        )
    return mesh, sigma


def benchmark_seed_strategies(
    endf_dict: dict,
    reaction: str,
    e_min: float,
    e_max: float,
    *,
    tol_abs: float = 0.0,
    tol_rel: float = 1e-3,
    options: RunOptions | None = None,
    strategies: tuple[str, ...] = _SEED_STRATEGIES,
    verify_mesh_size: int = 50_000,
    verify_log_spaced: bool = True,
    max_points: int = 200_000,
    max_iterations: int = 25,
) -> SeedStrategyBenchmark:
    """A/B compare seed strategies for :func:`linearize_reaction_xs`.

    Runs each requested strategy with the same tolerance and
    options, times each run, then measures the linear-interpolation
    error of each strategy's output on a shared dense verification
    mesh of ``verify_mesh_size`` points.

    Intended for picking a default on a new file class or for
    sanity-checking the two strategies on an unfamiliar corpus
    file. Not a production hot path; the verification call to
    :func:`endf_userpy.quantities.get_reaction_xs` dominates
    wall time.

    Parameters
    ----------
    endf_dict, reaction, e_min, e_max, tol_abs, tol_rel, options,
    max_points, max_iterations
        Forwarded verbatim to :func:`linearize_reaction_xs`.
    strategies : tuple of str, optional
        Which seed strategies to benchmark, in order. Defaults to
        every registered strategy.
    verify_mesh_size : int, optional
        Number of points in the shared dense verification mesh.
        Default 50 000 covers most RRR windows well; raise it for
        high-level-density actinides where sub-resonance structure
        is finer than a 50 k log-spaced sample can resolve.
    verify_log_spaced : bool, optional
        If True (default), build the verification mesh with
        :func:`numpy.geomspace`; else :func:`numpy.linspace`.
        Log-spaced is usually the right choice for an RRR window
        spanning several decades of Ein.

    Returns
    -------
    bench : :class:`SeedStrategyBenchmark`
        Comparison bundle; one :class:`SeedStrategyComparison`
        entry per requested strategy.
    """
    import time
    unknown = [s for s in strategies if s not in _SEED_STRATEGIES]
    if unknown:
        raise ValueError(
            f'unknown seed strategies: {unknown}; '
            f'known: {_SEED_STRATEGIES}'
        )
    if options is None:
        options = RunOptions()

    if verify_log_spaced:
        if e_min <= 0:
            raise ValueError(
                'verify_log_spaced=True requires e_min > 0; got '
                f'e_min={e_min!r}'
            )
        verify_mesh = np.geomspace(e_min, e_max, verify_mesh_size)
    else:
        verify_mesh = np.linspace(e_min, e_max, verify_mesh_size)
    verify_truth = np.asarray(
        get_reaction_xs(endf_dict, reaction, verify_mesh, options=options),
        dtype=float,
    )

    comparisons = []
    for strat in strategies:
        t0 = time.perf_counter()
        result = linearize_reaction_xs(
            endf_dict, reaction, e_min, e_max,
            tol_abs=tol_abs, tol_rel=tol_rel, options=options,
            max_points=max_points, max_iterations=max_iterations,
            seed_strategy=strat,
            return_diagnostics=True,
        )
        wall = time.perf_counter() - t0
        xs_interp = np.interp(verify_mesh, result.mesh, result.sigma)
        denom = np.maximum(np.abs(verify_truth), 1e-30)
        rel_err = float(np.nanmax(np.abs(verify_truth - xs_interp) / denom))
        seed_size = result.history[0]['n_mesh'] if result.history else result.mesh.size
        comparisons.append(SeedStrategyComparison(
            strategy=strat,
            seed_size=int(seed_size),
            final_size=int(result.mesh.size),
            iterations=int(result.iterations),
            wall_time_s=float(wall),
            verify_max_rel_err=rel_err,
            status=result.status,
        ))

    return SeedStrategyBenchmark(
        verify_mesh_size=int(verify_mesh_size),
        results=comparisons,
    )


# ----------------------------------------------------------------------
# Seed construction
# ----------------------------------------------------------------------


def _build_seed_mesh(
    endf_dict, e_min, e_max,
    *, poles_window_widths, poles_window_npts, background_per_decade,
) -> np.ndarray:
    """Seed mesh for the iterative refinement.

    Combines endpoints, per-pole windows, URR table knots, and a
    weak log-spaced background. Deduplicates and sorts.
    """
    pts = [float(e_min), float(e_max)]

    # Resonance-aware seed: pole-centred windows.
    offsets = np.linspace(
        -poles_window_widths, poles_window_widths, poles_window_npts,
    )
    for E_r, gamma_total in _extract_resonance_seed_points(endf_dict):
        if not (gamma_total > 0) or not np.isfinite(E_r):
            continue
        pts.extend(float(E_r + d * gamma_total) for d in offsets)

    # URR energy knots (natural mesh for the averaged XS).
    pts.extend(float(e) for e in _extract_urr_energy_knots(endf_dict))

    # Log-spaced background filler.
    if background_per_decade > 0:
        n_bg = max(2, int(np.log10(e_max / e_min) * background_per_decade))
        pts.extend(np.geomspace(e_min, e_max, n_bg).tolist())

    arr = np.asarray(pts, dtype=float)
    arr = arr[(arr >= e_min) & (arr <= e_max)]
    arr = np.unique(arr)   # unique() returns sorted
    return arr


def _build_seed_mesh_territory_adaptive(
    endf_dict, e_min, e_max,
    *, max_wing_steps, background_per_decade,
) -> np.ndarray:
    """Territory-tiled seed mesh: each pole owns the interval from
    the midpoint to its left neighbour (or ``e_min``) to the
    midpoint to its right neighbour (or ``e_max``), and seeds that
    interval with ``Γ/2``-spaced samples capped at
    ``max_wing_steps`` per side. Inter-pole midpoints are seeded
    too so interference dips have a knot near them from iteration 0.

    URR knots and a log-spaced background are added as in the
    windowed-fixed strategy so URR-only regions and resonance-less
    files still get sensible coverage.
    """
    pts = [float(e_min), float(e_max)]

    poles = [
        (float(er), float(gt))
        for er, gt in _extract_resonance_seed_points(endf_dict)
        if gt > 0 and np.isfinite(er) and e_min <= er <= e_max
    ]
    poles.sort(key=lambda p: p[0])
    n_poles = len(poles)

    for i, (er, gamma) in enumerate(poles):
        # Left territory boundary: midpoint to previous in-range pole,
        # or e_min if this is the first.
        if i == 0:
            left_bnd = float(e_min)
        else:
            left_bnd = 0.5 * (poles[i - 1][0] + er)
        # Right territory boundary: midpoint to next in-range pole,
        # or e_max if this is the last.
        if i == n_poles - 1:
            right_bnd = float(e_max)
        else:
            right_bnd = 0.5 * (er + poles[i + 1][0])

        pts.append(er)
        pts.append(left_bnd)
        pts.append(right_bnd)

        step = 0.5 * gamma
        # Walk left of the pole in Γ/2 steps, stopping at the
        # territory boundary or after ``max_wing_steps`` steps.
        for k in range(1, max_wing_steps + 1):
            e_k = er - k * step
            if e_k <= left_bnd:
                break
            pts.append(e_k)
        # Walk right of the pole likewise.
        for k in range(1, max_wing_steps + 1):
            e_k = er + k * step
            if e_k >= right_bnd:
                break
            pts.append(e_k)

    # URR knots (natural mesh for the averaged URR XS).
    pts.extend(float(e) for e in _extract_urr_energy_knots(endf_dict))

    # Log-spaced background filler.
    if background_per_decade > 0:
        n_bg = max(2, int(np.log10(e_max / e_min) * background_per_decade))
        pts.extend(np.geomspace(e_min, e_max, n_bg).tolist())

    arr = np.asarray(pts, dtype=float)
    arr = arr[(arr >= e_min) & (arr <= e_max)]
    return np.unique(arr)


def _extract_resonance_seed_points(endf_dict):
    """Yield ``(E_r, Γ_total)`` for every resolved resonance in
    MF2, across isotopes and ranges. Silent no-op on files without
    MF2/MT151, isotopes, or supported LRU=1 formalisms.
    """
    if 2 not in endf_dict or 151 not in endf_dict[2]:
        return
    isotopes = endf_dict[2][151].get('isotope', {})
    for iso_i in sorted(isotopes):
        d_iso = isotopes[iso_i]
        for rng_i in sorted(d_iso.get('range', {})):
            rng = d_iso['range'][rng_i]
            if int(rng.get('LRU', 0)) != 1:
                continue
            lrf = int(rng.get('LRF', 0))
            try:
                if lrf == 2:
                    yield from _poles_from_mlbw(endf_dict, iso_i, rng_i)
                elif lrf == 3:
                    yield from _poles_from_rm(endf_dict, iso_i, rng_i)
                elif lrf == 7:
                    yield from _poles_from_rml(endf_dict, iso_i, rng_i)
                # Other LRFs (4 Adler-Adler, 7 with KRM!=3) are not
                # reconstructed by this package; skip seeding, let the
                # iterative refinement discover whatever shape exists.
            except Exception as exc:
                warnings.warn(
                    f'linearization seed: skipping iso={iso_i} rng={rng_i} '
                    f'LRF={lrf} ({type(exc).__name__}: {exc}). Iterative '
                    f'refinement will still find peaks on its own.',
                    UserWarning, stacklevel=2,
                )


def _poles_from_mlbw(endf_dict, iso_i, rng_i):
    from .mfsec_interpretation import mf2_interpretation_mlbw_preproc as pre
    data = pre.mlbw_data_from_endf_dict(endf_dict, iso_i, rng_i)
    gamma_total = (
        np.abs(data.res_gn) + np.abs(data.res_gg)
        + np.abs(data.res_gf) + np.abs(data.res_gx)
    )
    for er, gt in zip(data.res_er, gamma_total):
        yield float(er), float(gt)


def _poles_from_rm(endf_dict, iso_i, rng_i):
    from .mfsec_interpretation import mf2_interpretation_reichmoore_preproc as pre
    data = pre.rm_data_from_endf_dict(endf_dict, iso_i, rng_i)
    gamma_total = (
        np.abs(data.res_gn) + np.abs(data.res_gg)
        + np.abs(data.res_gf1) + np.abs(data.res_gf2)
    )
    for er, gt in zip(data.res_er, gamma_total):
        yield float(er), float(gt)


def _poles_from_rml(endf_dict, iso_i, rng_i):
    from .mfsec_interpretation import mf2_interpretation_rml_preproc as pre
    data = pre.rml_data_from_endf_dict(endf_dict, iso_i, rng_i)
    # res_gam is (nres, max_nch) padded; sum of |widths| across
    # channels approximates Γ_total.
    gamma_total = np.sum(np.abs(data.res_gam), axis=1)
    for er, gt in zip(data.res_er, gamma_total):
        yield float(er), float(gt)


def _extract_urr_energy_knots(endf_dict):
    """Yield energies in URR energy tables across every LRU=2 range.
    These are the natural knots for the averaged URR cross section.
    """
    if 2 not in endf_dict or 151 not in endf_dict[2]:
        return
    isotopes = endf_dict[2][151].get('isotope', {})
    for iso_i in sorted(isotopes):
        d_iso = isotopes[iso_i]
        for rng_i in sorted(d_iso.get('range', {})):
            rng = d_iso['range'][rng_i]
            if int(rng.get('LRU', 0)) != 2:
                continue
            try:
                from .mfsec_interpretation import (
                    mf2_interpretation_urr_preproc as pre,
                )
                data = pre.urr_data_from_endf_dict(endf_dict, iso_i, rng_i)
                for e in np.asarray(data.table_es, dtype=float):
                    yield float(e)
            except Exception as exc:
                warnings.warn(
                    f'linearization seed: skipping URR iso={iso_i} '
                    f'rng={rng_i} ({type(exc).__name__}: {exc}).',
                    UserWarning, stacklevel=2,
                )


# ----------------------------------------------------------------------
# Iterative refinement
# ----------------------------------------------------------------------


def _iterative_refine(
    mesh, sigma, evaluate,
    *, tol_abs, tol_rel, max_points, max_iterations,
):
    """Chord-error bisection with batched evaluation.

    Returns ``(mesh, sigma, history, status)``.
    """
    history: list[dict] = []
    status = 'max_iterations'
    for it in range(max_iterations):
        mids = 0.5 * (mesh[:-1] + mesh[1:])
        predicted = 0.5 * (sigma[:-1] + sigma[1:])
        actual = evaluate(mids)
        err = np.abs(actual - predicted)
        tol = tol_abs + tol_rel * np.maximum(np.abs(actual), np.abs(predicted))
        # NaN propagation: if actual is NaN, err is NaN, (err > tol) is
        # False -> that segment is skipped. Matches warn_nan above-range
        # semantics: the driver won't densify where sigma is undefined.
        keep = err > tol
        max_err = float(np.nanmax(err)) if err.size else 0.0
        n_add = int(keep.sum())
        history.append({
            'iter': it, 'n_mesh': int(mesh.size),
            'n_add': n_add, 'max_err': max_err,
        })
        if n_add == 0:
            status = 'converged'
            break
        if mesh.size + n_add > max_points:
            budget = max_points - mesh.size
            if budget <= 0:
                status = 'point_budget'
                break
            # Keep only the worst `budget` midpoints by error.
            worst = np.argpartition(np.where(keep, err, -np.inf), -budget)[-budget:]
            keep = np.zeros_like(keep)
            keep[worst] = True
            mesh, sigma = _merge_sorted(mesh, sigma, mids[keep], actual[keep])
            status = 'point_budget'
            break
        mesh, sigma = _merge_sorted(mesh, sigma, mids[keep], actual[keep])
    return mesh, sigma, history, status


def _merge_sorted(mesh, sigma, new_x, new_y):
    """Insert ``(new_x, new_y)`` into the sorted ``(mesh, sigma)``
    arrays and return the merged, sorted pair.

    ``new_x`` entries are each strictly between some ``mesh[i]`` and
    ``mesh[i+1]`` (they are segment midpoints), so a straight
    concatenate + argsort is both correct and faster than a Python
    insertion loop for the sizes we care about.
    """
    combined_x = np.concatenate([mesh, new_x])
    combined_y = np.concatenate([sigma, new_y])
    order = np.argsort(combined_x, kind='mergesort')
    return combined_x[order], combined_y[order]
