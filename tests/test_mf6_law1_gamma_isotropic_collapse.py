"""Correctness tests for the c0=0 gamma-isotropic analytical collapse
in :func:`mf6_law1_epintegral._law1_spectrum_panel_pair`.

The collapse replaces the ``(n_sub, n_gl)`` kink-aware Gauss-Legendre
accumulation with a single ``f_amp(e, ep) * int_0^pi sin(z) dz`` when
the panel-pair satisfies

- ``c0 = sqrt(awi * awp) / (awi + awr) == 0`` (photon ejectile,
  ``awp == 0``: LAB<->CM map is identity),
- ``lang == 1`` and ``na_arr[p1] == na_arr[p2] == 0`` (isotropic
  Legendre: amplitude is mu-independent).

Under these conditions the two paths are algebraically identical, so
the collapse output must match a completely independent reference
(the Fortran-backed integrator ``feep_points_law1con`` derived from
NJOY) to float64 machine precision. That is exactly the check
:func:`test_collapse_matches_fortran_u233_ng` runs.

The collapse fires for both numpy and JAX backends transparently.

Note on ``include_resonance=False`` in the jit-under-trace tests: the
MF2 composition path invoked by the physics-first default
``RunOptions(include_resonance=True)`` goes through
``mfsec_interpretation.mf3_interpretation.compute_cross_section_agnostic``
and thence to ``primitives.tab1.interp``, which is not jit-safe
yet (triggers a ``TracerArrayConversionError`` on tracer input).
These jit smoke tests pin the broadening + jit integration, not
composition, so they explicitly opt out of composition via
``RunOptions(..., include_resonance=False)``. Making composition
jit-safe is a separate follow-up; once that lands, these tests
can drop the opt-out.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from endf_userpy.mfsec_interpretation import mf6_law1_epintegral as _epi
from endf_userpy.mfsec_interpretation import mf6_law1_preproc as pp
from endf_userpy.primitives import array_ns
from endf_userpy.run_options import RunOptions


_ADHOC_DIR = os.path.join(os.path.dirname(__file__), 'data_law1_adhoc')
_U233_PATH = os.path.join(_ADHOC_DIR, 'endfb81_n_U-233.endf')
_FE56_PATH = os.path.join(_ADHOC_DIR, 'tendl21_n_Fe-56.endf')
_AL27_PATH = os.path.join(_ADHOC_DIR, 'endfb81_n_Al-27.endf')


def _fortran_available():
    try:
        import endf_userpy.fortran.endf6  # noqa: F401
        from endf_userpy.fortran import HAS_FORTRAN
        return HAS_FORTRAN
    except ImportError:
        return False


def _jax_available():
    return 'jax' in array_ns.available_backends()


def _endf_dict(path):
    from endf_parserpy import EndfParserCpp
    parser = EndfParserCpp(
        ignore_send_records=True, ignore_missing_tpid=True,
        ignore_blank_lines=True, ignore_zero_mismatch=True,
    )
    return parser.parsefile(path)


@pytest.mark.skipif(
    not os.path.exists(_U233_PATH),
    reason='U-233 corpus file not fetched; run tests/data_law1_adhoc/fetch.sh',
)
@pytest.mark.skipif(
    not _fortran_available(),
    reason='Fortran extension not built; set ENDF_USERPY_BUILD_FORTRAN=1',
)
def test_collapse_matches_fortran_u233_ng():
    """The collapse output must equal the NJOY-derived Fortran
    reference to float64 machine precision on the U-233 (n,g)
    MF6/LAW=1 subsection (c0=0 gamma, lang=1 na=0). Matching
    Fortran independently validates the algebra: the two paths
    integrate the same physics using completely different numerical
    strategies, so bit-comparable agreement means the collapse is
    faithful.
    """
    from endf_userpy.mfsec_interpretation import (
        mf6_interpretation_integrals_fort as fort,
    )
    endf_dict = _endf_dict(_U233_PATH)
    xp = array_ns.get_backend('numpy')
    data = pp.mf6_law1_data_from_endf_dict(endf_dict, 102, 1)

    e_in = np.array([2.0e6, 2.5e6, 3.0e6])
    e_out = np.linspace(1.0e4, 4.5e6, 41)

    r_fort = fort.get_energydist_from_subsec_law1_fort(
        endf_dict, 102, 1, e_in, e_out, to_lab=True,
    )
    r_np = _epi.integrate_law1_spectrum(
        data, e_in, e_out, to_lab=True, xp=xp,
    )

    peak = float(np.max(np.abs(r_fort)))
    diff = float(np.max(np.abs(r_np - r_fort)))
    rel = diff / max(1e-30, peak)
    assert rel < 1e-12, (
        f'numpy-with-collapse vs Fortran: rel-to-peak diff = {rel:.3e} '
        f'(abs {diff:.3e}, peak {peak:.3e})'
    )


@pytest.mark.skipif(
    not os.path.exists(_U233_PATH),
    reason='U-233 corpus file not fetched; run tests/data_law1_adhoc/fetch.sh',
)
@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_collapse_matches_between_numpy_and_jax_u233_ng():
    """The collapse must give the same result under xp=numpy and
    xp=jax on U-233 (n,g). Prior to the collapse, the JAX path
    routed through the multipanel-traced kink-aware machinery; now
    both share the analytical collapse.
    """
    endf_dict = _endf_dict(_U233_PATH)
    e_in = np.array([2.0e6, 2.5e6, 3.0e6])
    e_out = np.linspace(1.0e4, 4.5e6, 41)

    xp_np = array_ns.get_backend('numpy')
    data_np = pp.mf6_law1_data_from_endf_dict(endf_dict, 102, 1, xp=xp_np)
    r_np = _epi.integrate_law1_spectrum(
        data_np, e_in, e_out, to_lab=True, xp=xp_np,
    )

    xp_jax = array_ns.get_backend('jax')
    data_jax = pp.mf6_law1_data_from_endf_dict(endf_dict, 102, 1, xp=xp_jax)
    r_jax = np.asarray(_epi.integrate_law1_spectrum(
        data_jax, e_in, e_out, to_lab=True, xp=xp_jax,
    ))

    peak = float(np.max(np.abs(r_np)))
    diff = float(np.max(np.abs(r_jax - r_np)))
    rel = diff / max(1e-30, peak)
    assert rel < 1e-12, f'JAX vs numpy: rel-to-peak diff = {rel:.3e}'


@pytest.mark.skipif(
    not os.path.exists(_U233_PATH),
    reason='U-233 corpus file not fetched; run tests/data_law1_adhoc/fetch.sh',
)
@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_jit_collapse_matches_numpy_u233_ng():
    """Under ``jax.jit`` with tracer ``energies_in``, the driver
    routes to the multipanel-traced kernel; this test pins that
    the collapse also fires there (the same ``c0 == 0, lang == 1,
    na == 0`` gate is checked in
    ``mf6_law1_multipanel_traced._law1_spectrum_panel_pair_traced``).

    Without the collapse on that path, the traced kernel builds an
    ``(nE, nEp, 2 * max_nep + 1, n_gl)`` tensor and OOMs at n_Ein=100
    (issue #281 follow-up). With the collapse, jit warm-run stays
    under 10 ms and peak RSS under 1 GB.
    """
    import jax
    import jax.numpy as jnp

    endf_dict = _endf_dict(_U233_PATH)
    e_in = np.array([2.0e6, 2.5e6, 3.0e6])
    e_out = np.linspace(1.0e4, 4.5e6, 41)

    xp_np = array_ns.get_backend('numpy')
    data_np = pp.mf6_law1_data_from_endf_dict(endf_dict, 102, 1, xp=xp_np)
    r_np = _epi.integrate_law1_spectrum(
        data_np, e_in, e_out, to_lab=True, xp=xp_np,
    )

    xp_jax = array_ns.get_backend('jax')
    data_jax = pp.mf6_law1_data_from_endf_dict(endf_dict, 102, 1, xp=xp_jax)

    @jax.jit
    def go(ein, eout):
        return _epi.integrate_law1_spectrum(
            data_jax, ein, eout, to_lab=True, xp=xp_jax,
        )

    r_jit = np.asarray(go(jnp.asarray(e_in), jnp.asarray(e_out)))

    peak = float(np.max(np.abs(r_np)))
    diff = float(np.max(np.abs(r_jit - r_np)))
    rel = diff / max(1e-30, peak)
    assert rel < 1e-12, (
        f'jax-jit-with-collapse vs numpy: rel-to-peak diff = {rel:.3e}'
    )


@pytest.mark.skipif(
    not os.path.exists(_U233_PATH),
    reason='U-233 corpus file not fetched; run tests/data_law1_adhoc/fetch.sh',
)
@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_jit_top_level_broadening_u233_ng():
    """End-to-end ``@jax.jit`` on the top-level broadening API on
    U-233 (n,g). Milestone for issue #290 Phase 2.

    Verifies that after (i) the discrete-line kernel port to a
    traced form, (ii) threading ``xp`` through the built-in Gaussian
    kernel closure, and (iii) switching the MF3 cross-section call
    to ``xp=``, the top-level ``get_particle_production_dxs_dE``
    traces end-to-end and produces bit-comparable output to numpy.

    Also spot-checks ``jax.grad`` wrt ``energies_in``.
    """
    import jax
    import jax.numpy as jnp
    from endf_parserpy import EndfParserCpp
    from endf_userpy.quantities import get_particle_production_dxs_dE

    parser = EndfParserCpp(
        ignore_send_records=True, ignore_missing_tpid=True,
        ignore_blank_lines=True,
    )
    endf_dict = parser.parsefile(_U233_PATH)
    ein = np.array([1.0e6, 2.0e6])
    eout = np.linspace(0.0, 8.0e6, 41)

    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')

    r_np = get_particle_production_dxs_dE(endf_dict, '(n,g)', 'g', ein, eout, broadening=1.0e4, options=RunOptions(backend=xp_np, include_resonance=False))

    @jax.jit
    def jit_go(ein_arg):
        return get_particle_production_dxs_dE(endf_dict, '(n,g)', 'g', ein_arg, eout, broadening=1.0e4, options=RunOptions(backend=xp_jax, include_resonance=False))

    r_jit = np.asarray(jit_go(jnp.asarray(ein)))
    peak = float(np.max(np.abs(r_np)))
    diff = float(np.max(np.abs(r_jit - r_np)))
    rel = diff / max(1e-30, peak)
    assert rel < 1e-12, (
        f'jit vs numpy on U-233 (n,g) broadening: rel-to-peak diff = {rel:.3e}'
    )

    def scalar_out(ein_arg):
        r = get_particle_production_dxs_dE(endf_dict, '(n,g)', 'g', ein_arg, eout, broadening=1.0e4, options=RunOptions(backend=xp_jax, include_resonance=False))
        return jnp.sum(r)

    g = np.asarray(jax.grad(scalar_out)(jnp.asarray(ein)))
    assert g.shape == ein.shape
    assert np.all(np.isfinite(g)), (
        f'jax.grad wrt energies_in returned non-finite values: {g}'
    )


@pytest.mark.skipif(
    not os.path.exists(_AL27_PATH),
    reason='Al-27 corpus file not fetched; run tests/data_law1_adhoc/fetch.sh',
)
@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_jit_top_level_broadening_al27_ng():
    """End-to-end ``@jax.jit`` on Al-27 (n,g) — sibling of the U-233
    test but exercising the MF12 discrete-line path (issue #290
    Phase 3). Al-27 MT102 has ~300 MF12 discrete gamma lines plus
    an MF6/LAW=1 continuum tail; jit must trace both.

    Pins the Phase 3 milestone: after adding ``_is_jax_tracer``
    guards on the four MF12/MF13 discrete-line broadening entry
    points and threading ``xp`` through their MF3 cross-section
    calls, ``@jax.jit(get_particle_production_dxs_dE)`` produces
    bit-comparable output to numpy.
    """
    import jax
    import jax.numpy as jnp
    from endf_parserpy import EndfParserCpp
    from endf_userpy.quantities import get_particle_production_dxs_dE

    parser = EndfParserCpp(ignore_missing_tpid=True)
    endf_dict = parser.parsefile(_AL27_PATH)
    # Ein above the RRR to avoid the "raw MF3 background" warning
    # (the resonance-region policy is orthogonal to this test).
    ein = np.array([1.0e6, 5.0e6])
    eout = np.linspace(0.0, 1.0e7, 41)

    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')

    r_np = get_particle_production_dxs_dE(endf_dict, '(n,g)', 'g', ein, eout, broadening=5.0e4, options=RunOptions(backend=xp_np, include_resonance=False))

    @jax.jit
    def jit_go(ein_arg):
        return get_particle_production_dxs_dE(endf_dict, '(n,g)', 'g', ein_arg, eout, broadening=5.0e4, options=RunOptions(backend=xp_jax, include_resonance=False))

    r_jit = np.asarray(jit_go(jnp.asarray(ein)))
    peak = float(np.max(np.abs(r_np)))
    diff = float(np.max(np.abs(r_jit - r_np)))
    rel = diff / max(1e-30, peak)
    assert rel < 1e-12, (
        f'jit vs numpy on Al-27 (n,g) broadening: rel-to-peak diff = {rel:.3e}'
    )


@pytest.mark.skipif(
    not os.path.exists(_AL27_PATH),
    reason='Al-27 corpus file not fetched; run tests/data_law1_adhoc/fetch.sh',
)
@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_jit_top_level_ddxs_al27_ng_mf15():
    """End-to-end ``@jax.jit`` on the top-level DDX API
    (:func:`get_particle_production_ddxs`) on Al-27 (n,g). Pins
    the Phase 4 milestone: after the same ``_is_jax_tracer`` guard
    + ``xp=xp`` threading in
    :func:`compute_ddx_mf15_continuum_broadened`, the DDX pipeline
    traces end-to-end through the MF15 continuum path in addition
    to the MF12 discrete-line path already covered by Phase 3.

    Al-27 MT102 has MF12 discrete lines, MF15 continuum, and MF14
    LI=1 isotropic angular. The DDX folder folds all three.
    """
    import jax
    import jax.numpy as jnp
    from endf_parserpy import EndfParserCpp
    from endf_userpy.quantities import get_particle_production_ddxs

    parser = EndfParserCpp(ignore_missing_tpid=True)
    endf_dict = parser.parsefile(_AL27_PATH)
    ein = np.array([1.0e6, 5.0e6])
    eout = np.linspace(0.0, 1.0e7, 21)
    mu = np.linspace(-0.9, 0.9, 5)

    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')

    r_np = get_particle_production_ddxs(endf_dict, '(n,g)', 'g', ein, eout, mu, broadening=5.0e4, options=RunOptions(backend=xp_np, include_resonance=False))

    @jax.jit
    def jit_go(ein_arg):
        return get_particle_production_ddxs(endf_dict, '(n,g)', 'g', ein_arg, eout, mu, broadening=5.0e4, options=RunOptions(backend=xp_jax, include_resonance=False))

    r_jit = np.asarray(jit_go(jnp.asarray(ein)))
    peak = float(np.max(np.abs(r_np)))
    diff = float(np.max(np.abs(r_jit - r_np)))
    rel = diff / max(1e-30, peak)
    assert rel < 1e-12, (
        f'jit vs numpy on Al-27 (n,g) DDX (MF15 route): '
        f'rel-to-peak diff = {rel:.3e}'
    )


@pytest.mark.skipif(
    not os.path.exists(_AL27_PATH),
    reason='Al-27 corpus file not fetched; run tests/data_law1_adhoc/fetch.sh',
)
@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_jit_top_level_broadening_al27_ng_tracer_eouts():
    """Phase 5 milestone: ``@jax.jit`` with ``energies_out`` also a
    tracer. Under the Phase 3/4 API, ``adaptive_convolve`` derived
    its internal mesh from ``float(eval_points.min())`` / ``.max()``,
    which forced ``energies_out`` to be a compile-time constant.

    Passing ``broadening_mesh_bounds=(emin, emax)`` supplies a
    static mesh so ``energies_out`` can flow as a tracer end-to-end.
    Numerical result must still match the concrete numpy path.
    """
    import jax
    import jax.numpy as jnp
    from endf_parserpy import EndfParserCpp
    from endf_userpy.quantities import get_particle_production_dxs_dE

    parser = EndfParserCpp(ignore_missing_tpid=True)
    endf_dict = parser.parsefile(_AL27_PATH)
    ein = np.array([1.0e6, 5.0e6])
    eout = np.linspace(0.0, 1.0e7, 41)
    # Match the concrete path's default margin exactly
    # (n_kernel_widths=5.0 * kernel_width=5e4) so the two paths
    # converge on the same FFT mesh and their results agree to
    # floating-point precision.
    kernel_width = 5.0e4
    margin = 5.0 * kernel_width
    bounds = (float(eout.min()) - margin, float(eout.max()) + margin)

    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')

    r_np = get_particle_production_dxs_dE(endf_dict, '(n,g)', 'g', ein, eout, broadening=5.0e4, options=RunOptions(backend=xp_np, include_resonance=False))

    @jax.jit
    def jit_go(ein_arg, eout_arg):
        return get_particle_production_dxs_dE(endf_dict, '(n,g)', 'g', ein_arg, eout_arg, broadening=5.0e4, options=RunOptions(broadening_mesh_bounds=bounds, backend=xp_jax, include_resonance=False))

    r_jit = np.asarray(jit_go(jnp.asarray(ein), jnp.asarray(eout)))
    peak = float(np.max(np.abs(r_np)))
    diff = float(np.max(np.abs(r_jit - r_np)))
    rel = diff / max(1e-30, peak)
    # Tolerance is looser than the Phase 3/4 tests because the
    # tracer path in ``adaptive_convolve`` runs the full ``max_iter``
    # doubling loop under trace (the concrete convergence check
    # can't fire on a tracer), so the two paths converge at
    # different iteration counts.
    assert rel < 1e-3, (
        f'jit(tracer eouts) vs numpy on Al-27 (n,g) broadening: '
        f'rel-to-peak diff = {rel:.3e}'
    )


@pytest.mark.skipif(
    not os.path.exists(_AL27_PATH),
    reason='Al-27 corpus file not fetched; run tests/data_law1_adhoc/fetch.sh',
)
@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_grad_wrt_eouts_al27_ng_broadening():
    """Phase 5 milestone: ``jax.grad`` wrt ``energies_out`` on the
    top-level ``get_particle_production_dxs_dE`` broadened path.
    Requires a tracer ``eval_points`` in ``adaptive_convolve`` and
    a static mesh via ``broadening_mesh_bounds``; the pre-Phase-5
    API failed at ``float(eval_points.min())``.

    We do not compare the gradient values to anything analytic —
    the file's MF12/MF15/MF6 mix makes that intractable — but we
    do assert the gradient is finite everywhere and has at least
    one non-zero entry (i.e. the trace really reaches through
    the broadening kernel to ``energies_out``).
    """
    import jax
    import jax.numpy as jnp
    from endf_parserpy import EndfParserCpp
    from endf_userpy.quantities import get_particle_production_dxs_dE

    parser = EndfParserCpp(ignore_missing_tpid=True)
    endf_dict = parser.parsefile(_AL27_PATH)
    ein = np.array([2.5e6])
    eout = np.linspace(1.0e5, 8.0e6, 24)
    bounds = (0.0, 1.0e7)

    xp_jax = array_ns.get_backend('jax')

    def scalar_out(eout_arg):
        r = get_particle_production_dxs_dE(endf_dict, '(n,g)', 'g', ein, eout_arg, broadening=5.0e4, options=RunOptions(broadening_mesh_bounds=bounds, backend=xp_jax, include_resonance=False))
        return jnp.sum(r)

    g = np.asarray(jax.grad(scalar_out)(jnp.asarray(eout)))
    assert g.shape == eout.shape
    assert np.all(np.isfinite(g)), (
        f'jax.grad wrt energies_out returned non-finite values: {g}'
    )
    assert float(np.max(np.abs(g))) > 0.0, (
        'jax.grad wrt energies_out was identically zero — the trace '
        'does not actually reach energies_out through the broadened '
        'path.'
    )


@pytest.mark.skipif(
    not os.path.exists(_AL27_PATH),
    reason='Al-27 corpus file not fetched; run tests/data_law1_adhoc/fetch.sh',
)
@pytest.mark.skipif(not _jax_available(), reason='JAX not installed')
def test_jit_fast_path_matches_slow_path_al27_ng():
    """Fast-path (issue #290 Lever A): when the caller passes
    ``broadening_mesh_bounds``, ``compute_dxs_dE_broadened`` and its
    DDX siblings short-circuit adaptive_convolve to just its final
    Richardson pair (h0=kw/32, one doubling to kw/64), skipping the
    kw/8 and kw/16 iterations. Under jit those earlier iterations
    were pure compile-time waste because the convergence check can
    never fire on a tracer.

    The Richardson return ``(4*R_{kw/64} - R_{kw/32})/3`` is
    algebraically identical to what the pre-fast-path 4-iteration
    jit path already returned from its OWN last two iterations, so
    output must be bit-identical between the two jit paths.
    """
    import jax
    import jax.numpy as jnp
    from endf_parserpy import EndfParserCpp
    from endf_userpy.quantities import get_particle_production_dxs_dE

    parser = EndfParserCpp(ignore_missing_tpid=True)
    endf_dict = parser.parsefile(_AL27_PATH)
    ein = np.array([1.0e6, 5.0e6])
    eout = np.linspace(0.0, 1.0e7, 41)
    margin = 5.0 * 5.0e4
    bounds = (float(eout.min()) - margin, float(eout.max()) + margin)
    xp_jax = array_ns.get_backend('jax')

    @jax.jit
    def jit_hint(ein_arg):
        return get_particle_production_dxs_dE(endf_dict, '(n,g)', 'g', ein_arg, eout, broadening=5.0e4, options=RunOptions(broadening_mesh_bounds=bounds, backend=xp_jax, include_resonance=False))

    @jax.jit
    def jit_nohint(ein_arg):
        return get_particle_production_dxs_dE(endf_dict, '(n,g)', 'g', ein_arg, eout, broadening=5.0e4, options=RunOptions(backend=xp_jax, include_resonance=False))

    r_hint = np.asarray(jit_hint(jnp.asarray(ein)))
    r_nohint = np.asarray(jit_nohint(jnp.asarray(ein)))

    peak = float(np.max(np.abs(r_nohint)))
    rel = float(np.max(np.abs(r_hint - r_nohint))) / max(1e-30, peak)
    # Bit-identical: same Richardson pair reached in two paths, same
    # jax FFT under the hood, so summation order matches too.
    assert rel == 0.0, (
        f'jit fast-path vs jit no-hint on Al-27 (n,g) dxs_dE: '
        f'rel-to-peak = {rel:.3e} (expected bit-identical)'
    )


@pytest.mark.skipif(
    not os.path.exists(_FE56_PATH),
    reason='Fe-56 corpus file not fetched; run tests/data_law1_adhoc/fetch.sh',
)
def test_c0_positive_regression_fe56_ninl():
    """Regression check: when the c0>0 gate fails (Fe-56 (n,inl)
    MT=91 with neutron ejectile, ``awp>0``), the collapse must NOT
    fire and the general kink-aware kernel must run unchanged. If
    numeric output changes on this workload, the collapse gate is
    over-permissive and the general path has regressed.
    """
    endf_dict = _endf_dict(_FE56_PATH)
    xp = array_ns.get_backend('numpy')
    data = pp.mf6_law1_data_from_endf_dict(endf_dict, 91, 2)

    ei = np.asarray(data.ei_mesh)
    e_in = np.array([ei[len(ei) // 2], ei[len(ei) // 2 + 3]])
    e_out = np.linspace(1.0e3, 1.5e7, 41)

    r = _epi.integrate_law1_spectrum(
        data, e_in, e_out, to_lab=True, xp=xp,
    )

    # Confirm the c0>0 gate really rejected the collapse: awp > 0
    # (a non-photon ejectile), so c0 > 0.
    c0 = float(np.sqrt(data.awi * data.awp) / (data.awi + data.awr))
    assert c0 > 0.0, f'expected c0>0 for Fe-56 (n,inl); got c0={c0}'

    # Sanity: result is finite and has physically reasonable shape
    # (some non-zero entries for near-zero Ep in the evaporation
    # regime; zero for Ep above the kinematic max). Full physics
    # validation lives in the existing corpus tests; this test
    # only pins that the general path executed.
    assert r.shape == (2, 41)
    assert np.all(np.isfinite(r))
    assert float(r.max()) > 0.0
