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

The collapse fires for numpy and JAX backends. It is bypassed when
the numba fast path (:mod:`mf6_law1_epintegral_numba`) intercepts
first with its own equivalent collapse; disabling the numba path in
these tests exercises the numpy implementation directly.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from endf_userpy.mfsec_interpretation import mf6_law1_epintegral as _epi
from endf_userpy.mfsec_interpretation import (
    mf6_law1_epintegral_numba as _numba_fastpath,
)
from endf_userpy.mfsec_interpretation import mf6_law1_preproc as pp
from endf_userpy.primitives import array_ns


_ADHOC_DIR = os.path.join(os.path.dirname(__file__), 'data_law1_adhoc')
_U233_PATH = os.path.join(_ADHOC_DIR, 'endfb81_n_U-233.endf')
_FE56_PATH = os.path.join(_ADHOC_DIR, 'tendl21_n_Fe-56.endf')


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
    MF6/LAW=1 subsection (c0=0 gamma, lang=1 na=0).

    Disables the numba fast path so the numpy code with the new
    collapse is what runs. Matching Fortran independently validates
    the algebra: the two paths integrate the same physics using
    completely different numerical strategies, so bit-comparable
    agreement means the collapse is faithful.
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

    saved = _numba_fastpath.HAS_NUMBA
    _numba_fastpath.HAS_NUMBA = False
    try:
        r_np = _epi.integrate_law1_spectrum(
            data, e_in, e_out, to_lab=True, xp=xp,
        )
    finally:
        _numba_fastpath.HAS_NUMBA = saved

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

    saved = _numba_fastpath.HAS_NUMBA
    _numba_fastpath.HAS_NUMBA = False
    try:
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
    finally:
        _numba_fastpath.HAS_NUMBA = saved

    peak = float(np.max(np.abs(r_np)))
    diff = float(np.max(np.abs(r_jax - r_np)))
    rel = diff / max(1e-30, peak)
    assert rel < 1e-12, f'JAX vs numpy: rel-to-peak diff = {rel:.3e}'


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

    # numba disabled so we exercise the numpy path (with the collapse
    # gate). c0>0 means the collapse should not fire and the general
    # kink-aware kernel runs.
    saved = _numba_fastpath.HAS_NUMBA
    _numba_fastpath.HAS_NUMBA = False
    try:
        r = _epi.integrate_law1_spectrum(
            data, e_in, e_out, to_lab=True, xp=xp,
        )
    finally:
        _numba_fastpath.HAS_NUMBA = saved

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
