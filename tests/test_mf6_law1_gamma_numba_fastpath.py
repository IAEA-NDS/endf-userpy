"""Equivalence tests for the gamma-ejectile numba fast path in
:mod:`endf_userpy.mfsec_interpretation.mf6_law1_epintegral_numba`.

Two coverage layers:

- **Synthetic dict** (fast, always runs): a hand-built
  :class:`MF6Law1Data` exercises the fast-path (gamma, lang=1, na=0,
  lep=2) and a scaffold of gate-failing cases that must fall through
  to the numpy path. Numeric output must match the numpy path
  bit-identically (rel-diff <= 1e-14).

- **Corpus** (skipped without ``data_law1_adhoc``): U-233 (n,g) end-to-
  end through ``get_particle_production_dxs_dE`` with broadening,
  toggling the fast path off and on; the sums must match to the same
  tolerance.
"""
import os
import numpy as np
import pytest

from endf_userpy.mfsec_interpretation import mf6_law1_epintegral as _epi
from endf_userpy.mfsec_interpretation import (
    mf6_law1_epintegral_numba as _numba_fastpath,
)
from endf_userpy.mfsec_interpretation.mf6_law1_preproc import MF6Law1Data
from endf_userpy.primitives import array_ns


pytestmark = pytest.mark.skipif(
    not _numba_fastpath.HAS_NUMBA,
    reason='numba not installed',
)


def _fortran_available():
    try:
        import endf_userpy.fortran.endf6  # noqa: F401
        from endf_userpy.fortran import HAS_FORTRAN
        return HAS_FORTRAN
    except ImportError:
        return False


def _make_synthetic_gamma_data(lang=1, na=0, lep=2, lei=22, awp=0.0):
    """Build a minimal MF6Law1Data for a gamma-ejectile section with
    two Ein panels and a small continuum tabulation each. Discrete
    (nd) count is 0 to keep the fixture simple; the fast path only
    requires ``nep > nd``.
    """
    ei_mesh = np.array([1.0e6, 2.0e6, 3.0e6])
    n_panels = 3
    max_nep = 6
    max_nt = na + 1

    ep_panels = np.zeros((n_panels, max_nep))
    b_panels = np.zeros((n_panels, max_nep, max_nt))
    for p in range(n_panels):
        # Continuum grid: 0, 0.5 MeV, 1 MeV, 1.5 MeV, 2 MeV, 2.5 MeV
        ep_panels[p] = np.array([0.0, 5e5, 1e6, 1.5e6, 2e6, 2.5e6])
        # b-coefficient: simple triangle peaked at 1 MeV, scaled by
        # (p+1) so each panel differs and the outer-Ein interp is
        # nontrivial.
        peak = (p + 1) * 1.0e-6
        b_panels[p, :, 0] = peak * np.array([0.0, 0.5, 1.0, 0.5, 0.25, 0.0])

    nep_arr = np.full(n_panels, max_nep, dtype=int)
    nd_arr = np.zeros(n_panels, dtype=int)
    na_arr = np.full(n_panels, na, dtype=int)

    return MF6Law1Data(
        awi=1.0,
        awr=232.9,
        awp=awp,
        q=0.0,
        za=92233,
        zai=1,
        zap=0,
        lct=2,
        lang=lang,
        lep=lep,
        ei_mesh=ei_mesh,
        int_arr=np.array([lei], dtype=int),
        nbt_arr=np.array([n_panels], dtype=int),
        nd_arr=nd_arr,
        na_arr=na_arr,
        nep_arr=nep_arr,
        ep_panels=ep_panels,
        b_panels=b_panels,
    )


def _integrate_via_numpy(data, e_sub, ep_out):
    """Force the numpy path by calling _law1_spectrum_panel_pair
    directly, bypassing the numba dispatcher wired into
    integrate_law1_spectrum.
    """
    xp = array_ns.get_backend('numpy')
    gl_x, gl_w = np.polynomial.legendre.leggauss(10)
    eff_lct = 2
    lei = int(data.int_arr[0])
    n_panels = len(data.ei_mesh) - 1
    result = np.zeros((len(e_sub), len(ep_out)))
    for p in range(n_panels):
        e1 = data.ei_mesh[p]
        e2 = data.ei_mesh[p + 1]
        in_panel = (e_sub >= e1) & (e_sub < e2) if p < n_panels - 1 \
                   else (e_sub >= e1) & (e_sub <= e2)
        if not np.any(in_panel):
            continue
        f = _epi._law1_spectrum_panel_pair(
            data, p, lei, e_sub[in_panel], ep_out,
            eff_lct, gl_x, gl_w, xp,
        )
        result[in_panel] = f
    return result


def _integrate_via_numba(data, e_sub, ep_out):
    """Force the numba path by calling try_panel_pair directly on
    each panel. Assumes every Ein query lands in a fast-path-eligible
    panel; test callers must ensure that.
    """
    gl_x, gl_w = np.polynomial.legendre.leggauss(10)
    xp = array_ns.get_backend('numpy')
    eff_lct = 2
    lei = int(data.int_arr[0])
    n_panels = len(data.ei_mesh) - 1
    result = np.zeros((len(e_sub), len(ep_out)))
    for p in range(n_panels):
        e1 = data.ei_mesh[p]
        e2 = data.ei_mesh[p + 1]
        in_panel = (e_sub >= e1) & (e_sub < e2) if p < n_panels - 1 \
                   else (e_sub >= e1) & (e_sub <= e2)
        if not np.any(in_panel):
            continue
        f = _numba_fastpath.try_panel_pair(
            data, p, lei, e_sub[in_panel], ep_out,
            eff_lct, gl_x, gl_w, xp,
        )
        assert f is not None, f'gate rejected fast path on panel {p}'
        result[in_panel] = f
    return result


def test_numba_matches_numpy_gamma_lang1_na0_lep2():
    data = _make_synthetic_gamma_data(lang=1, na=0, lep=2, lei=22, awp=0.0)
    e_sub = np.array([1.2e6, 1.7e6, 2.3e6])
    ep_out = np.linspace(0.0, 3.0e6, 61)
    r_np = _integrate_via_numpy(data, e_sub, ep_out)
    r_nb = _integrate_via_numba(data, e_sub, ep_out)
    peak = float(np.max(np.abs(r_np)))
    rel = float(np.max(np.abs(r_nb - r_np))) / max(1e-30, peak)
    assert rel < 1e-14, f'rel-to-peak diff = {rel:.3e}'


def test_numba_matches_numpy_gamma_lep1_histogram():
    data = _make_synthetic_gamma_data(lang=1, na=0, lep=1, lei=22, awp=0.0)
    e_sub = np.array([1.2e6, 1.7e6, 2.3e6])
    ep_out = np.linspace(0.0, 3.0e6, 61)
    r_np = _integrate_via_numpy(data, e_sub, ep_out)
    r_nb = _integrate_via_numba(data, e_sub, ep_out)
    peak = float(np.max(np.abs(r_np)))
    rel = float(np.max(np.abs(r_nb - r_np))) / max(1e-30, peak)
    assert rel < 1e-14, f'rel-to-peak diff = {rel:.3e}'


@pytest.mark.parametrize('override', [
    {'lang': 2},         # Kalbach-Mann rejected
    {'na': 2},           # na>0 rejected (needs Legendre expansion)
    {'lep': 3},          # log-lin LEP rejected
    {'lei': 3},          # lei%10=3 (log-lin) rejected
    {'awp': 1.0},        # c0>0 rejected (non-photon ejectile)
])
def test_gate_falls_through_when_specialisation_missing(override):
    """try_panel_pair must return None outside the specialisation."""
    kwargs = dict(lang=1, na=0, lep=2, lei=22, awp=0.0)
    kwargs.update(override)
    data = _make_synthetic_gamma_data(**kwargs)
    xp = array_ns.get_backend('numpy')
    gl_x, gl_w = np.polynomial.legendre.leggauss(10)
    result = _numba_fastpath.try_panel_pair(
        data, 0, int(data.int_arr[0]),
        np.array([1.2e6]), np.linspace(0.0, 3.0e6, 21),
        2, gl_x, gl_w, xp,
    )
    assert result is None, f'gate should reject override={override}'


def test_gate_falls_through_when_xp_is_not_numpy():
    """The fast path is numpy-only; jax must fall through."""
    pytest.importorskip('jax')
    data = _make_synthetic_gamma_data()
    xp = array_ns.get_backend('jax')
    gl_x, gl_w = np.polynomial.legendre.leggauss(10)
    result = _numba_fastpath.try_panel_pair(
        data, 0, int(data.int_arr[0]),
        np.array([1.2e6]), np.linspace(0.0, 3.0e6, 21),
        2, gl_x, gl_w, xp,
    )
    assert result is None


_ADHOC_DIR = os.path.join(
    os.path.dirname(__file__), 'data_law1_adhoc',
)
_U233_PATH = os.path.join(_ADHOC_DIR, 'endfb81_n_U-233.endf')


@pytest.mark.skipif(
    not os.path.exists(_U233_PATH),
    reason='U-233 corpus file not fetched; run tests/data_law1_adhoc/fetch.sh',
)
def test_u233_ng_broadened_matches_numpy():
    """End-to-end equivalence on the U-233 (n,g) corpus file for
    ``get_particle_production_dxs_dE`` with a 10 keV Gaussian.
    """
    from endf_parserpy import EndfParserCpp
    from endf_userpy.quantities import get_particle_production_dxs_dE

    parser = EndfParserCpp(
        ignore_send_records=True, ignore_missing_tpid=True,
        ignore_blank_lines=True,
    )
    endf_dict = parser.parsefile(_U233_PATH)
    ein = np.array([2.0e6])
    eout = np.linspace(0.0, 5.0e6, 401)

    # numpy-only reference: temporarily disable the numba fast path.
    saved_has = _numba_fastpath.HAS_NUMBA
    _numba_fastpath.HAS_NUMBA = False
    try:
        r_np = get_particle_production_dxs_dE(
            endf_dict, '(n,g)', 'g', ein, eout, broadening=1.0e4,
        )
    finally:
        _numba_fastpath.HAS_NUMBA = saved_has

    r_nb = get_particle_production_dxs_dE(
        endf_dict, '(n,g)', 'g', ein, eout, broadening=1.0e4,
    )
    peak = float(np.nanmax(np.abs(r_np)))
    rel = float(np.nanmax(np.abs(r_nb - r_np))) / max(1e-30, peak)
    assert rel < 1e-14, f'rel-to-peak diff = {rel:.3e}'


@pytest.mark.skipif(
    not os.path.exists(_U233_PATH),
    reason='U-233 corpus file not fetched; run tests/data_law1_adhoc/fetch.sh',
)
@pytest.mark.skipif(
    not _fortran_available(),
    reason='Fortran extension not built; set ENDF_USERPY_BUILD_FORTRAN=1',
)
def test_numba_fastpath_matches_fortran_oracle_u233_ng():
    """Cross-check the numba fast path against the Fortran-backed
    reference implementation on the U-233 (n,g) MF6/LAW=1 subsection.

    Fortran ``feep_points_law1con`` is the NJOY-derived reference
    integrator for MF6 LAW=1 continuum spectra; it uses its own
    kink-aware Ep-mesh handling internally. Bit-comparable agreement
    (rel-to-peak diff <= 1e-12) between the numba fast path and the
    Fortran reference validates that the ``c0=0`` analytical collapse
    used in :mod:`mf6_law1_epintegral_numba` reproduces the reference
    physics exactly, not merely the numpy re-implementation of it.
    """
    from endf_parserpy import EndfParserCpp
    from endf_userpy.mfsec_interpretation import mf6_law1_preproc as pp
    from endf_userpy.mfsec_interpretation import (
        mf6_interpretation_integrals_fort as fort,
    )

    parser = EndfParserCpp(
        ignore_send_records=True, ignore_missing_tpid=True,
        ignore_blank_lines=True,
    )
    endf_dict = parser.parsefile(_U233_PATH)

    e_in = np.array([2.0e6, 2.5e6, 3.0e6])
    e_out = np.linspace(1.0e4, 4.5e6, 41)

    xp = array_ns.get_backend('numpy')
    data = pp.mf6_law1_data_from_endf_dict(endf_dict, 102, 1)

    # Reference: Fortran feep_points_law1con.
    r_fort = fort.get_energydist_from_subsec_law1_fort(
        endf_dict, 102, 1, e_in, e_out, to_lab=True,
    )

    # Fast path: numba dispatches when the gate matches; for U-233
    # (n,g) it always matches (gamma out, na=0, lep=2, lei_law=2).
    r_nb = _epi.integrate_law1_spectrum(
        data, e_in, e_out, to_lab=True, xp=xp,
    )

    peak = float(np.max(np.abs(r_fort)))
    diff = float(np.max(np.abs(r_nb - r_fort)))
    rel = diff / max(1e-30, peak)
    assert rel < 1e-12, (
        f'numba vs fortran rel-to-peak diff = {rel:.3e} '
        f'(abs {diff:.3e}, peak {peak:.3e})'
    )
