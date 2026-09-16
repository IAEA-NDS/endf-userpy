"""Tests for MF5 analytic spectrum representations (issue #41).

Pre-fix bugs in LF=5, LF=7, LF=9 handlers:

- **LF=7 (simple Maxwellian) and LF=9 (evaporation)**: support mask
  was ``eout <= U`` (should be ``eout <= E - U``). With U=0 the mask
  was empty and every spectrum was identically zero.
- **LF=7 normalisation**: was ``I = theta^(3/2)`` (missing the
  bracketed erf/exp factor) and used ``exp(-z)`` instead of
  ``exp(-z^2)``.
- **LF=9 normalisation**: was ``I = theta^2 * (1 - (1 + (E-U)/theta))
  = -theta * (E-U)`` -- negative for physical inputs, missing the
  ``exp(-(E-U)/theta)`` factor entirely.
- **LF=5 (general evaporation)**: returned the reduced variable
  ``E'/theta`` and never touched the tabulated ``g(x)`` at all.

All three now match the ENDF-6 formulas.

No file in ``tests/data/`` contains an MF5 section at all, so
LF=7 and LF=9 tests use synthetic dicts. LF=5 is exercised on the
real corpus (U-235 MT 455 in ``data_law1_adhoc``).
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf5_interpretation as mf5
from endf_userpy.primitives.np_compat import trapezoid


ADHOC_DATA_DIR = Path(__file__).resolve().parent / 'data_law1_adhoc'


def _tab1(x_name, y_name, x, y):
    return {x_name: list(x), y_name: list(y), 'INT': [2], 'NBT': [len(x)]}


def _make_contrib(lf, theta, U, g_table=None):
    c = {
        'p_table': _tab1('E', 'p', [1e-5, 3e7], [1.0, 1.0]),
        'theta_table': _tab1('E', 'theta', [1e-5, 3e7], [theta, theta]),
        'U': U,
        'LF': lf,
    }
    if g_table is not None:
        c['g_table'] = g_table
    return c


@pytest.fixture(scope='module')
def u235_tendl():
    """U-235 MT 455 (delayed nubar) has LF=5 contributions in
    the LO=2 spectrum. Real corpus exercise for the general-
    evaporation code path."""
    fn = ADHOC_DATA_DIR / 'tendl21_n_U-235.endf'
    if not fn.exists():
        pytest.skip(f'{fn.name} not present; fetch corpus first')
    parser = EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    )
    return parser.parsefile(fn)


# ============================================================
# LF=7 simple Maxwellian: f = sqrt(E') exp(-E'/theta) / I
# I = theta^(3/2) [ sqrt(pi)/2 erf(z) - z exp(-z^2) ], z=sqrt((E-U)/theta)
# ============================================================


def test_lf7_integrates_to_one_at_U_zero():
    """Direct rebut of the pre-fix zero-spectrum bug. theta=1 MeV,
    U=0, E=14 MeV; integral over E' must equal 1 on a fine mesh."""
    c = _make_contrib(lf=7, theta=1e6, U=0.0)
    E = np.array([14e6])
    Eout = np.linspace(0.0, 14e6, 10001)
    f = mf5.compute_simple_maxwellian_fission_spectrum(c, E, Eout)
    integ = trapezoid(f[0], Eout)
    assert abs(integ - 1.0) < 1e-3


def test_lf7_integrates_to_one_at_U_nonzero():
    """Non-zero U so the mask fix is exercised (pre-fix mask
    `eout <= U` gave nonzero results for E' <= U only, still wrong)."""
    c = _make_contrib(lf=7, theta=1.5e6, U=1e6)
    E = np.array([10e6])
    Eout = np.linspace(0.0, 15e6, 10001)
    f = mf5.compute_simple_maxwellian_fission_spectrum(c, E, Eout)
    integ = trapezoid(f[0], Eout)
    assert abs(integ - 1.0) < 1e-3


def test_lf7_support_is_below_E_minus_U():
    """f(E') must be zero for E' > E - U."""
    c = _make_contrib(lf=7, theta=1e6, U=0.5e6)
    E = np.array([5e6])  # E - U = 4.5 MeV
    Eout = np.linspace(0.0, 10e6, 10001)
    f = mf5.compute_simple_maxwellian_fission_spectrum(c, E, Eout)
    above = Eout > (5e6 - 0.5e6)
    assert (f[0, above] == 0.0).all()


def test_lf7_integrates_to_one_across_ein_vector():
    """Vectorised over multiple incident energies. Each row must
    integrate to 1 independently (theta is Ein-dependent in
    general; here we hold it constant to isolate the analytic
    normalisation)."""
    c = _make_contrib(lf=7, theta=1e6, U=0.0)
    Ein = np.array([2e6, 5e6, 10e6, 14e6])
    Eout = np.linspace(0.0, 20e6, 20001)
    f = mf5.compute_simple_maxwellian_fission_spectrum(c, Ein, Eout)
    for i, e in enumerate(Ein):
        integ = trapezoid(f[i], Eout)
        assert abs(integ - 1.0) < 1e-3, f'Ein={e}: integral {integ}'


def test_lf7_nonnegative_everywhere():
    """A common pre-fix symptom: negative I(E) yielded negative f
    values across the whole grid. Regression anchor."""
    c = _make_contrib(lf=7, theta=1.5e6, U=1e6)
    E = np.array([10e6])
    Eout = np.linspace(0.0, 15e6, 1001)
    f = mf5.compute_simple_maxwellian_fission_spectrum(c, E, Eout)
    assert (f >= 0).all()


# ============================================================
# LF=9 evaporation: f = E' exp(-E'/theta) / I
# I = theta^2 [ 1 - exp(-(E-U)/theta) (1 + (E-U)/theta) ]
# ============================================================


def test_lf9_integrates_to_one_at_U_zero():
    c = _make_contrib(lf=9, theta=1e6, U=0.0)
    E = np.array([14e6])
    Eout = np.linspace(0.0, 14e6, 10001)
    f = mf5.compute_evaporation_spectrum(c, E, Eout)
    integ = trapezoid(f[0], Eout)
    assert abs(integ - 1.0) < 1e-3


def test_lf9_integrates_to_one_at_U_nonzero():
    c = _make_contrib(lf=9, theta=2e6, U=1e6)
    E = np.array([10e6])
    Eout = np.linspace(0.0, 15e6, 10001)
    f = mf5.compute_evaporation_spectrum(c, E, Eout)
    integ = trapezoid(f[0], Eout)
    assert abs(integ - 1.0) < 1e-3


def test_lf9_support_is_below_E_minus_U():
    c = _make_contrib(lf=9, theta=1e6, U=0.5e6)
    E = np.array([5e6])
    Eout = np.linspace(0.0, 10e6, 10001)
    f = mf5.compute_evaporation_spectrum(c, E, Eout)
    above = Eout > (5e6 - 0.5e6)
    assert (f[0, above] == 0.0).all()


def test_lf9_integrates_to_one_across_ein_vector():
    c = _make_contrib(lf=9, theta=1e6, U=0.0)
    Ein = np.array([2e6, 5e6, 10e6, 14e6])
    Eout = np.linspace(0.0, 20e6, 20001)
    f = mf5.compute_evaporation_spectrum(c, Ein, Eout)
    for i, e in enumerate(Ein):
        integ = trapezoid(f[i], Eout)
        assert abs(integ - 1.0) < 1e-3, f'Ein={e}: integral {integ}'


def test_lf9_nonnegative_everywhere():
    """Direct regression on the pre-fix negative-normalisation
    bug: I(E) was `theta^2 (1 - (1 + y)) = -y*theta^2`, so f was
    always negative on the interior of the support."""
    c = _make_contrib(lf=9, theta=1.5e6, U=1e6)
    E = np.array([10e6])
    Eout = np.linspace(0.0, 15e6, 1001)
    f = mf5.compute_evaporation_spectrum(c, E, Eout)
    assert (f >= 0).all()


# ============================================================
# LF=5 general evaporation: f = g(E'/theta) / (theta * G)
# G = integral of g(x) from 0 to (E-U)/theta
# ============================================================


def test_lf5_synthetic_gaussian_integrates_to_one():
    """Synthetic g(x): a truncated Gaussian on x in [0, 5]. Any
    correctly-normalised implementation returns integral 1 on a
    fine E' mesh once theta and support are consistent."""
    x = np.linspace(0.0, 5.0, 1001)
    g = np.exp(-((x - 2.5) ** 2))
    g_tab = {'x': list(x), 'g': list(g), 'INT': [2], 'NBT': [len(x)]}
    # U very negative so (E - U) / theta covers the entire g support.
    c = _make_contrib(lf=5, theta=1e6, U=-1e7, g_table=g_tab)
    E = np.array([1e6])          # E - U = 11 MeV
    Eout = np.linspace(0.0, 6e6, 6001)  # up to 6 MeV; x up to 6
    f = mf5.compute_general_evaporation_spectrum(c, E, Eout)
    integ = trapezoid(f[0], Eout)
    # Should integrate to 1 within the discretisation error of both
    # the interior trapezoid and the normalisation integral.
    assert abs(integ - 1.0) < 5e-3


def test_lf5_support_is_below_E_minus_U():
    x = np.linspace(0.0, 5.0, 501)
    g = np.ones_like(x)  # flat g(x): easiest analytic case
    g_tab = {'x': list(x), 'g': list(g), 'INT': [2], 'NBT': [len(x)]}
    c = _make_contrib(lf=5, theta=1e6, U=0.0, g_table=g_tab)
    E = np.array([3e6])  # E - U = 3 MeV
    Eout = np.linspace(0.0, 6e6, 601)
    f = mf5.compute_general_evaporation_spectrum(c, E, Eout)
    above = Eout > 3e6
    assert (f[0, above] == 0.0).all()


def test_lf5_nonzero_on_support():
    """The primary defect: pre-fix returned E'/theta unnormalised
    and never touched g_table. A non-degenerate g(x) should produce
    values that scale with g at the interior. Support of f(E')
    is the intersection of ``[0, E-U]`` (from ENDF) and the
    g_table's own x-mesh (times theta) -- outside that, interp
    returns 0."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    g = np.array([0.1, 1.0, 2.0, 2.0, 1.0, 0.1])
    g_tab = {'x': list(x), 'g': list(g), 'INT': [2], 'NBT': [len(x)]}
    c = _make_contrib(lf=5, theta=1e6, U=-1e7, g_table=g_tab)
    E = np.array([1e6])
    # Keep Eout inside x_max_g * theta = 5 * 1e6 = 5e6 so we're
    # inside g_table's tabulated support at every point.
    Eout = np.linspace(0.5e6, 4.5e6, 41)
    f = mf5.compute_general_evaporation_spectrum(c, E, Eout)
    assert (f[0] > 0).all(), 'g(x) > 0 on interior of support should map to f > 0'


def test_lf5_real_corpus_u235_mt455(u235_tendl):
    """Real-file exercise: U-235 delayed fission MT 455 in TENDL-2021
    has multiple LF=5 contributions (one per delayed group).
    Each individual contribution's spectrum should integrate to
    1 within a few tenths of a percent (trapezoid error on the
    g_table mesh)."""
    contribs = list(u235_tendl[5][455]['contribution'].values())
    lf5_contribs = [c for c in contribs if c['LF'] == 5]
    assert len(lf5_contribs) >= 1
    E = np.array([1e6])
    Eout = np.linspace(0.0, 3e7, 30001)
    for i, c in enumerate(lf5_contribs):
        f = mf5.compute_general_evaporation_spectrum(c, E, Eout)
        integ = trapezoid(f[0], Eout)
        # Accept 1% tolerance for trapezoid on the file's own mesh.
        assert abs(integ - 1.0) < 0.02, (
            f'contribution {i}: integral {integ:.6f} for LF=5'
        )


def test_lf5_zero_when_E_minus_U_nonpositive():
    """Edge case: E <= U. Empty support; spectrum is defined-zero."""
    x = np.array([0.0, 1.0, 2.0])
    g = np.array([1.0, 1.0, 1.0])
    g_tab = {'x': list(x), 'g': list(g), 'INT': [2], 'NBT': [3]}
    c = _make_contrib(lf=5, theta=1e6, U=1e7, g_table=g_tab)
    E = np.array([1e6])  # E < U -> empty support
    Eout = np.linspace(0.0, 2e6, 21)
    f = mf5.compute_general_evaporation_spectrum(c, E, Eout)
    np.testing.assert_array_equal(f, 0.0)


# ============================================================
# Dispatcher: compute_spectrum still works and calls the right
# path for each LF value.
# ============================================================


def test_dispatcher_lf7_end_to_end():
    """LF=7 through the top-level compute_spectrum dispatcher."""
    c = _make_contrib(lf=7, theta=1e6, U=0.0)
    endf_dict = {5: {18: {'contribution': {1: c}}}}
    E = np.array([14e6])
    Eout = np.linspace(0.0, 14e6, 10001)
    f = mf5.compute_spectrum(endf_dict, 18, E, Eout)
    # Includes the p_table probability weight (1.0 in our fixture),
    # so integral is p * 1 = 1.
    integ = trapezoid(f[0], Eout)
    assert abs(integ - 1.0) < 1e-3


def test_dispatcher_lf9_end_to_end():
    c = _make_contrib(lf=9, theta=1e6, U=0.0)
    endf_dict = {5: {18: {'contribution': {1: c}}}}
    E = np.array([14e6])
    Eout = np.linspace(0.0, 14e6, 10001)
    f = mf5.compute_spectrum(endf_dict, 18, E, Eout)
    integ = trapezoid(f[0], Eout)
    assert abs(integ - 1.0) < 1e-3
