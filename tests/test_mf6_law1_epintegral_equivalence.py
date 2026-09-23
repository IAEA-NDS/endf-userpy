"""MF6 LAW=1 continuum ``E'``-integral: Python port vs Fortran
reference plus numpy-vs-JAX parity plus ``jax.grad`` sanity.

The port replaces the Fortran per-subpanel Romberg-Richardson
extrapolation with polar-angle Gauss-Legendre quadrature, so
agreement with the Fortran is not bit-identical. The Fortran
``feep_uint_law1con`` runs Romberg with a rtol of 1e-3 and the
integrand has kinks from the LEP-piecewise Ep structure as
``mu`` varies, so neither Romberg nor fixed-order Gauss-Legendre
converges below a few 1e-3 relative on the (E, E') grid. A
scipy.integrate.quad reference at epsrel=1e-10 confirms both
implementations sit within a few 1e-3 of the "truth" and within
~1e-2 of each other; the tolerance below reflects that.

Pins:
- numpy port matches Fortran to rtol=1.5e-2 (median ~1e-3).
- numpy port bit-identical to JAX port (rtol=1e-11).
- jax.grad reaches back from a dict-stored ``b`` leaf and matches
  finite-diff.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_integrals as py_int,
    mf6_interpretation_integrals_fort as fort_int,
)
from endf_userpy.primitives import array_ns

from _corpus import resolve_al27


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip(
            'Al-27 ENDF file not available (run '
            'tests/data_law1_adhoc/fetch.sh)'
        )
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def _query_grid(subsec, panel_idx, n_ein=3, n_eout=40):
    """(E, E') grid inside the given panel, avoiding the panel
    boundaries where the LEP-piecewise structure is sharpest."""
    E1 = float(subsec['E'][panel_idx + 1])
    E2 = float(subsec['E'][panel_idx + 2])
    e_in = np.linspace(E1, E2, n_ein + 2)[1:-1]
    ep_vals = list(subsec['Ep'][panel_idx + 1].values())
    # Skip the first few Ep (may be discrete lines / low-signal);
    # end short of the tail where the amplitude tapers.
    ep_lo = float(ep_vals[3])
    ep_hi = float(ep_vals[-1])
    e_out = np.linspace(ep_lo * 1.05, ep_hi * 0.9, n_eout)
    return e_in, e_out


def test_law1_epintegral_python_matches_fortran_al27_mt91(al27_endf_dict):
    """Al-27 MT=91 (n, n') continuum: numpy Python port vs Fortran
    Romberg reference. Rtol=1.5e-2 reflects the Romberg-vs-GL
    tradeoff at kink-heavy integrands (see module docstring)."""
    mt, sn = 91, 1
    subsec = al27_endf_dict[6][mt]['subsection'][sn]
    # Panel_idx=7 (0-indexed) = incident-energy panel [1e7, 1.1e7]
    # eV, well inside the tabulated range.
    e_in, e_out = _query_grid(subsec, panel_idx=7)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fo = fort_int.get_energydist_from_subsec_law1_fort(
            al27_endf_dict, mt, sn, e_in, e_out, True,
        )
        py = py_int.get_energydist_from_subsec_law1(
            al27_endf_dict, mt, sn, e_in, e_out, True,
        )
    py = np.asarray(py)
    fo = np.asarray(fo)
    mask = fo > 1e-15
    rd = np.abs(py[mask] - fo[mask]) / fo[mask]
    assert rd.max() < 1.5e-2, (
        f'max rel diff {rd.max():.3e} exceeds 1.5e-2; median '
        f'{np.median(rd):.3e}'
    )
    # Median should be well inside Fortran Romberg's rtol=1e-3.
    assert np.median(rd) < 3e-3


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law1_epintegral_numpy_jax_parity_al27_mt91(al27_endf_dict):
    """Numpy vs JAX bit-identical: same algorithm, same nodes and
    weights, only the backend adapter differs."""
    mt, sn = 91, 1
    subsec = al27_endf_dict[6][mt]['subsection'][sn]
    e_in, e_out = _query_grid(subsec, panel_idx=7)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        a = np.asarray(py_int.get_energydist_from_subsec_law1(
            al27_endf_dict, mt, sn, e_in, e_out, True, xp=xp_np,
        ))
        b = np.asarray(py_int.get_energydist_from_subsec_law1(
            al27_endf_dict, mt, sn, e_in, e_out, True, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law1_epintegral_jax_grad_from_dict_leaf(al27_endf_dict):
    """``jax.grad`` of a scalar summary of ``f(E, E')`` reaches back
    to a dict-stored ``b`` leaf and matches finite-difference.

    Selects panel_key=8 (dict is 1-indexed; corresponds to
    panel_idx=7 in the query) so the tracer sits on the enclosing
    incident-energy panel and the gradient is nonzero."""
    import jax
    import jax.numpy as jnp
    mt, sn = 91, 1
    subsec = al27_endf_dict[6][mt]['subsection'][sn]
    e_in, e_out = _query_grid(subsec, panel_idx=7, n_ein=1, n_eout=20)
    xp_jx = array_ns.get_backend('jax')
    panel_key = 8      # 1-indexed dict key = 0-indexed panel 7
    ep_row = 5         # 1-indexed
    coef = 1           # 1-indexed: b[panel][row][col]
    original = float(
        al27_endf_dict[6][mt]['subsection'][sn]['b'][panel_key][ep_row][coef]
    )

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[6][mt]['subsection'][sn]['b'][panel_key][ep_row][coef] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return jnp.sum(py_int.get_energydist_from_subsec_law1(
                d_t, mt, sn, e_in, e_out, True, xp=xp_jx,
            ))

    val = float(loss(jnp.array(original)))
    grad = float(jax.grad(loss)(jnp.array(original)))
    assert val > 0.0
    assert np.isfinite(grad)
    # The chosen leaf is inside the enclosing panel -> nonzero grad.
    assert abs(grad) > 0.0
    eps = 1e-4
    lp = float(loss(jnp.array(original + eps)))
    lm = float(loss(jnp.array(original - eps)))
    fd = (lp - lm) / (2.0 * eps)
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-12)
