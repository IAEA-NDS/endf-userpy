"""MF6 LAW=4 recoil angular distribution.

Phase 2 base-physics coverage for LAW=4 (previously raised
NotImplementedError; roadmap #198). LAW=4 has no LAW-dependent
structure; the recoil's angular and energy distributions are
determined by two-body kinematics from a sibling LAW=2 or LAW=3
subsection in the same MT.

Reconstruction:

- Sibling is found in the same MT (first LAW in {2,3}).
- Recoil goes in the opposite direction to the ejectile in CM:
  ``f_recoil_CM(mu) = f_ejectile_CM(-mu)``.
- CM<->LAB Jacobian applied with THIS subsection's ``AWP`` (the
  recoil's mass), not the sibling's.

Uses Be-9 MT=600 (n,p_0 -> proton + Li-9) sub 2 (LAW=4 recoil,
sibling sub 1 is LAW=3 proton, ZAP=3009 Li-9, LCT=2 CM).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_subsecs as mf6subsec,
)
from endf_userpy.primitives import array_ns


DATA_DIR = Path(__file__).parent / 'data'


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def be9_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-004_Be_009.endf'),
    )


def test_law4_previously_notimplemented_now_returns(be9_endf_dict):
    """Regression pin: LAW=4 dispatch through
    ``compute_angdist_from_subsec`` no longer raises."""
    ein = np.array([1.6e7, 1.8e7])
    mu = np.linspace(-0.9, 0.9, 21)
    f = np.asarray(mf6subsec.compute_angdist_from_subsec(
        be9_endf_dict, 600, 2, ein, mu, to_lab=True,
    ))
    assert f.shape == (2, 21)
    assert np.isfinite(f).all()
    assert (f >= 0).all()


def test_law4_lab_angular_single_branch_regression(be9_endf_dict):
    """Regression pin for the single-branch CM<->LAB limitation
    (issue #210): for heavy recoils (r^2 < 1) the LAB density is
    a 2-branch sum but the primitive returns only one branch, so
    the integral is less than 1. Pin the current value so a fix
    to #210 flips this test and forces reconsideration.

    The forward-cone shape and per-mu values from the single
    branch are still correct; only the overall normalisation is
    affected."""
    ein = np.array([1.8e7])
    mu = np.linspace(-1.0, 1.0, 40001)
    f = np.asarray(mf6subsec.get_angdist_from_subsec_law4(
        be9_endf_dict, 600, 2, ein, mu, to_lab=True,
    ))
    integral = float(np.trapezoid(f[0], mu))
    # Current single-branch integral is ~0.66; well below 1.0 and
    # well above 0. When #210 is fixed to sum both branches the
    # integral should be ~1.0 and this test should flip.
    assert 0.5 < integral < 0.9, (
        f'LAW=4 single-branch integral outside expected range: '
        f'{integral:.6f}'
    )


def test_law4_recoil_narrow_forward_peak(be9_endf_dict):
    """Physical sanity: the Li-9 recoil (AWP ~9) from a light
    proton ejectile (AWP ~1) is heavy and slow, so its LAB angular
    distribution is confined to a narrow forward cone. Assert that
    the LAB f near mu=+1 is nonzero while the LAB f near mu=0 (or
    negative) is zero (below the kinematic cone edge)."""
    ein = np.array([1.8e7])
    mu = np.array([-0.9, 0.0, 0.5, 0.9, 0.95, 0.99])
    f = np.asarray(mf6subsec.get_angdist_from_subsec_law4(
        be9_endf_dict, 600, 2, ein, mu, to_lab=True,
    ))
    # Forbidden angles (well outside the forward cone): zero.
    assert f[0, 0] == 0.0                 # mu = -0.9
    assert f[0, 1] == 0.0                 # mu =  0.0
    assert f[0, 2] == 0.0                 # mu =  0.5
    # Inside the forward cone: nonzero.
    assert f[0, 3] > 0.0                  # mu =  0.9
    assert f[0, 4] > 0.0                  # mu =  0.95
    assert f[0, 5] > 0.0                  # mu =  0.99


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law4_numpy_jax_parity(be9_endf_dict):
    """xp=jax and xp=numpy return bit-identical values."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1.6e7, 1.7e7, 1.8e7, 1.9e7])
    mu = np.linspace(-0.9, 0.9, 21)

    f_np = np.asarray(mf6subsec.get_angdist_from_subsec_law4(
        be9_endf_dict, 600, 2, ein, mu, to_lab=True, xp=xp_np,
    ))
    f_jx = np.asarray(mf6subsec.get_angdist_from_subsec_law4(
        be9_endf_dict, 600, 2,
        jnp.asarray(ein), jnp.asarray(mu),
        to_lab=True, xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-11, atol=1e-30)


def test_law4_gamma_zap_short_circuits_with_warning():
    """A LAW=4 subsection storing a gamma (ZAP=0) short-circuits
    with a warning and returns zeros."""
    endf = EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-004_Be_009.endf'),
    )
    endf_mut = copy.deepcopy(endf)
    endf_mut[6][600]['subsection'][2]['ZAP'] = 0.0

    with pytest.warns(UserWarning, match='ZAP=0'):
        f = np.asarray(mf6subsec.get_angdist_from_subsec_law4(
            endf_mut, 600, 2, np.array([1.8e7]), np.linspace(-0.9, 0.9, 5),
            to_lab=True,
        ))
    assert (f == 0.0).all()


def test_law4_missing_sibling_raises():
    """When no LAW=2 or LAW=3 sibling is present in the MT, a
    LAW=4 subsection must raise a clear ValueError rather than
    silently returning zeros."""
    endf = EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-004_Be_009.endf'),
    )
    endf_mut = copy.deepcopy(endf)
    # Flip the sibling's LAW so it's no longer a valid sibling.
    endf_mut[6][600]['subsection'][1]['LAW'] = 1
    with pytest.raises(ValueError, match='no sibling'):
        mf6subsec.get_angdist_from_subsec_law4(
            endf_mut, 600, 2, np.array([1.8e7]),
            np.linspace(-0.9, 0.9, 5), to_lab=True,
        )
