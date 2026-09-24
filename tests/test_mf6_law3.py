"""MF6 LAW=3 isotropic discrete emission angular distribution.

Phase 2 base-physics coverage for LAW=3 (previously raised
NotImplementedError; roadmap #198). LAW=3 has no LAW-dependent
structure: the angular distribution is isotropic-in-CM
(``f_CM(mu) = 1/2`` for all mu, all E), and two-body kinematics
determine E' from AWR / AWP / QI (delta on E' at
``E'_kin(mu, E)``).

This file pins:

- Regression: LAW=3 no longer raises through ``compute_angdist_from_subsec``.
- LCT=1 case gives literal ``f_LAB = 0.5`` uniformly.
- LCT=2 case: the CM<->LAB Jacobian is applied; ``f_LAB`` is a
  valid probability density in ``mu_LAB`` (non-negative, integrates
  to 1 to sub-permille on a fine mu grid).
- Backend parity (numpy vs jax).

Uses Be-9 MT=600 (n,p_0 -> He-6 + p) subsection 1: LAW=3, ZAP=1001
(proton), LCT=2 CM frame. Real committed corpus file.
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


def test_law3_previously_notimplemented_now_returns(be9_endf_dict):
    """Regression pin: LAW=3 dispatch through
    ``compute_angdist_from_subsec`` no longer raises."""
    # Above threshold for Be-9(n,p_0) MT=600: Q ~ -13.8 MeV so
    # threshold ~ 15.4 MeV.
    ein = np.array([1.6e7, 1.8e7])
    mu = np.linspace(-0.9, 0.9, 21)
    f = np.asarray(mf6subsec.compute_angdist_from_subsec(
        be9_endf_dict, 600, 1, ein, mu, to_lab=True,
    ))
    assert f.shape == (2, 21)
    assert np.isfinite(f).all()
    assert (f >= 0).all()


def test_law3_lct1_gives_uniform_half(be9_endf_dict):
    """With ``to_lab=False`` the section's LCT is overridden to 1
    (LAB), so no CM<->LAB conversion runs and the isotropic-in-CM
    distribution surfaces directly as ``f_LAB = 0.5`` everywhere."""
    ein = np.array([1.6e7, 1.8e7])
    mu = np.linspace(-0.95, 0.95, 25)
    f = np.asarray(mf6subsec.get_angdist_from_subsec_law3(
        be9_endf_dict, 600, 1, ein, mu, to_lab=False,
    ))
    np.testing.assert_allclose(f, 0.5, rtol=0.0, atol=0.0)


def test_law3_lab_angular_normalises_to_one(be9_endf_dict):
    """The LAB-frame angular distribution is a probability density
    on mu_LAB with the CM->LAB Jacobian folded in; its integral
    over the allowed mu_LAB range equals 1 to sub-permille on a
    fine grid."""
    ein = np.array([1.6e7, 1.8e7])
    # Fine mu grid across the full LAB range for trapezoid.
    mu = np.linspace(-1.0, 1.0, 4000)
    f = np.asarray(mf6subsec.get_angdist_from_subsec_law3(
        be9_endf_dict, 600, 1, ein, mu, to_lab=True,
    ))
    for i in range(f.shape[0]):
        integral = float(np.trapezoid(f[i], mu))
        assert abs(integral - 1.0) < 1e-3, (
            f'LAW=3 LAB angdist not normalised at E={ein[i]}: '
            f'integral={integral:.6f}'
        )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law3_numpy_jax_parity(be9_endf_dict):
    """xp=jax and xp=numpy return bit-identical values."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1.6e7, 1.7e7, 1.8e7, 1.9e7])
    mu = np.linspace(-0.9, 0.9, 21)

    f_np = np.asarray(mf6subsec.get_angdist_from_subsec_law3(
        be9_endf_dict, 600, 1, ein, mu, to_lab=True, xp=xp_np,
    ))
    f_jx = np.asarray(mf6subsec.get_angdist_from_subsec_law3(
        be9_endf_dict, 600, 1,
        jnp.asarray(ein), jnp.asarray(mu),
        to_lab=True, xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-11, atol=1e-30)


def test_law3_gamma_zap_short_circuits_with_warning():
    """A LAW=3 subsection storing a gamma (ZAP=0) short-circuits
    with a warning and returns zeros. Constructed synthetically
    on a copy of Be-9 MT=600 with ZAP flipped to 0.0."""
    endf = EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-004_Be_009.endf'),
    )
    endf_mut = copy.deepcopy(endf)
    endf_mut[6][600]['subsection'][1]['ZAP'] = 0.0

    with pytest.warns(UserWarning, match='ZAP=0'):
        f = np.asarray(mf6subsec.get_angdist_from_subsec_law3(
            endf_mut, 600, 1, np.array([1.6e7]), np.linspace(-0.9, 0.9, 5),
            to_lab=True,
        ))
    assert (f == 0.0).all()
