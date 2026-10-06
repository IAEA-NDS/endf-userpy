"""MF6 LAW=5 DDX pipeline (issue #335 wiring).

The first #264 increment implemented the angular distribution so
``get_particle_production_dxs_dmu`` works for LAW=5 subsections.
``get_particle_production_dxs_dE`` and ``get_particle_production_ddxs``
still raised because the kinematic-delta folder in
``ddx_broadening.compute_ddx_discrete_broadened`` gated on
``has_angdist_part``, which recognised only LAW in (2, 3, 4). Issue
#335 adds LAW=5 to that whitelist; the folder then composes the
LAW=5 angular distribution with the two-body elastic kinematic
``E_out_kin(E_in, mu)`` to produce a broadened DDX on a finite
grid, same pattern as LAW=2 / LAW=3 / LAW=4.

Pins:
- ``has_angdist_part`` admits LAW=5 subsections (synthetic-dict
  check, no corpus required).
- ``get_particle_production_dxs_dmu`` reaches the LAW=5 handler and
  returns finite output.
- ``get_particle_production_dxs_dE`` with ``broadening=sigma_eV``
  returns a finite non-negative DDX.
- ``get_particle_production_ddxs`` with ``broadening=sigma_eV``
  returns a finite non-negative DDX of shape (n_Ein, n_Eout, n_mu).
- Area-under-E_out at fixed Ein, mu matches
  ``get_particle_production_dxs_dmu(Ein, mu)`` within integration
  tolerance (the kernel integrates to 1 so the kinematic-delta
  area is preserved).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation.mf6_interpretation_helpers import (
    has_angdist_part,
)
from endf_userpy.quantities import (
    get_particle_production_ddxs,
    get_particle_production_dxs_dE,
    get_particle_production_dxs_dmu,
)

from _corpus import resolve_p_he3_law5


# ---- Fixtures --------------------------------------------------------


@pytest.fixture(scope='module')
def p_he3_endf_dict():
    path = resolve_p_he3_law5()
    if path is None:
        pytest.skip('p + He-3 LAW=5 corpus file not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


# ---- Synthetic-dict predicate check ---------------------------------


def test_has_angdist_part_admits_law5():
    """Synthetic MF6 subsection with LAW=5 must be recognised by
    ``has_angdist_part`` as a two-body kinematic-delta source. Pins
    the one-line whitelist extension in
    mf6_interpretation_helpers.py."""
    # Minimal subsection stub that `has_angdist_part` reads: it only
    # needs LAW. ZAP is required by `contains_zap`.
    d = {
        1: {451: {'ZA': 1001.0, 'AWR': 1.0, 'PROJECTILE': 1001}},
        3: {2: {'ZA': 1001.0, 'AWR': 1.0,
                'xstable': {'E': [1.0e3, 2.0e7],
                             'xs': [1.0, 1.0],
                             'INT': [2], 'NBT': [2]}}},
        6: {
            2: {
                'LCT': 2, 'NK': 1,
                'AWR': 1.0, 'ZA': 1001.0,
                'subsection': {
                    1: {
                        'ZAP': 1001.0, 'AWP': 1.0,
                        'LIP': 0, 'LAW': 5,
                        'LIDP': 0, 'SPI': 0.5,
                        'yields': {'Eint': [1e3, 2e7], 'yi': [1., 1.],
                                    'INT': [2], 'NBT': [2]},
                    },
                },
            },
        },
    }
    assert has_angdist_part(d, mt=2, zap=1001) is True

    # LAW=1 same subsection must NOT be recognised as angdist-part.
    d[6][2]['subsection'][1]['LAW'] = 1
    assert has_angdist_part(d, mt=2, zap=1001) is False


# ---- Corpus-file pipeline smokes ------------------------------------


def test_dxs_dmu_runs_on_law5_file(p_he3_endf_dict):
    """``get_particle_production_dxs_dmu`` on p+He-3 (LAW=5 LTP=1
    LIDP=0) must return a finite non-negative array. Validated
    against the direct LAW=5 handler output in a different test; here
    we only pin that the top-level entry does not raise."""
    e_in = np.array([5.0e6, 1.0e7])
    mu = np.linspace(-0.5, 0.9, 11)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.asarray(get_particle_production_dxs_dmu(
            p_he3_endf_dict, '(p,p_0)', 'p', e_in, mu,
        ))
    assert out.shape == (2, 11)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.any(out > 0.0)


def test_dxs_dE_broadened_runs_on_law5_file(p_he3_endf_dict):
    """``get_particle_production_dxs_dE`` with ``broadening`` must
    produce finite non-negative DDX by folding the kinematic delta
    into the user kernel, same pattern as LAW=2 / LAW=3 / LAW=4."""
    e_in = np.array([5.0e6, 1.0e7])
    e_out = np.linspace(1.0e5, 1.0e7, 25)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.asarray(get_particle_production_dxs_dE(
            p_he3_endf_dict, '(p,p_0)', 'p', e_in, e_out,
            broadening=1.0e6,
        ))
    assert out.shape == (2, 25)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.any(out > 0.0)


def test_ddxs_broadened_runs_on_law5_file(p_he3_endf_dict):
    """``get_particle_production_ddxs`` with ``broadening`` must
    produce finite non-negative DDX of shape (n_Ein, n_Eout, n_mu)."""
    e_in = np.array([5.0e6, 1.0e7])
    e_out = np.linspace(1.0e5, 1.0e7, 25)
    mu = np.linspace(-0.5, 0.9, 11)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.asarray(get_particle_production_ddxs(
            p_he3_endf_dict, '(p,p_0)', 'p', e_in, e_out, mu,
            broadening=1.0e6,
        ))
    assert out.shape == (2, 25, 11)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.any(out > 0.0)


def test_ddx_integrates_to_dxs_dmu_over_eout(p_he3_endf_dict):
    """The broadening kernel integrates to 1, so the broadened DDX
    integrated over E_out at fixed (E_in, mu) must recover the
    unbroadened ``dxs/dmu(E_in, mu)``. Pins the kinematic-delta
    folder's area-conservation property for the LAW=5 branch; same
    invariant that holds for LAW=2 / LAW=3 / LAW=4."""
    e_in = np.array([1.0e7])
    mu = np.array([0.5])
    # Dense E_out grid centred on the elastic kinematic value so the
    # broadened kernel integral captures ~all the area.
    e_out = np.linspace(1.0e6, 1.0e7, 400)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ddx = np.asarray(get_particle_production_ddxs(
            p_he3_endf_dict, '(p,p_0)', 'p', e_in, e_out, mu,
            broadening=5.0e5,
        ))
        dxs_dmu = np.asarray(get_particle_production_dxs_dmu(
            p_he3_endf_dict, '(p,p_0)', 'p', e_in, mu,
        ))
    area = np.trapezoid(ddx[0, :, 0], e_out)
    expected = dxs_dmu[0, 0]
    rel = abs(area - expected) / expected
    assert rel < 5e-2, (
        f'DDX integral over E_out = {area:.3e}, dxs/dmu = '
        f'{expected:.3e}, rel diff = {rel:.3e}. The kinematic '
        f'kernel integrates to 1 so these must agree within '
        f'integration tolerance.'
    )
