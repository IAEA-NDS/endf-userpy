"""Tests for energy-distribution reconstruction from MF4 angular data.

Pinned by issue #21 / the elastic-peak investigation: the LAB <-> CM
kinematic conversion in primitives.conversion_relativistic uses a
"cos(pi - theta_lab)" convention internally, which is the *opposite*
of the convention used by the angular-distribution evaluators
(mu = cos theta_lab). The bridging negation lives in
distribution1d_helpers._prepare_angdist_to_energydist_conversion;
without it, the dxs/dE shape for elastic on a heavy target ends up
mirror-flipped and the elastic peak appears at the lower end of the
kinematic window instead of the upper end (E_in).
"""

from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities_mt_zap import distribution1d as d1d
from endf_userpy.quantities_mt_zap.distribution1d_helpers import (
    _prepare_angdist_to_energydist_conversion,
)
from endf_userpy.primitives.physical_constants import get_zap_for_particle


DATA_DIR = Path(__file__).resolve().parent / 'data'


@pytest.fixture(scope='module')
def al27_endf_dict():
    parser = EndfParserCpp(ignore_missing_tpid=True)
    return parser.parsefile(DATA_DIR / 'jeff33_13-Al-27g_mf4_mt2.endf')


def test_kinematic_conversion_mu_increases_with_E_out(al27_endf_dict):
    """For elastic scattering on a heavy target, the LAB outgoing
    energy E_out increases monotonically with mu = cos(theta_lab):
    forward scattering (mu = +1) gives the kinematic maximum E_out
    (= E_in), backscattering (mu = -1) gives the minimum. The bridging
    negation in _prepare_angdist_to_energydist_conversion is what
    makes this monotonicity hold; without it the mapping is inverted
    and the elastic dxs/dE peak ends up at the wrong window edge.
    """
    e_in = 1.4e7
    awr = al27_endf_dict[1][451]['AWR']
    e_out_min_approx = e_in * ((awr - 1) / (awr + 1)) ** 2
    # Sample inside the window with margin from the boundaries.
    width = e_in - e_out_min_approx
    e_out = e_out_min_approx + 0.05 * width + 0.9 * width * np.linspace(0, 1, 9)
    mu, _ = _prepare_angdist_to_energydist_conversion(
        al27_endf_dict, 2, get_zap_for_particle('n'),
        np.array([e_in]), e_out, to_lab=True,
    )
    # mu monotonically increasing with e_out
    assert np.all(np.diff(mu[0]) > 0), f'mu not monotonic in E_out: {mu[0]}'
    # Lower sample (5% above kinematic min) backward-leaning,
    # upper sample (5% below kinematic max) forward-leaning.
    assert mu[0, 0] < 0, f'low-E sample should map to backward mu, got {mu[0, 0]}'
    assert mu[0, -1] > 0, f'high-E sample should map to forward mu, got {mu[0, -1]}'


def test_elastic_dxs_dE_peak_in_upper_kinematic_half(al27_endf_dict):
    """End-to-end: peak of dxs/dE for elastic on a heavy target sits
    in the upper half of the kinematic window.

    Al-27 elastic at 14 MeV is forward-peaked, so the dxs/dE profile
    must concentrate at high E_out, near the kinematic maximum
    (= E_in). If the kinematic conversion sign is wrong the profile
    is mirror-flipped about the window centre and the peak lands at
    the lower end instead.
    """
    awr = al27_endf_dict[1][451]['AWR']
    e_in = 1.4e7
    e_out_max = e_in
    e_out_min = e_in * ((awr - 1) / (awr + 1)) ** 2
    e_window_centre = 0.5 * (e_out_min + e_out_max)
    # Sample the window densely so the peak is well-resolved.
    e_out = np.linspace(e_out_min * 0.999, e_out_max * 1.001, 5000)
    edist = d1d.compute_energydist_values(
        al27_endf_dict, 2, get_zap_for_particle('n'),
        np.array([e_in]), e_out, to_lab=True,
    )
    peak_e_out = e_out[np.argmax(edist[0])]
    assert peak_e_out > e_window_centre, (
        f'expected dxs/dE peak in upper half of [{e_out_min/1e6:.3f}, '
        f'{e_out_max/1e6:.3f}] MeV (centre {e_window_centre/1e6:.3f} MeV), '
        f'got peak at {peak_e_out/1e6:.3f} MeV'
    )


def test_kinematic_conversion_round_trip_unchanged(al27_endf_dict):
    """The internal convention of conversion_relativistic is unchanged
    by the boundary fix: cos_phi(Ekin) and Ekin(cos_phi) still
    round-trip exactly."""
    from endf_userpy.primitives.conversion_relativistic import (
        compute_cos_phi_from_Ekin,
        compute_Ekin_from_cos_phi,
    )
    from endf_userpy.primitives.properties import (
        get_projectile_mass, get_target_mass, get_reaction_qvalue,
    )
    from endf_userpy.primitives.physical_constants import get_particle_mass_for_zap

    m_i = get_projectile_mass(al27_endf_dict)
    m_t = get_target_mass(al27_endf_dict)
    m_e = get_particle_mass_for_zap(get_zap_for_particle('n'))
    qval = get_reaction_qvalue(al27_endf_dict, 2)
    m_r = m_t + (m_i - m_e) - qval

    e_in = np.array([[1.4e7]])
    cos_in = np.linspace(-0.99, 0.99, 11).reshape(1, -1)
    e_out = compute_Ekin_from_cos_phi(cos_in, e_in, m_i, m_t, m_e, m_r)
    cos_back = compute_cos_phi_from_Ekin(e_out, e_in, m_i, m_t, m_e, m_r)
    assert np.allclose(cos_in, cos_back, atol=1e-6)
