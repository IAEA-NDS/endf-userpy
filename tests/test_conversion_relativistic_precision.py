"""Relativistic two-body kinematics without catastrophic cancellation.

The residual mass shell ``2 P p mu = m_r^2 + p^2 + P^2 - E_r^2``
subtracts two numbers of order (target mass)^2 (5e22 eV^2 for U-235)
to get a result of order p P (1e15 eV^2 at MeV). The SymPy expansions
the module used before lost up to 1.1e-5 relative in mu and 2.6e-9 in
Ekin on every backend, which also showed up as a 3.9e-9 jit-vs-numpy
difference in U-235 dxs/dE near the inelastic thresholds. Checked here
against an mpmath evaluation of the exact relation on synthetic masses.
"""
import numpy as np
import pytest

from endf_userpy.primitives import conversion_relativistic as cr

mp = pytest.importorskip('mpmath')

M_N = 939.56542052e6              # neutron mass (eV)
M_U = 235.0439299 * 931.49410242e6
M_AL = 26.98153841 * 931.49410242e6


def _masses(m_t, q):
    """``(m_i, m_t, m_e, m_r)`` for (n, n') with reaction Q-value ``q``."""
    return M_N, m_t, M_N, m_t - q


def _mu_exact(E, Ei, m_i, m_t, m_e, m_r):
    """Exact mu at the current mpmath precision (callers raise it)."""
    E, Ei, m_i, m_t, m_e, m_r = (mp.mpf(x) for x in (E, Ei, m_i, m_t, m_e, m_r))
    pe2 = E**2 + 2 * E * m_e
    pi2 = Ei**2 + 2 * Ei * m_i
    Er = m_t + m_i + Ei - m_e - E
    return (m_r**2 + pe2 + pi2 - Er**2) / (2 * mp.sqrt(pe2 * pi2))


CASES = [
    (M_U, 0.0, 1e6),        # U-235 elastic
    (M_U, -0.2e6, 6.2e5),   # inelastic level just above threshold
    (M_AL, -0.84e6, 1.0e6),
    (M_AL, 0.0, 1.4e7),
]


@pytest.mark.parametrize('m_t,q,Ei', CASES)
def test_mu_from_ekin_matches_exact(m_t, q, Ei):
    m = _masses(m_t, q)
    mus = np.linspace(-0.98, 0.98, 9)
    E = cr.compute_Ekin_from_mu(mus, np.full_like(mus, Ei), *m)
    assert np.all(np.isfinite(E)) and np.all(E > 0)
    for e in E:
        with mp.workdps(60):
            exact = float(_mu_exact(e, Ei, *m))
        got = float(cr.compute_mu_from_Ekin(np.array(e), np.array(Ei), *m))
        assert abs(got - exact) <= 1e-12 * abs(exact) + 1e-14


@pytest.mark.parametrize('m_t,q,Ei', CASES)
def test_ekin_from_mu_matches_exact(m_t, q, Ei):
    m = _masses(m_t, q)
    for mu in (-0.9, -0.2, 0.3, 0.95):
        E = float(cr.compute_Ekin_from_mu(np.array(mu), np.array(Ei), *m))
        with mp.workdps(60):
            # Ekin error implied by the residual in mu (first order)
            dmu = mp.diff(lambda x: _mu_exact(x, Ei, *m), mp.mpf(E))
            err = abs(float((_mu_exact(E, Ei, *m) - mu) / dmu)) / E
        assert err < 1e-13


@pytest.mark.parametrize('m_t,q,Ei', CASES)
def test_derivatives_match_exact_and_are_reciprocal(m_t, q, Ei):
    m = _masses(m_t, q)
    for mu in (-0.7, 0.1, 0.8):
        E = float(cr.compute_Ekin_from_mu(np.array(mu), np.array(Ei), *m))
        with mp.workdps(60):
            exact = float(mp.diff(lambda x: _mu_exact(x, Ei, *m), mp.mpf(E)))
        dmu_dE = float(cr.compute_dmu_dEkin(np.array(E), np.array(Ei), *m))
        dE_dmu = float(cr.compute_dEkin_dmu(np.array(mu), np.array(Ei), *m))
        np.testing.assert_allclose(dmu_dE, exact, rtol=1e-11)
        np.testing.assert_allclose(dE_dmu * dmu_dE, 1.0, rtol=1e-11)
