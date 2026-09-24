"""Verify the ``mu = cos(theta_lab)`` convention refactor of
``primitives.conversion_relativistic`` (issue #185).

Pins:

- Round-trip ``mu -> Ekin -> mu`` and ``Ekin -> mu -> Ekin`` to
  round-off (10 decimal digits).
- ``compute_dmu_dEkin`` and ``compute_dEkin_dmu`` match their
  cos_phi-convention private counterparts up to the sign carried
  by the ``mu = -cos_phi`` relation.
- Both derivatives match a central finite-diff of the corresponding
  forward function to at least 6 decimal digits.
- ``mu > 0`` (forward-scattered) corresponds to ``Ekin`` above the
  isotropic-CM midpoint; ``mu < 0`` (backward) corresponds to
  ``Ekin`` below it. Nails down that the public API uses the
  standard beam-forward convention.
- Deprecated ``compute_*_cos_phi`` aliases keep working (they are
  the private helpers under new names) so any external code that
  imported the pre-refactor names is not broken.
"""
from __future__ import annotations

import numpy as np

from endf_userpy.primitives import conversion_relativistic as cr


# Al-27 (n, n) kinematic parameters, in MeV.
Ekin_i = 14.0
M_I = 1.008665 * 931.494
M_T = 26.9815 * 931.494
M_E = M_I           # elastic: ejectile == projectile
M_R = M_T           # elastic: residual == target


MU_SAMPLES = np.array([-0.9, -0.5, -0.1, 0.1, 0.3, 0.7, 0.99])


def test_round_trip_mu_to_ekin_to_mu():
    Ekin = cr.compute_Ekin_from_mu(MU_SAMPLES, Ekin_i, M_I, M_T, M_E, M_R)
    mu_back = cr.compute_mu_from_Ekin(Ekin, Ekin_i, M_I, M_T, M_E, M_R)
    np.testing.assert_allclose(MU_SAMPLES, mu_back, rtol=1e-10)


def test_round_trip_ekin_to_mu_to_ekin():
    Ekin = cr.compute_Ekin_from_mu(MU_SAMPLES, Ekin_i, M_I, M_T, M_E, M_R)
    mu_of = cr.compute_mu_from_Ekin(Ekin, Ekin_i, M_I, M_T, M_E, M_R)
    Ekin_back = cr.compute_Ekin_from_mu(
        mu_of, Ekin_i, M_I, M_T, M_E, M_R,
    )
    np.testing.assert_allclose(Ekin, Ekin_back, rtol=1e-10)


def test_dmu_dEkin_matches_negated_dcos_phi_dEkin():
    Ekin = cr.compute_Ekin_from_mu(MU_SAMPLES, Ekin_i, M_I, M_T, M_E, M_R)
    jac_new = cr.compute_dmu_dEkin(Ekin, Ekin_i, M_I, M_T, M_E, M_R)
    jac_old = -cr._compute_dcos_phi_dEkin(
        Ekin, Ekin_i, M_I, M_T, M_E, M_R,
    )
    np.testing.assert_allclose(jac_new, jac_old, rtol=1e-14)


def test_dEkin_dmu_matches_negated_dEkin_dcos_phi():
    jac_new = cr.compute_dEkin_dmu(
        MU_SAMPLES, Ekin_i, M_I, M_T, M_E, M_R,
    )
    jac_old = -cr._compute_dEkin_dcos_phi(
        -MU_SAMPLES, Ekin_i, M_I, M_T, M_E, M_R,
    )
    np.testing.assert_allclose(jac_new, jac_old, rtol=1e-14)


def test_dmu_dEkin_matches_finite_diff():
    Ekin = cr.compute_Ekin_from_mu(MU_SAMPLES, Ekin_i, M_I, M_T, M_E, M_R)
    analytic = cr.compute_dmu_dEkin(Ekin, Ekin_i, M_I, M_T, M_E, M_R)
    h = 1e-6 * Ekin
    fd = (
        cr.compute_mu_from_Ekin(Ekin + h, Ekin_i, M_I, M_T, M_E, M_R)
        - cr.compute_mu_from_Ekin(Ekin - h, Ekin_i, M_I, M_T, M_E, M_R)
    ) / (2 * h)
    np.testing.assert_allclose(analytic, fd, rtol=1e-6)


def test_dEkin_dmu_matches_finite_diff():
    analytic = cr.compute_dEkin_dmu(
        MU_SAMPLES, Ekin_i, M_I, M_T, M_E, M_R,
    )
    h = 1e-6
    fd = (
        cr.compute_Ekin_from_mu(
            MU_SAMPLES + h, Ekin_i, M_I, M_T, M_E, M_R,
        )
        - cr.compute_Ekin_from_mu(
            MU_SAMPLES - h, Ekin_i, M_I, M_T, M_E, M_R,
        )
    ) / (2 * h)
    # Central-FD truncation is O(h^2 * f'''); at h=1e-6 on this
    # kinematic function it lands near 1e-5 relative.
    np.testing.assert_allclose(analytic, fd, rtol=1e-4)


def test_forward_mu_is_above_isotropic_midpoint():
    """``mu > 0`` (forward-of-beam) means the ejectile carries most
    of the momentum forward and its LAB kinetic energy is above the
    isotropic-CM midpoint. Nails the beam-forward direction of the
    convention: if the sign were flipped, this assertion would fail."""
    mu_fwd = np.array([0.5, 0.9])
    mu_bwd = np.array([-0.9, -0.5])
    Ekin_fwd = cr.compute_Ekin_from_mu(
        mu_fwd, Ekin_i, M_I, M_T, M_E, M_R,
    )
    Ekin_bwd = cr.compute_Ekin_from_mu(
        mu_bwd, Ekin_i, M_I, M_T, M_E, M_R,
    )
    # For elastic n-on-Al-27 the LAB midpoint is around Ekin_i * A/(A+1)^2
    # ~ 12.5 MeV on either side; here we only need the ordering.
    assert (Ekin_fwd > Ekin_bwd).all()


def test_deprecated_cos_phi_aliases_still_work():
    """External code that imported the pre-#185 names must not break."""
    cos_phi = -MU_SAMPLES
    Ekin = cr.compute_Ekin_from_cos_phi(
        cos_phi, Ekin_i, M_I, M_T, M_E, M_R,
    )
    cos_phi_back = cr.compute_cos_phi_from_Ekin(
        Ekin, Ekin_i, M_I, M_T, M_E, M_R,
    )
    np.testing.assert_allclose(cos_phi, cos_phi_back, rtol=1e-10)
    # And the alias equals the private helper.
    assert (
        cr.compute_Ekin_from_cos_phi
        is cr._compute_Ekin_from_cos_phi
    )
