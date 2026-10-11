"""Relativistic 2-body LAB-frame kinematics for the emitted particle.

The public API uses the standard scattering-angle convention:
``mu = cos(theta_lab)`` where ``theta_lab`` is the angle between the
ejectile momentum and the beam direction.

Round-trip identities (up to floating-point error):
- ``compute_mu_from_Ekin(compute_Ekin_from_mu(mu, ...), ...) == mu``
- ``compute_dmu_dEkin`` and ``compute_dEkin_dmu`` are reciprocal at
  compatible ``(mu, Ekin)`` pairs.

Meaning of variable names
-------------------------
``Ekin_i``: kinetic energy of the incident particle (MeV).
``m_i``:   mass of the incident particle (MeV).
``m_t``:   mass of the target nucleus (MeV).
``m_r``:   mass of the residual nucleus (MeV).
``m_e``:   mass of the ejectile (MeV).
``Ekin``:  kinetic energy of the ejectile (MeV).
``mu``:    cosine of ``theta_lab`` for the ejectile.

Backend-agnostic (issue #169)
-----------------------------
All four public routines accept an optional ``xp`` adapter from
:func:`endf_userpy.primitives.array_ns.get_backend`. ``xp=None``
(default) is numpy. Under ``xp=jax`` the ``sqrt`` inside the
kinematic expression is dispatched through the backend so JAX
tracers on ``mu`` / ``Ekin`` / ``Ekin_i`` (the physically variable
quantities in a fit) propagate to ``jax.grad``. The mass parameters
are inherently concrete file leaves and stay as Python floats.

Internal ``cos_phi`` convention (private ``_*_cos_phi`` helpers)
----------------------------------------------------------------
The four ``_*_cos_phi`` helpers below (originally SymPy-generated,
now rewritten in cancellation-free form) use
``cos_phi = cos(pi - theta_lab) = -mu``, i.e.
the cosine of the angle from the *opposite* of the beam direction.
The public ``*_mu`` API wraps them with the ``mu <-> cos_phi``
negation on both the angle argument and the derivative sign.

That internal convention was inherited from the original SymPy
derivation (issue #185). Callers previously had to remember to
negate ``cos_phi`` at the call site (see the pre-refactor version of
``distribution1d_helpers._prepare_angdist_to_energydist_conversion``
and ``ddx_broadening._compute_eout_kin``); the public wrappers here
remove that trap so new callers can use the standard mu convention.
"""
from . import array_ns


# --- Public API (standard mu = cos(theta_lab) convention) ------------------

def compute_Ekin_from_mu(mu, Ekin_i, m_i, m_t, m_e, m_r, xp=None):
    """Ejectile kinetic energy at the given LAB-frame ``mu``.

    ``mu`` is the standard ``cos(theta_lab)``; positive means forward
    of the beam, negative means backward.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    return _compute_Ekin_from_cos_phi(-mu, Ekin_i, m_i, m_t, m_e, m_r, xp=xp)


def compute_mu_from_Ekin(Ekin, Ekin_i, m_i, m_t, m_e, m_r, xp=None):
    """LAB-frame ``mu = cos(theta_lab)`` of the ejectile at the given
    kinetic energy ``Ekin``.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    return -_compute_cos_phi_from_Ekin(Ekin, Ekin_i, m_i, m_t, m_e, m_r, xp=xp)


def compute_dEkin_dmu(mu, Ekin_i, m_i, m_t, m_e, m_r, xp=None):
    """``dEkin / dmu`` at the given ``mu``.

    By the chain rule, ``dEkin/dmu = dEkin/d(cos_phi) * d(cos_phi)/dmu
    = -dEkin/d(cos_phi)`` since ``cos_phi = -mu``.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    return -_compute_dEkin_dcos_phi(-mu, Ekin_i, m_i, m_t, m_e, m_r, xp=xp)


def compute_dmu_dEkin(Ekin, Ekin_i, m_i, m_t, m_e, m_r, xp=None):
    """``dmu / dEkin`` at the given ejectile kinetic energy."""
    if xp is None:
        xp = array_ns.get_backend('numpy')
    return -_compute_dcos_phi_dEkin(Ekin, Ekin_i, m_i, m_t, m_e, m_r, xp=xp)


# --- Private helpers (cos_phi convention) ----------------------------------
#
# With W = m_i + Ekin_i + m_t (total LAB energy), P^2 = Ekin_i^2 +
# 2 Ekin_i m_i (beam momentum squared), the ejectile momentum p and
# energy eps = m_e + Ekin, and the residual energy E_r = W - eps, the
# residual mass shell reads
#
#     2 P p mu = m_r^2 + p^2 + P^2 - E_r^2.
#
# The terms m_r^2 and E_r^2 are ~(target mass)^2 (5e22 eV^2 for U-235)
# while the left side is ~p P (1e15 eV^2 at MeV energies), so the
# SymPy expansions this module used before cancelled catastrophically:
# relative errors up to 1.5e-8 in mu and 2.6e-9 in Ekin, on every
# backend. The forms below rewrite each large difference through the
# mass excess q = (m_t - m_r) + (m_i - m_e) (= Q for consistent masses,
# accurate to an ulp of the masses):
#
#     m_r - E_r          = (Ekin - Ekin_i) - q
#     2 (A - m_e W)      = (Ekin_i + q) (W - m_e + m_r) - P^2 =: B
#
# with A = (W^2 - P^2 + m_e^2 - m_r^2) / 2, and keep every remaining
# difference well conditioned (relative error ~1e-14 against an
# mpmath reference on U-235, Al-27 and H-1).

def _beam(Ekin_i, m_i, m_t, m_e, m_r):
    """``(P^2, W, q)``: beam momentum squared, total LAB energy and
    the mass excess ``(m_t - m_r) + (m_i - m_e)``."""
    P2 = Ekin_i**2 + 2*Ekin_i*m_i
    W = m_i + Ekin_i + m_t
    q = (m_t - m_r) + (m_i - m_e)
    return P2, W, q


def _mass_shell_numerator(Ekin, Ekin_i, m_i, m_t, m_e, m_r):
    """``(N, P^2, p^2, E_r)`` with ``N = m_r^2 + p^2 + P^2 - E_r^2 =
    2 P p mu``, without the cancellation of the direct form."""
    P2, W, q = _beam(Ekin_i, m_i, m_t, m_e, m_r)
    p2 = Ekin**2 + 2*Ekin*m_e
    E_r = W - m_e - Ekin
    N = p2 + P2 + ((Ekin - Ekin_i) - q) * (m_r + E_r)
    return N, P2, p2, E_r


def _compute_Ekin_from_cos_phi(cos_phi, Ekin_i, m_i, m_t, m_e, m_r, xp=None):
    """``Ekin(cos_phi)`` with ``cos_phi = cos(pi - theta_lab) = -mu``.

    The root ``p = (A P mu + W sqrt(D)) / (W^2 - P^2 mu^2)`` of the
    mass-shell quadratic, ``D = A^2 - m_e^2 (W^2 - P^2 mu^2)``; for
    ``A P mu < 0`` it is evaluated as ``(m_e^2 W^2 - A^2) / (A P mu -
    W sqrt(D))`` to avoid cancellation. ``Ekin = p^2 / (sqrt(p^2 +
    m_e^2) + m_e)``. NaN where no solution exists (``D < 0``).
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    sqrt = xp.sqrt
    mu = -cos_phi
    P2, W, q = _beam(Ekin_i, m_i, m_t, m_e, m_r)
    half_B = 0.5 * ((Ekin_i + q) * (W - m_e + m_r) - P2)    # A - m_e W
    A = m_e * W + half_B
    c = half_B * (2*m_e*W + half_B)                         # A^2 - m_e^2 W^2
    D = c + m_e**2 * P2 * mu**2
    W_sqrt_D = W * sqrt(D)
    a = A * sqrt(P2) * mu
    p_direct = (a + W_sqrt_D) / (W**2 - P2 * mu**2)
    p_conj = -c / (a - W_sqrt_D)
    p = xp.where(a >= 0, p_direct, p_conj)
    p2 = p * p
    denom = sqrt(p2 + m_e**2) + m_e
    return xp.where(denom == 0, 0.0, p2 / xp.where(denom == 0, 1.0, denom))


def _compute_cos_phi_from_Ekin(Ekin, Ekin_i, m_i, m_t, m_e, m_r, xp=None):
    """``cos_phi(Ekin)`` with ``cos_phi = cos(pi - theta_lab) = -mu``."""
    if xp is None:
        xp = array_ns.get_backend('numpy')
    sqrt = xp.sqrt
    N, P2, p2, _ = _mass_shell_numerator(Ekin, Ekin_i, m_i, m_t, m_e, m_r)
    return -N / (2 * sqrt(p2) * sqrt(P2))


def _compute_dcos_phi_dEkin(Ekin, Ekin_i, m_i, m_t, m_e, m_r, xp=None):
    """``d(cos_phi) / dEkin``; see module docstring for convention.

    ``d/dEkin [-N / (2 P p)]`` with ``dN/dEkin = 2 (eps + E_r)`` and
    ``dp/dEkin = eps / p``."""
    if xp is None:
        xp = array_ns.get_backend('numpy')
    sqrt = xp.sqrt
    N, P2, p2, E_r = _mass_shell_numerator(Ekin, Ekin_i, m_i, m_t, m_e, m_r)
    eps = Ekin + m_e
    P = sqrt(P2)
    p = sqrt(p2)
    return -(eps + E_r) / (P * p) + eps * N / (2 * P * p2 * p)


def _compute_dEkin_dcos_phi(cos_phi, Ekin_i, m_i, m_t, m_e, m_r, xp=None):
    """``dEkin / d(cos_phi)``: the reciprocal of ``d(cos_phi)/dEkin``
    at ``Ekin(cos_phi)``."""
    if xp is None:
        xp = array_ns.get_backend('numpy')
    Ekin = _compute_Ekin_from_cos_phi(cos_phi, Ekin_i, m_i, m_t, m_e, m_r, xp=xp)
    return 1.0 / _compute_dcos_phi_dEkin(Ekin, Ekin_i, m_i, m_t, m_e, m_r, xp=xp)


# --- Deprecated cos_phi aliases -------------------------------------------
# Retained so external code (if any) that imported the pre-#185 names keeps
# working. New code should use the standard-mu API above.

compute_Ekin_from_cos_phi = _compute_Ekin_from_cos_phi
compute_cos_phi_from_Ekin = _compute_cos_phi_from_Ekin
compute_dEkin_dcos_phi = _compute_dEkin_dcos_phi
compute_dcos_phi_dEkin = _compute_dcos_phi_dEkin
