"""LAB <-> CM angular kinematics for two-body reactions.

Backend-agnostic: every function accepts an optional ``xp`` (the
adapter returned by :func:`array_ns.get_backend`). Passing
``xp=None`` (the default) resolves to the numpy backend for
back-compat with existing callers that predate the array-namespace
adapter.

Physics: non-relativistic two-body kinematics per ENDF-6 manual
sec. 6.2 / MF4 discussion. The full relativistic replacement lives
in :mod:`conversion_relativistic`.
"""
from . import array_ns


def _resolve_xp(xp):
    return xp if xp is not None else array_ns.get_backend('numpy')


def _correct_r2(r2, xp):
    """Clip ``r2`` below to a small positive floor so downstream
    ``sqrt(r2)`` and ``1/r`` do not blow up at zero mass ratio.

    Functional (returns a new array): the previous numpy
    implementation mutated ``r2`` in place, which is illegal on
    JAX-tracer inputs. Numerical result is bit-identical: the
    floor ``1e-76`` sits far below any physical r^2 value.
    """
    r2min = 1e-76
    return xp.where(r2 < r2min, r2min, r2)


def compute_r2(E_lab, awi, awr, awp, q, xp=None):
    """Kinematic ``r^2`` factor for the LAB <-> CM angular
    transformation.

    ``r^2 = (awr (awr+awi-awp) / (awi awp)) * (1 + ((awr+awi)/awr)
    * q / E_lab)``. Vector over ``E_lab``.
    """
    xp = _resolve_xp(xp)
    E_lab = xp.asarray(E_lab)
    return (awr * (awr + awi - awp) / (awi * awp)
            * (1.0 + (awr + awi) / awr * q / E_lab))


def convert_angcos_to_cmsys(mu_lab, r2, xp=None):
    """Map a LAB-frame ``cos(theta)`` to CM-frame ``cos(theta)``
    given ``r2 = compute_r2(...)``.

    Kinematically forbidden LAB angles (``z = mu_lab^2 + r^2 - 1
    < 0`` for r < 1, i.e. the equal-mass / light-ejectile
    back-scatter region) produce NaN -- the caller is expected to
    clip NaN to 0 in the LAB return since no scattering into
    unreachable angles is physically possible.

    Returns the "primary" CM branch. For ``r < 1`` the LAB->CM
    mapping is 2-to-1 in the forward cone (see
    :func:`convert_angcos_to_cmsys_second_branch` and issue #210
    for the physics). Callers who need the full LAB density on a
    heavy recoil should also call the second-branch variant and
    sum the two Jacobian contributions.

    Returns an ``(len(r2), len(mu_lab))`` array.
    """
    xp = _resolve_xp(xp)
    mu_lab = xp.asarray(mu_lab).reshape(1, -1)
    r2 = xp.asarray(r2).reshape(-1, 1)
    r2 = _correct_r2(r2, xp)
    r = xp.sqrt(r2)
    u = mu_lab
    u2 = u * u
    z = u2 + r2 - 1.0
    z1 = (1.0 - u2 - r2 * u2)
    # Numpy raises a harmless ``invalid value in sqrt`` warning on
    # the forbidden region; JAX does not. Guard z with ``where`` so
    # both backends produce NaN via the algebra (the ``where(z<0,
    # nan, sqrt(z))`` pattern is what leaves a clean NaN on JAX
    # too).
    z_safe = xp.where(z < 0.0, xp.nan, z)
    z2 = r * (u2 - 1.0 - u * xp.sqrt(z_safe))
    mu_cm = z1 / z2
    # For r^2 < 1 (heavy scattered particle) the physical LAB
    # angular range is the forward cone mu_LAB > 0 only. Negative
    # mu_LAB back-transforms to positive mu_LAB in either branch
    # (unphysical); mask those to NaN so the caller's clip
    # suppresses them. r^2 >= 1 covers the full [-1, 1] LAB range.
    unphysical = (r2 < 1.0) & (u < 0.0)
    return xp.where(unphysical, xp.nan, mu_cm)


def convert_angcos_to_cmsys_second_branch(mu_lab, r2, xp=None):
    """Second CM cosine root of the two-body LAB->CM mapping
    (issue #210).

    The LAB<->CM mapping is a quadratic in ``mu_CM``:

        r^2 mu_CM^2 + 2 r (1 - mu_LAB^2) mu_CM
        + (1 - mu_LAB^2 - r^2 mu_LAB^2) = 0

    For ``r < 1`` (heavy scattered particle in CM, e.g. the recoil
    in a light-ejectile reaction), both roots are physical in the
    forward LAB cone: two distinct ``mu_CM`` values map to the
    same ``mu_LAB``, and the LAB angular density is the sum over
    both branches' Jacobian contributions. ``convert_angcos_to_
    cmsys`` returns the "primary" root (sign-choice matching
    the historical Fortran endf6.f90); this function returns the
    "secondary" root (sign-flipped sqrt in the denominator).

    For ``r >= 1`` (light scattered particle), the secondary root
    is spurious -- its back-transform gives ``-mu_LAB``, not
    ``+mu_LAB`` -- so this function returns NaN so the caller's
    ``xp.where(isnan, 0, ...)`` clip suppresses it.

    Same shape convention as :func:`convert_angcos_to_cmsys`:
    ``(len(r2), len(mu_lab))``.
    """
    xp = _resolve_xp(xp)
    mu_lab_arr = xp.asarray(mu_lab).reshape(1, -1)
    r2_arr = xp.asarray(r2).reshape(-1, 1)
    r2_arr = _correct_r2(r2_arr, xp)
    r = xp.sqrt(r2_arr)
    u = mu_lab_arr
    u2 = u * u
    z = u2 + r2_arr - 1.0
    z1 = (1.0 - u2 - r2_arr * u2)
    z_safe = xp.where(z < 0.0, xp.nan, z)
    # Sign of sqrt flipped: the secondary root.
    z2 = r * (u2 - 1.0 + u * xp.sqrt(z_safe))
    mu_cm = z1 / z2
    # Suppress the spurious "second branch" for r >= 1, and the
    # unphysical backward LAB angles when r^2 < 1.
    unphysical = (r2_arr >= 1.0) | (u < 0.0)
    return xp.where(unphysical, xp.nan, mu_cm)


def convert_angdist_to_labsys(mu_cm, f_cm, r2, xp=None):
    """Multiply the CM-frame angular distribution by the magnitude
    of the CM -> LAB Jacobian ``xw^(3/2) / (r^2 |r + mu_cm|)`` with
    ``xw = 1 + 2 r mu_cm + r^2``.

    ``|r + mu_cm|`` (not the unsigned ``r + mu_cm``) is the correct
    Jacobian magnitude: for ``r < 1`` and ``mu_cm < -r`` (which is
    normal on the second CM branch of a heavy recoil, see
    :func:`convert_angcos_to_cmsys_second_branch` and issue #210)
    the unsigned form flips sign and yields a spurious negative
    density.
    """
    xp = _resolve_xp(xp)
    mu_cm = xp.asarray(mu_cm)
    f_cm = xp.asarray(f_cm)
    if mu_cm.ndim == 1:
        mu_cm = mu_cm.reshape(1, -1)
        f_cm = f_cm.reshape(1, -1)
    r2 = xp.asarray(r2).reshape(-1, 1)
    r2 = _correct_r2(r2, xp)
    r = xp.sqrt(r2)
    xw = 1.0 + 2.0 * r * mu_cm + r2
    f_lab = f_cm * xw * xp.sqrt(xw) / (r2 * xp.abs(r + mu_cm))
    return f_lab
