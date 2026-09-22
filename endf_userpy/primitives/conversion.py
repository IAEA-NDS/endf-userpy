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
    return mu_cm


def convert_angdist_to_labsys(mu_cm, f_cm, r2, xp=None):
    """Multiply the CM-frame angular distribution by the CM -> LAB
    Jacobian ``xw^(3/2) / (r^2 (r + mu_cm))`` with ``xw = 1 + 2 r
    mu_cm + r^2``.
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
    f_lab = f_cm * xw * xp.sqrt(xw) / (r2 * (r + mu_cm))
    return f_lab
