"""Phase-1 autodiff-coverage pins: ``jax.grad`` wrt query
incident energy E and outgoing energy E' through the analytic
MF5 spectra (LF=7 Maxwellian, LF=9 evaporation).

Companion to ``test_mf5_backend_agnostic.py`` (which grads wrt
file-side theta and U). This file grads wrt the two query-side
axes exposed by ``mf5.compute_spectrum(endf_dict, mt,
energies_in, energies_out, xp=xp)``.

Both closed-form spectra are pure analytic functions of (E, E',
theta, U), so under ``xp=jax`` the tracer flows through
``compute_theta`` (interp on the theta_table) and the analytic
normalisation without any special panel machinery. Undefined at
E = U (spectrum support boundary) but well-behaved elsewhere.

LF=5 general evaporation is NOT covered here: it interps a
tabulated g-function via ``interp_tab1`` and the tabulated path
has a known tracer-materialisation gap (see the note in
``test_mf5_backend_agnostic.py``). Tracked as a Phase 3 item.

LF=11 (energy-dependent Watt) and LF=12 (Madland-Nix) raise
``NotImplementedError`` today (Phase 2 base-physics items).
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.mfsec_interpretation import mf5_interpretation as mf5
from endf_userpy.primitives import array_ns


def _jax_available():
    return 'jax' in array_ns.available_backends()


def _tab1(x_name, y_name, x, y):
    return {x_name: list(x), y_name: list(y), 'INT': [2], 'NBT': [len(x)]}


def _make_contrib(lf, theta, U):
    return {
        'p_table': _tab1('E', 'p', [1e-5, 3e7], [1.0, 1.0]),
        'theta_table': _tab1('E', 'theta', [1e-5, 3e7], [theta, theta]),
        'U': U,
        'LF': lf,
    }


def _make_endf_dict(contrib):
    return {5: {18: {'contribution': {1: contrib}}}}


def _fd5(f, x, h):
    """5-point central FD stencil, O(h^4) error. Kept tight for
    analytic spectra where the standard 3-point stencil sometimes
    picks up curvature near the support boundary (E'=0, E'=E-U)."""
    return (-float(f(x + 2 * h)) + 8 * float(f(x + h))
            - 8 * float(f(x - h)) + float(f(x - 2 * h))) / (12 * h)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('lf,theta,U', [
    (7, 1.2e6, 1.0e5),
    (7, 5.0e5, 0.0),
    (9, 1.0e6, 2.0e5),
    (9, 8.0e5, 0.0),
])
def test_grad_wrt_E_matches_fd(lf, theta, U):
    """``jax.grad(sum(f_spec(E, E')))`` wrt tracer E matches FD.

    Fixed E' grid (well within support so no boundary effects);
    tracer is the incident energy E. Multiple (theta, U) settings
    exercise both zero-U and non-zero-U branches."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    d = _make_endf_dict(_make_contrib(lf, theta, U))
    eout = np.linspace(5e4, 2.5e6, 40, dtype=np.float64)

    def loss(E_scalar):
        E_arr = jnp.array([E_scalar])
        return jnp.sum(mf5.compute_spectrum(
            d, 18, E_arr, jnp.asarray(eout), xp=xp_jx,
        ))

    for E_val in (2.0e6, 5.0e6, 1.0e7):
        grad = float(jax.grad(loss)(jnp.array(E_val)))
        fd = _fd5(loss, jnp.array(E_val), E_val * 1e-4)
        assert np.isfinite(grad), f'LF={lf} grad not finite @ E={E_val}'
        if abs(fd) < 1e-30:
            continue
        np.testing.assert_allclose(
            grad, fd, rtol=1e-3, atol=1e-30,
            err_msg=f'LF={lf} theta={theta} U={U} E={E_val}: '
                    f'ad={grad:.4e} fd={fd:.4e}',
        )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('lf,theta,U', [
    (7, 1.2e6, 1.0e5),
    (9, 1.0e6, 2.0e5),
])
def test_grad_wrt_Ep_matches_fd(lf, theta, U):
    """``jax.grad(f_spec(E, E'))`` wrt tracer E' matches FD at
    Es well inside the emission support ``[0, E - U]``.

    E' is a query-side leaf: the analytic form differentiates
    smoothly wrt it wherever the support mask is True."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    d = _make_endf_dict(_make_contrib(lf, theta, U))
    E_in = 8.0e6                    # keeps E - U comfortably large

    def loss(Ep_scalar):
        return mf5.compute_spectrum(
            d, 18, jnp.array([E_in]), jnp.array([Ep_scalar]),
            xp=xp_jx,
        )[0, 0]

    for Ep_val in (1.0e5, 5.0e5, 2.0e6, 5.0e6):
        grad = float(jax.grad(loss)(jnp.array(Ep_val)))
        fd = _fd5(loss, jnp.array(Ep_val), Ep_val * 1e-4)
        assert np.isfinite(grad)
        if abs(fd) < 1e-30:
            continue
        np.testing.assert_allclose(
            grad, fd, rtol=1e-3, atol=1e-30,
            err_msg=f'LF={lf} theta={theta} U={U} Ep={Ep_val}: '
                    f'ad={grad:.4e} fd={fd:.4e}',
        )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_normalisation_holds_along_grad_path_lf7():
    """The Maxwellian spectrum integrates to 1 at every E in
    support. This is not a grad test per se; it pins that the
    analytic implementation preserves normalisation under xp=jax
    (a common way autodiff regressions manifest is a slightly
    non-normalised spectrum whose derivatives look reasonable but
    whose absolute values drift)."""
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    U = 1.0e5
    d = _make_endf_dict(_make_contrib(7, theta=1.2e6, U=U))
    for E_in in (2.0e6, 5.0e6, 1.0e7):
        # Cover the full support [0, E - U] with a fine grid so
        # the trapezoid error stays sub-permille on the Maxwellian
        # tail.
        eout = jnp.linspace(1e2, E_in - U, 4000)
        f = np.asarray(mf5.compute_spectrum(
            d, 18, jnp.array([E_in]), eout, xp=xp_jx,
        ))[0]
        integral = float(np.trapezoid(f, np.asarray(eout)))
        assert abs(integral - 1.0) < 1e-3, (
            f'LF=7 spectrum not normalised at E={E_in}: '
            f'integral={integral:.6f}'
        )
