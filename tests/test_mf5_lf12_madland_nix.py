"""MF5 LF=12 Madland-Nix fission spectrum.

Phase 2 base-physics coverage for LF=12 (previously raised
ValueError; #198 roadmap). Verifies the closed-form implementation
against:

- Support: ``f = 0`` for E' < 0; finite and non-negative for E' > 0.
- Normalisation: ``integral_0^inf f(E, E') dE' == 1`` to a few
  parts per thousand on a fine grid extending well past the
  spectrum tail.
- Symmetry-under-swap: the (EFL, EFH) contributions enter through
  ``0.5 * [S(E', EFL, T) + S(E', EFH, T)]`` so swapping EFL and
  EFH must give the same spectrum.
- Backend-agnostic (numpy vs jax) on concrete inputs.
- ``jax.grad`` wrt file-side EFL and T_M tracers (fast, via the
  custom_jvp exp1 / gammainc path landed for issue #207).
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


def _make_mn_contrib(efl, efh, tm, U=0.0):
    return {
        'p_table': _tab1('E', 'p', [1e-5, 3e7], [1.0, 1.0]),
        'tm_table': _tab1('E', 'TM', [1e-5, 3e7], [tm, tm]),
        'EFL': efl,
        'EFH': efh,
        'U': U,
        'LF': 12,
    }


def _make_endf_dict(contrib):
    return {5: {18: {'contribution': {1: contrib}}}}


@pytest.mark.parametrize('efl,efh,tm,E_in', [
    (1.03e6, 5.5e5, 1.0e6, 5.0e6),
    (1.03e6, 5.5e5, 1.2e6, 1.5e7),
    (9.0e5,  6.0e5, 8.0e5, 8.0e6),
])
def test_lf12_madland_nix_support_boundary(efl, efh, tm, E_in):
    """f(E, E'<0) is zero and f is finite and non-negative for
    E' > 0."""
    d = _make_endf_dict(_make_mn_contrib(efl, efh, tm))
    ein = np.array([E_in])
    eout = np.array([-1.0, 0.0, 1e4, 1e5, 1e6, 5e6, 2e7])
    f = np.asarray(mf5.compute_spectrum(d, 18, ein, eout))[0]
    assert f[0] == 0.0                            # E' < 0
    assert np.isfinite(f[1:]).all()
    assert (f[1:] >= 0.0).all()


@pytest.mark.parametrize('efl,efh,tm,E_in', [
    (1.03e6, 5.5e5, 1.0e6, 5.0e6),
    (1.03e6, 5.5e5, 1.2e6, 1.5e7),
    (9.0e5,  6.0e5, 8.0e5, 8.0e6),
])
def test_lf12_madland_nix_normalisation(efl, efh, tm, E_in):
    """Integral over [0, inf) of f dE' equals 1 to a few parts per
    thousand. Uses a fine grid extending to ~30 T_M which covers
    the physical spectrum tail comfortably."""
    d = _make_endf_dict(_make_mn_contrib(efl, efh, tm))
    ein = np.array([E_in])
    # 30 T_M is well past where the spectrum drops below 1e-8.
    eout = np.linspace(1e2, 30.0 * tm, 8000)
    f = np.asarray(mf5.compute_spectrum(d, 18, ein, eout))[0]
    integral = float(np.trapezoid(f, eout))
    assert abs(integral - 1.0) < 5e-3, (
        f'LF=12 spectrum not normalised at (efl={efl}, efh={efh}, '
        f'tm={tm}, E={E_in}): integral={integral:.6f}'
    )


def test_lf12_madland_nix_symmetric_under_efl_efh_swap():
    """chi(E, E') is symmetric under swap of EFL and EFH."""
    ein = np.array([5.0e6, 1.5e7])
    eout = np.linspace(1e4, 8.0e6, 300)
    d_ab = _make_endf_dict(_make_mn_contrib(
        efl=1.03e6, efh=5.5e5, tm=1.0e6,
    ))
    d_ba = _make_endf_dict(_make_mn_contrib(
        efl=5.5e5, efh=1.03e6, tm=1.0e6,
    ))
    f_ab = np.asarray(mf5.compute_spectrum(d_ab, 18, ein, eout))
    f_ba = np.asarray(mf5.compute_spectrum(d_ba, 18, ein, eout))
    np.testing.assert_allclose(f_ab, f_ba, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_lf12_madland_nix_numpy_jax_parity():
    """numpy vs jax parity at physics-level precision. Not to machine
    precision because the JAX exp1 path uses the A&S 5.1.53 / 5.1.56
    rational approximation (accuracy ~2e-7 relative) for
    jit-friendliness; scipy uses a full-precision implementation.
    The gap is well below any physical measurement uncertainty."""
    d = _make_endf_dict(_make_mn_contrib(
        efl=1.03e6, efh=5.5e5, tm=1.2e6,
    ))
    ein = np.array([5.0e6, 1.0e7, 1.5e7])
    eout = np.linspace(1e4, 8e6, 200)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    f_np = np.asarray(mf5.compute_spectrum(d, 18, ein, eout, xp=xp_np))
    f_jx = np.asarray(mf5.compute_spectrum(d, 18, ein, eout, xp=xp_jx))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-6, atol=1e-30)


def _fd5(f, x, h):
    return (-float(f(x + 2 * h)) + 8 * float(f(x + h))
            - 8 * float(f(x - h)) + float(f(x - 2 * h))) / (12 * h)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_lf12_madland_nix_jax_grad_wrt_efl_matches_fd():
    """``jax.grad`` wrt EFL matches 5-point central FD. Only fast
    after the #207 custom_jvp path lands for exp1 / gammainc;
    without it, a single grad call takes ~1 minute on this grid."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    efh = 5.5e5
    tm = 1.0e6
    ein = jnp.array([5.0e6, 1.0e7])
    eout = jnp.linspace(1e4, 6e6, 100)

    def loss(efl_val):
        c = _make_mn_contrib(efl=0.0, efh=efh, tm=tm)
        c['EFL'] = efl_val
        return jnp.sum(mf5.compute_spectrum(
            _make_endf_dict(c), 18, ein, eout, xp=xp_jx,
        ))

    efl0 = 1.03e6
    grad = float(jax.grad(loss)(jnp.array(efl0)))
    fd = _fd5(lambda v: loss(jnp.array(v)), efl0, efl0 * 1e-4)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=2e-3, atol=1e-20)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_lf12_madland_nix_jax_grad_wrt_tm_matches_fd():
    """``jax.grad`` wrt the tabulated T_M parameter matches FD.
    Fast after the #207 custom_jvp path."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    efl = 1.03e6
    efh = 5.5e5
    ein = jnp.array([5.0e6, 1.0e7])
    eout = jnp.linspace(1e4, 6e6, 100)

    def loss(tm_val):
        c = _make_mn_contrib(efl=efl, efh=efh, tm=0.0)
        c['tm_table']['TM'] = [tm_val, tm_val]
        return jnp.sum(mf5.compute_spectrum(
            _make_endf_dict(c), 18, ein, eout, xp=xp_jx,
        ))

    tm0 = 1.0e6
    grad = float(jax.grad(loss)(jnp.array(tm0)))
    fd = _fd5(lambda v: loss(jnp.array(v)), tm0, tm0 * 1e-4)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=2e-3, atol=1e-20)


def test_lf12_madland_nix_previously_not_implemented_error_is_gone():
    """Regression pin: LF=12 no longer raises ValueError."""
    d = _make_endf_dict(_make_mn_contrib(
        efl=1.03e6, efh=5.5e5, tm=1.0e6,
    ))
    ein = np.array([5.0e6])
    eout = np.array([1e5, 1e6, 3e6])
    f = np.asarray(mf5.compute_spectrum(d, 18, ein, eout))
    assert f.shape == (1, 3)
    assert np.isfinite(f).all()
