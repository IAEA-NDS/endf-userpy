"""Phase-1 autodiff-coverage pins for MF4 angular distributions.

Covers all three LTT variants (Legendre, tabulated, mixed) with
concrete-input parity plus regression pins for the query-side
tracer gap they inherit from the shared interp primitives.

Current state (as of the #201 Legendre-path fix):

- numpy vs jax parity on concrete inputs works for LTT=1, 2, 3.
- ``jax.grad`` wrt query-side E or mu on LTT=1 (Legendre): works
  (unblocked by the #201 Legendre-path fix). Positive assertions
  below.
- ``jax.grad`` wrt query-side E or mu on LTT=2 (tabulated) /
  LTT=3 (mixed): still blocked on the ``interp_tab2`` traced-x
  fast path (#201 PR-B). Regression pins below still raise.

Files used:

- LTT=1 (Legendre): ``tests/data/n-004_Be_009.endf`` MT=2
  (committed, small).
- LTT=2 (tabulated): ``tests/data_law1_adhoc/endfb81_n_Al-27.endf``
  MT=2 (skips cleanly when corpus not fetched).
- LTT=3 (mixed):    ``tests/data/n-001_H_002.endf`` MT=2
  (committed, small).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf4_interpretation as mf4
from endf_userpy.primitives import array_ns

from _corpus import resolve_al27


DATA_DIR = Path(__file__).parent / 'data'


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def be9_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-004_Be_009.endf'),
    )


@pytest.fixture(scope='module')
def h2_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-001_H_002.endf'),
    )


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip('Al-27 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def _tracer_exc():
    try:
        import jax.errors
        return jax.errors.TracerArrayConversionError
    except AttributeError:
        return Exception


# -------------------------------------------------------------------
# LTT=1 (Legendre) on Be-9 MT=2 elastic
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf4_ltt1_numpy_jax_parity(be9_endf_dict):
    """xp=jax and xp=numpy return the same values on Be-9 MT=2
    LTT=1 Legendre angular distribution across representative Es
    and mus. Pins the working concrete-input path."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1e5, 1e6, 5e6, 1.5e7], dtype=np.float64)
    mu = np.linspace(-0.9, 0.9, 21, dtype=np.float64)

    f_np = np.asarray(mf4.compute_angdist_values(
        be9_endf_dict, 2, ein, mu, xp=xp_np,
    ))
    f_jx = np.asarray(mf4.compute_angdist_values(
        be9_endf_dict, 2, jnp.asarray(ein), jnp.asarray(mu),
        xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-10, atol=1e-30)


def _fd5(f, x, h):
    return (-float(f(x + 2 * h)) + 8 * float(f(x + h))
            - 8 * float(f(x - h)) + float(f(x - 2 * h))) / (12 * h)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('E_val', [1e6, 5e6, 1.5e7])
def test_mf4_ltt1_grad_wrt_E_matches_fd(be9_endf_dict, E_val):
    """``jax.grad`` wrt E on MF4 LTT=1 (Legendre) matches central
    FD (unblocked by the #201 Legendre-path fix)."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    mu = jnp.linspace(-0.9, 0.9, 21)

    def loss(E_scalar):
        return jnp.sum(mf4.compute_angdist_values(
            be9_endf_dict, 2, jnp.array([E_scalar]), mu, xp=xp_jx,
        ))

    grad = float(jax.grad(loss)(jnp.array(E_val)))
    fd = _fd5(lambda v: loss(jnp.array(v)), E_val, E_val * 1e-4)
    assert np.isfinite(grad)
    # atol=1e-6 covers near-stationary Es where 5-point FD noise
    # dominates the true grad.
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
@pytest.mark.parametrize('mu_val', [-0.5, 0.0, 0.3, 0.7])
def test_mf4_ltt1_grad_wrt_mu_matches_fd(be9_endf_dict, mu_val):
    """``jax.grad`` wrt mu on MF4 LTT=1 (Legendre) matches central
    FD (unblocked by the #201 Legendre-path fix)."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([1e6, 5e6, 1.5e7])

    def loss(mu_scalar):
        return jnp.sum(mf4.compute_angdist_values(
            be9_endf_dict, 2, ein, jnp.array([mu_scalar]), xp=xp_jx,
        ))

    grad = float(jax.grad(loss)(jnp.array(mu_val)))
    fd = _fd5(lambda v: loss(jnp.array(v)), mu_val, 1e-4)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


# -------------------------------------------------------------------
# LTT=2 (tabulated) on Al-27 MT=2 elastic
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf4_ltt2_numpy_jax_parity(al27_endf_dict):
    """xp=jax and xp=numpy return the same values on Al-27 MT=2
    LTT=2 tabulated angular distribution. Uses interp_tab2 under
    the hood."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1e5, 1e6, 5e6, 1.5e7], dtype=np.float64)
    mu = np.linspace(-0.9, 0.9, 21, dtype=np.float64)

    f_np = np.asarray(mf4.compute_angdist_values(
        al27_endf_dict, 2, ein, mu, xp=xp_np,
    ))
    f_jx = np.asarray(mf4.compute_angdist_values(
        al27_endf_dict, 2, jnp.asarray(ein), jnp.asarray(mu),
        xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-10, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf4_ltt2_grad_wrt_E_currently_raises(al27_endf_dict):
    """Regression pin for the query-side E-tracer gap on MF4 LTT=2.
    Same root cause as issue #201 via interp_tab2."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    mu = jnp.linspace(-0.9, 0.9, 21)

    def loss(E_scalar):
        return jnp.sum(mf4.compute_angdist_values(
            al27_endf_dict, 2, jnp.array([E_scalar]), mu, xp=xp_jx,
        ))

    with pytest.raises(_tracer_exc()):
        jax.grad(loss)(jnp.array(5.0e6))


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf4_ltt2_grad_wrt_mu_currently_raises(al27_endf_dict):
    """Regression pin for the query-side mu-tracer gap on MF4 LTT=2."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([1e6, 5e6, 1.5e7])

    def loss(mu_scalar):
        return jnp.sum(mf4.compute_angdist_values(
            al27_endf_dict, 2, ein, jnp.array([mu_scalar]), xp=xp_jx,
        ))

    with pytest.raises(_tracer_exc()):
        jax.grad(loss)(jnp.array(0.3))


# -------------------------------------------------------------------
# LTT=3 (mixed) on H-2 MT=2 elastic
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf4_ltt3_numpy_jax_parity(h2_endf_dict):
    """xp=jax and xp=numpy agree on H-2 MT=2 LTT=3 mixed
    representation (Legendre below the split energy, tabulated
    above). Uses both primitives underneath."""
    import jax.numpy as jnp

    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    # H-2 elastic covers 1e-5 .. 2e7. Pick Es spanning the LTT=3
    # Legendre/tabulated split.
    ein = np.array([1e3, 1e5, 1e6, 1e7], dtype=np.float64)
    mu = np.linspace(-0.9, 0.9, 21, dtype=np.float64)

    f_np = np.asarray(mf4.compute_angdist_values(
        h2_endf_dict, 2, ein, mu, xp=xp_np,
    ))
    f_jx = np.asarray(mf4.compute_angdist_values(
        h2_endf_dict, 2, jnp.asarray(ein), jnp.asarray(mu),
        xp=xp_jx,
    ))
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-10, atol=1e-30)
