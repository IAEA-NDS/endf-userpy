"""MF5 LF=5 general-evaporation autodiff wrt shape leaves.

Companion to ``test_mf5_backend_agnostic.py`` (which covers LF=7
and LF=9 analytic spectra). The LF=5 reconstruction is fully
vectorised over the incident-energy axis, so ``jax.grad`` composes
with ``jax.jit`` and grads flow through:

- ``contrib['theta_table']['theta']`` -- evaporation temperature
- ``contrib['U']`` -- kinematic offset
- ``contrib['g_table']['g'][idx]`` -- shape's per-knot values

The primary tests use a synthetic **LIN-LIN** g_table so the
spectrum is smooth wrt theta / U and finite-difference is a valid
reference. A separate corpus regression pins numerical parity with
the pre-vectorised implementation on an INT=1 (histogram) file,
where FD wrt theta is not a valid reference (histogram
interpolation is knot-discontinuous by construction) but the
reconstruction values themselves must be bit-identical.
"""
from __future__ import annotations

import copy
import os
import warnings

import numpy as np
import pytest

from endf_userpy.mfsec_interpretation import mf5_interpretation as mf5
from endf_userpy.primitives import array_ns


def _jax_available():
    return 'jax' in array_ns.available_backends()


pytestmark = pytest.mark.skipif(
    not _jax_available(), reason='jax not installed',
)


def _tab1(x_name, y_name, x, y, int_=2):
    return {
        x_name: list(x), y_name: list(y),
        'INT': [int_], 'NBT': [len(x)],
    }


def _make_lf5_contrib(theta, U, g_x, g_y):
    return {
        'p_table': _tab1('E', 'p', [1e-5, 3e7], [1.0, 1.0]),
        'theta_table': _tab1('E', 'theta', [1e-5, 3e7], [theta, theta]),
        'U': U,
        'LF': 5,
        'g_table': _tab1('x', 'g', g_x, g_y, int_=2),
    }


def _make_endf_dict(contrib):
    return {5: {18: {'contribution': {1: contrib}}}}


def _default_g_lin():
    # smooth triangular ramp on [0, 1] with peak at 0.4
    x = np.linspace(0.0, 1.0, 21).tolist()
    y = [1.0 - abs(v - 0.4) * 2.0 for v in x]
    y = [max(v, 0.05) for v in y]
    return x, y


def test_numpy_default_matches_xp_numpy_and_jax_synthetic():
    """xp=None default, xp=numpy explicit, xp=jax all agree on a
    synthetic LIN-LIN LF=5 section."""
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    g_x, g_y = _default_g_lin()
    d = _make_endf_dict(_make_lf5_contrib(1.2e6, 1.0e5, g_x, g_y))
    ein = np.array([2.0e6, 5.0e6, 1.0e7])
    eout = np.linspace(1e3, 5.0e6, 30)

    default = np.asarray(mf5.compute_spectrum(d, 18, ein, eout))
    with_np = np.asarray(
        mf5.compute_spectrum(d, 18, ein, eout, xp=xp_np),
    )
    with_jx = np.asarray(
        mf5.compute_spectrum(d, 18, ein, eout, xp=xp_jx),
    )
    np.testing.assert_array_equal(default, with_np)
    np.testing.assert_allclose(default, with_jx, rtol=1e-10, atol=1e-14)


def test_lf5_grad_wrt_theta_matches_fd():
    """``jax.grad`` wrt evaporation temperature matches central FD
    on a synthetic LIN-LIN section (histogram INT files are
    knot-discontinuous by construction and are covered by the
    corpus regression below)."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    g_x, g_y = _default_g_lin()
    ein = jnp.array([2.0e6, 5.0e6])
    eout = jnp.linspace(1e3, 4.0e6, 30)

    def loss(theta_val):
        c = _make_lf5_contrib(1.2e6, 1.0e5, g_x, g_y)
        c['theta_table']['theta'] = [theta_val, theta_val]
        d = _make_endf_dict(c)
        return jnp.sum(mf5.compute_spectrum(d, 18, ein, eout, xp=xp_jx))

    theta0 = 1.2e6
    val = float(loss(jnp.array(theta0)))
    grad = float(jax.grad(loss)(jnp.array(theta0)))
    eps = 1e3
    fd = (
        float(loss(jnp.array(theta0 + eps)))
        - float(loss(jnp.array(theta0 - eps)))
    ) / (2 * eps)
    assert val > 0.0
    assert np.isfinite(grad)
    assert abs(fd) > 0.0, 'FD is exactly zero; test parameters give no signal'
    np.testing.assert_allclose(grad, fd, rtol=1e-3, atol=1e-20)


def test_lf5_grad_wrt_U_matches_fd():
    """``jax.grad`` wrt the kinematic offset ``U`` matches central FD."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    g_x, g_y = _default_g_lin()
    ein = jnp.array([2.0e6, 5.0e6])
    eout = jnp.linspace(1e3, 4.0e6, 30)

    def loss(U_val):
        c = _make_lf5_contrib(1.2e6, U_val, g_x, g_y)
        d = _make_endf_dict(c)
        return jnp.sum(mf5.compute_spectrum(d, 18, ein, eout, xp=xp_jx))

    U0 = 1.0e5
    val = float(loss(jnp.array(U0)))
    grad = float(jax.grad(loss)(jnp.array(U0)))
    eps = 1e3
    fd = (
        float(loss(jnp.array(U0 + eps)))
        - float(loss(jnp.array(U0 - eps)))
    ) / (2 * eps)
    assert val > 0.0
    assert np.isfinite(grad)
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-3, atol=1e-20)


def test_lf5_grad_wrt_g_leaf_matches_fd():
    """``jax.grad`` wrt an individual ``g_table['g'][idx]`` value
    matches central FD."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    g_x, g_y = _default_g_lin()
    ein = jnp.array([2.0e6, 5.0e6])
    eout = jnp.linspace(1e3, 4.0e6, 30)
    idx = 5  # in the interior of the mesh

    def loss(g_val):
        gy = list(g_y)
        gy[idx] = g_val
        c = _make_lf5_contrib(1.2e6, 1.0e5, g_x, gy)
        d = _make_endf_dict(c)
        return jnp.sum(mf5.compute_spectrum(d, 18, ein, eout, xp=xp_jx))

    g0 = float(g_y[idx])
    val = float(loss(jnp.array(g0)))
    grad = float(jax.grad(loss)(jnp.array(g0)))
    eps = abs(g0) * 1e-4 or 1e-6
    fd = (
        float(loss(jnp.array(g0 + eps)))
        - float(loss(jnp.array(g0 - eps)))
    ) / (2 * eps)
    assert val > 0.0
    assert np.isfinite(grad)
    assert abs(fd) > 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-3, atol=1e-20)


def test_lf5_jit_composability_theta():
    """Both ``jax.grad(jax.jit(loss))`` and ``jax.jit(jax.grad(loss))``
    match eager ``jax.grad`` for LF=5 wrt theta."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    g_x, g_y = _default_g_lin()
    ein = jnp.array([2.0e6, 5.0e6])
    eout = jnp.linspace(1e3, 4.0e6, 30)

    def loss(theta_val):
        c = _make_lf5_contrib(1.2e6, 1.0e5, g_x, g_y)
        c['theta_table']['theta'] = [theta_val, theta_val]
        d = _make_endf_dict(c)
        return jnp.sum(mf5.compute_spectrum(d, 18, ein, eout, xp=xp_jx))

    theta0 = jnp.array(1.2e6)
    g_eager = float(jax.grad(loss)(theta0))
    g_grad_of_jit = float(jax.grad(jax.jit(loss))(theta0))
    jit_grad = jax.jit(jax.grad(loss))
    _ = jit_grad(theta0)  # warmup
    g_jit_of_grad = float(jit_grad(theta0))
    for label, val in (
        ('grad(jit(loss))', g_grad_of_jit),
        ('jit(grad(loss))', g_jit_of_grad),
    ):
        assert np.isfinite(val)
        np.testing.assert_allclose(
            val, g_eager, rtol=1e-6, atol=1e-20,
            err_msg=f'{label} vs eager grad',
        )


def test_lf5_corpus_numpy_jax_parity():
    """Bit-parity between the numpy and jax backends on the U-235
    MT=455 LF=5 corpus file. This is a purely numerical parity
    check (INT=1 histogram, so FD-based autodiff tests do not
    apply). Skips cleanly if the corpus is absent.
    """
    path = 'tests/data_law1_adhoc/tendl21_n_U-235.endf'
    if not os.path.exists(path):
        pytest.skip('U-235 corpus not available')
    from endf_parserpy import EndfParserCpp
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    contribs = d[5][455]['contribution']
    lf5_idx = next(k for k, c in contribs.items() if c.get('LF') == 5)
    c = contribs[lf5_idx]
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1e6, 5e6, 1.5e7])
    eout = np.linspace(1e3, 5e5, 40)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f_np = np.asarray(
            mf5.compute_general_evaporation_spectrum(c, ein, eout, xp=xp_np),
        )
        f_jx = np.asarray(
            mf5.compute_general_evaporation_spectrum(c, ein, eout, xp=xp_jx),
        )
    np.testing.assert_allclose(f_np, f_jx, rtol=1e-10, atol=1e-30)


def test_lf5_corpus_grad_does_not_crash():
    """On the histogram-INT U-235 corpus file, ``jax.grad`` wrt g
    leaf runs end-to-end without raising. Value is unchecked (INT=1
    is knot-discontinuous, so FD is not a smooth reference) but the
    tracer must reach the leaf and produce a finite scalar.
    """
    path = 'tests/data_law1_adhoc/tendl21_n_U-235.endf'
    if not os.path.exists(path):
        pytest.skip('U-235 corpus not available')
    import jax
    import jax.numpy as jnp
    from endf_parserpy import EndfParserCpp

    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    contribs = d[5][455]['contribution']
    lf5_idx = next(k for k, c in contribs.items() if c.get('LF') == 5)
    xp_jx = array_ns.get_backend('jax')
    ein = np.array([1e6])
    eout = np.linspace(1e3, 1e5, 20)
    g_orig = float(contribs[lf5_idx]['g_table']['g'][2])

    def loss(g_val):
        d_t = copy.deepcopy(d)
        d_t[5][455]['contribution'][lf5_idx]['g_table']['g'][2] = g_val
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = mf5.compute_general_evaporation_spectrum(
                d_t[5][455]['contribution'][lf5_idx], ein, eout, xp=xp_jx,
            )
        return jnp.sum(f)

    grad = float(jax.grad(loss)(jnp.array(g_orig)))
    assert np.isfinite(grad)
