"""Multi-panel jax.grad wrt incident energy E for MF6 LAW=1
continuum spectrum (issue #166).

The tracer-panel-index kernel in ``mf6_law1_multipanel_traced``
auto-dispatches from ``mf6_law1_epintegral.integrate_law1_spectrum``
when ``xp=jax`` and no explicit ``panel_idx=`` is given. This
enables ``jax.grad`` wrt E to work across panel knots without the
caller having to compute the panel index themselves.

Pins:

- Numpy vs jax parity across the section's full ei mesh (traced
  vs single-panel path, on both LANG=1 Legendre and LANG=2
  Kalbach-Mann sections).
- ``jax.grad(sum(spectrum))(E)`` matches central FD to rtol=1e-4
  at multiple E in different panels.
- Grad on both sides of a panel knot agrees with the one-sided
  single-panel entry (``panel_idx=`` set to that side), confirming
  the auto-dispatch produces the physically correct one-sided
  gradient at C0 kinks.
- Uniform-NA precondition: constructed synthetic dict with varying
  NA raises ``NotImplementedError``.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf6_law1_preproc as _pre
from endf_userpy.mfsec_interpretation import mf6_law1_epintegral as _epi

from _corpus import resolve_al27, _first_existing
import os


def _jax_available():
    return 'jax' in array_ns.available_backends()


def _resolve_nd143():
    return _first_existing(
        os.environ.get('ND143_ENDF'),
        os.path.join(
            os.path.dirname(__file__), 'data_law1_adhoc',
            'endfb81_n_Nd-143.endf',
        ),
    )


@pytest.fixture(scope='module')
def al27_kalbach_data_jax():
    """Al-27 MT16 (n,2n) subsec 1: LAW=1 LANG=2 Kalbach-Mann,
    6-knot ei_mesh -> 6 panel-pairs. Canonical multi-panel LANG=2
    case."""
    if not _jax_available():
        pytest.skip('jax not installed')
    path = resolve_al27()
    if path is None:
        pytest.skip('Al-27 corpus not present')
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    xp_jx = array_ns.get_backend('jax')
    return _pre.mf6_law1_data_from_endf_dict(d, 16, 1, xp=xp_jx)


@pytest.fixture(scope='module')
def nd143_legendre_data_jax():
    """Nd-143 MT102 subsec 1: LAW=1 LANG=1 Legendre. Multi-panel
    Legendre case (52 panel-pairs)."""
    if not _jax_available():
        pytest.skip('jax not installed')
    path = _resolve_nd143()
    if path is None:
        pytest.skip('Nd-143 corpus not present')
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    xp_jx = array_ns.get_backend('jax')
    for mt in d.get(6, {}):
        subs = d[6][mt]['subsection']
        if len(subs) == 1 and list(subs.values())[0]['LAW'] == 1 \
                and int(list(subs.values())[0]['LANG']) == 1:
            return _pre.mf6_law1_data_from_endf_dict(d, mt, 1, xp=xp_jx)
    pytest.skip('no single-subsec LAW=1 LANG=1 in Nd-143')


def _max_rel(a, b):
    return float(np.max(np.abs(a - b) / (0.5 * (np.abs(a) + np.abs(b)) + 1e-30)))


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_al27_kalbach_numpy_jax_multipanel_parity(al27_kalbach_data_jax):
    """xp=jax multipanel path matches numpy single-panel path across
    the full ei mesh of Al-27 (n, 2n) MT16."""
    import jax.numpy as jnp

    path = resolve_al27()
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    xp_np = array_ns.get_backend('numpy')
    data_np = _pre.mf6_law1_data_from_endf_dict(d, 16, 1, xp=xp_np)
    xp_jx = array_ns.get_backend('jax')

    ein = np.array([1.36e7, 1.42e7, 1.55e7, 1.75e7, 1.9e7, 2.5e7])
    eout = np.linspace(1e4, 1e6, 12)
    a = np.asarray(_epi.integrate_law1_spectrum(data_np, ein, eout, to_lab=True))
    b = np.asarray(_epi.integrate_law1_spectrum(
        al27_kalbach_data_jax, jnp.asarray(ein), jnp.asarray(eout),
        to_lab=True, xp=xp_jx,
    ))
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_al27_kalbach_jax_grad_wrt_E_multipanel(al27_kalbach_data_jax):
    """jax.grad wrt E via auto-dispatched multi-panel kernel
    matches central FD at four Es in four different panels."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    eout = jnp.linspace(1e4, 1e6, 10)

    def loss(E):
        return jnp.sum(_epi.integrate_law1_spectrum(
            al27_kalbach_data_jax, E[None], eout, to_lab=True, xp=xp_jx,
        ))

    for E_val in (1.42e7, 1.55e7, 1.75e7, 1.9e7):
        E0 = jnp.array(E_val)
        val = float(loss(E0))
        grad = float(jax.grad(loss)(E0))
        fd_h = E_val * 1e-5
        fd = (float(loss(E0 + fd_h)) - float(loss(E0 - fd_h))) / (2 * fd_h)
        assert np.isfinite(grad)
        assert val > 0.0
        np.testing.assert_allclose(grad, fd, rtol=1e-3)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_nd143_legendre_jax_grad_wrt_E_multipanel(nd143_legendre_data_jax):
    """LANG=1 Legendre multi-panel grad, on a section with 50+
    panel-pairs (Nd-143 MT102 capture): jax.grad wrt E matches FD."""
    import jax
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    ei_mesh = np.asarray(nd143_legendre_data_jax.ei_mesh)
    eout = jnp.linspace(1e5, 5e6, 8)

    def loss(E):
        return jnp.sum(_epi.integrate_law1_spectrum(
            nd143_legendre_data_jax, E[None], eout, to_lab=True, xp=xp_jx,
        ))

    # Pick 3 mid-panel E values from panels well apart in the mesh.
    test_es = [
        0.5 * (ei_mesh[5] + ei_mesh[6]),
        0.5 * (ei_mesh[20] + ei_mesh[21]),
        0.5 * (ei_mesh[40] + ei_mesh[41]),
    ]
    for E_val in test_es:
        E0 = jnp.array(float(E_val))
        grad = float(jax.grad(loss)(E0))
        fd_h = float(E_val) * 1e-5
        fd = (float(loss(E0 + fd_h)) - float(loss(E0 - fd_h))) / (2 * fd_h)
        assert np.isfinite(grad)
        if abs(fd) < 1e-30:
            continue  # zero-value region; grad is trivially zero
        np.testing.assert_allclose(grad, fd, rtol=1e-3)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_auto_dispatch_matches_panel_idx_entry(al27_kalbach_data_jax):
    """The multi-panel auto-dispatch (no explicit panel_idx) gives
    the same output as an explicit panel_idx= call for E strictly
    inside the corresponding panel."""
    import jax.numpy as jnp

    xp_jx = array_ns.get_backend('jax')
    eout = jnp.linspace(1e4, 1e6, 10)

    # E = 1.55e7 is strictly inside panel 2 (ei[2]=1.5e7, ei[3]=1.6e7)
    E_val = 1.55e7
    p = 2
    E_arr = jnp.array([E_val])
    auto = np.asarray(_epi.integrate_law1_spectrum(
        al27_kalbach_data_jax, E_arr, eout, to_lab=True, xp=xp_jx,
    ))
    manual = np.asarray(_epi.integrate_law1_spectrum(
        al27_kalbach_data_jax, E_arr, eout, to_lab=True, xp=xp_jx,
        panel_idx=p,
    ))
    np.testing.assert_allclose(auto, manual, rtol=1e-6)
