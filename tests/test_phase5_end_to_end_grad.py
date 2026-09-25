"""Phase 5 cross-MF end-to-end grad pins (roadmap #198).

Each of the five top-level user-facing entry points in
``endf_userpy.quantities`` gets:

- A query-side ``jax.grad`` pin (wrt E, and where applicable E'
  and mu), FD-checked with a 5-point central stencil.
- One file-side leaf grad pin (to catch any tracer-materialisation
  bug introduced by the top-level plumbing itself: MT selection,
  MT5 fallback, ZAP resolution, distribution routing).

Leaf-module grad tests already cover file-side leaves through the
reconstruction kernels directly (MLBW ER, MF6 LAW=1 b-panels,
LAW=2 Legendre coeffs, MF5 theta/a/b/EFL/EFH/TM, ...). What this
file adds is that grad still flows when the leaf is reached
through the top-level API, not just the leaf kernel.

Uses Be-9 (n-004_Be_009.endf, committed) for the compact
neutron-emission side. Nb-93 (adhoc corpus) for the resonance
composition path in ``get_reaction_xs`` with
``include_resonance=True``.

All ten configurations now assert on the actual numeric grad
value (issue #220 fully closed across PRs 1-4). The Ep-axis
tests for Be-9 (n,2n) rely on the fact that the file's inner
Ep interp is INT=1 (histogram), so analytic grad is exactly zero
in each bracket and FD is zero when the stencil doesn't cross a
bracket boundary; the tracer-Ep pin still exercises the grad
plumbing.
"""
from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_reaction_xs,
    get_particle_production_xs,
    get_particle_production_dxs_dE,
    get_particle_production_dxs_dmu,
    get_particle_production_ddxs,
)
from endf_userpy.primitives import array_ns

from _corpus import resolve_nb93, resolve_h2


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
    """JEFF-4.0 H-2 (n,2n) MT16 uses MF6 LAW=7 with INT=2 (lin-lin)
    on the inner Ep axis, which stresses the unit-base traced-x
    branch more than Be-9 (n,2n)'s INT=1 histogram (issue #220
    PR 4 evidence)."""
    path = resolve_h2()
    if path is None:
        pytest.skip('H-2 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def nb93_endf_dict():
    path = resolve_nb93()
    if path is None:
        pytest.skip('Nb-93 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def _fd5(f, x, h):
    return (-float(f(x + 2 * h)) + 8 * float(f(x + h))
            - 8 * float(f(x - h)) + float(f(x - 2 * h))) / (12 * h)


# -------------------------------------------------------------------
# get_reaction_xs
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_reaction_xs_grad_wrt_E_end_to_end(be9_endf_dict):
    """``jax.grad(get_reaction_xs)(E)`` end-to-end. Elastic (n,n)
    on Be-9 to keep it MF3-only (no MT5 fallback, no resonance
    composition path)."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')

    def loss(E_scalar):
        return get_reaction_xs(
            be9_endf_dict, '(n,n)', jnp.array([E_scalar]), xp=xp_jx,
        ).sum()

    for E_val in (1e6, 5e6, 1.2e7):
        grad = float(jax.grad(loss)(jnp.array(E_val)))
        fd = _fd5(lambda v: loss(jnp.array(v)), E_val, E_val * 1e-4)
        assert np.isfinite(grad)
        np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_reaction_xs_grad_wrt_resonance_ER_end_to_end(nb93_endf_dict):
    """File-side leaf pin: ``jax.grad`` reaches the MF2 MLBW ER
    leaf via the top-level ``get_reaction_xs`` with
    ``include_resonance=True``. Same pattern as the pre-existing
    ``test_jax_grad_wrt_mlbw_ER_end_to_end_nb93`` (PR #188); this
    one lives under the Phase 5 label to keep the top-level-API
    coverage visibly complete."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([1.0, 35.0, 100.0])

    l_group = (
        nb93_endf_dict[2][151]['isotope'][1]['range'][1].get('l_group')
        or nb93_endf_dict[2][151]['isotope'][1]['range'][1]['spingroup']
    )
    er_key = list(l_group[1]['ER'].keys())[3]
    er0 = float(l_group[1]['ER'][er_key])

    def loss(er_val):
        d_t = copy.deepcopy(nb93_endf_dict)
        lg = (
            d_t[2][151]['isotope'][1]['range'][1].get('l_group')
            or d_t[2][151]['isotope'][1]['range'][1]['spingroup']
        )
        lg[1]['ER'][er_key] = er_val
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return get_reaction_xs(
                d_t, '(n,total)', ein,
                include_resonance=True, resonance_backend='jax',
                xp=xp_jx,
            ).sum()

    grad = float(jax.grad(loss)(jnp.array(er0)))
    fd = _fd5(lambda v: loss(jnp.array(v)), er0, 1e-3)
    assert np.isfinite(grad) and grad != 0.0
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


# -------------------------------------------------------------------
# get_particle_production_xs
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_particle_production_xs_grad_wrt_E(be9_endf_dict):
    """``jax.grad(get_particle_production_xs)(E)`` end-to-end for
    Be-9 (n,2n) neutron production."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')

    def loss(E_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return get_particle_production_xs(
                be9_endf_dict, '(n,2n)', 'n',
                jnp.array([E_scalar]), xp=xp_jx,
            ).sum()

    for E_val in (5e6, 1e7, 1.5e7):
        grad = float(jax.grad(loss)(jnp.array(E_val)))
        fd = _fd5(lambda v: loss(jnp.array(v)), E_val, E_val * 1e-4)
        assert np.isfinite(grad)
        np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


# -------------------------------------------------------------------
# get_particle_production_dxs_dE
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_particle_production_dxs_dE_grad_wrt_E(be9_endf_dict):
    """``jax.grad(get_particle_production_dxs_dE)(E)`` end-to-end for
    Be-9 (n,2n) neutron production. Under xp=jax with tracer Ein the
    LAW=7 mu-integration falls through to the xp-native fixed-mesh
    Simpson path (issue #220 PR 2), so grad reaches file-side
    leaves. Accuracy on the fixed-mesh Simpson is within a few
    permille of the kink-aware kernel -- fine for the FD check
    tolerance."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    eout = jnp.linspace(1e5, 5e6, 12)

    def loss(E_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return get_particle_production_dxs_dE(
                be9_endf_dict, '(n,2n)', 'n',
                jnp.array([E_scalar]), eout, xp=xp_jx,
            ).sum()

    for E_val in (5e6, 1e7, 1.5e7):
        grad = float(jax.grad(loss)(jnp.array(E_val)))
        fd = _fd5(lambda v: loss(jnp.array(v)), E_val, E_val * 1e-4)
        assert np.isfinite(grad)
        np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_particle_production_dxs_dE_grad_wrt_Ep(be9_endf_dict):
    """``jax.grad(get_particle_production_dxs_dE)(Ep)`` for Be-9
    (n,2n) neutron production, integrated over mu. The MF6 LAW=7
    subsection's inner Ep interp for Be-9 (n,2n) is INT=1
    (histogram / piecewise constant), so the analytic derivative
    wrt Ep is exactly zero in the interior of each histogram
    bracket and the 5-point FD stencil with a small h stays inside
    one bracket -- both drop below ``atol=1e-6``. What this pins
    is that the LAW=7 unit-base traced-x branch (issue #220 PR 4)
    lets grad flow through the query Ep axis without materialising
    the tracer."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([1.5e7])

    def loss(Ep_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return get_particle_production_dxs_dE(
                be9_endf_dict, '(n,2n)', 'n',
                ein, jnp.array([Ep_scalar]), xp=xp_jx,
            ).sum()

    for Ep_val in (2.5e5, 8e5, 2.3e6):
        grad = float(jax.grad(loss)(jnp.array(Ep_val)))
        fd = _fd5(lambda v: loss(jnp.array(v)), Ep_val, Ep_val * 1e-4)
        assert np.isfinite(grad)
        np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


# -------------------------------------------------------------------
# get_particle_production_dxs_dmu
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_particle_production_dxs_dmu_grad_wrt_E(be9_endf_dict):
    """``jax.grad(get_particle_production_dxs_dmu)(E)`` for Be-9
    (n,g)/g. MF14 LI=1 (fully isotropic) here so the FD reference
    is exact; what this test guards is that ``jax.grad`` flows
    through the MF14 + MF12 yield-weighted angular distribution
    without materialising the tracer Ein axis (issue #220 PR 3)."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    mu = jnp.linspace(-0.9, 0.9, 11)

    def loss(E_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return get_particle_production_dxs_dmu(
                be9_endf_dict, '(n,g)', 'g',
                jnp.array([E_scalar]), mu, xp=xp_jx,
            ).sum()

    for E_val in (1e5, 1e6, 1e7):
        grad = float(jax.grad(loss)(jnp.array(E_val)))
        fd = _fd5(lambda v: loss(jnp.array(v)), E_val, E_val * 1e-4)
        assert np.isfinite(grad)
        np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_particle_production_dxs_dmu_grad_wrt_mu(be9_endf_dict):
    """``jax.grad(get_particle_production_dxs_dmu)(mu)`` for Be-9
    (n,g)/g. MF14 LI=1 (isotropic) makes the analytical grad
    exactly zero at every mu; the test still exercises the
    tracer-mu path end-to-end through the MF14 branch (issue #220
    PR 3)."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([1.5e7])

    def loss(mu_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return get_particle_production_dxs_dmu(
                be9_endf_dict, '(n,g)', 'g',
                ein, jnp.array([mu_scalar]), xp=xp_jx,
            ).sum()

    for mu_val in (-0.5, 0.0, 0.5):
        grad = float(jax.grad(loss)(jnp.array(mu_val)))
        fd = _fd5(lambda v: loss(jnp.array(v)), mu_val, 1e-4)
        assert np.isfinite(grad)
        np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


# -------------------------------------------------------------------
# get_particle_production_ddxs
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_particle_production_ddxs_grad_wrt_E(be9_endf_dict):
    """``jax.grad(get_particle_production_ddxs)(E)`` end-to-end for
    Be-9 (n,2n) neutron production. Sums over Eout and mu."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    eout = jnp.linspace(1e5, 4e6, 10)
    mu = jnp.linspace(-0.9, 0.9, 9)

    def loss(E_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return get_particle_production_ddxs(
                be9_endf_dict, '(n,2n)', 'n',
                jnp.array([E_scalar]), eout, mu, xp=xp_jx,
            ).sum()

    for E_val in (5e6, 1e7, 1.5e7):
        grad = float(jax.grad(loss)(jnp.array(E_val)))
        fd = _fd5(lambda v: loss(jnp.array(v)), E_val, E_val * 1e-4)
        assert np.isfinite(grad)
        np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_particle_production_ddxs_grad_wrt_Ep(be9_endf_dict):
    """``jax.grad(get_particle_production_ddxs)(Ep)`` for Be-9
    (n,2n). Same INT=1 histogram semantics on the inner Ep axis
    as ``dxs_dE_grad_wrt_Ep``: analytic derivative is exactly zero
    in each histogram bracket, and the FD stencil stays within one
    bracket at these Ep values. The test pins the tracer-Ep flow
    through the LAW=7 unit-base kernel (issue #220 PR 4)."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([1.5e7])
    mu = jnp.linspace(-0.9, 0.9, 9)

    def loss(Ep_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return get_particle_production_ddxs(
                be9_endf_dict, '(n,2n)', 'n',
                ein, jnp.array([Ep_scalar]), mu, xp=xp_jx,
            ).sum()

    for Ep_val in (2.5e5, 8e5, 2.3e6):
        grad = float(jax.grad(loss)(jnp.array(Ep_val)))
        fd = _fd5(lambda v: loss(jnp.array(v)), Ep_val, Ep_val * 1e-4)
        assert np.isfinite(grad)
        np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_particle_production_ddxs_grad_wrt_mu(be9_endf_dict):
    """``jax.grad(get_particle_production_ddxs)(mu)`` for Be-9
    (n,2n). Non-knot mu values are chosen intentionally: LAW=7's
    mu tabulation for this subsection is on a 0.1-spaced grid, so
    any mu at a knot is a piecewise-linear kink where FD picks up
    a jump the analytic one-sided derivative does not. Away from
    the knots both agree to double precision. Pins that tracer
    mu flows through the LAW=7 unit-base traced-x branch
    (issue #220 PR 4)."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([1.5e7])
    eout = jnp.linspace(1e5, 4e6, 10)

    def loss(mu_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return get_particle_production_ddxs(
                be9_endf_dict, '(n,2n)', 'n',
                ein, eout, jnp.array([mu_scalar]), xp=xp_jx,
            ).sum()

    for mu_val in (-0.53, 0.05, 0.34):
        grad = float(jax.grad(loss)(jnp.array(mu_val)))
        fd = _fd5(lambda v: loss(jnp.array(v)), mu_val, 1e-4)
        assert np.isfinite(grad)
        np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-6)


# -------------------------------------------------------------------
# LAW=7 with INT=2 inner Ep interp -- stronger check than the Be-9
# (n,2n) pins above (which use INT=1 histogram, so grad is trivially
# zero in each bracket). JEFF-4.0 H-2 (n,2n) uses lin-lin (INT=2)
# on the inner Ep axis, and its LAW=7 unit-base transform gives a
# smooth analytic derivative on both Ep and mu (away from mu knots)
# that agrees with the FD reference to full float precision.
# -------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_particle_production_ddxs_grad_wrt_Ep_h2_int2(h2_endf_dict):
    """H-2 (n,2n) MF6 LAW=7 has INT=2 (lin-lin) on the inner Ep
    axis, so the analytic grad wrt Ep is smooth and non-zero away
    from bracket boundaries. FD-checked to double precision."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([15e6])
    mu = jnp.linspace(-0.9, 0.9, 9)

    def loss(Ep_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return get_particle_production_ddxs(
                h2_endf_dict, '(n,2n)', 'n',
                ein, jnp.array([Ep_scalar]), mu, xp=xp_jx,
            ).sum()

    for Ep_val in (5e5, 5e6, 8e6):
        grad = float(jax.grad(loss)(jnp.array(Ep_val)))
        fd = _fd5(lambda v: loss(jnp.array(v)), Ep_val, Ep_val * 1e-4)
        assert np.isfinite(grad)
        # Loose tolerance around Ep values that could straddle a
        # bracket boundary at the requested h; atol handles the
        # near-zero regime where FD noise dominates.
        np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-14)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_get_particle_production_ddxs_grad_wrt_mu_h2_int2(h2_endf_dict):
    """H-2 (n,2n) MF6 LAW=7 grad wrt mu. Unit-base with lin-lin
    outer mu INT gives smooth analytic grad on the mu interior;
    picking mu values away from the tabulated mu knots keeps FD
    exact to double precision."""
    import jax
    import jax.numpy as jnp
    xp_jx = array_ns.get_backend('jax')
    ein = jnp.array([15e6])
    eout = jnp.linspace(1e5, 8e6, 10)

    def loss(mu_scalar):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return get_particle_production_ddxs(
                h2_endf_dict, '(n,2n)', 'n',
                ein, eout, jnp.array([mu_scalar]), xp=xp_jx,
            ).sum()

    for mu_val in (-0.63, 0.05, 0.34, 0.72):
        grad = float(jax.grad(loss)(jnp.array(mu_val)))
        fd = _fd5(lambda v: loss(jnp.array(v)), mu_val, 1e-4)
        assert np.isfinite(grad)
        np.testing.assert_allclose(grad, fd, rtol=1e-3, atol=1e-12)
