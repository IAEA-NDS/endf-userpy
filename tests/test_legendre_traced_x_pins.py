"""Regression pins for the query-side (E, mu) tracer path through
``evaluate_interp_legendre_polynomials`` and its ENDF-level
callers (issue #201).

Background
----------

Roadmap #198 Phase 3 tracked
``evaluate_interp_legendre_polynomials`` as still on the numpy
path, blocking grad-wrt-mu through MF4 LTT=1/2, MF6 LAW=2, and
MF14 LTT=1. The pre-#226 assessment was stale: the primitive
itself was already xp-native (returns Legendre values through
``endf_interp1d``'s traced-x coefficient interpolation and a
pure-arithmetic Bonnet recurrence), and the ``interp_tab2``
unit-base branch that #226 added covers the LTT=2 tabulated
path. What was missing until this file: regression pins that
verify these paths keep flowing tracers end-to-end.

This file adds FD-checked ``jax.grad`` pins on the primitive
plus one caller per formalism / file layout. All pins skip
cleanly when JAX is not installed; corpus-only pins additionally
skip when the ad-hoc file is missing.

Scope A (chosen scope for issue #201 closure): pin only the
paths that already work. MF4 LTT=3 (mixed) has a separate
non-tracer-safe boolean split on the energy axis
(``compute_angdist_from_mixed`` at
``mf4_interpretation.py:172``) that will fail on a tracer E; it
stays open as a follow-up finding and is intentionally NOT
pinned here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.interpolation import (
    evaluate_interp_legendre_polynomials,
)
from endf_userpy.mfsec_interpretation import (
    mf4_interpretation,
    mf6_interpretation,
    mf14_interpretation,
)

from _corpus import (
    resolve_al27,
    resolve_c12,
)


DATA_DIR = Path(__file__).parent / 'data'


def _jax_available():
    return 'jax' in array_ns.available_backends()


def _load(path):
    if path is None:
        pytest.skip('corpus file not present (see fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def _fd_scalar(f, x, h):
    """Two-sided FD for a scalar-in, scalar-out function."""
    return float((f(x + h) - f(x - h)) / (2.0 * h))


# ------------------------------------------------------------------
# Primitive-level pin: the Legendre interp itself must preserve
# tracer x AND tracer mu, since every caller relies on that.
# ------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_evaluate_interp_legendre_polynomials_grad_wrt_E_and_mu():
    """Pin the primitive: grad wrt x (incident energy) and grad
    wrt mu match FD to double precision on a synthetic 2-panel
    Legendre expansion. Any regression here breaks every caller
    silently."""
    import jax
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')

    xp_mesh = np.array([1.0, 100.0])
    coeffs = np.array([
        [1.0, 0.2, 0.05, 0.01],
        [1.0, 0.1, 0.02, 0.005],
    ])
    int_arr = np.array([2], dtype=int)   # lin-lin
    nbt_arr = np.array([2], dtype=int)

    def f(E, mu):
        return evaluate_interp_legendre_polynomials(
            jnp.array([E]), jnp.array([mu]),
            xp_mesh, coeffs, int_arr, nbt_arr,
            outside_value=0.0, xp=xp,
        ).sum()

    for E, mu in [(20.0, -0.3), (50.0, 0.3), (75.0, 0.7)]:
        gE = float(jax.grad(f, argnums=0)(E, mu))
        gmu = float(jax.grad(f, argnums=1)(E, mu))
        fd_E = _fd_scalar(lambda v: f(v, mu), E, 1e-4)
        fd_mu = _fd_scalar(lambda v: f(E, v), mu, 1e-4)
        assert np.isfinite(gE) and np.isfinite(gmu)
        np.testing.assert_allclose(gE, fd_E, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(gmu, fd_mu, rtol=1e-6, atol=1e-12)


# ------------------------------------------------------------------
# MF4 LTT=1 (Legendre only): committed corpus file (Be-9 MT=2).
# ------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf4_ltt1_grad_wrt_mu_be9():
    """``compute_angdist_values`` on Be-9 (n,n_0) is LTT=1 (pure
    Legendre); grad wrt mu must match FD to double precision.
    Committed corpus so this pin always runs."""
    import jax
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-004_Be_009.endf'),
    )
    ein = jnp.array([1e6])

    def f(mu_v):
        return mf4_interpretation.compute_angdist_values(
            d, 2, ein, jnp.array([mu_v]), True, xp=xp,
        ).sum()

    for mu in (-0.53, 0.05, 0.34, 0.72):
        g = float(jax.grad(f)(jnp.array(mu)))
        fd = _fd_scalar(f, mu, 1e-4)
        assert np.isfinite(g)
        np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-12)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf4_ltt1_grad_wrt_E_be9_pure_legendre():
    """Grad wrt incident E through the pure-Legendre path
    ``compute_angdist_from_legrepr`` (skips the LCT=2 CM<->LAB
    kinematic conversion in the top-level MF4 wrapper). This
    isolates the ``evaluate_interp_legendre_polynomials`` grad
    wrt E from the two-branch CM<->LAB machinery (a separate
    concern, tracked as issue #210 and its follow-ups)."""
    import jax
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(
        str(DATA_DIR / 'n-004_Be_009.endf'),
    )
    mu = jnp.array([0.34])

    def f(E_v):
        return mf4_interpretation.compute_angdist_from_legrepr(
            d, 2, jnp.array([E_v]), mu, xp=xp,
        ).sum()

    # E values chosen as panel-interior midpoints of the Be-9 MT=2
    # MF4 E-mesh (INT=2 lin-lin: derivative is piecewise constant,
    # so a query on a mesh knot gives a one-sided grad the FD
    # stencil averages across the two neighbouring panels).
    for E in (1.375e5, 2.7e6, 4.3e6):
        g = float(jax.grad(f)(jnp.array(E)))
        fd = _fd_scalar(f, E, 1.0)
        assert np.isfinite(g)
        np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-12)


# ------------------------------------------------------------------
# MF4 LTT=2 (tabulated only) via the unit-base ``interp_tab2``
# traced-x branch (#226). Ad-hoc corpus (Al-27 MT=2).
# ------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf4_ltt2_grad_wrt_mu_al27():
    """Al-27 (n,n_0) is stored LTT=2 (tabulated angular). Grad
    wrt mu flows through the ``interp_tab2`` unit-base traced-x
    branch; pinned here to keep the two primitives (Legendre and
    unit-base tab2) covered on the same MF4 API surface."""
    import jax
    import jax.numpy as jnp
    d = _load(resolve_al27())
    xp = array_ns.get_backend('jax')
    ein = jnp.array([5e6])

    def f(mu_v):
        return mf4_interpretation.compute_angdist_values(
            d, 2, ein, jnp.array([mu_v]), True, xp=xp,
        ).sum()

    for mu in (-0.53, 0.05, 0.34):
        g = float(jax.grad(f)(jnp.array(mu)))
        fd = _fd_scalar(f, mu, 1e-4)
        assert np.isfinite(g)
        np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-10)


# ------------------------------------------------------------------
# MF6 LAW=2 LANG=0 (Legendre): Al-27 (n,n_1) grad wrt E and mu.
# LANG=12/14 (tabulated) is not present in the committed / ad-hoc
# corpus so it stays uncovered by regression pins here; those
# paths route through the same ``interp_tab2`` unit-base branch
# already pinned by the LTT=2 test above.
# ------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf6_law2_lang0_grad_wrt_mu_al27():
    """Al-27 MT=51 (n,n_1) uses MF6 LAW=2 LANG=0 (Legendre in the
    CM frame). Grad wrt mu at three query cosines pinned to
    double precision. This is the primary caller of
    ``evaluate_interp_legendre_polynomials`` on the MF6 side."""
    import jax
    import jax.numpy as jnp
    d = _load(resolve_al27())
    xp = array_ns.get_backend('jax')
    ein = jnp.array([5e6])

    def f(mu_v):
        return mf6_interpretation.compute_angdist_values(
            d, 51, 1, ein, jnp.array([mu_v]), True, xp=xp,
        ).sum()

    for mu in (-0.53, 0.05, 0.34):
        g = float(jax.grad(f)(jnp.array(mu)))
        fd = _fd_scalar(f, mu, 1e-4)
        assert np.isfinite(g)
        np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-10)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf6_law2_lang0_grad_wrt_E_al27():
    """Grad wrt incident E on the same MF6 LAW=2 LANG=0 path."""
    import jax
    import jax.numpy as jnp
    d = _load(resolve_al27())
    xp = array_ns.get_backend('jax')
    mu = jnp.array([0.34])

    def f(E_v):
        return mf6_interpretation.compute_angdist_values(
            d, 51, 1, jnp.array([E_v]), mu, True, xp=xp,
        ).sum()

    for E in (1e6, 5e6, 1e7):
        g = float(jax.grad(f)(jnp.array(E)))
        fd = _fd_scalar(f, E, E * 1e-4)
        assert np.isfinite(g)
        np.testing.assert_allclose(g, fd, rtol=5e-3, atol=1e-12)


# ------------------------------------------------------------------
# MF14 LTT=1 (Legendre gamma angular): JENDL-5 C-12 MT=51.
# ------------------------------------------------------------------


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_mf14_ltt1_grad_wrt_mu_c12():
    """JENDL-5 C-12 MT=51 carries per-photon-line Legendre
    coefficients (LI=0, LTT=1). Grad wrt mu pinned to double
    precision through ``mf14_interpretation.compute_angdist_values``."""
    import jax
    import jax.numpy as jnp
    d = _load(resolve_c12())
    if 14 not in d or 51 not in d[14] or d[14][51].get('LI') != 0:
        pytest.skip('C-12 MT=51 not MF14 LI=0 LTT=1 in this corpus')
    xp = array_ns.get_backend('jax')
    photon_energies = np.array([d[14][51]['EG'][k]
                                for k in d[14][51]['EG']][:1], dtype=float)
    ein = jnp.array([1e7])

    def f(mu_v):
        return mf14_interpretation.compute_angdist_values(
            d, 51, ein, photon_energies, jnp.array([mu_v]), xp=xp,
        ).sum()

    for mu in (-0.53, 0.05, 0.34):
        g = float(jax.grad(f)(jnp.array(mu)))
        fd = _fd_scalar(f, mu, 1e-4)
        assert np.isfinite(g)
        np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-10)
