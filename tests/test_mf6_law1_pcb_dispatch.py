"""Multi-panel jax.grad wrt E via numpy panel lookup + JIT single-
panel kernel (issue #190 follow-up prototype).

``integrate_law1_spectrum_pcb`` in
``mf6_law1_multipanel_traced``'s sibling module
``mf6_law1_pcb_dispatch`` implements the "hand-tuned fast lane" for
the multi-panel autodiff use case: ``jax.pure_callback`` runs the
numpy panel lookup, then dispatches to a JIT-compiled single-panel
primitive per panel. Whole thing wrapped in ``jax.custom_vjp`` so
``jax.grad`` routes through the same pattern on the backward pass.

Pins:

- Numpy vs pcb parity across a set of Es spanning several panels.
- ``jax.grad(sum(pcb_spectrum))(E)`` matches central FD at multiple
  mid-panel Es.
- pcb result equals the multipanel-traced result to floating-point
  round-off (same physics, different dispatch pattern).
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns

from _corpus import resolve_al27


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def al27_data_jx():
    if not _jax_available():
        pytest.skip('jax not installed')
    path = resolve_al27()
    if path is None:
        pytest.skip('Al-27 corpus not present')
    from endf_userpy.mfsec_interpretation import mf6_law1_preproc as _pre
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    xp_jx = array_ns.get_backend('jax')
    return _pre.mf6_law1_data_from_endf_dict(d, 16, 1, xp=xp_jx)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_pcb_dispatch_requires_jax(al27_data_jx):
    """Numpy backend raises with a suggestive message."""
    from endf_userpy.mfsec_interpretation import mf6_law1_pcb_dispatch as _pcb
    xp_np = array_ns.get_backend('numpy')
    with pytest.raises(ValueError, match='requires xp=jax'):
        _pcb.integrate_law1_spectrum_pcb(
            al27_data_jx, np.array([1.5e7]), np.linspace(1e5, 1e6, 5),
            xp=xp_np,
        )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_pcb_parity_with_numpy_multipanel(al27_data_jx):
    """pcb dispatch matches numpy single-panel loop to round-off
    across multiple Es in different panels."""
    import jax.numpy as jnp
    from endf_userpy.mfsec_interpretation import (
        mf6_law1_epintegral as _epi,
        mf6_law1_pcb_dispatch as _pcb,
        mf6_law1_preproc as _pre,
    )

    path = resolve_al27()
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    data_np = _pre.mf6_law1_data_from_endf_dict(d, 16, 1, xp=xp_np)

    ein = np.array([1.42e7, 1.55e7, 1.7e7, 1.9e7])
    eout = np.linspace(1e5, 5e6, 15)
    ref = _epi.integrate_law1_spectrum(data_np, ein, eout, to_lab=True)
    pcb = np.asarray(_pcb.integrate_law1_spectrum_pcb(
        al27_data_jx, jnp.asarray(ein), jnp.asarray(eout), xp=xp_jx,
    ))
    np.testing.assert_allclose(ref, pcb, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_pcb_grad_wrt_E_matches_fd(al27_data_jx):
    """jax.grad wrt E via pcb matches central FD at four mid-panel
    Es across four different panels."""
    import jax
    import jax.numpy as jnp
    from endf_userpy.mfsec_interpretation import mf6_law1_pcb_dispatch as _pcb

    xp_jx = array_ns.get_backend('jax')
    eout = jnp.linspace(1e5, 5e6, 15)

    def loss(E):
        return jnp.sum(_pcb.integrate_law1_spectrum_pcb(
            al27_data_jx, E[None], eout, xp=xp_jx,
        ))

    for E_val in (1.42e7, 1.55e7, 1.7e7, 1.9e7):
        E0 = jnp.array(E_val)
        val = float(loss(E0))
        grad = float(jax.grad(loss)(E0))
        fd_h = E_val * 1e-5
        fd = (float(loss(E0 + fd_h)) - float(loss(E0 - fd_h))) / (2 * fd_h)
        assert np.isfinite(grad)
        assert val > 0.0
        if abs(fd) < 1e-30:
            continue
        np.testing.assert_allclose(grad, fd, rtol=1e-3, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_pcb_matches_multipanel_traced(al27_data_jx):
    """pcb dispatch and multipanel-traced auto-dispatch give the
    same value (they compute the same physical quantity, different
    dispatch pattern)."""
    import jax.numpy as jnp
    from endf_userpy.mfsec_interpretation import (
        mf6_law1_epintegral as _epi,
        mf6_law1_pcb_dispatch as _pcb,
    )

    xp_jx = array_ns.get_backend('jax')
    ein = jnp.asarray(np.array([1.42e7, 1.55e7, 1.7e7]))
    eout = jnp.linspace(1e5, 5e6, 10)

    mp = np.asarray(_epi.integrate_law1_spectrum(
        al27_data_jx, ein, eout, to_lab=True, xp=xp_jx,
    ))
    pcb = np.asarray(_pcb.integrate_law1_spectrum_pcb(
        al27_data_jx, ein, eout, xp=xp_jx,
    ))
    np.testing.assert_allclose(mp, pcb, rtol=1e-9, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_pcb_batched_partial_chunk_padding(al27_data_jx):
    """Non-multiple-of-chunk-size input: the padded slots at the tail
    of a partial chunk must not leak into the returned rows."""
    import jax.numpy as jnp
    from endf_userpy.mfsec_interpretation import (
        mf6_law1_epintegral as _epi,
        mf6_law1_pcb_dispatch as _pcb,
        mf6_law1_preproc as _pre,
    )

    path = resolve_al27()
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    data_np = _pre.mf6_law1_data_from_endf_dict(d, 16, 1, xp=xp_np)

    # Es across three panels with counts that don't fill any single
    # chunk size cleanly: adaptive picks a different size per panel
    # and each has padded slots. The dispatcher must return exactly
    # rows[:actual] for every panel.
    ein = np.array([1.42e7, 1.45e7, 1.48e7, 1.5e7, 1.55e7, 1.62e7, 1.75e7])
    eout = np.linspace(1e5, 5e6, 12)
    ref = _epi.integrate_law1_spectrum(data_np, ein, eout, to_lab=True)
    pcb = np.asarray(_pcb.integrate_law1_spectrum_pcb(
        al27_data_jx, jnp.asarray(ein), jnp.asarray(eout), xp=xp_jx,
    ))
    np.testing.assert_allclose(ref, pcb, rtol=1e-10, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_pcb_batched_cache_bounded(al27_data_jx):
    """Repeated calls with different n_E over the same panels must
    not grow the cache unboundedly: sizes are quantised to the
    fixed ``_CHUNK_SIZES`` set, so cache is bounded by
    n_panels * len(_CHUNK_SIZES) per kind (fwd + grad)."""
    import jax
    import jax.numpy as jnp
    from endf_userpy.mfsec_interpretation import mf6_law1_pcb_dispatch as _pcb

    xp_jx = array_ns.get_backend('jax')
    eout = jnp.linspace(1e5, 5e6, 8)
    _pcb._CACHE_REGISTRY.clear()

    def total_cache_size():
        return sum(len(c._cache) for c in _pcb._CACHE_REGISTRY.values())

    # Two panels only. Vary n_E; sizes get quantised.
    rng = np.random.default_rng(0)
    mid1 = 1.42e7
    mid2 = 1.7e7
    max_expected = 2 * len(_pcb._CHUNK_SIZES)  # forward only
    for n in (3, 5, 10, 17, 60, 130):
        Es = np.concatenate([
            rng.uniform(mid1 - 1e5, mid1 + 1e5, size=n // 2),
            rng.uniform(mid2 - 1e5, mid2 + 1e5, size=n - n // 2),
        ])
        _pcb.integrate_law1_spectrum_pcb(
            al27_data_jx, jnp.asarray(Es), eout, xp=xp_jx,
        ).block_until_ready()

    fwd_cache = total_cache_size()
    assert fwd_cache <= max_expected, (
        f'forward cache exceeded bound: got {fwd_cache}, '
        f'expected <= {max_expected}'
    )

    # Backward path must stay under the same doubled bound
    # (fwd + grad kernels, both quantised).
    def loss(E):
        return _pcb.integrate_law1_spectrum_pcb(
            al27_data_jx, E, eout, xp=xp_jx,
        ).sum()

    for n in (3, 5, 10, 17):
        Es = np.concatenate([
            rng.uniform(mid1 - 1e5, mid1 + 1e5, size=n // 2),
            rng.uniform(mid2 - 1e5, mid2 + 1e5, size=n - n // 2),
        ])
        jax.grad(loss)(jnp.asarray(Es)).block_until_ready()
    assert total_cache_size() <= 2 * max_expected


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_pcb_grad_across_many_Es(al27_data_jx):
    """Vector grad wrt (n_E,) with Es spread across several panels
    matches per-E FD element-wise."""
    import jax
    import jax.numpy as jnp
    from endf_userpy.mfsec_interpretation import mf6_law1_pcb_dispatch as _pcb

    xp_jx = array_ns.get_backend('jax')
    eout = jnp.linspace(1e5, 5e6, 10)
    Es = jnp.asarray(np.array([1.42e7, 1.55e7, 1.7e7, 1.9e7]))

    def loss(E):
        return _pcb.integrate_law1_spectrum_pcb(
            al27_data_jx, E, eout, xp=xp_jx,
        ).sum()

    grad_vec = np.asarray(jax.grad(loss)(Es))
    assert grad_vec.shape == (4,)

    # Element-wise central FD, one E at a time.
    for i, E_val in enumerate([1.42e7, 1.55e7, 1.7e7, 1.9e7]):
        h = E_val * 1e-5
        Es_p = Es.at[i].set(E_val + h)
        Es_m = Es.at[i].set(E_val - h)
        fd = (float(loss(Es_p)) - float(loss(Es_m))) / (2 * h)
        if abs(fd) < 1e-30:
            continue
        np.testing.assert_allclose(grad_vec[i], fd, rtol=2e-3, atol=1e-30)
