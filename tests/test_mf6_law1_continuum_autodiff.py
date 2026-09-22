"""Backend-agnostic MF6 LAW=1 continuum reconstruction tests
mirroring the MF6 LAW=2 pattern from #158.

Pins:

1. Numpy-vs-JAX parity on the full LAW=1 chain across LANG=1
   (photon, Be-9) and LANG=2 (Kalbach-Mann, Al-27).
2. ``jax.grad`` reaches back from the reconstructed continuum
   distribution to file-stored angular parameters
   (``data.b_panels``), for the dataclass-first entry path.
3. ``jax.grad`` reaches back through the dict-facing entry point
   when a JAX tracer is stored at an ``endf_dict[6][mt]
   ['subsection'][sn]['b'][panel][ep_idx][coef_idx]`` leaf.
"""
from __future__ import annotations

import copy
import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_subsecs as mf6subsec,
    mf6_law1_preproc,
    mf6_law1_kernel,
)
from endf_userpy.primitives import array_ns

from _corpus import resolve_al27


DATA_DIR = Path(__file__).resolve().parent / 'data'


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip(
            'Al-27 ENDF file not available (run '
            'tests/data_law1_adhoc/fetch.sh)'
        )
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def be9_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        DATA_DIR / 'n-004_Be_009.endf',
    )


def _query_grid_inside_panel(subsec, panel_idx):
    """Build (E, E', mu) inside the given panel so the physics is
    nonzero."""
    Ep_dict = subsec['Ep'][panel_idx + 1]
    ep_vals = list(Ep_dict.values())
    ep_lo = float(ep_vals[0])
    ep_hi = float(ep_vals[-1])
    e_val = float(subsec['E'][panel_idx + 1])
    e_next = float(subsec['E'][panel_idx + 2])
    e_in = np.array([0.5 * (e_val + e_next)])
    e_out = np.linspace(max(1e3, ep_lo * 1.1), ep_hi * 0.9, 4)
    mu = np.array([-0.5, 0.0, 0.5])
    return e_in, e_out, mu


def test_kernel_from_dataclass_matches_dict_entry_point_on_numpy(al27_endf_dict):
    """Bit-identical parity between the preproc+kernel entry path
    and the dict-facing entry point on numpy."""
    xp_np = array_ns.get_backend('numpy')
    data = mf6_law1_preproc.mf6_law1_data_from_endf_dict(al27_endf_dict, 91, 1)
    e_in, e_out, mu = _query_grid_inside_panel(
        al27_endf_dict[6][91]['subsection'][1], panel_idx=7,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        kernel_out = np.asarray(mf6_law1_kernel.reconstruct(
            data, e_in, e_out, mu, True, xp=xp_np,
        ))
        dict_out = np.asarray(mf6subsec.get_dist2d_from_subsec_law1(
            al27_endf_dict, 91, 1, e_in, e_out, mu, True, xp=xp_np,
        ))
    np.testing.assert_array_equal(kernel_out, dict_out)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law1_continuum_numpy_jax_parity_al27_kalbach(al27_endf_dict):
    """Al-27 MT=91 (n, n') continuum LANG=2 Kalbach-Mann: numpy vs
    JAX to machine precision."""
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')
    data = mf6_law1_preproc.mf6_law1_data_from_endf_dict(al27_endf_dict, 91, 1)
    e_in, e_out, mu = _query_grid_inside_panel(
        al27_endf_dict[6][91]['subsection'][1], panel_idx=7,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        out_np = np.asarray(mf6_law1_kernel.reconstruct(
            data, e_in, e_out, mu, True, xp=xp_np,
        ))
        out_jax = np.asarray(mf6_law1_kernel.reconstruct(
            data, e_in, e_out, mu, True, xp=xp_jax,
        ))
    np.testing.assert_allclose(out_np, out_jax, rtol=1e-11, atol=1e-14)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law1_continuum_numpy_jax_parity_be9_photon(be9_endf_dict):
    """Be-9 MT=701 continuum LANG=1 photon (Legendre): numpy vs
    JAX to machine precision."""
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')
    data = mf6_law1_preproc.mf6_law1_data_from_endf_dict(be9_endf_dict, 701, 3)
    # Be-9 MT=701 panel 1 has only a placeholder Ep=1; pick a panel
    # further along with real range.
    subsec = be9_endf_dict[6][701]['subsection'][3]
    n_panels = len(subsec['E'])
    panel_idx = min(2, n_panels - 2)
    e_in, e_out, mu = _query_grid_inside_panel(subsec, panel_idx=panel_idx)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        out_np = np.asarray(mf6_law1_kernel.reconstruct(
            data, e_in, e_out, mu, True, xp=xp_np,
        ))
        out_jax = np.asarray(mf6_law1_kernel.reconstruct(
            data, e_in, e_out, mu, True, xp=xp_jax,
        ))
    np.testing.assert_allclose(out_np, out_jax, rtol=1e-11, atol=1e-14)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_flows_from_dataclass_b_panels(al27_endf_dict):
    """Primary issue #154 use case for LAW=1: ``jax.grad`` of a
    scalar summary of the reconstructed continuum distribution wrt
    the tracer-replaced ``b_panels`` field returns a finite,
    non-zero gradient with a nonzero count of nonzero entries.

    Finite-diff sanity on one specific coefficient.
    """
    import jax
    import jax.numpy as jnp
    xp_jax = array_ns.get_backend('jax')
    data = mf6_law1_preproc.mf6_law1_data_from_endf_dict(al27_endf_dict, 91, 1)
    e_in, e_out, mu = _query_grid_inside_panel(
        al27_endf_dict[6][91]['subsection'][1], panel_idx=7,
    )
    b0 = jnp.asarray(data.b_panels)

    def loss(b):
        d_traced = dataclasses.replace(data, b_panels=b)
        return jnp.sum(mf6_law1_kernel.reconstruct(
            d_traced, e_in, e_out, mu, True, xp=xp_jax,
        ))

    val = float(loss(b0))
    grad = np.asarray(jax.grad(loss)(b0))
    assert val > 0.0
    assert grad.shape == b0.shape
    assert np.all(np.isfinite(grad))
    assert int(np.sum(grad != 0.0)) >= 1
    # Finite-diff sanity on one representative entry.
    # Panel 8 (0-indexed = 8) is inside the query bracket.
    ip, jp, kp = 8, 5, 0
    eps = 1e-4
    lp = float(loss(b0.at[ip, jp, kp].add(eps)))
    lm = float(loss(b0.at[ip, jp, kp].add(-eps)))
    fd = (lp - lm) / (2.0 * eps)
    np.testing.assert_allclose(grad[ip, jp, kp], fd, rtol=1e-3, atol=1e-8)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_flows_from_dict_stored_tracer(al27_endf_dict):
    """Dict-first path: user writes a JAX tracer directly into
    ``endf_dict[6][mt]['subsection'][sn]['b'][panel][ep_row][coef_idx]``
    and calls the dict-facing entry point with a JAX backend. The
    tracer-preserving ``dict2array`` in the preproc keeps the
    tracer alive, and jax.grad reaches back through it."""
    import jax
    import jax.numpy as jnp
    xp_jax = array_ns.get_backend('jax')
    e_in, e_out, mu = _query_grid_inside_panel(
        al27_endf_dict[6][91]['subsection'][1], panel_idx=7,
    )
    # Panel 8 (1-indexed 8+1=9 -> use 8 with 0-indexed logic; here
    # dict is 1-indexed so panel_idx=7 -> dict key 8), pick a
    # tabulated (Ep_row, coef) leaf. Al-27 LANG=2 has NA=1 -> b has
    # 2 cols per row (f0, r).
    panel_key = 8
    ep_row_key = 5     # 1-indexed
    coef_idx_key = 1   # 1-indexed: b[panel][ep_row][coef]
    original = float(
        al27_endf_dict[6][91]['subsection'][1]['b'][panel_key][ep_row_key][coef_idx_key]
    )

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[6][91]['subsection'][1]['b'][panel_key][ep_row_key][coef_idx_key] = theta
        return jnp.sum(mf6subsec.get_dist2d_from_subsec_law1(
            d_t, 91, 1, e_in, e_out, mu, True, xp=xp_jax,
        ))

    grad = float(jax.grad(loss)(jnp.array(original)))
    assert np.isfinite(grad)
    # Finite-diff sanity
    eps = 1e-4
    lp = float(loss(jnp.array(original + eps)))
    lm = float(loss(jnp.array(original - eps)))
    fd = (lp - lm) / (2.0 * eps)
    # Rel tolerance loose because Kalbach loss surface has some
    # tricky gradient scales.
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-8)
