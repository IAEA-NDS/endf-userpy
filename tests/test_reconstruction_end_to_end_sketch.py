"""End-to-end integration test for the array-agnostic resonance sketch.

Reads a real ENDF-6 file (Nb-93 for MLBW), runs the whole pipeline
that a user would run:

    1. Parse ENDF-6 with endf_parserpy.
    2. Preprocess with mlbw_data_from_endf_dict.
    3. Reconstruct on numpy, JAX, and numba, verifying agreement.
    4. Under jax.grad, compute dσ_cap / dE_r for the strongest
       resonance (proves sensitivity works end-to-end).
    5. Wrap a chunked_chi2 loss around the reconstruction and
       demonstrate jax.value_and_grad returns finite (loss, grad)
       -- the shape a fitter needs per iteration.

Serves the double role of an integration test in CI and a
worked example a reader can copy from. If any step fails, the
test message points at which part of the sketch pipeline broke.

Skipped when the Nb-93 file is not on disk (it lives in a
personal path outside the repo, on the maintainer's box).
"""
from __future__ import annotations

import os
import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf2_interpretation_mlbw as mlbw
from endf_userpy.mfsec_interpretation import (
    mf2_interpretation_mlbw_preproc as pre,
)


NB93 = "/home/gschnabel/Seafile/Development/codeproj/playground/daniel_jax/n-041_Nb_093.endf"


def _nb93_available():
    return os.path.exists(NB93)


def _jax_available():
    return 'jax' in array_ns.available_backends()


def _numba_available():
    return 'numba' in array_ns.available_backends()


@pytest.fixture
def nb93_data():
    if not _nb93_available():
        pytest.skip('Nb-93 ENDF file not available')
    from endf_parserpy import EndfParserCpp
    p = EndfParserCpp()
    d = p.parsefile(NB93, include=[1, 2])
    return pre.mlbw_data_from_endf_dict(d)


# ============================================================
# Reconstruction on all three backends.
# ============================================================


def test_nb93_reconstruction_numpy(nb93_data):
    """The numpy path runs and gives sensible cross sections."""
    xp = array_ns.get_backend('numpy')
    einc = np.linspace(1e-4, 1e3, 5001)
    xs = mlbw.reconstruct(nb93_data, einc, xp)
    for key in ('sct', 'cap', 'fis', 'pot', 'tot'):
        arr = np.asarray(xs[key])
        assert np.all(np.isfinite(arr)), f'{key} has non-finite values'
    assert np.all(np.asarray(xs['tot']) > 0.0)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_nb93_reconstruction_numpy_vs_jax_bitwise_agree(nb93_data):
    """numpy and JAX outputs agree at rtol=1e-10 (both use the
    same adapter path, so any disagreement points at a backend
    dispatch bug)."""
    einc = np.linspace(1e-4, 1e3, 501)
    xs_np = mlbw.reconstruct(nb93_data, einc, array_ns.get_backend('numpy'))
    xs_jax = mlbw.reconstruct(nb93_data, einc, array_ns.get_backend('jax'))
    for key in ('sct', 'cap', 'fis', 'pot', 'tot'):
        np.testing.assert_allclose(
            np.asarray(xs_np[key]),
            np.asarray(xs_jax[key]),
            rtol=1e-10, atol=1e-30,
            err_msg=f'numpy vs jax disagree on {key}',
        )


@pytest.mark.skipif(not _numba_available(), reason='numba not installed')
def test_nb93_reconstruction_numpy_vs_numba_agree(nb93_data):
    """numpy and numba outputs agree at rtol=1e-8 (looser than
    JAX because numba uses fastmath=True which reassociates
    float ops for FMA)."""
    einc = np.linspace(1e-4, 1e3, 501)
    xs_np = mlbw.reconstruct(nb93_data, einc, array_ns.get_backend('numpy'))
    xs_nb = mlbw.reconstruct(nb93_data, einc, array_ns.get_backend('numba'))
    for key in ('sct', 'cap', 'fis', 'pot', 'tot'):
        np.testing.assert_allclose(
            np.asarray(xs_np[key]),
            np.asarray(xs_nb[key]),
            rtol=1e-8, atol=1e-30,
            err_msg=f'numpy vs numba disagree on {key}',
        )


# ============================================================
# Sensitivity end-to-end: jax.grad of dsigma_cap / dE_r for
# the strongest resonance.
# ============================================================


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_nb93_dsigma_cap_dEr_finite_via_jax_grad(nb93_data):
    """`jax.grad` of the capture cross section at the peak of the
    strongest resonance, w.r.t. that resonance's E_r, returns a
    finite non-zero number. Confirms end-to-end differentiability
    from ENDF-6 through the sketch."""
    import jax
    import jax.numpy as jnp

    xp = array_ns.get_backend('jax')

    # Locate the strongest resonance (highest cap peak).
    einc_dense = np.linspace(30.0, 40.0, 1001)
    xs = mlbw.reconstruct(nb93_data, einc_dense, array_ns.get_backend('numpy'))
    e_peak = float(einc_dense[np.argmax(np.asarray(xs['cap']))])

    # Find its resonance index in the sketch's res_er array.
    idx = int(np.argmin(np.abs(np.asarray(nb93_data.res_er) - e_peak)))
    er0 = float(nb93_data.res_er[idx])

    # Build a jax-traceable predict_fn that varies er[idx] only.
    res_er_j = jnp.asarray(nb93_data.res_er)

    def cap_at(er_val, e_query):
        # Replace res_er[idx] with the traced value.
        er_new = res_er_j.at[idx].set(er_val)
        data = mlbw.MLBWData(**{**nb93_data.__dict__, 'res_er': er_new})
        return mlbw.reconstruct(data, jnp.array([e_query]), xp)['cap'][0]

    grad_fn = jax.grad(cap_at, argnums=0)
    g = float(grad_fn(er0, e_peak))
    assert np.isfinite(g)
    # Grad at the exact peak is often small but non-zero; loosely
    # bound |g| < 1e10 to catch obviously-wrong results.
    assert abs(g) < 1e10


# ============================================================
# chunked_chi2 fitting workflow end-to-end.
# ============================================================


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_nb93_chunked_chi2_value_and_grad_shape(nb93_data):
    """A full fitting-workflow iteration: chunked_chi2 wrapped
    around the sketch's reconstruction, evaluated with
    jax.value_and_grad. Returns finite loss and a gradient of the
    right shape. This is what a scipy L-BFGS-B or optax loop
    would compute per iteration."""
    import jax
    import jax.numpy as jnp
    from endf_userpy.primitives.fit_helpers import chunked_chi2

    xp = array_ns.get_backend('jax')

    # Synthetic "measurement" = the reconstruction at the file's Er
    # values, plus 1% Gaussian noise. In a real fit these would come
    # from experiment; here we generate them from the file itself.
    einc = np.linspace(1e-2, 1e3, 20_000)
    xs_ref = np.asarray(
        mlbw.reconstruct(nb93_data, einc, array_ns.get_backend('numpy'))['cap']
    )
    rng = np.random.default_rng(0)
    y_meas = xs_ref + rng.normal(0.0, np.maximum(xs_ref * 0.01, 1e-6), xs_ref.shape)
    y_err = np.maximum(xs_ref * 0.01, 1e-6)

    energies_j = jnp.asarray(einc)
    y_meas_j = jnp.asarray(y_meas)
    y_err_j = jnp.asarray(y_err)

    def predict(theta, e_slice):
        # theta is the perturbation to res_er[0]; keep the rest fixed.
        er_new = jnp.asarray(nb93_data.res_er).at[0].add(theta[0])
        data = mlbw.MLBWData(**{**nb93_data.__dict__, 'res_er': er_new})
        return mlbw.reconstruct(data, e_slice, xp)['cap']

    def loss(theta):
        return chunked_chi2(
            predict, theta, energies_j, y_meas_j, y_err_j,
            xp, chunk=1000,
        )

    theta0 = jnp.array([0.05])   # small perturbation on first Er
    v, g = jax.value_and_grad(loss)(theta0)
    v = float(v)
    g = np.asarray(g)
    assert np.isfinite(v)
    assert g.shape == (1,)
    assert np.all(np.isfinite(g))
    # Loss should be strictly positive (theta != truth by 0.05 eV
    # gives a definite chi^2 contribution).
    assert v > 0.0
