"""Numba-path pins for the vectorised R-matrix accumulator shared by the
Reich-Moore (LRF=3) and R-Matrix Limited (LRF=7 KRM=3) kernels.

- :func:`accumulate_r_matrix` (real / imaginary split, width products
  precomputed) equals the direct complex sum for 1, 2 and 3 channels
  and leaves absent channels at 0.
- Multi-group reconstruction with interleaved ``res_group`` and
  0 / 1 / 2 fission channels matches the numpy backend, which pins the
  per-group slicing of the sorted resonance arrays.

Resonances with zero eliminated width (singular at their energy) are
covered in ``test_rm_zero_width_limit.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.tab1 import TAB1
from endf_userpy.mfsec_interpretation import mf2_interpretation_reichmoore as rm

numba = pytest.importorskip('numba')

from endf_userpy.mfsec_interpretation.mf2_interpretation_reichmoore_numba import (  # noqa: E402
    accumulate_r_matrix,
)


def _constant_tab1(value):
    return TAB1(
        x=np.array([1e-5, 1e10], dtype=np.float64),
        y=np.array([value, value], dtype=np.float64),
        nbt=np.array([1], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )


@numba.njit
def _accumulate(E, er, hg, a, nch):
    return accumulate_r_matrix(
        E, er, hg, a[0], a[1], a[2], a[3], a[4], a[5], nch,
    )


@pytest.mark.parametrize('nch', [1, 2, 3])
def test_accumulate_r_matrix_matches_complex_sum(nch):
    rng = np.random.default_rng(nch)
    nres = 37    # not a multiple of the SIMD width: exercises the tail
    er = np.sort(rng.uniform(-50.0, 500.0, nres))
    hg = rng.uniform(0.01, 0.2, nres)
    gam = rng.normal(size=(3, nres))
    pairs = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)]
    a = np.stack([gam[c] * gam[d] for c, d in pairs])
    for E in (1e-3, 3.7, float(er[5]), 499.0):
        out = _accumulate(E, er, hg, a, nch)
        inv = 1.0 / (er - E - 1j * hg)
        used = {1: 1, 2: 3, 3: 6}[nch]
        for k in range(6):
            expected = np.sum(a[k] * inv) if k < used else 0.0
            np.testing.assert_allclose(out[k], expected, rtol=1e-13,
                                       atol=1e-13 * np.abs(a[k]).sum())


def test_rm_numba_matches_numpy_interleaved_groups():
    """Three groups (0, 1 and 2 fission channels) with resonances listed
    out of group order: the numba wrapper sorts by group and the kernel
    slices each group's contiguous block."""
    rng = np.random.default_rng(7)
    nres = 60
    res_group = rng.integers(0, 3, nres).astype(np.int32)
    data = rm.RMData(
        abn=1.0, spi=3.5, ki=2.19e-4,
        r_a=_constant_tab1(0.96), r_ap=_constant_tab1(0.96),
        group_l=np.array([0, 1, 0], dtype=np.int32),
        group_g=np.array([0.4375, 0.5625, 0.3], dtype=np.float64),
        group_nfis=np.array([0, 1, 2], dtype=np.int32),
        res_group=res_group,
        res_er=rng.uniform(-20.0, 300.0, nres),
        res_gn=rng.uniform(1e-4, 1e-2, nres) * rng.choice([-1, 1], nres),
        res_gg=rng.uniform(0.02, 0.05, nres),
        res_gf1=np.where(res_group >= 1, rng.normal(0, 0.1, nres), 0.0),
        res_gf2=np.where(res_group == 2, rng.normal(0, 0.1, nres), 0.0),
    )
    e = np.geomspace(1e-3, 300.0, 997)
    xs_np = rm.reconstruct(data, e, array_ns.get_backend('numpy'))
    xs_nb = rm.reconstruct(data, e, array_ns.get_backend('numba'))
    # Absorption is pi/k^2 g (1 - sum |U|^2): the cancellation in the
    # bracket loses digits on the pi/k^2 scale (large at low E) in every
    # backend, so the per-point tolerance is absolute on that scale.
    pi_k2 = np.pi / (data.ki ** 2 * e)
    for key in ('sct', 'fis', 'pot'):
        np.testing.assert_allclose(
            np.asarray(xs_nb[key]), np.asarray(xs_np[key]),
            rtol=1e-12, atol=0.0, err_msg=key,
        )
    for key in ('cap', 'tot'):
        diff = np.abs(np.asarray(xs_nb[key]) - np.asarray(xs_np[key]))
        assert np.all(diff < 1e-13 * pi_k2), key
