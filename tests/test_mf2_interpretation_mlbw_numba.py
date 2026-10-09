"""Numba-path pins for the per-channel MLBW kernel.

The numba kernel sorts the resonances by channel, evaluates the
energy-dependent penetrability / shift once per channel (every
resonance of a channel shares its L) and accumulates each channel over
its contiguous slice.

- Interleaved ``res_channel`` (channels not contiguous in the input),
  mixed L, a competitive channel and a spin-0 target agree with the
  numpy backend.
- Data that breaks the ``res_l == ch_l[res_channel]`` invariant the
  per-channel evaluation relies on is rejected instead of silently
  using the channel's L.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.tab1 import TAB1
from endf_userpy.mfsec_interpretation import mf2_interpretation_mlbw as mlbw

pytest.importorskip('numba')

KEYS = ('sct', 'cap', 'fis', 'pot', 'rxx', 'tot')


def _constant_tab1(value):
    return TAB1(
        x=np.array([1e-5, 1e10], dtype=np.float64),
        y=np.array([value, value], dtype=np.float64),
        nbt=np.array([1], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )


def _interleaved_data(spi, seed=3, nres=83):
    rng = np.random.default_rng(seed)
    ch_l = np.array([0, 1, 2, 1, 3], dtype=np.int32)
    res_channel = rng.integers(0, ch_l.size, nres).astype(np.int32)
    return mlbw.MLBWData(
        abn=1.0, spi=spi, ki=2.196771e-3, qx=-50.0,
        r_a=_constant_tab1(0.62), r_ap=_constant_tab1(0.65),
        ch_l=ch_l,
        ch_g=np.array([0.25, 0.75, 0.5, 0.4, 0.6]),
        res_channel=res_channel,
        res_l=ch_l[res_channel],
        res_er=rng.uniform(-20.0, 1000.0, nres),
        res_gn=rng.uniform(1e-3, 5e-2, nres),
        res_gg=rng.uniform(0.02, 0.05, nres),
        res_gf=rng.uniform(0.0, 0.02, nres),
        res_gx=rng.uniform(0.0, 0.03, nres),
    )


@pytest.mark.parametrize('spi', [0.0, 2.5])
def test_mlbw_numba_matches_numpy_interleaved_channels(spi):
    data = _interleaved_data(spi)
    e = np.geomspace(1e-3, 1100.0, 1501)
    ref = mlbw.reconstruct(data, e, array_ns.get_backend('numpy'))
    got = mlbw.reconstruct(data, e, array_ns.get_backend('numba'))
    for key in KEYS:
        np.testing.assert_allclose(
            np.asarray(got[key]), np.asarray(ref[key]),
            rtol=1e-11, atol=1e-12 * np.max(np.abs(ref[key])), err_msg=key,
        )
    # The competitive channel opens above -qx = 50 eV.
    assert np.max(np.asarray(got['rxx'])[e > 60.0]) > 0.0


def test_mlbw_numba_rejects_inconsistent_resonance_l():
    data = _interleaved_data(2.5)
    res_l = np.array(data.res_l, copy=True)
    res_l[0] = (res_l[0] + 1) % 4
    data.res_l = res_l
    with pytest.raises(ValueError, match='res_l'):
        mlbw.reconstruct(data, np.array([1.0, 10.0]),
                         array_ns.get_backend('numba'))
