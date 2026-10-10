"""Reich-Moore / R-Matrix Limited capture without cancellation.

Capture (the eliminated channels) is ``(pi/k^2) g (1 - sum_c |U_0c|^2)``.
Evaluated literally, the subtraction cancels catastrophically wherever
capture is small compared with the unitarity bound: on the synthetic L=1
group below every backend was off by up to 5e-4 relative against a
50-digit reference, and numba / numpy disagreed by up to 3e-8 on
JEFF-4.0 Cu-63. All backends now use the exact identity

    1 - sum_c |U_0c|^2 = 4 P_0 w Im(R) w^H,   w = row 0 of (I - i R P)^-1,

a sum of non-negative terms (``Im R`` is positive semi-definite).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.tab1 import TAB1
from endf_userpy.mfsec_interpretation import mf2_interpretation_reichmoore as rm
from endf_userpy.quantities_mt_zap import resonance_composition as rc

from _corpus import resolve_cu63_rml, resolve_pu239_rml

mp = pytest.importorskip('mpmath')

KI, RA = 2.19e-4, 0.96
RNG = np.random.default_rng(7)
N = 20
ER = RNG.uniform(-20.0, 300.0, N)          # includes bound levels
GN = RNG.uniform(1e-4, 1e-2, N)
GG = RNG.uniform(0.02, 0.05, N)
GF1 = RNG.normal(0.0, 0.1, N)
GF2 = RNG.normal(0.0, 0.1, N)
E = np.concatenate([np.geomspace(1e-3, 300.0, 40), ER[ER > 0][:4] + 1e-7])


def _p1(rho):
    return rho ** 3 / (1 + rho ** 2)       # L = 1 penetrability


def _capture_reference(e, nfis):
    """(pi/k^2) (1 - sum_c |U_0c|^2) for one L=1 group, g=1, 50 digits."""
    mp.mp.dps = 50
    f = mp.mpf
    rho = f(KI) * mp.sqrt(f(e)) * f(RA)
    p = _p1(rho)
    nch = 1 + nfis
    R = mp.matrix(nch, nch)
    for r in range(N):
        rho_r = f(KI) * mp.sqrt(abs(f(ER[r]))) * f(RA)
        g = [mp.sign(f(GN[r])) * mp.sqrt(abs(f(GN[r])) / (2 * _p1(rho_r)))]
        for gf in (GF1, GF2)[:nfis]:
            g.append(mp.sign(f(gf[r])) * mp.sqrt(abs(f(gf[r])) / 2))
        inv = 1 / (f(ER[r]) - f(e) - mp.mpc(0, 1) * f(GG[r]) / 2)
        for a in range(nch):
            for b in range(nch):
                R[a, b] += g[a] * g[b] * inv
    P = [p] + [f(1)] * nfis
    W = mp.matrix(nch, nch)
    for a in range(nch):
        for b in range(nch):
            W[a, b] = (1 if a == b else 0) - mp.mpc(0, 1) * R[a, b] * P[b]
    X = W ** -1 * R
    s = abs(1 + 2j * p * X[0, 0]) ** 2
    for b in range(1, nch):
        s += abs(2j * mp.sqrt(p) * X[0, b]) ** 2
    return float(mp.pi / (f(KI) ** 2 * f(e)) * (1 - s))


def _constant_tab1(v):
    return TAB1(x=np.array([1e-5, 1e10]), y=np.array([v, v]),
                nbt=np.array([1], dtype=np.int32),
                intp=np.array([2], dtype=np.int32))


@pytest.fixture(scope='module')
def references():
    return {nfis: np.array([_capture_reference(e, nfis) for e in E])
            for nfis in (0, 1, 2)}


@pytest.mark.parametrize('backend', ['numpy', 'numba', 'jax'])
@pytest.mark.parametrize('nfis', [0, 1, 2])
def test_rm_capture_matches_50_digit_reference(references, backend, nfis):
    if backend not in array_ns.available_backends():
        pytest.skip(f'{backend} not installed')
    data = rm.RMData(
        abn=1.0, spi=0.5, ki=KI, r_a=_constant_tab1(RA), r_ap=_constant_tab1(RA),
        group_l=np.array([1], dtype=np.int32), group_g=np.array([1.0]),
        group_nfis=np.array([nfis], dtype=np.int32),
        res_group=np.zeros(N, dtype=np.int32), res_er=ER, res_gn=GN, res_gg=GG,
        res_gf1=GF1 if nfis >= 1 else np.zeros(N),
        res_gf2=GF2 if nfis >= 2 else np.zeros(N),
    )
    cap = np.asarray(rm.reconstruct(data, E, array_ns.get_backend(backend))['cap'])
    ref = references[nfis]
    assert np.all(ref > 0.0)
    np.testing.assert_allclose(cap, ref, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize('resolve', [resolve_pu239_rml, resolve_cu63_rml])
def test_rml_capture_agrees_across_backends(resolve):
    path = resolve()
    if path is None:
        pytest.skip('LRF=7 corpus file not present (see fetch.sh)')
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    rngs = d[2][151]['isotope'][1]['range']
    ri = next(k for k in rngs
              if int(rngs[k]['LRU']) == 1 and int(rngs[k]['LRF']) == 7)
    e = np.geomspace(max(float(rngs[ri]['EL']), 1e-4), float(rngs[ri]['EH']), 3000)
    caps = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for be in ('numpy', 'numba', 'jax'):
            if be not in array_ns.available_backends():
                continue
            out = rc._reconstruct_lru1_range(
                d, 1, ri, rngs[ri], e, array_ns.get_backend(be))[0]
            caps[be] = np.asarray(out['cap'])
    ref = caps.pop('numpy')
    assert np.all(ref > 0.0)
    for be, cap in caps.items():
        np.testing.assert_allclose(cap, ref, rtol=1e-10, atol=0.0, err_msg=be)
