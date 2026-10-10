"""Reich-Moore / RML resonances with zero eliminated (capture) width.

Such a resonance adds the real pole ``γγᵀ / (E_r - E)`` to the R-matrix.
Folded into ``R`` it made every backend lose precision near ``E_r``
(relative error ~ eps / |E/E_r - 1|, 2e-5 one ULP away) and return NaN
exactly at ``E_r``, although the cross sections are smooth there. The
resonance is now kept out of ``R`` and applied as an exact rank-one
(Sherman-Morrison) update of ``(I - i R P)⁻¹``.

No resonance in ENDF/B-VIII.1 (71 RM / RML ranges) has a zero
eliminated width; the cases are synthetic or derived from corpus data.

Also pinned here: the RML numpy backend returned NaN for every query
exactly at a resonance energy (any width), because the R-matrix
denominators used group-masked widths (0 for the zero-amplitude
resonances of other groups: 0 / 0).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.tab1 import TAB1
from endf_userpy.mfsec_interpretation import mf2_interpretation_reichmoore as rm
from endf_userpy.mfsec_interpretation import mf2_interpretation_rml as rml
from endf_userpy.mfsec_interpretation.mf2_interpretation_rml_preproc import (
    rml_data_from_endf_dict,
)

from _corpus import resolve_cu63_rml, resolve_pu239_rml

mp = pytest.importorskip('mpmath')

KI, RA, N = 2.19e-4, 0.96, 8
RNG = np.random.default_rng(3)
ER = np.sort(RNG.uniform(1.0, 100.0, N))
GN = RNG.uniform(1e-3, 1e-2, N)
GG = RNG.uniform(0.02, 0.05, N)
GF = RNG.normal(0.0, 0.05, N)
GG[3] = 0.0
ES = float(ER[3])
EPS = [-1e-3, -1e-12, -2.2e-16, 0.0, 2.2e-16, 1e-14, 1e-9]


def _reference(eps):
    """sct / cap / fis of one L=0 group with one fission channel,
    60 digits; at eps == 0 the energy is moved off E_s by 1e-45."""
    mp.mp.dps = 60
    f = mp.mpf
    e = f(ES) * (1 + (f(eps) if eps else f('1e-45')))
    rho = f(KI) * mp.sqrt(e) * f(RA)
    p = rho
    R = mp.matrix(2, 2)
    for r in range(N):
        g = [mp.sign(f(GN[r])) * mp.sqrt(abs(f(GN[r]))
                                          / (2 * f(KI) * mp.sqrt(f(ER[r])) * f(RA))),
             mp.sign(f(GF[r])) * mp.sqrt(abs(f(GF[r])) / 2)]
        inv = 1 / (f(ER[r]) - e - mp.mpc(0, 1) * f(GG[r]) / 2)
        for a in range(2):
            for b in range(2):
                R[a, b] += g[a] * g[b] * inv
    W = mp.matrix([[1 - 1j * R[0, 0] * p, -1j * R[0, 1]],
                   [-1j * R[1, 0] * p, 1 - 1j * R[1, 1]]])
    X = W ** -1 * R
    om = mp.e ** (-1j * rho)
    u00 = om * om * (1 + 2j * p * X[0, 0])
    u01 = om * 2j * mp.sqrt(p) * X[0, 1]
    pk = mp.pi / (f(KI) ** 2 * e)
    return np.array([float(pk * abs(1 - u00) ** 2),
                     float(pk * (1 - abs(u00) ** 2 - abs(u01) ** 2)),
                     float(pk * abs(u01) ** 2)])


def _constant_tab1(v):
    return TAB1(x=np.array([1e-5, 1e10]), y=np.array([v, v]),
                nbt=np.array([1], dtype=np.int32),
                intp=np.array([2], dtype=np.int32))


@pytest.mark.parametrize('backend', ['numpy', 'numba', 'jax'])
def test_rm_zero_width_resonance_matches_reference_at_and_near_er(backend):
    if backend not in array_ns.available_backends():
        pytest.skip(f'{backend} not installed')
    data = rm.RMData(
        abn=1.0, spi=0.5, ki=KI, r_a=_constant_tab1(RA), r_ap=_constant_tab1(RA),
        group_l=np.array([0], dtype=np.int32), group_g=np.array([1.0]),
        group_nfis=np.array([1], dtype=np.int32),
        res_group=np.zeros(N, dtype=np.int32), res_er=ER, res_gn=GN, res_gg=GG,
        res_gf1=GF, res_gf2=np.zeros(N),
    )
    e = ES * (1.0 + np.array(EPS))
    out = rm.reconstruct(data, e, array_ns.get_backend(backend))
    got = np.stack([np.asarray(out[k]) for k in ('sct', 'cap', 'fis')], axis=1)
    ref = np.stack([_reference(x) for x in EPS])
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


@pytest.fixture(scope='module')
def pu239_zero_width():
    path = resolve_pu239_rml()
    if path is None:
        pytest.skip('Pu-239 corpus file not present (see fetch.sh)')
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    data = rml_data_from_endf_dict(d)
    er = np.asarray(data.res_er)
    r0 = int(np.nonzero(er > 1.0)[0][0])
    g = int(data.res_group[r0])
    gam = np.array(data.res_gam, copy=True)
    gam[r0, rml._classify_group_channels(data, g)[2]] = 0.0
    data.res_gam = gam
    return data, float(er[r0])


def test_rml_zero_width_resonance_smooth_and_backend_consistent(pu239_zero_width):
    data, es = pu239_zero_width
    e = es * (1.0 + np.array(EPS + [-1e-14]))
    outs = {}
    for be in ('numpy', 'numba', 'jax'):
        if be in array_ns.available_backends():
            outs[be] = rml.reconstruct(data, e, array_ns.get_backend(be))
    for k in ('sct', 'cap', 'fis'):
        ref = np.asarray(outs['numpy'][k])
        assert np.all(np.isfinite(ref)) and np.all(ref > 0.0)
        at, near = ref[3], ref[[2, 4, 5, 7]]          # E_s and +-1e-14 .. 1 ULP
        np.testing.assert_allclose(near, at, rtol=1e-12)
        for be, out in outs.items():
            np.testing.assert_allclose(np.asarray(out[k]), ref, rtol=1e-12,
                                       err_msg=f'{k} {be}')


@pytest.mark.parametrize('resolve', [resolve_pu239_rml, resolve_cu63_rml])
def test_rml_numpy_finite_exactly_at_resonance_energies(resolve):
    path = resolve()
    if path is None:
        pytest.skip('LRF=7 corpus file not present (see fetch.sh)')
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    data = rml_data_from_endf_dict(d)
    er = np.asarray(data.res_er)
    e = er[er > 0.0][:100]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = rml.reconstruct(data, e, array_ns.get_backend('numba'))
        got = rml.reconstruct(data, e, array_ns.get_backend('numpy'))
    for k in ('sct', 'cap', 'fis', 'tot'):
        assert np.all(np.isfinite(np.asarray(got[k]))), k
        np.testing.assert_allclose(np.asarray(got[k]), np.asarray(ref[k]),
                                   rtol=1e-10, atol=1e-12, err_msg=k)


def test_rm_zero_width_resonance_under_jax_jit():
    """Inside a jit trace the widths read through ``xp.asarray`` are
    tracers; the zero-width detection must use the concrete dataclass
    fields, or the singular path (NaN at E_s) comes back."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')

    def tot(gn, e):
        data = rm.RMData(
            abn=1.0, spi=0.5, ki=KI, r_a=_constant_tab1(RA),
            r_ap=_constant_tab1(RA), group_l=np.array([0], dtype=np.int32),
            group_g=np.array([1.0]), group_nfis=np.array([1], dtype=np.int32),
            res_group=np.zeros(N, dtype=np.int32), res_er=ER, res_gn=gn,
            res_gg=GG, res_gf1=GF, res_gf2=np.zeros(N),
        )
        return rm.reconstruct(data, e, xp)['tot']

    e = jnp.asarray(ES * (1.0 + np.array(EPS)))
    got = np.asarray(jax.jit(tot)(jnp.asarray(GN), e))
    ref = np.array([_reference(x).sum() for x in EPS])
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)
