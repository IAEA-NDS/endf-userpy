"""Reconstruction pins for the MF2 LRF=7 KRM=3 arc (#228 PR 2).

Two flavours of test:

- **Corpus smoke tests**: on each of the three ad-hoc LRF=7 files
  (Rh-103, Pu-239, Cu-63), reconstruct the RRR at a spread of
  incident energies and assert the returned cross sections are
  finite, non-negative, and consistent (``tot == sct + cap +
  fis``). Full numerical verification against NJOY / SAMMY is a
  later PR of the arc.

- **Equivalence-to-R-M pin**: synthesise a minimal LRF=7 case
  with one elastic channel + one eliminated gamma channel per
  J-group and reconstruct it via the new KRM=3 code path. Then
  build the exact analogous R-M (LRF=3) input from the same
  parameters and reconstruct via the existing (well-tested)
  Reich-Moore reconstruction. The two curves must agree to
  double precision; any mismatch is a bug in the new KRM=3
  arithmetic, not in the physics.

The equivalence test is what actually pins the arithmetic; the
corpus smoke tests are shape / regression guards that would only
fail on a preproc breakage or a NaN-producing kernel bug.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation.mf2_interpretation_rml import (
    RMLData, reconstruct as rml_reconstruct,
)
from endf_userpy.mfsec_interpretation.mf2_interpretation_rml_preproc import (
    rml_data_from_endf_dict,
)
from endf_userpy.mfsec_interpretation.mf2_interpretation_reichmoore import (
    RMData, reconstruct as rm_reconstruct,
)
from endf_userpy.primitives import array_ns
from endf_userpy.primitives.tab1 import TAB1

from _corpus import (
    resolve_rh103,
    resolve_pu239_rml,
    resolve_cu63_rml,
)


def _load(path):
    if path is None:
        pytest.skip('LRF=7 corpus file not present (see fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


# ---------------------------------------------------------------
# Corpus smoke tests: reconstruction returns physically sensible
# arrays on real LRF=7 files.
# ---------------------------------------------------------------


@pytest.mark.parametrize('path_fn,label', [
    (resolve_rh103, 'Rh-103'),
    (resolve_pu239_rml, 'Pu-239'),
    (resolve_cu63_rml, 'Cu-63'),
])
def test_rml_corpus_reconstruction_smoke(path_fn, label):
    d = _load(path_fn())
    data = rml_data_from_endf_dict(d)
    xp = array_ns.get_backend('numpy')
    # Sample a handful of energies inside the RRR to keep the
    # test fast; corpus RRRs span 1e-5 -- (a few) 1e3 eV.
    ein = np.geomspace(1e-3, 500.0, 12)
    out = rml_reconstruct(data, ein, xp)
    for key in ('sct', 'cap', 'fis', 'pot', 'tot'):
        arr = out[key]
        assert arr.shape == ein.shape, f'{label} {key} shape'
        assert np.all(np.isfinite(arr)), f'{label} {key} finite'
        assert np.all(arr >= 0.0), (
            f'{label} {key} non-negative (min={arr.min():.3g})'
        )
    # Consistency: tot = sct + cap + fis.
    np.testing.assert_allclose(
        out['tot'], out['sct'] + out['cap'] + out['fis'],
        rtol=1e-10, atol=0.0,
    )


# ---------------------------------------------------------------
# Equivalence with LRF=3 R-M for the minimal 1-elastic + 1-gamma
# per group case.
# ---------------------------------------------------------------


def _make_synthetic_rml_and_rm(
    n_res: int,
    er_seed: float,
    gn_seed: float,
    gg_seed: float,
    L: int,
    aj: float,
    spi: float,
    ap: float,
    ki: float,
):
    """Build a minimal LRF=7 KRM=3 + LRF=3 R-M pair that describe
    the SAME resonance system: one J-group with one elastic
    channel + one eliminated gamma channel, ``n_res`` resonances.

    Parameters chosen so P_L(rho) at |E_r| stays away from zero
    for all resonances (keeps the reduced-width-amplitude scale
    well-defined).
    """
    rng = np.random.default_rng(0)
    er = np.linspace(1.0, 100.0, n_res) + er_seed * rng.standard_normal(n_res)
    gn = gn_seed * np.abs(rng.standard_normal(n_res)) + 0.1
    gg = gg_seed * np.ones(n_res)

    r_ap = TAB1(
        x=np.array([1e-5, 1e11]),
        y=np.array([ap, ap]),
        nbt=np.array([1], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )
    # g_J denominator uses target spin I=spi + incident spin 1/2:
    g_J = (2.0 * abs(aj) + 1.0) / ((2.0 * spi + 1.0) * 2.0)

    # ------- Build RMData (LRF=3) -------
    rmdata = RMData(
        abn=1.0,
        spi=spi,
        ki=ki,
        r_a=r_ap,
        r_ap=r_ap,
        group_l=np.array([L], dtype=np.int32),
        group_g=np.array([g_J], dtype=np.float64),
        group_nfis=np.array([0], dtype=np.int32),
        res_group=np.zeros(n_res, dtype=np.int32),
        res_er=er,
        res_gn=gn,
        res_gg=gg,
        res_gf1=np.zeros(n_res),
        res_gf2=np.zeros(n_res),
    )

    # ------- Build the analogous RMLData (LRF=7 KRM=3) -------
    # Two particle pairs: gamma (MT=102, MA=0) and elastic (MT=2,
    # MA=1). Per-group NCH=2: gamma channel (PPI=1) + elastic
    # channel (PPI=2). Per-resonance widths at slot 0 = gamma
    # (GG), slot 1 = elastic (GN). Same statistical weight.
    max_nch = 2
    ch_ppi = np.array([[1.0, 2.0]])              # (1, 2)
    ch_l   = np.array([[float(L), float(L)]])
    ch_sch = np.zeros_like(ch_l)
    ch_bnd = np.zeros_like(ch_l)
    ch_ape = np.array([[ap, ap]])
    ch_apt = np.array([[ap, ap]])
    ch_active = np.array([[True, True]])
    res_gam = np.stack([gg, gn], axis=1)          # (n_res, 2)

    rmldata = RMLData(
        abn=1.0,
        spi=spi,
        ki=ki,
        r_ap=r_ap,
        krm=3,
        ifg=0,
        krl=0,
        naps=1,
        pp_ma=np.array([0.0, 1.0]),               # gamma, neutron
        pp_mb=np.array([100.0, 100.0]),
        pp_za=np.zeros(2),
        pp_zb=np.zeros(2),
        pp_ia=np.array([1.0, 0.5]),
        pp_ib=np.array([spi, spi]),
        pp_q=np.zeros(2),
        pp_pnt=np.array([0.0, 1.0]),
        pp_shf=np.array([0.0, 0.0]),
        pp_mt=np.array([102.0, 2.0]),
        pp_pa=np.zeros(2),
        pp_pb=np.zeros(2),
        pp_incident_idx=2,
        group_aj=np.array([aj]),
        group_pj=np.array([0.0]),
        group_g=np.array([g_J]),
        group_nch=np.array([max_nch], dtype=np.int32),
        ch_ppi=ch_ppi,
        ch_l=ch_l,
        ch_sch=ch_sch,
        ch_bnd=ch_bnd,
        ch_ape=ch_ape,
        ch_apt=ch_apt,
        ch_active=ch_active,
        res_group=np.zeros(n_res, dtype=np.int32),
        res_er=er,
        res_gam=res_gam,
    )
    return rmldata, rmdata


@pytest.mark.parametrize('L,aj', [(0, 0.5), (1, 1.5), (2, 2.5)])
def test_rml_krm3_agrees_with_lrf3_reichmoore(L, aj):
    """A single J-group, 1 elastic + 1 gamma channel LRF=7 KRM=3
    should give identical cross sections to the analogous LRF=3
    R-M reconstruction. The two implementations share NONE of
    their code paths (independent R-matrix builders), so
    agreement to double precision is a strong correctness pin
    on the KRM=3 arithmetic."""
    spi = 0.5
    ap = 0.7   # fm
    # ki matching the MLBW preproc convention.
    ki = 2.196771e-3     # cm^-1 / sqrt(eV)  representative scale
    rmldata, rmdata = _make_synthetic_rml_and_rm(
        n_res=8, er_seed=0.3, gn_seed=2.0, gg_seed=0.15,
        L=L, aj=aj, spi=spi, ap=ap, ki=ki,
    )
    xp = array_ns.get_backend('numpy')
    ein = np.geomspace(0.1, 200.0, 40)
    out_rml = rml_reconstruct(rmldata, ein, xp)
    out_rm = rm_reconstruct(rmdata, ein, xp)
    # Elastic and capture: the two formalisms should match to
    # double precision away from the pot-scattering-only limit.
    np.testing.assert_allclose(
        out_rml['sct'], out_rm['sct'], rtol=1e-10, atol=1e-30,
    )
    np.testing.assert_allclose(
        out_rml['cap'], out_rm['cap'], rtol=1e-10, atol=1e-30,
    )
    # Fission: both formalisms should return zero when there is
    # no fission channel.
    np.testing.assert_array_equal(out_rml['fis'], 0.0)
    np.testing.assert_array_equal(out_rm['fis'], 0.0)


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_rml_reconstruct_jax_matches_numpy_on_rh103():
    """The reconstruction is array-agnostic via ``array_ns``.
    Numpy and JAX must produce the same cross sections to double
    precision on a real LRF=7 file."""
    import jax.numpy as jnp
    d = _load(resolve_rh103())
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    data_np = rml_data_from_endf_dict(d, xp=xp_np)
    data_jx = rml_data_from_endf_dict(d, xp=xp_jx)
    ein = np.geomspace(1e-3, 500.0, 12)
    out_np = rml_reconstruct(data_np, ein, xp_np)
    out_jx = rml_reconstruct(data_jx, jnp.asarray(ein), xp_jx)
    for k in ('sct', 'cap', 'fis', 'pot', 'tot'):
        np.testing.assert_allclose(
            np.asarray(out_np[k]),
            np.asarray(out_jx[k]),
            rtol=1e-10, atol=1e-12,
        )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_rml_reconstruct_jax_grad_wrt_er():
    """``jax.grad`` reaches the resonance-energy leaves of an
    ``RMLData`` — the whole point of array-agnostic reconstruction
    is that fits and sensitivity workflows on top of LRF=7 files
    work without a physics-code fork."""
    import jax
    import jax.numpy as jnp
    xp = array_ns.get_backend('jax')
    rmldata, _ = _make_synthetic_rml_and_rm(
        n_res=4, er_seed=0.0, gn_seed=2.0, gg_seed=0.15,
        L=0, aj=0.5, spi=0.5, ap=0.7, ki=2.196771e-3,
    )
    # Cast res_er to a jax array so autodiff can flow through it.
    from dataclasses import replace
    data_jx = replace(rmldata, res_er=jnp.asarray(rmldata.res_er))

    def loss(shift):
        d2 = replace(data_jx, res_er=data_jx.res_er + shift)
        out = rml_reconstruct(d2, jnp.array([50.0]), xp)
        return out['tot'].sum()

    g = float(jax.grad(loss)(jnp.array(0.0)))
    # FD reference at shift=0 with h=1e-3.
    h = 1e-3
    fd = float((loss(jnp.array(h)) - loss(jnp.array(-h))) / (2.0 * h))
    assert np.isfinite(g)
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


def test_rml_reconstruct_rejects_non_krm3():
    """Non-KRM=3 dataclasses raise a clear NotImplementedError."""
    rmldata, _ = _make_synthetic_rml_and_rm(
        n_res=2, er_seed=0.0, gn_seed=1.0, gg_seed=0.1,
        L=0, aj=0.5, spi=0.5, ap=0.7, ki=2.196771e-3,
    )
    # Manually override krm to something other than 3.
    from dataclasses import replace
    bad = replace(rmldata, krm=4)
    xp = array_ns.get_backend('numpy')
    with pytest.raises(NotImplementedError, match='KRM=4'):
        rml_reconstruct(bad, np.array([1.0]), xp)
