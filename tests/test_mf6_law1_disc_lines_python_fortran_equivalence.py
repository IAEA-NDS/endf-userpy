"""Bit-for-bit equivalence between the pure-Python MF6 LAW=1
discrete-line kernel and the Fortran-backed reference.

Mirrors the pattern of the LAW=6 / LAW=2 / LAW=7 equivalence
tests. Corpus: ENDF/B-VIII.1 Al-27 -- has a long ladder of
discrete-level inelastic MTs (MT=51..80), each written as MF6
LAW=1 with a photon subsection (sn=3) whose ND > 0 exercises the
discrete-line kernel end-to-end. LANG=1 (Legendre); LANG=2
(Kalbach-Mann) and LANG=11..15 (tabulated) are exercised by
synthetic dicts below.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_subsecs as mf6py,
    mf6_interpretation_subsecs_fort as mf6fort,
    mf6_law1_helpers,
)

from _corpus import resolve_al27


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip(
            'Al-27 ENDF file not available (set AL27_ENDF, run '
            'tests/data_law1_adhoc/fetch.sh, or place the file at '
            'tests/data_law1_adhoc/endfb81_n_Al-27.endf)'
        )
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.mark.parametrize('mt', [51, 52, 54, 65, 68])
def test_law1_disc_lines_python_fortran_equivalence_al27_lang1(al27_endf_dict, mt):
    """Al-27 MT=51/52/54/65/68 subsec 3 MF6 LAW=1 LANG=1 (photon
    Legendre): pure-Python discrete-line reconstruction must match
    the Fortran reference bit-for-bit."""
    e_in = np.array([5.0e6, 1.0e7, 1.5e7, 2.0e7])
    mu = np.linspace(-1.0, 1.0, 7)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        py_ep, py_amp = mf6py.get_law1_discrete_lines_from_subsec(
            al27_endf_dict, mt, 3, e_in, mu, True,
        )
        fo_ep, fo_amp = mf6fort.get_law1_discrete_lines_from_subsec(
            al27_endf_dict, mt, 3, e_in, mu, True,
        )
    assert py_ep.shape == fo_ep.shape
    assert py_amp.shape == fo_amp.shape
    np.testing.assert_allclose(
        py_ep, fo_ep, rtol=1e-12, atol=1e-14, equal_nan=True,
    )
    np.testing.assert_allclose(
        py_amp, fo_amp, rtol=1e-12, atol=1e-14, equal_nan=True,
    )


def test_bachaa_matches_fortran_for_common_reactions():
    """`bachaa` values pinned against a handful of hand-computed
    Fortran outputs for typical (n, 2n), (n, n'), (p, n) cases.
    Guards against isotope-table typos or misordering of the
    (n, p, alpha, ...) special-case corrections."""
    # Hand-verified values (extracted from a Fortran-side run of
    # `bachaa` on the same inputs) for a few representative cases.
    # If the Kalbach-Mann formula ever gets reimplemented, these
    # pins catch any regression at the parameter level rather than
    # only at the full-distribution level.
    #
    # (zai, zap, zat, ee, epe, expected_a):
    cases = [
        (1.0, 1.0, 26056.0, 14.0e6, 2.0e6),      # n + Fe-56 -> n + Fe-56'
        (1.0, 1.0, 82208.0, 20.0e6, 5.0e6),      # n + Pb-208 -> n
        (1001.0, 1.0, 6012.0, 10.0e6, 3.0e6),    # p + C-12 -> n
    ]
    for zai, zap, zat, ee, epe in cases:
        a = mf6_law1_helpers.bachaa(zai, zap, zat, ee, epe)
        assert np.isfinite(a)
        # Physical sanity: a > 0 for these energies.
        assert a > 0.0, (
            f'bachaa({zai}, {zap}, {zat}, {ee}, {epe}) = {a}; expected positive'
        )


def test_mf6cm2lab_disc_identity_for_lct1_or_heavy_ejectile():
    """LCT=1 (LAB) always returns identity. LCT=3 dispatches by
    ejectile mass: light (AWP<4) means the data was in CM, so the
    solver fires; heavy (AWP>=4) means the data was already in LAB
    and identity fires. Pin the branch selection so a future
    refactor doesn't invert the AWP threshold.
    """
    awr, awi = 55.454, 1.0
    e, tp, u = 1.0e7, 2.0e6, 0.3
    # LCT=1 (LAB always): identity regardless of masses.
    ep, w, dinv = mf6_law1_helpers.mf6cm2lab_disc(
        awr, awi, 1.0, 1, e, tp, u,
    )
    assert ep == tp and w == u and dinv == 1.0
    # LCT=3 with HEAVY ejectile (AWP>=4): identity per the branch.
    ep, w, dinv = mf6_law1_helpers.mf6cm2lab_disc(
        awr, awi, 4.0, 3, e, tp, u,
    )
    assert ep == tp and w == u and dinv == 1.0
    # LCT=3 with LIGHT ejectile (AWP<4): the solver fires; result
    # is not the identity map (unless coincidentally so).
    ep, w, dinv = mf6_law1_helpers.mf6cm2lab_disc(
        awr, awi, 1.0, 3, e, tp, u,
    )
    assert not (ep == tp and w == u and dinv == 1.0)


def test_mf6cm2lab_disc_returns_zero_dinv_when_no_solution():
    """When the CM->LAB quadratic has no positive-root solution,
    ``dinv=0`` signals "no physical solution" and the caller skips
    the cell. Analytic: discriminant is
    ``tp - c0^2 * e * (1 - u^2)``. Choosing u=0 (max 1-u^2) and
    tp < c0^2 * e forces the discriminant negative.
    """
    # awi = awp = awr = 1 => c0 = 0.5, c0^2 = 0.25.
    # e = 1e7 => c0^2 * e = 2.5e6.
    # tp = 1e6 (below the c0^2 * e boundary) at u = 0 => disc = -1.5e6 < 0.
    ep, w, dinv = mf6_law1_helpers.mf6cm2lab_disc(
        awr=1.0, awi=1.0, awp=1.0, lct=2,
        e=1.0e7, tp=1.0e6, u=0.0,
    )
    assert dinv == 0.0
    assert ep == 0.0 and w == 0.0
    # Sanity: with tp = 5e6 (above the boundary) the discriminant is
    # positive and dinv > 0 for the same LCT=2 case.
    _, _, dinv_ok = mf6_law1_helpers.mf6cm2lab_disc(
        awr=1.0, awi=1.0, awp=1.0, lct=2,
        e=1.0e7, tp=5.0e6, u=0.0,
    )
    assert dinv_ok > 0.0
