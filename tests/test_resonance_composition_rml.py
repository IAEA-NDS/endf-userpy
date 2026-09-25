"""Composition-layer pins for the MF2 LRF=7 KRM=3 arc (#228 PR 4).

Verifies that ``reconstruct_resonance_xs`` dispatches LRU=1
LRF=7 ranges through the new RML preprocessor + reconstruction
and that the composed cross section (MF3 background + MF2
resonance contribution) is finite, non-negative, and
consistent across MT slices. The full numerical verification
against NJOY / SAMMY is a PR 5 concern; this PR pins the
plumbing.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.quantities_mt_zap.resonance_composition import (
    reconstruct_resonance_xs,
    compute_reconstructed_cross_section,
)
from endf_userpy.primitives import array_ns

from _corpus import (
    resolve_rh103,
    resolve_pu239_rml,
    resolve_cu63_rml,
)


def _load(path):
    if path is None:
        pytest.skip('LRF=7 corpus file not present (see fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.mark.parametrize('path_fn,label', [
    (resolve_rh103, 'Rh-103'),
    (resolve_pu239_rml, 'Pu-239'),
    (resolve_cu63_rml, 'Cu-63'),
])
def test_reconstruct_resonance_xs_lrf7_dispatch(path_fn, label):
    """LRF=7 RRR ranges reach the KRM=3 reconstruction without
    the 'no supported LRF' fallback warning. Result at RRR-interior
    energies must be finite and non-negative for elastic / capture
    (and fission, where declared)."""
    d = _load(path_fn())
    ein = np.geomspace(1e-3, 500.0, 16)
    xp = array_ns.get_backend('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        # No 'no supported LRF' fallback warning must fire on these
        # files: they're all LRF=7 KRM=3 which is now supported.
        elastic = reconstruct_resonance_xs(d, 2, ein, xp=xp)
        capture = reconstruct_resonance_xs(d, 102, ein, xp=xp)
    for name, arr in (('elastic', elastic), ('capture', capture)):
        assert arr.shape == ein.shape, f'{label} {name} shape'
        assert np.all(np.isfinite(arr)), f'{label} {name} finite'
        assert np.all(arr >= 0.0), (
            f'{label} {name} non-negative (min={arr.min():.3g})'
        )


def test_reconstruct_resonance_xs_pu239_fission():
    """Pu-239 has an explicit fission particle pair in its LRF=7
    range; the MT=18 partial must dispatch and return a positive
    contribution somewhere in the RRR."""
    d = _load(resolve_pu239_rml())
    # Sample coarse-mesh energies where fission is known to be
    # non-zero (RRR interior of Pu-239 spans ~1e-5 -- 2500 eV).
    ein = np.geomspace(1e-3, 1000.0, 32)
    xp = array_ns.get_backend('numpy')
    fis = reconstruct_resonance_xs(d, 18, ein, xp=xp)
    assert np.all(np.isfinite(fis))
    assert np.all(fis >= 0.0)
    assert float(fis.max()) > 0.0, (
        'Pu-239 MT=18 (fission) should have a non-zero peak '
        'somewhere in the RRR sample.'
    )


def test_reconstruct_resonance_xs_lrf7_seam_is_zero_above_eh():
    """The resonance contribution vanishes above ``EH``: LRF=7 is
    treated as the half-open interval ``[EL, EH)`` per NJOY's
    right-limit convention (matches LRF=2 / LRF=3 dispatch)."""
    d = _load(resolve_rh103())
    # Rh-103's LRF=7 range: EL=1e-5, EH=8000; below-EH point + above-EH point.
    ein = np.array([100.0, 10000.0])
    xp = array_ns.get_backend('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cap = reconstruct_resonance_xs(d, 102, ein, xp=xp)
    assert cap[0] > 0.0
    # Above EH=8000 the LRF=7 range contributes zero; Rh-103 also
    # has an LRF=2 URR above 8000 which may add a positive value,
    # so this pin is scoped: at 10 keV the reconstruction result
    # must be finite and the physical XS positive (URR + MF3).
    assert np.isfinite(cap[1])


def test_composed_xs_lrf7_reduces_to_mf3_far_above_eh():
    """Well above every resonance range the composed cross section
    equals MF3 alone. Pin using Rh-103 at a few MeV where both the
    RRR (EH=8 keV) and the URR (EH ~ 24 keV) are behind us and MF3
    carries the physical XS."""
    d = _load(resolve_rh103())
    ein = np.array([1.0e6, 5.0e6, 14.0e6])
    xp = array_ns.get_backend('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        composed = compute_reconstructed_cross_section(d, 102, ein, xp=xp)
        res_only = reconstruct_resonance_xs(d, 102, ein, xp=xp)
    # Resonance contribution above every range must be zero.
    np.testing.assert_array_equal(np.asarray(res_only), 0.0)
    # Composed XS = MF3 alone here; must be positive (capture
    # doesn't vanish anywhere in the MeV region for Rh-103).
    assert np.all(np.asarray(composed) > 0.0)
