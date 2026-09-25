"""End-to-end corpus verification pins for the MF2 LRF=7 KRM=3
arc (#228 PR 5).

Compares the reconstructed cross sections for the three ad-hoc
LRF=7 corpus files against widely-published thermal-neutron
values and against qualitative resonance behaviour (first strong
capture resonance appears at the tabulated ``E_r``, cross section
at that ``E_r`` is markedly above the surrounding baseline). All
tolerances are 15 % or looser to accommodate the spread in
handbook values across ENDF / JEFF / JENDL evaluations while
still being tight enough to catch a >~1 order-of-magnitude
regression.

The tests skip cleanly when the ad-hoc corpus is not present.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import get_reaction_xs

from _corpus import (
    resolve_rh103,
    resolve_pu239_rml,
    resolve_cu63_rml,
)


def _load(path):
    if path is None:
        pytest.skip('LRF=7 corpus file not present (see fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


# --- Thermal-value pins. References collected from
# NNDC / IAEA-NDS thermal-neutron-constant tabulations; values
# quoted are consistent across ENDF/B-VIII.1 (source of the
# tested files), JEFF-3.3, and JENDL-5 to within a few percent.
# Tolerance is 15 % so tests survive minor cross-library drift.


THERMAL = np.array([0.0253])
_RTOL = 0.15


def _thermal_xs(d, reaction):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(get_reaction_xs(
            d, reaction, THERMAL, include_resonance=True,
        ))


def test_rh103_thermal_capture():
    """Rh-103 (n,g) thermal capture ~ 145 b (NNDC evaluated
    thermal constants)."""
    d = _load(resolve_rh103())
    xs = _thermal_xs(d, '(n,g)')
    assert np.isclose(xs, 145.0, rtol=_RTOL), (
        f'Rh-103 (n,g) thermal = {xs:.2f} b; expected ~145 b '
        f'within {_RTOL * 100:.0f}%.'
    )


def test_rh103_thermal_elastic():
    """Rh-103 elastic thermal ~ 4.5 b (NNDC)."""
    d = _load(resolve_rh103())
    xs = _thermal_xs(d, '(n,n_0)')
    assert np.isclose(xs, 4.5, rtol=_RTOL), (
        f'Rh-103 (n,n_0) thermal = {xs:.2f} b; expected ~4.5 b '
        f'within {_RTOL * 100:.0f}%.'
    )


def test_pu239_thermal_capture():
    """Pu-239 (n,g) thermal capture ~ 270 b (NNDC)."""
    d = _load(resolve_pu239_rml())
    xs = _thermal_xs(d, '(n,g)')
    assert np.isclose(xs, 270.0, rtol=_RTOL), (
        f'Pu-239 (n,g) thermal = {xs:.2f} b; expected ~270 b '
        f'within {_RTOL * 100:.0f}%.'
    )


def test_pu239_thermal_fission():
    """Pu-239 (n,fission) thermal ~ 748 b (NNDC thermal fission
    cross section, one of the best-measured resonance-region
    values in nuclear data)."""
    d = _load(resolve_pu239_rml())
    xs = _thermal_xs(d, '(n,fission)')
    assert np.isclose(xs, 748.0, rtol=_RTOL), (
        f'Pu-239 (n,fission) thermal = {xs:.2f} b; expected ~748 b '
        f'within {_RTOL * 100:.0f}%.'
    )


def test_pu239_thermal_elastic():
    """Pu-239 elastic thermal ~ 7.7 b (NNDC)."""
    d = _load(resolve_pu239_rml())
    xs = _thermal_xs(d, '(n,n_0)')
    assert np.isclose(xs, 7.7, rtol=_RTOL), (
        f'Pu-239 (n,n_0) thermal = {xs:.2f} b; expected ~7.7 b '
        f'within {_RTOL * 100:.0f}%.'
    )


def test_cu63_thermal_capture():
    """Cu-63 (n,g) thermal ~ 4.5 b (NNDC)."""
    d = _load(resolve_cu63_rml())
    xs = _thermal_xs(d, '(n,g)')
    assert np.isclose(xs, 4.5, rtol=_RTOL), (
        f'Cu-63 (n,g) thermal = {xs:.2f} b; expected ~4.5 b '
        f'within {_RTOL * 100:.0f}%.'
    )


def test_cu63_thermal_elastic():
    """Cu-63 elastic thermal ~ 5.8 b (NNDC)."""
    d = _load(resolve_cu63_rml())
    xs = _thermal_xs(d, '(n,n_0)')
    assert np.isclose(xs, 5.8, rtol=_RTOL), (
        f'Cu-63 (n,n_0) thermal = {xs:.2f} b; expected ~5.8 b '
        f'within {_RTOL * 100:.0f}%.'
    )


# --- Resonance-peak positions: the LRF=7 reconstruction should
# reproduce a peak at each tabulated ``E_r``. Sample fine-mesh
# around a strong known resonance and assert the peak-fine value
# is a factor of >~ 3 above the surrounding baseline.


def test_rh103_capture_resonance_peak_at_1p26eV():
    """Rh-103 has a strong s-wave capture resonance near 1.257 eV;
    it dominates the thermal region. Verify a fine-mesh scan shows
    a sharp peak within ~ 1 eV of 1.257 eV."""
    d = _load(resolve_rh103())
    ein = np.linspace(0.5, 2.5, 401)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = np.asarray(get_reaction_xs(
            d, '(n,g)', ein, include_resonance=True,
        ))
    peak_idx = int(np.argmax(xs))
    e_peak = float(ein[peak_idx])
    xs_peak = float(xs[peak_idx])
    baseline = float(np.median(xs))
    assert 1.0 <= e_peak <= 1.5, (
        f'Rh-103 peak found at E={e_peak:.3f} eV; expected ~1.257 eV.'
    )
    assert xs_peak > 3.0 * baseline, (
        f'Rh-103 peak height {xs_peak:.1f} b is not clearly above '
        f'baseline {baseline:.1f} b in the 0.5..2.5 eV window.'
    )


def test_pu239_fission_resonance_peak_below_1eV():
    """Pu-239 has a strong fission resonance at 0.296 eV; the peak
    dominates the thermal-neighbourhood fission cross section."""
    d = _load(resolve_pu239_rml())
    ein = np.linspace(0.1, 0.7, 601)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = np.asarray(get_reaction_xs(
            d, '(n,fission)', ein, include_resonance=True,
        ))
    peak_idx = int(np.argmax(xs))
    e_peak = float(ein[peak_idx])
    xs_peak = float(xs[peak_idx])
    assert 0.2 <= e_peak <= 0.4, (
        f'Pu-239 fission peak at E={e_peak:.3f} eV; expected ~0.296 eV.'
    )
    # Reference peak XS at 0.296 eV in ENDF/B-VIII.1 evaluations is
    # in the 5000+ barn range.
    assert xs_peak > 3000.0, (
        f'Pu-239 fission peak height {xs_peak:.1f} b is well below '
        f'the expected ~5000+ b at the 0.296 eV resonance.'
    )


# --- Total = sum of partials consistency (composed with MF3).


def test_pu239_total_consistency_across_thermal_region():
    """(n,total) should equal (n,n_0) + (n,g) + (n,fission) at
    thermal (no other significant channels open). Composed XS
    (MF3 + reconstructed MF2) is expected to be internally
    consistent to within ~ 5 % at the 0.0253 eV point after
    MF3 background subtraction contributions are folded in."""
    d = _load(resolve_pu239_rml())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tot = _thermal_xs(d, '(n,total)')
        elastic = _thermal_xs(d, '(n,n_0)')
        cap = _thermal_xs(d, '(n,g)')
        fis = _thermal_xs(d, '(n,fission)')
    parts_sum = elastic + cap + fis
    np.testing.assert_allclose(tot, parts_sum, rtol=0.05)
