"""Regression tests for endf_userpy.quantities_mt_zap.selectors.

Pinned by issue #19: a JENDL-5 + NJOY (HEATR) merged file contains
MTs that are not reaction channels (MT 301..450) plus the X-particle
production sums (MT 201..207). Public APIs that iterate every MT in
MF3 must not raise on those rows.
"""

import pytest

import endf_userpy.quantities_mt_zap.selectors as selectors
import endf_userpy.primitives.physical_constants as physconst


def _neutron_endf_dict():
    """Smallest dict that satisfies prop.get_projectile (NSUB=10)."""
    return {1: {451: {'NSUB': 10}}}


@pytest.mark.parametrize("mt", [301, 302, 318, 444, 445, 446, 447, 450])
def test_contains_zap_skips_heatr_mts(mt):
    endf_dict = _neutron_endf_dict()
    zap_n = physconst.PARTICLE_ZAP['n']
    assert selectors.contains_zap(endf_dict, mt, zap_n) is False


@pytest.mark.parametrize("mt", [251, 252, 253])
def test_contains_zap_skips_dosimetry_like_mts(mt):
    endf_dict = _neutron_endf_dict()
    zap_n = physconst.PARTICLE_ZAP['n']
    assert selectors.contains_zap(endf_dict, mt, zap_n) is False


@pytest.mark.parametrize("mt,particle", [
    (201, 'n'), (202, 'g'), (203, 'p'), (204, 'd'),
    (205, 't'), (206, 'h'), (207, 'a'),
])
def test_contains_zap_skips_x_particle_production_mts(mt, particle):
    endf_dict = _neutron_endf_dict()
    zap = physconst.PARTICLE_ZAP[particle]
    assert selectors.contains_zap(endf_dict, mt, zap) is False


def test_contains_zap_real_channel_still_resolves():
    endf_dict = _neutron_endf_dict()
    zap_n = physconst.PARTICLE_ZAP['n']
    zap_g = physconst.PARTICLE_ZAP['g']
    # (n,2n) — emits two neutrons, no gamma.
    assert selectors.contains_zap(endf_dict, 16, zap_n) is True
    assert selectors.contains_zap(endf_dict, 16, zap_g) is False


def test_contains_zap_fission_special_case_preserved():
    endf_dict = _neutron_endf_dict()
    zap_n = physconst.PARTICLE_ZAP['n']
    zap_p = physconst.PARTICLE_ZAP['p']
    assert selectors.contains_zap(endf_dict, 18, zap_n) is True
    assert selectors.contains_zap(endf_dict, 18, zap_p) is False
