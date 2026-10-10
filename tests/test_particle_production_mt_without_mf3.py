"""Particle production when an MT is described only through its photons.

ENDF/B-VIII.1 (and TENDL-2021) N-14 give MT28 (n,np) and MT32 (n,nd)
in MF13 / MF14 only; their neutrons, protons and deuterons are lumped
into MT5. Production of anything but photons is yield x the MF3 cross
section, so such an MT must not be admitted for those particles:
before the fix the admission (`contains_zap`'s reaction-name fallback,
then `satisfies_particle_production_select`) let it through and
`get_particle_production_xs(..., 'n' | 'p' | 'd')` raised KeyError on
the missing MF3 section.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.physical_constants import PARTICLE_ZAP
from endf_userpy.quantities import get_particle_production_xs
from endf_userpy.quantities_mt_zap import selectors
from endf_userpy.mfsec_interpretation import mf3_interpretation as mf3
from endf_userpy.run_options import RunOptions

from _corpus import resolve_n14


def test_production_admission_needs_mf3_for_non_photon_particles():
    """Synthetic dict: an MT present only in MF13 is not admitted for
    the production of any particle but photons (no cross section to
    attach a yield to), although the reaction itself does emit them."""
    d = {1: {451: {'NSUB': 10}}, 3: {2: {}}, 13: {28: {}}}
    for particle in ('n', 'p'):
        zap = PARTICLE_ZAP[particle]
        assert selectors.contains_zap(d, 28, zap) is True
        assert selectors.satisfies_particle_production_select(
            d, 28, [3], zap) is False


@pytest.fixture(scope='module')
def n14():
    path = resolve_n14()
    if path is None:
        pytest.skip('N-14 corpus file not present (see fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.mark.parametrize('particle,partial_mt', [
    ('n', 2),       # every elastic collision emits one neutron
    ('p', 103),     # (n,p)
    ('d', 104),     # (n,d)
])
def test_n14_production_bounded_below_by_its_main_channel(n14, particle,
                                                          partial_mt):
    assert 28 not in n14[3] and 28 in n14[13]
    e = np.geomspace(1e-3, 2e7, 400)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prod = np.asarray(get_particle_production_xs(
            n14, '(n,total)', particle, e, options=RunOptions(backend='numpy'),
        ))
        part = np.asarray(mf3.compute_cross_section_agnostic(
            n14, partial_mt, e, array_ns.get_backend('numpy'),
        ))
    fin = np.isfinite(prod)
    assert fin.sum() > 300
    assert np.all(prod[fin] >= part[fin] * (1.0 - 1e-12))
    assert np.max(prod[fin]) > 0.0
