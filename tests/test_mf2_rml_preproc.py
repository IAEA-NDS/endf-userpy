"""Preproc-side pins for the MF2 LRF=7 (R-Matrix Limited) arc
(#228 PR 1). Verifies that ``rml_data_from_endf_dict`` extracts
the three ad-hoc corpus files' LRF=7 ranges into an ``RMLData``
with the expected shapes, per-pair MT / mass assignments, per-
J-group AJ / NCH counts, statistical weights, resonance counts,
and padded channel-width layout.

Each test skips cleanly when its corpus file is not present.
Numeric spot-checks target invariant physical quantities (widths
that appear in the file's first row for a given group) so tests
would flake only on genuine preproc regressions, not on any
downstream reconstruction change.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation.mf2_interpretation_rml import (
    RMLData,
)
from endf_userpy.mfsec_interpretation.mf2_interpretation_rml_preproc import (
    rml_data_from_endf_dict,
)

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
# Rh-103: compact LRF=7 case (5 J-groups, 2 pairs, NCH in {2, 3})
# ---------------------------------------------------------------


def test_rml_preproc_rh103_shapes_and_scalars():
    d = _load(resolve_rh103())
    data = rml_data_from_endf_dict(d)

    assert isinstance(data, RMLData)
    # Range scalars.
    assert data.krm == 3
    assert data.krl == 0
    assert data.ifg == 0
    assert data.naps == 1
    assert data.n_pp() == 2
    assert data.n_groups() == 5
    # Two particle pairs: gamma+capture (MT=102) and neutron+elastic (MT=2).
    mts = [int(round(m)) for m in np.asarray(data.pp_mt)]
    assert sorted(mts) == [2, 102]
    # Elastic pair index (1-based) points at the pair with MT=2.
    assert int(round(float(data.pp_mt[data.pp_incident_idx - 1]))) == 2
    # Per-group NCH matches the file dump (dict-order-preserving).
    assert list(np.asarray(data.group_nch)) == [2, 2, 2, 3, 2]
    # ``max_nch`` propagates to the per-channel padding dim.
    assert data.max_nch() == 3
    # Padded channel rows carry ``ch_active=False`` beyond the
    # group's true NCH.
    assert not bool(data.ch_active[0, 2])   # NCH=2 group has no ch idx 2
    assert bool(data.ch_active[3, 2])       # NCH=3 group has ch idx 2


def test_rml_preproc_rh103_statistical_weights():
    d = _load(resolve_rh103())
    data = rml_data_from_endf_dict(d)
    spi = data.spi
    # Rh-103 target has I=1/2 -> denominator (2I+1)(2i+1) = 2*2 = 4.
    denom = (2.0 * spi + 1.0) * (2.0 * 0.5 + 1.0)
    for g in range(data.n_groups()):
        aj = float(data.group_aj[g])
        expected = (2.0 * abs(aj) + 1.0) / denom
        np.testing.assert_allclose(
            data.group_g[g], expected, rtol=1e-12,
        )


def test_rml_preproc_rh103_resonance_counts():
    d = _load(resolve_rh103())
    data = rml_data_from_endf_dict(d)
    # NRS per group from the file dump: [46, 45, 155, 158, 151].
    expected_nrs = [46, 45, 155, 158, 151]
    assert data.n_res() == sum(expected_nrs)
    # ``res_group`` runs 0, 0, ..., 1, 1, ..., ..., 4, 4, ....
    for gi, n in enumerate(expected_nrs):
        assert int((np.asarray(data.res_group) == gi).sum()) == n


def test_rml_preproc_rh103_first_resonance_widths_pin():
    """Pins the padded (nres, max_nch) width layout: the first
    resonance of the first J-group has group NCH=2, so its ``res_gam``
    row must carry the file's two GAM values in slots [0, 1] and 0.0
    in the padded slot [2]."""
    d = _load(resolve_rh103())
    data = rml_data_from_endf_dict(d)
    # First resonance of J-group 0 has ER = -1491.814 eV (bound state).
    er_first = float(np.asarray(data.res_er)[0])
    np.testing.assert_allclose(er_first, -1491.814, rtol=1e-6)
    # File dump: GAM[1] first row = 0.1816456; GAM[2] first row = ??
    # We only pin the padded slot: NCH=2, so ch_gam[0, 2] == 0.
    gam0 = np.asarray(data.res_gam)[0]
    assert gam0.shape == (3,)
    np.testing.assert_allclose(gam0[0], 0.1816456, rtol=1e-6)
    assert gam0[2] == 0.0


# ---------------------------------------------------------------
# Pu-239: fission-bearing LRF=7 (3 pairs, 2 J-groups, up to 1604
# resonances per group)
# ---------------------------------------------------------------


def test_rml_preproc_pu239_shapes():
    d = _load(resolve_pu239_rml())
    data = rml_data_from_endf_dict(d)
    assert data.krm == 3
    assert data.n_pp() == 3
    assert data.n_groups() == 2
    # Three MTs: capture (102), elastic (2), fission (18).
    mts = sorted(int(round(m)) for m in np.asarray(data.pp_mt))
    assert mts == [2, 18, 102]
    # Per-group NCH from the dump: [4, 3]; padded dim = 4.
    assert list(np.asarray(data.group_nch)) == [4, 3]
    assert data.max_nch() == 4
    # NRS per group: [481, 1604] -> total 2085.
    assert data.n_res() == 2085
    assert int((np.asarray(data.res_group) == 0).sum()) == 481
    assert int((np.asarray(data.res_group) == 1).sum()) == 1604


def test_rml_preproc_pu239_incident_pair_is_elastic():
    d = _load(resolve_pu239_rml())
    data = rml_data_from_endf_dict(d)
    assert int(
        round(float(data.pp_mt[data.pp_incident_idx - 1]))
    ) == 2


# ---------------------------------------------------------------
# JEFF-4.0 Cu-63: mid-mass LRF=7 (6 J-groups)
# ---------------------------------------------------------------


def test_rml_preproc_cu63_shapes():
    d = _load(resolve_cu63_rml())
    data = rml_data_from_endf_dict(d)
    assert data.krm == 3
    assert data.n_pp() == 2
    assert data.n_groups() == 6
    assert list(np.asarray(data.group_nch)) == [2, 2, 2, 3, 3, 2]
    assert data.max_nch() == 3
    # NRS per group from the dump.
    expected_nrs = [189, 356, 45, 156, 148, 204]
    assert data.n_res() == sum(expected_nrs)


# ---------------------------------------------------------------
# Reject-cases: preproc raises on inputs outside the initial scope.
# ---------------------------------------------------------------


def test_rml_preproc_rejects_non_lrf7_range():
    """A range with ``LRF != 7`` is rejected with a clear error."""
    d = _load(resolve_rh103())
    # Rh-103's second range (URR) is LRU=2 LRF=2 -> should be
    # rejected as it isn't LRU=1 LRF=7.
    with pytest.raises(ValueError, match='LRU=1, LRF=7'):
        rml_data_from_endf_dict(d, isotope_idx=1, range_idx=2)
