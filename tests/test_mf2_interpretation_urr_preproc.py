"""Preproc tests for :mod:`mf2_interpretation_urr_preproc`.

Same pattern as the MLBW / R-M preproc tests: hand-built synthetic
ENDF-dict fixtures for field-level correctness, plus one U-235
structural round-trip (skipped if the corpus file is absent).
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_userpy.mfsec_interpretation import mf2_interpretation_urr as urr
from endf_userpy.mfsec_interpretation import (
    mf2_interpretation_urr_preproc as pre,
)

from _corpus import resolve_u235


# ============================================================
# Synthetic dict fixture.
# ============================================================


def _minimal_urr_endf_dict(
    nsub=10,           # neutron incident
    spi=3.5, ap=0.6, naps=0, awri=232.0,
    j_groups=None,     # list of (L, [(aj, amun, amug, amuf, amux, es_arr,
                       #               d_arr, gn0_arr, gg_arr, gf_arr,
                       #               gx_arr), ...])
    intp=2,            # lin-lin
):
    """Build a minimal parsed-ENDF-6-style dict with one MF2 URR
    range at the standard layout (LRU=2 LRF=2 at range 2, with a
    dummy LRU=1 LRF=2 range at range 1 to satisfy the standard
    two-range shape).
    """
    if j_groups is None:
        # Default: one L-group, one J-group, NE=2 with constant widths.
        j_groups = [(0, [(3.5, 1.0, 0.0, 1.0, 1.0,
                          [1e3, 1e4], [1.0, 1.0],
                          [0.1, 0.1], [0.05, 0.05],
                          [0.0, 0.0], [0.0, 0.0])])]

    # Group by L (all J's under one L go into one l_group entry).
    l_group = {}
    for l_i, (L, jlist) in enumerate(j_groups, start=1):
        j_entries = {}
        for j_idx, (aj, amun, amug, amuf, amux,
                    es, d, gn0, gg, gf, gx) in enumerate(jlist, start=1):
            ne = len(es)
            j_entries[j_idx] = {
                'AJ': aj,
                'AMUN': amun, 'AMUG': amug,
                'AMUF': amuf, 'AMUX': amux,
                'NE': ne, 'INT': intp,
                'ES': {i + 1: es[i] for i in range(ne)},
                'D':  {i + 1: d[i]  for i in range(ne)},
                'GN0': {i + 1: gn0[i] for i in range(ne)},
                'GG':  {i + 1: gg[i]  for i in range(ne)},
                'GF':  {i + 1: gf[i]  for i in range(ne)},
                'GX':  {i + 1: gx[i]  for i in range(ne)},
            }
        l_group[l_i] = {
            'L': L, 'AWRI': awri, 'NJS': len(jlist),
            'j_group': j_entries,
        }

    return {
        1: {451: {'NSUB': nsub}},
        2: {151: {'isotope': {1: {
            'ABN': 1.0,
            'range': {
                # Range 1: minimal LRU=1 shell (not touched by URR preproc).
                1: {
                    'LRU': 1, 'LRF': 2, 'NAPS': 0, 'NRO': 0,
                    'SPI': spi, 'AP': ap, 'NLS': 0,
                    'EL': 1e-5, 'EH': 1e3, 'LAD': 0,
                    'l_group': {},
                },
                # Range 2: the URR range under test.
                2: {
                    'LRU': 2, 'LRF': 2, 'LSSF': 1,
                    'NAPS': naps, 'NRO': 0,
                    'SPI': spi, 'AP': ap,
                    'NLS': len(j_groups),
                    'EL': 1e3, 'EH': 1e5,
                    'l_group': l_group,
                },
            },
        }}}},
    }


# ============================================================
# Preproc: field-level correctness on a synthetic dict.
# ============================================================


def test_wrong_lrf_raises():
    """LRU=2 LRF=1 (Case A, constant widths) is out of scope for
    this preproc; caller should get a clear message and not
    silently produce garbage."""
    d = _minimal_urr_endf_dict()
    d[2][151]['isotope'][1]['range'][2]['LRF'] = 1
    with pytest.raises(ValueError, match=r'LRU=2 LRF=2'):
        pre.urr_data_from_endf_dict(d)


def test_single_group_scalars_carry_through():
    d = _minimal_urr_endf_dict()
    data = pre.urr_data_from_endf_dict(d)
    assert data.group_l.shape == (1,)
    assert int(data.group_l[0]) == 0
    assert int(data.group_j2[0]) == 7          # 2 * 3.5
    assert float(data.group_amun[0]) == 1.0
    assert float(data.group_amuf[0]) == 1.0
    assert float(data.group_amug[0]) == 0.0
    assert int(data.group_int[0]) == 2
    # Statistical weight g_J = (2J + 1) / (2 (2I + 1)) with I=3.5,
    # J=3.5 -> (8) / (2 * 8) = 0.5.
    assert float(data.group_g[0]) == pytest.approx(0.5)


def test_scalar_fields_are_array_shaped_not_python_float():
    """URRData follows the same pytree-shaped-scalar convention
    as MLBWData / RMData: 0-d numpy arrays, not Python floats."""
    d = _minimal_urr_endf_dict()
    data = pre.urr_data_from_endf_dict(d)
    for field_name in ('abn', 'spi', 'ap', 'awri', 'ki'):
        value = getattr(data, field_name)
        assert not isinstance(value, float), (
            f'URRData.{field_name} is a Python float ({value!r}); '
            f'expected numpy scalar / 0-d array for JAX '
            f'substitution.'
        )


def test_rectangular_width_tables_have_expected_shape():
    j0 = (3.5, 1.0, 0.0, 1.0, 1.0,
          [1e3, 3e3, 1e4], [1.5, 1.6, 1.7],
          [0.1, 0.11, 0.12], [0.05, 0.052, 0.054],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    j1 = (2.5, 1.0, 0.0, 2.0, 1.0,
          [1e3, 3e3, 1e4], [1.9, 2.0, 2.1],
          [0.08, 0.09, 0.10], [0.06, 0.062, 0.064],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    d = _minimal_urr_endf_dict(j_groups=[(0, [j0]), (1, [j1])])
    data = pre.urr_data_from_endf_dict(d)
    assert data.table_es.shape == (2, 3), (
        f'table_es shape {data.table_es.shape}, expected (2, 3)'
    )
    np.testing.assert_array_equal(data.table_es[0], [1e3, 3e3, 1e4])
    np.testing.assert_array_equal(data.table_gn0[1], [0.08, 0.09, 0.10])


def test_variable_ne_across_groups_raises_clearly():
    j0 = (3.5, 1.0, 0.0, 1.0, 1.0,
          [1e3, 1e4], [1.5, 1.7],
          [0.1, 0.12], [0.05, 0.054],
          [0.0, 0.0], [0.0, 0.0])
    j1 = (2.5, 1.0, 0.0, 1.0, 1.0,
          [1e3, 3e3, 1e4], [1.9, 2.0, 2.1],
          [0.08, 0.09, 0.10], [0.06, 0.062, 0.064],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    d = _minimal_urr_endf_dict(j_groups=[(0, [j0, j1])])
    with pytest.raises(ValueError, match=r'NE'):
        pre.urr_data_from_endf_dict(d)


# ============================================================
# Reconstruction wrapper stub raises the "not implemented" flag
# until the kernel lands in the next commit on this branch.
# ============================================================


def test_reconstruct_stub_raises_clearly_pending_kernel():
    """The kernel arrives in the next commit; until then a caller
    that hits :func:`reconstruct` gets a clear NotImplementedError
    naming the follow-up rather than a silent bad number."""
    d = _minimal_urr_endf_dict()
    data = pre.urr_data_from_endf_dict(d)
    from endf_userpy.primitives import array_ns
    xp = array_ns.get_backend('numpy')
    with pytest.raises(NotImplementedError, match=r'kernel not yet implemented'):
        urr.reconstruct(data, np.array([5e3]), xp)


# ============================================================
# U-235 (TENDL-2021) structural round-trip: 6 spin groups
# (2 L=0 + 4 L=1), NE=14 per group, shared ES grid.
# ============================================================


@pytest.fixture
def u235_urr_data():
    path = resolve_u235()
    if path is None:
        pytest.skip(
            'U-235 corpus file not available (run '
            'tests/data_law1_adhoc/fetch.sh)'
        )
    from endf_parserpy import EndfParserCpp
    d = EndfParserCpp().parsefile(path, include=[1, 2])
    return pre.urr_data_from_endf_dict(d)


def test_u235_urr_structural_counts(u235_urr_data):
    """TENDL-2021 U-235 URR: 2 L=0 spin groups + 4 L=1 spin groups
    = 6 groups; NE=14 per group; energy range [2.25 keV, 46.2 keV]."""
    data = u235_urr_data
    assert data.group_l.shape == (6,), (
        f'expected 6 spin groups, got shape {data.group_l.shape}'
    )
    assert int((data.group_l == 0).sum()) == 2
    assert int((data.group_l == 1).sum()) == 4
    assert data.table_es.shape == (6, 14)
    for row in data.table_es:
        assert row[0] == pytest.approx(2250.0)
        assert row[-1] == pytest.approx(46200.0)
    # AWRI ~= 233 for U-235 in TENDL.
    assert 232.0 < float(data.awri) < 234.0


def test_u235_urr_dof_values_match_file(u235_urr_data):
    """TENDL-2021 U-235 URR: AMUN=1 everywhere; AMUG=0 (folded
    into the Reich-Moore-style capture channel); AMUF is a
    per-group DOF (1 for two of the L=1 groups, 2 for two of
    them, 1 for both L=0 groups)."""
    data = u235_urr_data
    np.testing.assert_array_equal(data.group_amun,
                                   np.ones(6, dtype=np.float64))
    np.testing.assert_array_equal(data.group_amug,
                                   np.zeros(6, dtype=np.float64))
    # Set of unique AMUF values across the 6 groups.
    assert set(map(int, np.unique(data.group_amuf))) == {1, 2}
