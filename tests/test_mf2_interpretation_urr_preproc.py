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
# Basic reconstruction sanity on synthetic dicts.
# ============================================================


def test_reconstruct_returns_finite_nonnegative_on_synthetic():
    """Baseline sanity: every partial (sct / cap / fis / rxx /
    pot / tot) is finite and non-negative at a handful of query
    energies inside the URR range."""
    d = _minimal_urr_endf_dict()
    data = pre.urr_data_from_endf_dict(d)
    from endf_userpy.primitives import array_ns
    xp = array_ns.get_backend('numpy')
    xs = urr.reconstruct(data, np.array([1e3, 5e3, 1e4]), xp)
    for k in ('sct', 'cap', 'fis', 'rxx', 'pot', 'tot'):
        arr = np.asarray(xs[k])
        assert np.all(np.isfinite(arr)), f'{k} has non-finite'
        assert np.all(arr >= 0.0), f'{k} went negative'


def test_reconstruct_accepts_int5_log_log():
    """INT=5 (log-log) is supported alongside INT=2 (lin-lin).
    Rows whose y values are all positive are interpolated in
    log-log space; rows with any non-positive y (typical: GF
    all-zero on a non-fissile group) silently fall back to
    lin-lin so the interpolation stays well defined."""
    d = _minimal_urr_endf_dict(intp=5)   # log-log
    data = pre.urr_data_from_endf_dict(d)
    from endf_userpy.primitives import array_ns
    xp = array_ns.get_backend('numpy')
    xs = urr.reconstruct(data, np.array([5e3]), xp)
    assert np.all(np.isfinite(np.asarray(xs['tot'])))


def test_reconstruct_rejects_unsupported_int_codes():
    """Non-{2,5} INT codes are still rejected up front so a caller
    gets a clear NotImplementedError rather than a silently
    mis-interpolated width."""
    d = _minimal_urr_endf_dict(intp=3)   # lin-log; not implemented
    data = pre.urr_data_from_endf_dict(d)
    from endf_userpy.primitives import array_ns
    xp = array_ns.get_backend('numpy')
    with pytest.raises(NotImplementedError, match=r'INT=2 \(lin-lin\) and INT=5'):
        urr.reconstruct(data, np.array([5e3]), xp)


def test_u235_reconstruction_plausible_vs_tabulated_mf3():
    """TENDL-2021 U-235 has LSSF=1 URR, so MF3 in the URR range
    IS the evaluator-processed average XS. Our reconstruction
    from the raw URR parameters should be in the same physical
    ballpark (within ~25%), giving a smoke check that the
    formulas are the right shape.

    Not a strict-NJOY-parity test: MF3 was produced by
    whatever URR processor TENDL used, with its own
    tabulated-fluctuation-factor conventions, DOF handling
    and possibly a resonance-potential interference term this
    kernel does not yet include. Sub-percent parity is a
    follow-up: run NJOY unresr on the same file's URR params
    (via scripts/njoy_compare/) and compare bit-for-bit,
    independent of MF3.

    What this test does catch: order-of-magnitude physics
    errors (unit slips, missing L-summation, wrong
    penetration-factor formula, etc.). If any partial drifts
    outside a factor of 1.25 vs MF3, physics is wrong somewhere,
    not just a small formula variant.
    """
    path = resolve_u235()
    if path is None:
        pytest.skip('U-235 corpus file not available')
    from endf_parserpy import EndfParserCpp
    from endf_userpy.mfsec_interpretation import mf3_interpretation as mf3
    from endf_userpy.primitives import array_ns
    d = EndfParserCpp().parsefile(path, include=[1, 2, 3])
    data = pre.urr_data_from_endf_dict(d)
    xp = array_ns.get_backend('numpy')

    einc = np.array([2500., 5000., 10000., 20000., 40000.])
    ours = urr.reconstruct(data, einc, xp)

    for mt, key in [(2, 'sct'), (18, 'fis'), (102, 'cap')]:
        ours_x = np.asarray(ours[key])
        mf3_x = np.asarray(mf3.compute_cross_section_agnostic(d, mt, einc, xp))
        rel = np.abs(ours_x - mf3_x) / mf3_x
        assert np.all(rel < 0.25), (
            f'{key} (MT={mt}): max relative error {rel.max():.3g} '
            f'exceeds 25% vs tabulated MF3 -- likely a physics bug '
            f'rather than a formula-variant difference; '
            f'ours={ours_x} vs MF3={mf3_x}'
        )


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


# ============================================================
# JAX autodiff: pinning test that jax.grad flows through the URR
# reconstruction wrt an average-width field. Enabling capability
# for autodiff-driven URR fitting.
# ============================================================


def _jax_available():
    from endf_userpy.primitives import array_ns
    return 'jax' in array_ns.available_backends()


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_flows_through_urr_wrt_gamma_gamma():
    """``jax.grad`` of the capture cross section wrt a scalar
    scaling of the tabulated Gamma_gamma table returns a finite,
    non-zero gradient with the right sign (positive: increasing
    capture width increases capture XS).

    Proves the JAX path is fully differentiable end-to-end
    (interpolation + factors.pnt_shf + fluctuation integrals +
    per-group sum), which is the enabling capability the user
    called out for automated URR fitting.
    """
    import dataclasses
    import jax
    import jax.numpy as jnp
    from endf_userpy.primitives import array_ns

    xp = array_ns.get_backend('jax')
    d = _minimal_urr_endf_dict(
        j_groups=[(0, [(3.5, 1.0, 0.5, 1.0, 0.0,
                        [1e3, 1e4], [1.0, 1.0],
                        [0.1, 0.1], [0.05, 0.05],
                        [0.0, 0.0], [0.0, 0.0])])],
    )
    data = pre.urr_data_from_endf_dict(d)

    einc = jnp.array([2.5e3, 5e3])

    def capture_from_scale(gg_scale):
        """Sum of capture XS at the two query energies, as a
        function of a scalar scaling applied to the entire
        Gamma_gamma table. Autodiff sees this whole chain."""
        gg_scaled = jnp.asarray(data.table_gg) * gg_scale
        data_s = dataclasses.replace(data, table_gg=gg_scaled)
        xs = urr.reconstruct(data_s, einc, xp)
        return jnp.sum(xs['cap'])

    val = float(capture_from_scale(1.0))
    grad = float(jax.grad(capture_from_scale)(1.0))

    assert np.isfinite(val), f'value not finite: {val}'
    assert val > 0.0, f'capture at scale=1 should be positive; got {val}'
    assert np.isfinite(grad), f'gradient not finite: {grad}'
    # Sign check: scaling Γ_γ up should increase Γ_n Γ_γ / Γ
    # (numerator grows faster than denominator at small Γ_γ),
    # so d capture / d gg_scale > 0.
    assert grad > 0.0, (
        f'expected positive gradient wrt Gamma_gamma scale; got {grad}. '
        f'Autodiff may not be flowing through the fluctuation integral '
        f'or through the tables interpolation.'
    )
    # Finite-difference sanity: gradient magnitude should match
    # (capture(1 + eps) - capture(1)) / eps within a few percent.
    eps = 1e-3
    val_p = float(capture_from_scale(1.0 + eps))
    fd = (val_p - val) / eps
    rel = abs(grad - fd) / abs(fd)
    assert rel < 0.05, (
        f'autodiff gradient {grad} vs finite-difference {fd}: '
        f'{rel:.3%} relative disagreement (expected < 5%). '
        f'A serious mismatch here indicates the autodiff graph is '
        f'not connected the way the physics is.'
    )


# ============================================================
# Numba backend parity: identical numerics to the numpy path.
# ============================================================


def _numba_available():
    from endf_userpy.primitives import array_ns
    return 'numba' in array_ns.available_backends()


@pytest.mark.skipif(not _numba_available(), reason='numba not installed')
def test_urr_numpy_numba_agree_on_synthetic():
    """URR numba kernel must reproduce the numpy path on the
    same synthetic dict at machine precision (both paths use
    the same Gauss-Legendre nodes and the same physics
    formulas)."""
    from endf_userpy.primitives import array_ns
    d = _minimal_urr_endf_dict(
        j_groups=[(0, [(3.5, 1.0, 0.5, 1.0, 0.0,
                        [1e3, 1e4], [1.0, 1.0],
                        [0.1, 0.1], [0.05, 0.05],
                        [0.03, 0.03], [0.0, 0.0])])],
    )
    data = pre.urr_data_from_endf_dict(d)
    einc = np.array([1.5e3, 3e3, 5e3, 8e3])
    xs_np = urr.reconstruct(data, einc, array_ns.get_backend('numpy'))
    xs_nb = urr.reconstruct(data, einc, array_ns.get_backend('numba'))
    for k in ('sct', 'cap', 'fis', 'rxx', 'pot', 'tot'):
        np.testing.assert_allclose(
            np.asarray(xs_np[k]), np.asarray(xs_nb[k]),
            rtol=1e-12, atol=1e-30,
            err_msg=f'URR numpy vs numba disagree on {k}',
        )


@pytest.mark.skipif(not _numba_available(), reason='numba not installed')
def test_urr_numpy_numba_agree_on_u235():
    """Numpy vs numba on the U-235 URR range: max relative error
    at the ~1e-15 level across every partial (both paths run
    the same physics; only the outer loop differs)."""
    path = resolve_u235()
    if path is None:
        pytest.skip('U-235 corpus file not available')
    from endf_parserpy import EndfParserCpp
    from endf_userpy.primitives import array_ns
    d = EndfParserCpp().parsefile(path, include=[1, 2])
    data = pre.urr_data_from_endf_dict(d)
    einc = np.array([2500., 5000., 10000., 20000., 40000.])
    xs_np = urr.reconstruct(data, einc, array_ns.get_backend('numpy'))
    xs_nb = urr.reconstruct(data, einc, array_ns.get_backend('numba'))
    for k in ('sct', 'cap', 'fis', 'rxx', 'pot', 'tot'):
        np.testing.assert_allclose(
            np.asarray(xs_np[k]), np.asarray(xs_nb[k]),
            rtol=1e-12, atol=1e-30,
            err_msg=f'URR numpy vs numba on U-235 disagree on {k}',
        )


# ============================================================
# NJOY-parity: reconstruct vs NJOY unresr's tabulated MT152.
# ============================================================


def test_reconstruct_close_to_njoy_unresr_u235():
    """``reconstruct`` (direct 1D Laplace-Gauss-Legendre) agrees
    with NJOY unresr's tabulated MT152 (infinite dilution,
    T = 0) to a few 1e-4 relative on capture and fission,
    ~1e-5 on elastic, across the full TENDL-2021 U-235 URR
    range. The small residual is NJOY's Ross-10-point
    quadrature error, not ours (verified against Monte Carlo:
    our integrand converges to the true chi-squared average to
    ~1e-5, NJOY has ~1e-4 systematic offset from that value at
    Porter-Thomas parameters).

    Reference values transcribed from a NJOY unresr run on
    TENDL-2021 U-235 (see ``scripts/njoy_compare/run_unresr.sh``);
    encoded here as literals so the test works without NJOY
    installed in CI.
    """
    path = resolve_u235()
    if path is None:
        pytest.skip('U-235 corpus file not available')
    from endf_parserpy import EndfParserCpp
    from endf_userpy.primitives import array_ns
    d = EndfParserCpp().parsefile(path, include=[1, 2])
    data = pre.urr_data_from_endf_dict(d)
    xp = array_ns.get_backend('numpy')

    # (E, el, fis, cap) from NJOY unresr MT152 tape22, sig0 = 1e10,
    # T = 0 (NJOY floors to 1 K). 7-digit precision from tape22.
    njoy = [
        (2250.0,   11.9542, 5.9600, 2.3881),
        (2500.0,   11.9309, 5.6553, 2.2547),
        (3000.0,   11.8877, 5.1670, 2.0415),
        (5000.0,   11.7485, 4.0425, 1.5528),
        (10000.0,  11.4989, 2.9701, 1.0919),
        (20000.0,  11.1531, 2.2866, 0.7961),
        (40000.0,  10.6822, 1.8784, 0.6107),
        (46200.0,  10.5668, 1.8175, 0.5813),
    ]
    einc = np.array([row[0] for row in njoy])
    xs = urr.reconstruct(data, einc, xp)

    tolerances = {'el': 1e-4, 'fis': 1e-3, 'cap': 1e-3}
    for i, (E, n_el, n_fis, n_cap) in enumerate(njoy):
        for label, ours, njoy_val in [
            ('el', float(xs['sct'][i]), n_el),
            ('fis', float(xs['fis'][i]), n_fis),
            ('cap', float(xs['cap'][i]), n_cap),
        ]:
            rel = abs(ours - njoy_val) / njoy_val
            assert rel < tolerances[label], (
                f'E={E} eV: {label} = {ours:.6f} vs NJOY {njoy_val} '
                f'({rel:.3%} relative -- expected < '
                f'{tolerances[label]:.0e})'
            )
