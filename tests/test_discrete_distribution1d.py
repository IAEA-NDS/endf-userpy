"""Tests for the WIP module `quantities_mt_zap.discrete_distribution1d`.

Currently only pins the fix for issue #44: the MF13 branch of
`compute_energydist_values` used to normalise the per-line yields
matrix along the wrong axis:

    yields_sum = np.sum(yields, axis=1)           # (n_einc,)
    energy_dist_values = yields / yields_sum.reshape(1, -1)   # BUG

`yields_sum` is per incident energy, so the reshape must be
`(-1, 1)` for `yields / yields_sum.reshape(-1, 1)` to divide each
row by its own sum. `reshape(1, -1)` produces `(1, n_einc)`, which
either raises a broadcast error when `n_lines != n_einc` or, if
the two happen to be equal, silently normalises by the wrong row.

The module is currently a WIP sketch and not wired into any public
entry point (issue #44 flagged it as a latent bug that would surface
the moment the MF13 path was wired into `get_particle_production_dxs_dE`
for gamma). The tests here inject the reader outputs via
`monkeypatch` rather than constructing a full synthetic ENDF dict,
which keeps the test focused on the shape / normalisation
arithmetic that this issue is about.
"""
import numpy as np
import pytest

from endf_userpy.quantities_mt_zap import discrete_distribution1d as ddist


G_ZAP = 0.0  # get_zap_for_particle('g')


def _patch_readers(monkeypatch, prodxs, xs):
    """Replace the two readers `compute_energydist_values` calls in
    its MF13 branch with fixed-array stand-ins. Also stub the
    `has_*_mt` predicates so the function takes the MF13 branch."""
    monkeypatch.setattr(ddist, 'has_mf12_mt', lambda d, m: False)
    monkeypatch.setattr(ddist, 'has_mf13_mt', lambda d, m: True)
    monkeypatch.setattr(
        ddist.mf13_interp, 'compute_photon_production_xs',
        lambda endf_dict, mt, eincs, pes: prodxs,
    )
    monkeypatch.setattr(
        ddist.mf3_interp, 'compute_cross_section',
        lambda endf_dict, mt, eincs: xs,
    )


def test_mf13_branch_shape_survives_unequal_einc_lines(monkeypatch):
    """The primary defect: n_lines != n_einc. Before the fix this
    raised a numpy broadcasting error at
    `yields / yields_sum.reshape(1, -1)`. After the fix the shapes
    line up and the function returns a well-formed (n_einc, n_lines)
    array."""
    n_einc, n_lines = 4, 3
    xs = np.full(n_einc, 2.0)                                       # (n_einc,)
    prodxs = np.arange(1.0, 1.0 + n_einc * n_lines).reshape(n_einc, n_lines)
    _patch_readers(monkeypatch, prodxs, xs)

    eincs = np.linspace(1e6, 5e6, n_einc)
    pes = np.array([1.0e5, 5.0e5, 9.0e5])
    r = ddist.compute_energydist_values(
        endf_dict={}, mt=51, zap=G_ZAP,
        energies_in=eincs, discrete_energies_out=pes,
    )
    assert r.shape == (n_einc, n_lines)
    assert not np.any(np.isnan(r))


def test_mf13_branch_rows_sum_to_one(monkeypatch):
    """Each row (incident energy) must sum to exactly 1 after the
    per-row normalisation. Before the fix the same call would either
    raise or, if `n_lines == n_einc`, produce rows that summed to
    various nonsense values because each entry was normalised by
    the wrong row's yield_sum."""
    n_einc, n_lines = 5, 7
    xs = np.linspace(0.5, 2.5, n_einc)
    # Non-uniform prodxs across (einc, line) so rows are genuinely
    # different -- rules out the accidental "sum is 1 anyway" case.
    prodxs = np.outer(np.arange(1, n_einc + 1), np.arange(1, n_lines + 1))
    prodxs = prodxs.astype(float)
    _patch_readers(monkeypatch, prodxs, xs)

    r = ddist.compute_energydist_values(
        endf_dict={}, mt=51, zap=G_ZAP,
        energies_in=np.linspace(1e6, 2e6, n_einc),
        discrete_energies_out=np.linspace(1e5, 9e5, n_lines),
    )
    row_sums = r.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-12)


def test_mf13_branch_equal_dims_normalises_by_correct_row(monkeypatch):
    """The subtle-silent case: n_lines == n_einc. Before the fix the
    reshape gave (1, n_einc) which broadcast fine but normalised each
    column by the corresponding row's yield_sum rather than each row
    by its own sum, so `r[i, :]` did not sum to 1 in general. After
    the fix, each row sums to 1."""
    n = 4
    xs = np.full(n, 1.0)
    # Deliberately asymmetric so wrong-axis normalisation would be
    # visibly wrong.
    prodxs = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.1, 0.2, 0.3, 0.4],
    ])
    _patch_readers(monkeypatch, prodxs, xs)

    r = ddist.compute_energydist_values(
        endf_dict={}, mt=51, zap=G_ZAP,
        energies_in=np.linspace(1e6, 4e6, n),
        discrete_energies_out=np.linspace(1e5, 4e5, n),
    )
    # Rows must sum to 1
    np.testing.assert_allclose(r.sum(axis=1), 1.0, rtol=1e-12)
    # Sanity: each row equals prodxs[i] / prodxs[i].sum() (since
    # xs is 1 here, yields == prodxs).
    for i in range(n):
        np.testing.assert_allclose(
            r[i, :], prodxs[i] / prodxs[i].sum(), rtol=1e-12,
        )


def test_non_gamma_zap_rejected():
    """Sanity: the function still refuses non-gamma ZAPs with a
    NotImplementedError. Unrelated to the axis fix; anchored here
    for one-file completeness."""
    with pytest.raises(NotImplementedError):
        ddist.compute_energydist_values(
            endf_dict={}, mt=51, zap=1.0,  # neutron
            energies_in=np.array([1e6]),
            discrete_energies_out=np.array([1e5]),
        )
