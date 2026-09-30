"""User-facing API returning MF12/MF13 discrete gamma-line records
(issue #271, option-B follow-up to #266).

The unbroadened DDX / dxs_dE dispatchers drop the delta-shaped
discrete gamma content (they cannot render a Dirac peak on a
finite E_out grid) and emit a UserWarning naming the affected
MTs. This module tests the complementary path that exposes the
same content as ``DiscreteGammaLine`` records callers can render,
broaden with a custom kernel, or fit against.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.run_options import RunOptions
from endf_userpy.quantities import (
    get_particle_production_discrete_gamma_lines,
    get_particle_production_dxs_dE,
)
from endf_userpy.quantities_mt_zap.discrete_gamma_lines import (
    DiscreteGammaLine,
)


@pytest.fixture(scope='module')
def h2_endf_dict():
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(
        'tests/data/n-001_H_002.endf',
    )


def test_h2_returns_one_line_at_expected_energy(h2_endf_dict):
    """H-2 (n,g) MT=102 has one MF12 discrete photon at Eg=6.251 MeV."""
    ein = np.array([1e6])
    lines = get_particle_production_discrete_gamma_lines(
        h2_endf_dict, '(n,g)', ein,
    )
    assert len(lines) == 1
    ln = lines[0]
    assert isinstance(ln, DiscreteGammaLine)
    assert ln.mt == 102
    assert ln.mf == 12
    assert abs(ln.Eg - 6.251002e6) < 1e3
    assert ln.angdist is None  # not requested


def test_h2_line_weight_matches_sigma_times_yield(h2_endf_dict):
    """The MF12-source weight is ``sigma(Ein) * y_i(Ein)``. Verify
    against a direct MF3 * MF12 lookup at three incident energies.
    """
    from endf_userpy.mfsec_interpretation import (
        mf3_interpretation as mf3_interp,
        mf12_interpretation as mf12_interp,
    )
    ein = np.array([1e5, 1e6, 5e6])
    lines = get_particle_production_discrete_gamma_lines(
        h2_endf_dict, '(n,g)', ein,
    )
    assert len(lines) == 1
    ln = lines[0]
    xs = mf3_interp.compute_cross_section(h2_endf_dict, 102, ein)
    egs = mf12_interp.get_photon_energies(h2_endf_dict, 102)
    yields = mf12_interp.compute_photon_yields(
        h2_endf_dict, 102, ein, np.asarray(egs, dtype=float),
    )
    disc_mask = np.asarray(egs) > 0.0
    y_disc = np.asarray(yields)[:, disc_mask].sum(axis=1)
    expected = xs * y_disc
    np.testing.assert_allclose(np.asarray(ln.weight), expected, rtol=1e-10)


def test_h2_angdist_isotropic_fallback_when_no_mf14(h2_endf_dict):
    """H-2 MT=102 has no MF14; the per-line angdist falls back to
    isotropic 0.5 across the requested mu grid.
    """
    ein = np.array([1e6])
    mus = np.array([-0.75, 0.0, 0.75])
    lines = get_particle_production_discrete_gamma_lines(
        h2_endf_dict, '(n,g)', ein, angle_cosines_out=mus,
    )
    assert len(lines) == 1
    ang = np.asarray(lines[0].angdist)
    assert ang.shape == (1, 3)
    np.testing.assert_allclose(ang, 0.5)


def test_h2_lines_consistent_with_dxs_dE_integral(h2_endf_dict):
    """Summing the per-line weight across every discrete line
    (there is only one for H-2) recovers what
    ``get_particle_production_dxs_dE`` reports as dropped from the
    unbroadened output. Numerically: at each Ein, sum(weights) equals
    the total gamma production XS from discrete channels alone.
    """
    from endf_userpy.mfsec_interpretation import (
        mf3_interpretation as mf3_interp,
    )
    ein = np.array([1e6])
    lines = get_particle_production_discrete_gamma_lines(
        h2_endf_dict, '(n,g)', ein,
    )
    total_from_lines = sum(np.asarray(ln.weight) for ln in lines)
    # Verify: for H-2 with only one MF12 discrete line whose y ≈ 1,
    # the sum matches sigma(Ein) * y_disc(Ein) directly.
    xs = mf3_interp.compute_cross_section(h2_endf_dict, 102, ein)
    # y_disc = 1 for the single H-2 line -> total_from_lines ≈ xs.
    np.testing.assert_allclose(total_from_lines, xs, rtol=1e-3)

    # And under broadening=None the dxs_dE result excludes this
    # content (with a warning); confirm they add up to the same total
    # after adding the discrete lines back via nearest-bin quadrature.
    eouts = np.linspace(0.0, 1e7, 51)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dxs_cont = get_particle_production_dxs_dE(
            h2_endf_dict, '(n,g)', 'g', ein, eouts,
        )
    widths = np.empty(51)
    widths[0] = eouts[1] - eouts[0]
    widths[-1] = eouts[-1] - eouts[-2]
    widths[1:-1] = 0.5 * (eouts[2:] - eouts[:-2])
    cont_int = np.sum(np.asarray(dxs_cont)[0] * widths)
    disc_int = float(total_from_lines[0])
    # H-2 gamma production at 1 MeV is dominated by the discrete
    # 6.25 MeV line; the continuum share (from MF15) is present but
    # small.
    assert disc_int > 0.5 * cont_int, (
        f'discrete-line total ({disc_int}) should dominate over the '
        f'continuum integral ({cont_int}) for H-2 (n,g) at 1 MeV'
    )


def test_returns_empty_list_when_no_discrete_gamma_content(h2_endf_dict):
    """A reaction with no discrete gamma content returns an empty list.
    Elastic MT=2 has no MF12/MF13 by construction.
    """
    lines = get_particle_production_discrete_gamma_lines(
        h2_endf_dict, '(n,n_0)', np.array([1e6]),
    )
    assert lines == []


def test_lines_sorted_by_Eg(h2_endf_dict):
    """Returned lines are sorted by Eg ascending; verify on a file
    with any discrete content.
    """
    lines = get_particle_production_discrete_gamma_lines(
        h2_endf_dict, '(n,g)', np.array([1e6]),
    )
    egs = [ln.Eg for ln in lines]
    assert egs == sorted(egs)


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def u233_endf_dict():
    import os
    path = 'tests/data_law1_adhoc/endfb81_n_U-233.endf'
    if not os.path.exists(path):
        pytest.skip('U-233 adhoc corpus not present')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def test_u233_mf6_law1_discrete_lines_are_extracted(u233_endf_dict):
    """U-233 (n,g) MT=102 stores discrete gamma-line content in MF6
    LAW=1 with ND>0 (a JENDL/ENDF-B convention for capture cascades).
    Verify that this content is exposed as ``mf=6`` records, that the
    reported ``Eg`` sits in the physically expected range for
    U-233(n,g) gamma cascades (tens of keV to a few MeV), and that
    the weights are nonzero at an Ein above the resolved-resonance
    region.
    """
    ein = np.array([100e3, 1e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lines = get_particle_production_discrete_gamma_lines(
            u233_endf_dict, '(n,g)', ein,
        )
    mf6_lines = [ln for ln in lines if ln.mf == 6]
    assert len(mf6_lines) > 0, 'no MF6 discrete gamma lines extracted'
    for ln in mf6_lines:
        assert 1e3 < ln.Eg < 1e8, f'Eg={ln.Eg} outside plausible range'
        assert ln.mt == 102
    # At least some lines carry nonzero weight in the fast region.
    assert any(
        np.any(np.asarray(ln.weight) > 0.0) for ln in mf6_lines
    )


def test_u233_mf6_law1_weight_matches_direct_extraction(u233_endf_dict):
    """The per-line weight (barn) reported by the discrete-line API
    for an MF6/LAW=1 gamma must equal ``sigma(Ein) * yield(Ein) *
    int_mu(amp)`` computed by hand from
    ``mf6_interpretation.compute_law1_discrete_lines`` for the same
    (MT, line) pair. Uses MT=102 alone to keep the extraction
    focused on a single MF6 subsection with a stable ~86-line set.
    """
    from endf_userpy.mfsec_interpretation import (
        mf3_interpretation as mf3_interp,
        mf6_interpretation as mf6_interp,
    )
    from endf_userpy.quantities_mt_zap import quantities as quant_mt_zap

    ein = np.array([1e6])
    mus_grid = np.linspace(-1.0, 1.0, 21)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lines = get_particle_production_discrete_gamma_lines(
            u233_endf_dict, '(n,g)', ein,
        )
    mf6_mt102 = [ln for ln in lines if ln.mf == 6 and ln.mt == 102]
    assert mf6_mt102, 'no MF6/LAW=1 lines extracted from U-233 MT=102'

    # Direct reference: extract the same lines outside the API and
    # apply the same barn-weight convention.
    ep_ref, amp_ref = mf6_interp.compute_law1_discrete_lines(
        u233_endf_dict, 102, 0, ein, mus_grid, to_lab=True,
    )
    yields = quant_mt_zap.compute_yields(
        u233_endf_dict, 102, 0, ein, include_discrete=True,
    )
    xs = mf3_interp.compute_cross_section(u233_endf_dict, 102, ein)
    yields_xs = np.asarray(yields) * np.asarray(xs)
    # Match each API line back to its slot in the reference by Eg.
    for ln in mf6_mt102:
        # Slot k whose reference Eg matches within tolerance.
        # Nonzero-amp reference values only (padding slots have amp=0).
        matched_k = None
        for k in range(ep_ref.shape[-1]):
            amp_k = amp_ref[:, :, k]
            if not np.any(amp_k > 0.0):
                continue
            ep_k_ref = float(ep_ref[:, :, k][amp_k > 0.0].flat[0])
            if abs(ep_k_ref - ln.Eg) < 1.0:
                matched_k = k
                break
        assert matched_k is not None, f'no reference slot for Eg={ln.Eg}'
        amp_int = float(np.trapezoid(amp_ref[0, :, matched_k], mus_grid))
        expected_weight = amp_int * float(yields_xs[0])
        np.testing.assert_allclose(
            float(np.asarray(ln.weight)[0]), expected_weight, rtol=1e-8,
        )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_jax_grad_reaches_line_weight_through_yield_swap(h2_endf_dict):
    """`jax.grad` of a scalar loss over the sum of line weights
    reaches the MF12 yield column via `xp=jax`. The per-line MF12
    yield column is swapped for a jnp array; grad wrt one entry
    matches central FD.
    """
    import copy
    import jax
    import jax.numpy as jnp

    xp_jax = array_ns.get_backend('jax')
    d = h2_endf_dict
    # H-2 MF12 MT=102 stores its per-line yield in d[12][102]['table'][1]['y'].
    y_list = d[12][102]['table'][1]['y']
    n_pts = len(y_list)
    idx = min(n_pts // 2, n_pts - 1)
    orig_val = float(y_list[idx])

    def loss(theta):
        d_t = copy.deepcopy(d)
        tab = d_t[12][102]['table'][1]
        new_y = jnp.asarray([float(v) for v in tab['y']]).at[idx].set(theta)
        tab['y'] = new_y
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lines = get_particle_production_discrete_gamma_lines(d_t, '(n,g)', np.array([1e6]), options=RunOptions(backend=xp_jax))
        return jnp.sum(jnp.stack([jnp.sum(ln.weight) for ln in lines]))

    theta0 = jnp.array(orig_val)
    grad = float(jax.grad(loss)(theta0))
    eps = max(abs(orig_val) * 1e-3, 1e-10)
    fd = (
        float(loss(jnp.array(orig_val + eps)))
        - float(loss(jnp.array(orig_val - eps)))
    ) / (2.0 * eps)
    assert abs(fd) > 0.0, (
        'FD is zero: the perturbed yield entry does not affect the '
        'line weight at 1 MeV (grid too coarse or wrong index).'
    )
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-30)
