"""Dispatcher-level tests for get_particle_production_ddxs(broadening=...).

These tests stub out the per-MT compute functions and the selector
predicates so we can verify the dispatcher's wiring -- normalisation
of the broadening argument, fanout to the continuous and discrete
folders, summation logic, and pass-through of the (kernel, width)
pair -- without needing real ENDF data.
"""
import numpy as np
import warnings

import pytest

import endf_userpy.quantities as quantities
from endf_userpy.primitives.np_compat import trapezoid


@pytest.fixture
def stub_quantities(monkeypatch):
    """Replace the per-MT compute functions and selector predicates
    so the dispatcher can be tested in isolation. Returns a state
    dict the test populates with the desired stand-ins."""
    state = {
        # Map mt -> 'cont' | 'disc' | 'none' indicating which folder
        # admits the MT. The dispatcher then asks the (stubbed)
        # compute_* for each admitted MT.
        'mt_kind': {},
        # Map mt -> shape-compatible ndarray returned by cont_folder.
        'cont_return': {},
        # Map mt -> ndarray returned by disc_folder.
        'disc_return': {},
        # The list of MTs the cumulative loop iterates over.
        'mt_list': [],
        # Calls captured for inspection: (folder, mt, kernel_id, width).
        'calls': [],
        # MTs the LAW=1 ND>0 detector should report as hits.
        'law1_disc_mts': set(),
        # Set of MTs the endf_dict pretends to have MF6 entries for
        # (used only by the warning helper's `6 in endf_dict` check).
        'mf6_mts': set(),
    }

    def fake_get_reaction_mts(endf_dict):
        return state['mt_list']

    def fake_translate_reaction_string_to_mt(reacstr):
        # The dispatcher only uses this for user_mts, but our stubbed
        # satisfies_select_heuristic ignores user_mts anyway.
        return 1

    def fake_contains_zap(endf_dict, mt, zap):
        return state['mt_kind'].get(mt, 'none') != 'none'

    def fake_has_continuous_ddx(endf_dict, mt, zap):
        return state['mt_kind'].get(mt) == 'cont'

    def fake_has_discrete_two_body_ddx(endf_dict, mt, zap):
        return state['mt_kind'].get(mt) == 'disc'

    def fake_has_mf6_law1_discrete_lines(endf_dict, mt, zap):
        return state['mt_kind'].get(mt) == 'law1_disc'

    def fake_satisfies_select_heuristic(endf_dict, mt, user_mts=None):
        return True

    def fake_cont_broadened(endf_dict, mt, zap, einc, eouts, mus,
                            kernel, kernel_width):
        state['calls'].append(('cont', mt, id(kernel), kernel_width))
        return state['cont_return'][mt]

    def fake_disc_broadened(endf_dict, mt, zap, einc, eouts, mus, kernel):
        state['calls'].append(('disc', mt, id(kernel), None))
        return state['disc_return'][mt]

    def fake_law1_disc_broadened(endf_dict, mt, zap, einc, eouts, mus, kernel):
        state['calls'].append(('law1_disc', mt, id(kernel), None))
        return state['law1_disc_return'][mt]

    def fake_compute_ddxs(endf_dict, mt, zap, einc, eouts, mus):
        # Unbroadened path used when broadening=None.
        state['calls'].append(('plain', mt, None, None))
        return state['cont_return'][mt]

    def fake_dxs_dE_broadened(endf_dict, mt, zap, einc, eouts,
                              kernel, kernel_width):
        state['calls'].append(('dexs_b', mt, id(kernel), kernel_width))
        return state['dexs_return'][mt]

    def fake_compute_dexs(endf_dict, mt, zap, einc, eouts):
        state['calls'].append(('dexs_plain', mt, None, None))
        return state['dexs_return'][mt]

    def fake_dxs_dE_law1_disc_broadened(endf_dict, mt, zap, einc, eouts, kernel):
        state['calls'].append(('dexs_law1_disc', mt, id(kernel), None))
        return state['dexs_law1_disc_return'][mt]

    # Keys default-populated only when used by a test.
    state.setdefault('dexs_return', {})
    state.setdefault('law1_disc_return', {})
    state.setdefault('dexs_law1_disc_return', {})

    monkeypatch.setattr(
        quantities.quant_mt_zap, 'get_reaction_mt_numbers',
        fake_get_reaction_mts,
    )
    # compute_cumulative_quantity in quantities_mt_zap.quantities imports
    # get_reaction_mts from mf3_interpretation under the alias
    # get_reaction_mt_numbers; patch that module-local reference too.
    monkeypatch.setattr(
        quantities.quant_mt_zap, 'get_reaction_mt_numbers',
        fake_get_reaction_mts,
    )
    import endf_userpy.quantities_mt_zap.quantities as qmtz_mod
    monkeypatch.setattr(qmtz_mod, 'get_reaction_mt_numbers', fake_get_reaction_mts)

    monkeypatch.setattr(
        quantities.reac, 'translate_reaction_string_to_mt',
        fake_translate_reaction_string_to_mt,
    )
    monkeypatch.setattr(
        quantities.selectors, 'contains_zap', fake_contains_zap
    )
    monkeypatch.setattr(
        quantities.selectors, 'has_continuous_ddx', fake_has_continuous_ddx
    )
    monkeypatch.setattr(
        quantities.selectors, 'has_discrete_two_body_ddx',
        fake_has_discrete_two_body_ddx,
    )
    monkeypatch.setattr(
        quantities.selectors, 'has_mf6_law1_discrete_lines',
        fake_has_mf6_law1_discrete_lines,
    )
    monkeypatch.setattr(
        quantities.selectors, 'satisfies_select_heuristic',
        fake_satisfies_select_heuristic,
    )
    monkeypatch.setattr(
        quantities.ddxb, 'compute_ddx_continuous_broadened',
        fake_cont_broadened,
    )
    monkeypatch.setattr(
        quantities.ddxb, 'compute_ddx_discrete_broadened',
        fake_disc_broadened,
    )
    monkeypatch.setattr(
        quantities.ddxb, 'compute_ddx_law1_discrete_broadened',
        fake_law1_disc_broadened,
    )
    monkeypatch.setattr(
        quantities.quant_mt_zap, 'compute_ddxs', fake_compute_ddxs
    )
    monkeypatch.setattr(
        quantities.ddxb, 'compute_dxs_dE_broadened', fake_dxs_dE_broadened
    )
    monkeypatch.setattr(
        quantities.quant_mt_zap, 'compute_dexs', fake_compute_dexs
    )
    monkeypatch.setattr(
        quantities.ddxb, 'compute_dxs_dE_law1_discrete_broadened',
        fake_dxs_dE_law1_disc_broadened,
    )
    return state


def _einc_eouts_mus():
    return (
        np.array([1.4e7]),
        np.linspace(1.0e6, 1.4e7, 5),
        np.linspace(-1.0, 1.0, 3),
    )


# ---------- behaviour when broadening=None ----------


def test_no_broadening_uses_only_continuous_path(stub_quantities):
    """When broadening is None, only continuous MTs contribute and
    the unbroadened compute_ddxs is invoked (not the folder)."""
    einc, eouts, mus = _einc_eouts_mus()
    shape = (len(einc), len(eouts), len(mus))
    stub_quantities['mt_list'] = [16, 51, 91]
    stub_quantities['mt_kind'] = {16: 'cont', 51: 'disc', 91: 'cont'}
    stub_quantities['cont_return'] = {
        16: np.full(shape, 1.0),
        91: np.full(shape, 2.0),
    }

    result = quantities.get_particle_production_ddxs(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        broadening=None,
    )
    np.testing.assert_array_equal(result, np.full(shape, 3.0))
    folders = sorted(c[0] for c in stub_quantities['calls'])
    assert folders == ['plain', 'plain'], (
        f"Expected only 'plain' compute_ddxs calls, got {stub_quantities['calls']}"
    )


# ---------- broadening sums continuous and discrete ----------


def test_broadening_sums_continuous_and_discrete(stub_quantities):
    """A scalar broadening admits both folders and the result is
    the sum of their per-MT outputs."""
    einc, eouts, mus = _einc_eouts_mus()
    shape = (len(einc), len(eouts), len(mus))
    stub_quantities['mt_list'] = [2, 16, 51]
    stub_quantities['mt_kind'] = {2: 'disc', 16: 'cont', 51: 'disc'}
    stub_quantities['cont_return'] = {16: np.full(shape, 7.0)}
    stub_quantities['disc_return'] = {
        2: np.full(shape, 1.5),
        51: np.full(shape, 0.25),
    }

    result = quantities.get_particle_production_ddxs(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        broadening=1.0e5,
    )

    expected = 7.0 + 1.5 + 0.25
    np.testing.assert_allclose(result, np.full(shape, expected))


def test_broadening_only_discrete_when_no_continuous(stub_quantities):
    """If only discrete channels match, the continuous folder result
    is None and the dispatcher returns the discrete result alone."""
    einc, eouts, mus = _einc_eouts_mus()
    shape = (len(einc), len(eouts), len(mus))
    stub_quantities['mt_list'] = [2]
    stub_quantities['mt_kind'] = {2: 'disc'}
    stub_quantities['disc_return'] = {2: np.full(shape, 4.2)}

    result = quantities.get_particle_production_ddxs(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        broadening=1.0e5,
    )
    np.testing.assert_array_equal(result, np.full(shape, 4.2))


def test_broadening_only_continuous_when_no_discrete(stub_quantities):
    """Symmetric to the above: only continuous matches -> continuous-only result."""
    einc, eouts, mus = _einc_eouts_mus()
    shape = (len(einc), len(eouts), len(mus))
    stub_quantities['mt_list'] = [16]
    stub_quantities['mt_kind'] = {16: 'cont'}
    stub_quantities['cont_return'] = {16: np.full(shape, 3.3)}

    result = quantities.get_particle_production_ddxs(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        broadening=1.0e5,
    )
    np.testing.assert_array_equal(result, np.full(shape, 3.3))


# ---------- normalisation of the broadening argument ----------


def test_broadening_scalar_passes_gaussian_with_right_width(stub_quantities):
    """A scalar sigma must reach the continuous folder as
    (callable, sigma)."""
    einc, eouts, mus = _einc_eouts_mus()
    shape = (len(einc), len(eouts), len(mus))
    stub_quantities['mt_list'] = [16]
    stub_quantities['mt_kind'] = {16: 'cont'}
    stub_quantities['cont_return'] = {16: np.zeros(shape)}

    sigma = 1.2345e5
    quantities.get_particle_production_ddxs(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        broadening=sigma,
    )
    cont_calls = [c for c in stub_quantities['calls'] if c[0] == 'cont']
    assert len(cont_calls) == 1
    _, _, _, width_passed = cont_calls[0]
    assert width_passed == sigma


def test_broadening_tuple_uses_provided_kernel_and_width(stub_quantities):
    """A (callable, width) tuple must pass both through unchanged."""
    einc, eouts, mus = _einc_eouts_mus()
    shape = (len(einc), len(eouts), len(mus))
    stub_quantities['mt_list'] = [16]
    stub_quantities['mt_kind'] = {16: 'cont'}
    stub_quantities['cont_return'] = {16: np.zeros(shape)}

    user_kernel = lambda d: np.exp(-np.abs(d))  # noqa: E731
    user_width = 3.0e4
    quantities.get_particle_production_ddxs(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        broadening=(user_kernel, user_width),
    )
    cont_calls = [c for c in stub_quantities['calls'] if c[0] == 'cont']
    _, _, kernel_id, width_passed = cont_calls[0]
    assert kernel_id == id(user_kernel)
    assert width_passed == user_width


def test_broadening_rejects_invalid_specs():
    """Bad values for broadening raise ValueError before any compute
    runs."""
    einc, eouts, mus = _einc_eouts_mus()

    with pytest.raises(ValueError, match="positive"):
        quantities._normalize_broadening(0.0)
    with pytest.raises(ValueError, match="positive"):
        quantities._normalize_broadening(-1.0)
    with pytest.raises(ValueError, match="callable"):
        quantities._normalize_broadening(("not a callable", 1e5))
    with pytest.raises(ValueError, match="positive"):
        quantities._normalize_broadening((lambda d: d, 0.0))
    with pytest.raises(ValueError, match="None|positive scalar|tuple"):
        quantities._normalize_broadening("invalid")


def test_normalize_broadening_none_is_identity():
    assert quantities._normalize_broadening(None) == (None, None)


def test_normalize_broadening_gaussian_is_unit_norm():
    """The scalar form must construct a kernel that integrates to 1."""
    kernel, width = quantities._normalize_broadening(1.0e5)
    # Sample over +/- 10 sigma -> well outside the kernel support.
    x = np.linspace(-1.0e6, 1.0e6, 50001)
    integral = trapezoid(kernel(x), x)
    assert abs(integral - 1.0) < 1e-6


# ============================================================
# get_particle_production_dxs_dE dispatcher
# ============================================================


def test_dxs_dE_no_broadening_uses_plain_compute_dexs(stub_quantities):
    """broadening=None routes to compute_dexs unchanged; no folder is
    called."""
    einc = np.array([1.4e7])
    eouts = np.linspace(1e6, 1.4e7, 5)
    shape = (len(einc), len(eouts))
    stub_quantities['mt_list'] = [16, 91]
    stub_quantities['mt_kind'] = {16: 'cont', 91: 'cont'}
    stub_quantities['dexs_return'] = {
        16: np.full(shape, 1.0),
        91: np.full(shape, 2.0),
    }

    result = quantities.get_particle_production_dxs_dE(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts,
        broadening=None,
    )
    np.testing.assert_array_equal(result, np.full(shape, 3.0))
    kinds = sorted(c[0] for c in stub_quantities['calls'])
    assert kinds == ['dexs_plain', 'dexs_plain']


def test_dxs_dE_broadening_routes_through_folder(stub_quantities):
    """A scalar broadening routes every admitted MT through the
    1D folder rather than plain compute_dexs."""
    einc = np.array([1.4e7])
    eouts = np.linspace(1e6, 1.4e7, 5)
    shape = (len(einc), len(eouts))
    stub_quantities['mt_list'] = [16, 51]
    # The dxs/dE dispatcher does not split continuous vs discrete -
    # compute_dexs handles both. Mark both as 'cont' so contains_zap
    # admits them.
    stub_quantities['mt_kind'] = {16: 'cont', 51: 'cont'}
    stub_quantities['dexs_return'] = {
        16: np.full(shape, 5.0),
        51: np.full(shape, 1.5),
    }

    sigma = 2.0e5
    result = quantities.get_particle_production_dxs_dE(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts,
        broadening=sigma,
    )
    np.testing.assert_array_equal(result, np.full(shape, 6.5))
    folder_calls = [c for c in stub_quantities['calls'] if c[0] == 'dexs_b']
    plain_calls = [c for c in stub_quantities['calls'] if c[0] == 'dexs_plain']
    assert len(folder_calls) == 2 and not plain_calls
    # Width is propagated.
    for _, _, _, w in folder_calls:
        assert w == sigma


def test_dxs_dE_broadening_tuple_passes_kernel_through(stub_quantities):
    """A (callable, width) tuple reaches the folder unchanged."""
    einc = np.array([1.4e7])
    eouts = np.linspace(1e6, 1.4e7, 5)
    shape = (len(einc), len(eouts))
    stub_quantities['mt_list'] = [91]
    stub_quantities['mt_kind'] = {91: 'cont'}
    stub_quantities['dexs_return'] = {91: np.zeros(shape)}

    user_kernel = lambda d: np.exp(-np.abs(d))  # noqa: E731
    user_width = 3.0e4
    quantities.get_particle_production_dxs_dE(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts,
        broadening=(user_kernel, user_width),
    )
    folder_calls = [c for c in stub_quantities['calls'] if c[0] == 'dexs_b']
    assert len(folder_calls) == 1
    _, _, kernel_id, width = folder_calls[0]
    assert kernel_id == id(user_kernel)
    assert width == user_width


# ============================================================
# Dispatcher wiring for MF6/LAW=1 ND>0 folders
# ============================================================


def _endf_with_mf6(mts):
    """Minimal endf-like dict with MF6 populated for the given MTs."""
    return {6: {mt: {} for mt in mts}}


def test_ddx_broadening_sums_all_three_paths(stub_quantities):
    """With broadening on, admitted MTs from all three families -
    continuous, 2-body discrete, and MF6/LAW=1 ND>0 - are routed to
    their respective folders and the results are summed."""
    einc, eouts, mus = _einc_eouts_mus()
    shape = (len(einc), len(eouts), len(mus))
    stub_quantities['mt_list'] = [16, 51, 702]
    stub_quantities['mt_kind'] = {16: 'cont', 51: 'disc', 702: 'law1_disc'}
    stub_quantities['cont_return'] = {16: np.full(shape, 4.0)}
    stub_quantities['disc_return'] = {51: np.full(shape, 1.5)}
    stub_quantities['law1_disc_return'] = {702: np.full(shape, 0.25)}

    result = quantities.get_particle_production_ddxs(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        broadening=1.0e5,
    )
    np.testing.assert_allclose(result, np.full(shape, 5.75))
    folders = sorted(c[0] for c in stub_quantities['calls'])
    assert folders == ['cont', 'disc', 'law1_disc']


def test_ddx_broadening_law1_only_returns_law1_alone(stub_quantities):
    """If only LAW=1 discrete MTs are admitted, the DDX result comes
    entirely from that folder."""
    einc, eouts, mus = _einc_eouts_mus()
    shape = (len(einc), len(eouts), len(mus))
    stub_quantities['mt_list'] = [702]
    stub_quantities['mt_kind'] = {702: 'law1_disc'}
    stub_quantities['law1_disc_return'] = {702: np.full(shape, 3.14)}

    result = quantities.get_particle_production_ddxs(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        broadening=1.0e5,
    )
    np.testing.assert_array_equal(result, np.full(shape, 3.14))


def test_ddx_broadening_no_warning_when_law1_handled(stub_quantities, monkeypatch):
    """The DDX dispatcher no longer emits the LAW=1 gap warning; the
    folder now handles those channels. Verify silence."""
    einc, eouts, mus = _einc_eouts_mus()
    shape = (len(einc), len(eouts), len(mus))
    stub_quantities['mt_list'] = [702]
    stub_quantities['mt_kind'] = {702: 'law1_disc'}
    stub_quantities['law1_disc_return'] = {702: np.zeros(shape)}

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        quantities.get_particle_production_ddxs(
            endf_dict=_endf_with_mf6([702]), reaction='(n,total)', particle='n',
            energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
            broadening=1.0e5,
        )
    law1 = [w for w in recorded if 'MF6/LAW=1' in str(w.message)]
    assert law1 == []


def test_dxs_dE_broadening_sums_cont_and_law1_disc(stub_quantities):
    """The 1D dispatcher, with broadening on, runs the continuous
    folder AND the LAW=1 discrete folder over their respective
    admitted MTs and sums the results.

    Note: the cont path runs on every admitted MT (including
    law1_disc-only ones); in production compute_dxs_dE_broadened
    catches the IndexError from compute_dexs and returns zeros. The
    stub emulates that by returning zeros for law1_disc MTs."""
    einc = np.array([1.4e7])
    eouts = np.linspace(1e6, 1.4e7, 5)
    shape = (len(einc), len(eouts))
    stub_quantities['mt_list'] = [16, 702]
    stub_quantities['mt_kind'] = {16: 'cont', 702: 'law1_disc'}
    stub_quantities['dexs_return'] = {
        16: np.full(shape, 5.0),
        702: np.zeros(shape),  # emulates the IndexError-swallow path
    }
    stub_quantities['dexs_law1_disc_return'] = {702: np.full(shape, 0.25)}

    result = quantities.get_particle_production_dxs_dE(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts,
        broadening=1.0e5,
    )
    np.testing.assert_allclose(result, np.full(shape, 5.25))
    folder_kinds = sorted(c[0] for c in stub_quantities['calls'])
    assert 'dexs_law1_disc' in folder_kinds
    assert folder_kinds.count('dexs_b') == 2  # once per admitted MT


def test_dxs_dE_broadening_law1_only_returns_law1_alone(stub_quantities):
    """If only LAW=1 discrete MTs are admitted, the 1D result comes
    entirely from that folder (the cont path returns zeros)."""
    einc = np.array([1.4e7])
    eouts = np.linspace(1e6, 1.4e7, 5)
    shape = (len(einc), len(eouts))
    stub_quantities['mt_list'] = [702]
    stub_quantities['mt_kind'] = {702: 'law1_disc'}
    stub_quantities['dexs_return'] = {702: np.zeros(shape)}  # cont path zeros
    stub_quantities['dexs_law1_disc_return'] = {702: np.full(shape, 2.5)}

    result = quantities.get_particle_production_dxs_dE(
        endf_dict=None, reaction='(n,total)', particle='n',
        energies_in=einc, energies_out=eouts,
        broadening=1.0e5,
    )
    np.testing.assert_array_equal(result, np.full(shape, 2.5))


def test_dxs_dE_broadening_no_warning_when_law1_handled(stub_quantities):
    """The dxs/dE dispatcher no longer emits the LAW=1 gap warning
    because the folder now handles those channels."""
    einc = np.array([1.4e7])
    eouts = np.linspace(1e6, 1.4e7, 5)
    shape = (len(einc), len(eouts))
    stub_quantities['mt_list'] = [702]
    stub_quantities['mt_kind'] = {702: 'law1_disc'}
    stub_quantities['dexs_return'] = {702: np.zeros(shape)}
    stub_quantities['dexs_law1_disc_return'] = {702: np.zeros(shape)}

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        quantities.get_particle_production_dxs_dE(
            endf_dict=_endf_with_mf6([702]), reaction='(n,total)', particle='n',
            energies_in=einc, energies_out=eouts,
            broadening=1.0e5,
        )
    law1 = [w for w in recorded if 'MF6/LAW=1' in str(w.message)]
    assert law1 == []
