"""Tests for ddx_broadening (continuous + discrete folders).

The tests monkeypatch the dist2d / angdist / mass / yield / xs
primitives so the broadening wrappers can be checked against
closed-form references without depending on any real ENDF data.
"""
import numpy as np
import pytest

from endf_userpy.quantities_mt_zap import ddx_broadening as ddxb
from endf_userpy.quantities_mt_zap import selectors


def _gaussian(x, sigma, mu=0.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


@pytest.fixture
def patched_environment(monkeypatch):
    """Stand-ins for the data-access primitives both folders consume.
    Each test installs the pieces it needs via the returned setter;
    unset entries fall back to harmless defaults."""
    state = {
        'dist2d': lambda einc, eouts, mus: np.zeros((len(einc), len(eouts), len(mus))),
        'angdist': lambda einc, mus: np.zeros((len(einc), len(mus))),
        'yields_all': lambda einc: np.ones(len(einc)),
        'yields_cont': lambda einc: np.zeros(len(einc)),
        'xs': lambda einc: np.ones(len(einc)),
        'has_mf4': False, 'has_mf5': False, 'has_mf6': False,
        'has_angdist_part': False,
        'projectile_mass': 939.565e6,
        'target_mass': 27 * 939.565e6,
        'ejectile_mass': 939.565e6,
        'qvalue': 0.0,
        'projectile': 'n',
        'multiplicity': 1.0,
    }

    def fake_dist2d(endf_dict, mt, zap, einc, eouts, mus, to_lab=True):
        return state['dist2d'](einc, eouts, mus)

    def fake_yields(endf_dict, mt, zap, einc, include_discrete=True, level=None):
        if include_discrete:
            return state['yields_all'](einc)
        return state['yields_cont'](einc)

    def fake_xs(endf_dict, mt, einc):
        return state['xs'](einc)

    def fake_angdist_mf4(endf_dict, mt, einc, mus, to_lab=True):
        return state['angdist'](einc, mus)

    def fake_angdist_mf6(endf_dict, mt, zap, einc, mus, to_lab=True):
        return state['angdist'](einc, mus)

    monkeypatch.setattr(ddxb, 'compute_dist2d_values', fake_dist2d)
    monkeypatch.setattr(ddxb, 'compute_yields', fake_yields)
    monkeypatch.setattr(ddxb.mf3_interp, 'compute_cross_section', fake_xs)
    monkeypatch.setattr(
        ddxb.mf4_interp, 'compute_angdist_values', fake_angdist_mf4
    )
    monkeypatch.setattr(
        ddxb.mf6_interp, 'compute_angdist_values', fake_angdist_mf6
    )
    monkeypatch.setattr(
        ddxb, 'has_mf4_mt', lambda d, mt: state['has_mf4']
    )
    monkeypatch.setattr(
        ddxb, 'has_mf5_mt', lambda d, mt: state['has_mf5']
    )
    monkeypatch.setattr(
        ddxb, 'has_mf6_mt', lambda d, mt: state['has_mf6']
    )
    monkeypatch.setattr(
        ddxb.mf6_help, 'has_angdist_part',
        lambda d, mt, zap: state['has_angdist_part'],
    )
    monkeypatch.setattr(
        ddxb, 'get_projectile_mass', lambda d: state['projectile_mass']
    )
    monkeypatch.setattr(
        ddxb, 'get_target_mass', lambda d: state['target_mass']
    )
    monkeypatch.setattr(
        ddxb, 'get_particle_mass_for_zap', lambda zap: state['ejectile_mass']
    )
    monkeypatch.setattr(
        ddxb, 'get_reaction_qvalue', lambda d, mt: state['qvalue']
    )
    monkeypatch.setattr(
        ddxb, 'get_projectile', lambda d: state['projectile']
    )
    monkeypatch.setattr(
        ddxb.reactions, 'get_multiplicity_for_zap',
        lambda proj, mt, zap: state['multiplicity'],
    )

    def setter(**overrides):
        state.update(overrides)

    return setter


def test_constant_spectrum_preserved(patched_environment):
    """Constant dist2d * unit-norm kernel = same constant, in the
    interior of the eout window. With xs=yield=1 the result is
    exactly 1/(2 pi) everywhere by the compute_ddxs normalization."""
    n_einc, n_mu = 2, 3
    patched_environment(
        dist2d=lambda einc, eouts, mus: np.ones((len(einc), len(eouts), len(mus))),
        yields_cont=lambda einc: np.ones(len(einc)),
        xs=lambda einc: np.ones(len(einc)),
    )

    einc = np.array([1.0e6, 5.0e6])
    eouts = np.linspace(-2.0e6, 2.0e6, 21)
    mus = np.linspace(-1.0, 1.0, n_mu)
    sigma = 3.0e5

    result = ddxb.compute_ddx_continuous_broadened(
        endf_dict=None, mt=0, zap=1,
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        kernel=lambda d: _gaussian(d, sigma),
        kernel_width=sigma,
        rtol=1e-5,
    )

    assert result.shape == (n_einc, len(eouts), n_mu)
    expected = 1.0 / (2 * np.pi)
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_gaussian_spectrum_broadens_analytically(patched_environment):
    """Gaussian dist2d (in E_out) folded with Gaussian kernel produces
    a wider Gaussian with sigma_y = sqrt(sigma_f^2 + sigma_k^2). The
    xs/yield/2pi prefactor follows the existing compute_ddxs convention."""
    sigma_f = 4.0e5
    sigma_k = 2.0e5
    e_centre = 1.4e7

    def dist2d(einc, eouts, mus):
        spec = _gaussian(eouts, sigma_f, mu=e_centre)
        return np.broadcast_to(
            spec[None, :, None],
            (len(einc), len(eouts), len(mus)),
        ).copy()

    patched_environment(
        dist2d=dist2d,
        yields_cont=lambda einc: np.full(len(einc), 2.5),
        xs=lambda einc: np.full(len(einc), 0.7),
    )

    einc = np.array([1.4e7])
    eouts = np.linspace(e_centre - 2.0e6, e_centre + 2.0e6, 41)
    mus = np.array([-0.5, 0.0, 0.5])

    result = ddxb.compute_ddx_continuous_broadened(
        endf_dict=None, mt=0, zap=1,
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        kernel=lambda d: _gaussian(d, sigma_k),
        kernel_width=sigma_k,
        rtol=1e-6,
        max_iter=10,
    )

    sigma_y = np.sqrt(sigma_f ** 2 + sigma_k ** 2)
    expected_spec = _gaussian(eouts, sigma_y, mu=e_centre)
    expected = expected_spec[None, :, None] * (2.5 * 0.7 / (2 * np.pi))

    # Broadcast expected across mu axis explicitly so assert_allclose
    # checks every slice.
    expected = np.broadcast_to(expected, result.shape)
    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-12)


def test_integral_preserved(patched_environment):
    """Convolution by a unit-norm kernel preserves the integral of f
    over E_out. So the total production cross section per (E_in, mu)
    bin -- integral of DDX over E_out -- should match before/after
    broadening."""
    sigma_f = 4.0e5
    sigma_k = 2.0e5
    e_centre = 1.4e7

    def dist2d(einc, eouts, mus):
        spec = _gaussian(eouts, sigma_f, mu=e_centre)
        return np.broadcast_to(
            spec[None, :, None],
            (len(einc), len(eouts), len(mus)),
        ).copy()

    patched_environment(
        dist2d=dist2d,
        yields_cont=lambda einc: np.full(len(einc), 1.5),
        xs=lambda einc: np.full(len(einc), 0.3),
    )

    einc = np.array([1.4e7])
    eouts = np.linspace(e_centre - 3.0e6, e_centre + 3.0e6, 401)
    mus = np.array([0.0])

    broadened = ddxb.compute_ddx_continuous_broadened(
        endf_dict=None, mt=0, zap=1,
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        kernel=lambda d: _gaussian(d, sigma_k),
        kernel_width=sigma_k,
        rtol=1e-5,
        max_iter=10,
    )

    # f integrates to 1 (unit-norm Gaussian over an infinite domain;
    # the eouts grid spans +/- 7.5 sigma so the truncation loss is
    # negligible). Hence the production-side integral is xs * yield
    # / (2 pi) per (E_in, mu) bin = 0.3 * 1.5 / (2 pi).
    integral = np.trapezoid(broadened[0, :, 0], eouts)
    expected_integral = 1.5 * 0.3 / (2 * np.pi)
    assert abs(integral - expected_integral) / expected_integral < 1e-3


# ============================================================
# Discrete-folder tests
# ============================================================


def test_discrete_peak_position_matches_kinematics(patched_environment):
    """For elastic on an Al-27-like target the broadened DDX peaks
    along E_out_kin(mu): mu=+1 gives E_out ~ E_in (forward), mu=-1
    gives E_out ~ E_in * ((A-1)/(A+1))^2 (backscatter, nonrelativistic).
    The peak location in the broadened spectrum is the kinematic
    locus regardless of kernel width."""
    m_n = 939.565e6
    A = 27.0
    patched_environment(
        has_mf4=True, has_mf5=False, has_mf6=False,
        # Isotropic angdist (g(mu) = 1/2 normalises ∫g dmu = 1).
        angdist=lambda einc, mus: np.full((len(einc), len(mus)), 0.5),
        projectile_mass=m_n, target_mass=A * m_n, ejectile_mass=m_n,
        qvalue=0.0,
        xs=lambda einc: np.ones(len(einc)),
        yields_all=lambda einc: np.ones(len(einc)),
        yields_cont=lambda einc: np.zeros(len(einc)),
    )

    e_in = 1.4e7
    sigma_k = 5.0e4
    einc = np.array([e_in])
    mus = np.array([+1.0, -1.0])
    eouts = np.linspace(0.8 * e_in, 1.02 * e_in, 5001)

    result = ddxb.compute_ddx_discrete_broadened(
        endf_dict=None, mt=2, zap=1,
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        kernel=lambda d: _gaussian(d, sigma_k),
    )

    peak_fwd = eouts[np.argmax(result[0, :, 0])]
    peak_bwd = eouts[np.argmax(result[0, :, 1])]

    expected_bwd_nr = e_in * ((A - 1) / (A + 1)) ** 2

    # Forward: relativistic correction is tiny at 14 MeV / A=27, so
    # E_out_kin(mu=+1) sits within 0.5% of E_in.
    assert abs(peak_fwd - e_in) / e_in < 5e-3
    # Backward: the non-relativistic prediction is accurate to a
    # similar level.
    assert abs(peak_bwd - expected_bwd_nr) / expected_bwd_nr < 5e-3


def test_discrete_isotropic_integrates_to_xs_times_yield(patched_environment):
    """∫∫ ddx_discrete dE_out d(mu) integrated over a fine grid and
    multiplied by 2*pi recovers xs * yield_discrete, since the kernel
    is unit-norm and the angdist ∫g(mu) dmu = 1."""
    m_n = 939.565e6
    A = 56.0  # heavier target -> wider kinematic window so eouts grid spans it
    xs_val = 0.42
    yield_val = 1.0
    patched_environment(
        has_mf4=True, has_mf5=False, has_mf6=False,
        angdist=lambda einc, mus: np.full((len(einc), len(mus)), 0.5),
        projectile_mass=m_n, target_mass=A * m_n, ejectile_mass=m_n,
        qvalue=0.0,
        xs=lambda einc: np.full(len(einc), xs_val),
        yields_all=lambda einc: np.full(len(einc), yield_val),
        yields_cont=lambda einc: np.zeros(len(einc)),
    )

    e_in = 1.4e7
    eout_min = e_in * ((A - 1) / (A + 1)) ** 2
    sigma_k = 5.0e4
    einc = np.array([e_in])
    mus = np.linspace(-1.0, 1.0, 401)
    eouts = np.linspace(eout_min - 10 * sigma_k, e_in + 10 * sigma_k, 4001)

    result = ddxb.compute_ddx_discrete_broadened(
        endf_dict=None, mt=2, zap=1,
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        kernel=lambda d: _gaussian(d, sigma_k),
    )

    # ∫∫ ddx dE_out d(Omega) = 2*pi ∫∫ ddx dE_out dmu
    integral = np.trapezoid(
        np.trapezoid(result[0], eouts, axis=0),
        mus,
    ) * 2 * np.pi

    expected = xs_val * yield_val
    assert abs(integral - expected) / expected < 5e-3


def test_discrete_yield_from_reaction_multiplicity(patched_environment):
    """The discrete folder reads the outgoing-particle multiplicity
    from the reaction-string table (always 1 for genuine 2-body
    channels like (n,n'), (n,p'), etc.). It does NOT consult the MF6
    yield bookkeeping, which in some evaluations only lists the heavy
    residual. Setting multiplicity=2 on the (synthetic) reaction
    table and checking the integral scales linearly verifies this."""
    m_n = 939.565e6
    A = 56.0
    xs_val = 0.5
    patched_environment(
        has_mf4=False, has_mf5=False, has_mf6=True,
        has_angdist_part=True,
        angdist=lambda einc, mus: np.full((len(einc), len(mus)), 0.5),
        projectile_mass=m_n, target_mass=A * m_n, ejectile_mass=m_n,
        qvalue=0.0,
        xs=lambda einc: np.full(len(einc), xs_val),
        multiplicity=2.0,
    )

    e_in = 1.4e7
    eout_min = e_in * ((A - 1) / (A + 1)) ** 2
    sigma_k = 5.0e4
    einc = np.array([e_in])
    mus = np.linspace(-1.0, 1.0, 401)
    eouts = np.linspace(eout_min - 10 * sigma_k, e_in + 10 * sigma_k, 4001)

    result = ddxb.compute_ddx_discrete_broadened(
        endf_dict=None, mt=51, zap=1,
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        kernel=lambda d: _gaussian(d, sigma_k),
    )

    integral = np.trapezoid(
        np.trapezoid(result[0], eouts, axis=0),
        mus,
    ) * 2 * np.pi
    expected = xs_val * 2.0
    assert abs(integral - expected) / expected < 5e-3


def test_discrete_yield_raises_when_no_multiplicity(patched_environment):
    """If the reaction table has no multiplicity for (proj, mt, zap),
    fail loudly rather than silently zeroing."""
    patched_environment(
        has_mf4=True, has_mf5=False, has_mf6=False,
        angdist=lambda einc, mus: np.full((len(einc), len(mus)), 0.5),
        multiplicity=None,
    )
    with pytest.raises(ValueError, match="No multiplicity"):
        ddxb.compute_ddx_discrete_broadened(
            endf_dict=None, mt=2, zap=1,
            energies_in=np.array([1.4e7]),
            energies_out=np.array([1.0e7]),
            angle_cosines_out=np.array([0.0]),
            kernel=lambda d: _gaussian(d, 1.0e5),
        )


def test_discrete_rejects_channel_without_two_body_data(patched_environment):
    """If neither MF6/LAW=2 nor MF4-only is present, the folder must
    raise. (It's the dispatcher's job to gate on the selector; the
    folder itself fails loudly rather than silently returning zeros.)"""
    patched_environment(has_mf4=True, has_mf5=True, has_mf6=False)
    with pytest.raises(ValueError, match="2-body discrete"):
        ddxb.compute_ddx_discrete_broadened(
            endf_dict=None, mt=91, zap=1,
            energies_in=np.array([1.4e7]),
            energies_out=np.array([1.0e7]),
            angle_cosines_out=np.array([0.0]),
            kernel=lambda d: _gaussian(d, 1.0e5),
        )


# ============================================================
# Selector predicate
# ============================================================


class _FakeMod:
    """Lightweight stand-in for the property accessors below."""


def test_has_discrete_two_body_ddx_mf4_only_neutron(monkeypatch):
    """MF4 only (no MF5, no MF6) for a neutron ejectile counts as
    a 2-body discrete channel (typical MT 2 elastic)."""
    monkeypatch.setattr(selectors.prop, 'has_mf6_mt', lambda d, mt: False)
    monkeypatch.setattr(selectors.prop, 'has_mf4_mt', lambda d, mt: True)
    monkeypatch.setattr(selectors.prop, 'has_mf5_mt', lambda d, mt: False)
    zap_n = selectors.physconst.PARTICLE_ZAP['n']
    assert selectors.has_discrete_two_body_ddx({}, 2, zap_n) is True


def test_has_discrete_two_body_ddx_mf6_with_angdist_part(monkeypatch):
    """MF6 with at least one LAW=2/3/4 subsection counts."""
    monkeypatch.setattr(selectors.prop, 'has_mf6_mt', lambda d, mt: True)
    monkeypatch.setattr(
        selectors.mf6help, 'has_angdist_part',
        lambda d, mt, zap: True,
    )
    zap_n = selectors.physconst.PARTICLE_ZAP['n']
    assert selectors.has_discrete_two_body_ddx({}, 51, zap_n) is True


def test_has_discrete_two_body_ddx_pure_mf6_law1_does_not(monkeypatch):
    """MF6 with no LAW=2/3/4 (e.g. only LAW=1 continuum) does not."""
    monkeypatch.setattr(selectors.prop, 'has_mf6_mt', lambda d, mt: True)
    monkeypatch.setattr(
        selectors.mf6help, 'has_angdist_part',
        lambda d, mt, zap: False,
    )
    zap_n = selectors.physconst.PARTICLE_ZAP['n']
    assert selectors.has_discrete_two_body_ddx({}, 16, zap_n) is False


def test_has_discrete_two_body_ddx_mf4_plus_mf5_does_not(monkeypatch):
    """Having MF5 in addition to MF4 means the channel is a continuum,
    not a 2-body kinematic delta."""
    monkeypatch.setattr(selectors.prop, 'has_mf6_mt', lambda d, mt: False)
    monkeypatch.setattr(selectors.prop, 'has_mf4_mt', lambda d, mt: True)
    monkeypatch.setattr(selectors.prop, 'has_mf5_mt', lambda d, mt: True)
    zap_n = selectors.physconst.PARTICLE_ZAP['n']
    assert selectors.has_discrete_two_body_ddx({}, 91, zap_n) is False
