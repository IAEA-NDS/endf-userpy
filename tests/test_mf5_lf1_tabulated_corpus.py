"""Real-corpus coverage for MF5 LF=1 tabulated spectrum (issue #48).

The MF5 LF=1 branch (`compute_tabulated_spectrum` +
`compute_spectrum` dispatcher route) was previously exercised only
by the `never_fails` synthetic test in
`tests/test_mf5_interpretation.py`. PR #63 (issue #41) rewrote and
tested the analytic LF=5/7/9 branches; this file closes the gap
for the LF=1 branch using U-235 MT 18 fission-neutron spectrum in
TENDL-2021.

Physics: U-235 fast fission neutron spectrum peaks around ~1 MeV
with a Watt-like shape; total spectrum integrates to 1.
"""
from pathlib import Path
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf5_interpretation as mf5


ADHOC_DATA_DIR = Path(__file__).resolve().parent / 'data_law1_adhoc'


@pytest.fixture(scope='module')
def u235_tendl():
    fn = ADHOC_DATA_DIR / 'tendl21_n_U-235.endf'
    if not fn.exists():
        pytest.skip(
            f'{fn.name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh` to populate the corpus'
        )
    return EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    ).parsefile(fn)


def test_u235_mt18_mf5_lf1_integrates_to_one(u235_tendl):
    """Full fission spectrum via `compute_spectrum` at 5 MeV
    incident. Contributions sum to 1 over the outgoing-energy
    range. Uses a very fine E' grid so trapezoid error doesn't
    intrude on the tolerance."""
    E = np.array([5.0e6])
    Eout = np.linspace(1.0e4, 2.0e7, 20001)
    f = mf5.compute_spectrum(u235_tendl, 18, E, Eout)
    integ = np.trapezoid(f[0], Eout)
    assert abs(integ - 1.0) < 5e-3, (
        f'MT 18 MF5 spectrum should integrate to 1; got {integ}'
    )


def test_u235_mt18_mf5_lf1_watt_peak_around_1MeV(u235_tendl):
    """U-235 fast-fission neutron spectrum is Watt-shaped with the
    peak around 0.7-1.0 MeV. Assert the mode of the spectrum lies
    in a physically reasonable band."""
    E = np.array([5.0e6])
    Eout = np.linspace(1.0e4, 1.5e7, 15001)
    f = mf5.compute_spectrum(u235_tendl, 18, E, Eout)
    peak_eout = Eout[np.argmax(f[0])]
    assert 5e5 < peak_eout < 1.5e6, (
        f'expected fission spectrum peak in [0.5, 1.5] MeV; '
        f'got {peak_eout/1e6:.3f} MeV'
    )


def test_u235_mt18_mf5_lf1_nonnegative_and_finite(u235_tendl):
    """Sanity: the reconstructed spectrum is non-negative and
    finite everywhere."""
    E = np.array([1e5, 1e6, 5e6, 1.4e7])
    Eout = np.linspace(1.0e4, 2.0e7, 501)
    f = mf5.compute_spectrum(u235_tendl, 18, E, Eout)
    assert f.shape == (len(E), len(Eout))
    assert np.all(np.isfinite(f))
    assert np.all(f >= 0)


def test_lf1_contribution_via_direct_dispatcher(u235_tendl):
    """The `compute_spectrum_contribution` dispatcher routes LF=1
    to `compute_tabulated_spectrum`. Verify the individual
    contribution integrates to 1 (the p_table probability weight
    is applied only in the top-level `compute_spectrum`)."""
    contribs = list(u235_tendl[5][18]['contribution'].values())
    lf1 = [c for c in contribs if c['LF'] == 1]
    assert len(lf1) >= 1, 'expected at least one LF=1 contribution'
    c = lf1[0]
    E = np.array([1e6])
    Eout = np.linspace(1.0e4, 2.0e7, 20001)
    f = mf5.compute_spectrum_contribution(c, E, Eout)
    integ = np.trapezoid(f[0], Eout)
    assert abs(integ - 1.0) < 5e-3
