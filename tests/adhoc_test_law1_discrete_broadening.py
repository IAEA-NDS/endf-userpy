"""Ad-hoc coverage tests for MF6/LAW=1 ND>0 discrete-line broadening.

Named `adhoc_test_*` so pytest does NOT auto-collect this module in
the default `pytest tests/` run. Invoke explicitly:

    pytest tests/adhoc_test_law1_discrete_broadening.py -v

or run as a plain script:

    python tests/adhoc_test_law1_discrete_broadening.py

The data files live in `tests/data_law1_adhoc/` (see that folder for
the picked candidates and their provenance). These files were chosen
to widen coverage of the LAW=1 ND>0 handling beyond the single
Be-9 case in the main suite:

    endfb81_n_Be-9.endf    ENDF-B-VIII.1  Be-9    single gamma line, minimal
    endfb81_n_B-11.endf    ENDF-B-VIII.1  B-11    3 MTs (22, 103, 107)
    endfb81_n_Al-27.endf   ENDF-B-VIII.1  Al-27   106 subsecs across MT 51..819
    tendl21_n_Fe-56.endf   TENDL-2021     Fe-56   mixed cont+disc, LCT=3, up to 42 panels
    tendl21_n_U-235.endf   TENDL-2021     U-235   LCT 2 and 3, actinide, mixed

Each test:
    - enumerates every (MT, ZAP=gamma) that has MF6/LAW=1 ND>0 content;
    - runs the LAW=1 discrete folder in isolation
      (compute_ddx_law1_discrete_broadened, and its 1D projection);
    - checks the result is finite and non-negative;
    - checks the integrated DDX (or 1D spectrum) matches the reference
      production xs = xs(MT) * yield(MT, ZAP=gamma) within the
      per-file tolerance.

Failures here don't imply a bug in the folder alone: a MT/ZAP pair
where the integrated result diverges from xs*yield may be highlighting
a limitation (panel interpolation edge case, mu-grid density, kernel
truncation near a peak that sits at the grid edge, mixed subsec
splits, ...). Each failure is a candidate for further investigation
before deciding whether to promote the file into the main suite.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf3_interpretation as mf3_interp
from endf_userpy.quantities_mt_zap import ddx_broadening as ddxb
from endf_userpy.quantities_mt_zap.quantities import compute_yields
from endf_userpy.quantities_mt_zap import selectors


DATA_DIR = Path(__file__).resolve().parent / 'data_law1_adhoc'
GAMMA_ZAP = 0.0

# Per-file settings. E_in picks the target incident energy; the eouts
# window and internal mu density are set so the peaks (which for
# gamma cascades span ~10 keV to a few MeV) are resolved. `eout_min`
# has to be well below the lowest discrete line in the file, or a
# kernel-width fraction of the low-energy peak's mass falls out of
# the integration window and shows up as a systematic shortfall.
# rtol is the fractional-error tolerance on the integrated
# production xs vs the reference.
FILE_CFG = {
    'endfb81_n_Be-9.endf':   dict(e_in=1.4e7, eout_min=1e3, eout_max=8e5,
                                  n_eouts=401, n_mus=21, sigma=3e4, rtol=5e-3),
    'endfb81_n_B-11.endf':   dict(e_in=1.4e7, eout_min=1e3, eout_max=2e7,
                                  n_eouts=1001, n_mus=21, sigma=1e5, rtol=5e-2),
    'endfb81_n_Al-27.endf':  dict(e_in=1.4e7, eout_min=1e3, eout_max=1.4e7,
                                  n_eouts=1501, n_mus=21, sigma=8e4, rtol=1e-1),
    'tendl21_n_Fe-56.endf':  dict(e_in=1.4e7, eout_min=1e3, eout_max=1.5e7,
                                  n_eouts=1501, n_mus=21, sigma=1e5, rtol=1e-1),
    'tendl21_n_U-235.endf':  dict(e_in=1.4e7, eout_min=1e3, eout_max=1.5e7,
                                  n_eouts=1501, n_mus=21, sigma=1e5, rtol=1e-1),
}


def _gaussian(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _parse(fn):
    parser = EndfParserCpp(
        ignore_missing_tpid=True,
        ignore_zero_mismatch=True,
        accept_spaces=True,
    )
    return parser.parsefile(str(fn))


def _admitted_gamma_mts(endf):
    """Every MT that has MF6/LAW=1 ND>0 subsections carrying gammas
    and passes contains_zap for gamma."""
    if 6 not in endf:
        return []
    out = []
    for mt in sorted(endf[6].keys()):
        if not selectors.contains_zap(endf, mt, GAMMA_ZAP):
            continue
        if selectors.has_mf6_law1_discrete_lines(endf, mt, GAMMA_ZAP):
            out.append(mt)
    return out


def _run_folder_ddx(endf, mt, cfg):
    """LAW=1 discrete-line folder result and, when the same MT/ZAP
    also has continuum content, its continuous-folder result. The
    two share the full MF6 yield (LAW=1 subsecs partition the
    distribution through their b(k) amplitudes rather than through
    the yield), so the discrete + continuum integrals sum to
    xs * yield for correct handling."""
    einc = np.array([cfg['e_in']])
    eouts = np.linspace(cfg['eout_min'], cfg['eout_max'], cfg['n_eouts'])
    mus = np.linspace(-1.0, 1.0, cfg['n_mus'])
    sigma = cfg['sigma']
    ddx_disc = ddxb.compute_ddx_law1_discrete_broadened(
        endf, mt=mt, zap=GAMMA_ZAP,
        energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
        kernel=lambda d: _gaussian(d, sigma),
    )
    ddx_cont = None
    if selectors.has_continuous_ddx(endf, mt, GAMMA_ZAP):
        try:
            ddx_cont = ddxb.compute_ddx_continuous_broadened(
                endf, mt=mt, zap=GAMMA_ZAP,
                energies_in=einc, energies_out=eouts, angle_cosines_out=mus,
                kernel=lambda d: _gaussian(d, sigma),
                kernel_width=sigma,
            )
        except Exception:
            ddx_cont = None
    return einc, eouts, mus, ddx_disc, ddx_cont


def _run_folder_dxs_de(endf, mt, cfg):
    einc = np.array([cfg['e_in']])
    eouts = np.linspace(cfg['eout_min'], cfg['eout_max'], cfg['n_eouts'])
    sigma = cfg['sigma']
    dexs_disc = ddxb.compute_dxs_dE_law1_discrete_broadened(
        endf, mt=mt, zap=GAMMA_ZAP,
        energies_in=einc, energies_out=eouts,
        kernel=lambda d: _gaussian(d, sigma),
    )
    dexs_cont = ddxb.compute_dxs_dE_broadened(
        endf, mt=mt, zap=GAMMA_ZAP,
        energies_in=einc, energies_out=eouts,
        kernel=lambda d: _gaussian(d, sigma),
        kernel_width=sigma,
    )
    return einc, eouts, dexs_disc, dexs_cont


def _reference_production_xs(endf, mt, einc):
    """xs(MT) * yield(MT, gamma) at each incident energy. Falls back
    to a zero array if yields aren't available (some MT/gamma
    combinations aren't tabulated -- treat as no reference)."""
    xs = mf3_interp.compute_cross_section(endf, mt, einc)
    try:
        y = compute_yields(endf, mt, GAMMA_ZAP, einc, include_discrete=True)
    except Exception:
        return None
    return xs * y


# The expected corpus, keyed off FILE_CFG so pytest always shows a
# parametrised entry per candidate file even when the folder is empty.
# Each entry either yields the parsed dict or skips with a message
# pointing at the fetch script.
EXPECTED_FILES = sorted(FILE_CFG.keys())


@pytest.fixture(scope='module', params=EXPECTED_FILES, ids=lambda n: n)
def endf_context(request):
    name = request.param
    fn = DATA_DIR / name
    if not fn.exists():
        pytest.skip(
            f'{name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh` to populate the corpus'
        )
    endf = _parse(fn)
    cfg = FILE_CFG[name]
    return fn, endf, cfg


def test_ddx_folder_finite_and_integrates_to_xs_times_yield(endf_context):
    """For every MT with MF6/LAW=1 ND>0 gamma content in the file:
    the LAW=1 DDX folder returns finite, non-negative values, and
    the integral over (E_out, Omega) matches xs * yield within the
    file's tolerance."""
    fn, endf, cfg = endf_context
    admitted = _admitted_gamma_mts(endf)
    if not admitted:
        pytest.skip(f'{fn.name}: no MF6/LAW=1 ND>0 gamma MTs')

    failures = []
    checks = 0
    for mt in admitted:
        try:
            einc, eouts, mus, ddx_disc, ddx_cont = _run_folder_ddx(endf, mt, cfg)
        except Exception as exc:
            failures.append(f'MT={mt}: folder raised {type(exc).__name__}: {exc}')
            continue
        for label, arr in (('LAW=1 disc', ddx_disc), ('cont', ddx_cont)):
            if arr is None:
                continue
            if np.any(np.isnan(arr)):
                failures.append(f'MT={mt}: NaN in {label} DDX')
                break
            # Tolerate FFT numerical noise (~1e-15 relative to peak).
            peak = float(np.abs(arr).max())
            if np.any(arr < -1e-10 * peak - 1e-30):
                failures.append(f'MT={mt}: negative value in {label} DDX')
                break
        total = ddx_disc if ddx_cont is None else ddx_disc + ddx_cont
        inner = np.trapezoid(total[0], eouts, axis=0)
        integ = np.trapezoid(inner, mus) * 2 * np.pi
        ref = _reference_production_xs(endf, mt, einc)
        if ref is None:
            continue
        checks += 1
        ref_val = float(ref[0])
        if ref_val <= 0:
            if integ > 1e-6:
                failures.append(
                    f'MT={mt}: ref xs*yield=0 but integ={integ:.3e}'
                )
            continue
        rel_err = abs(integ - ref_val) / ref_val
        if rel_err > cfg['rtol']:
            failures.append(
                f'MT={mt}: integ={integ:.4e} vs ref xs*yield={ref_val:.4e} '
                f'(rel err {rel_err:.2%}, tol {cfg["rtol"]:.1%})'
            )
    if failures:
        joined = '\n  '.join(failures)
        pytest.fail(
            f'{fn.name}: {len(failures)} MT(s) failed the DDX check '
            f'({checks} MTs compared to reference):\n  {joined}'
        )


def test_dxs_dE_folder_finite_and_integrates_to_xs_times_yield(endf_context):
    """1D analogue of the DDX check: for every MT with MF6/LAW=1 ND>0
    gamma content, discrete + continuum 1D folders sum to a spectrum
    whose integral matches xs * yield."""
    fn, endf, cfg = endf_context
    admitted = _admitted_gamma_mts(endf)
    if not admitted:
        pytest.skip(f'{fn.name}: no MF6/LAW=1 ND>0 gamma MTs')

    failures = []
    checks = 0
    for mt in admitted:
        try:
            einc, eouts, dexs_disc, dexs_cont = _run_folder_dxs_de(endf, mt, cfg)
        except Exception as exc:
            failures.append(f'MT={mt}: folder raised {type(exc).__name__}: {exc}')
            continue
        for label, arr in (('LAW=1 disc', dexs_disc), ('cont', dexs_cont)):
            if np.any(np.isnan(arr)):
                failures.append(f'MT={mt}: NaN in {label} dxs/dE')
                break
            peak = float(np.abs(arr).max())
            if np.any(arr < -1e-10 * peak - 1e-30):
                failures.append(f'MT={mt}: negative value in {label} dxs/dE')
                break
        total = dexs_disc + dexs_cont
        integ = np.trapezoid(total[0], eouts)
        ref = _reference_production_xs(endf, mt, einc)
        if ref is None:
            continue
        checks += 1
        ref_val = float(ref[0])
        if ref_val <= 0:
            if integ > 1e-6:
                failures.append(
                    f'MT={mt}: ref xs*yield=0 but integ={integ:.3e}'
                )
            continue
        rel_err = abs(integ - ref_val) / ref_val
        if rel_err > cfg['rtol']:
            failures.append(
                f'MT={mt}: integ={integ:.4e} vs ref xs*yield={ref_val:.4e} '
                f'(rel err {rel_err:.2%}, tol {cfg["rtol"]:.1%})'
            )
    if failures:
        joined = '\n  '.join(failures)
        pytest.fail(
            f'{fn.name}: {len(failures)} MT(s) failed the 1D check '
            f'({checks} MTs compared to reference):\n  {joined}'
        )


def test_public_ddx_dispatcher_runs_end_to_end(endf_context):
    """Smoke test on the public API: get_particle_production_ddxs
    with broadening on completes, is finite, non-negative, no
    warnings emitted about LAW=1. Uses reaction='(n,total)' to admit
    every neutron-emitting MT (which for these gamma-heavy files
    still includes the LAW=1 gamma path via the ZAP='g' query)."""
    from endf_userpy.quantities import get_particle_production_ddxs
    fn, endf, cfg = endf_context
    einc = np.array([cfg['e_in']])
    eouts = np.linspace(cfg['eout_min'], cfg['eout_max'], 201)
    mus = np.linspace(-1.0, 1.0, 15)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        try:
            result = get_particle_production_ddxs(
                endf, '(n,total)', 'g', einc, eouts, mus,
                broadening=cfg['sigma'],
            )
        except Exception as exc:
            pytest.fail(f'public API raised: {type(exc).__name__}: {exc}')
    assert result is not None, 'no MTs admitted for gamma production'
    assert not np.any(np.isnan(result)), 'NaN in dispatcher result'
    peak = float(np.abs(result).max())
    assert not np.any(result < -1e-10 * peak - 1e-30), (
        'negative in dispatcher result (beyond FFT noise tolerance)'
    )
    law1 = [w for w in recorded if 'MF6/LAW=1' in str(w.message)]
    assert not law1, f'unexpected LAW=1 warning: {law1[0].message}'


if __name__ == '__main__':
    # Human-readable summary when run as a script.
    for name in EXPECTED_FILES:
        fn = DATA_DIR / name
        if not fn.exists():
            print(f'{name}: missing (run tests/data_law1_adhoc/fetch.sh)')
            continue
        try:
            endf = _parse(fn)
        except Exception as exc:
            print(f'{name}: parse failed: {exc}')
            continue
        admitted = _admitted_gamma_mts(endf)
        print(f'{name}: {len(admitted)} MT(s) with MF6/LAW=1 ND>0 gamma content')
        cfg = FILE_CFG[name]
        for mt in admitted[:5]:
            einc, eouts, mus, ddx_disc, ddx_cont = _run_folder_ddx(endf, mt, cfg)
            total = ddx_disc if ddx_cont is None else ddx_disc + ddx_cont
            inner = np.trapezoid(total[0], eouts, axis=0)
            integ_total = np.trapezoid(inner, mus) * 2 * np.pi
            inner_d = np.trapezoid(ddx_disc[0], eouts, axis=0)
            integ_disc = np.trapezoid(inner_d, mus) * 2 * np.pi
            ref = _reference_production_xs(endf, mt, einc)
            ref_val = None if ref is None else float(ref[0])
            mix = ' (mixed)' if ddx_cont is not None else ''
            print(
                f'  MT={mt:>3d}{mix}: disc={integ_disc:.4e} total={integ_total:.4e} b'
                + ('' if ref_val is None else f' (ref xs*yield={ref_val:.4e} b)')
            )
        if len(admitted) > 5:
            print(f'  ... and {len(admitted)-5} more MT(s)')
