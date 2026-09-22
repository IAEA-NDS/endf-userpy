"""Bit-for-bit equivalence between the pure-Python MF6 LAW=2
(discrete two-body reaction) angular-distribution reconstruction
and the Fortran-backed reference implementation, plus a JAX-parity
smoke test on the ported backend-agnostic path.

Mirrors ``test_mf6_law6_python_fortran_equivalence.py``.

Corpus: ENDF/B-VIII.1 Al-27 (fetch-on-demand;
``tests/data_law1_adhoc/endfb81_n_Al-27.endf``). Al-27 has a long
discrete-level inelastic-scattering ladder MT=51..80, each written
as MF6 LAW=2 LANG=0 (Legendre) with a neutron ejectile
(``ZAP=1.0``); this is the canonical non-gamma-ZAP LAW=2 case that
exercises the Legendre + LAB-CM Jacobian + kinematic-forbidden-
region clipping path end-to-end. LANG=12 / LANG=14 tabulated
subsections are rare in real evaluations and don't appear in
either corpus; the code path is covered by a hand-built synthetic
dict below.
"""
from pathlib import Path
import numpy as np
import pytest
import warnings

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_subsecs as mf6py,
    mf6_interpretation_subsecs_fort as mf6fort,
)
from endf_userpy.primitives import array_ns

from _corpus import resolve_al27


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip(
            'Al-27 ENDF file not available (set AL27_ENDF, run '
            'tests/data_law1_adhoc/fetch.sh, or place the file at '
            'tests/data_law1_adhoc/endfb81_n_Al-27.endf)'
        )
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.mark.parametrize('mt', [51, 52, 53])
def test_law2_lang0_python_fortran_equivalence(al27_endf_dict, mt):
    """Al-27 discrete inelastic MT=51/52/53 MF6 LAW=2 LANG=0
    (Legendre). Pure-Python path must match the Fortran path to
    machine precision at every (E, mu) query point, since both
    evaluate the same closed-form combination of Legendre-in-CM +
    lin-lin panel interpolation + Jacobian to LAB.
    """
    E_inc = np.array([5.0e6, 1.0e7, 1.5e7, 2.0e7])
    mu = np.linspace(-1.0, 1.0, 9)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        py_res = np.asarray(mf6py.get_angdist_from_subsec_law2(
            al27_endf_dict, mt, 1, E_inc, mu, True,
        ))
        fort_res = mf6fort.get_angdist_from_subsec_law2(
            al27_endf_dict, mt, 1, E_inc, mu, True,
        )
    assert py_res.shape == (len(E_inc), len(mu))
    assert py_res.shape == fort_res.shape
    np.testing.assert_allclose(
        py_res, fort_res, rtol=1e-12, atol=1e-14, equal_nan=True,
    )


def test_law2_gamma_zap_short_circuit_returns_zeros():
    """Photon LAW=2 (ZAP=0) short-circuits to zeros with a
    UserWarning (issue #78). Pre-port and post-port both take this
    branch; pins the guard so nobody removes it during future
    cleanup."""
    parser = EndfParserCpp(ignore_missing_tpid=True)
    data_dir = Path(__file__).resolve().parent / 'data'
    d = parser.parsefile(data_dir / 'n-001_H_001.endf')
    E_inc = np.array([5.0e4])
    mu = np.array([-0.5, 0.5])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = np.asarray(mf6py.get_angdist_from_subsec_law2(
            d, 102, 1, E_inc, mu, True,
        ))
    assert out.shape == (1, 2)
    np.testing.assert_array_equal(out, np.zeros((1, 2)))
    zap_warnings = [w for w in caught if 'ZAP=0' in str(w.message)]
    assert len(zap_warnings) == 1


def _synthetic_mf6_law2_lang12_dict():
    """Hand-built MF6/MT=51 subsection with LAW=2 LANG=12
    (tabulated lin-lin in mu). Covers the tabulated code path
    which the shipped corpus does not exercise."""
    return {
        1: {451: {
            'ZA': 26056, 'AWR': 55.454, 'NSUB': 10, 'AWI': 1.0,
        }},
        6: {51: {
            'LCT': 1,   # LAB, no CM->LAB conversion, keeps the test
                        # focused on the tabulated evaluation itself
            'subsection': {
                1: {
                    'ZAP': 1.0, 'AWP': 1.0, 'LAW': 2, 'LANG': 12,
                    'NR': 1, 'NE': 2,
                    'NBT': [2], 'INT': [2],
                    'E': {1: 1.0e6, 2: 2.0e7},
                    'NL': {1: 3, 2: 3},
                    'A': {
                        # [u1, p1, u2, p2, u3, p3] per panel
                        1: [-1.0, 0.25, 0.0, 0.5, 1.0, 0.75],
                        2: [-1.0, 0.1, 0.0, 0.5, 1.0, 0.9],
                    },
                },
            },
        }},
        3: {51: {'QI': -0.5, 'QM': -0.5, 'LR': 0}},
    }


def test_law2_lang12_tabulated_evaluates_at_expected_endpoints():
    """LANG=12 tabulated angular distribution: at exactly a mu
    endpoint of the tab1 record, the Python path must return that
    tabulated value (no interpolation smoothing). Pins the LANG=12
    branch of `_law2_tab1_records` + `interp_tab2` on a synthetic
    dict without needing a corpus file.
    """
    d = _synthetic_mf6_law2_lang12_dict()
    E_inc = np.array([1.0e6, 2.0e7])   # exactly at the two panel knots
    mu = np.array([-1.0, 0.0, 1.0])
    out = np.asarray(mf6py.get_angdist_from_subsec_law2(
        d, 51, 1, E_inc, mu, True,
    ))
    # At E=1e6, mu=(-1, 0, 1): expected (0.25, 0.5, 0.75)
    np.testing.assert_allclose(out[0], [0.25, 0.5, 0.75], rtol=1e-12)
    # At E=2e7, mu=(-1, 0, 1): expected (0.1, 0.5, 0.9)
    np.testing.assert_allclose(out[1], [0.1, 0.5, 0.9], rtol=1e-12)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law2_numpy_jax_parity(al27_endf_dict):
    """LAW=2 LANG=0 Al-27 path: numpy vs JAX backend must return
    the same numbers to machine precision. Exercises the whole
    ported chain (`convert_angcos_to_cmsys`,
    `evaluate_interp_legendre_polynomials`,
    `convert_angdist_to_labsys`) on the JAX backend end-to-end."""
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')
    E_inc = np.array([5.0e6, 1.0e7])
    mu = np.linspace(-0.9, 0.9, 7)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        out_np = np.asarray(mf6py.get_angdist_from_subsec_law2(
            al27_endf_dict, 51, 1, E_inc, mu, True, xp=xp_np,
        ))
        out_jax = np.asarray(mf6py.get_angdist_from_subsec_law2(
            al27_endf_dict, 51, 1, E_inc, mu, True, xp=xp_jax,
        ))
    np.testing.assert_allclose(
        out_np, out_jax, rtol=1e-12, atol=1e-14, equal_nan=True,
    )
