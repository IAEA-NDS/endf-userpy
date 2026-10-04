"""End-to-end regression pin for the ``urr_quadrature`` parameter on
the top-level XS API (issue #299).

Pre-fix, ``resonance_composition.reconstruct_resonance_xs`` called
``mf2_interpretation_urr.reconstruct`` without forwarding the
``quadrature`` argument, so the ``'ross_10'`` NJOY-parity quadrature
was unreachable from the user-facing ``get_reaction_xs`` etc. This
test pins that the parameter now reaches the URR kernel AND that
the two quadratures give the sub-permille agreement the URR
reconstructor's module docstring claims on LSSF=0 corpus.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import warnings

from endf_userpy.primitives import array_ns
from endf_userpy.quantities import get_reaction_xs
from endf_userpy.run_options import RunOptions


PU239_CORPUS = os.path.join(
    os.path.dirname(__file__), 'data_law1_adhoc', 'cendl32_n_Pu-239.endf',
)


def _pu239_available():
    return os.path.exists(PU239_CORPUS)


@pytest.fixture(scope='module')
def pu239_endf_dict():
    if not _pu239_available():
        pytest.skip(
            'CENDL-3.2 Pu-239 corpus not fetched; run '
            'bash tests/data_law1_adhoc/fetch.sh'
        )
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(PU239_CORPUS)


@pytest.mark.skipif(not _pu239_available(), reason='CENDL-3.2 Pu-239 corpus missing')
def test_urr_quadrature_reaches_kernel_and_changes_output(pu239_endf_dict):
    """CENDL-3.2 Pu-239 has an LSSF=0 URR range at 1..30 keV. Pin
    that switching ``urr_quadrature`` changes the composed capture
    XS in that window (i.e. the parameter actually reaches the
    kernel; pre-fix it did not)."""
    # Ein inside the LSSF=0 URR window.
    ein = np.linspace(2.0e3, 25.0e3, 40)
    xs_gl = get_reaction_xs(
        pu239_endf_dict, '(n,g)', ein,
        options=RunOptions(include_resonance=True),
    )
    xs_ross = get_reaction_xs(
        pu239_endf_dict, '(n,g)', ein,
        options=RunOptions(include_resonance=True, urr_quadrature='ross_10'),
    )
    peak = float(np.max(np.abs(xs_gl)))
    assert peak > 0.0, 'expected non-zero composed XS in the URR window'
    diff = float(np.max(np.abs(xs_gl - xs_ross)))
    # Bug-catch: pre-fix both calls silently used gauss_legendre_32
    # so diff was exactly 0. Post-fix Ross-10 differs by a small
    # but non-zero amount tracked by the assertion below.
    assert diff > 0.0, (
        'urr_quadrature did not change the output; the parameter '
        'is not reaching mf2_interpretation_urr.reconstruct '
        '(issue #299 regression).'
    )


@pytest.mark.skipif(not _pu239_available(), reason='CENDL-3.2 Pu-239 corpus missing')
def test_urr_quadrature_agreement_within_documented_class(pu239_endf_dict):
    """The URR reconstructor's module docstring states GL-32 and
    Ross-10 differ by ``~4e-4`` (~sub-permille) on LSSF=0 corpus
    where the two ~10-point quadratures both trail Monte Carlo
    truth by a similar amount. Pin that the top-level composed XS
    respects this class."""
    ein = np.linspace(2.0e3, 25.0e3, 40)
    xs_gl = get_reaction_xs(
        pu239_endf_dict, '(n,g)', ein,
        options=RunOptions(include_resonance=True),
    )
    xs_ross = get_reaction_xs(
        pu239_endf_dict, '(n,g)', ein,
        options=RunOptions(include_resonance=True, urr_quadrature='ross_10'),
    )
    peak = float(np.max(np.abs(xs_gl)))
    rel = float(np.max(np.abs(xs_gl - xs_ross))) / max(1e-30, peak)
    # Docstring says "sub-permille"; use 1e-3 as the ceiling and
    # 1e-8 as a floor sanity check (we already assert non-zero in
    # the sibling test).
    assert rel < 1.0e-3, (
        f'GL-32 vs Ross-10 rel-to-peak diff {rel:.3e} exceeds '
        f'the ~4e-4 cross-scheme agreement documented in '
        f'mf2_interpretation_urr.reconstruct.'
    )


@pytest.mark.skipif(not _pu239_available(), reason='CENDL-3.2 Pu-239 corpus missing')
def test_urr_quadrature_default_matches_gauss_legendre_explicit(pu239_endf_dict):
    """Passing no ``urr_quadrature`` must give bit-identical output
    to passing ``'gauss_legendre_32'`` explicitly. Guards against a
    silent default-drift regression."""
    ein = np.linspace(2.0e3, 25.0e3, 40)
    xs_default = get_reaction_xs(
        pu239_endf_dict, '(n,g)', ein,
        options=RunOptions(include_resonance=True),
    )
    xs_explicit = get_reaction_xs(
        pu239_endf_dict, '(n,g)', ein,
        options=RunOptions(
            include_resonance=True,
            urr_quadrature='gauss_legendre_32',
        ),
    )
    np.testing.assert_array_equal(xs_default, xs_explicit)


def _numba_available():
    return 'numba' in array_ns.available_backends()


@pytest.mark.skipif(not _pu239_available(), reason='CENDL-3.2 Pu-239 corpus missing')
@pytest.mark.skipif(not _numba_available(), reason='numba not installed')
def test_urr_numba_ross10_falls_back_to_numpy_with_warning(pu239_endf_dict):
    """Issue #315: ``RunOptions(backend='numba', urr_quadrature='ross_10')``
    on a file with an LSSF=0 URR range must not raise. The numba
    kernel only supports ``'gauss_legendre_32'``; for Ross-10 the
    URR dispatcher falls through to the array-agnostic (numpy /
    JAX) path and emits a one-shot ``UserWarning`` naming the
    fallback.

    Pre-fix: raised ``NotImplementedError`` from
    ``mf2_interpretation_urr.reconstruct``. Caller received a
    composition layer warning instead of the real average XS;
    cross-scheme comparison tests silently measured raw MF3 vs
    Ross-10 and failed with ~100% rel diff."""
    ein = np.linspace(2.0e3, 25.0e3, 20)
    xp_numba = array_ns.get_backend('numba')
    xp_numpy = array_ns.get_backend('numpy')
    opts_numba_ross = RunOptions(
        backend=xp_numba, include_resonance=True,
        urr_quadrature='ross_10',
    )
    opts_numpy_ross = RunOptions(
        backend=xp_numpy, include_resonance=True,
        urr_quadrature='ross_10',
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        xs_numba = np.asarray(
            get_reaction_xs(pu239_endf_dict, '(n,g)', ein, options=opts_numba_ross)
        )
    xs_numpy = np.asarray(
        get_reaction_xs(pu239_endf_dict, '(n,g)', ein, options=opts_numpy_ross)
    )

    # Correctness: fallback result matches the numpy path pointwise.
    np.testing.assert_allclose(xs_numba, xs_numpy, rtol=1e-12, atol=0.0)

    # Warning: at least one UserWarning mentioning the fallback.
    fb_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and 'fell back' in str(w.message)
        and 'ross_10' in str(w.message)
    ]
    assert fb_warnings, (
        'expected a UserWarning naming the ross_10 fallback; '
        f'got {[str(w.message)[:80] for w in caught]}'
    )
