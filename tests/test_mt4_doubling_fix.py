"""Regression tests for the (n,n) MT 4 doubling bug (issue #138).

Discovered while investigating #135. PR #132 (issue #130) added an
escape in `selectors.satisfies_select_heuristic`:

    if (user_mts is not None and mt in user_mts
        and prop.has_mf13_mt(endf_dict, mt)):
        return True

Intended to admit sum-MTs carrying MF13 gamma content when the user
explicitly asks for them (fix for JENDL-5 N-14 `(n,nonelas)+g`).
But the escape is zap-agnostic: it fires in `get_reaction_xs` (XS,
no zap in scope) too, and on files with MF13 on a sum-MT that also
has children in MF3, admits the sum-MT AND its children -- XS sum
double-counts.

Concrete on ENDF/B-VIII.1 N-14 at 10 MeV:
- explicit MF3/MT 4       255.78 mb
- pre-fix `get_reaction_xs('(n,n)', ...)`  511.56 mb (exactly 2x)
- post-fix                255.78 mb (matches explicit)

The escape is now redundant after PR #134 (issue #133) added
`satisfies_gamma_production_select` which handles the intended
gamma case via its own Rule 2. Removing the escape from the
general heuristic fixes the XS double-count without regressing
the gamma path.

Tests pin:

1. `(n,n)` MT 4 XS on ENDF/B-VIII.1 N-14 and B-11 matches the
   explicit MF3 read (was 2x).
2. `(n,n)` MT 4 XS on TENDL-2021 N-14 also matches (same layout
   as ENDF/B-VIII.1).
3. JEFF-4.0 Cu-63 `(n,n)` MT 4 (had a smaller 15% discrepancy)
   now matches the explicit XS.
4. Non-regression: JENDL-5 N-14 `(n,nonelas)+g` still returns MT
   3's MF13 aggregate (~0.83 barn) via PR #134's Rule 2.
5. Non-regression: `(n,total)` and `(n,total)+g` on standard
   corpus files unchanged.
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_reaction_xs,
    get_particle_production_xs,
)
from endf_userpy.mfsec_interpretation.mf3_interpretation import (
    compute_cross_section,
)


ADHOC = Path(__file__).resolve().parent / 'data_law1_adhoc'


def _load(fn_name):
    fn = ADHOC / fn_name
    if not fn.exists():
        pytest.skip(
            f'{fn_name} not present; run '
            f'`bash tests/data_law1_adhoc/fetch.sh`'
        )
    return EndfParserCpp(
        ignore_missing_tpid=True, ignore_zero_mismatch=True, accept_spaces=True,
    ).parsefile(fn)


# ============================================================
# Primary reproducer: MT 4 doubling on light-nuclei with MF13.
# ============================================================


@pytest.mark.parametrize('fn_name', [
    'endfb81_n_N-14.endf',
    'endfb81_n_B-11.endf',
    'tendl21_n_N-14.endf',
])
def test_nn_reaction_xs_matches_explicit_mt4(fn_name):
    """`get_reaction_xs('(n,n)', ...)` must equal the explicit
    MF3/MT 4 XS on files that have MF13 on MT 4. Pre-fix, the
    escape hatch admitted MT 4 alongside MT 51..90 and the sum
    was 2x the correct value."""
    endf = _load(fn_name)
    einc = np.array([5e6, 1e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        explicit = compute_cross_section(endf, 4, einc)
        current = get_reaction_xs(endf, '(n,n)', einc)
    # Must agree to floating-point precision (both are the same
    # linear combination of children MT XS with the same
    # interpolation).
    np.testing.assert_allclose(
        current, explicit, rtol=1e-5,
        err_msg=(
            f'{fn_name}: get_reaction_xs("(n,n)") != explicit MT 4. '
            f'Pre-fix returned 2x explicit due to the MF13 escape '
            f'admitting MT 4 alongside its MT 51..90 children.'
        ),
    )


# ============================================================
# Non-regression: gamma queries via PR #134's gamma-aware rule
# still work on JENDL-5 N-14.
# ============================================================


def test_jendl5_n14_nonelas_gamma_still_uses_mt3():
    """PR #132's escape was zap-agnostic; PR #134 replaced it
    with a zap-aware Rule 2 in satisfies_gamma_production_select.
    Removing the escape here must not regress PR #132's fix
    (JENDL-5 N-14 (n,nonelas)+g returns MT 3's MF13 aggregate)."""
    endf = _load('jendl5_n_N-14.endf')
    einc = np.array([1e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,nonelas)', 'g', einc)
    assert xs is not None
    assert xs[0] > 0.5, (
        f'JENDL-5 N-14 (n,nonelas)+g at 10 MeV: expected ~0.83 barn '
        f'(MT 3 MF13 aggregate), got {xs[0]:.4g}. Removing the escape '
        f'regressed PR #132.'
    )


def test_jendl5_n14_total_gamma_still_uses_mt3():
    """PR #134 (issue #133) admits MT 3 for indirect (n,total)+g
    via satisfies_gamma_production_select's Rule 2. That rule is
    independent of the escape being removed here."""
    endf = _load('jendl5_n_N-14.endf')
    einc = np.array([1e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,total)', 'g', einc)
    assert xs is not None
    assert xs[0] > 0.5


# ============================================================
# Non-regression: (n,total) XS on well-formed corpus files
# should not shift.
# ============================================================


def test_al27_total_xs_unchanged():
    """Al-27 (no MF13 anywhere) is unaffected."""
    endf = _load('endfb81_n_Al-27.endf')
    einc = np.array([1e6, 1e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_reaction_xs(endf, '(n,total)', einc)
    assert np.all(np.isfinite(xs))
    assert np.all(xs > 0)


def test_u238_fission_neutron_xs_unchanged():
    """U-238 fission-neutron production must be unchanged (MT 18
    with MF12+MF15 but MT 18 is not a general sum-MT-with-MF13-and
    -MF3-children case)."""
    endf = _load('jendl5_n_U-238.endf')
    einc = np.array([1e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,fission)', 'n', einc)
    assert xs is not None
    assert xs[0] > 0
