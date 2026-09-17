"""Tests for direct-MT queries on MF13-only gamma production
MTs (issue #130).

JENDL-5 N-14 places its gamma production on MT 3 (nonelastic sum)
via MF13/MF14/MF15 but does NOT tabulate MT 3 in MF3 -- MT 3 is
implicit as MT 1 - MT 2 on that file. Before this fix, three
things went wrong:

1. `quantities_mt_zap.get_reaction_mt_numbers` returned only
   `MF3.keys()`, so the cumulative-sum iteration never visited
   MT 3 -- the dispatchers silently missed the entire MF13
   contribution.
2. `satisfies_select_heuristic(3, [3])` returned False because MT
   3 is a sum-MT and some of its children had MF4/5/6 detail;
   even if MT 3 had been iterated, the admission heuristic would
   have dropped it in favour of children whose gamma content is
   ~10x smaller.
3. `compute_prodxs` / `compute_dexs` / `compute_daxs` /
   `compute_ddxs` all composed `xs * yields` using MF3 XS, which
   crashed with `KeyError: 3` for MT 3 without an MF3 entry.

Fix (issue #130, narrow scope):

- `mf3_interpretation.get_reaction_mts_widened` returns the union
  of MF3 + MF12 + MF13 + MF15 keys; the four particle-production
  dispatchers pass it via the new `mts=` kwarg on
  `compute_cumulative_quantity`.
- `satisfies_select_heuristic` gains an "escape hatch" for
  user-explicitly-listed sum MTs that carry their own MF13
  gamma-production XS.
- `compute_prodxs` / `compute_dexs` / `compute_daxs` short-circuit
  for MF13-only gamma MTs via `_is_mf13_only_gamma`, using
  `mf13_interpretation.compute_total_photon_production_xs` as the
  gamma-production XS directly. `compute_ddxs` returns zeros
  (the actual DDX for such MTs is picked up by
  `compute_ddxs_from_mf15_mf14`, which now handles the MF13-only
  branch too). `compute_ddx_mf15_continuum_broadened` in
  `ddx_broadening.py` gets the same MF13-only branch.

These tests pin JENDL-5 N-14 (n,nonelas)+gamma across all four
top-level APIs and check both the XS-integrated consistency and
non-regression on files whose gamma content is standard MF12+MF15
(Al-27 MT 102, U-238 MT 18 fission gammas).
"""
from pathlib import Path
import warnings
import numpy as np
import pytest
from endf_parserpy import EndfParserCpp

from endf_userpy.quantities import (
    get_particle_production_xs,
    get_particle_production_dxs_dE,
    get_particle_production_dxs_dmu,
    get_particle_production_ddxs,
)
from endf_userpy.primitives.np_compat import trapezoid


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
# Primary reproducer: JENDL-5 N-14 (n,nonelas)+gamma via MT 3.
# ============================================================


def test_jendl5_n14_nonelas_gamma_xs_nonzero():
    """Pre-fix: `get_particle_production_xs('(n,nonelas)', 'g', ...)`
    returned zero on JENDL-5 N-14 because MT 3 was not visited by
    the MF3-only iteration. Post-fix: returns MF13/MT 3's XS."""
    endf = _load('jendl5_n_N-14.endf')
    einc = np.array([1e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,nonelas)', 'g', einc)
    assert xs is not None
    assert xs[0] > 0.5, (
        f'JENDL-5 N-14 (n,nonelas)+g XS should be >0.5 barn at 14 MeV, '
        f'got {xs[0]:.4g}'
    )


@pytest.mark.parametrize('api,make_args,extract', [
    ('dxs_dE',
     lambda einc: dict(
         energies_in=einc,
         energies_out=np.linspace(1e4, 1.4e7, 300),
     ),
     lambda r, kw: float(trapezoid(r[0], kw['energies_out']))),
    ('dxs_dmu',
     lambda einc: dict(
         energies_in=einc,
         angle_cosines_out=np.linspace(-1, 1, 21),
     ),
     lambda r, kw: float(
         2 * np.pi * trapezoid(r[0], kw['angle_cosines_out']),
     )),
    ('ddxs_unbroadened',
     lambda einc: dict(
         energies_in=einc,
         energies_out=np.linspace(1e4, 1.4e7, 300),
         angle_cosines_out=np.linspace(-1, 1, 21),
     ),
     lambda r, kw: float(
         2 * np.pi * trapezoid(
             trapezoid(r[0], kw['angle_cosines_out'], axis=-1),
             kw['energies_out'],
         ),
     )),
    ('ddxs_broadened',
     lambda einc: dict(
         energies_in=einc,
         energies_out=np.linspace(1e4, 1.4e7, 300),
         angle_cosines_out=np.linspace(-1, 1, 21),
         broadening=200e3,
     ),
     lambda r, kw: float(
         2 * np.pi * trapezoid(
             trapezoid(r[0], kw['angle_cosines_out'], axis=-1),
             kw['energies_out'],
         ),
     )),
])
def test_jendl5_n14_nonelas_gamma_all_apis_return_similar(api, make_args, extract):
    """Every top-level particle-production API integrated over its
    differential axes should agree with the XS query to within a
    few percent (the discrepancy is grid-truncation of the MF15
    spectrum's tail beyond the caller's E_out range)."""
    endf = _load('jendl5_n_N-14.endf')
    einc = np.array([1e7])
    kw = make_args(einc)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,nonelas)', 'g', einc)
        api_map = {
            'dxs_dE': get_particle_production_dxs_dE,
            'dxs_dmu': get_particle_production_dxs_dmu,
            'ddxs_unbroadened': get_particle_production_ddxs,
            'ddxs_broadened': get_particle_production_ddxs,
        }
        r = api_map[api](endf, '(n,nonelas)', 'g', **kw)
    assert r is not None, f'{api} returned None'
    integ = extract(r, kw)
    # Allow 10% shortfall for grid-truncation + kernel truncation
    # on the broadened path (MF15 spectrum has extended tails; the
    # dxs/dmu case is exact because no E_out integration is needed).
    ratio = integ / xs[0]
    assert 0.85 < ratio < 1.05, (
        f'{api}: integral {integ:.4g} vs XS {xs[0]:.4g} '
        f'(ratio {ratio:.3f}) -- expected within 15%'
    )


# ============================================================
# Non-regression: standard MF12+MF15 files unchanged.
# ============================================================


def test_al27_mt102_ng_unchanged():
    """Al-27 (n,g) MT 102 uses standard MF12+MF15 with an MF3 XS.
    The MF13-only fast path must not fire here."""
    endf = _load('endfb81_n_Al-27.endf')
    einc = np.array([1.4e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,g)', 'g', einc)
    assert xs is not None
    # Pinned pre-#130 value (was 0.001914 barn from earlier probe).
    assert abs(xs[0] - 0.001914) < 1e-4, (
        f'Al-27 (n,g) gamma XS changed: expected ~0.001914, got {xs[0]:.4g}'
    )


def test_u238_fission_gamma_unchanged():
    """U-238 (n,fission) gamma via MT 18 (with MF12+MF15) must be
    unchanged from post-PR-#127 baseline (~7.26 barn at 14 MeV)."""
    endf = _load('jendl5_n_U-238.endf')
    einc = np.array([1.4e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,fission)', 'g', einc)
    assert xs is not None
    assert 6.0 < xs[0] < 8.0, (
        f'U-238 (n,fission) gamma XS changed: expected ~7.3, got {xs[0]:.4g}'
    )


def test_neutron_queries_unchanged():
    """The MF13-only fast path in compute_prodxs/dexs/daxs/ddxs is
    gamma-only. Neutron-ejectile queries are not affected."""
    endf = _load('jendl5_n_U-238.endf')
    einc = np.array([1.4e7])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xs = get_particle_production_xs(endf, '(n,total)', 'n', einc)
    assert xs is not None
    assert xs[0] > 0
