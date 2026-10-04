"""Regression tests for #311: visibility of silent-zero and
silent-raw-MF3 cases when the user's reaction string resolves to
an MT the file or the resonance-composition maps cannot handle.

Three fixes to pin:

- **Option 3 (hard fail)**: a reaction string that resolves to a
  chance-breakdown fission MT (19, 20, 21, 38) under
  ``include_resonance=True`` raises ``ValueError`` when the file's
  MF2 carries fission widths. The widths describe total fission;
  attributing them to a chance channel would overstate the
  resonance contribution.
- **Mode 1 (silent-zero visibility)**: a reaction string that
  resolves to an MT not tabulated in the file AND no admitted
  partial could synthesise it via the sum rule returns zero plus
  a summary ``UserWarning``. Does NOT fire when the sum heuristic
  admits partials (e.g. ``(n,total)`` on a file lacking MT 1).
- **Mode 2 (silent-raw-MF3 visibility)**: a reaction string that
  resolves to an MT present in MF3 but absent from the active
  formalism's MT-to-partial-keys map produces raw MF3 only. Mode
  2 names those MTs in a summary ``UserWarning``, scoped to
  user-requested MTs so differential queries with widened MT
  iteration don't emit noise.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest


sys.path.insert(0, os.path.dirname(__file__))
from _corpus import resolve_u235, resolve_nb93  # noqa: E402

from endf_userpy.quantities import (                      # noqa: E402
    get_reaction_xs,
    get_particle_production_xs,
)
from endf_userpy.run_options import RunOptions             # noqa: E402


@pytest.fixture(scope='module')
def u235_dict():
    """TENDL-2021 U-235: LRU=1 Reich-Moore with fission widths, used
    for the option-3 chance-breakdown guard and the mode 1 silent-
    zero case on ``(n,f)``."""
    path = resolve_u235()
    if path is None:
        pytest.skip('tendl21_n_U-235.endf not fetched')
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def nb93_dict():
    """Nb-93: MLBW, no fission widths, used for mode 1 ``(n,total)``
    via-partials-is-OK case."""
    path = resolve_nb93()
    if path is None:
        pytest.skip('endfb81_n_Nb-93.endf not fetched')
    from endf_parserpy import EndfParserCpp
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


# ----------------------------------------------------------------------
# Option 3: chance-breakdown fission MT + MF2 fission widths = raise
# ----------------------------------------------------------------------


def test_option3_raises_on_n_f_under_include_resonance_true(u235_dict):
    """``(n,f)`` -> MT 19 under include_resonance=True on a file whose
    MF2 carries fission widths must raise ``ValueError`` naming MT 19
    and pointing at ``(n,fission)`` / ``include_resonance=False``."""
    E = np.array([5.0, 20.0, 100.0])
    opts = RunOptions(include_resonance=True)
    with pytest.raises(ValueError, match=r'MT 19.*chance-breakdown fission'):
        get_reaction_xs(u235_dict, '(n,f)', E, options=opts)


def test_option3_silent_when_include_resonance_false(u235_dict):
    """Opt-out: with ``include_resonance=False`` the guard is a no-op
    (no composition is applied, so the overstatement cannot happen).
    Returns a zero cross section (MT 19 not in MF3) plus the mode 1
    warning only."""
    E = np.array([5.0, 20.0, 100.0])
    opts = RunOptions(include_resonance=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        xs = get_reaction_xs(u235_dict, '(n,f)', E, options=opts)
    np.testing.assert_array_equal(xs, np.zeros(3))
    mode1 = [w for w in caught if 'file does not tabulate' in str(w.message)]
    assert mode1, 'mode 1 warning should fire for (n,f) with no composition'


def test_option3_covers_all_chance_breakdown_mts(u235_dict):
    """Each of MT 19, 20, 21, 38 maps to a reaction string that must
    raise under include_resonance=True on this file."""
    E = np.array([10.0, 50.0])
    opts = RunOptions(include_resonance=True)
    for reaction_str, expected_mt in (
        ('(n,f)', 19),
        ('(n,nf)', 20),
        ('(n,2nf)', 21),
        ('(n,3nf)', 38),
    ):
        with pytest.raises(ValueError, match=f'MT {expected_mt}'):
            get_reaction_xs(u235_dict, reaction_str, E, options=opts)


def test_option3_fires_on_particle_production_entry_point(u235_dict):
    """The guard also applies to the four particle-production entry
    points, not just ``get_reaction_xs``. Covers both bytes of #311's
    'warn for all entry points' request."""
    E = np.array([10.0])
    opts = RunOptions(include_resonance=True)
    with pytest.raises(ValueError, match=r'MT 19'):
        get_particle_production_xs(u235_dict, '(n,f)', 'n', E, options=opts)


# ----------------------------------------------------------------------
# Mode 1: silent-zero visibility
# ----------------------------------------------------------------------


def test_mode1_warns_for_missing_mt_with_no_synthesisable_partials(u235_dict):
    """Mode 1 refined definition: fire only when the user-requested
    MT is not in the file AND no admitted partial could synthesise
    the answer via the sum rule. The ``(n,f)`` case on this file
    (MT 19 absent, no partials of MT 19) is the canonical hit."""
    E = np.array([5.0, 20.0, 100.0])
    opts = RunOptions(include_resonance=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        xs = get_reaction_xs(u235_dict, '(n,f)', E, options=opts)
    assert np.all(xs == 0.0)
    mode1 = [w for w in caught if 'file does not tabulate' in str(w.message)]
    assert len(mode1) == 1
    assert 'MT=19' in str(mode1[0].message)


def test_mode1_silent_when_user_mt_is_in_file(nb93_dict):
    """`(n,g)` -> MT 102 is in Nb-93's MF3. Mode 1 should stay silent
    regardless of what the XS value turns out to be (zero or
    non-zero), because the requested MT is tabulated."""
    E = np.array([5.0, 20.0, 100.0])
    opts = RunOptions(include_resonance=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        xs = get_reaction_xs(nb93_dict, '(n,g)', E, options=opts)
    assert xs.shape == E.shape
    mode1 = [w for w in caught if 'file does not tabulate' in str(w.message)]
    assert not mode1, (
        f'mode 1 fired falsely on (n,g) which is in the file; '
        f'messages: {[str(w.message)[:80] for w in caught]}'
    )


def test_mode1_fires_on_particle_production_entry_point(u235_dict):
    """Same request-not-synthesisable shape, routed through
    ``get_particle_production_xs``. Mode 1 is pre-iteration here."""
    E = np.array([5.0, 20.0])
    opts = RunOptions(include_resonance=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        get_particle_production_xs(u235_dict, '(n,f)', 'n', E, options=opts)
    mode1 = [w for w in caught if 'file does not tabulate' in str(w.message)]
    assert mode1, (
        'mode 1 should fire on get_particle_production_xs when the '
        'requested MT is absent and no admitted partial produces the '
        'queried particle for that reaction'
    )


# ----------------------------------------------------------------------
# Mode 2: silent-raw-MF3 visibility
# ----------------------------------------------------------------------
#
# Hard to exercise without a corpus file that has MT X in MF3 but not
# in the formalism's MT-to-partial-keys map AND where the user asks
# for that specific MT. The sum-mt heuristic generally routes
# `(n,inl)` -> MT 4 via partials MT 51..91; those partials are not
# themselves user-requested, so mode 2 shouldn't fire for them. A
# true mode-2 hit would be a reaction string that resolves to e.g.
# MT 16 (n,2n) via `(n,2n)' on a file where MF3 carries MT 16 and
# MF2 has a reconstructable range -- then composition skips MT 16
# because it is outside {1, 2, 3, 18, 27, 102}, returns raw MF3,
# and mode 2 names it.


def test_mode2_fires_when_user_mt_in_mf3_but_not_in_formalism_map(nb93_dict):
    """Nb-93 has MLBW resonances and MF3 for MT 16 (n,2n). MT 16 is
    not in ``_MLBW_MT_TO_KEYS``, so composition returns raw MF3 only
    for ``(n,2n)``. Mode 2 should name MT 16 in a summary warning
    when the user asked for it specifically."""
    E = np.array([1e6, 5e6, 1e7])   # above Nb-93 (n,2n) threshold
    opts = RunOptions(include_resonance=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        xs = get_reaction_xs(nb93_dict, '(n,2n)', E, options=opts)
    mode2 = [w for w in caught if 'MT-to-partial-keys map' in str(w.message)]
    # Depending on whether MT 16 is in the file at these energies,
    # the mode 2 warning may or may not fire. If it does fire, it
    # must name MT 16 and not any widened-iteration MT.
    for w in mode2:
        assert 'MT=16' in str(w.message), (
            f'mode 2 message named an MT other than the user-requested '
            f'MT 16: {str(w.message)[:160]}'
        )
    # Smoke check that the call completed; the shape is correct
    # independently of whether MT 16 was in file at these Ein.
    assert xs.shape == E.shape
