"""Tests for the narrowed `is_unique_path_to_residual` ejectile
whitelist.

The pre-fix whitelist was `('g', 'n', 'p', 'd', 't', 'h', 'a')`.
`d`, `t`, `h` (He-3), and `a` (alpha) were physically incorrect
inclusions: their residual ZA on a neutron projectile is also
produced by a multi-particle-exit MT that the code correctly
rejects at the `len(ejectiles) > 1` filter but does NOT compensate
for downstream.

Concrete overlaps present in REACTION_DICT, neutron projectile
(enumerated by scanning every multi-particle MT for a matching
residual delta):

- MT104 (n,d)    residual (Z-1, A-1) also from MT28 (n,np)
- MT105 (n,t)    residual (Z-1, A-2) also from MT32 (n,nd),
                                             MT41 (n,2np)
- MT106 (n,3He)  residual (Z-2, A-2) also from MT44 (n,n2p),
                                             MT115 (n,pd)
- MT107 (n,a)    residual (Z-2, A-3) also from MT34 (n,n3He),
                                             MT116 (n,pt),
                                             MT183 (n,npd),
                                             MT190 (n,2n2p)

On files that lump both a single-particle and a multi-particle
exit into MT5 catch-all, the pre-fix rescue attributed the entire
MT5 contribution to the single-particle MT, over-counting the
(n,d)/(n,t)/(n,3He)/(n,a) query. The narrowed whitelist
`('g', 'n', 'p')` restricts the rescue to genuinely
unshared-residual MTs: gamma emission (MT102), single-proton
emission (MT103), and neutron multiplication with mult >= 2
(MT16, MT17).

These tests pin the whitelist boundary as a predicate check on
`is_unique_path_to_residual` -- they do not need a corpus file
because the function is purely about MT numbers.
"""
import pytest

from endf_userpy.primitives.reactions import (
    is_unique_path_to_residual,
    get_ejectiles,
    REACTION_DICT,
)


# ============================================================
# Post-fix: MTs that MUST still be unique paths.
# ============================================================


@pytest.mark.parametrize('mt,description', [
    (102, '(n,g) -> Z, A+1 -- unshared residual'),
    (103, '(n,p) -> Z-1, A -- unshared residual'),
    (16, '(n,2n) -> Z, A-1 -- neutron mult=2, unshared'),
    (17, '(n,3n) -> Z, A-2 -- neutron mult=3, unshared'),
])
def test_still_unique_after_narrowing(mt, description):
    """These MTs' residuals are physically unshared with any
    multi-particle-exit MT and MUST still trigger the MT5 rescue."""
    assert is_unique_path_to_residual('n', mt), (
        f'MT{mt} ({description}) should still be a unique path'
    )


# ============================================================
# Post-fix: MTs that MUST no longer be unique paths.
# ============================================================


@pytest.mark.parametrize('mt,ejectile,sibling_mt', [
    (104, 'd', 28),   # MT28 (n,np) shares MT104's residual
    (105, 't', 32),   # MT32 (n,nd) shares MT105's residual
    (106, 'h', 44),   # MT44 (n,n2p) shares MT106's residual
    (107, 'a', 34),   # MT34 (n,n3He) shares MT107's residual
])
def test_composite_ejectile_no_longer_unique(mt, ejectile, sibling_mt):
    """These MTs' residuals overlap with a multi-particle-exit
    sibling; they were the physically wrong inclusions in the
    pre-fix whitelist and MUST NOT trigger the MT5 rescue anymore."""
    # Confirm the sibling exists in REACTION_DICT (i.e. the
    # overlap the docstring claims is a real overlap).
    assert sibling_mt in REACTION_DICT, (
        f'sibling MT{sibling_mt} is missing from REACTION_DICT; '
        f'test premise is invalid'
    )
    sibling_ejs = get_ejectiles('n', sibling_mt)
    assert sibling_ejs is not None
    # The sibling must have >1 ejectile so the len(ejectiles)>1
    # check filters it too -- otherwise it would ALSO count as
    # unique and there'd be no over-count.
    assert len(sibling_ejs) > 1
    assert not is_unique_path_to_residual('n', mt), (
        f'MT{mt} (n,{ejectile}) should no longer be a unique path'
    )


# ============================================================
# Pre-existing exclusions must continue to hold.
# ============================================================


@pytest.mark.parametrize('mt,reason', [
    (2, 'elastic (single n mult=1 with n projectile)'),
    (4, 'inelastic sum (n mult=1 shared with MT2 elastic)'),
    (28, '(n,np) multi-particle exit'),
    (32, '(n,nd) multi-particle exit'),
    (22, '(n,na) multi-particle exit'),
    (51, 'discrete-level scattering'),
    (91, 'continuum-channel'),
    (18, '(n,f) fission -- multiple ejectiles conceptually'),
    (201, 'particle-production sum'),
    (207, 'particle-production sum'),
])
def test_pre_existing_exclusions_unchanged(mt, reason):
    """MTs that were already NOT unique paths must still not be."""
    assert not is_unique_path_to_residual('n', mt), (
        f'MT{mt} ({reason}) was rejected pre-fix and must stay '
        f'rejected'
    )


# ============================================================
# Whitelist contains no accidental extra ejectile after the
# narrowing.
# ============================================================


def test_no_composite_ejectile_passes_for_any_mt():
    """Scan every MT in REACTION_DICT for the neutron projectile
    and confirm no MT with a composite (d/t/h/a) ejectile passes
    is_unique_path_to_residual after the narrowing."""
    for mt, (proj, _) in REACTION_DICT.items():
        # REACTION_DICT stores 'z' for projectile-agnostic strings;
        # we're asking is_unique_path with proj='n', so evaluate
        # each MT under that assumption.
        ejs = get_ejectiles('n', mt)
        if not ejs or len(ejs) != 1:
            continue
        _, e = ejs[0]
        if e in ('d', 't', 'h', 'a'):
            assert not is_unique_path_to_residual('n', mt), (
                f'MT{mt} has composite ejectile {e!r} and slipped '
                f'through the narrowed whitelist'
            )
