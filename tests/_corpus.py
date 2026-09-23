"""Locate on-disk ENDF files used by the resonance-reconstruction
integration tests, portable across a fresh checkout and the
maintainer's personal layout.

Lookup order per name:

1. Environment variable (e.g. ``NB93_ENDF``).
2. ``tests/data_law1_adhoc/<corpus_name>`` (populated by
   ``bash tests/data_law1_adhoc/fetch.sh``).

A returned path is guaranteed to exist. ``None`` means every
candidate was missing and the test should ``pytest.skip``.

To use a private / out-of-tree copy of a file, point the
corresponding environment variable at it.
"""
from __future__ import annotations

import os


_HERE = os.path.dirname(__file__)
_CORPUS_DIR = os.path.join(_HERE, 'data_law1_adhoc')


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def resolve_nb93() -> str | None:
    """Resolve the Nb-93 ENDF file. ENDF/B-VIII.1 preferred."""
    return _first_existing(
        os.environ.get('NB93_ENDF'),
        os.path.join(_CORPUS_DIR, 'endfb81_n_Nb-93.endf'),
    )


def resolve_u235() -> str | None:
    """Resolve the U-235 ENDF file (TENDL-2021, used for Reich-Moore)."""
    return _first_existing(
        os.environ.get('U235_ENDF'),
        os.path.join(_CORPUS_DIR, 'tendl21_n_U-235.endf'),
    )


def resolve_nd143() -> str | None:
    """Resolve the Nd-143 ENDF file (ENDF/B-VIII.1, used for the
    RRR (MLBW) + LSSF=0 URR + MF3-above-URR seam tests in
    issue #149)."""
    return _first_existing(
        os.environ.get('ND143_ENDF'),
        os.path.join(_CORPUS_DIR, 'endfb81_n_Nd-143.endf'),
    )


def resolve_al27() -> str | None:
    """Resolve the Al-27 ENDF file (ENDF/B-VIII.1). Used for the
    MF6 LAW=2 discrete-two-body port (issue #47): Al-27 has a
    long ladder of discrete-level inelastic-scattering MTs
    (MT=51..80) each written as MF6 LAW=2 LANG=0 (Legendre) with
    a neutron ejectile (ZAP=1), which is the canonical
    non-gamma-ZAP LAW=2 case."""
    return _first_existing(
        os.environ.get('AL27_ENDF'),
        os.path.join(_CORPUS_DIR, 'endfb81_n_Al-27.endf'),
    )


def resolve_h2() -> str | None:
    """Resolve the H-2 ENDF file (JEFF-4.0). Used by the LAW=7
    kink-aware mu-integration parity tests: JEFF-4.0 H-2 MT16
    (n, 2n) is the corpus's canonical single-subsection LAW=7
    case where the general adaptive-Simpson mu-integrator undershoots
    the effective knot structure."""
    return _first_existing(
        os.environ.get('H2_ENDF'),
        os.path.join(_CORPUS_DIR, 'jeff40_n_H-2.endf'),
    )


def resolve_c12() -> str | None:
    """Resolve the C-12 ENDF file (JENDL-5). Used by the MF14
    Legendre-branch parity tests (issue #169 tier-2 photon path):
    JENDL-5 C-12 MT51 is one of the rare corpus MTs that carries
    MF14 LI=0 LTT=1 per-line Legendre coefficients rather than
    the fully-isotropic LI=1 fallback."""
    return _first_existing(
        os.environ.get('C12_ENDF'),
        os.path.join(_CORPUS_DIR, 'jendl5_n_C-12.endf'),
    )
