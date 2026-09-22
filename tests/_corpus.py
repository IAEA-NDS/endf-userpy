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
