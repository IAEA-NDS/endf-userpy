"""Numba-path pins for the URR fluctuation-integral kernel.

:func:`_channel_factors_scalar` returns all three expectation orders of
the chi-squared width factor from one power (integer ``nu`` without
``pow``, zero widths short-circuited); each order must equal the
closed form ``(1 + 2/nu)**[order == 2] * b**(-nu/2 - order)`` with
``b = 1 + 2 t alpha / nu`` (``exp(-t alpha)`` for ``nu == 0``).

The kernel looks the energy-table interval up once per (energy, group)
and takes the per-row INT=5 log-log decision from the wrapper; the
corpus parity tests below cover INT=5 groups (Nb-93), including rows
that fall back to lin-lin, and INT=2 (Nd-143) against the numpy
backend.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.primitives import array_ns
from endf_userpy.mfsec_interpretation import mf2_interpretation_urr as urr
from endf_userpy.mfsec_interpretation.mf2_interpretation_urr_preproc import (
    urr_data_from_endf_dict,
)

from _corpus import resolve_nb93, resolve_nd143

numba = pytest.importorskip('numba')

from endf_userpy.mfsec_interpretation.mf2_interpretation_urr_numba import (  # noqa: E402
    _channel_factors_scalar,
)


@numba.njit
def _factors(alpha, nu, t):
    return _channel_factors_scalar(alpha, nu, t)


def _reference(alpha, nu, t, order):
    if nu <= 1e-38:
        return math.exp(-t * alpha)
    b = 1.0 + 2.0 * t * alpha / nu
    pref = (1.0 + 2.0 / nu) if order == 2 else 1.0
    return pref * b ** (-nu / 2.0 - order)


@pytest.mark.parametrize('nu', [0.0, 0.5, 1.0, 1.7, 2.0, 3.0, 4.0, 5.0])
@pytest.mark.parametrize('alpha', [0.0, 1e-6, 0.3, 40.0])
def test_channel_factors_match_closed_form(nu, alpha):
    for t in (0.0, 1e-3, 0.7, 25.0, 1e4):
        got = _factors(alpha, nu, t)
        for order in range(3):
            np.testing.assert_allclose(
                got[order], _reference(alpha, nu, t, order),
                rtol=4e-15, atol=0.0,
                err_msg=f'order={order} t={t}',
            )


@pytest.mark.parametrize('resolve', [resolve_nb93, resolve_nd143])
def test_urr_numba_matches_numpy_on_corpus(resolve):
    path = resolve()
    if path is None:
        pytest.skip('corpus file not present (see fetch.sh)')
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(path)
    rngs = d[2][151]['isotope'][1]['range']
    ri = next(k for k in rngs if int(rngs[k]['LRU']) == 2)
    data = urr_data_from_endf_dict(d, isotope_idx=1, range_idx=ri)
    group_int = np.asarray(data.group_int)
    if resolve is resolve_nb93:
        # INT=5 groups with both all-positive rows (log-log) and rows
        # with zeros (lin-lin fallback): both branches run.
        assert np.any(group_int == 5)
        rows = [np.asarray(getattr(data, f'table_{k}'))[group_int == 5]
                for k in ('gn0', 'gg', 'gf', 'gx', 'd')]
        assert any(np.any(np.all(r > 0.0, axis=1)) for r in rows)
        assert any(np.any(~np.all(r > 0.0, axis=1)) for r in rows)
    el, eh = float(rngs[ri]['EL']), float(rngs[ri]['EH'])
    # Below, inside and above the tables' energy span.
    e = np.concatenate([[el * 0.5], np.geomspace(el, eh, 397), [eh * 2.0]])
    ref = urr.reconstruct(data, e, array_ns.get_backend('numpy'))
    got = urr.reconstruct(data, e, array_ns.get_backend('numba'))
    for key in ('sct', 'cap', 'fis', 'rxx', 'pot', 'tot'):
        np.testing.assert_allclose(
            np.asarray(got[key]), np.asarray(ref[key]),
            rtol=1e-12, atol=1e-14 * np.max(np.abs(np.asarray(ref[key]))),
            err_msg=key,
        )
