"""Numba-path pins for the URR fluctuation-integral kernel.

:func:`_channel_factors_scalar` returns all three expectation orders of
the chi-squared width factor from one power (integer ``nu`` without
``pow``, zero widths short-circuited); each order must equal the
closed form ``(1 + 2/nu)**[order == 2] * b**(-nu/2 - order)`` with
``b = 1 + 2 t alpha / nu`` (``exp(-t alpha)`` for ``nu == 0``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

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
