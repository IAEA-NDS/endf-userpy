"""Numba fast path of :func:`interpolation.endf_interp1d`.

On the numba backend ``endf_interp1d`` hands the deduplicated mesh and
a per-panel law table to
:func:`interpolation_numba._endf_interp1d_kernel` instead of the
per-region numpy loop. Pins:

- the numba backend actually takes the kernel (and plain numpy does
  not);
- on randomised tables (doubled points, doubled zeros, zero ordinates,
  1-3 regions, every INT code) and queries (mesh nodes, interior
  points, NaN, out-of-range with and without ``outside_value``) the
  result equals the numpy path: identical NaN / inf pattern and
  exceptions, bit-identical for INT 1 / 2, within 1e-13 for the
  log-based laws (numpy's and libm's ``log`` / ``exp`` differ by ULPs);
- a NaN query uses the last panel, as numpy's NaN-sorts-last
  ``searchsorted`` does (an INT=1 last region returns its ``y1``).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from endf_userpy.primitives import array_ns
from endf_userpy.primitives.interpolation import endf_interp1d

pytest.importorskip('numba')

from endf_userpy.primitives import interpolation_numba  # noqa: E402

NB = array_ns.get_backend('numba')
NP = array_ns.get_backend('numpy')


def _random_table(rng, max_law):
    n = int(rng.integers(2, 30))
    scale = 0.0 if rng.random() < 0.1 else 1.0
    mesh = np.sort(scale * rng.uniform(0.0, 100.0, n))
    if n > 3 and rng.random() < 0.5:
        k = int(rng.integers(1, n - 1))
        mesh[k] = mesh[k - 1]                    # doubled point
    if rng.random() < 0.1:
        mesh[:2] = 0.0                           # doubled zero
    fp = rng.choice([0.0, 1.0], size=n, p=[0.1, 0.9]) * rng.uniform(0.1, 10.0, n)
    nreg = int(rng.integers(1, min(4, n)))
    inner = (np.sort(rng.choice(np.arange(2, n), nreg - 1, replace=False))
             if nreg > 1 else np.array([], dtype=int))
    nbt = np.append(inner, n).astype(int)
    int_arr = rng.integers(1, max_law + 1, nbt.size)
    return mesh, fp, int_arr, nbt


def _both(x, mesh, fp, int_arr, nbt, outside_value):
    out = []
    for xp in (NP, NB):
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            try:
                out.append((np.asarray(endf_interp1d(
                    x, mesh, fp, int_arr, nbt,
                    outside_value=outside_value, xp=xp)), None))
            except Exception as exc:     # compare the exception type
                out.append((None, type(exc)))
    return out


@pytest.mark.parametrize('max_law,rtol', [(2, 0.0), (5, 1e-13)])
def test_numba_path_matches_numpy_path(max_law, rtol):
    rng = np.random.default_rng(max_law)
    for _ in range(300):
        mesh, fp, int_arr, nbt = _random_table(rng, max_law)
        q = np.concatenate([mesh, rng.uniform(mesh[0], mesh[-1], 40), [np.nan]])
        for outside_value in (None, 0.0, -1.0):
            x = q if outside_value is None else np.concatenate(
                [q, [mesh[0] - 1.0, mesh[-1] + 1.0]])
            (a, ea), (b, eb) = _both(x, mesh, fp, int_arr, nbt, outside_value)
            assert ea == eb
            if ea is not None:
                continue
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
            np.testing.assert_array_equal(np.isinf(a), np.isinf(b))
            fin = np.isfinite(a)
            if rtol == 0.0:
                np.testing.assert_array_equal(a[fin], b[fin])
            else:
                np.testing.assert_allclose(b[fin], a[fin], rtol=rtol, atol=0.0)


def test_nan_query_uses_last_panel():
    mesh = np.array([1.0, 2.0, 3.0, 4.0])
    fp = np.array([1.0, 5.0, 7.0, 9.0])
    int_arr = np.array([2, 1])
    nbt = np.array([2, 4])
    for xp in (NP, NB):
        y = np.asarray(endf_interp1d(np.array([np.nan, 1.5]), mesh, fp,
                                     int_arr, nbt, xp=xp))
        assert y[0] == 7.0           # INT=1 on the last panel [3, 4]
        assert y[1] == 3.0


def test_numba_backend_takes_kernel(monkeypatch):
    calls = []
    kernel = interpolation_numba._endf_interp1d_kernel

    def spy(*args):
        calls.append(1)
        return kernel(*args)

    monkeypatch.setattr(interpolation_numba, '_endf_interp1d_kernel', spy)
    mesh = np.array([1.0, 2.0, 4.0])
    fp = np.array([0.0, 1.0, 3.0])
    x = np.linspace(1.0, 4.0, 7)
    y_np = endf_interp1d(x, mesh, fp, np.array([2]), np.array([3]), xp=NP)
    assert not calls
    y_nb = endf_interp1d(x, mesh, fp, np.array([2]), np.array([3]), xp=NB)
    assert len(calls) == 1
    np.testing.assert_array_equal(np.asarray(y_nb), np.asarray(y_np))


def test_panel_laws_rejects_incomplete_or_unknown_regions():
    assert interpolation_numba.panel_laws([2], [3], 5) is None    # panels 3, 4 uncovered
    assert interpolation_numba.panel_laws([7], [5], 5) is None    # INT=7
    np.testing.assert_array_equal(
        interpolation_numba.panel_laws([2, 5], [3, 5], 5), [2, 2, 5, 5])
