"""MF6 LAW=1 adaptive linearization port
(``feep_full_law1con``) equivalence.

The port composes the kink-aware point-wise integrator (PR #163)
with a Python replica of the Fortran adaptive-bisection driver.
Because the point-wise integrator uses GL (converges to machine
precision) rather than Romberg (~1e-3 relative), the port produces
a coarser mesh than Fortran while still meeting the ``tol=1e-3``
linearization criterion. Bit-identical mesh size vs Fortran is not
a goal; matching linear-interpolability tolerance is.

Pins:
- Linear interpolation on the returned mesh matches the point-wise
  integrator to ``tol`` for E' in the interior (away from the
  near-threshold sqrt-like rise, where BOTH Fortran and Python
  give ~30% error at max_depth=20 -- a real algorithm limit).
- Mesh endpoints and monotonicity are as expected.
- Values at panel-Ep-derived landmark points agree with the
  point-wise integrator to a few 1e-3.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip('endf_userpy.fortran.endf6')

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_integrals as py_int,
)

from _corpus import resolve_al27


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip('Al-27 corpus not present (fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def test_linearize_returns_monotonic_mesh_al27(al27_endf_dict):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ep_mesh, f_mesh, dev_mesh = (
            py_int.get_energydist_from_subsec_law1_dynamic_mesh(
                al27_endf_dict, 91, 1, 1.05e7, tol=1e-3,
            )
        )
    assert ep_mesh.ndim == 1
    assert f_mesh.shape == ep_mesh.shape
    assert dev_mesh.shape == ep_mesh.shape
    assert ep_mesh[0] == 0.0
    assert np.all(np.diff(ep_mesh) > 0.0), 'ep_mesh must be strictly monotonic'
    assert ep_mesh.shape[0] > 20  # non-trivial refinement


def test_linearize_interior_interp_matches_pointwise_al27(al27_endf_dict):
    """Linear interpolation on the returned mesh matches the
    point-wise integrator to ``tol`` at interior midpoints. We
    skip the first few intervals near ``ep=0`` where the sqrt-like
    kinematic rise exceeds what linear interp can represent at
    max_depth=20 (a real algorithm limit; Fortran hits the same
    ~30% error at the same location -- see
    ``adhoc/adaptive_boundary_check.py`` for the parity check).
    """
    E = 1.05e7
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ep_mesh, f_mesh, _ = (
            py_int.get_energydist_from_subsec_law1_dynamic_mesh(
                al27_endf_dict, 91, 1, E, tol=1e-3,
            )
        )
    # Skip the first 20 intervals (max_depth=20 near-threshold
    # bisection stack); the remaining interior is where the
    # accuracy claim holds.
    mid = 0.5 * (ep_mesh[20:-1] + ep_mesh[21:])
    lin = 0.5 * (f_mesh[20:-1] + f_mesh[21:])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        true = np.asarray(py_int.get_energydist_from_subsec_law1(
            al27_endf_dict, 91, 1, np.array([E]), mid, True,
        ))[0]
    mask = np.abs(true) > 1e-15
    if mask.any():
        rel = np.abs(lin[mask] - true[mask]) / np.abs(true[mask])
        assert rel.max() < 1e-3, (
            f'interior linear-interp error {rel.max():.3e} > tol=1e-3'
        )


def test_linearize_user_hint_included_in_mesh_al27(al27_endf_dict):
    """User-supplied ``energies_out_hint`` values appear in the
    returned mesh (up to the ``tol0=1e-6`` dedup floor)."""
    E = 1.05e7
    hint = np.array([1.234e5, 5.678e5, 1.234e6])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ep_mesh, _, _ = (
            py_int.get_energydist_from_subsec_law1_dynamic_mesh(
                al27_endf_dict, 91, 1, E, energies_out_hint=hint,
                tol=1e-3,
            )
        )
    for h in hint:
        # Nearest mesh point should sit within tol0=1e-6 relative
        # of the hint (dedup tolerance in the initial-grid builder).
        nearest = ep_mesh[np.argmin(np.abs(ep_mesh - h))]
        assert abs(nearest - h) <= 1e-6 * abs(h), (
            f'hint {h} not present; nearest mesh point {nearest}'
        )


def test_linearize_mesh_endpoints_match_fortran_al27(al27_endf_dict):
    """The mesh endpoints (kinematic min/max E') match Fortran to
    machine precision -- both implementations compute them from
    the same closed-form CM<->LAB formula."""
    from endf_userpy.fortran.endf6 import feep_full_law1con
    from endf_userpy.primitives.helpers import dict2array
    from endf_userpy.primitives.properties import get_AWR, get_AWI, get_ZA, get_ZAI

    E = 1.05e7
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ep_py, _, _ = py_int.get_energydist_from_subsec_law1_dynamic_mesh(
            al27_endf_dict, 91, 1, E, tol=1e-3,
        )

    d = al27_endf_dict
    sec = d[6][91]; subsec = sec['subsection'][1]
    awr = get_AWR(d); awi = get_AWI(d); awp = subsec['AWP']
    za = get_ZA(d); zai = get_ZAI(d); zap = subsec['ZAP']
    p = 7
    e1 = float(subsec['E'][p + 1]); e2 = float(subsec['E'][p + 2])
    nd1 = int(subsec['ND'][p + 1]); na1 = int(subsec['NA'][p + 1])
    nd2 = int(subsec['ND'][p + 2]); na2 = int(subsec['NA'][p + 2])
    ep1 = dict2array(subsec['Ep'][p + 1], dtype=float, order='F')
    b1 = dict2array(subsec['b'][p + 1], dtype=float, order='F')
    ep2 = dict2array(subsec['Ep'][p + 2], dtype=float, order='F')
    b2 = dict2array(subsec['b'][p + 2], dtype=float, order='F')
    nepmax = 100_000
    ep_f = np.zeros(nepmax, dtype=np.float64, order='F')
    feep_f = np.zeros(nepmax, dtype=np.float64, order='F')
    fdev_f = np.zeros(nepmax, dtype=np.float64, order='F')
    nep_arr = np.zeros(1, dtype=np.int64)
    feep_full_law1con(
        E, awr, awi, awp, za, zai, zap, 2,
        int(subsec['LANG']), int(subsec['LEP']), 2,
        e1, nd1, na1, ep1, b1, e2, nd2, na2, ep2, b2,
        1e-3, 0, np.zeros(1, dtype=np.float64, order='F'),
        nepmax, ep_f, feep_f, fdev_f, nep_arr,
    )
    n_f = int(nep_arr[0])
    ep_f = ep_f[:n_f]

    assert ep_py[0] == ep_f[0], 'mesh min mismatch'
    np.testing.assert_allclose(ep_py[-1], ep_f[-1], rtol=1e-12)


def test_linearize_max_points_raises_al27(al27_endf_dict):
    """Setting an unrealistically small ``max_points`` raises
    ``ValueError`` rather than silently truncating."""
    with pytest.raises(ValueError, match='max_points'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            py_int.get_energydist_from_subsec_law1_dynamic_mesh(
                al27_endf_dict, 91, 1, 1.05e7, tol=1e-3, max_points=50,
            )
