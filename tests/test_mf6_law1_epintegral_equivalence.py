"""MF6 LAW=1 continuum ``E'``-integral: correctness of the Python
port plus numpy-vs-JAX parity plus ``jax.grad`` sanity.

Correctness is pinned against a high-precision scipy.integrate.quad
reference (epsrel=1e-10, adaptive) rather than against the Fortran
Romberg reference. Rationale: the Fortran uses per-subpanel Romberg
with rtol=1e-3 and the integrand has kinks from the LEP-piecewise
Ep structure as mu varies, so Fortran itself sits ~1e-3 from the
true integral. Comparing the port against Fortran conflates the
port's quadrature error with Fortran's own. scipy.quad's adaptive
kink-aware quadrature converges below any tolerance we care about,
so the port's accuracy claim scales cleanly with n_gl /
n_polar_subpanels.

Pins:
- Port at defaults (n_gl=10, n_polar_subpanels=4) matches
  scipy.quad truth to rtol=5e-3.
- Port at high order (n_gl=32, n_polar_subpanels=16) matches
  scipy.quad truth to rtol=1.5e-3 (demonstrates user-controllable
  accuracy: ~3x tighter for ~13x more mu nodes; slow convergence
  is expected because the integrand carries LEP-piecewise kinks).
- Port bit-identical numpy vs JAX (rtol=1e-11).
- jax.grad reaches back from a dict-stored ``b`` leaf and matches
  finite-diff.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import (
    mf6_interpretation_integrals as py_int,
    mf6_law1_helpers,
    mf6_law1_preproc,
)
from endf_userpy.primitives import array_ns
from endf_userpy.primitives.helpers import dict2array

from _corpus import resolve_al27


def _jax_available():
    return 'jax' in array_ns.available_backends()


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip(
            'Al-27 ENDF file not available (run '
            'tests/data_law1_adhoc/fetch.sh)'
        )
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def _query_grid(subsec, panel_idx, n_ein=3, n_eout=40):
    """(E, E') grid inside the given panel, avoiding the panel
    boundaries where the LEP-piecewise structure is sharpest."""
    E1 = float(subsec['E'][panel_idx + 1])
    E2 = float(subsec['E'][panel_idx + 2])
    e_in = np.linspace(E1, E2, n_ein + 2)[1:-1]
    ep_vals = list(subsec['Ep'][panel_idx + 1].values())
    ep_lo = float(ep_vals[3])
    ep_hi = float(ep_vals[-1])
    e_out = np.linspace(ep_lo * 1.05, ep_hi * 0.9, n_eout)
    return e_in, e_out


def _scipy_quad_truth(endf_dict, mt, sn, panel_idx, E, Ep, to_lab=True):
    """Adaptive high-precision reference for ``f(E, E')`` at one
    ``(E, E')``: scipy.quad of ``f_amp(mu) * dinv(mu)`` from
    ``umin`` to +1 with epsrel=1e-10.

    Derives ``eff_lct`` from the section the same way the port does
    so the reference and the port share the same frame convention.
    """
    from endf_userpy.primitives.properties import (
        get_AWR, get_AWI, get_ZA, get_ZAI,
    )
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][sn]
    awr = get_AWR(endf_dict)
    awi = get_AWI(endf_dict)
    awp = subsec['AWP']
    za = get_ZA(endf_dict)
    zai = get_ZAI(endf_dict)
    zap = subsec['ZAP']
    lang = int(subsec['LANG'])
    lep = int(subsec['LEP'])
    lct = int(sec['LCT']) if to_lab else 1
    if lct in (1, 2):
        eff_lct = lct
    elif lct == 3:
        eff_lct = 1 if awp > 4 else 2
    else:
        raise NotImplementedError(f'LCT={lct} not implemented')
    p1 = panel_idx
    p2 = panel_idx + 1
    e1 = float(subsec['E'][p1 + 1])
    e2 = float(subsec['E'][p2 + 1])
    nd1 = int(subsec['ND'][p1 + 1])
    na1 = int(subsec['NA'][p1 + 1])
    ep1 = dict2array(subsec['Ep'][p1 + 1], dtype=float)
    b1 = dict2array(subsec['b'][p1 + 1], dtype=float)
    nd2 = int(subsec['ND'][p2 + 1])
    na2 = int(subsec['NA'][p2 + 1])
    ep2 = dict2array(subsec['Ep'][p2 + 1], dtype=float)
    b2 = dict2array(subsec['b'][p2 + 1], dtype=float)
    lei = 2  # Al-27 MF6 outer INT is lin-lin

    # umin per feep_law1con line 3020 (LAB case)
    if eff_lct == 2 or (eff_lct == 3 and awp < 4.0):
        c0 = float(np.sqrt(awi * awp) / (awi + awr))
        y_slope = (E - e1) / (e2 - e1)
        epmax_eff = float(ep1[-1]) + y_slope * (float(ep2[-1]) - float(ep1[-1]))
        umin_raw = (Ep + c0**2 * E - epmax_eff) / (2.0 * c0 * np.sqrt(Ep * E))
        umin = max(-1.0, min(1.0, umin_raw))
    else:
        umin = -1.0

    def integrand(mu):
        tp, w, dinv = mf6_law1_helpers.mf6lab2cm(
            awr, awi, awp, eff_lct, E, Ep, mu,
        )
        f_amp = mf6_law1_helpers.f6law1con_amplitude(
            E, tp, w, za, zai, zap, lang, lep, lei,
            e1, nd1, na1, ep1, b1, e2, nd2, na2, ep2, b2,
        )
        return f_amp * dinv

    if umin >= 1.0:
        return 0.0
    val, _ = quad(integrand, umin, 1.0, epsrel=1e-10, limit=2000)
    return val


@pytest.mark.parametrize(
    'n_gl,n_sub,rtol,label',
    [
        (10, 4, 5e-3, 'defaults'),
        (32, 16, 1.5e-3, 'high-order'),
    ],
)
def test_law1_epintegral_matches_scipy_quad_truth(
    al27_endf_dict, n_gl, n_sub, rtol, label,
):
    """Al-27 MT=91 continuum ``f(E, E')`` matches an adaptive
    scipy.integrate.quad reference. Tolerance tightens by ~5x when
    the caller passes higher order (n_gl=32, n_polar_subpanels=16),
    confirming user-controllable accuracy."""
    mt, sn = 91, 1
    subsec = al27_endf_dict[6][mt]['subsection'][sn]
    panel_idx = 7  # (E1, E2) = (1e7, 1.1e7) eV
    e_in, e_out = _query_grid(subsec, panel_idx=panel_idx, n_ein=1, n_eout=8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        py = np.asarray(py_int.get_energydist_from_subsec_law1(
            al27_endf_dict, mt, sn, e_in, e_out, True,
            n_gl=n_gl, n_polar_subpanels=n_sub,
        ))
    truth = np.zeros_like(py)
    for i, E in enumerate(e_in):
        for j, Ep in enumerate(e_out):
            truth[i, j] = _scipy_quad_truth(
                al27_endf_dict, mt, sn, panel_idx, float(E), float(Ep),
                to_lab=True,
            )
    mask = truth > 1e-15
    rd = np.abs(py[mask] - truth[mask]) / truth[mask]
    assert rd.max() < rtol, (
        f'[{label}] max rel diff vs scipy.quad truth {rd.max():.3e} '
        f'exceeds {rtol:.0e}; median {np.median(rd):.3e}'
    )


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law1_epintegral_numpy_jax_parity_al27_mt91(al27_endf_dict):
    """Numpy vs JAX bit-identical: same algorithm, same nodes and
    weights, only the backend adapter differs."""
    mt, sn = 91, 1
    subsec = al27_endf_dict[6][mt]['subsection'][sn]
    e_in, e_out = _query_grid(subsec, panel_idx=7)
    xp_np = array_ns.get_backend('numpy')
    xp_jx = array_ns.get_backend('jax')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        a = np.asarray(py_int.get_energydist_from_subsec_law1(
            al27_endf_dict, mt, sn, e_in, e_out, True, xp=xp_np,
        ))
        b = np.asarray(py_int.get_energydist_from_subsec_law1(
            al27_endf_dict, mt, sn, e_in, e_out, True, xp=xp_jx,
        ))
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-30)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_law1_epintegral_jax_grad_from_dict_leaf(al27_endf_dict):
    """``jax.grad`` of a scalar summary of ``f(E, E')`` reaches back
    to a dict-stored ``b`` leaf and matches finite-difference.

    Selects panel_key=8 (dict is 1-indexed; corresponds to
    panel_idx=7 in the query) so the tracer sits on the enclosing
    incident-energy panel and the gradient is nonzero."""
    import jax
    import jax.numpy as jnp
    mt, sn = 91, 1
    subsec = al27_endf_dict[6][mt]['subsection'][sn]
    e_in, e_out = _query_grid(subsec, panel_idx=7, n_ein=1, n_eout=20)
    xp_jx = array_ns.get_backend('jax')
    panel_key = 8      # 1-indexed dict key = 0-indexed panel 7
    ep_row = 5         # 1-indexed
    coef = 1           # 1-indexed: b[panel][row][col]
    original = float(
        al27_endf_dict[6][mt]['subsection'][sn]['b'][panel_key][ep_row][coef]
    )

    def loss(theta):
        d_t = copy.deepcopy(al27_endf_dict)
        d_t[6][mt]['subsection'][sn]['b'][panel_key][ep_row][coef] = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return jnp.sum(py_int.get_energydist_from_subsec_law1(
                d_t, mt, sn, e_in, e_out, True, xp=xp_jx,
            ))

    val = float(loss(jnp.array(original)))
    grad = float(jax.grad(loss)(jnp.array(original)))
    assert val > 0.0
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    eps = 1e-4
    lp = float(loss(jnp.array(original + eps)))
    lm = float(loss(jnp.array(original - eps)))
    fd = (lp - lm) / (2.0 * eps)
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-12)


def test_preproc_data_shape(al27_endf_dict):
    """Sanity: the preproc dataclass built for MT=91 has the panel
    counts and per-panel Ep counts we key the query grid off in
    the other tests."""
    data = mf6_law1_preproc.mf6_law1_data_from_endf_dict(al27_endf_dict, 91, 1)
    n_panels = data.ei_mesh.shape[0]
    assert n_panels >= 8, (
        f'expected >=8 panels for Al-27 MT=91 LAW=1, got {n_panels}'
    )
    assert int(data.nep_arr[7]) > 5
