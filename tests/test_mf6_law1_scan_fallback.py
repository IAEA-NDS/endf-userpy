"""Scan-safe panel-loop fallback in ``mf6_law1_kernel.reconstruct``
(issue #328).

The Python-loop fallback at the bottom of ``reconstruct`` unrolls
one ``_f6law1con_panel_pair_bc`` call per panel pair into the
jaxpr under ``@jax.jit``. On U-235 (n,g) MT=102 (41 panels) at a
modest DDX grid, XLA preparing the computation asked for 60 GB of
buffer and OOMed. ``_reconstruct_fallback_scanned`` collapses the
panel loop to one body in the jaxpr via ``xp.scan``.

Pins:
  * numerical equivalence to the numpy baseline on Al-27 (lang=2)
    and U-235 (lang=1, lct=1) and Nd-143 (lang=1, lct=2) corpus
    subsections,
  * same equivalence under ``@jax.jit``,
  * ``_pack_panels_for_scan`` returns ``n_panels - 1`` pairs with
    correctly aligned ``is_last`` flag,
  * fallback for unsupported shapes (lang=11) raises
    ``NotImplementedError`` so the kernel can route to the legacy
    Python-loop path.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation import mf6_law1_preproc as _pre
from endf_userpy.mfsec_interpretation.mf6_law1_kernel import (
    _pack_panels_for_scan,
    _reconstruct_fallback_scanned,
    reconstruct,
)
from endf_userpy.primitives import array_ns

from _corpus import resolve_al27, resolve_u235, resolve_nd143


def _jax_available():
    return 'jax' in array_ns.available_backends()


def _rel_max(f_new, f_old):
    diff = float(np.max(np.abs(np.asarray(f_new) - np.asarray(f_old))))
    peak = max(float(np.max(np.abs(np.asarray(f_old)))), 1e-30)
    return diff / peak


@pytest.fixture(scope='module')
def al27_endf_dict():
    path = resolve_al27()
    if path is None:
        pytest.skip('Al-27 corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def u235_endf_dict():
    path = resolve_u235()
    if path is None:
        pytest.skip('U-235 corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.fixture(scope='module')
def nd143_endf_dict():
    path = resolve_nd143()
    if path is None:
        pytest.skip('Nd-143 corpus not available')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


def _reconstruct_via_scan_path(data, e_in, e_out, mu, xp):
    """Drive ``_reconstruct_fallback_scanned`` with the same frame
    handling as ``reconstruct`` so the two outputs are comparable."""
    from endf_userpy.mfsec_interpretation.mf6_law1_kernel import _mf6lab2cm_bc

    n_e, n_ep, n_mu = e_in.shape[0], e_out.shape[0], mu.shape[0]
    eff_lct = int(data.lct)
    e_bc = xp.asarray(e_in)[:, None, None]
    ep_bc = xp.asarray(e_out)[None, :, None]
    mu_bc = xp.asarray(mu)[None, None, :]
    if eff_lct == 1 or (eff_lct == 3 and float(data.awp) >= 4.0):
        tp_bc = xp.broadcast_to(ep_bc, (n_e, n_ep, n_mu))
        w_bc = xp.broadcast_to(mu_bc, (n_e, n_ep, n_mu))
        dinv_bc = xp.ones((n_e, n_ep, n_mu), dtype=xp.float64)
    else:
        e_full = xp.broadcast_to(e_bc, (n_e, n_ep, n_mu))
        ep_full = xp.broadcast_to(ep_bc, (n_e, n_ep, n_mu))
        mu_full = xp.broadcast_to(mu_bc, (n_e, n_ep, n_mu))
        tp_bc, w_bc, dinv_bc = _mf6lab2cm_bc(
            data.awr, data.awi, data.awp, eff_lct,
            e_full, ep_full, mu_full, xp,
        )
    e_bc_full = xp.broadcast_to(e_bc, (n_e, n_ep, n_mu))
    return _reconstruct_fallback_scanned(
        data, e_in, e_bc_full, tp_bc, w_bc, dinv_bc,
        n_e, n_ep, n_mu, xp,
    )


def test_pack_panels_shapes_and_is_last(al27_endf_dict):
    data = _pre.mf6_law1_data_from_endf_dict(al27_endf_dict, mt=91,
                                              subsec_num=1)
    xp = array_ns.get_backend('numpy')
    packed = _pack_panels_for_scan(data, xp=xp)
    n_pairs = int(data.ei_mesh.shape[0]) - 1
    assert packed['e1'].shape == (n_pairs,)
    assert packed['e2'].shape == (n_pairs,)
    assert packed['ep1'].shape[0] == n_pairs
    assert packed['b1'].shape[0] == n_pairs
    assert bool(packed['is_last'][-1]) is True
    assert all(bool(x) is False for x in packed['is_last'][:-1])
    # The (i, i+1) alignment the scan body relies on.
    assert np.array_equal(np.asarray(packed['e1'])[1:],
                          np.asarray(packed['e2'])[:-1])


@pytest.mark.parametrize('mt,sub', [(16, 1), (22, 1), (91, 1)])
def test_scan_fallback_matches_baseline_lang2_al27(
    al27_endf_dict, mt, sub,
):
    data = _pre.mf6_law1_data_from_endf_dict(al27_endf_dict, mt=mt,
                                              subsec_num=sub)
    if int(data.lang) != 2:
        pytest.skip(f'MT={mt} sub={sub} is not lang=2')
    xp = array_ns.get_backend('numpy')
    e_min = float(data.ei_mesh[0])
    e_max = float(data.ei_mesh[-1])
    e_in = np.linspace(e_min * 1.01, e_max * 0.99, 7)
    e_out = np.linspace(1e4, 1e7, 11)
    mu = np.linspace(-0.9, 0.9, 5)
    f_old = np.asarray(reconstruct(data, e_in, e_out, mu,
                                    to_lab=True, xp=xp))
    f_new = np.asarray(_reconstruct_via_scan_path(data, e_in, e_out,
                                                   mu, xp))
    assert _rel_max(f_new, f_old) < 1e-10


@pytest.mark.parametrize('mt,sub', [(5, 6), (16, 2), (22, 3), (91, 2)])
def test_scan_fallback_matches_baseline_lang1_al27(
    al27_endf_dict, mt, sub,
):
    data = _pre.mf6_law1_data_from_endf_dict(al27_endf_dict, mt=mt,
                                              subsec_num=sub)
    if int(data.lang) != 1:
        pytest.skip(f'MT={mt} sub={sub} is not lang=1')
    xp = array_ns.get_backend('numpy')
    e_min = float(data.ei_mesh[0])
    e_max = float(data.ei_mesh[-1])
    e_in = np.linspace(e_min * 1.01, e_max * 0.99, 7)
    e_out = np.linspace(1e4, 1e7, 11)
    mu = np.linspace(-0.9, 0.9, 5)
    f_old = np.asarray(reconstruct(data, e_in, e_out, mu,
                                    to_lab=True, xp=xp))
    f_new = np.asarray(_reconstruct_via_scan_path(data, e_in, e_out,
                                                   mu, xp))
    assert _rel_max(f_new, f_old) < 1e-10


def test_scan_fallback_matches_baseline_u235_ng_lang1(u235_endf_dict):
    data = _pre.mf6_law1_data_from_endf_dict(u235_endf_dict, mt=102,
                                              subsec_num=1)
    xp = array_ns.get_backend('numpy')
    e_in = np.geomspace(1e-5, 2e7, 10)
    e_out = np.linspace(0, 2e7, 20)
    mu = np.linspace(-0.9, 0.9, 5)
    f_old = np.asarray(reconstruct(data, e_in, e_out, mu,
                                    to_lab=True, xp=xp))
    f_new = np.asarray(_reconstruct_via_scan_path(data, e_in, e_out,
                                                   mu, xp))
    assert _rel_max(f_new, f_old) < 1e-10


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_reconstruct_under_jit_matches_numpy_u235(u235_endf_dict):
    """The issue-#328 fingerprint test: @jax.jit over the full
    reconstruct must produce bit-identical output to the numpy
    baseline and finish without OOM on the 41-panel actinide file
    at the DDX grid shape that originally triggered the 60 GB
    allocation.
    """
    import jax
    import jax.numpy as jnp

    data = _pre.mf6_law1_data_from_endf_dict(u235_endf_dict, mt=102,
                                              subsec_num=1)
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')

    e_in_np = np.geomspace(1e-5, 2e7, 20)
    e_out_np = np.linspace(0, 2e7, 50)
    mu_np = np.linspace(-0.9, 0.9, 10)

    f_old = np.asarray(reconstruct(data, e_in_np, e_out_np, mu_np,
                                    to_lab=True, xp=xp_np))

    e_in_j = jnp.asarray(e_in_np)
    e_out_j = jnp.asarray(e_out_np)
    mu_j = jnp.asarray(mu_np)

    @jax.jit
    def fn(e):
        return reconstruct(data, e, e_out_j, mu_j,
                           to_lab=True, xp=xp_jax)

    f_jit = np.asarray(fn(e_in_j))
    assert _rel_max(f_jit, f_old) < 1e-10


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_reconstruct_under_jit_matches_numpy_al27_lang2(al27_endf_dict):
    import jax
    import jax.numpy as jnp

    data = _pre.mf6_law1_data_from_endf_dict(al27_endf_dict, mt=91,
                                              subsec_num=1)
    xp_np = array_ns.get_backend('numpy')
    xp_jax = array_ns.get_backend('jax')
    e_in_np = np.linspace(1e5, 2e7, 7)
    e_out_np = np.linspace(1e4, 1e7, 11)
    mu_np = np.linspace(-0.9, 0.9, 5)

    f_old = np.asarray(reconstruct(data, e_in_np, e_out_np, mu_np,
                                    to_lab=True, xp=xp_np))

    @jax.jit
    def fn(e):
        return reconstruct(data, e, jnp.asarray(e_out_np),
                           jnp.asarray(mu_np),
                           to_lab=True, xp=xp_jax)

    f_jit = np.asarray(fn(jnp.asarray(e_in_np)))
    assert _rel_max(f_jit, f_old) < 1e-10


def test_scan_fallback_nd143_lang1_lct2(nd143_endf_dict):
    """Nd-143 is lang=1 lct=2 (CM frame); exercises the LCT=2
    LAB-to-CM transform path in the scan driver."""
    xp = array_ns.get_backend('numpy')
    found_sub = None
    for mt in sorted(nd143_endf_dict.get(6, {}).keys()):
        for sub in range(1, 20):
            try:
                data = _pre.mf6_law1_data_from_endf_dict(
                    nd143_endf_dict, mt=mt, subsec_num=sub,
                )
            except Exception:
                break
            if int(data.lang) == 1 and int(data.lct) == 2:
                found_sub = (mt, sub, data)
                break
        if found_sub is not None:
            break
    if found_sub is None:
        pytest.skip('No lang=1 lct=2 subsection found in Nd-143')
    mt, sub, data = found_sub
    e_min = float(data.ei_mesh[0])
    e_max = float(data.ei_mesh[-1])
    e_in = np.linspace(e_min * 1.01, e_max * 0.99, 5)
    e_out = np.linspace(1e4, 1e7, 11)
    mu = np.linspace(-0.9, 0.9, 5)
    f_old = np.asarray(reconstruct(data, e_in, e_out, mu,
                                    to_lab=True, xp=xp))
    f_new = np.asarray(_reconstruct_via_scan_path(data, e_in, e_out,
                                                   mu, xp))
    assert _rel_max(f_new, f_old) < 1e-9


def test_scan_fallback_rejects_unsupported_lang(al27_endf_dict):
    """lang in (11..15) is not covered by the scan path; the
    fallback must raise ``NotImplementedError`` so the caller can
    route to the legacy Python-loop path."""
    data = _pre.mf6_law1_data_from_endf_dict(al27_endf_dict, mt=91,
                                              subsec_num=1)
    import dataclasses
    data_patched = dataclasses.replace(data, lang=11)
    xp = array_ns.get_backend('numpy')
    e_in = np.linspace(float(data.ei_mesh[0]) * 1.01,
                       float(data.ei_mesh[-1]) * 0.99, 3)
    e_out = np.linspace(1e4, 1e7, 5)
    mu = np.linspace(-0.9, 0.9, 3)
    with pytest.raises(NotImplementedError, match='lang'):
        _reconstruct_via_scan_path(data_patched, e_in, e_out, mu, xp)


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_reconstruct_warns_on_scan_fallthrough_unsupported_lang(
    al27_endf_dict,
):
    """When the scan-safe path cannot handle the subsection (here:
    lang=11 forced on an Al-27 LAW=1 case) and the kernel is in
    the fallback branch (tracer mesh / query under @jax.jit),
    ``reconstruct`` must emit one UserWarning that names the
    reason AND the OOM-risk implication under @jax.jit, so the
    user is not surprised by a silent slow / OOM fallback
    (follow-up to issue #328).
    """
    import dataclasses
    import warnings as _warnings

    import jax
    import jax.numpy as jnp

    data = _pre.mf6_law1_data_from_endf_dict(al27_endf_dict, mt=91,
                                              subsec_num=1)
    data_patched = dataclasses.replace(data, lang=11)
    xp = array_ns.get_backend('jax')
    e_out_j = jnp.linspace(1e4, 1e7, 5)
    mu_j = jnp.linspace(-0.9, 0.9, 3)

    @jax.jit
    def fn(e):
        return reconstruct(data_patched, e, e_out_j, mu_j,
                           to_lab=True, xp=xp)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        try:
            fn(jnp.linspace(float(data.ei_mesh[0]) * 1.01,
                            float(data.ei_mesh[-1]) * 0.99, 3))
        except Exception:
            pass
    scan_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and 'scan-safe' in str(w.message)
    ]
    assert len(scan_warnings) >= 1, (
        f'expected a scan-fallthrough UserWarning, got '
        f'{[str(w.message) for w in caught]}'
    )
    msg = str(scan_warnings[0].message)
    assert 'lang' in msg
    assert 'OOM' in msg or 'issue #328' in msg
    assert 'eager' in msg


@pytest.mark.skipif(not _jax_available(), reason='jax not installed')
def test_reconstruct_does_not_warn_on_supported_subsection(
    al27_endf_dict,
):
    """The common lang=1 / lang=2 uniform-ei case is handled by
    the scan path and must NOT emit the fallthrough UserWarning,
    including in the jit-fallback branch (tracer e_in)."""
    import warnings as _warnings
    import jax
    import jax.numpy as jnp

    data = _pre.mf6_law1_data_from_endf_dict(al27_endf_dict, mt=91,
                                              subsec_num=1)
    xp = array_ns.get_backend('jax')
    e_out_j = jnp.linspace(1e4, 1e7, 5)
    mu_j = jnp.linspace(-0.9, 0.9, 3)

    @jax.jit
    def fn(e):
        return reconstruct(data, e, e_out_j, mu_j,
                           to_lab=True, xp=xp)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        fn(jnp.linspace(float(data.ei_mesh[0]) * 1.01,
                        float(data.ei_mesh[-1]) * 0.99, 3))
    scan_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and 'scan-safe' in str(w.message)
    ]
    assert scan_warnings == []
