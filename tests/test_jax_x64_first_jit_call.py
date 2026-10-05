"""Regression test for the jax x64-config race.

Before the fix, ``JaxBackend.__init__`` called ``jax.config.update(
'jax_enable_x64', True)``. When ``RunOptions(backend='jax')`` was
constructed INSIDE an active ``@jax.jit`` trace (the natural
pattern: build options in the function body), the config update
came too late for that trace -- the tracers were already float32
and the resonance 1/v tail underflowed at low Ein. A SECOND jit
call to the same function then saw x64 globally enabled, compiled
afresh and matched numpy, so the bug only showed on the first
call of a fresh python session.

Fix: move ``jax.config.update('jax_enable_x64', True)`` to
module-import time (``_enable_jax_x64()`` in
``endf_userpy/primitives/array_ns.py``). This test must be run
as a standalone invocation (fresh process) so no earlier jit
has activated x64 by side effect; the harness enforces this via
``pytest-forked`` when available, or spawns a subprocess
otherwise.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


SCRIPT = textwrap.dedent('''
    import numpy as np, jax, jax.numpy as jnp
    from endf_parserpy import EndfParserCpp
    from endf_userpy.quantities_mt_zap.quantities import compute_xs
    from endf_userpy.run_options import RunOptions

    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(
        "tests/data_law1_adhoc/tendl21_n_U-235.endf",
    )
    ein = np.geomspace(1e-5, 2e7, 10)
    r_np = compute_xs(d, 102, ein, options=RunOptions(backend="numpy"))

    @jax.jit
    def fn(e):
        return compute_xs(d, 102, e, options=RunOptions(backend="jax"))

    r_jit = np.asarray(fn(jnp.asarray(ein)))
    peak = float(np.abs(r_np).max())
    diff = float(np.abs(r_jit - r_np).max())
    rel = diff / max(peak, 1e-30)
    print(f"REL={rel:.3e} JIT0={r_jit[0]:.6e} NP0={r_np[0]:.6e}")
''')


U235_PATH = 'tests/data_law1_adhoc/tendl21_n_U-235.endf'


@pytest.mark.skipif(
    not os.path.exists(U235_PATH),
    reason='tendl21_n_U-235.endf not fetched',
)
@pytest.mark.skipif(
    sys.platform not in ('linux', 'darwin'),
    reason='subprocess spawn behaviour differs on Windows',
)
def test_first_jit_call_matches_numpy_in_resonance_region():
    """A fresh python session's first jit call on compute_xs
    with Ein covering the resonance region must bit-equal numpy
    (modulo the usual XLA FP noise ~1e-10).

    Pre-fix: first-call rel-diff was 1.0 (jit returned 0 in the
    resonance region; numpy returned the 1/v tail xs). Post-fix:
    rel-diff ~1e-10.
    """
    result = subprocess.run(
        [sys.executable, '-c', SCRIPT],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f'subprocess failed: {result.returncode}\n'
        f'stdout: {result.stdout}\nstderr: {result.stderr[-2000:]}'
    )
    # Parse "REL=... JIT0=... NP0=..." from stdout.
    rel_line = [
        line for line in result.stdout.splitlines()
        if line.startswith('REL=')
    ]
    assert rel_line, (
        f'did not find REL line in output: {result.stdout!r}'
    )
    kvs = dict(kv.split('=') for kv in rel_line[-1].split())
    rel = float(kvs['REL'])
    jit0 = float(kvs['JIT0'])
    np0 = float(kvs['NP0'])
    assert rel < 1e-6, (
        f'first-call jit result diverges from numpy by rel={rel:.3e} '
        f'(pre-#x64-fix regression likely re-introduced). '
        f'jit[0]={jit0:.3e}, numpy[0]={np0:.3e}'
    )
