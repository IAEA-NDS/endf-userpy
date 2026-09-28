"""Peak-RSS regression test for the MF6/LAW=1 continuum broadening
path (issue #275).

On files with dense discrete-line cascades in MF6/LAW=1 ND>0
subsections (U-233 (n,g) has ~86 discrete gamma lines per Ein
panel), a plain broadened dxs/dE call:

    get_particle_production_dxs_dE(u233, '(n,g)', 'g', ein,
                                   eouts=lin(0, 8e6, 401),
                                   broadening=10e3)

used to allocate ``(nE, nEp, n_sub, n_gl, nt)`` intermediates
where ``n_sub`` was the discrete + continuum knot count per panel
pair (~173 for U-233) and ``nEp`` was the ``adaptive_convolve``
internal mesh (~131k points on the default settings). That
inflated to **27 GB peak RSS** for one 1D dxs/dE call.

Two fixes together brought it to ~1 GB:

1. Slice out the discrete-line Ep positions when building the
   polar-angle kink set. Those are delta locations that the
   continuous folder does not integrate over; they were being
   fed to Gauss-Legendre as spurious kink boundaries, clamped
   to the endpoint by ``xp.minimum(xp.maximum(., umin), 1)``
   and contributing zero-width subpanels.

2. Chunk the E' query axis in
   ``integrate_law1_spectrum``'s per-Ein-panel loop so
   intermediates never exceed a bounded fraction of the total
   mesh at any time.

This test spawns a subprocess with ``resource.setrlimit`` capping
address space to a low ceiling and asserts the broadened call
completes. Fails hard (subprocess OOM-killed by RLIMIT_AS) if the
fixes ever regress.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


ADHOC_PATH = 'tests/data_law1_adhoc/endfb81_n_U-233.endf'


@pytest.mark.skipif(
    not os.path.exists(ADHOC_PATH),
    reason='U-233 adhoc corpus not fetched',
)
@pytest.mark.skipif(
    sys.platform != 'linux',
    reason='resource.setrlimit(RLIMIT_AS) is Linux-only',
)
def test_u233_broadened_dxs_dE_stays_under_4gb():
    """A single ``get_particle_production_dxs_dE`` call with a
    Gaussian broadening on U-233 (n,g) on the default 401-point
    E_out grid must complete under a 4 GB address-space ceiling.
    Historical baseline before the #275 fix was ~27 GB, so this
    test catches a regression to that regime with 7x margin.
    """
    script = textwrap.dedent('''
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    import warnings
    import numpy as np
    from endf_parserpy import EndfParserCpp
    from endf_userpy.quantities import get_particle_production_dxs_dE
    d = EndfParserCpp(ignore_missing_tpid=True).parsefile(
        "tests/data_law1_adhoc/endfb81_n_U-233.endf",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = get_particle_production_dxs_dE(
            d, "(n,g)", "g",
            np.array([1e6]),
            np.linspace(0.0, 8e6, 401),
            broadening=10e3,
        )
    assert r.shape == (1, 401), r.shape
    print("OK")
    ''')
    proc = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, (
        f'U-233 broadened dxs_dE exceeded the 4 GB address-space '
        f'cap (issue #275 regression?):\n'
        f'stdout: {proc.stdout}\nstderr: {proc.stderr}'
    )
    assert 'OK' in proc.stdout
