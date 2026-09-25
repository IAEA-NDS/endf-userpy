"""Numba-backend parity pins for MF2 LRF=7 KRM=3 (#228 PR 3).

Two flavours:

- **Numpy-numba equivalence**: on the three ad-hoc corpus LRF=7
  files, reconstruct via the numpy backend AND via the numba
  backend and assert the returned cross sections agree to double
  precision. This pins the ``@njit`` kernel arithmetic against
  the well-tested numpy path (which itself is pinned against
  LRF=3 R-M in PR 2's suite).
- **Kernel-limit rejection**: a synthetic RMLData with more than
  three particle channels per group must raise a clear
  ``NotImplementedError`` from the numba wrapper rather than
  silently truncating.
"""
from __future__ import annotations

import numpy as np
import pytest

from endf_parserpy import EndfParserCpp

from endf_userpy.mfsec_interpretation.mf2_interpretation_rml import (
    reconstruct as rml_reconstruct,
)
from endf_userpy.mfsec_interpretation.mf2_interpretation_rml_preproc import (
    rml_data_from_endf_dict,
)
from endf_userpy.primitives import array_ns

from _corpus import (
    resolve_rh103,
    resolve_pu239_rml,
    resolve_cu63_rml,
)


try:
    import numba  # noqa: F401
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _load(path):
    if path is None:
        pytest.skip('LRF=7 corpus file not present (see fetch.sh)')
    return EndfParserCpp(ignore_missing_tpid=True).parsefile(path)


@pytest.mark.skipif(not _HAS_NUMBA, reason='numba not installed')
@pytest.mark.parametrize('path_fn,label', [
    (resolve_rh103, 'Rh-103'),
    (resolve_pu239_rml, 'Pu-239'),
    (resolve_cu63_rml, 'Cu-63'),
])
def test_rml_numba_matches_numpy(path_fn, label):
    """Numpy vs numba backend equivalence on real LRF=7 corpus files."""
    d = _load(path_fn())
    data = rml_data_from_endf_dict(d)
    ein = np.geomspace(1e-3, 500.0, 32)
    out_np = rml_reconstruct(data, ein, array_ns.get_backend('numpy'))
    out_nb = rml_reconstruct(data, ein, array_ns.get_backend('numba'))
    for k in ('sct', 'cap', 'fis', 'pot', 'tot'):
        np.testing.assert_allclose(
            out_nb[k], out_np[k], rtol=1e-8, atol=1e-30,
            err_msg=f'{label} {k}',
        )


@pytest.mark.skipif(not _HAS_NUMBA, reason='numba not installed')
def test_rml_numba_rejects_wide_group():
    """Groups with more than 3 particle channels are rejected by
    the numba wrapper with a clear NotImplementedError."""
    from endf_userpy.mfsec_interpretation.mf2_interpretation_rml import (
        RMLData,
    )
    from endf_userpy.primitives.tab1 import TAB1
    r_ap = TAB1(
        x=np.array([1e-5, 1e11]),
        y=np.array([0.7, 0.7]),
        nbt=np.array([1], dtype=np.int32),
        intp=np.array([2], dtype=np.int32),
    )
    # 5-channel group: 1 gamma + 4 particle (elastic + 3 fission).
    data = RMLData(
        abn=1.0, spi=0.5, ki=2.196771e-3, r_ap=r_ap,
        krm=3, ifg=0, krl=0, naps=1,
        pp_ma=np.array([0.0, 1.0, 1.0]),
        pp_mb=np.array([100.0, 99.0, 99.0]),
        pp_za=np.zeros(3), pp_zb=np.zeros(3),
        pp_ia=np.array([1.0, 0.5, 0.0]),
        pp_ib=np.array([0.5, 0.5, 0.0]),
        pp_q=np.zeros(3), pp_pnt=np.array([0.0, 1.0, -1.0]),
        pp_shf=np.zeros(3),
        pp_mt=np.array([102.0, 2.0, 18.0]),
        pp_pa=np.zeros(3), pp_pb=np.zeros(3),
        pp_incident_idx=2,
        group_aj=np.array([0.5]),
        group_pj=np.array([0.0]),
        group_g=np.array([0.5]),
        group_nch=np.array([5], dtype=np.int32),
        # 5 channels: gamma, elastic, fis, fis, fis
        ch_ppi=np.array([[1.0, 2.0, 3.0, 3.0, 3.0]]),
        ch_l=np.zeros((1, 5)),
        ch_sch=np.zeros((1, 5)),
        ch_bnd=np.zeros((1, 5)),
        ch_ape=np.full((1, 5), 0.7),
        ch_apt=np.full((1, 5), 0.7),
        ch_active=np.ones((1, 5), dtype=bool),
        res_group=np.zeros(1, dtype=np.int32),
        res_er=np.array([50.0]),
        res_gam=np.array([[0.1, 1.0, 0.05, 0.03, 0.02]]),
    )
    with pytest.raises(NotImplementedError, match='particle channels'):
        rml_reconstruct(data, np.array([10.0]), array_ns.get_backend('numba'))
