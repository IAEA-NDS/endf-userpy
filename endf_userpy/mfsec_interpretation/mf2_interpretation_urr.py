"""Unresolved-resonance region (LRU=2) reconstruction: dataclass +
kernel.

ENDF-6 MF2/MT151 LRU=2 (URR) parameterises the average cross
section over the region where individual resonances can no longer
be resolved. Instead of resonance-by-resonance parameters, the
file lists, per spin group (L, J), the average widths
(``GN0``, ``GG``, ``GF``, ``GX``) and level spacing (``D``) as
functions of incident energy, along with degrees-of-freedom
(``AMUN``, ``AMUG``, ``AMUF``, ``AMUX``) that specify how each
width fluctuates around its average via chi-squared distributions.

Layout mirrors :class:`~mf2_interpretation_mlbw.MLBWData` /
:class:`~mf2_interpretation_reichmoore.RMData`: scalars for the
range, per-group arrays, tabulated widths. See
:mod:`mf2_interpretation_urr_preproc` for the ENDF-dict lift.

Scope
-----

- **LRF=2** (Case C): energy-dependent widths tabulated at knot
  energies per J-group. Covers essentially every modern actinide
  URR range. LRF=1 (Case A, constant widths) is a special case
  of LRF=2 with NE=1 and is deferred to a follow-up.
- **LSSF=0 and LSSF=1**: the reconstruction kernel is the same
  either way; composition
  (:mod:`~endf_userpy.quantities_mt_zap.resonance_composition`)
  is what decides whether to add this contribution to MF3
  (LSSF=0) or leave MF3 alone (LSSF=1, where MF3 already carries
  the average XS).
- **Backend-agnostic**: numpy / JAX via
  :mod:`endf_userpy.primitives.array_ns`; numba path in
  :mod:`mf2_interpretation_urr_numba`. JAX is a first-class
  consumer here since the whole motivation for including it is
  autodiff-through-URR fitting workflows.

Not covered in this module (deliberate scope):

- **Doppler broadening** of the reconstructed average XS: a
  separate convolution layer, out of scope until Doppler lands.
- **Case B** (LRF=1 with fission-only overrides): if a real file
  ever needs it, plug in with a preproc-only shim; the kernel
  itself is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..primitives import tab1


@dataclass
class URRData:
    """Natural-size URR (LRU=2 LRF=2) input for one isotope / one range.

    Fields:

    Scalars (per range):
        ``abn``:  isotopic abundance in the material (0-d array).
        ``spi``:  target spin ``I`` (0-d array; used for
                  statistical weight and hard-sphere phase).
        ``ap``:   scattering radius (0-d array; fed into r_ap).
        ``awri``: mass ratio ``m_target / m_neutron`` from the
                  first L-group (all L-groups share a single
                  AWRI in every real file; enforced at preproc).
        ``ki``:   wavenumber coefficient
                  ``k(E) = ki * sqrt(E)``.
        ``naps``: NAPS flag (0/1/2) as Python int for control
                  flow in the preproc / channel-radius helper.

    Per J-group (shape ``(ngroups,)``):
        ``group_l``:    L per group.
        ``group_j2``:   ``2*J`` per group (integer).
        ``group_g``:    statistical weight
                        ``g_J = (2J + 1) / (2 (2I + 1))``.
        ``group_amun``, ``group_amug``, ``group_amuf``,
        ``group_amux``: chi-squared degrees of freedom for the
                        neutron / gamma / fission / competitive
                        widths.
        ``group_int``: interpolation code for the energy tables
                       (ENDF INT convention; 2 = lin-lin, most
                       common; up to 5 = log-log).

    Per J-group tabulated widths (shape ``(ngroups, ne)`` where
    ``ne`` is common across groups in this implementation; real
    URR files fix NE per range):
        ``table_es``:  knot energies (eV).
        ``table_d``:   average level spacing ``<D>``.
        ``table_gn0``: reduced average neutron width; the
                       physical width at ``E`` is recovered as
                       ``Gn(E) = GN0(E) * sqrt(E) * factor(L, rho)``
                       inside the reconstruction kernel.
        ``table_gg``:  average gamma width.
        ``table_gf``:  average fission width.
        ``table_gx``:  average competitive width.

    Radii:
        ``r_a``:  channel radius as :class:`~primitives.tab1.TAB1`
                  vs E (constant TAB1 unless NAPS=2).
        ``r_ap``: scattering radius as TAB1 vs E (may be
                  energy-dependent via NRO=1 / APE tables).
    """

    abn: np.ndarray
    spi: np.ndarray
    ap: np.ndarray
    awri: np.ndarray
    ki: np.ndarray
    naps: int

    group_l: np.ndarray
    group_j2: np.ndarray
    group_g: np.ndarray
    group_amun: np.ndarray
    group_amug: np.ndarray
    group_amuf: np.ndarray
    group_amux: np.ndarray
    group_int: np.ndarray

    table_es: np.ndarray
    table_d: np.ndarray
    table_gn0: np.ndarray
    table_gg: np.ndarray
    table_gf: np.ndarray
    table_gx: np.ndarray

    r_a: tab1.TAB1
    r_ap: tab1.TAB1


def reconstruct(data: URRData, energies_in, xp) -> Any:
    """URR average cross sections at ``energies_in``.

    NOT YET IMPLEMENTED. Preproc + dataclass land first so
    integration tests and downstream composition wiring can
    proceed against a stable shape; the physics kernel arrives
    in the next commit on this branch.

    Callers that hit this today get a clear ``NotImplementedError``
    naming the follow-up commit, rather than a silent bad number.
    """
    raise NotImplementedError(
        "URR reconstruction kernel not yet implemented. "
        "This is the preproc-only landing commit; the kernel "
        "(chi-squared fluctuation-factor integrals + per-group "
        "sum) follows in the next commit on this branch. See PR "
        "for progress."
    )
