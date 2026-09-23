"""Public MF6 LAW=1 continuum ``E'``-integral wrappers.

Backend-agnostic (numpy default; ``xp=array_ns.get_backend('jax')``
threads a JAX tracer through). Composes the preproc dataclass
(:mod:`mf6_law1_preproc`) with the backend-agnostic
``mu``-integrator (:mod:`mf6_law1_epintegral`).

The historical Fortran-backed implementations remain in
:mod:`mf6_interpretation_integrals_fort` as the equivalence oracle
for the port; ``get_energydist_from_subsec_law1_dynamic_mesh`` is a
compatibility re-export of the Fortran adaptive variant (PR-B in
the LAW=1 series will replace it with a pure-Python adaptive port).
"""
import numpy as np

from ..primitives import array_ns
from .mf6_interpretation_helpers import (
    pad_outside_energydist_values,
)
from . import mf6_law1_epintegral, mf6_law1_preproc
# Re-export the Fortran adaptive-mesh wrapper under the historical
# name so callers that reach for the adaptive integrator can still
# find it. Will be replaced by a Python port in PR-B (issue #47).
from .mf6_interpretation_integrals_fort import (
    get_energydist_from_subsec_law1_dynamic_mesh_fort
    as get_energydist_from_subsec_law1_dynamic_mesh,
)

__all__ = [
    'get_energydist_from_subsec_law1',
    'get_energydist_from_subsec_law1_dynamic_mesh',
]


def get_energydist_from_subsec_law1(
    endf_dict, mt, subsec_num, energies_in, energies_out, to_lab,
    xp=None, n_gl=10, panel_idx=None,
):
    """Backend-agnostic ``f(E, E')`` for one MF6 LAW=1 continuum
    subsection.

    Passing ``xp=array_ns.get_backend('jax')`` and storing a tracer
    in ``endf_dict[6][mt]['subsection'][subsec_num]['b'][panel][row]
    [col]`` lets ``jax.grad`` reach back to the file-stored
    parameters.

    ``panel_idx`` (Python int) is the entry point for ``jax.grad``
    wrt ``energies_in`` / ``energies_out``: with a static panel
    choice, both arrays are kept xp-native and flow through the
    integrator as tracers. The caller must keep ``energies_in``
    inside ``[ei_mesh[panel_idx], ei_mesh[panel_idx + 1]]`` --
    grad across a panel knot is undefined (the amplitude has
    physical C0 kinks there). The default ``panel_idx=None`` uses
    the section-wide multi-panel dispatcher and applies the
    standard outside-range zero-padding.

    Numerics: kink-aware polar-angle Gauss-Legendre. Subpanels are
    aligned with the LEP-piecewise ``E'`` knots of the two
    bracketing panels, so ``n_gl`` controls exponential convergence
    within each smooth piece. See :mod:`mf6_law1_epintegral` for
    accuracy notes.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    if panel_idx is not None:
        return _get_energydist_from_subsec_law1_single_panel(
            endf_dict, mt, subsec_num, energies_in, energies_out,
            to_lab, xp=xp, n_gl=n_gl, panel_idx=panel_idx,
        )
    return _get_energydist_from_subsec_law1_multi_panel(
        endf_dict, mt, subsec_num, energies_in, energies_out,
        to_lab, xp=xp, n_gl=n_gl,
    )


@pad_outside_energydist_values
def _get_energydist_from_subsec_law1_multi_panel(
    endf_dict, mt, subsec_num, energies_in, energies_out, to_lab,
    xp=None, n_gl=10,
):
    """Section-wide dispatcher with outside-range zero-padding."""
    data = mf6_law1_preproc.mf6_law1_data_from_endf_dict(
        endf_dict, mt, subsec_num, xp=xp,
    )
    ein = np.asarray(energies_in, dtype=float)
    eout = np.asarray(energies_out, dtype=float)
    return mf6_law1_epintegral.integrate_law1_spectrum(
        data, ein, eout, to_lab, xp=xp, n_gl=n_gl,
    )


def _get_energydist_from_subsec_law1_single_panel(
    endf_dict, mt, subsec_num, energies_in, energies_out, to_lab,
    xp, n_gl, panel_idx,
):
    """Autodiff-safe single-panel path. No decorator, no numpy
    conversion of ``energies_in`` / ``energies_out`` (tracers pass
    through)."""
    data = mf6_law1_preproc.mf6_law1_data_from_endf_dict(
        endf_dict, mt, subsec_num, xp=xp,
    )
    return mf6_law1_epintegral.integrate_law1_spectrum(
        data, energies_in, energies_out, to_lab,
        xp=xp, n_gl=n_gl, panel_idx=panel_idx,
    )


