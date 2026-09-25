import warnings
import numpy as np
from ..mfsec_interpretation import mf4_interpretation as mf4_interp
from ..mfsec_interpretation import mf5_interpretation as mf5_interp
from ..mfsec_interpretation import mf6_interpretation as mf6_interp
from ..mfsec_interpretation import mf6_interpretation_helpers as mf6_help
from ..mfsec_interpretation import mf12_interpretation as mf12_interp
from ..mfsec_interpretation import mf14_interpretation as mf14_interp
from ..mfsec_interpretation import mf15_interpretation as mf15_interp
from ..primitives import array_ns
from ..primitives.physical_constants import get_zap_for_particle
from ..primitives.properties import (
    is_zap_consistent,
    has_mf4_mt,
    has_mf5_mt,
    has_mf6_mt,
    has_mf12_mt,
    has_mf14_mt,
    has_mf15_mt,
)


_N_ZAP = get_zap_for_particle('n')
_G_ZAP = get_zap_for_particle('g')
from .distribution1d_helpers import (
    integrate_mf6_dist2d_over_eout,
    integrate_mf6_dist2d_over_mu,
    convert_angdist_to_energydist,
)
import logging


module_logger = logging.getLogger(__name__)


# Per-(file, MT) dedup for the "MF15 present, MF12 has no Eg=0
# continuum placeholder" warning (issue #103 / audit D3). `id()`
# can be reused after GC, but the worst case is a missed warning
# rather than a wrong result -- same trade as `_isomer_warning_seen`
# in endf_userpy.quantities.
_mf15_no_placeholder_warned = set()


def compute_angdist_values(
    endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab=True,
    xp=None,
):
    """Composition-layer angular distribution.

    ``xp=None`` (default) is numpy. Passing a JAX adapter threads
    tracers through the MF4, MF6 and MF14 (gamma) branches end-to-
    end so ``jax.grad`` reaches file-side leaves in every path.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    if not is_zap_consistent(endf_dict, mt, zap):
        raise ValueError(f'MT={mt} and ZAP={zap} are not consistent')

    module_logger.debug(f'determine angular distribution for MT: {mt}')

    if has_mf4_mt(endf_dict, mt) and zap == _N_ZAP:
        module_logger.debug('--> found discrete LAW in MF4')
        return mf4_interp.compute_angdist_values(
            endf_dict, mt, energies_in, angle_cosines_out, to_lab, xp=xp,
        )

    elif has_mf6_mt(endf_dict, mt):
        found_angdist = False
        angdist = 0.0
        if mf6_help.has_cont_part(endf_dict, mt, zap):
            module_logger.debug('--> integrate MF6')
            found_angdist = True
            angdist = angdist + integrate_mf6_dist2d_over_eout(
                endf_dict, mt, zap, energies_in, angle_cosines_out,
                to_lab, xp=xp,
            )
        if mf6_help.has_angdist_part(endf_dict, mt, zap):
            module_logger.debug('--> found discrete LAW in MF6')
            found_angdist = True
            angdist = angdist + mf6_interp.compute_angdist_values(
                endf_dict, mt, zap, energies_in, angle_cosines_out,
                to_lab, xp=xp,
            )
        if mf6_help.has_disc_part(endf_dict, mt, zap):
            module_logger.debug('--> found LAW=1 discrete-line angular in MF6')
            found_angdist = True
            # The LAW=1 discrete-line path is currently numpy-only
            # (compute_law1_discrete_lines has not been xp-threaded);
            # convert to xp at the boundary so the composed sum stays
            # on the caller's backend.
            angdist = angdist + xp.asarray(_compute_mf6_law1_disc_angdist(
                endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab,
            ))
        if found_angdist:
            return angdist

    if zap == _G_ZAP and has_mf14_mt(endf_dict, mt):
        module_logger.debug('--> found gamma angular distribution in MF14')
        return _compute_mf14_gamma_angdist(
            endf_dict, mt, energies_in, angle_cosines_out, xp=xp,
        )

    module_logger.debug(
        f'no angular distribution reconstructable for MT={mt}, '
        f'ZAP={zap}; returning zeros'
    )
    return xp.zeros(
        (len(energies_in), len(angle_cosines_out)), dtype=xp.float64,
    )


def _compute_mf6_law1_disc_angdist(
    endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab,
):
    """Angular projection of MF6/LAW=1 discrete-line content
    (issue #55).

    Returns ``f_disc(mu | Ein)`` shape ``(n_einc, n_mus)``. The per-
    line ``amp_disc(einc, mu, k)`` from
    ``mf6_interp.compute_law1_discrete_lines`` already has the per-
    line yield weight embedded (the b(k) amplitudes carry the split
    between discrete and continuum), so the sum over k directly
    gives the discrete-only contribution to the total angular
    distribution with the correct normalisation relative to
    ``compute_daxs``: for a subsection whose LAW=1 yield is entirely
    in discrete lines and whose per-line angular distribution is
    isotropic, ``sum_k amp_disc`` integrates to 1 over mu (the full
    normalisation); for mixed discrete + continuum subsections it
    integrates to ``y_disc / Y_total``, and the sibling continuum
    branch of `compute_angdist_values` (integrate_mf6_dist2d_over_eout)
    contributes the complementary ``y_cont / Y_total``.

    Cases where the b(k) amplitudes are all zero -- a TENDL/JEFF
    file quirk noted in the LAW=1 discrete-line diagnosis history
    -- return an all-zero angular distribution; the continuum
    branch above handles those files' actual gamma content.
    """
    _, amp_disc = mf6_interp.compute_law1_discrete_lines(
        endf_dict, mt, zap, energies_in, angle_cosines_out, to_lab,
    )
    # amp_disc shape (n_einc, n_mus, K).
    return amp_disc.sum(axis=-1)


def _compute_mf14_gamma_angdist(
    endf_dict, mt, energies_in, angle_cosines_out, xp=None,
):
    """Yield-weighted gamma angular distribution from MF14 + MF12.

    Combines per-photon-line angular distributions from MF14 with
    per-photon-line yields from MF12 into a single, MU-normalised
    angular distribution ``f(mu | Ein)`` (i.e. ``int f dmu = 1``)
    suitable for consumption by ``compute_daxs``. Handles three
    layouts:

    - MF14 LI=1 (fully isotropic, the common case in every ad-hoc
      corpus file today): returns ``0.5`` regardless of yields or
      photon energies.
    - MF14 LI=0 with Legendre coefficients per photon line, plus a
      MF12 Eg=0 continuum placeholder: yield-weighted average of the
      MF14 discrete-line distributions plus an isotropic continuum
      contribution weighted by ``y_cont / Y_total``.
    - No MF12 (unusual for a gamma-emitting MT with MF14): treated
      as fully isotropic, since without yield information the only
      neutral choice is to trust MF14's LI flag.

    The MF14 tabulated (LTT=2) branch of the reader is not implemented
    upstream; a MT reaching that branch will raise NotImplementedError
    from ``mf14_interp.compute_angdist_values``.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    # ``energies_in`` and ``angle_cosines_out`` may be JAX tracers
    # (issue #220 PR 3). Keep them as xp arrays throughout so
    # ``jax.grad`` reaches file-side MF12 / MF14 leaves; only
    # ``photon_energies`` (a static file-side attribute) stays
    # numpy. ``len()`` on a shape-known jax array returns the static
    # leading dim so ``n_einc`` / ``n_mus`` remain Python ints.
    n_einc = len(energies_in)
    n_mus = len(angle_cosines_out)

    mtsec = endf_dict[14][mt]
    if mtsec['LI'] == 1:
        # Fully isotropic: f(mu) = 1/2, integrates to 1 over mu.
        return xp.full((n_einc, n_mus), 0.5, dtype=xp.float64)

    if not has_mf12_mt(endf_dict, mt):
        # LI=0 without MF12 yields is a degenerate case: fall back
        # to isotropic rather than guess a partition.
        return xp.full((n_einc, n_mus), 0.5, dtype=xp.float64)

    pes = np.asarray(mf12_interp.get_photon_energies(endf_dict, mt), dtype=float)
    yields_all = mf12_interp.compute_photon_yields(
        endf_dict, mt, energies_in, pes, xp=xp,
    )
    y_total = xp.sum(yields_all, axis=1)

    disc_mask = pes > 0.0
    disc_pes = pes[disc_mask]
    disc_idcs = np.where(disc_mask)[0]
    cont_idcs = np.where(~disc_mask)[0]
    disc_yields = yields_all[:, disc_idcs]
    if cont_idcs.size > 0:
        y_cont = xp.sum(yields_all[:, cont_idcs], axis=1)
    else:
        y_cont = xp.zeros(n_einc, dtype=xp.float64)

    if len(disc_pes) == 0:
        # Only continuum-placeholder photons (no MF14 per-line data
        # applies): isotropic.
        return xp.full((n_einc, n_mus), 0.5, dtype=xp.float64)

    per_line = mf14_interp.compute_angdist_values(
        endf_dict, mt, energies_in, disc_pes, angle_cosines_out, xp=xp,
    )  # (n_einc, n_disc, n_mus)

    # xp-safe divide: zero-out where y_total == 0 without producing
    # NaN under jax (grad would then be NaN).
    safe_total = xp.where(y_total > 0, y_total, 1.0)
    weights_disc = xp.where(
        (y_total > 0).reshape(-1, 1),
        disc_yields / safe_total.reshape(-1, 1),
        xp.asarray(0.0),
    )
    cont_frac = xp.where(y_total > 0, y_cont / safe_total, xp.asarray(0.0))

    f_disc_avg = xp.einsum('ei,eim->em', weights_disc, per_line)
    f_cont = 0.5 * cont_frac.reshape(-1, 1)
    return f_disc_avg + f_cont


def compute_energydist_values(
    endf_dict, mt, zap, energies_in, energies_out, to_lab=True, xp=None,
):
    """Composition-layer energy distribution.

    ``xp=None`` (default) is numpy. Passing a JAX adapter threads
    tracers through the MF5, MF6 and MF15 (gamma) branches end-to-
    end so ``jax.grad`` reaches file-side leaves. The
    ``convert_angdist_to_energydist`` fallback (MF4 angdist -> E'
    via LAB Jacobian) runs on numpy internally and returns xp-
    native at the boundary.
    """
    if xp is None:
        xp = array_ns.get_backend('numpy')
    if not is_zap_consistent(endf_dict, mt, zap):
        raise ValueError(f'MT={mt} and ZAP={zap} are not consistent')

    module_logger.debug(f'determine energy distribution for MT: {mt}')

    if has_mf5_mt(endf_dict, mt) and zap == _N_ZAP:
        if to_lab is not True:
            raise ValueError(
                f"Energy spectrum for MT={mt}, ZAP={zap} reconstruction "
                "from MF5 only possible with `to_lab=True` argument."
            )
        module_logger.debug('--> found energy spectrum in MF5')
        return mf5_interp.compute_spectrum(
            endf_dict, mt, energies_in, energies_out, xp=xp,
        )

    elif has_mf4_mt(endf_dict, mt) and zap == _N_ZAP:
        module_logger.debug('--> found discrete angdist in MF4')
        return convert_angdist_to_energydist(
            lambda endf_dict, mt, _, energies_in, energies_out, to_lab: (
                mf4_interp.compute_angdist_values(
                    endf_dict, mt, energies_in, energies_out, to_lab
                )
            ),
            endf_dict, mt, zap, energies_in, energies_out, to_lab, xp=xp,
        )

    elif has_mf6_mt(endf_dict, mt):
        found_energydist = False
        energydist = 0.0
        if mf6_help.has_cont_part(endf_dict, mt, zap):
            module_logger.debug('--> integrate MF6')
            found_energydist = True
            energydist = energydist + integrate_mf6_dist2d_over_mu(
                endf_dict, mt, zap, energies_in, energies_out, to_lab, xp=xp,
            )
        if mf6_help.has_angdist_part(endf_dict, mt, zap):
            module_logger.debug('--> found discrete angdist in MF6')
            found_energydist = True
            energydist = energydist + convert_angdist_to_energydist(
                mf6_interp.compute_angdist_values,
                endf_dict, mt, zap, energies_in, energies_out, to_lab, xp=xp,
            )
        if found_energydist:
            return energydist

    elif has_mf15_mt(endf_dict, mt) and zap == get_zap_for_particle('g'):
        module_logger.debug('--> found continuous gamma energy spectrum in MF15')
        spec = mf15_interp.compute_spectrum(
            endf_dict, mt, energies_in, energies_out, xp=xp,
        )
        # Weight the MF15 continuum shape by the continuum yield
        # fraction from MF12 (issue #54). `compute_dexs` multiplies
        # what we return here by `compute_yields`, which sums ALL
        # photon yields declared in MF12 (discrete lines + Eg=0
        # continuum placeholder). Without this weighting the MF15
        # spectrum is multiplied by Y_total instead of y_cont, and
        # at incident energies where the MF12 discrete lines carry
        # nonzero weight but the Eg=0 placeholder does not (Al-27
        # MT 102 below ~10 keV is the corpus example: Y_disc ~ 2.17,
        # Y_cont = 0) the current form injects a spurious continuum
        # contribution proportional to the discrete-line yield.
        # Above the discrete/continuum crossover (~10 keV for Al-27)
        # Y_disc drops to zero and the two forms are identical; the
        # fix is a no-op there.
        if has_mf12_mt(endf_dict, mt):
            pes = np.asarray(
                mf12_interp.get_photon_energies(endf_dict, mt),
                dtype=float,
            )
            cont_mask = pes == 0.0
            if np.any(cont_mask):
                yields_all = mf12_interp.compute_photon_yields(
                    endf_dict, mt, energies_in, pes, xp=xp,
                )
                cont_idcs = np.where(cont_mask)[0]
                y_cont = xp.sum(yields_all[:, cont_idcs], axis=1)
                y_total = xp.sum(yields_all, axis=1)
                # xp-safe divide: zero-out where y_total == 0 without
                # producing NaN under jax (which would break grad).
                safe_total = xp.where(y_total > 0, y_total, 1.0)
                frac = xp.where(
                    y_total > 0, y_cont / safe_total, xp.asarray(0.0),
                )
                spec = spec * frac.reshape(-1, 1)
            else:
                # MF12 declares no Eg=0 continuum placeholder, so
                # the file's own convention says there is no
                # continuum for this MT. Any MF15 content here is
                # inconsistent with MF12; drop it rather than let
                # it inflate the sum. Emit one warning per
                # (file, MT) so users see why their MF15 content
                # vanished (issue #103 / audit D3).
                key = (id(endf_dict), int(mt))
                if key not in _mf15_no_placeholder_warned:
                    _mf15_no_placeholder_warned.add(key)
                    warnings.warn(
                        f"MT={mt} has MF15 continuous gamma "
                        f"spectrum data but MF12 declares no Eg=0 "
                        f"continuum placeholder, so the file's own "
                        f"y_cont normalisation is zero and the MF15 "
                        f"contribution to dxs/dE is dropped to avoid "
                        f"inflating the sum. If the MF15 content is "
                        f"physical, the file's MF12 needs an Eg=0 "
                        f"row with the appropriate continuum yield; "
                        f"otherwise the drop is correct.",
                        UserWarning, stacklevel=2,
                    )
                spec = xp.zeros_like(spec)
        # No MF12 at all: MF15 stands alone, keep unweighted
        # behaviour so the reaction-string yield fallback (mult=1
        # for (n,g)) times MF15 still integrates to sigma.
        return spec

    # No representable continuous energy spectrum for this (MT, ZAP):
    # either MF6 with only LAW=1 ND>0 discrete-line content (handled
    # separately by the LAW=1 discrete-line folder in the broadening
    # pipeline, and by MF12/14 for the unbroadened case), MF15 without
    # gamma ZAP, or no relevant MF section at all. Return zeros so
    # cumulative summation over MTs is well-defined; used to raise
    # IndexError, which forced defensive try/except workarounds in
    # downstream callers (issue #31).
    module_logger.debug(
        f'no continuum or angdist energy spectrum reconstructable for '
        f'MT={mt}, ZAP={zap}; returning zeros'
    )
    return xp.zeros(
        (len(energies_in), len(energies_out)), dtype=xp.float64,
    )
