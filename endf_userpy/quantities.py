import numpy as np
import warnings
from .primitives import physical_constants as physconst
from .primitives import properties as prop
from .primitives import reactions as reac
from .primitives.helpers import unpack_za
from .quantities_mt_zap import quantities as quant_mt_zap
from .quantities_mt_zap import distribution1d as dist1d
from .quantities_mt_zap import selectors
from .quantities_mt_zap import ddx_broadening as ddxb
import logging
# TODO: Remove direct use of mf6_interpretation module in this module
from .mfsec_interpretation import mf6_interpretation as mf6interp
from .mfsec_interpretation import mf8_interpretation as mf8interp


# Cache of (id(endf_dict), mt, zap, lfs) tuples we have already warned
# about, so the warning fires once per file+MT+residual+isomer rather
# than once per call. id() can be reused after garbage collection but
# the worst case is a missed warning, never a wrong result.
_isomer_warning_seen = set()


def _warn_if_missing_isomer_routing(
    endf_dict, residual_str, za_residual, lfs, mt5_contrib
):
    """Warn for MTs that could physically produce ``za_residual`` but
    lack an MF8 entry to resolve the isomer state. Called only when
    the user explicitly requested a non-ground LFS.
    """
    avail_mts = quant_mt_zap.get_reaction_mt_numbers(endf_dict)
    has_mf8 = (8 in endf_dict)
    for mt in sorted(avail_mts):
        if mt == 5 and not mt5_contrib:
            continue
        if not selectors.contains_residual_za(endf_dict, mt, za_residual):
            continue
        if has_mf8 and mt in endf_dict[8]:
            continue
        key = (id(endf_dict), mt, int(za_residual), int(lfs))
        if key in _isomer_warning_seen:
            continue
        _isomer_warning_seen.add(key)
        warnings.warn(
            f"MT={mt} has no MF8 isomer routing in this file; "
            f"production of '{residual_str}' from MT={mt} is unresolved "
            f"at the LFS level. Returning 0 for this isomer. "
            f"The metastable may still be produced according to other "
            f"evaluations.",
            UserWarning,
            stacklevel=3,
        )


def _format_lfs_suffix(lfs):
    if lfs == 0:
        return 'g'
    if lfs == 1:
        return 'm'
    return f'm{lfs}'


def _format_residual(zap, lfs):
    z, a = unpack_za(zap)
    sym = physconst.ELEMENT_SYMBOLS[z]
    return f'{sym}-{a}{_format_lfs_suffix(lfs)}'


module_logger = logging.getLogger(__name__)


# TODO: check and complete


def get_available_reactions(endf_dict):
    mts = quant_mt_zap.get_reaction_mt_numbers(endf_dict)
    reacs = []
    for mt in mts:
        if not reac.is_known_reaction_mt(mt):
            module_logger.debug(
                f'skipping MT={mt} from available reactions '
                f'(not in reaction table; e.g. HEATR heating number)'
            )
            continue
        reacs.append(prop.get_reaction_string_for_mt(endf_dict, mt))
    return reacs


def get_declared_residuals(endf_dict):
    """List residuals declared via MF8, with isomer suffix.

    Returns a sorted list of strings like ``["Co-58g", "Co-58m",
    "Co-60g", "Co-60m"]``. Empty list if MF8 is absent. The result
    is what the file explicitly says, not what physics-based MT/ZAP
    accounting would predict.
    """
    pairs = mf8interp.get_declared_zap_lfs(endf_dict)
    return [
        _format_residual(zap, lfs)
        for zap, lfs in sorted(pairs)
    ]


def is_residual_declared(endf_dict, residual_nucleus):
    """True iff the residual (with isomer state) is declared in MF8.

    The residual string follows the same format as for
    ``get_residual_production_xs`` (e.g. ``"Co-58m"``,
    ``"27-Co-58g"``). If no isomer suffix is given, returns True iff
    the bare nucleus is declared in any state.
    """
    zap, lfs = physconst.get_za_for_residual_nucleus(residual_nucleus)
    if lfs is None:
        return len(mf8interp.get_declared_lfs_for_zap(endf_dict, zap)) > 0
    return mf8interp.is_zap_lfs_declared(endf_dict, zap, lfs)


def get_declared_isomer_states(endf_dict, residual_nucleus):
    """Isomer-state suffixes declared in MF8 for a given nucleus.

    Any isomer suffix on the input is ignored: the function asks
    "which states of this nucleus does the file declare?". For
    ``"Co-58"`` it returns e.g. ``["g", "m"]``. Empty list if the
    nucleus has no MF8 entry.
    """
    zap, _ = physconst.get_za_for_residual_nucleus(residual_nucleus)
    return [
        _format_lfs_suffix(lfs)
        for lfs in mf8interp.get_declared_lfs_for_zap(endf_dict, zap)
    ]


def get_incident_energies(endf_dict, reaction):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    mts = quant_mt_zap.get_reaction_mt_numbers(endf_dict)
    select_mts = [
        mt for mt in mts
        if selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
    ]
    module_logger.debug('selected ' + ','.join(str(mt) for mt in select_mts))
    energy_meshes = [
        quant_mt_zap.get_incident_energies(endf_dict, mt)
        for mt in select_mts
    ]
    return np.unique(np.concatenate(energy_meshes))


def get_emission_energies(endf_dict, reaction, particle, nofail=False):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    mts = quant_mt_zap.get_reaction_mt_numbers(endf_dict)
    select_mts = [
        mt for mt in mts
        if selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        and selectors.contains_zap(endf_dict, mt, zap)
    ]
    module_logger.debug('selected ' + ','.join(str(mt) for mt in select_mts))
    energy_meshes = [
        mf6interp.get_emission_energies(endf_dict, mt, zap, nofail)
        for mt in select_mts
    ]
    return np.unique(np.concatenate(energy_meshes))


def get_reaction_xs(endf_dict, reaction, energies_in, mt5_contrib=True):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    avail_mts = set(quant_mt_zap.get_reaction_mt_numbers(endf_dict))
    iter_mts = avail_mts.copy()
    iter_mts.update(user_mts)
    xs = np.zeros_like(energies_in, dtype=float)
    proj = prop.get_projectile(endf_dict)
    for mt in sorted(iter_mts):
        module_logger.debug(f'consider MT={mt} for reaction xs')
        should_select = selectors.satisfies_select_heuristic(
            endf_dict, mt, user_mts
        )
        mt_available = mt in avail_mts
        if should_select and mt_available:
            module_logger.debug(f'select MT={mt} for reaction xs')
            cur_xs = quant_mt_zap.compute_xs(endf_dict, mt, energies_in)
            xs += cur_xs

        # add associated MT5 component if available and permissible
        if (mt in user_mts
                and mt5_contrib
                and 5 not in user_mts
                and not reac.any_ancestor_in_mts(5, user_mts)
                and reac.is_unique_path_to_residual(proj, mt)):
            if not mt_available or not should_select:
                cur_xs = np.zeros_like(energies_in, dtype=float)
            eincs_sel = (~np.bool(mt_available)) | (cur_xs == 0.0)
            mt5_xs = quant_mt_zap.compute_xs_mt5_contrib(
                endf_dict, mt, energies_in[eincs_sel]
            )
            xs[eincs_sel] += mt5_xs
            if np.any(mt5_xs != 0.0):
                module_logger.debug(f'include MF6/MT5 component for MT={mt}')
    return xs


def get_residual_production_xs(endf_dict, residual_nucleus, energies_in, mt5_contrib=True):
    za_residual, level = physconst.get_za_for_residual_nucleus(residual_nucleus)
    if level is not None:
        module_logger.debug(f'user requested isomeric state LFS={level}')
    if level not in (None, 0):
        _warn_if_missing_isomer_routing(
            endf_dict, residual_nucleus, za_residual, level, mt5_contrib
        )
    xs = quant_mt_zap.compute_cumulative_quantity(
        lambda endf_dict, mt: quant_mt_zap.compute_residual_xs(
            endf_dict, mt, za_residual, level, energies_in
        ),
        lambda endf_dict, mt: (
            (mt5_contrib or mt != 5) and
            selectors.contains_residual_za_and_lfs(
                endf_dict, mt, za_residual, level
            ) and
            selectors.satisfies_select_heuristic(endf_dict, mt)
        ),
        endf_dict
    )
    if xs is None:
        xs = np.zeros_like(energies_in, dtype=float)
    return xs


def get_particle_production_xs(endf_dict, reaction, particle, energies_in):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        quant_mt_zap.compute_prodxs,
        lambda endf_dict, mt, zap, energies_in: (
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
            and selectors.contains_zap(endf_dict, mt, zap)
        ),
        endf_dict, zap, energies_in
    )


def get_particle_production_dxs_dE(
    endf_dict, reaction, particle, energies_in, energies_out
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        quant_mt_zap.compute_dexs,
        lambda endf_dict, mt, zap, energies_in, energies_out: (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        ),
        endf_dict, zap, energies_in, energies_out
    )


def get_particle_production_dxs_dmu(
    endf_dict, reaction, particle, energies_in, angle_cosines_out
):
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)
    return quant_mt_zap.compute_cumulative_quantity(
        quant_mt_zap.compute_daxs,
        lambda endf_dict, mt, zap, energies_in, angle_cosines_out: (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        ),
        endf_dict, zap, energies_in, angle_cosines_out
    )


def get_particle_production_ddxs(
    endf_dict, reaction, particle, energies_in, energies_out, angle_cosines_out,
    broadening=None,
):
    """Double-differential cross section for particle production.

    Parameters
    ----------
    endf_dict, reaction, particle, energies_in, energies_out, angle_cosines_out
        As before.
    broadening : None or float or (callable, float), optional
        If None (default), behaviour is unchanged: only channels with
        a true continuous (E_out, mu) distribution contribute, and
        the result is the bare DDX. If broadening is provided, the
        result instead sums (i) the continuous DDX folded with the
        kernel along E_out, and (ii) the discrete two-body channels
        (MT 2 elastic, MT 51..90 etc.) with the kinematic delta
        replaced by the kernel.

        Accepted forms:
          - scalar `sigma` (eV) -> Gaussian kernel of that width.
          - tuple `(kernel_callable, width)` -> custom kernel; the
            callable is `kernel(delta_E)` and `width` is its
            characteristic scale (passed to the FFT mesh control).
    """
    user_mts = [reac.translate_reaction_string_to_mt(reaction)]
    zap = physconst.get_zap_for_particle(particle)

    kernel, kernel_width = _normalize_broadening(broadening)
    if kernel is None:
        return quant_mt_zap.compute_cumulative_quantity(
            quant_mt_zap.compute_ddxs,
            lambda endf_dict, mt, zap, energies_in, energies_out, angle_cosines_out: (
                selectors.contains_zap(endf_dict, mt, zap) and
                selectors.has_continuous_ddx(endf_dict, mt, zap) and
                selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
            ),
            endf_dict, zap, energies_in, energies_out, angle_cosines_out
        )

    def cont_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_continuous_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel, kernel_width=kernel_width,
        )

    def cont_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_continuous_ddx(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        )

    def disc_compute(endf_dict, mt, zap, einc, eouts, mus):
        return ddxb.compute_ddx_discrete_broadened(
            endf_dict, mt, zap, einc, eouts, mus,
            kernel=kernel,
        )

    def disc_select(endf_dict, mt, zap, einc, eouts, mus):
        return (
            selectors.contains_zap(endf_dict, mt, zap) and
            selectors.has_discrete_two_body_ddx(endf_dict, mt, zap) and
            selectors.satisfies_select_heuristic(endf_dict, mt, user_mts)
        )

    cont = quant_mt_zap.compute_cumulative_quantity(
        cont_compute, cont_select,
        endf_dict, zap, energies_in, energies_out, angle_cosines_out,
    )
    disc = quant_mt_zap.compute_cumulative_quantity(
        disc_compute, disc_select,
        endf_dict, zap, energies_in, energies_out, angle_cosines_out,
    )
    if cont is None:
        return disc
    if disc is None:
        return cont
    return cont + disc


def _normalize_broadening(broadening):
    """Translate a user broadening spec into (kernel, width) for the
    low-level folders. None propagates as (None, None)."""
    if broadening is None:
        return None, None
    if isinstance(broadening, (int, float, np.integer, np.floating)):
        sigma = float(broadening)
        if sigma <= 0:
            raise ValueError("broadening sigma must be positive")
        norm = 1.0 / (sigma * np.sqrt(2 * np.pi))

        def gaussian_kernel(d):
            return norm * np.exp(-0.5 * (d / sigma) ** 2)
        return gaussian_kernel, sigma
    try:
        kernel, width = broadening
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "broadening must be None, a positive scalar sigma, or a "
            "(kernel_callable, width) tuple"
        ) from exc
    if not callable(kernel):
        raise ValueError("broadening tuple element 0 must be a callable kernel")
    width = float(width)
    if width <= 0:
        raise ValueError("broadening tuple element 1 (width) must be positive")
    return kernel, width
