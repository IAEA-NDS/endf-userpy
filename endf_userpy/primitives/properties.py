from .physical_constants import (
    PARTICLE_MASSES_AMU,
    PARTICLE_ZAP,
    get_particle_mass,
)
from .reactions import (
    get_ejectiles,
    get_raw_reaction_string_for_mt,
)


def get_ZA(endf_dict):
    return endf_dict[1][451]['ZA']


def get_ZAI(endf_dict):
    proj = get_projectile(endf_dict)
    zai = PARTICLE_ZAP[proj]
    return zai


def get_ELIS(endf_dict):
    return endf_dict[1][451]['ELIS']


def get_QM(endf_dict, mt):
    return endf_dict[3][mt]['QM']


def get_QI(endf_dict, mt):
    return endf_dict[3][mt]['QI']


def get_reaction_qvalue(endf_dict, mt):
    return get_QI(endf_dict, mt)


def get_LR(endf_dict, mt):
    return endf_dict[3][mt]['LR']


def get_AWR(endf_dict):
    return endf_dict[1][451]['AWR']


def get_AWI(endf_dict):
    return endf_dict[1][451]['AWI']


def get_ejectile(endf_dict, mt):
    projectile = get_projectile(endf_dict)
    if mt in (18, 19, 20, 21, 38):
        ejectile = 'n'
    else:
        ejectiles = get_ejectiles(projectile, mt)
        if ejectiles is None:
            raise ValueError(f'AWP cannot be determined for MT={mt}.')
        # Return the primary ejectile: for single-ejectile MTs there is
        # only one, and for multi-ejectile MTs whose reaction string
        # lists a neutron first (like (n,n'X)) that neutron is the one
        # carrying MF4/MF6 angular data. Multi-ejectile MTs whose first
        # ejectile is not a neutron (e.g. MT 112 = (n,p a), MT 115..117)
        # do not have a well-defined single "the ejectile"; refuse
        # rather than silently guessing. Callers who want a boolean
        # "does mt emit zap" should use is_zap_consistent, which
        # handles that case; callers who need a unique ZAP should not
        # be calling this for multi-ejectile MTs anyway.
        if len(ejectiles) > 1 and ejectiles[0][1] != 'n':
            ejectile_names = [e[1] for e in ejectiles]
            raise ValueError(
                f'MT={mt} has multiple ejectiles {ejectile_names} and none '
                f'is a neutron; no unique "the ejectile" to return.'
            )
        ejectile = ejectiles[0][1]
    return ejectile


def get_AWP(endf_dict, mt):
    ejectile = get_ejectile(endf_dict, mt)
    awp = PARTICLE_MASSES_AMU[ejectile] / PARTICLE_MASSES_AMU['n']
    return awp


def get_ZAP(endf_dict, mt):
    ejectile = get_ejectile(endf_dict, mt)
    zap = PARTICLE_ZAP[ejectile]
    return zap


def is_zap_consistent(endf_dict, mt, zap):
    """Whether an MT could produce a particle with ZAP=zap.

    Handles multi-ejectile MTs correctly by consulting the reaction-
    string ejectile table rather than relying on get_ZAP's single-
    ejectile answer. Gamma (zap=0) is always accepted: gammas
    accompany many reactions and are typically encoded in MF12/14
    rather than in the primary MF6 subsections that get_ejectile
    reasons about.
    """
    from . import reactions as _reactions
    if zap == PARTICLE_ZAP['g']:
        return True
    try:
        projectile = get_projectile(endf_dict)
    except (KeyError, ValueError):
        return True
    contains = _reactions.contains_zap(projectile, mt, zap)
    if contains is None:
        # Unknown reaction table for this MT: be permissive so the
        # caller can still attempt the reconstruction and fail with
        # a domain-specific error if the data really is missing.
        return True
    return contains


def get_projectile(endf_dict):
    sec = endf_dict[1][451]
    nsub = sec['NSUB']
    part_dict = {
        0: 'g', 1: 'g', 3: 'g',
        4: None, 5: None, 6: None,
        10: 'n', 11: 'n', 12: 'n',
        #  (113, 11, 3): 'e',
        10010: 'p', 10011: 'p',
        10020: 'd', 10030: 't',
        20030: 'h', 20040: 'a',
    }
    projectile = part_dict[nsub]
    return projectile


def get_projectile_mass(endf_dict):
    projectile = get_projectile(endf_dict)
    mass = get_particle_mass(projectile)
    return mass


def get_target_mass(endf_dict):
    awr = get_AWR(endf_dict)
    mass_neutron = get_particle_mass('n')
    return awr * mass_neutron


def get_reaction_string_for_mt(endf_dict, mt):
    proj = get_projectile(endf_dict)
    r = get_raw_reaction_string_for_mt(mt)
    r = r.replace('(z,', f'({proj},')
    r = r.replace('(y,', f'({proj},')
    r = r.replace(',z', f',{proj}')
    return r


def has_mf4_mt(endf_dict, mt):
    return 4 in endf_dict and mt in endf_dict[4]


def has_mf5_mt(endf_dict, mt):
    return 5 in endf_dict and mt in endf_dict[5]


def has_mf6_mt(endf_dict, mt):
    return 6 in endf_dict and mt in endf_dict[6]


def has_mf12_mt(endf_dict, mt):
    return 12 in endf_dict and mt in endf_dict[12]


def has_mf13_mt(endf_dict, mt):
    return 13 in endf_dict and mt in endf_dict[13]


def has_mf14_mt(endf_dict, mt):
    return 14 in endf_dict and mt in endf_dict[14]


def has_mf15_mt(endf_dict, mt):
    return 15 in endf_dict and mt in endf_dict[15]


def has_mf2_fission_widths(endf_dict):
    """True iff any MF2 resonance range carries non-zero fission
    widths (``GF`` for MLBW, ``GFA`` / ``GFB`` for Reich-Moore,
    per-channel ``GAM`` for RML).

    Walks parsed-dict metadata only; does no reconstruction. Used by
    the chance-breakdown-fission guard in the user-facing XS entry
    points (issue #311): if the file carries total-fission widths in
    MF2 and the user asks for a chance-breakdown MT (19/20/21/38),
    composing the widths with the chance MT would overstate the
    resonance contribution.
    """
    import numpy as np
    if 2 not in endf_dict or 151 not in endf_dict[2]:
        return False
    for iso in endf_dict[2][151].get('isotope', {}).values():
        for rng in iso.get('range', {}).values():
            lru = int(rng.get('LRU', 0))
            lrf = int(rng.get('LRF', 0))
            if lru != 1:
                continue
            # ``endf_parserpy`` names this ``l_group`` (>=0.17) or
            # ``spingroup`` (older); same content.
            d_grp = rng.get('l_group') or rng.get('spingroup') or {}
            for d_l in d_grp.values():
                # MLBW / RM share GF / GFA / GFB columns.
                for key in ('GF', 'GFA', 'GFB'):
                    column = d_l.get(key, None)
                    if column is None:
                        continue
                    values = (
                        column.values() if isinstance(column, dict)
                        else column
                    )
                    for v in values:
                        if abs(float(v)) > 0.0:
                            return True
                # LRF=7 RML stores per-channel widths under GAM.
                # Pragmatic shortcut: treat any non-zero per-channel
                # GAM as a potential fission signal on an RML file;
                # a false positive just makes the top-level guard
                # raise a shade more aggressively.
                if lrf == 7:
                    gam = d_l.get('GAM', None)
                    if gam is None:
                        continue
                    try:
                        values = (
                            gam.values() if isinstance(gam, dict)
                            else np.asarray(gam, dtype=float).ravel()
                        )
                    except Exception:
                        continue
                    for v in values:
                        if abs(float(v)) > 0.0:
                            return True
    return False
