from .physical_constants import (
    PARTICLE_MASSES_AMU,
    PARTICLE_ZAP,
)
from .reactions import (
    is_binary_reaction,
    get_ejectiles,
    get_raw_reaction_string_for_mt,
)


def get_ZA(endf_dict):
    return endf_dict[1][451]['ZA']


def get_ZAI(endf_dict):
    proj = get_projectile(endf_dict)
    zai = PARTICLE_ZAP[proj]
    return zai


def get_QM(endf_dict, mt):
    return endf_dict[3][mt]['QM']


def get_QI(endf_dict, mt):
    return endf_dict[3][mt]['QI']


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
        assert len(ejectiles) == 1 or ejectiles[0][1] == 'n'
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


def get_zap_for_particle(particle):
    return PARTICLE_ZAP[particle]


def is_zap_consistent(endf_dict, mt, zap):
    try:
        zap_mt = get_ZAP(endf_dict, mt)
        return zap_mt == zap
    except:
        return True


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
