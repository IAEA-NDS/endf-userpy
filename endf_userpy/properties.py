from .physical_constants import PARTICLE_MASSES_AMU
from .reactions import (
    is_binary_reaction,
    get_ejectiles,
)


def get_ZA(endf_dict):
    return endf_dict[1][451]['ZA']


def get_ZAI(endf_dict):
    proj = get_projectile(endf_dict)
    zai = {
        'g': 0*1000. + 0,
        'n': 0*1000. + 1,
        'p': 1*1000. + 1,
        'd': 1*1000. + 2,
        't': 1*1000. + 3,
        'h': 2*1000. + 3,
        'a': 2*1000. + 4,
    }[proj]
    return endf_dict[1][451]['ZAI']


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


def get_AWP(endf_dict, mt):
    projectile = get_projectile(endf_dict)
    # fission reaction
    if mt in (18, 19, 20, 21, 38):
        ejectile = 'n'
    else:
        ejectiles = get_ejectiles(projectile, mt)
        if ejectiles is None:
            raise ValueError(f'AWP cannot be determined for MT={mt}.')
        assert len(ejectiles) == 1 or ejectiles[0][1] == 'n'
        ejectile = ejectiles[0][1]
    awp = PARTICLE_MASSES_AMU[ejectile] / PARTICLE_MASSES_AMU[projectile]
    return awp


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
