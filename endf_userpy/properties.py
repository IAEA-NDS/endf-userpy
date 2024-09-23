from .physical_constants import PARTICLE_MASSES_AMU
from .reactions import (
    is_binary_reaction,
    get_ejectiles,
)


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
    if not is_binary_reaction(projectile, mt):
        raise ValueError('AWP can only be determined for binary reactions')
    ejectiles = get_ejectiles(projectile, mt)
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
