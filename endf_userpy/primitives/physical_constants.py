PARTICLE_MASSES_AMU = {
    'n': 1.00866491578, 
    'p': 1.00727646688, 
    'd': 2.01355321271, 
    't': 3.015500713,
    'h': 3.01493223469, 
    'a': 4.0015061747, 
}


PARTICLE_ZAP = {
    'g': 1000 * 0.0 + 0.0,
    'n': 1000 * 0.0 + 1.0,
    'p': 1000 * 1.0 + 1.0,
    'd': 1000 * 1.0 + 2.0,
    't': 1000 * 1.0 + 3.0,
    'h': 1000 * 2.0 + 3.0,
    'a': 1000 * 2.0 + 4.0,
}


ZAP_PARTICLE = {v: k for k, v in PARTICLE_ZAP.items()}


def get_zap_for_particle(particle):
    return PARTICLE_ZAP[particle]


def get_particle_for_zap(zap):
    return ZAP_PARTICLE[zap]
