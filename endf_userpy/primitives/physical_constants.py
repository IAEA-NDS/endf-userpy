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


PARTICLE_NAME_SYNONYMS = {
    'g': ('gamma',),
    'n': ('neutron',),
    'p': ('proton',),
    'd': ('deuteron',),
    't': ('triton',),
    'h': ('helion', 'helium-3'),
    'a': ('alpha',),
}


PARTICLE_NAME_MAP = {
    v: k for k, t in PARTICLE_NAME_SYNONYMS.items() for v in t
}
PARTICLE_NAME_MAP.update({k: k for k in PARTICLE_NAME_SYNONYMS})


def is_valid_particle_string(particle):
    return particle.lower() in PARTICLE_NAME_MAP


def get_zap_for_particle(particle):
    part = PARTICLE_NAME_MAP[particle.lower()]
    return PARTICLE_ZAP[part]


def get_particle_for_zap(zap):
    return ZAP_PARTICLE[zap]
