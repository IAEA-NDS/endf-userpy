import re


AMU_TO_EV = 9.3149410242e8  # eV / c^2
AMU_TO_MEV = AMU_TO_EV * 1e-6  # MeV / c^2


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


# Element symbols indexed by atomic number; index 0 is the neutron
# placeholder so ELEMENT_SYMBOLS[Z] gives the symbol for charge Z.
ELEMENT_SYMBOLS = (
    'n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
)
ELEMENT_Z_MAP = {sym.lower(): z for z, sym in enumerate(ELEMENT_SYMBOLS)}


def is_valid_particle_string(particle):
    return particle.lower() in PARTICLE_NAME_MAP


def get_zap_for_particle(particle):
    part = PARTICLE_NAME_MAP[particle.lower()]
    return PARTICLE_ZAP[part]


def get_particle_for_zap(zap):
    return ZAP_PARTICLE[zap]


def get_particle_mass(particle):
    return PARTICLE_MASSES_AMU[particle] * AMU_TO_EV


def get_particle_mass_for_zap(zap):
    part = get_particle_for_zap(zap)
    return get_particle_mass(part)


def _parse_isomer_suffix(text):
    """Strip a g, m, or mN suffix and return (text_without_suffix, lfs).

    `g` -> 0, `m` -> 1, `m<N>` -> N (e.g. `m1` -> 1, `m22` -> 22).
    No suffix returns lfs=None.
    """
    if text.endswith('g'):
        return text[:-1], 0
    match = re.fullmatch(r'(\d+)m(\d*)', text)
    if match is not None:
        body, num = match.groups()
        return body, int(num) if num else 1
    return text, None


def get_za_for_residual_nucleus(residual_nucleus):
    s = residual_nucleus.replace(' ', '')
    parts = s.split('-')
    if len(parts) == 3:
        charge_str, _sym, mass = parts
        z = int(charge_str)
    elif len(parts) == 2:
        sym, mass = parts
        if sym.lower() not in ELEMENT_Z_MAP:
            raise ValueError(
                f'unknown element symbol "{sym}" in residual nucleus '
                f'"{residual_nucleus}"'
            )
        z = ELEMENT_Z_MAP[sym.lower()]
    else:
        raise ValueError(
            f'cannot parse residual nucleus "{residual_nucleus}"; '
            f'expected format "Z-Sym-A" (e.g. "27-Co-60") or "Sym-A" '
            f'(e.g. "Co-60"), optionally with isomer suffix g, m, m1, m2, ...'
        )
    mass, level = _parse_isomer_suffix(mass)
    return z*1000.0 + int(mass), level
