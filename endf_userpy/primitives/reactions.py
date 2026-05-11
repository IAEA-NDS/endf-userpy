import types
from collections.abc import Sequence
from .physical_constants import (
    get_particle_for_zap,
    get_zap_for_particle,
)



def _rng(start, stop):
    return tuple(range(start, stop+1))


def _create_tuple(*args):
    res = []
    for arg in args:
        if isinstance(arg, types.GeneratorType):
            res.extend(arg)
        else:
            res.append(arg)
    return tuple(res)


PARTICLES = ('g', 'n', 'p', 'd', 't', 'h', 'a')

# variables in tuples: MT, projectile, ejectile string
# z is a placeholder for
# n(eutron), p(roton), d(euteron), t(riton), h(elium-3), a(lpha)
# y can contain particles of z except the one in the exit channel 
REACTIONS = _create_tuple(
    (1, 'n', 'total'),
    (2, 'z', 'z_0'),
    (3, 'z', 'nonelas'),
    (4, 'z', 'n'),
    (5, 'z', 'anything'),
    (10, 'z', 'contin'),
    (11, 'z', '2nd'),
    (16, 'z', '2n'),
    (17, 'z', '3n'),
    (18, 'z', 'fission'),
    (19, 'n', 'f'),
    (20, 'n', 'nf'),  # TODO: Can this appear in MF4?
    (21, 'n', '2nf'),
    (22, 'z', 'na'),
    (23, 'n', 'n3a'),
    (24, 'z', '2na'),
    (25, 'z', '3na'),
    (27, 'z', 'abs'),
    (28, 'z', 'np'),
    (29, 'z', 'n2a'),
    (30, 'z', '2n2a'),
    (32, 'z', 'nd'),
    (33, 'z', 'nt'),
    (34, 'z', 'nh'),
    (35, 'z', 'nd2a'),
    (36, 'z', 'nt2a'),
    (37, 'z', '4n'),
    (38, 'n', '3nf'),
    (41, 'z', '2np'),
    (42, 'z', '3np'),
    (44, 'z', 'n2p'),
    (45, 'z', 'npa'),
    (50, 'y', 'n_0'), 
    ((mt, 'z', f'n_{mt-50}') for mt in range(51, 91)),
    (91, 'z', 'n_c'),
    (101, 'n', 'disap'),
    (102, 'z', 'g'),
    (103, 'z', 'p'),
    (104, 'z', 'd'),
    (105, 'z', 't'),
    (106, 'z', 'h'),
    (107, 'z', 'a'),
    (108, 'z', '2a'),
    (109, 'z', '3a'),
    (111, 'z', '2p'),
    (112, 'z', 'pa'),
    (113, 'z', 't2a'),
    (114, 'z', 'd2a'),
    (115, 'z', 'pd'),
    (116, 'z', 'pt'),
    (117, 'z', 'da'),
    (151, 'n', 'RES'),
    (152, 'z', '5n'),
    (153, 'z', '6n'),
    (154, 'z', '2nt'),
    (155, 'z', 'ta'),
    (156, 'z', '4np'),
    (157, 'z', '3nd'),
    (158, 'z', 'nda'),
    (159, 'z', '2npa'),
    (160, 'z', '7n'),
    (161, 'z', '8n'),
    (162, 'z', '5np'),
    (163, 'z', '6np'),
    (164, 'z', '7np'),
    (165, 'z', '4na'),
    (166, 'z', '5na'),
    (167, 'z', '6na'),
    (168, 'z', '7na'),
    (169, 'z', '4nd'),
    (170, 'z', '5nd'),
    (171, 'z', '6nd'),
    (172, 'z', '3nt'),
    (173, 'z', '4nt'),
    (174, 'z', '5nt'),
    (175, 'z', '6nt'),
    (176, 'z', '2nh'),
    (177, 'z', '3nh'),
    (178, 'z', '4nh'),
    (179, 'z', '3n2p'),
    (180, 'z', '3n2a'),
    (181, 'z', '3npa'),
    (182, 'z', 'dt'),
    (183, 'z', 'npd'),
    (184, 'z', 'npt'),
    (185, 'z', 'ndt'),
    (186, 'z', 'nph'),
    (187, 'z', 'ndh'),
    (188, 'z', 'nth'),
    (189, 'z', 'nta'),
    (190, 'z', '2n2p'),
    (191, 'z', 'ph'),
    (192, 'z', 'dh'),
    (193, 'z', 'ha'),
    (194, 'z', '4n2p'),
    (195, 'z', '4n2a'),
    (196, 'z', '4npa'),
    (197, 'z', '3p'),
    (198, 'z', 'n3p'),
    (199, 'z', '3n2pa'),
    (200, 'z', '5n2p'),
    (201, 'z', 'Xn'),  # TODO: Can this appear in MF4?
    (202, 'z', 'Xg'),
    (203, 'z', 'Xp'),
    (204, 'z', 'Xd'),
    (205, 'z', 'Xt'),
    (206, 'z', 'Xh'),
    (207, 'z', 'Xa'),
    (600, 'y', 'p_0'),
    ((mt, 'z', f'p_{mt-600}') for mt in range(601, 649)),
    (649, 'z', 'p_c'),
    ((mt, 'z', f'd_{mt-650}') for mt in range(650, 699)),
    (699, 'z', 'd_c'),
    ((mt, 'z', f't_{mt-700}') for mt in range(700, 749)),
    (749, 'z', 't_c'),
    ((mt, 'z', f'h_{mt-750}') for mt in range(750, 799)),
    (799, 'z', 'h_c'),
    ((mt, 'z', f'a_{mt-800}') for mt in range(800, 849)),
    (849, 'z', 'a_c'),
    (891, 'z', '2n_c')
)


REACTION_DICT = {t[0]: (t[1], t[2]) for t in REACTIONS}  
INV_REACTION_DICT = {v: k for k, v in REACTION_DICT.items()}


SUM_RULES = {
      1: (2, 3),
      3: ((4, 5, 11, 16, 17) + _rng(22, 37) + (41, 42, 44, 45)
          + _rng(152, 154) + _rng(156, 181) + _rng(183, 190)
          + _rng(194, 196) + _rng(198, 200)),
      4: _rng(50, 91),
     16: _rng(875, 891),
     18: (19, 20, 21, 38),
     27: (18, 101),
    101: _rng(102, 117) + (155, 182) + _rng(191, 193) + (197,),
    103: _rng(600, 649),
    104: _rng(650, 699),
    105: _rng(700, 749),
    106: _rng(750, 799),
    107: _rng(800, 849),
    516: (515, 517),
}

SUM_RULE_MAP = {mt: k for k, v in SUM_RULES.items() for mt in v}


def is_known_reaction_mt(mt):
    """True iff `mt` is a reaction MT recognised by this package.

    HEATR-injected heating numbers (MT 301..450), MF1/MF8-only book-
    keeping MTs, and other non-reaction MT values that may appear in
    a merged ENDF file return False.
    """
    return mt in REACTION_DICT


X_PRODUCTION_SUM_TO_PARTICLE = {
    201: 'n', 202: 'g', 203: 'p', 204: 'd',
    205: 't', 206: 'h', 207: 'a',
}
PARTICLE_TO_X_PRODUCTION_SUM = {
    v: k for k, v in X_PRODUCTION_SUM_TO_PARTICLE.items()
}
X_PARTICLE_PRODUCTION_MTS = frozenset(X_PRODUCTION_SUM_TO_PARTICLE)


def is_x_particle_production_mt(mt):
    """True for MT 201..207, the X-particle production cross sections.

    These are sums over every channel emitting the labelled particle
    weighted by the (energy-dependent) particle multiplicity. The
    multiplicity is folded into the MF3 cross section directly, so
    code paths that ordinarily multiply by a per-channel yield should
    treat MT 201..207 as if the yield were 1.
    """
    return mt in X_PARTICLE_PRODUCTION_MTS


def get_x_production_sum_mt(particle):
    """MT 201..207 number that aggregates production of `particle`."""
    return PARTICLE_TO_X_PRODUCTION_SUM.get(particle)


def get_x_production_partials(proj, sum_mt, available_mts):
    """Partial reaction MTs that contribute to MT `sum_mt`.

    Filters `available_mts` (typically `endf_dict[3].keys()`) down to
    those reaction channels that emit at least one of the labelled
    particle for `proj`. Used to decide whether the partial-channel
    path can answer a particle-production query, in which case the
    aggregate MT 201..207 entry is excluded to avoid double counting.
    """
    if sum_mt not in X_PRODUCTION_SUM_TO_PARTICLE:
        return []
    particle = X_PRODUCTION_SUM_TO_PARTICLE[sum_mt]
    zap = get_zap_for_particle(particle)
    out = []
    for mt in available_mts:
        if is_x_particle_production_mt(mt):
            continue
        if not is_known_reaction_mt(mt):
            continue
        mult = get_multiplicity_for_zap(proj, mt, zap)
        if mult is not None and mult > 0:
            out.append(mt)
    return out


def get_ejectiles(proj, mt):
    """Get ejectiles and their multiplicities.

    Returns None for MTs that are not in the reaction table (e.g.
    HEATR heating numbers MT 301..450 in a merged file).
    """
    result = []
    if mt not in REACTION_DICT:
        return None
    ejectiles = REACTION_DICT[mt][1]
    i = 0
    while i < len(ejectiles):
        c = ejectiles[i]
        multiplicity = 1
        if c.isdigit():
            multiplicity = int(c)
            i += 1
            if i == len(ejectiles):
                return None
        particle = ejectiles[i]
        particle = particle if particle != 'z' else proj 
        if particle not in PARTICLES:
            return None
        result.append((multiplicity, particle))
        # level definitions are expected to come last
        if ejectiles[i+1:i+2] == '_':
            level = ejectiles[i+2:]
            if not level.isdigit() and level != 'c':
                return None
            return tuple(result)
        i += 1
    return tuple(result)


def get_unique_ejectile(proj, mt):
    ejectiles = get_ejectiles(proj, mt)
    if ejectiles is None or len(ejectiles) != 1:
        raise ValueError('Not possible to identiy the unique ejectile')
    return ejectiles[0][1]


def get_multiplicity_for_ejectile(proj, mt, ejectile):
    ejectiles = get_ejectiles(proj, mt)
    if ejectiles is None:
        return None
    for m, part in ejectiles:
        if part == ejectile:
            return m
    return 0


def get_multiplicity_for_zap(proj, mt, zap):
    ejectile = get_particle_for_zap(zap)
    return get_multiplicity_for_ejectile(proj, mt, ejectile)


def contains_zap(proj, mt, zap):
    mult = get_multiplicity_for_zap(proj, mt, zap)
    return mult > 0 if mult is not None else None


def is_discrete_level_scattering(mt):
    if mt not in REACTION_DICT:
        return False
    ejstr = REACTION_DICT[mt][1]
    if '_' not in ejstr:
        return False
    return ejstr.rsplit('_', 1)[-1].isdigit()


def is_continuum_channel(mt):
    if mt not in REACTION_DICT:
        return False
    return REACTION_DICT[mt][1].endswith('_c')


def get_raw_reaction_string_for_mt(mt):
    if mt not in REACTION_DICT:
        raise ValueError(
            f'MT={mt} is not a known ENDF reaction; cannot form a '
            f'reaction string. Use is_known_reaction_mt() to filter.'
        )
    t = REACTION_DICT[mt]
    return f'({t[0]},{t[1]})'


def translate_reaction_string_to_mt(reacstr):
    reacstr = reacstr.lstrip('(').rstrip(')').replace(' ','')
    proj, ejectiles = reacstr.split(',')
    for _ in range(2):
        mt = INV_REACTION_DICT.get(('z', ejectiles), None)
        if mt is not None:
            return mt
        mt = INV_REACTION_DICT.get(('y', ejectiles), None)
        if mt is not None:
            proj_zap = get_zap_for_particle(proj)
            if not contains_zap(proj, mt, proj_zap):
                return mt
        if proj == 'n':
            mt = INV_REACTION_DICT.get(('n', ejectiles), None)
            if mt is not None:
                return mt
        # try again but replace occurrences of z
        # in ejectile string by projectile
        ejectiles = ejectiles.replace(proj, 'z')
    raise ValueError('invalid reaction string')


def is_binary_reaction(proj, mt):
    """Determine if binary reaction"""
    ejectiles = get_ejectiles(proj, mt)
    if ejectiles is None:
        return False
    return len(ejectiles) == 1 and ejectiles[0][0] == 1 


def is_sum_mt(mt):
    return mt in SUM_RULES


def is_in_sum_mt(mt):
    return mt in SUM_RULE_MAP


def any_ancestor_in_mts(mt, mts):
    if not is_in_sum_mt(mt):
        return False
    cur_parent = get_sum_mt_from_part_mt(mt)
    if cur_parent in mts:
        return True
    return any_ancestor_in_mts(cur_parent, mts)


def get_sum_mt_from_part_mt(mt):
    return SUM_RULE_MAP[mt]


def get_part_mts_from_sum_mt(mt):
    return SUM_RULES[mt]


def exist_associated_child_mts(mt, avail_mts):
    if not is_sum_mt(mt):
        return False
    part_mts = get_part_mts_from_sum_mt(mt)
    for part_mt in part_mts:
        if part_mt in avail_mts:
            return True
        if exist_associated_child_mts(part_mt, avail_mts):
            return True
    return False


def is_unique_path_to_residual(proj, mt):
    ejectiles = get_ejectiles(proj, mt)
    # exactly one particle must be in exit channel
    if ejectiles is None or len(ejectiles) > 1:
        return False
    mult, ejectile = ejectiles[0]
    # this particle must be a single light-ion ejectile whose residual
    # ZA is uniquely determined by the projectile, target, and ejectile
    if ejectile not in ('g', 'n', 'p', 'd', 't', 'h', 'a'):
        return False
    # for neutron-induced reactions, there is not an MT number
    # corresponding to n,el + n,inl, so whenever only one neutron is
    # in the exit channel, MT=4 is only a partial component and
    # therefore not the unique path. Photon and charged-particle
    # projectiles do not have this ambiguity, since no elastic MT
    # alongside MT=4 shares the n ejectile.
    if proj == 'n' and ejectile == 'n' and mult == 1:
        return False
    # discrete-level scattering MTs (e.g. MT=51..90, 600..648, ...)
    # share their residual nucleus with their continuum-channel sibling
    # (MT=91, 649, ...) and with each other, so none of them is alone
    # the unique path to that residual.
    if is_discrete_level_scattering(mt) or is_continuum_channel(mt):
        return False
    return True
