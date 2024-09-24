import types
from collections.abc import Sequence


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
    (20, 'n', 'nf'),
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
    (201, 'z', 'Xn'),
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


def get_ejectiles(proj, mt):
    """Get ejectiles and their multiplicities"""
    result = []
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


def is_binary_reaction(proj, mt):
    """Determine if binary reaction"""
    ejectiles = get_ejectiles(proj, mt)
    if ejectiles is None:
        return False
    return len(ejectiles) == 1 and ejectiles[0][0] == 1 
