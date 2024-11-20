from .properties import get_ZAP


def is_dist2d_law(law):
    return law in (1, 6, 7)


def contains_subsec_dist2d(endf_dict, mt, subsec_num):
    sec = endf_dict[6][mt]
    subsec = sec['subsection'][subsec_num]
    law = subsec['LAW']
    return is_dist2d_law(law)


def get_zap_with_check(endf_dict, mt, zap):
    if zap is None:
        zap = get_ZAP(endf_dict, mt)
    elif not is_zap_consistent(endf_dict, mt, zap):
        raise ValueError(f'provided ZAP={zap} not consistent with MT={mt}')


def check_mf6_exists(endf_dict):
    if 6 not in endf_dict:
        raise IndexError(
            f'No information on product-angle distributions found '
            f'(MF=6 section missing)'
        )


def check_mt_exists_in_mf6(endf_dict, mt):
    if mt not in endf_dict[6]:
        raise IndexError(
            f'No information on product-angle distributions found for MT={mt} '
            f'(MF6/MT{mt} missing)'
        )


def find_subsec_num(endf_dict, mt, zap): 
    sec = endf_dict[6][mt]
    for idx, subsec in sec['subsection'].items():
        if subsec['ZAP'] == zap:
            return idx
    raise IndexError(
        f'subsection with ZAP={zap} not found in MF6/MT{mt}'
    )


def get_zaps_for_mt(endf_dict, mt, dist2d_only=False):
    sec = endf_dict[6][mt]
    subsecs = sec['subsection']
    zaps = []
    for idx, subsec in subsecs.items():
        law = subsec['LAW']
        zap = subsec['ZAP']
        if dist2d_only and not _is_dist2d_law(law):
            continue
        zaps.append(zap)
    return zaps
