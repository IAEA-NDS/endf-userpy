def get_declared_zap_lfs(endf_dict):
    """All (ZAP, LFS) pairs declared anywhere in MF8.

    Returns an empty set if MF8 is not present in the file.
    """
    if 8 not in endf_dict:
        return set()
    pairs = set()
    for sec in endf_dict[8].values():
        for subsec in sec['subsection'].values():
            pairs.add((int(subsec['ZAP']), int(subsec['LFS'])))
    return pairs


def get_declared_lfs_for_zap(endf_dict, zap):
    """LFS values declared for a given ZAP in MF8.

    Returns a sorted list, possibly empty.
    """
    if 8 not in endf_dict:
        return []
    zap = int(zap)
    lfs_values = set()
    for sec in endf_dict[8].values():
        for subsec in sec['subsection'].values():
            if int(subsec['ZAP']) == zap:
                lfs_values.add(int(subsec['LFS']))
    return sorted(lfs_values)


def is_zap_lfs_declared(endf_dict, zap, lfs):
    """True iff the (ZAP, LFS) pair appears in any MF8 subsection."""
    if 8 not in endf_dict:
        return False
    zap = int(zap)
    lfs = int(lfs)
    for sec in endf_dict[8].values():
        for subsec in sec['subsection'].values():
            if int(subsec['ZAP']) == zap and int(subsec['LFS']) == lfs:
                return True
    return False


def get_mts_producing(endf_dict, zap, lfs):
    """MTs whose MF8 entries route to the given (ZAP, LFS) residual.

    Returns a sorted list, possibly empty.
    """
    if 8 not in endf_dict:
        return []
    zap = int(zap)
    lfs = int(lfs)
    mts = []
    for mt, sec in endf_dict[8].items():
        for subsec in sec['subsection'].values():
            if int(subsec['ZAP']) == zap and int(subsec['LFS']) == lfs:
                mts.append(mt)
                break
    return sorted(mts)


def list_subsecs(endf_dict, mt=None, zap=None, level=None):
    mt_secs = endf_dict[8]
    mtnums = list(mt_secs) if mt is None else [mt]
    subsec_list = []
    for mtnum in mtnums:
        sec = mt_secs[mtnum]
        subsecs = sec['subsection'].values()
        for subsec in subsecs:
            cur_zap = subsec['ZAP']
            if zap is not None and cur_zap != zap:
                continue
            cur_level = subsec['LFS']
            if level is not None and cur_level != level:
                continue
            subsec_list.append((mtnum, cur_zap, cur_level))
    return subsec_list


def find_subsec_nums(endf_dict, mt, zap, level=None):
    sec = endf_dict[8][mt]
    idcs = tuple()
    nums = []
    final_states = []
    for idx, subsec in sec['subsection'].items():
        if subsec['ZAP'] != zap:
            continue
        if level is not None and subsec['LFS'] != level:
            continue
        nums.append(idx)
    return nums


def get_subsecs(endf_dict, mt, zap, level=None):
    subsec_nums = find_subsec_nums(endf_dict, mt, zap, level)
    subsecs = endf_dict[8][mt]['subsection']
    return tuple(subsecs[idx] for idx in subsec_nums)


def get_mf_switch(endf_dict, mt, zap, level=None):
    subsecs = get_subsecs(endf_dict, mt, zap, level)
    if len(subsecs) == 0:
        levelstr = f', level={level}' if level is not None else ''
        raise IndexError(
            f'No subsection associated with MF=8, MT={mt}, ZAP={zap}{levelstr}'
        )
    if len(subsecs) > 1:
        levelstr = f', level={level}' if level is not None else ''
        raise IndexError(
            f'Multiple subsections associated with MF=8, MT={mt}, ZAP={zap}{levelstr}'
        )
    return subsecs[0]['LMF']


def get_mf6_subsec_position_for(endf_dict, mt, zap, lfs):
    """Position of the MF6 subsection matching (mt, zap, lfs).

    For LMF=6 (data routed to MF6) the convention is that MF6
    subsections appear in the same order as the corresponding MF8
    entries for the same (mt, zap). This function walks MF8/mt
    subsections in order, counts those with matching ZAP, and returns
    the 0-based position whose LFS matches the request. The caller
    can then index into MF6 subsections filtered by the same ZAP.
    """
    mf8sec = endf_dict[8][mt]['subsection']
    pos = 0
    for sub in mf8sec.values():
        if sub['ZAP'] != zap:
            continue
        if sub['LFS'] == lfs:
            return pos
        pos += 1
    raise IndexError(
        f'MF8/MT={mt} has no subsection with ZAP={zap} and LFS={lfs}'
    )
