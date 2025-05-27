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
