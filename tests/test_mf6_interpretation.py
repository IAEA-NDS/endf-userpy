from pathlib import Path
import os
import pytest
import numpy as np
from endf_parserpy import EndfParserCpp
from endf_userpy.mf6_interpretation import (
    compute_dist2d_from_subsec,
    compute_dist1d_from_subsec,
)
from endf_userpy.helpers import deg2rad
from endf_parserpy.utils.user_tools import show_content

import subprocess
import re
import shutil
import tempfile
from endf_parserpy.utils.user_tools import locate, get_endf_values
from endf_parserpy import EndfPath


def parse_fortran_test_output(cont, mt, subsec_num):
    pat1 = r' *imt= *(\d+) *MT= *(\d+)'
    pat2 = r' *Particle *(\d+)'
    pat3 = r'\*\*\*\*\*\*\*'
    pat4 = r' *(-?\d\.[^ ]+)' * 7
    rex1 = re.compile(pat1)
    rex2 = re.compile(pat2)
    rex3 = re.compile(pat3)
    rex4 = re.compile(pat4)
    # locate MT section
    found_idx1 = None
    for idx, line in enumerate(cont):
        m1 = rex1.match(line)
        if m1:
            curmt = int(m1.group(2))
            if curmt != mt:
                continue
            found_idx1 = idx
            break
    if found_idx1 is None:
        return None
    # locate subsection
    found_idx2 = None
    for idx, line in enumerate(cont[found_idx1+3:], start=found_idx1+3):
        if rex1.match(line):
            break
        m2 = rex2.match(line)
        if not m2:
            continue
        cur_subsec_num = int(m2.group(1))
        if cur_subsec_num == subsec_num:
            found_idx2 = idx
            break
    if found_idx2 is None:
        return None
    colnames =  ('ei', 'ep', 'u', 'tp', 'w', 'f6dis', 'f6con') 
    values = list()
    for line in cont[found_idx2+3:]:
        if rex3.match(line):
            break
        m = rex4.match(line) 
        if m:
            curvals = {n: float(v) for n, v in zip(colnames, m.groups())}
            values.append(curvals)
    # change schema
    res = {}
    for col in colnames: 
        res[col] = [x[col] for x in values]
    return res


def find_subsections_by_law(endf_dict, law):
    mf6 = endf_dict[6] 
    locs = locate(mf6, 'LAW')
    vals = get_endf_values(mf6, locs)
    locs = [l for l, v in zip(locs, vals) if v == law]
    mt_ss = [(x[0], x[2]) for x in locs]
    return mt_ss


def call_fortran_test(fortran_test_exe, endf_file, include=None):
    parser = EndfParserCpp(ignore_missing_tpid=True)
    endf_dict = parser.parsefile(endf_file, include=include)
    mat = endf_dict[1][451]['MAT']
    with tempfile.TemporaryDirectory() as tmpdirname:
        shutil.copy2(fortran_test_exe, os.path.join(tmpdirname, 'runtest'))
        shutil.copyfile(endf_file, os.path.join(tmpdirname, 'endffile'))
        orig_cwd = os.getcwd()
        os.chdir(tmpdirname)
        test_inp = '\n'.join([
            "endffile",
            "output",
            str(mat),
            ""
        ])
        result = subprocess.run(['./runtest'], input=test_inp, text=True)
        with open('output', 'r') as f:
            cont = f.readlines()
        os.chdir(orig_cwd)
    return cont, endf_dict


def test_dist2d_law1_python_interface(endf_file):
    cont, endf_dict = call_fortran_test('../tests_fortran/test_mf6', endf_file)
    if 6 not in endf_dict:
        return
    mt_ss = find_subsections_by_law(endf_dict, 1)
    for mt, subsec_num in mt_ss:
        ref_res = parse_fortran_test_output(cont, mt, subsec_num)
        # calculate using python interface
        energies_in = np.unique(ref_res['ei'])
        energies_out = np.unique(ref_res['ep'])
        mu_out = np.unique(ref_res['u'])
        cont_arr = compute_dist2d_from_subsec(
            endf_dict, mt, subsec_num, energies_in, energies_out, mu_out
        )
        # retrieve corresponding values from fortran test output
        eis = np.array(ref_res['ei'])
        eps = np.array(ref_res['ep'])
        us = np.array(ref_res['u'])
        f6cons = np.array(ref_res['f6con'])
        for ei, ep, u, f6con in zip(eis, eps, us, f6cons):
            idx1 = np.where(ei == energies_in)[0]
            assert len(idx1) == 1
            idx1 = idx1[0]
            idx2 = np.where(ep == energies_out)[0]
            assert len(idx2) == 1
            idx2 = idx2[0]
            idx3 = np.where(u == mu_out)[0]
            assert len(idx3) == 1
            idx3 = idx3[0]
            cur_res = cont_arr[idx1, idx2, idx3]
            if not np.isclose(cur_res, f6con, rtol=1e-5, atol=1e-15):
                print(f'mt: {mt} subsec: {subsec_num} ei: {ei}, ep: {ep}, u: {u}, f6con: {f6con} vs {cur_res}')
                # return


def test_dist1d_law2_interface():
    parser = EndfParserCpp()
    endf_file = "data/n-001_H_001.endf"
    endf_dict = parser.parsefile(endf_file) 
    Einc = [50000, 70000]  
    mu = np.cos(deg2rad([30, 50, 70]))
    cont_arr = compute_dist1d_from_subsec(
        endf_dict, 102, 1, Einc, mu
    )


def test_dist2d_law6_interface():
    parser = EndfParserCpp()
    endf_file = "data/n-001_H_002.endf"
    endf_dict = parser.parsefile(endf_file) 
    Einc = [50000, 70000]  
    mu = np.cos(deg2rad([30, 50, 70]))
    Eout = np.linspace(10000, 70000, 5) 
    cont_arr = compute_dist2d_from_subsec(
        endf_dict, 16, 1, Einc, Eout, mu
    )


def test_dist2d_law7_interface():
    parser = EndfParserCpp()
    endf_file = "data/n-004_Be_009.endf"
    endf_dict = parser.parsefile(endf_file)
    Einc = [1.8e6, 2e6]
    mu = np.cos(deg2rad([30, 50, 70]))
    Eout = np.linspace(10000, 70000, 5)
    compute_dist2d_from_subsec(
        endf_dict, 16, 1, Einc, Eout, mu
    )
