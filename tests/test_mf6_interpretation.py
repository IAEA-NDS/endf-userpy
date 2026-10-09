from pathlib import Path
import os
import sys
import numpy as np
from endf_parserpy import EndfParserCpp
from endf_userpy.mfsec_interpretation.mf6_interpretation_subsecs import (
    compute_dist2d_from_subsec,
    compute_angdist_from_subsec,
)
from endf_userpy.primitives.helpers import deg2rad

import subprocess
import re
import shutil
import tempfile
from endf_parserpy.utils.user_tools import locate, get_endf_values


TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / 'data'
FORTRAN_TESTS_DIR = TESTS_DIR.parent / 'tests_fortran'


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
    """Run the Fortran reference program on ``endf_file`` in a scratch
    directory and return ``(output_lines, endf_dict)``.

    The program runs with ``cwd=`` the scratch directory instead of
    ``os.chdir`` into it: a failure (e.g. the binary cannot start) used
    to leave the whole pytest process inside the deleted temp
    directory, so every later test with a relative corpus path failed
    or skipped. A binary that exists but cannot be loaded (missing
    runtime library, foreign architecture) is skipped like a missing
    one; any other failure raises with the program's stderr.
    """
    import pytest
    parser = EndfParserCpp(ignore_missing_tpid=True)
    endf_dict = parser.parsefile(endf_file, include=include)
    mat = endf_dict[1][451]['MAT']
    with tempfile.TemporaryDirectory() as tmpdirname:
        shutil.copy2(fortran_test_exe, os.path.join(tmpdirname, 'runtest'))
        shutil.copyfile(endf_file, os.path.join(tmpdirname, 'endffile'))
        test_inp = '\n'.join([
            "endffile",
            "output",
            str(mat),
            ""
        ])
        try:
            proc = subprocess.run(
                [os.path.join(tmpdirname, 'runtest')], input=test_inp,
                text=True, cwd=tmpdirname, capture_output=True,
            )
        except OSError as exc:      # exec format error etc.
            pytest.skip(f'{fortran_test_exe} cannot be executed here: {exc}')
        if proc.returncode != 0 and (
                'error while loading shared libraries' in proc.stderr):
            pytest.skip(
                f'{fortran_test_exe} cannot be loaded here: '
                f'{proc.stderr.strip()}'
            )
        output_path = os.path.join(tmpdirname, 'output')
        if proc.returncode != 0 or not os.path.exists(output_path):
            raise RuntimeError(
                f'{fortran_test_exe} failed (exit {proc.returncode}):\n'
                f'{proc.stderr}'
            )
        with open(output_path, 'r') as f:
            cont = f.readlines()
    return cont, endf_dict


def test_dist2d_law1_python_interface(endf_file):
    exefile = 'test_mf6'
    if sys.platform == 'win32':
        exefile += '.exe'
    exe_path = FORTRAN_TESTS_DIR / exefile
    if not exe_path.exists():
        # tests_fortran/ reference binaries are not committed to the
        # repository (issue #95); on a fresh clone the binary is
        # absent and the test would fail with FileNotFoundError from
        # shutil.copy2 inside call_fortran_test. Skip cleanly so this
        # test does not block the CI pytest job (#94). Contributors
        # who want to run the fortran/python equivalence check
        # locally must build tests_fortran/test_mf6 themselves.
        import pytest
        pytest.skip(
            f'{exe_path} not built; run the local fortran build in '
            f'tests_fortran/ to enable this cross-check (issue #95).'
        )
    cont, endf_dict = call_fortran_test(exe_path, endf_file)
    if 6 not in endf_dict:
        return
    mt_ss = find_subsections_by_law(endf_dict, 1)
    mismatches = []
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
                mismatches.append(
                    f'mt={mt} subsec={subsec_num} ei={ei} ep={ep} u={u}: '
                    f'fortran {f6con} vs python {cur_res}'
                )
    # Used to only print mismatches, so the comparison could never fail.
    assert not mismatches, (
        f'{len(mismatches)} LAW=1 points differ from the Fortran '
        f'reference beyond rtol=1e-5, atol=1e-15; first 10:\n'
        + '\n'.join(mismatches[:10])
    )


def test_dist1d_law2_interface():
    parser = EndfParserCpp()
    endf_file = DATA_DIR / "n-001_H_001.endf"
    endf_dict = parser.parsefile(endf_file)
    Einc = np.array([50000, 70000], dtype=float)
    mu = np.cos(deg2rad([30, 50, 70]))
    compute_angdist_from_subsec(
        endf_dict, 102, 1, Einc, mu
    )


def test_dist2d_law6_interface():
    parser = EndfParserCpp()
    endf_file = DATA_DIR / "n-001_H_002.endf"
    endf_dict = parser.parsefile(endf_file)
    Einc = np.array([50000, 70000])
    mu = np.cos(deg2rad([30, 50, 70]))
    Eout = np.linspace(10000, 70000, 5)
    compute_dist2d_from_subsec(
        endf_dict, 16, 1, Einc, Eout, mu
    )


def test_dist2d_law7_interface():
    parser = EndfParserCpp()
    endf_file = DATA_DIR / "n-004_Be_009.endf"
    endf_dict = parser.parsefile(endf_file)
    Einc = np.array([1.8e6, 2e6])
    mu = np.cos(deg2rad([30, 50, 70]))
    Eout = np.linspace(10000, 70000, 5)
    compute_dist2d_from_subsec(
        endf_dict, 16, 1, Einc, Eout, mu
    )


def _fake_exe(tmp_path, stderr, code):
    exe = tmp_path / 'fake_runtest'
    exe.write_text(f'#!/bin/sh\necho "{stderr}" >&2\nexit {code}\n')
    exe.chmod(0o755)
    return exe


def test_call_fortran_test_failure_keeps_cwd(tmp_path):
    """A Fortran reference binary that cannot load is skipped, one that
    crashes raises; in both cases the process cwd is unchanged. The
    old os.chdir into the temp dir leaked on failure and broke every
    later test that opens a relative corpus path."""
    import pytest
    if sys.platform == 'win32':
        pytest.skip('POSIX shell script stands in for the binary')
    endf_file = DATA_DIR / 'n-001_H_001.endf'
    cwd = os.getcwd()
    loader = _fake_exe(
        tmp_path, 'runtest: error while loading shared libraries: '
        'libgfortran.so.5: cannot open shared object file', 127)
    with pytest.raises(pytest.skip.Exception, match='cannot be loaded'):
        call_fortran_test(loader, endf_file)
    assert os.getcwd() == cwd
    crash = _fake_exe(tmp_path, 'Segmentation fault', 139)
    with pytest.raises(RuntimeError, match='Segmentation fault'):
        call_fortran_test(crash, endf_file)
    assert os.getcwd() == cwd
