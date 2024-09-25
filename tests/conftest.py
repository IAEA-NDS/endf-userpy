from pathlib import Path


def pytest_addoption(parser):
    parser.addoption("--endfdir", action="store", default="data")
    parser.addoption("--endffile", action="store", default=None)
    parser.addoption("--ignore_zero_mismatch", action="store", default="true")
    parser.addoption("--ignore_number_mismatch", action="store", default="false")
    parser.addoption("--ignore_varspec_mismatch", action="store", default="false")
    parser.addoption("--accept_spaces", action="store", default="true")
    parser.addoption("--ignore_blank_lines", action="store", default="False")
    parser.addoption("--ignore_send_records", action="store", default="False")
    parser.addoption("--ignore_missing_tpid", action="store", default="False")


def pytest_generate_tests(metafunc):
    endf_dir = Path(__file__).parent / metafunc.config.option.endfdir
    if "endf_file" in metafunc.fixturenames:
        file_opt = metafunc.config.option.endffile
        if file_opt is not None:
            endf_files = [endf_dir / file_opt]
        else:
            endf_files = list(endf_dir.glob("*.endf"))
        metafunc.parametrize(
            "endf_file", endf_files, ids=[str(file.name) for file in endf_files]
        )

    parse_opts = (
        "ignore_zero_mismatch",
        "ignore_number_mismatch",
        "ignore_varspec_mismatch",
        "accept_spaces",
        "ignore_blank_lines",
        "ignore_send_records",
        "ignore_missing_tpid",
    )

    opts = metafunc.config.option
    for curopt in parse_opts:
        if curopt in metafunc.fixturenames:
            argval = opts.__dict__[curopt].lower().strip()
            argval = argval == "true"
            metafunc.parametrize(curopt, [argval], scope="module")
