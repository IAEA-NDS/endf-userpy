import resource
from pathlib import Path

import pytest


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
    parser.addoption(
        "--memtrack", action="store_true", default=False,
        help="Log peak-RSS growth per test and print the top offenders "
             "at session end. Uses resource.getrusage(RUSAGE_SELF).ru_maxrss, "
             "which only grows monotonically; a nonzero 'delta' identifies "
             "the test during which peak RSS advanced.",
    )


# --- Memory tracking (opt-in via --memtrack) ---

_mem_records: list[tuple[str, int, int]] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if not item.config.getoption("--memtrack"):
        yield
        return
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    yield
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _mem_records.append((item.nodeid, before, after))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("--memtrack") or not _mem_records:
        return
    # Report top 20 tests by RSS growth (delta > 0 means peak advanced
    # during this test). Followed by the top 20 in absolute post-test
    # peak so the last-known-bad-neighbours are also visible.
    rows = [
        (nodeid, before, after, after - before)
        for nodeid, before, after in _mem_records
    ]
    by_delta = sorted(rows, key=lambda r: r[3], reverse=True)[:20]
    by_peak = sorted(rows, key=lambda r: r[2], reverse=True)[:20]

    def _fmt(kb):
        return f"{kb / 1024:8.1f} MB"

    tr = terminalreporter
    tr.write_sep("=", "peak-RSS growth per test (top 20 by delta)")
    for nodeid, before, after, delta in by_delta:
        tr.write_line(
            f"  +{_fmt(delta)}  peak-after {_fmt(after)}  {nodeid}"
        )
    tr.write_sep("=", "absolute peak RSS after test (top 20)")
    for nodeid, before, after, delta in by_peak:
        tr.write_line(f"  peak {_fmt(after)}  (+{_fmt(delta)})  {nodeid}")


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
