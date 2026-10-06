# MF6 LAW=5 test corpus (fetch on demand)

Fetch-on-demand ENDF files used by `tests/test_mf6_law5_charged_particle.py`
to exercise the charged-particle elastic reconstruction (issue #264).

Populate with:

    bash tests/data_law5_adhoc/fetch.sh

Tests that use these files call `pytest.skip` when the corpus is
absent, so a fresh checkout can run `pytest tests/` without the
download. The script pins the SHA-256 of each committed file so an
upstream edit that silently changes the data is caught.

## Files

| File | Target | AWR | NE | Why |
|---|---|---|---|---|
| `p-002_He_003.endf` | p + He-3 | 2.99 | 42 | Smallest LTP=1 LIDP=0 proton evaluation. Non-trivial NL (NL=4) and well in the Rutherford-dominated regime at low Ein. |
| `p-005_B_010.endf` | p + B-10 | 9.92 | 68 | Second LTP=1 LIDP=0 test target to catch any species-specific regression (heavier target, more panels in the Ein grid). |

Both files are the ENDF/B-VIII.0 incident-proton evaluations pulled
from BNL's download archive (not the IAEA mirror, which sits behind
a Cloudflare challenge that a plain `curl` cannot pass). The fetch
script downloads the sublibrary zip (~13 MB) once and extracts just
the files above.
