#!/usr/bin/env bash
#
# Fetch the five candidate ENDF files listed in README.md into this
# directory. Files are pulled straight from nds.iaea.org, unzipped,
# renamed to the <library>_n_<isotope>.endf convention, and their
# SHA-256 checksums verified against the values below (updated only
# when the upstream evaluators intentionally change).
#
# Idempotent: skips files already present with the expected checksum.
# Runs a single curl + unzip per candidate on first fetch.
#
# Usage:
#     bash tests/data_law1_adhoc/fetch.sh
#
# Requires: bash, curl, unzip, sha256sum.

set -euo pipefail

cd "$(dirname "$0")"

# name-in-repo  upstream-library  upstream-zip-basename  sha256
files=(
  "endfb81_n_Be-9.endf   ENDF-B-VIII.1  n_004-Be-9_0425   6153d5a05dd1bf903b2b3e28d6f03cc09bde85c1c80eb266dd924dbe14f93560"
  "endfb81_n_B-11.endf   ENDF-B-VIII.1  n_005-B-11_0528   9a10957ddcd8924a4c1d7456ed16da89ee749c20d057d78bcd573a9ce009d97f"
  "endfb81_n_Al-27.endf  ENDF-B-VIII.1  n_013-Al-27_1325  f68a0fd25f921c59730555d767212b5f30e68f9693b70134d214c94af71eb3a5"
  "tendl21_n_Fe-56.endf  TENDL-2021     n_026-Fe-56_2631  34fb36fe4f20d3263d22f3011d0af32d277ee5f68a99901b2d4ca70ce846ac18"
  "tendl21_n_U-235.endf  TENDL-2021     n_092-U-235_9228  876a66be00cfd8fb0d65f04f678b92bc03ebf501e2430b420a77ab782ed15dca"
  # Cu-63 from JEFF-4.0: the exact file behind Pablo's four-library
  # 63Cu(n,xg) plot that motivated issue #29. MF12 partial channels
  # for MT 51..79 carry the (n,n_i) de-excitation photons that were
  # silently dropped by the pre-PR#35 gamma production dispatcher.
  "jeff40_n_Cu-63.endf   JEFF-4.0       n_029-Cu-63_2925  7831a5bab6cac44d56d019f775062112cb38741287cca9745d9cb132413ebafb"
  # Cu-63 from JENDL-5: MT 51..90 put the (n,n_i) de-excitation
  # photons in MF6/LAW=1 as pure-discrete ND=1 subsections (rather
  # than MF12+MF14 as every other library). Motivating file for the
  # MF6/LAW=1 gamma-angular support (issue #55): before that fix,
  # the entire JENDL-5 (n,n_i) gamma angular structure was silently
  # dropped from dxs/dmu / DDX because `has_angdist_part` only
  # recognised LAW=2/3/4.
  "jendl5_n_Cu-63.endf   JENDL-5        n_029-Cu-63_2925  cf81b57249cd33cc787c197f21ce560cda4f711b6fd77141710ce90f1fcf6a7f"
)

ua='Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)'

check_sum() {
  local expected="$1"
  local file="$2"
  [ -f "$file" ] || return 1
  echo "$expected  $file" | sha256sum --check --status
}

for line in "${files[@]}"; do
  read -r name library upstream expected <<<"$line"
  if check_sum "$expected" "$name"; then
    echo "OK    $name"
    continue
  fi
  echo "FETCH $name  (${library}/${upstream}.zip)"
  url="https://nds.iaea.org/public/download-endf/${library}/n/${upstream}.zip"
  tmpzip=$(mktemp --suffix=.zip)
  tmpdir=$(mktemp -d)
  trap 'rm -f "$tmpzip"; rm -rf "$tmpdir"' EXIT
  curl -fsSL -A "$ua" "$url" -o "$tmpzip"
  unzip -oq "$tmpzip" -d "$tmpdir"
  extracted=$(find "$tmpdir" -type f -name "${upstream}.dat" | head -n 1)
  if [ -z "$extracted" ]; then
    echo "ERROR $name: expected ${upstream}.dat inside archive"
    exit 1
  fi
  mv "$extracted" "$name"
  rm -f "$tmpzip"; rm -rf "$tmpdir"
  trap - EXIT
  if ! check_sum "$expected" "$name"; then
    actual=$(sha256sum "$name" | awk '{print $1}')
    echo "ERROR $name: sha256 mismatch"
    echo "  expected $expected"
    echo "  got      $actual"
    exit 1
  fi
  echo "OK    $name"
done

echo
echo "All files present in $(pwd)."
