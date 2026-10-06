#!/usr/bin/env bash
#
# Fetch MF6 LAW=5 test files from the ENDF/B-VIII.0 incident-proton
# sublibrary at BNL, verify SHA-256 of the files we need, and leave
# the extracted .endf files in this directory. Idempotent: skips
# the download when every target file already exists with the
# expected checksum.
#
# Usage:
#     bash tests/data_law5_adhoc/fetch.sh
#
# Requires: bash, curl, unzip, sha256sum.
#
# The target files are the LAW=5 LTP=1 LIDP=0 proton evaluations
# used by tests/test_mf6_law5_charged_particle.py:
#
#   p-002_He_003.endf  (p+He-3, small, LTP=1 LIDP=0, NE=42)
#   p-005_B_010.endf   (p+B-10, LTP=1 LIDP=0, NE=68)

set -euo pipefail

cd "$(dirname "$0")"

# name-in-repo      sha256
files=(
  "p-002_He_003.endf  dd68902b7acc518799581ed992a0271291d77f10a0713523a2619f6147d7d115"
  "p-005_B_010.endf   6109e7be3daf4ef88911b78d8fe1a18921cdbf6eeeb5e2f09e7692da65cbba45"
  "p-006_C_012.endf   771f7c5f7f3336a05a05463fe711593284af39c8c93f3bdf4b3a1e76640302a0"
)

zip_url="https://www.nndc.bnl.gov/endf-b8.0/zips/ENDF-B-VIII.0_protons.zip"
zip_sha="27bcafb89cf0444c53c6b9f3dd17618c62e2e6b3694f70d31c4f502399f103f7"
zip_subdir="ENDF-B-VIII.0_protons"

check_sum() {
  local expected="$1"
  local file="$2"
  [ -f "$file" ] || return 1
  echo "$expected  $file" | sha256sum --check --status
}

need_fetch=0
for line in "${files[@]}"; do
  read -r name expected <<<"$line"
  if ! check_sum "$expected" "$name"; then
    need_fetch=1
  fi
done

if [ "$need_fetch" -eq 0 ]; then
  for line in "${files[@]}"; do
    read -r name _expected <<<"$line"
    echo "OK    $name"
  done
  echo
  echo "All files present in $(pwd)."
  exit 0
fi

echo "FETCH ENDF-B-VIII.0_protons.zip from BNL"
tmpzip=$(mktemp --suffix=.zip)
tmpdir=$(mktemp -d)
trap 'rm -f "$tmpzip"; rm -rf "$tmpdir"' EXIT

curl -fsSL "$zip_url" -o "$tmpzip"
echo "$zip_sha  $tmpzip" | sha256sum --check --status

unzip -oq "$tmpzip" -d "$tmpdir"

for line in "${files[@]}"; do
  read -r name expected <<<"$line"
  src="$tmpdir/$zip_subdir/$name"
  if [ ! -f "$src" ]; then
    echo "ERROR $name: not found inside archive"
    exit 1
  fi
  cp "$src" "./$name"
  if ! check_sum "$expected" "$name"; then
    actual=$(sha256sum "$name" | awk '{print $1}')
    echo "ERROR $name: sha256 mismatch"
    echo "  expected $expected"
    echo "  got      $actual"
    exit 1
  fi
  echo "OK    $name"
done

rm -f "$tmpzip"; rm -rf "$tmpdir"
trap - EXIT

echo
echo "All files present in $(pwd)."
