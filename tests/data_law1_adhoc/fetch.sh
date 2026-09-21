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
  # Broad public-API stress-test corpus: one representative file
  # each from ENDF/B-VIII.1, JEFF-4.0, TENDL-2021 and JENDL-5 across
  # a wide mass range (H, D, C, N, O, Ni, Au, U, Pu). Used by the
  # ad-hoc stress-test driver that walks every user-facing function
  # in endf_userpy.quantities.
  "endfb81_n_H-1.endf     ENDF-B-VIII.1  n_001-H-1_0125    3442a583d436fe3317a87b7d5518b1e64fae11a73bbd25aa9ea7af2e115c0b52"
  "jeff40_n_H-2.endf      JEFF-4.0       n_001-H-2_0128    8f807ac6a507557a645f13d216672eca872a9120b375e2f70b18598cf290fac3"
  "jendl5_n_C-12.endf     JENDL-5        n_006-C-12_0625   82b4fd9a757513d276badc2981212bd5c0b0669ad576b3313bed68fcd989950f"
  "endfb81_n_N-14.endf    ENDF-B-VIII.1  n_007-N-14_0725   fb15493d2b3425af84cedb3018dc7cf18c399d9bfc9ad574ed7551b2841d6a8f"
  "jeff40_n_O-16.endf     JEFF-4.0       n_008-O-16_0825   099c015375ce869c215a937f47f39b58e709664c55e6ac29fbc3e3c3e9bddf43"
  "endfb81_n_Ni-58.endf   ENDF-B-VIII.1  n_028-Ni-58_2825  6b3616ae6ec985933f4fe393c7158f985c802cf3b03ce281ae364e4a8620f7c9"
  "tendl21_n_Au-197.endf  TENDL-2021     n_079-Au-197_7925 265ef2bd036f2e16cf007473c4d91caa714b190276673b018d8578ba74721896"
  "jendl5_n_U-238.endf    JENDL-5        n_092-U-238_9237  cb987b2c672e2bf3280950d774fc6b316b2efa5e64a28d24f0b05d2be55e5868"
  "endfb81_n_Pu-239.endf  ENDF-B-VIII.1  n_094-Pu-239_9437 e465d4f589b750cd7a6c1ae0e700367446a71ecad0186cb6b2294c56d36efd9a"
  # MF13 cross-library coverage (issue #92 / follow-up to #79).
  # ENDF/B-VIII.1 N-14 already covers the NK=1 and NK>1 branches of
  # mf13_interpretation.compute_total_photon_production_xs, but only
  # from one library. TENDL-2021 N-14 replicates the same seven-MT
  # layout (MT 4 / 28 / 32 / 103 / 104 / 105 / 107) under a
  # different evaluator so a library-specific totals-header quirk
  # would be caught. JENDL-5 N-14 writes MF13 differently -- one MT
  # (MT 3, the nonelastic sum) with NK=1 -- giving orthogonal
  # coverage of the NK=1 branch on a distinct data shape.
  "tendl21_n_N-14.endf   TENDL-2021     n_007-N-14_0725   1f29bf7206a014a0c82272d83d032a80cc115d332cbeefb99e72acd81c03528c"
  "jendl5_n_N-14.endf    JENDL-5        n_007-N-14_0725   1e23b9c02627a58a3cb718b087acbf0a5b0016629a9347eb735d659a2e7cc00f"
  # Nb-93 for MLBW resonance-reconstruction integration tests. Clean
  # mid-mass MLBW range with LRX competitive channels above the
  # inelastic threshold, so it exercises both the plain-MLBW and
  # LRX-competitive branches. Used by the array-agnostic MF2
  # reconstruction and end-to-end composition tests.
  "endfb81_n_Nb-93.endf  ENDF-B-VIII.1  n_041-Nb-93_4125  6ae5b6e7ee35b1d7e9460a9067b986de057ea632a7673a8bfc3125dbd3a222bd"
  # ---- R-M / URR / composition coverage corners (from the NJOY
  # coverage sweep summarised in the branch history). Each file
  # exercises a corner of the reconstruction the pre-existing
  # corpus does not.
  #
  # Pb-208 (ENDF-B-VIII.1): closed-shell + per-L APL (L=0: 0.967;
  # L=1..3: 0.975). Regression pin for the per-L APL fix, and
  # tracks the residual high-E R-M discrepancy driven by the
  # large-Γ_n negative-E resonances (task #45; ~1% ratio bias
  # above ~50 keV that survives the per-L APL fix).
  "endfb81_n_Pb-208.endf  ENDF-B-VIII.1  n_082-Pb-208_8237  c6088040dcfcd68ebcd010faf3b1c98b90701de12beb0d7f2dd3d4e2c3c49688"
  # K-39 (ENDF-B-VIII.1): worst LRF=3 NAPS=0 mismatch found in the
  # sweep (median 17%, max 11.3 barn) driven by the same neg-E
  # extrapolation issue as Pb-208 but on a light nucleus with
  # different resonance density. Regression pin for the eventual
  # fix of task #45.
  "endfb81_n_K-39.endf    ENDF-B-VIII.1  n_019-K-39_1925    c99f5435fbb2d8975188142507edc89cc0ebb67996e7eac1039b1e4c0190b6e9"
  # Rh-103 (ENDF-B-VIII.1): small RRR (6 resonances) plus a big
  # LSSF=0 URR with the full AMU set (AMUN, AMUG, AMUF, AMUX all
  # populated). Exercises the URR reconstruction path without the
  # RRR crowding an actinide file forces.
  "endfb81_n_Rh-103.endf  ENDF-B-VIII.1  n_045-Rh-103_4525  898d98b8945ac1c9debc9a39ceae93c47f1564f27c191e1501b86c42ba4e715b"
  # Au-197 (JEFF-4.0): LSSF=0 URR Case C on a non-fissile standard
  # reference nucleus. Complements the fissile-actinide URR
  # coverage the U-235/U-238 files provide.
  "jeff40_n_Au-197.endf   JEFF-4.0       n_079-Au-197_7925  b342274b21d26862fe8e1a29cf32e53f2c5a1e7dc6408041882e4c11542c6526"
  # U-233 (ENDF-B-VIII.1): the biggest LRF=3 RRR in the surveyed
  # corpus (6122 resonances). Performance stress and correctness
  # cross-check for the numba/JAX kernels under a resonance load
  # ~2x the U-235/U-238 files.
  "endfb81_n_U-233.endf   ENDF-B-VIII.1  n_092-U-233_9222   a51575252b97fe6605da3dbcbc7af4798d854f756fdd698d84b64f7466f983b4"
  # Pu-239 (CENDL-3.2): MT=1 pathology unique to CENDL in the
  # sweep: MT=1 max/median rel err both saturate to 1.0 while
  # MT=2/18/102 stay clean. Flag file for a follow-up sum-rule
  # / MF3-declaration investigation; kept in the corpus so that
  # any future fix can be pinned against it.
  "cendl32_n_Pu-239.endf  CENDL-3.2      n_094-Pu-239_9437  45f038bbef1a3b6a55a758879c0720f8aecdbd22b7d8f087abb1cc459c73dd42"
  # Nd-143 (ENDF-B-VIII.1): LRF=2 MLBW with an LSSF=0 URR at
  # higher energy (10% max on MT=1 in the sweep). Exercises the
  # MLBW <-> URR handoff in resonance_composition, which none of
  # the other LSSF=0 URR files in the corpus does (they all pair
  # URR with LRF=3 R-M).
  "endfb81_n_Nd-143.endf  ENDF-B-VIII.1  n_060-Nd-143_6028  a2c1a5d579f52edd84626477ed55a3e03c38c2efb0c651f6ab60fbd40418c1de"
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
