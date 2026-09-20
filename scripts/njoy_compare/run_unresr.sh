#!/bin/bash
# Run NJOY reconr + unresr on an ENDF-6 file to produce a PENDF
# file with the URR average cross sections written to MF2/MT152
# at infinite dilution (sigma0 = 1e10 b), T = 0 K. Comparison
# target for our URR reconstruction. See README.md.
#
# Usage:
#     ./run_unresr.sh <source.endf> <output-dir>
#
# Produces:
#     <output-dir>/tape22   PENDF with URR MF2/MT152 tabulated.
#
# reconr runs first because unresr consumes reconr's output
# PENDF, not the raw ENDF. reconr also normalises the file's
# format for the downstream unresr pass.

set -eu

if [ $# -ne 2 ]; then
    echo "Usage: $0 <source.endf> <output-dir>" >&2
    exit 1
fi

SRC=$1
OUT=$2
NJOY_BIN=${NJOY:-njoy}

if ! command -v "$NJOY_BIN" >/dev/null 2>&1; then
    CANDIDATE=$(ls -1 "$HOME"/miniconda3/pkgs/njoy2016*/bin/njoy 2>/dev/null | head -n 1 || true)
    if [ -n "${CANDIDATE:-}" ]; then
        NJOY_BIN=$CANDIDATE
    else
        echo "NJOY not found on PATH; set NJOY=<path-to-njoy> or install NJOY 2016." >&2
        exit 1
    fi
fi

mkdir -p "$OUT"
cd "$OUT"
cp "$SRC" tape20

MAT=$(awk 'NR==2 { print substr($0, 67, 4) }' tape20 | tr -d ' ')
echo "Source: $SRC"
echo "MAT   : $MAT"
echo "NJOY  : $NJOY_BIN"

# reconr: normalise + tolerance-based pointwise reconstruction in
# the RRR. unresr: URR average XS at infinite dilution (sig0 =
# 1e10 b), T=0 K. Writing to tape22.
cat > njoy.inp << EOF
reconr
20 21
'pointwise reconstruction, T=0 K, TOL=1e-3'/
${MAT} 0 0/
0.001 0.0/
0/
unresr
20 21 22
${MAT} 1 1 1/
0.0/
1.0e10/
0/
stop
EOF

echo "===== running NJOY reconr + unresr ====="
time "$NJOY_BIN" < njoy.inp 2>&1 | tail -20
echo "===== output ====="
ls -la tape22
echo "PENDF+URR written to $OUT/tape22"
