#!/bin/bash
# Run NJOY reconr on an ENDF-6 file to produce a PENDF file at
# T=0 K, TOL=1e-3, for comparison against our resonance
# reconstruction. See README.md.

set -eu

if [ $# -ne 2 ]; then
    echo "Usage: $0 <source.endf> <output-dir>" >&2
    exit 1
fi

SRC=$1
OUT=$2
NJOY_BIN=${NJOY:-njoy}

if ! command -v "$NJOY_BIN" >/dev/null 2>&1; then
    # Fall back to the conda-installed one that many boxes have.
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

# MAT number is in columns 67..70 of the second line of every
# ENDF-6 file.
MAT=$(awk 'NR==2 { print substr($0, 67, 4) }' tape20 | tr -d ' ')
echo "Source: $SRC"
echo "MAT   : $MAT"
echo "NJOY  : $NJOY_BIN"

cat > njoy.inp << EOF
reconr
20 21
'pointwise reconstruction, T=0 K, TOL=1e-3'/
${MAT} 0 0/
0.001 0.0/
0/
stop
EOF

echo "===== running NJOY reconr ====="
time "$NJOY_BIN" < njoy.inp 2>&1 | tail -6
echo "===== output ====="
ls -la tape21
echo "PENDF written to $OUT/tape21"
