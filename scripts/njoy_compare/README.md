# NJOY reconr comparison

Utility scripts for comparing our resonance reconstruction sketch
(`endf_userpy.mfsec_interpretation.mf2_interpretation_*`) against
NJOY reconr on real ENDF-6 files. Numerical agreement and a rough
speed comparison, per MT, on NJOY's own adaptive PENDF grid.

Not part of the automated test suite because they need external
tools (NJOY 2016) and specific ENDF files on disk. Point them at
whatever files you have.

## Prerequisites

- NJOY 2016 on `$PATH` (or edit `NJOY_BIN` in the shell script).
  The `miniconda` build at
  `.../miniconda3/pkgs/njoy2016-<ver>/bin/njoy` works.
- The Python package installed (this repo), with `numba` and
  `endf_parserpy` importable.
- An ENDF-6 file with either LRF=2 (MLBW, e.g. Nb-93) or LRF=3
  (Reich-Moore, e.g. U-235).

## Usage

```
# 1. Run reconr to produce a PENDF file at T=0 K, TOL=1e-3.
./run_reconr.sh <path-to-source.endf> <output-dir>

# 2. Compare our reconstruction against the PENDF.
python compare_vs_njoy.py \
    --source <path-to-source.endf> \
    --pendf  <output-dir>/tape21 \
    --formalism {mlbw,rm} \
    --rrr-hi  <upper energy of RRR in eV> \
    --mts 1,2,102[,18]
```

Prints for each MT: median / mean / max-abs-scaled relative
difference against NJOY, the point of maximum disagreement, and
the numba backend wall-clock timing on that grid.

## Reference numbers (against NJOY 2016.78, T=0 K, TOL=1e-3)

Nb-93 (`n-041_Nb_093.endf`, JEFF-3.3, RRR to 7 keV, 41 797 pts):

    MT=  2 (sct):  median rel err ~ 5e-8    numba wall  ~19 ms
    MT=102 (cap):  median rel err ~ 7e-8    numba wall  ~19 ms
    MT=  1 (tot):  median rel err ~ 7e-8    numba wall  ~18 ms

U-235 (ENDF/B-VIII.1, RRR to 2.25 keV, 232 410 pts):

    MT=  2 (sct):  median rel err ~ 1e-7    numba wall ~2.4 s
    MT= 18 (fis):  median rel err ~ 7e-8    numba wall ~2.6 s
    MT=102 (cap):  median rel err ~ 7e-8    numba wall ~2.6 s
    MT=  1 (tot):  median rel err ~ 2e-7    numba wall ~2.6 s

That is: our reconstruction agrees with NJOY to roughly float64
accumulation noise (~1e-7 relative) across all MTs on both
formalisms. NJOY reconr itself took 0.8 s for Nb-93 and 88 s
for U-235 producing the same PENDFs, so our numba path is
30-40× faster.

## Safety note

The Reich-Moore numpy path materialises an ``(NE, nres)`` complex
intermediate that at NE~200k, nres~3k reaches ~10 GB. On a machine
without swap, an unchecked OOM in that region can freeze the box.
The comparison script uses the numba backend by default, which
does not materialise that tensor and runs comfortably at a few
hundred MB regardless of grid size. Prepend `ulimit -v` to the
invocation as a belt-and-suspenders guard if you have a
non-standard memory configuration.
