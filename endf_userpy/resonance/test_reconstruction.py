from reconstruction import *
import time

#==============================================================================================
# Printing routine
#==============================================================================================

def print_diff(x, ycal, yref, label):
    n = x.size
    y = ycal[:n]
    y0 = yref[:n]
    erra = y - y0
    err = jnp.where(y0 != 0.0, erra * 100.0 / y0, 0.0)
    erramax = jnp.max(jnp.abs(erra))
    errmax = jnp.max(jnp.abs(err))

    lines = []
    lines.append(" ")
    lines.append(f"{label}")
    lines.append("-" * 95)
    lines.append(
        f"{'i':>7s} {'E':>14s} {'NJOY':>14s} {'CALC':>14s} {'DIFF[barn]':>14s} {'DIFF[%]':>14s}"
    )
    lines.append("-" * 95)

    for i in range(n):
        lines.append(
            f"{i:7d} {float(x[i]):14.6e} {float(y0[i]):14.6e} {float(y[i]):14.6e} "
            f"{float(erra[i]):14.6e} {float(err[i]):14.3f}"
        )

    lines.append(f" Max. Rel. Diff.   [%]  = {float(errmax):14.3f}")
    lines.append(f" Max. Abs. Diff. [barn] = {float(erramax):14.6e}")
    print("\n".join(lines))


#==============================================================================================
# Force completion of JAX work
#==============================================================================================

def block_all(*arrays):
    for a in arrays:
        a.block_until_ready()


#==============================================================================================
# MAIN PROGRAM
#==============================================================================================

t0 = time.perf_counter()
c0 = time.process_time()

#----------------------------------------------------------------------------------------------
# Input files
#----------------------------------------------------------------------------------------------

endf_file = "n-041_Nb_093.endf"
pendf_file = "n_41-Nb-93.pendf"

# Alternative example:
# endf_file = "n_28-Ni-059g.jeff"
# pendf_file = "n_28-Ni-59.pendf"

print(" endf6 data file:", endf_file)
print(" pendf data file:", pendf_file)
print()

#----------------------------------------------------------------------------------------------
# Read NJOY pendf for reference
#----------------------------------------------------------------------------------------------

t_read0 = time.perf_counter()
c_read0 = time.process_time()

parser_pendf = EndfParserFactory.create(endf_format="pendf")
d_pendf = parser_pendf.parsefile(pendf_file, include=[3])

ein_pnd = jnp.array(d_pendf[3][1]['xstable']['E'], dtype=jnp.float64)
tot_pnd = jnp.array(d_pendf[3][1]['xstable']['xs'], dtype=jnp.float64)
sct_pnd = jnp.array(d_pendf[3][2]['xstable']['xs'], dtype=jnp.float64)
cap_pnd = jnp.array(d_pendf[3][102]['xstable']['xs'], dtype=jnp.float64)

if 18 in d_pendf[3]:
    fis_pnd = jnp.array(d_pendf[3][18]['xstable']['xs'], dtype=jnp.float64)
else:
    fis_pnd = jnp.zeros(ein_pnd.size, dtype=jnp.float64)

del d_pendf

t_read1 = time.perf_counter()
c_read1 = time.process_time()

#----------------------------------------------------------------------------------------------
# Select incident energy grid
#----------------------------------------------------------------------------------------------

e_vec = ein_pnd
#e_vec = ein_pnd[:10]   # quick test

print(" Number of incident energies:", e_vec.size)
print()

#----------------------------------------------------------------------------------------------
# Prepare reconstruction objects once
#----------------------------------------------------------------------------------------------

t_prep0 = time.perf_counter()
c_prep0 = time.process_time()

el, eh, iso_kernels, bkg_data = prepare_reconstruction(endf_file)
validate_range_capacity(e_vec, el, eh)
e_blocks = blockify_vector(e_vec)

t_prep1 = time.perf_counter()
c_prep1 = time.process_time()

print(" Prepare reconstruction:")
print("   elapsed time =", t_prep1 - t_prep0)
print("   cpu time     =", c_prep1 - c_prep0)
print()

#----------------------------------------------------------------------------------------------
# Resonance warm-up: includes JIT compilation of isotope kernels
#----------------------------------------------------------------------------------------------

t1 = time.perf_counter()
c1 = time.process_time()

sct_res, cap_res, fis_res, pot, rxx = get_resxs(
    e_vec, e_blocks, el, eh, iso_kernels
)
block_all(sct_res, cap_res, fis_res, pot, rxx)

t2 = time.perf_counter()
c2 = time.process_time()

print(" Resonance warm-up (includes compilation):")
print("   elapsed time =", t2 - t1)
print("   cpu time     =", c2 - c1)
print()

#----------------------------------------------------------------------------------------------
# Resonance second call: already compiled execution
#----------------------------------------------------------------------------------------------

t3 = time.perf_counter()
c3 = time.process_time()

sct_res, cap_res, fis_res, pot, rxx = get_resxs(
    e_vec, e_blocks, el, eh, iso_kernels
)
block_all(sct_res, cap_res, fis_res, pot, rxx)

t4 = time.perf_counter()
c4 = time.process_time()

print(" Resonance second call (compiled execution):")
print("   elapsed time =", t4 - t3)
print("   cpu time     =", c4 - c3)
print()

#----------------------------------------------------------------------------------------------
# Background warm-up
#----------------------------------------------------------------------------------------------

t5 = time.perf_counter()
c5 = time.process_time()

sigmas_bkg = get_bkg(e_vec, e_blocks, bkg_data)
tot, sct, cap, fis = add_resxs_bkg(sct_res, cap_res, fis_res, sigmas_bkg)
block_all(tot, sct, cap, fis)

t6 = time.perf_counter()
c6 = time.process_time()

print(" Background warm-up:")
print("   elapsed time =", t6 - t5)
print("   cpu time     =", c6 - c5)
print()

#----------------------------------------------------------------------------------------------
# Background second call
#----------------------------------------------------------------------------------------------

t7 = time.perf_counter()
c7 = time.process_time()

sigmas_bkg = get_bkg(e_vec, e_blocks, bkg_data)
tot, sct, cap, fis = add_resxs_bkg(sct_res, cap_res, fis_res, sigmas_bkg)
block_all(tot, sct, cap, fis)

t8 = time.perf_counter()
c8 = time.process_time()

print(" Background second call (compiled execution):")
print("   elapsed time =", t8 - t7)
print("   cpu time     =", c8 - c7)
print()

#----------------------------------------------------------------------------------------------
# Full compiled calculation
#----------------------------------------------------------------------------------------------

t9 = time.perf_counter()
c9 = time.process_time()

sct_res, cap_res, fis_res, pot, rxx = get_resxs(
    e_vec, e_blocks, el, eh, iso_kernels
)
sigmas_bkg = get_bkg(e_vec, e_blocks, bkg_data)
tot, sct, cap, fis = add_resxs_bkg(sct_res, cap_res, fis_res, sigmas_bkg)
block_all(tot, sct, cap, fis, pot, rxx)

t10 = time.perf_counter()
c10 = time.process_time()

print(" Full compiled calculation:")
print("   elapsed time =", t10 - t9)
print("   cpu time     =", c10 - c9)
print()

#----------------------------------------------------------------------------------------------
# Comparison against NJOY
#----------------------------------------------------------------------------------------------

t_cmp0 = time.perf_counter()
c_cmp0 = time.process_time()

print(" Comparison endf6+reconstruction(CALC) vs NJOY_pendf(NJOY)")
print(" =========================================================")

print_diff(e_vec, tot, tot_pnd, " Total (MT=1)")
print_diff(e_vec, sct, sct_pnd, " Elastic scattering (MT=2)")
print_diff(e_vec, cap, cap_pnd, " Radiative capture (MT=102)")
print_diff(e_vec, fis, fis_pnd, " Fission (MT=18)")
print_diff(e_vec, pot, pot, " Potential scattering (SIGPOT)")
print_diff(e_vec, rxx, rxx, " Competitive reaction (SIGX)")

t_cmp1 = time.perf_counter()
c_cmp1 = time.process_time()

#----------------------------------------------------------------------------------------------
# Timing summary
#----------------------------------------------------------------------------------------------

print()
print(" Execution time summary:")
print(" Read NJOY pendf:                     cpu time =", c_read1 - c_read0, " elapsed time =", t_read1 - t_read0)
print(" Prepare reconstruction:              cpu time =", c_prep1 - c_prep0, " elapsed time =", t_prep1 - t_prep0)
print(" Resonance warm-up (compile):         cpu time =", c2 - c1,           " elapsed time =", t2 - t1)
print(" Resonance second call:               cpu time =", c4 - c3,           " elapsed time =", t4 - t3)
print(" Background warm-up:                  cpu time =", c6 - c5,           " elapsed time =", t6 - t5)
print(" Background second call:              cpu time =", c8 - c7,           " elapsed time =", t8 - t7)
print(" Full compiled calculation:           cpu time =", c10 - c9,          " elapsed time =", t10 - t9)
print(" Comparison and printing:             cpu time =", c_cmp1 - c_cmp0,   " elapsed time =", t_cmp1 - t_cmp0)

t11 = time.perf_counter()
c11 = time.process_time()

print(" Overall runtime summary:             cpu time =", c11 - c0,          " elapsed time =", t11 - t0)