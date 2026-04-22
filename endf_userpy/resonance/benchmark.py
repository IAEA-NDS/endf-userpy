import time
from sensitivity import *

endf_file = "n-041_Nb_093.endf"
el, eh, res_data, bkg_data = endf_preprocessor(endf_file)

idx = -1
print("\n=== Available resonance ranges and models ===")
for i in range(len(res_data)):
    imodel = res_data[i].model_type
    print(f"range {i}: El={el[i]}, Eh={eh[i]}, model={imodel}")
    if imodel == 2:  # mlbw
        idx = i

assert idx >= 0, f"no mlbw model"

data = res_data[idx].data

print("\n=== Selected resonance range ===")
print("idx =", idx)
print("El, Eh =", el[idx], eh[idx])
print("model =", res_data[idx].model_type)

NE         = 200_000
CHUNK_SIZE = 100
eb = jnp.linspace(el[idx], eh[idx], NE, dtype=jnp.float64)

active_idx = MLBWActiveParams.active_indices(data)
theta      = MLBWActiveParams.extract_theta(data)

print(f"NE={NE}, N_theta={theta.shape[0]}, chunk_size={CHUNK_SIZE}")
print(f"Full Jacobian shape will be ({NE}, {theta.shape[0]})\n")

# --- warm-up: compile all kernels on a tiny batch ---
print("Warming up JIT compilation...")
t_wu = time.perf_counter()
_xs_fn  = lambda e: MLBWSensitivity.xs_all(data.er, data.gn, data.gg, data.gf, data.gx, data, e)
_jac_fn = lambda e: jac_tot_theta(theta, data, active_idx, e)
chunked_vmap(_xs_fn,  eb[:CHUNK_SIZE], chunk_size=CHUNK_SIZE)
chunked_vmap(_jac_fn, eb[:CHUNK_SIZE], chunk_size=CHUNK_SIZE)
jax.effects_barrier()
print(f"  warm-up done in {time.perf_counter() - t_wu:.2f}s\n")

# --- benchmark cross sections ---
print(f"Benchmarking cross sections ({NE:,} energies)...")
t0 = time.perf_counter()
tot, sct, cap, fis, pot, rxx = chunked_vmap(_xs_fn, eb, chunk_size=CHUNK_SIZE)
jax.effects_barrier()
t_xs = time.perf_counter() - t0
print(f"  elapsed : {t_xs:.3f} s  ({t_xs/NE*1e6:.1f} µs/energy)")
print(f"  tot[0]  = {float(tot[0]):.6e}\n")

# --- benchmark Jacobian ---
print(f"Benchmarking full Jacobian ({NE:,} x {theta.shape[0]})...")
t0 = time.perf_counter()
J = chunked_vmap(_jac_fn, eb, chunk_size=CHUNK_SIZE)
jax.effects_barrier()
t_jac = time.perf_counter() - t0
print(f"  elapsed : {t_jac:.3f} s  ({t_jac/NE*1e6:.1f} µs/energy)")
print(f"  J.shape = {J.shape}")
print(f"  J[0,:5] = {J[0,:5]}\n")

print(f"Total (xs + jac): {t_xs + t_jac:.3f} s")
