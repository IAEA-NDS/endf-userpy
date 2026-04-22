import sys
import time
from sensitivity import *

#-------------
# Main
#-------------

if __name__ == "__main__":

    t0 = time.perf_counter()
    c0 = time.process_time()

    #------------------------------------------------------------
    # User inputs (files and incident energies)
    #------------------------------------------------------------

    endf_file = "n-041_Nb_093.endf"

    eb = jnp.array(
        [1.0e-5, 1.25e-5, 1.5e-5, 2.0e-5, 3.0e-5, 5.0e-5],
        dtype=jnp.float64
    )

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

    #------------------------------------------------------------
    # Compute cross sections via chunked vmap (chunk_size=100)
    #------------------------------------------------------------

    CHUNK_SIZE = 100

    _xs_fn = lambda e: MLBWSensitivity.xs_all(
        data.er, data.gn, data.gg, data.gf, data.gx, data, e
    )
    tot, sct, cap, fis, pot, rxx = chunked_vmap(_xs_fn, eb, chunk_size=CHUNK_SIZE)

    print("\n=== Cross sections ===")
    print("E   =", eb)
    print("TOT =", tot)
    print("SCT =", sct)
    print("CAP =", cap)
    print("FIS =", fis)
    print("POT =", pot)
    print("RXX =", rxx)

    # Internal check at the first energy point
    e0   = eb[0]
    info = MLBWSensitivity.check(data, e0)
    print("\n=== Internal check at e =", float(e0), "eV ===")
    print("loss0 =", info["loss0"])
    print("grad_max_abs =", info["grad_max_abs"])
    print("jac_shapes =", info["jac_shapes"])

    #------------------------------------------------------------
    # Finite-difference validation at the first valid resonance
    #------------------------------------------------------------

    valid     = MLBWSensitivity.valid_res_mask(data)
    valid_idx = jnp.where(valid)[0]

    if valid_idx.shape[0] > 0:
        jtest = int(valid_idx[0])
        e_test = eb[0]

        print("\n=== Finite-difference check at e =", float(e_test), "eV, j =", jtest, "===")
        for pname in ["er", "gn", "gg", "gf", "gx"]:
            out = finite_difference_check(data, e_test, param_name=pname, j=jtest, eps=1.0e-6)
            print(f"  {pname}: jac={float(out['jac_val']):.6e}  fd={float(out['fd_val']):.6e}"
                  f"  abs_err={float(out['abs_err']):.2e}  rel_err={float(out['rel_err']):.2e}")

    #------------------------------------------------------------
    # Active-parameter check at first energy
    #------------------------------------------------------------

    info2 = MLBWActiveParams.check(data, e0)
    print("\n=== Active-parameter check at e =", float(e0), "eV ===")
    print(info2)

    active_idx = MLBWActiveParams.active_indices(data)
    theta      = MLBWActiveParams.extract_theta(data)

    tot_scalar = MLBWActiveParams.sigma_tot(theta, data, active_idx, e0)
    print("\nTOT at e0 =", tot_scalar)

    # Gradient at a single energy
    sigma_ref = tot_scalar
    g = grad_loss_tot_theta(theta, data, active_idx, e0, sigma_ref, None)
    # Jacobian row at a single energy (shape: N_theta)
    J_row = jac_tot_theta(theta, data, active_idx, e0)

    print("\ngrad shape =", g.shape)
    print("J_row shape (one energy) =", J_row.shape)

    # Assemble full Jacobian via chunked vmap
    _jac_fn = lambda e: jac_tot_theta(theta, data, active_idx, e)
    J_rows = chunked_vmap(_jac_fn, eb, chunk_size=CHUNK_SIZE)
    print("Full Jacobian shape =", J_rows.shape)

    tf = time.perf_counter()
    cf = time.process_time()
    print('\nOverall runtime summary: cpu time =', cf-c0, '  elapsed time =', tf-t0)
