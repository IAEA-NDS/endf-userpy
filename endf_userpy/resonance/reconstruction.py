from preprocessing import *

#====================================================================================
# Configurable data for blocking incident energies
#====================================================================================

BLOCK_SIZE = 512                   # Incident energy block size (for vmapping)
MAX_BLOCKS = 256                   # Max. number of incident energy blocks

#====================================================================================
# Optional safety check (no jittable)
#====================================================================================

def validate_range_capacity(e_vec, el, eh):
    """
    validate arrays range
    Input parameters:
       e_vec: incident energy vector
       el: array of lower energy limits of resonant ranges (N_TOTAL,)
       eh: array of higher energy limits of resonant ranges (N_TOTAL)
    Note:
       N_TOTAL = MAX_ISO * MAX_RANGES imported via preprocessing <- physiscs
    """
    
    e_np = np.asarray(e_vec)
    el_np = np.asarray(el)
    eh_np = np.asarray(eh)

    max_supported = MAX_BLOCKS * BLOCK_SIZE
    ne = e_np.size

    if ne > max_supported:
        raise ValueError(
            f"Incident energy array size = {ne} exceeds "
            f"MAX_BLOCKS * BLOCK_SIZE = {max_supported}"
        )

    i_start = np.searchsorted(e_np, el_np, side="left")
    i_end   = np.searchsorted(e_np, eh_np, side="right")
    lengths = np.maximum(i_end - i_start, 0)

    max_length = int(lengths.max()) if lengths.size > 0 else 0
    if max_length > max_supported:
        bad = np.where(lengths > max_supported)[0].tolist()
        raise ValueError(
            f"Some resonance ranges exceed capacity MAX_BLOCKS * BLOCK_SIZE = "
            f"{max_supported}. Maximum range length found: {max_length}. "
            f"Offending global range indices: {bad}"
        )

#====================================================================================
# Blockify a 1D vector in blocks of (n_blocks, BLOCK_SIZE)
#====================================================================================

@jax.jit
def blockify_vector(X):
    
    n_blocks = (X.size + BLOCK_SIZE - 1) // BLOCK_SIZE
    pad = n_blocks * BLOCK_SIZE - X.size
    X_blocks = jnp.pad(X, (0, pad))
    
    return X_blocks.reshape(n_blocks, BLOCK_SIZE)

#====================================================================================
# Precompute block indexes by intervals (X1, X2) (used for one isotope)
#====================================================================================

@jax.jit
def precompute_ranges(X, X1, X2):

    n  = X.size
    nr = X1.shape[0]

    i_start = jnp.searchsorted(X, X1, side="left")
    i_end   = jnp.searchsorted(X, X2, side="right")

    i_start = jnp.clip(i_start, 0, n)
    i_end   = jnp.clip(i_end, 0, n)

    lengths = i_end - i_start

    pad = MAX_RANGES - nr

    i_start = jnp.pad(i_start, (0, pad))
    lengths = jnp.pad(lengths, (0, pad))

    active = jnp.arange(MAX_RANGES) < nr

    return i_start, lengths, active

#====================================================================================
# Build blocks from precomputed indexes
# Output shape: (MAX_RANGES, MAX_BLOCKS, BLOCK_SIZE)
#====================================================================================

@jax.jit
def build_blocks_by_ranges(X_blocks, i_start, lengths, active):
    
    offsets = (
        jnp.arange(MAX_BLOCKS)[:, None] * BLOCK_SIZE
        + jnp.arange(BLOCK_SIZE)[None, :]
    ).reshape(-1)

    Xflat = X_blocks.reshape(-1)

    def build_range(start, length, act):
        idx = start + offsets
        idx_safe = jnp.clip(idx, 0, Xflat.size - 1)

        vals = Xflat[idx_safe]
        mask = (offsets < length) & act
        vals = jnp.where(mask, vals, 0.0)

        return vals.reshape(MAX_BLOCKS, BLOCK_SIZE)

    return jax.vmap(build_range)(i_start, lengths, active)

#====================================================================================
# Prepare a batch of TAB1s for parallel interpolation using vmap
#====================================================================================

def batch_tab1(bkg_data):
    
    tab1_list = [bkg_data.tot, bkg_data.sct, bkg_data.cap, bkg_data.fis]
    
    return tree_util.tree_map(lambda *xs: jnp.stack(xs), *tab1_list)

#====================================================================================
# Compute MF3 background on the incident energy grid
#====================================================================================

@jax.jit
def get_bkg(X, X_blocks, bkg_data):
    
    bkg_TAB1_batch = batch_tab1(bkg_data)

    sigma_blocks = jax.vmap(                    # vmap over TAB1
        jax.vmap(                               # vmap over X_blocks
            tab1_intp_vec,
            in_axes=(0, None)
        ),
        in_axes=(None, 0)
    )(X_blocks, bkg_TAB1_batch)                 # shape: (4, nblocks, BLOCK_SIZE)

    sigma_flat = lax.dynamic_slice(
        sigma_blocks.reshape(sigma_blocks.shape[0], -1),
        (0,0),
        (sigma_blocks.shape[0], X.size)
    )
    
    #(bkg_tot, bkg_sct, bkg_cap, bkg_fis)
    # shape = (X.shape,)
    return tuple(sigma_flat)                   

#====================================================================================
# Add MF3 background and compute total
#====================================================================================

@jax.jit
def add_resxs_bkg(sct_res, cap_res, fis_res, sigmas_bkg):
    
    tot = sct_res + cap_res + fis_res + sigmas_bkg[0]
    sct = sct_res + sigmas_bkg[1]
    cap = cap_res + sigmas_bkg[2]
    fis = fis_res + sigmas_bkg[3]
    
    return tot, sct, cap, fis

#====================================================================================
# Build one JIT kernel per isotope, process ONE BLOCK AT A TIME
#====================================================================================

def make_isotope_kernel(res_data_iso):

    # Static tuple of range functions; each one expects shape (BLOCK_SIZE,)
    
    res_fun = tuple(model.build_jax_vmap_function() for model in res_data_iso)
    
    local_offsets = jnp.arange(BLOCK_SIZE, dtype=jnp.int32)

    def apply_model(idx, e_block):
        return lax.switch(idx, res_fun, e_block)

    @jax.jit
    def isotope_kernel(e_vec, e_blocks, el_iso, eh_iso):
        
        i_start, lengths, active = precompute_ranges(e_vec, el_iso, eh_iso)
        eres_b = build_blocks_by_ranges(e_blocks, i_start, lengths, active)
        
        sink = e_vec.size
        zeros = jnp.zeros((sink + 1,), dtype=e_vec.dtype)

        def range_body(r, carry):
            sct_acc, cap_acc, fis_acc, pot_acc, rxx_acc = carry

            start_r = i_start[r]
            len_r   = lengths[r]
            act_r   = active[r]
            blocks_r = eres_b[r]                 # (MAX_BLOCKS, BLOCK_SIZE)

            def block_scan_fn(block_carry, b):
                sct_c, cap_c, fis_c, pot_c, rxx_c = block_carry

                e_block = blocks_r[b]
                
                sct_b, cap_b, fis_b, pot_b, rxx_b = apply_model(r, e_block)

                base = start_r + b * BLOCK_SIZE
                idx = base + local_offsets
                mask = act_r & ((b * BLOCK_SIZE + local_offsets) < len_r)

                idx_safe = jnp.where(mask, idx, sink)

                sct_c = sct_c.at[idx_safe].set(jnp.where(mask, sct_b, 0.0))
                cap_c = cap_c.at[idx_safe].set(jnp.where(mask, cap_b, 0.0))
                fis_c = fis_c.at[idx_safe].set(jnp.where(mask, fis_b, 0.0))
                pot_c = pot_c.at[idx_safe].set(jnp.where(mask, pot_b, 0.0))
                rxx_c = rxx_c.at[idx_safe].set(jnp.where(mask, rxx_b, 0.0))

                return (sct_c, cap_c, fis_c, pot_c, rxx_c), None

            (sct_acc, cap_acc, fis_acc, pot_acc, rxx_acc), _ = lax.scan(
                block_scan_fn,
                (sct_acc, cap_acc, fis_acc, pot_acc, rxx_acc),
                jnp.arange(MAX_BLOCKS, dtype=jnp.int32),
            )

            return (sct_acc, cap_acc, fis_acc, pot_acc, rxx_acc)

        sct_out, cap_out, fis_out, pot_out, rxx_out = lax.fori_loop(
            0,
            MAX_RANGES,
            range_body,
            (zeros, zeros, zeros, zeros, zeros),
        )

        return (
            sct_out[:-1],
            cap_out[:-1],
            fis_out[:-1],
            pot_out[:-1],
            rxx_out[:-1],
        )

    return isotope_kernel

#====================================================================================
# Prepare reconstruction objects
#====================================================================================

def prepare_reconstruction(endf_file):
    
    el, eh, res_data, bkg_data = endf_preprocessor(endf_file)

    iso_kernels = []
    for iso in range(MAX_ISO):
        imin = iso * MAX_RANGES
        imax = imin + MAX_RANGES
        iso_kernels.append(make_isotope_kernel(res_data[imin:imax]))

    return el, eh, tuple(iso_kernels), bkg_data

#====================================================================================
# Compute resonance contributions using prepared isotope kernels
#====================================================================================

def get_resxs(e_vec, e_blocks, el, eh, iso_kernels):
    sct = jnp.zeros_like(e_vec)
    cap = jnp.zeros_like(e_vec)
    fis = jnp.zeros_like(e_vec)
    pot = jnp.zeros_like(e_vec)
    rxx = jnp.zeros_like(e_vec)

    for iso in range(MAX_ISO):
        imin = iso * MAX_RANGES
        imax = imin + MAX_RANGES

        sct_i, cap_i, fis_i, pot_i, rxx_i = iso_kernels[iso](
            e_vec, e_blocks, el[imin:imax], eh[imin:imax]
        )

        sct = sct + sct_i
        cap = cap + cap_i
        fis = fis + fis_i
        pot = pot + pot_i
        rxx = rxx + rxx_i

    return sct, cap, fis, pot, rxx

#====================================================================================
# High-level reconstruction driver
#====================================================================================

def reconstruct_xs(e_vec, endf_file):
    """
    Reconstruct the resonant cross sections
    Input parameters:
        e_vec: Incident energy vector
        endf_file: ENDF-6 formatted evaluated nuclear data file
    Return:
        tot: Total cross section (from MF2 and MF3 backgrounds)
        sct: Elastic scattering cross section
        cap: Radiative capture cross section
        fis: Fission cross section
        pot: Potential scattering cross section
        rxx: Competive cross section, if any        
    """
    
    # Read ENDF-6 data, preprocess MF1-MF3, and prepare reconstruction objects
    el, eh, iso_kernels, bkg_data = prepare_reconstruction(endf_file)
    
    # Check fixed arrays size
    validate_range_capacity(e_vec, el, eh)
    
    # Prepare global incident energy blocks (MAX_BLOCKS, BLOCK_SIZES) 
    e_blocks = blockify_vector(e_vec)

    # Compute resonant contribution (MF2)
    sct_res, cap_res, fis_res, pot, rxx = get_resxs(
        e_vec, e_blocks, el, eh, iso_kernels
    )
    
    # Compute background (MF3)
    sigmas_bkg = get_bkg(e_vec, e_blocks, bkg_data)
    
    # Add resonant contribution and background
    tot, sct, cap, fis = add_resxs_bkg(sct_res, cap_res, fis_res, sigmas_bkg)

    return tot, sct, cap, fis, pot, rxx