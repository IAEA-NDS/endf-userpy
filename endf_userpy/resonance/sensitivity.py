from preprocessing import *

#===================================================================================
# Functions for cross sections, gradients and Jacobians
# with respect to: er, gn, gg, gf, gx
# All methods take a scalar incident energy e.
#===================================================================================

class MLBWSensitivity:
    """
    Helper for cross-section reconstruction, losses, gradients and Jacobians
    at a single incident energy e (scalar).

    Loop over energies in Python and JIT-compile the scalar functions.
    """

    @staticmethod
    def replace_params(data, er=None, gn=None, gg=None, gf=None, gx=None):
        kwargs = {}
        if er is not None: kwargs["er"] = er
        if gn is not None: kwargs["gn"] = gn
        if gg is not None: kwargs["gg"] = gg
        if gf is not None: kwargs["gf"] = gf
        if gx is not None: kwargs["gx"] = gx
        return data.replace(**kwargs)

    @staticmethod
    @jax.jit
    def valid_res_mask(data):
        return data.ich < data.nch

    @staticmethod
    @jax.jit
    def xs_all(er, gn, gg, gf, gx, data, e):
        """
        Scalar e. Returns scalars: tot, sct, cap, fis, pot, rxx
        """
        data2 = MLBWSensitivity.replace_params(data, er=er, gn=gn, gg=gg, gf=gf, gx=gx)
        sct, cap, fis, pot, rxx = mlbw(data2, e)
        tot = sct + cap + fis
        return tot, sct, cap, fis, pot, rxx

    @staticmethod
    @jax.jit
    def sigma_tot(er, gn, gg, gf, gx, data, e):
        return MLBWSensitivity.xs_all(er, gn, gg, gf, gx, data, e)[0]

    @staticmethod
    @jax.jit
    def sigma_sct(er, gn, gg, gf, gx, data, e):
        return MLBWSensitivity.xs_all(er, gn, gg, gf, gx, data, e)[1]

    @staticmethod
    @jax.jit
    def sigma_cap(er, gn, gg, gf, gx, data, e):
        return MLBWSensitivity.xs_all(er, gn, gg, gf, gx, data, e)[2]

    @staticmethod
    @jax.jit
    def sigma_fis(er, gn, gg, gf, gx, data, e):
        return MLBWSensitivity.xs_all(er, gn, gg, gf, gx, data, e)[3]

    @staticmethod
    @jax.jit
    def loss_tot(er, gn, gg, gf, gx, data, e, sigma_ref, weights=None):
        resid = MLBWSensitivity.sigma_tot(er, gn, gg, gf, gx, data, e) - sigma_ref
        return resid**2 if weights is None else weights * resid**2

    @staticmethod
    @jax.jit
    def loss_sct(er, gn, gg, gf, gx, data, e, sigma_ref, weights=None):
        resid = MLBWSensitivity.sigma_sct(er, gn, gg, gf, gx, data, e) - sigma_ref
        return resid**2 if weights is None else weights * resid**2

    @staticmethod
    @jax.jit
    def loss_cap(er, gn, gg, gf, gx, data, e, sigma_ref, weights=None):
        resid = MLBWSensitivity.sigma_cap(er, gn, gg, gf, gx, data, e) - sigma_ref
        return resid**2 if weights is None else weights * resid**2

    @staticmethod
    @jax.jit
    def loss_fis(er, gn, gg, gf, gx, data, e, sigma_ref, weights=None):
        resid = MLBWSensitivity.sigma_fis(er, gn, gg, gf, gx, data, e) - sigma_ref
        return resid**2 if weights is None else weights * resid**2

    @staticmethod
    @jax.jit
    def mask_grads(data, dEr, dGn, dGg, dGf, dGx):
        mask = MLBWSensitivity.valid_res_mask(data).astype(dEr.dtype)
        return dEr * mask, dGn * mask, dGg * mask, dGf * mask, dGx * mask

    @staticmethod
    def gradients_tot(data, e, sigma_ref, weights=None, apply_mask=True):
        grads = _grad_loss_tot_params(
            data.er, data.gn, data.gg, data.gf, data.gx, data, e, sigma_ref, weights
        )
        return MLBWSensitivity.mask_grads(data, *grads) if apply_mask else grads

    @staticmethod
    def gradients_sct(data, e, sigma_ref, weights=None, apply_mask=True):
        grads = _grad_loss_sct_params(
            data.er, data.gn, data.gg, data.gf, data.gx, data, e, sigma_ref, weights
        )
        return MLBWSensitivity.mask_grads(data, *grads) if apply_mask else grads

    @staticmethod
    def gradients_cap(data, e, sigma_ref, weights=None, apply_mask=True):
        grads = _grad_loss_cap_params(
            data.er, data.gn, data.gg, data.gf, data.gx, data, e, sigma_ref, weights
        )
        return MLBWSensitivity.mask_grads(data, *grads) if apply_mask else grads

    @staticmethod
    def gradients_fis(data, e, sigma_ref, weights=None, apply_mask=True):
        grads = _grad_loss_fis_params(
            data.er, data.gn, data.gg, data.gf, data.gx, data, e, sigma_ref, weights
        )
        return MLBWSensitivity.mask_grads(data, *grads) if apply_mask else grads

    @staticmethod
    def jacobians_tot(data, e, apply_mask=True):
        """Returns per-parameter Jacobian vectors of shape (NR,) at scalar e."""
        J_er, J_gn, J_gg, J_gf, J_gx = _jac_tot_all_params(
            data.er, data.gn, data.gg, data.gf, data.gx, data, e
        )
        if not apply_mask:
            return J_er, J_gn, J_gg, J_gf, J_gx
        mask = MLBWSensitivity.valid_res_mask(data).astype(J_er.dtype)
        return J_er * mask, J_gn * mask, J_gg * mask, J_gf * mask, J_gx * mask

    @staticmethod
    def check(data, e):
        """
        Quick internal-consistency check at scalar energy e.
        loss0 and grad_max_abs should both be ~0.
        Jacobian shapes are (NR,).
        """
        sigma_ref = MLBWSensitivity.sigma_tot(
            data.er, data.gn, data.gg, data.gf, data.gx, data, e
        )
        loss0 = MLBWSensitivity.loss_tot(
            data.er, data.gn, data.gg, data.gf, data.gx, data, e, sigma_ref
        )
        grads = MLBWSensitivity.gradients_tot(data, e, sigma_ref, apply_mask=True)
        J_er, J_gn, J_gg, J_gf, J_gx = MLBWSensitivity.jacobians_tot(data, e)

        return {
            "loss0": loss0,
            "grad_max_abs": {
                "er": jnp.max(jnp.abs(grads[0])),
                "gn": jnp.max(jnp.abs(grads[1])),
                "gg": jnp.max(jnp.abs(grads[2])),
                "gf": jnp.max(jnp.abs(grads[3])),
                "gx": jnp.max(jnp.abs(grads[4])),
            },
            "jac_shapes": {
                "er": J_er.shape,
                "gn": J_gn.shape,
                "gg": J_gg.shape,
                "gf": J_gf.shape,
                "gx": J_gx.shape,
            },
        }


#==========================================================================================
# JAX compiled reverse-mode gradients and Jacobians
# (scalar e → (NR,) gradient vectors)
#==========================================================================================

# Gradients of scalar losses w.r.t. each of the 5 parameter arrays (NR,) each
_grad_loss_tot_params = jax.jit(jax.grad(MLBWSensitivity.loss_tot, argnums=(0, 1, 2, 3, 4)))
_grad_loss_sct_params = jax.jit(jax.grad(MLBWSensitivity.loss_sct, argnums=(0, 1, 2, 3, 4)))
_grad_loss_cap_params = jax.jit(jax.grad(MLBWSensitivity.loss_cap, argnums=(0, 1, 2, 3, 4)))
_grad_loss_fis_params = jax.jit(jax.grad(MLBWSensitivity.loss_fis, argnums=(0, 1, 2, 3, 4)))

# Jacobian of sigma_tot w.r.t. all 5 parameter arrays in one reverse-mode pass
_jac_tot_all_params = jax.jit(
    jax.jacrev(MLBWSensitivity.sigma_tot, argnums=(0, 1, 2, 3, 4))
)

#=========================================================================================
# Finite difference checking
#=========================================================================================

def finite_difference_check(data, e, param_name="er", j=0, eps=1.0e-6):
    """
    Compare Jacobian element vs centered finite difference for parameter index j
    at scalar energy e.

    Returns dict with jac_val, fd_val, abs_err, rel_err.
    """
    er, gn, gg, gf, gx = data.er, data.gn, data.gg, data.gf, data.gx

    J_er, J_gn, J_gg, J_gf, J_gx = MLBWSensitivity.jacobians_tot(data, e, apply_mask=True)
    jac_map = {"er": J_er, "gn": J_gn, "gg": J_gg, "gf": J_gf, "gx": J_gx}
    jac_val = jac_map[param_name][j]

    if param_name == "er":
        sp = MLBWSensitivity.sigma_tot(er.at[j].add( eps), gn, gg, gf, gx, data, e)
        sm = MLBWSensitivity.sigma_tot(er.at[j].add(-eps), gn, gg, gf, gx, data, e)
    elif param_name == "gn":
        sp = MLBWSensitivity.sigma_tot(er, gn.at[j].add( eps), gg, gf, gx, data, e)
        sm = MLBWSensitivity.sigma_tot(er, gn.at[j].add(-eps), gg, gf, gx, data, e)
    elif param_name == "gg":
        sp = MLBWSensitivity.sigma_tot(er, gn, gg.at[j].add( eps), gf, gx, data, e)
        sm = MLBWSensitivity.sigma_tot(er, gn, gg.at[j].add(-eps), gf, gx, data, e)
    elif param_name == "gf":
        sp = MLBWSensitivity.sigma_tot(er, gn, gg, gf.at[j].add( eps), gx, data, e)
        sm = MLBWSensitivity.sigma_tot(er, gn, gg, gf.at[j].add(-eps), gx, data, e)
    elif param_name == "gx":
        sp = MLBWSensitivity.sigma_tot(er, gn, gg, gf, gx.at[j].add( eps), data, e)
        sm = MLBWSensitivity.sigma_tot(er, gn, gg, gf, gx.at[j].add(-eps), data, e)
    else:
        raise ValueError(f"Unknown parameter name: {param_name}")

    fd_val  = (sp - sm) / (2.0 * eps)
    abs_err = jnp.abs(fd_val - jac_val)
    rel_err = abs_err / (jnp.abs(fd_val) + 1.0e-30)

    return {"jac_val": jac_val, "fd_val": fd_val, "abs_err": abs_err, "rel_err": rel_err}

#=========================================================================================
# Chunked vmap: Vectorize function f over xs in chunks of chunk_size using vmap
#=========================================================================================

def chunked_vmap(f, xs, chunk_size=100):
    """
    Apply jax.vmap(f) over xs in chunks to bound peak memory usage.
    Returns the same result structure as jax.vmap(f)(xs), computed chunk-by-chunk.
    """
    n = xs.shape[0]
    results = []
    for start in range(0, n, chunk_size):
        results.append(jax.vmap(f)(xs[start:start + chunk_size]))
    if isinstance(results[0], tuple):
        return tuple(jnp.concatenate([r[i] for r in results]) for i in range(len(results[0])))
    return jnp.concatenate(results)


#==========================================================================================
# Active-parameter utilities for MLBW fitting
# All methods take scalar e; loop over energies in Python.
#==========================================================================================

class MLBWActiveParams:
    """
    Compact interface to optimise only active (non-padded) resonance parameters.

    theta = [er_active | gn_active | gg_active | gf_active | gx_active]
    Each block has length N_active.

    sigma_* and loss_* operate at a single scalar energy e.
    Assemble the full Jacobian matrix by looping over energies in Python.
    """

    @staticmethod
    def active_mask(data):
        return data.ich < data.nch

    @staticmethod
    def active_indices(data):
        mask = jax.device_get(MLBWActiveParams.active_mask(data))
        return jnp.array([i for i, m in enumerate(mask) if bool(m)], dtype=jnp.int32)

    @staticmethod
    def n_active_from_idx(active_idx):
        return active_idx.shape[0]

    @staticmethod
    def extract_theta(data):
        idx = MLBWActiveParams.active_indices(data)
        return jnp.concatenate([
            data.er[idx], data.gn[idx], data.gg[idx],
            data.gf[idx], data.gx[idx]
        ])

    @staticmethod
    def split_theta(theta, active_idx):
        na = active_idx.shape[0]
        return (theta[0:na], theta[na:2*na], theta[2*na:3*na],
                theta[3*na:4*na], theta[4*na:5*na])

    @staticmethod
    def inject_theta(theta, data, active_idx):
        er_a, gn_a, gg_a, gf_a, gx_a = MLBWActiveParams.split_theta(theta, active_idx)
        return data.replace(
            er=data.er.at[active_idx].set(er_a),
            gn=data.gn.at[active_idx].set(gn_a),
            gg=data.gg.at[active_idx].set(gg_a),
            gf=data.gf.at[active_idx].set(gf_a),
            gx=data.gx.at[active_idx].set(gx_a),
        )

    @staticmethod
    @jax.jit
    def xs_all(theta, data, active_idx, e):
        """Scalar e. Returns scalars: tot, sct, cap, fis, pot, rxx"""
        data2 = MLBWActiveParams.inject_theta(theta, data, active_idx)
        sct, cap, fis, pot, rxx = mlbw(data2, e)
        return sct + cap + fis, sct, cap, fis, pot, rxx

    @staticmethod
    @jax.jit
    def sigma_tot(theta, data, active_idx, e):
        return MLBWActiveParams.xs_all(theta, data, active_idx, e)[0]

    @staticmethod
    @jax.jit
    def sigma_sct(theta, data, active_idx, e):
        return MLBWActiveParams.xs_all(theta, data, active_idx, e)[1]

    @staticmethod
    @jax.jit
    def sigma_cap(theta, data, active_idx, e):
        return MLBWActiveParams.xs_all(theta, data, active_idx, e)[2]

    @staticmethod
    @jax.jit
    def sigma_fis(theta, data, active_idx, e):
        return MLBWActiveParams.xs_all(theta, data, active_idx, e)[3]

    @staticmethod
    @jax.jit
    def loss_tot(theta, data, active_idx, e, sigma_ref, weights=None):
        resid = MLBWActiveParams.sigma_tot(theta, data, active_idx, e) - sigma_ref
        return resid**2 if weights is None else weights * resid**2

    @staticmethod
    @jax.jit
    def loss_sct(theta, data, active_idx, e, sigma_ref, weights=None):
        resid = MLBWActiveParams.sigma_sct(theta, data, active_idx, e) - sigma_ref
        return resid**2 if weights is None else weights * resid**2

    @staticmethod
    @jax.jit
    def loss_cap(theta, data, active_idx, e, sigma_ref, weights=None):
        resid = MLBWActiveParams.sigma_cap(theta, data, active_idx, e) - sigma_ref
        return resid**2 if weights is None else weights * resid**2

    @staticmethod
    @jax.jit
    def loss_fis(theta, data, active_idx, e, sigma_ref, weights=None):
        resid = MLBWActiveParams.sigma_fis(theta, data, active_idx, e) - sigma_ref
        return resid**2 if weights is None else weights * resid**2

    @staticmethod
    def check(data, e):
        active_idx = MLBWActiveParams.active_indices(data)
        theta      = MLBWActiveParams.extract_theta(data)

        loss0  = active_self_loss(theta, data, active_idx, e)
        grad0  = active_self_grad(theta, data, active_idx, e)
        J0     = jac_tot_theta(theta, data, active_idx, e)

        return {
            "n_active":     active_idx.shape[0],
            "theta_shape":  theta.shape,
            "loss0":        loss0,
            "grad_max_abs": jnp.max(jnp.abs(grad0)),
            "jac_shape":    J0.shape,
        }


#===================================================================================
# Exact self-consistency check
#===================================================================================

@jax.jit
def active_self_loss(theta, data, active_idx, e):
    """
    Self-consistency check at scalar e.
    loss = (sigma(theta) - stop_gradient(sigma(theta)))^2  →  should be 0.
    """
    sigma = MLBWActiveParams.sigma_tot(theta, data, active_idx, e)
    resid = sigma - jax.lax.stop_gradient(sigma)
    return resid * resid


active_self_grad = jax.jit(jax.grad(active_self_loss, argnums=0))

#===================================================================================
# Gradient and Jacobian functions (scalar e → (N_theta,) vectors)
# Loop over energies in Python to assemble the full Jacobian matrix.
#===================================================================================

grad_loss_tot_theta = jax.jit(jax.grad(MLBWActiveParams.loss_tot, argnums=0))
grad_loss_sct_theta = jax.jit(jax.grad(MLBWActiveParams.loss_sct, argnums=0))
grad_loss_cap_theta = jax.jit(jax.grad(MLBWActiveParams.loss_cap, argnums=0))
grad_loss_fis_theta = jax.jit(jax.grad(MLBWActiveParams.loss_fis, argnums=0))

# jacrev on a scalar output = gradient; shape (N_theta,) per energy point
jac_tot_theta = jax.jit(jax.jacrev(MLBWActiveParams.sigma_tot, argnums=0))
jac_sct_theta = jax.jit(jax.jacrev(MLBWActiveParams.sigma_sct, argnums=0))
jac_cap_theta = jax.jit(jax.jacrev(MLBWActiveParams.sigma_cap, argnums=0))
jac_fis_theta = jax.jit(jax.jacrev(MLBWActiveParams.sigma_fis, argnums=0))
