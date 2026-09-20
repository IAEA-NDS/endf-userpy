"""Fit-workflow helpers that plug into the array-namespace design.

The one function that lives here today is :func:`chunked_chi2`: a
memory-bounded chi-square loss for wrapping any array-agnostic
reconstruction into a fittable objective.

Why the helper exists
---------------------

For a JAX-based resonance / cross-section fit, the natural code
shape is::

    def chi2(theta):
        data = build_data(theta)
        sigma_pred = reconstruct(data, energies, xp)   # (N,)
        return jnp.sum(((sigma_pred - y_meas) / y_err) ** 2)

    grad = jax.grad(chi2)(theta)

That works at small ``N`` but fails silently at any real actinide
scale, because reverse-mode autodiff has to store activations from
the forward pass to run the backward pass. Chunking the loop in
Python doesn't help -- it just makes the traced graph bigger. The
correct pattern combines two JAX primitives:

    - ``jax.lax.scan`` to keep compile time flat as ``NE`` grows,
    - ``jax.checkpoint`` on the scan body to recompute per-chunk
      activations during backward instead of storing them.

Measured on this sketch (MLBW, NRES=500, NE=200 000, chunk=2000):

    - scan alone under jax.grad:            6.6 GB peak,  4.3 s/iter
    - scan + checkpoint under jax.grad:     0.60 GB peak, 1.5 s/iter

Not only bounded memory but faster wall clock (less cache thrashing).
:func:`chunked_chi2` bakes this combination in so callers get the
right behaviour without knowing about it.

Non-JAX backends do not run autodiff, so their implementation is a
plain Python loop over chunks. The API and the numerical result are
identical across backends; only the wall clock and memory profile
change.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def chunked_chi2(
    predict_fn: Callable[[Any, Any], Any],
    theta,
    energies,
    y_meas,
    y_err,
    xp,
    chunk: int = 5000,
) -> Any:
    """Chunked chi-square loss with memory-bounded gradients on JAX.

    Computes::

        L(theta) = sum_i ((predict_fn(theta, energies[i]) - y_meas[i]) / y_err[i]) ** 2

    but evaluates ``predict_fn`` on energy chunks of size ``chunk``
    rather than the whole grid at once.

    Parameters
    ----------
    predict_fn : callable
        ``predict_fn(theta, energies_slice) -> sigma_pred``. The
        slice has shape ``(chunk,)`` (or the remainder for the tail
        chunk if ``chunk`` does not divide ``len(energies)``); the
        return has the same shape. ``theta`` is passed through
        unchanged so that ``jax.grad(chunked_chi2)(theta)`` walks
        the whole pipeline (dataclass construction, reconstruction,
        residual weighting) end-to-end.
    theta : array_like
        Parameter vector. Traced under ``jax.grad``.
    energies, y_meas, y_err : array_like
        Measurement grid, target values, and errors. Same length.
        Must be numpy on numpy / numba backends, jnp on JAX.
    xp : backend
        As returned by :func:`endf_userpy.primitives.array_ns.get_backend`.
    chunk : int, optional
        Energy chunk size. Trades peak memory (larger chunk -> more
        per-iteration state) for per-iteration overhead (smaller
        chunk -> more iterations). 2000-10000 is a good starting
        range for MLBW-scale problems.

    Returns
    -------
    Scalar loss value (backend-typed).

    Notes
    -----
    On the JAX backend, uses ``jax.lax.scan`` for the loop body and
    wraps the body in ``jax.checkpoint`` so that ``jax.grad`` of
    this function has memory bounded by one chunk's worth of state,
    regardless of ``len(energies)``. See the module docstring for
    measured numbers.

    On the numpy and numba backends, uses a plain Python loop; the
    result is numerically identical but there is no gradient (those
    backends do not do autodiff).

    A trailing partial chunk (when ``chunk`` does not divide
    ``len(energies)``) is handled separately in a single extra
    ``predict_fn`` call after the scan. This can slightly enlarge
    the traced graph but keeps the API forgiving; pick a ``chunk``
    that divides ``len(energies)`` if trace-cleanliness matters.
    """
    name = getattr(xp, 'name', 'numpy')
    N = int(np.asarray(energies).shape[0])
    if chunk <= 0:
        raise ValueError(f'chunk must be positive, got {chunk!r}')

    if name == 'jax':
        return _chunked_chi2_jax(
            predict_fn, theta, energies, y_meas, y_err, chunk, N,
        )
    # numpy / numba / any other backend: plain Python loop.
    return _chunked_chi2_python(
        predict_fn, theta, energies, y_meas, y_err, chunk, N,
    )


def _chunked_chi2_python(
    predict_fn, theta, energies, y_meas, y_err, chunk, N,
):
    energies = np.asarray(energies)
    y_meas = np.asarray(y_meas)
    y_err = np.asarray(y_err)
    total = 0.0
    for start in range(0, N, chunk):
        e_c = energies[start:start + chunk]
        y_c = y_meas[start:start + chunk]
        err_c = y_err[start:start + chunk]
        sig_c = np.asarray(predict_fn(theta, e_c))
        total += float(np.sum(((sig_c - y_c) / err_c) ** 2))
    return total


def _chunked_chi2_jax(
    predict_fn, theta, energies, y_meas, y_err, chunk, N,
):
    # Local imports so numpy-only users don't pay the JAX import cost.
    import jax
    import jax.numpy as jnp

    n_full = (N // chunk) * chunk
    tail = N - n_full

    energies = jnp.asarray(energies)
    y_meas = jnp.asarray(y_meas)
    y_err = jnp.asarray(y_err)

    e_full = energies[:n_full].reshape(-1, chunk)
    y_full = y_meas[:n_full].reshape(-1, chunk)
    err_full = y_err[:n_full].reshape(-1, chunk)

    @jax.checkpoint
    def body(carry, x):
        e_c, y_c, err_c = x
        sig_c = predict_fn(theta, e_c)
        return carry + jnp.sum(((sig_c - y_c) / err_c) ** 2), None

    loss, _ = jax.lax.scan(body, 0.0, (e_full, y_full, err_full))

    if tail > 0:
        e_tail = energies[n_full:]
        y_tail = y_meas[n_full:]
        err_tail = y_err[n_full:]
        # Wrap the tail in checkpoint too so it doesn't spoil the
        # memory bound.
        @jax.checkpoint
        def tail_fn(theta):
            sig_tail = predict_fn(theta, e_tail)
            return jnp.sum(((sig_tail - y_tail) / err_tail) ** 2)
        loss = loss + tail_fn(theta)

    return loss
