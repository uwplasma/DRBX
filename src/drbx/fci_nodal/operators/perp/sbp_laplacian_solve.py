"""Preconditioned conjugate gradients."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _dot(a, b):
    return jnp.sum(a * b)


def _pcg(matvec, b, precond, norm, x0, rtol, maxit, *, zero_rhs_exact=False, dot=_dot):
    """Flexible Polak-Ribiere preconditioned CG ``while_loop`` from ``x0``; returns ``(x, info)``.

    ``zero_rhs_exact``: a zero right-hand side returns exactly zero with no iterations (instead of ``rtol``-sized noise).
    ``dot``: the inner product (a ``psum`` reduction under ``shard_map``).
    """
    nb0 = norm(b)
    nb = jnp.where(nb0 > 0.0, nb0, 1.0)
    if zero_rhs_exact:
        x0 = jnp.where(nb0 > 0.0, x0, 0.0)
    r0 = b - matvec(x0)
    z0 = precond(r0)

    def cond(st):
        _x, _r, _z, _p, _rz, it, rel = st
        return jnp.logical_and(rel > rtol, it < maxit)

    def body(st):
        x, r, z, p, rz, it, _rel = st
        Ap = matvec(p)
        alpha = rz / dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        z_new = precond(r_new)
        rz_new = dot(r_new, z_new)
        beta = jnp.maximum(dot(z_new, r_new - r) / rz, 0.0)
        return x, r_new, z_new, z_new + beta * p, rz_new, it + 1, norm(r_new) / nb

    init = (x0, r0, z0, z0, dot(r0, z0), jnp.asarray(0), norm(r0) / nb)
    x, _r, _z, _p, _rz, it, rel = jax.lax.while_loop(cond, body, init)
    return x, {"iterations": it, "relative_residual": rel, "converged": rel <= rtol}


