"""Preconditioned CG Dirichlet solve ``L f = s`` of the nodal SBP perpendicular Laplacian (the phi / polarization inversion).

With ``L f = -H^-1 (M f - b)`` the Dirichlet problem is ``M f = b - H s``, ``M`` symmetric positive definite.
:func:`solve_dirichlet` runs CG (flexible Polak-Ribiere form, identical to CG for a fixed SPD preconditioner and robust to the
float32 rounding of the factors) on the matrix-free JAX form of :mod:`drbx.native.fci_perpendicular_sbp_laplacian`, jitted with
the plan, the preconditioner, the coefficient and the data as arguments. The preconditioner is the block-Jacobi over eta planes
of :mod:`drbx.native.fci_perpendicular_plane_preconditioner` (exact banded block-LDU of every in-plane block), factorised from
the in-plane blocks assembled on the host by :class:`~drbx.validation.sbp_laplacian_audit.LaplacianAssembly` with the node keys
of :func:`~drbx.geometry.sbp_laplacian.plane_keys`. The core is a dense block that becomes the first super-ring.

Typical use, the Boussinesq potential ``div(grad phi) = tau div(grad q) - omega`` with ``q = n Ti`` (the P07 convention
``A(q) = -L_perp(q)``; ``phi`` and ``q`` carry their own wall data)::

    prec = build_dirichlet_preconditioner(lplan)                        # once per geometry / coefficient
    s = tau * laplacian_action(lplan, q, bcd_q, "dirichlet") - omega
    phi, info = solve_dirichlet(lplan, s, bcd_phi, prec, x0=phi_prev)    # every stage
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from drbx.geometry.sbp_laplacian import LaplacianPlan, plane_keys
from drbx.native.fci_perpendicular_plane_preconditioner import (
    PlanePreconditioner, apply_plane_preconditioner, build_plane_preconditioner)
from drbx.native.fci_perpendicular_sbp_laplacian import laplacian_form
from drbx.validation.sbp_laplacian_audit import LaplacianAssembly

__all__ = ["build_dirichlet_preconditioner", "solve_dirichlet", "solve_dirichlet_jit"]


def build_dirichlet_preconditioner(lp: LaplacianPlan, coeff=None, c_kappa: float = 1.0, *, factor_dtype: str = "float64",
                                   max_block: int | None = None, assembly: LaplacianAssembly | None = None
                                   ) -> PlanePreconditioner:
    """Plane-block preconditioner of the Dirichlet matrix (host; ``coeff (E, P)`` and ``c_kappa`` as in the apply).

    ``factor_dtype`` is the storage and apply dtype of the factors (``"float32"`` halves the memory traffic).
    ``prec.info`` records the block layout (``S`` super-rings of ``B`` slots, bandwidth ``w``, padding).
    """
    asm = LaplacianAssembly(lp, coeff, c_kappa) if assembly is None else assembly
    ring, plane, theta = plane_keys(lp.structure)
    return build_plane_preconditioner(asm.matrix("dirichlet", inplane=True), ring, plane, theta,
                                      factor_dtype=factor_dtype, max_block=max_block)


def solve_dirichlet(lp: LaplacianPlan, s, bcd, prec: PlanePreconditioner, *, coeff=None, c_kappa=1.0, x0=None,
                    rtol: float = 1e-10, maxit: int = 200, residual_norm: str = "h_inv"):
    """Solve ``L f = s`` for ``f (E, P)`` with Dirichlet wall data ``bcd`` (``s (E, P)`` the per-volume right-hand side).

    The relative residual is ``|r| / |rhs|`` of ``r = rhs - M f`` in the ``H^-1`` norm (``residual_norm="h_inv"``, the
    discrete ``L^2`` norm of the residual of ``L``) or the Euclidean norm (``"euclid"``). Returns ``(f, info)`` with
    ``iterations``, ``relative_residual`` and ``converged`` (jax scalars). ``rtol`` and ``maxit`` may be traced.
    """
    if residual_norm not in ("h_inv", "euclid"):
        raise ValueError(f"residual_norm must be 'h_inv' or 'euclid', got {residual_norm!r}")
    s = jnp.asarray(s)
    E, P = s.shape
    H = lp.Hp * lp.structure.deta
    rhs = -laplacian_form(lp, jnp.zeros_like(s), bcd, "dirichlet", coeff, c_kappa) - H * s

    def matvec(v):
        return laplacian_form(lp, v, None, "dirichlet", coeff, c_kappa)

    def precond(r):
        return apply_plane_preconditioner(prec, r.reshape(-1)).reshape(E, P)

    if residual_norm == "h_inv":
        def norm(r):
            return jnp.sqrt(jnp.sum(r * r / H))
    else:
        def norm(r):
            return jnp.sqrt(jnp.sum(r * r))

    x0 = jnp.zeros_like(s) if x0 is None else jnp.asarray(x0)
    r0 = rhs - matvec(x0)
    z0 = precond(r0)
    nb = norm(rhs)
    nb = jnp.where(nb > 0.0, nb, 1.0)

    def cond(st):
        _x, _r, _z, _p, _rz, it, rel = st
        return jnp.logical_and(rel > rtol, it < maxit)

    def body(st):
        x, r, z, p, rz, it, _rel = st
        Ap = matvec(p)
        alpha = rz / jnp.sum(p * Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        z_new = precond(r_new)
        rz_new = jnp.sum(r_new * z_new)
        beta = jnp.maximum(jnp.sum(z_new * (r_new - r)) / rz, 0.0)
        return x, r_new, z_new, z_new + beta * p, rz_new, it + 1, norm(r_new) / nb

    init = (x0, r0, z0, z0, jnp.sum(r0 * z0), jnp.asarray(0), norm(r0) / nb)
    x, _r, _z, _p, _rz, it, rel = jax.lax.while_loop(cond, body, init)
    return x, {"iterations": it, "relative_residual": rel, "converged": rel <= rtol}


solve_dirichlet_jit = jax.jit(solve_dirichlet, static_argnames=("residual_norm",))
