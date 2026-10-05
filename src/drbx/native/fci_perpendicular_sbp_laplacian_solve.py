"""Preconditioned CG Dirichlet solve ``L f = s`` of the nodal SBP perpendicular Laplacian (the phi / polarization inversion).

With ``L f = -H^-1 (M f - b)`` the Dirichlet problem is ``M f = b - H s``, ``M`` symmetric positive definite.
:func:`solve_dirichlet` runs CG (flexible Polak-Ribiere form, identical to CG for a fixed SPD preconditioner and robust to the
float32 rounding of the factors) on the matrix-free JAX form of :mod:`drbx.native.fci_perpendicular_sbp_laplacian`, jitted with
the plan, the preconditioner, the coefficient and the data as arguments. The preconditioner is the block-Jacobi over eta planes
of :mod:`drbx.native.fci_perpendicular_plane_preconditioner`: every in-plane block ``[core, rings]`` is inverted exactly, the
rings by a banded block ``L D L^T`` (``rings_per_block`` rings per super-ring) and the dense core by its Schur complement
(``method="core_schur"``, the default), so the block size is set by the rings only. The in-plane blocks are assembled on the host
by :mod:`drbx.geometry.sbp_laplacian_assembly` a group of planes at a time (bounded memory, bitwise the blocks of the full
assembly). ``method="core_super_ring"`` is the previous layout (the P07 banded LDU with the core as the first super-ring).

Typical use, the Boussinesq potential ``div(grad phi) = omega - tau div(grad q)`` with ``q = n Ti`` (from
``omega = div(grad(phi + tau q))``; ``phi`` and ``q`` carry their own wall data)::

    prec = build_dirichlet_preconditioner(lplan)                        # once per geometry / coefficient
    s = omega - tau * laplacian_action(lplan, q, bcd_q, "dirichlet")
    phi, info = solve_dirichlet(lplan, s, bcd_phi, prec, x0=phi_prev)    # every stage

For gradients use :func:`solve_dirichlet_implicit` (same arguments and values): the solution carries the exact implicit
derivative of ``M f = b(bcd) - H s`` (forward and reverse mode through one adjoint solve), while :func:`solve_dirichlet` is the
bare CG ``while_loop`` (no reverse mode; forward mode would differentiate the iterations).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from drbx.geometry.sbp_laplacian import LaplacianPlan, plane_keys
from drbx.geometry.sbp_laplacian_assembly import LaplacianAssembly, iter_inplane_blocks
from drbx.native.fci_perpendicular_plane_preconditioner import (
    CoreSchurPreconditioner, PlanePreconditioner, apply_core_schur_preconditioner, apply_plane_preconditioner,
    build_core_schur_preconditioner, build_plane_preconditioner)
from drbx.native.fci_perpendicular_sbp_laplacian import laplacian_form

__all__ = ["apply_preconditioner", "build_dirichlet_preconditioner", "solve_dirichlet", "solve_dirichlet_implicit",
           "solve_dirichlet_implicit_jit", "solve_dirichlet_jit"]

_METHODS = ("core_schur", "core_super_ring")


def apply_preconditioner(prec, r):
    """``M^-1 r`` of either preconditioner type (the type is static under ``jit``)."""
    if isinstance(prec, CoreSchurPreconditioner):
        return apply_core_schur_preconditioner(prec, r)
    return apply_plane_preconditioner(prec, r)


def build_dirichlet_preconditioner(lp: LaplacianPlan, coeff=None, c_kappa: float = 1.0, *, factor_dtype: str = "float64",
                                   max_block: int | None = None, assembly: LaplacianAssembly | None = None,
                                   method: str = "core_schur", rings_per_block: int | str = "auto", group_planes: int | None = None
                                   ) -> PlanePreconditioner | CoreSchurPreconditioner:
    """Plane-block preconditioner of the Dirichlet matrix (host; ``coeff (E, P)`` and ``c_kappa`` as in the apply).

    ``factor_dtype`` is the storage and apply dtype of the factors (``"float32"`` halves the memory traffic).
    ``method="core_schur"`` (default): exact symmetric inverse of every in-plane block, rings by banded ``L D L^T`` with
    ``rings_per_block`` rings per super-ring (``"auto"``: the cheapest ring factor) and the core by its Schur complement
    (:func:`~drbx.native.fci_perpendicular_plane_preconditioner.build_core_schur_preconditioner`); the in-plane matrix is
    assembled ``group_planes`` planes at a time (default: about 12e6 stored entries per group, see
    :func:`~drbx.geometry.sbp_laplacian_assembly.plane_group_size`), so the host memory does not grow with the number of
    planes. ``method="core_super_ring"``: the P07 banded block-LDU of the full in-plane matrix with the core as the first
    super-ring (``max_block`` as in :func:`~drbx.native.fci_perpendicular_plane_preconditioner.build_plane_preconditioner`).
    A given ``assembly`` supplies its in-plane matrix instead of the windowed assembly. ``prec.info`` records the layout.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    st = lp.structure
    if method == "core_super_ring":
        asm = LaplacianAssembly(lp, coeff, c_kappa) if assembly is None else assembly
        ring, plane, theta = plane_keys(st)
        return build_plane_preconditioner(asm.matrix("dirichlet", inplane=True), ring, plane, theta,
                                          factor_dtype=factor_dtype, max_block=max_block)
    if max_block is not None:
        raise ValueError("max_block applies to method='core_super_ring'")
    if assembly is None:
        blocks = iter_inplane_blocks(lp, coeff, c_kappa, group_planes=group_planes)
    else:
        mat = assembly.matrix("dirichlet", inplane=True).tocsr()
        blocks = ((k0, k1, mat[k0 * st.P:k1 * st.P, k0 * st.P:k1 * st.P])
                  for k0, k1 in _groups(st.n_eta, group_planes))
    return build_core_schur_preconditioner(blocks, n_planes=st.n_eta, n_core=st.Nc, n_rings=st.m, ring_size=st.N,
                                           factor_dtype=factor_dtype, rings_per_block=rings_per_block)


def _groups(n: int, size: int | None):
    size = n if size is None else int(size)
    return [(k, min(k + size, n)) for k in range(0, n, size)]


def solve_dirichlet(lp: LaplacianPlan, s, bcd, prec: PlanePreconditioner | CoreSchurPreconditioner, *, coeff=None, c_kappa=1.0, x0=None,
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
        return apply_preconditioner(prec, r.reshape(-1)).reshape(E, P)

    if residual_norm == "h_inv":
        def norm(r):
            return jnp.sqrt(jnp.sum(r * r / H))
    else:
        def norm(r):
            return jnp.sqrt(jnp.sum(r * r))

    x0 = jnp.zeros_like(s) if x0 is None else jnp.asarray(x0)
    return _pcg(matvec, rhs, precond, norm, x0, rtol, maxit)


def _pcg(matvec, b, precond, norm, x0, rtol, maxit, *, zero_rhs_exact=False):
    """Flexible Polak-Ribiere preconditioned CG ``while_loop`` from ``x0``; returns ``(x, info)``.

    ``zero_rhs_exact``: a zero right-hand side returns exactly zero with no iterations (instead of ``rtol``-sized noise).
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


def solve_dirichlet_implicit(lp: LaplacianPlan, s, bcd, prec: PlanePreconditioner | CoreSchurPreconditioner, *, coeff=None,
                             c_kappa=1.0, x0=None, rtol: float = 1e-10, maxit: int = 200, residual_norm: str = "h_inv"):
    """:func:`solve_dirichlet` with implicit derivatives: same arguments, same ``(f, info)``, same values (the same CG).

    The solve is wrapped in :func:`jax.lax.custom_linear_solve` (``symmetric=True``: ``M`` is symmetric positive definite), so
    ``f = M^-1 (b(bcd) - H s)`` is differentiated at the solution instead of through the CG iterations. The JVP solves
    ``M df = db - H ds - dM f`` and the VJP (reverse mode) solves the adjoint system ``M w = f_bar`` once, both by the same
    preconditioned CG at relative tolerance ``rtol`` (relative to the norm of their own right-hand side, in the
    ``residual_norm`` norm) and ``maxit``.

    Differentiable: ``s``; the Dirichlet data in ``bcd`` (``b`` is affine in it, any wall array in ``bcd.value``); and ``coeff``
    and ``c_kappa`` (the operator enters through the ``matvec`` closure, ``dM f`` is the tangent of the matvec at the solution;
    the polarization-coefficient derivative is verified against finite differences in the tests). The operator itself is
    only piecewise smooth in ``coeff``: its penalty scalings use ``max |coeff trace|``, so at exactly tied extrema (e.g. a
    mirror-symmetric coefficient) the derivative is a subgradient, not the two-sided limit. Non-differentiable (treated
    as constants, their tangents are ignored): the preconditioner ``prec`` (only an approximate inverse used to accelerate the
    CG, it does not change the solution), the warm start ``x0`` (the result is independent of it up to ``rtol``), ``rtol`` and
    ``maxit``. Only ``f`` carries derivatives; ``info`` is non-differentiable diagnostics (``stop_gradient``).

    ``x0`` also warm-starts the tangent / adjoint solves (a poor guess for them, but they converge to the same relative
    tolerance); a zero tangent or cotangent right-hand side returns exactly zero.
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
        return apply_preconditioner(prec, r.reshape(-1)).reshape(E, P)

    if residual_norm == "h_inv":
        def norm(r):
            return jnp.sqrt(jnp.sum(r * r / H))
    else:
        def norm(r):
            return jnp.sqrt(jnp.sum(r * r))

    x0 = jnp.zeros_like(s) if x0 is None else jax.lax.stop_gradient(jnp.asarray(x0))
    rtol = jax.lax.stop_gradient(jnp.asarray(rtol))
    maxit = jax.lax.stop_gradient(jnp.asarray(maxit))

    def solve(mv, b):                                          # the CG of solve_dirichlet on the operator handed in
        return _pcg(mv, b, precond, norm, x0, rtol, maxit, zero_rhs_exact=True)

    x, info = jax.lax.custom_linear_solve(matvec, rhs, solve=solve, transpose_solve=solve, symmetric=True, has_aux=True)
    return x, jax.lax.stop_gradient(info)


solve_dirichlet_implicit_jit = jax.jit(solve_dirichlet_implicit, static_argnames=("residual_norm",))
