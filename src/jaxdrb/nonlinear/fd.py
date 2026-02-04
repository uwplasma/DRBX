from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D


def _pad_x(u: jnp.ndarray, dx: float, bc: BC2D) -> jnp.ndarray:
    # Return u padded with one ghost cell in x at both ends: shape (nx+2, ny).
    if bc.kind_x == 0:
        gl = u[-1:, :]
        gr = u[0:1, :]
    elif bc.kind_x == 1:
        gl = 2.0 * bc.x_value - u[1:2, :]
        gr = 2.0 * bc.x_value - u[-2:-1, :]
    else:
        gl = u[1:2, :] - 2.0 * dx * bc.x_grad
        gr = u[-2:-1, :] + 2.0 * dx * bc.x_grad
    return jnp.concatenate([gl, u, gr], axis=0)


def _pad_y(u: jnp.ndarray, dy: float, bc: BC2D) -> jnp.ndarray:
    # Return u padded with one ghost cell in y at both ends: shape (nx, ny+2).
    if bc.kind_y == 0:
        gl = u[:, -1:]
        gr = u[:, 0:1]
    elif bc.kind_y == 1:
        gl = 2.0 * bc.y_value - u[:, 1:2]
        gr = 2.0 * bc.y_value - u[:, -2:-1]
    else:
        gl = u[:, 1:2] - 2.0 * dy * bc.y_grad
        gr = u[:, -2:-1] + 2.0 * dy * bc.y_grad
    return jnp.concatenate([gl, u, gr], axis=1)


def ddx(u: jnp.ndarray, dx: float, bc: BC2D) -> jnp.ndarray:
    if bc.kind_x == 0:
        return (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2.0 * dx)
    up = _pad_x(u, dx, bc)
    return (up[2:, :] - up[:-2, :]) / (2.0 * dx)


def ddy(u: jnp.ndarray, dy: float, bc: BC2D) -> jnp.ndarray:
    if bc.kind_y == 0:
        return (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2.0 * dy)
    up = _pad_y(u, dy, bc)
    return (up[:, 2:] - up[:, :-2]) / (2.0 * dy)


def laplacian(u: jnp.ndarray, dx: float, dy: float, bc: BC2D) -> jnp.ndarray:
    if bc.kind_x == 0 and bc.kind_y == 0:
        return (jnp.roll(u, -1, axis=0) - 2.0 * u + jnp.roll(u, 1, axis=0)) / dx**2 + (
            jnp.roll(u, -1, axis=1) - 2.0 * u + jnp.roll(u, 1, axis=1)
        ) / dy**2

    upx = _pad_x(u, dx, bc)
    d2x = (upx[2:, :] - 2.0 * upx[1:-1, :] + upx[:-2, :]) / dx**2
    upy = _pad_y(u, dy, bc)
    d2y = (upy[:, 2:] - 2.0 * upy[:, 1:-1] + upy[:, :-2]) / dy**2
    return d2x + d2y


def biharmonic(u: jnp.ndarray, dx: float, dy: float, bc: BC2D) -> jnp.ndarray:
    """Return ∇⁴(u) using two applications of the FD Laplacian."""

    return laplacian(laplacian(u, dx, dy, bc), dx, dy, bc)


def boundary_mask(nx: int, ny: int, *, bc: BC2D) -> jnp.ndarray:
    """Mask for boundary nodes relevant to non-periodic BCs."""

    x_b = (bc.kind_x != 0) * (jnp.arange(nx) == 0) + (bc.kind_x != 0) * (jnp.arange(nx) == nx - 1)
    y_b = (bc.kind_y != 0) * (jnp.arange(ny) == 0) + (bc.kind_y != 0) * (jnp.arange(ny) == ny - 1)
    mx = x_b.astype(bool)[:, None]
    my = y_b.astype(bool)[None, :]
    return mx | my


def enforce_bc_relaxation(
    u: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    bc: BC2D,
    nu: float,
) -> jnp.ndarray:
    """Return an RHS term that relaxes boundary values toward the BC targets.

    - Dirichlet: u(boundary) -> value
    - Neumann:   u(boundary) -> u(neighbor) ± h*grad  (1st-order implied value)
    - Periodic:  no enforcement term
    """

    if nu == 0.0 or (bc.kind_x == 0 and bc.kind_y == 0):
        return jnp.zeros_like(u)

    nx, ny = u.shape
    mask = boundary_mask(nx, ny, bc=bc).astype(u.dtype)

    # Default target = current (no forcing) then override edges.
    target = u

    # X boundaries
    if bc.kind_x == 1:
        target = target.at[0, :].set(bc.x_value)
        target = target.at[-1, :].set(bc.x_value)
    elif bc.kind_x == 2:
        target = target.at[0, :].set(u[1, :] - dx * bc.x_grad)
        target = target.at[-1, :].set(u[-2, :] + dx * bc.x_grad)

    # Y boundaries
    if bc.kind_y == 1:
        target = target.at[:, 0].set(bc.y_value)
        target = target.at[:, -1].set(bc.y_value)
    elif bc.kind_y == 2:
        target = target.at[:, 0].set(u[:, 1] - dy * bc.y_grad)
        target = target.at[:, -1].set(u[:, -2] + dy * bc.y_grad)

    return -nu * mask * (u - target)


def _laplacian_homogeneous(u: jnp.ndarray, dx: float, dy: float, bc: BC2D) -> jnp.ndarray:
    """Linear Laplacian with homogeneous BCs.

    - periodic: periodic wrapping
    - dirichlet: homogeneous (value=0) via zero padding ghosts
    - neumann: homogeneous (grad=0) via reflection ghosts
    """

    if bc.kind_x == 0 and bc.kind_y == 0:
        return laplacian(u, dx, dy, bc)

    def pad_x_h(u_):
        if bc.kind_x == 1:
            return jnp.pad(u_, ((1, 1), (0, 0)), mode="constant", constant_values=0.0)
        if bc.kind_x == 2:
            gl = u_[1:2, :]
            gr = u_[-2:-1, :]
            return jnp.concatenate([gl, u_, gr], axis=0)
        # periodic
        gl = u_[-1:, :]
        gr = u_[0:1, :]
        return jnp.concatenate([gl, u_, gr], axis=0)

    def pad_y_h(u_):
        if bc.kind_y == 1:
            return jnp.pad(u_, ((0, 0), (1, 1)), mode="constant", constant_values=0.0)
        if bc.kind_y == 2:
            gl = u_[:, 1:2]
            gr = u_[:, -2:-1]
            return jnp.concatenate([gl, u_, gr], axis=1)
        gl = u_[:, -1:]
        gr = u_[:, 0:1]
        return jnp.concatenate([gl, u_, gr], axis=1)

    upx = pad_x_h(u)
    d2x = (upx[2:, :] - 2.0 * upx[1:-1, :] + upx[:-2, :]) / dx**2
    upy = pad_y_h(u)
    d2y = (upy[:, 2:] - 2.0 * upy[:, 1:-1] + upy[:, :-2]) / dy**2
    return d2x + d2y


def inv_laplacian_cg(
    rhs: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    bc: BC2D,
    maxiter: int = 200,
    tol: float = 0.0,
) -> jnp.ndarray:
    """Solve ∇² u = rhs with FD Laplacian and configurable BCs using fixed-iter CG.

    Notes
    -----
    - For pure Neumann problems, the Laplacian has a constant nullspace; we project
      rhs to zero-mean and return the zero-mean solution.
    - This solver is end-to-end differentiable through JAX's CG implementation.
    """

    nx, ny = rhs.shape

    if bc.kind_x == 0 and bc.kind_y == 0:
        # Periodic: solve full system (nullspace fixed by zero-mean gauge).
        rhs0 = rhs - jnp.mean(rhs)

        def mv(v_flat):
            v = v_flat.reshape((nx, ny))
            out = laplacian(v, dx, dy, bc)
            return out.reshape((-1,))

        b = rhs0.reshape((-1,))
        x0 = jnp.zeros_like(b)
        x, _ = jax.scipy.sparse.linalg.cg(mv, b, x0=x0, tol=tol, atol=tol, maxiter=maxiter)
        u = x.reshape((nx, ny))
        return u - jnp.mean(u)

    if bc.kind_x == 1 and bc.kind_y == 1:
        # Dirichlet: solve for interior unknowns with boundary fixed.
        value = float(bc.x_value)
        b_int = rhs[1:-1, 1:-1].reshape((-1,))

        def mv(v_flat):
            v = v_flat.reshape((nx - 2, ny - 2))
            u = jnp.full((nx, ny), value, dtype=rhs.dtype)
            u = u.at[1:-1, 1:-1].set(v)
            Lu = (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 + (
                u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]
            ) / dy**2
            return Lu.reshape((-1,))

        x0 = jnp.zeros_like(b_int)
        x, _ = jax.scipy.sparse.linalg.cg(mv, b_int, x0=x0, tol=tol, atol=tol, maxiter=maxiter)
        u = jnp.full((nx, ny), value, dtype=rhs.dtype)
        u = u.at[1:-1, 1:-1].set(x.reshape((nx - 2, ny - 2)))
        return u

    if bc.kind_x == 2 and bc.kind_y == 2:
        # Neumann: project rhs to the range of the Laplacian and solve full system,
        # choosing the zero-mean representative for the solution.
        rhs0 = rhs - jnp.mean(rhs)

        # Build a particular solution for constant boundary gradients.
        x = jnp.linspace(0.0, dx * (nx - 1), nx)[:, None]
        y = jnp.linspace(0.0, dy * (ny - 1), ny)[None, :]
        u_bc = bc.x_grad * x + bc.y_grad * y
        rhs_eff = rhs0 - _laplacian_homogeneous(u_bc, dx, dy, bc)

        def mv(v_flat):
            v = v_flat.reshape((nx, ny))
            out = _laplacian_homogeneous(v, dx, dy, bc)
            return out.reshape((-1,))

        b = rhs_eff.reshape((-1,))
        x0 = jnp.zeros_like(b)
        x, _ = jax.scipy.sparse.linalg.cg(mv, b, x0=x0, tol=tol, atol=tol, maxiter=maxiter)
        u = x.reshape((nx, ny)) + u_bc
        return u - jnp.mean(u)

    raise ValueError(
        "inv_laplacian_cg supports periodic, dirichlet/dirichlet, or neumann/neumann BCs."
    )
