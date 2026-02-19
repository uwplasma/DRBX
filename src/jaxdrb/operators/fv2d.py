from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.bc import BC2D


def _pad_x(u: jnp.ndarray, dx: float, bc: BC2D) -> jnp.ndarray:
    if bc.kind_x == 0:
        gl = u[-1:, :]
        gr = u[0:1, :]
    elif bc.kind_x == 1:
        gl = 2.0 * bc.x_value - u[0:1, :]
        gr = 2.0 * bc.x_value - u[-1:, :]
    else:
        gl = u[0:1, :] - dx * bc.x_grad
        gr = u[-1:, :] + dx * bc.x_grad
    return jnp.concatenate([gl, u, gr], axis=0)


def _pad_y(u: jnp.ndarray, dy: float, bc: BC2D) -> jnp.ndarray:
    if bc.kind_y == 0:
        gl = u[:, -1:]
        gr = u[:, 0:1]
    elif bc.kind_y == 1:
        gl = 2.0 * bc.y_value - u[:, 0:1]
        gr = 2.0 * bc.y_value - u[:, -1:]
    else:
        gl = u[:, 0:1] - dy * bc.y_grad
        gr = u[:, -1:] + dy * bc.y_grad
    return jnp.concatenate([gl, u, gr], axis=1)


def ddx(u: jnp.ndarray, dx: float, bc: BC2D) -> jnp.ndarray:
    """Conservative finite-volume style first derivative in x."""

    up = _pad_x(u, dx, bc)
    f_face = 0.5 * (up[1:, :] + up[:-1, :])  # (nx+1, ny)
    return (f_face[1:, :] - f_face[:-1, :]) / dx


def ddy(u: jnp.ndarray, dy: float, bc: BC2D) -> jnp.ndarray:
    """Conservative finite-volume style first derivative in y."""

    up = _pad_y(u, dy, bc)
    f_face = 0.5 * (up[:, 1:] + up[:, :-1])  # (nx, ny+1)
    return (f_face[:, 1:] - f_face[:, :-1]) / dy


def laplacian(u: jnp.ndarray, dx: float, dy: float, bc: BC2D) -> jnp.ndarray:
    """Finite-volume flux-form Laplacian with periodic/Dirichlet/Neumann BCs."""

    upx = _pad_x(u, dx, bc)
    gx_face = (upx[1:, :] - upx[:-1, :]) / dx
    div_x = (gx_face[1:, :] - gx_face[:-1, :]) / dx

    upy = _pad_y(u, dy, bc)
    gy_face = (upy[:, 1:] - upy[:, :-1]) / dy
    div_y = (gy_face[:, 1:] - gy_face[:, :-1]) / dy

    return div_x + div_y


def biharmonic(u: jnp.ndarray, dx: float, dy: float, bc: BC2D) -> jnp.ndarray:
    return laplacian(laplacian(u, dx, dy, bc), dx, dy, bc)
