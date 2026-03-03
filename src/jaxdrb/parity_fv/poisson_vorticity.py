from __future__ import annotations

import jax
import jax.numpy as jnp


def laplacian_xy_periodic(phi: jnp.ndarray, *, dx: float, dy: float) -> jnp.ndarray:
    """Periodic Cartesian Laplacian over the last two axes.

    Supports arrays shaped `(nx, ny)` or `(nz, nx, ny)`.
    """

    inv_dx2 = 1.0 / max(float(dx) ** 2, 1e-30)
    inv_dy2 = 1.0 / max(float(dy) ** 2, 1e-30)
    dxx = (jnp.roll(phi, -1, axis=-2) - 2.0 * phi + jnp.roll(phi, 1, axis=-2)) * inv_dx2
    dyy = (jnp.roll(phi, -1, axis=-1) - 2.0 * phi + jnp.roll(phi, 1, axis=-1)) * inv_dy2
    return dxx + dyy


def solve_poisson_xy_spectral(
    omega: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    gauge_fix: bool = True,
) -> jnp.ndarray:
    """Solve ∇⊥²φ = ω with periodic x/y using FFT over last two axes."""

    nx = int(omega.shape[-2])
    ny = int(omega.shape[-1])
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=float(dx))
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=float(dy))
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    denom = -k2
    denom_safe = jnp.where(k2 > 0.0, denom, 1.0)

    omega_hat = jnp.fft.fftn(omega, axes=(-2, -1))
    phi_hat = omega_hat / denom_safe
    phi_hat = phi_hat.at[..., 0, 0].set(0.0)
    phi = jnp.fft.ifftn(phi_hat, axes=(-2, -1)).real
    return phi


def laplacian_xy_spectral(phi: jnp.ndarray, *, dx: float, dy: float) -> jnp.ndarray:
    """Spectral Cartesian Laplacian over last two axes."""

    nx = int(phi.shape[-2])
    ny = int(phi.shape[-1])
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=float(dx))
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=float(dy))
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    phi_hat = jnp.fft.fftn(phi, axes=(-2, -1))
    omega_hat = -k2 * phi_hat
    return jnp.fft.ifftn(omega_hat, axes=(-2, -1)).real


def apply_invert_set_x_guard(
    phi_plus_pi: jnp.ndarray,
    *,
    xstart: int,
    xend: int,
    ystart: int,
    yend: int,
) -> jnp.ndarray:
    """Apply Hermes/BOUT INVERT_SET x-boundary midpoint guard update.

    Layout is `(nz, nx, ny)` with guard cells present in `x` and `y`.
    """

    ys = slice(int(ystart), int(yend) + 1)
    out = phi_plus_pi
    out = out.at[:, int(xstart) - 1, ys].set(
        0.5 * (phi_plus_pi[:, int(xstart) - 1, ys] + phi_plus_pi[:, int(xstart), ys])
    )
    out = out.at[:, int(xend) + 1, ys].set(
        0.5 * (phi_plus_pi[:, int(xend) + 1, ys] + phi_plus_pi[:, int(xend), ys])
    )
    return out


def copy_outer_x_guard_cells(
    phi: jnp.ndarray,
    *,
    xstart: int,
    xend: int,
    ystart: int,
    yend: int,
) -> jnp.ndarray:
    """Copy outer x guard cells as in Hermes vorticity post-solve path."""

    ys = slice(int(ystart), int(yend) + 1)
    nx_tot = int(phi.shape[1])

    def _left_body(i: int, arr: jnp.ndarray) -> jnp.ndarray:
        idx = int(xstart) - 2 - i
        return arr.at[:, idx, ys].set(arr[:, idx + 1, ys])

    def _right_body(i: int, arr: jnp.ndarray) -> jnp.ndarray:
        idx = int(xend) + 2 + i
        return arr.at[:, idx, ys].set(arr[:, idx - 1, ys])

    out = phi
    n_left = max(int(xstart) - 1, 0)
    n_right = max(nx_tot - (int(xend) + 2), 0)
    out = jax.lax.fori_loop(0, n_left, _left_body, out)
    out = jax.lax.fori_loop(0, n_right, _right_body, out)
    return out


def apply_parallel_free_y_guard(
    phi: jnp.ndarray,
    *,
    ystart: int,
    yend: int,
    xstart: int,
    xend: int,
) -> jnp.ndarray:
    """Apply free-gradient y-guard update used before vorticity derivatives."""

    xs = slice(int(xstart), int(xend) + 1)
    out = phi
    out = out.at[:, xs, int(ystart) - 1].set(
        2.0 * phi[:, xs, int(ystart)] - phi[:, xs, int(ystart) + 1]
    )
    out = out.at[:, xs, int(yend) + 1].set(2.0 * phi[:, xs, int(yend)] - phi[:, xs, int(yend) - 1])
    return out


def prepare_phi_plus_pi_for_poisson(
    phi: jnp.ndarray,
    pi_hat: jnp.ndarray,
    *,
    xstart: int,
    xend: int,
    ystart: int,
    yend: int,
) -> jnp.ndarray:
    """Hermes-equivalent pre-inversion state for the vorticity solve."""

    return apply_invert_set_x_guard(
        phi + pi_hat,
        xstart=xstart,
        xend=xend,
        ystart=ystart,
        yend=yend,
    )


def finalize_phi_after_poisson(
    phi: jnp.ndarray,
    *,
    xstart: int,
    xend: int,
    ystart: int,
    yend: int,
    parallel_free_y: bool,
) -> jnp.ndarray:
    """Hermes-equivalent post-inversion guard-cell updates for `phi`."""

    out = copy_outer_x_guard_cells(
        phi,
        xstart=xstart,
        xend=xend,
        ystart=ystart,
        yend=yend,
    )
    if parallel_free_y:
        out = apply_parallel_free_y_guard(
            out,
            ystart=ystart,
            yend=yend,
            xstart=xstart,
            xend=xend,
        )
    return out
