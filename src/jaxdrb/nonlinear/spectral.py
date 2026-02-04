from __future__ import annotations

import jax.numpy as jnp


def rfft2(field: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.fft2(field)


def irfft2(field_hat: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(field_hat).real


def dealias(field: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """2/3-rule dealiasing by zeroing high modes in Fourier space."""

    return irfft2(rfft2(field) * mask)


def laplacian(field: jnp.ndarray, k2: jnp.ndarray) -> jnp.ndarray:
    return irfft2(-k2 * rfft2(field))


def inv_laplacian(rhs: jnp.ndarray, k2: jnp.ndarray, *, k2_min: float = 1e-12) -> jnp.ndarray:
    """Solve ∇² u = rhs on a periodic domain with zero-mean gauge.

    For Fourier mode k=0, the inverse is singular; we set û(0)=0.
    """

    rhs_hat = rfft2(rhs)
    denom = jnp.where(k2 > 0.0, k2, 1.0)
    u_hat = -rhs_hat / jnp.maximum(denom, k2_min)
    u_hat = u_hat.at[0, 0].set(0.0 + 0.0j)
    return irfft2(u_hat)


def ddx(field: jnp.ndarray, kx: jnp.ndarray) -> jnp.ndarray:
    return irfft2(1j * kx * rfft2(field))


def ddy(field: jnp.ndarray, ky: jnp.ndarray) -> jnp.ndarray:
    return irfft2(1j * ky * rfft2(field))


def poisson_bracket_spectral(
    phi: jnp.ndarray,
    f: jnp.ndarray,
    *,
    kx: jnp.ndarray,
    ky: jnp.ndarray,
    dealias_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Pseudo-spectral Poisson bracket [phi, f] on a periodic domain."""

    dphi_dx = ddx(phi, kx)
    dphi_dy = ddy(phi, ky)
    df_dx = ddx(f, kx)
    df_dy = ddy(f, ky)
    bracket = dphi_dx * df_dy - dphi_dy * df_dx
    if dealias_mask is None:
        return bracket
    return dealias(bracket, dealias_mask)
