from __future__ import annotations

import jax
import jax.numpy as jnp


def rfft2(field: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.fft2(field)


def irfft2(field_hat: jnp.ndarray, *, real_output: bool = True) -> jnp.ndarray:
    """Inverse FFT returning either real or complex output.

    Most nonlinear examples in `jaxdrb` evolve real-valued fields. In those cases
    the inverse FFT should be real up to roundoff, and returning `.real` avoids
    dtype upcasting and small imaginary noise.

    For linearized workflows that use complex Fourier-mode representations
    (e.g. constant `k_par` modeled via `∂_|| -> i k_par`), we must preserve the
    complex result. Call with `real_output=False` in those cases.
    """

    out = jnp.fft.ifft2(field_hat)
    return out.real if real_output else out


def _real_output_like(field: jnp.ndarray) -> bool:
    return not jnp.iscomplexobj(field)


def dealias(field: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """2/3-rule dealiasing by zeroing high modes in Fourier space."""

    return irfft2(rfft2(field) * mask, real_output=_real_output_like(field))


def laplacian(field: jnp.ndarray, k2: jnp.ndarray) -> jnp.ndarray:
    return irfft2(-k2 * rfft2(field), real_output=_real_output_like(field))


def biharmonic(field: jnp.ndarray, k2: jnp.ndarray) -> jnp.ndarray:
    """Return ∇⁴(field) on a periodic domain via FFTs."""

    return irfft2((k2**2) * rfft2(field), real_output=_real_output_like(field))


def inv_laplacian(rhs: jnp.ndarray, k2: jnp.ndarray, *, k2_min: float = 1e-12) -> jnp.ndarray:
    """Solve ∇² u = rhs on a periodic domain with zero-mean gauge.

    For Fourier mode k=0, the inverse is singular; we set û(0)=0.
    """

    rhs_hat = rfft2(rhs)
    denom = jnp.where(k2 > 0.0, k2, 1.0)
    u_hat = -rhs_hat / jnp.maximum(denom, k2_min)
    u_hat = u_hat.at[0, 0].set(0.0 + 0.0j)
    return irfft2(u_hat, real_output=_real_output_like(rhs))


def ddx(field: jnp.ndarray, kx: jnp.ndarray) -> jnp.ndarray:
    return irfft2(1j * kx * rfft2(field), real_output=_real_output_like(field))


def ddy(field: jnp.ndarray, ky: jnp.ndarray) -> jnp.ndarray:
    return irfft2(1j * ky * rfft2(field), real_output=_real_output_like(field))


def poisson_bracket_spectral(
    phi: jnp.ndarray,
    f: jnp.ndarray,
    *,
    kx: jnp.ndarray,
    ky: jnp.ndarray,
    dealias_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Pseudo-spectral Poisson bracket [phi, f] on a periodic domain.

    Notes
    -----
    This uses a *skew-symmetric* discretization of the incompressible advection operator,
    which is significantly more robust in long-time nonlinear runs on collocation grids:

      [phi,f] = u·∇f,   u = (-∂y phi, ∂x phi)

    We compute

      u·∇f      (advective form)
      ∇·(u f)   (flux/divergence form)

    and average them. In the continuous periodic system they are identical; the averaged form
    improves discrete conservation properties.
    """

    dphi_dx = ddx(phi, kx)
    dphi_dy = ddy(phi, ky)
    u_x = -dphi_dy
    u_y = dphi_dx

    df_dx = ddx(f, kx)
    df_dy = ddy(f, ky)
    adv = u_x * df_dx + u_y * df_dy

    flux = ddx(u_x * f, kx) + ddy(u_y * f, ky)
    bracket = 0.5 * (adv + flux)
    if dealias_mask is None:
        return bracket
    return dealias(bracket, dealias_mask)


def poisson_bracket_spectral_multi(
    phi: jnp.ndarray,
    fields: jnp.ndarray,
    *,
    kx: jnp.ndarray,
    ky: jnp.ndarray,
    dealias_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute Poisson brackets [phi, f_i] for a stack of fields.

    This reuses dphi/dx and dphi/dy across all fields to reduce work.
    """

    dphi_dx = ddx(phi, kx)
    dphi_dy = ddy(phi, ky)
    u_x = -dphi_dy
    u_y = dphi_dx

    df_dx = jax.vmap(lambda f: ddx(f, kx))(fields)
    df_dy = jax.vmap(lambda f: ddy(f, ky))(fields)
    adv = u_x * df_dx + u_y * df_dy

    flux = jax.vmap(lambda f: ddx(u_x * f, kx) + ddy(u_y * f, ky))(fields)
    bracket = 0.5 * (adv + flux)

    if dealias_mask is None:
        return bracket
    return jax.vmap(lambda b: dealias(b, dealias_mask))(bracket)
