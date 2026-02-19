from __future__ import annotations

import jax.numpy as jnp


def zonal_fraction_y(f: jnp.ndarray, *, eps: float = 1e-30) -> jnp.ndarray:
    """Return the fraction of fluctuation RMS living in the zonal (ky=0) component.

    We define:

    - fluctuation: f' = f - <f> where <·> is the spatial mean over (x,y)
    - zonal part:  f_z = <f'>_y (mean over y, function of x only)

    Then the diagnostic is:

        Z = ||f_z||_2 / ||f'||_2

    Values near 1 indicate a nearly pure banded/zonal state. Values near 0 indicate
    non-zonal turbulence.
    """

    f = jnp.asarray(f)
    f_fluct = f - jnp.mean(f)
    f_zonal = jnp.mean(f_fluct, axis=1, keepdims=True)
    rms_total = jnp.sqrt(jnp.mean(jnp.square(f_fluct)))
    rms_zonal = jnp.sqrt(jnp.mean(jnp.square(f_zonal)))
    return rms_zonal / (rms_total + eps)


def isotropic_power_spectrum_2d(
    f: jnp.ndarray,
    *,
    Lx: float,
    Ly: float,
    nbins: int = 32,
    eps: float = 1e-30,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute an isotropic shell-binned power spectrum from a 2D periodic field.

    Parameters
    ----------
    f:
        2D array with shape (nx, ny).
    Lx, Ly:
        Domain size in x and y.
    nbins:
        Number of isotropic shells.

    Returns
    -------
    k_centers:
        Wavenumber bin centers.
    Pk:
        Shell-summed power in each bin.
    """

    f = jnp.asarray(f)
    nx, ny = f.shape
    dx = Lx / float(nx)
    dy = Ly / float(ny)

    fhat = jnp.fft.fft2(f)
    ps2d = jnp.real(fhat * jnp.conj(fhat))

    kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    kx, ky = jnp.meshgrid(kx_1d, ky_1d, indexing="ij")
    k = jnp.sqrt(kx**2 + ky**2)

    kmax = jnp.max(k)
    dk = (kmax + eps) / float(nbins)
    bin_index = jnp.floor(k / dk).astype(jnp.int32)
    bin_index = jnp.clip(bin_index, 0, nbins - 1)

    Pk = jnp.bincount(bin_index.reshape((-1,)), weights=ps2d.reshape((-1,)), length=nbins)
    k_centers = (jnp.arange(nbins) + 0.5) * dk
    return k_centers, Pk


def spectrum_loglog_slope(
    k: jnp.ndarray,
    Pk: jnp.ndarray,
    *,
    kmin: float,
    kmax: float,
    eps: float = 1e-30,
) -> jnp.ndarray:
    """Fit a log-log slope of P(k) over a finite band [kmin, kmax]."""

    k = jnp.asarray(k)
    Pk = jnp.asarray(Pk)
    mask = (k >= float(kmin)) & (k <= float(kmax)) & (Pk > 0.0)
    x = jnp.log(jnp.maximum(k[mask], eps))
    y = jnp.log(jnp.maximum(Pk[mask], eps))
    x = x - jnp.mean(x)
    y = y - jnp.mean(y)
    denom = jnp.sum(jnp.square(x)) + eps
    return jnp.sum(x * y) / denom
