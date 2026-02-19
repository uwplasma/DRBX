from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def fft_power2d(f: jnp.ndarray) -> jnp.ndarray:
    """Return unnormalized 2D power spectrum |FFT(f)|^2 for a real field."""

    fhat = jnp.fft.rfft2(f)
    return jnp.real(fhat * jnp.conj(fhat))


def radial_spectrum(
    power2d: jnp.ndarray,
    kx: jnp.ndarray,
    ky: jnp.ndarray,
    *,
    nbins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a 2D power spectrum into a 1D radial spectrum.

    Returns (k_centers, spectrum). Uses numpy for binning.
    """

    kx_np = np.asarray(kx)
    ky_np = np.asarray(ky)
    p_np = np.asarray(power2d)

    kkx, kky = np.meshgrid(kx_np, ky_np, indexing="ij")
    kvals = np.sqrt(kkx**2 + kky**2).ravel()
    pvals = p_np.ravel()

    if nbins is None:
        nbins = max(8, int(np.sqrt(pvals.size)))

    kmax = np.max(kvals)
    bins = np.linspace(0.0, kmax, nbins + 1)
    which = np.digitize(kvals, bins) - 1

    spectrum = np.zeros(nbins)
    counts = np.zeros(nbins)
    for i, w in enumerate(which):
        if 0 <= w < nbins:
            spectrum[w] += pvals[i]
            counts[w] += 1

    counts = np.where(counts > 0, counts, 1.0)
    spectrum = spectrum / counts
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    return k_centers, spectrum
