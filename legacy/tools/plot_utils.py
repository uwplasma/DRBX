from __future__ import annotations

import numpy as np


def lowpass_2d(field: np.ndarray, frac: float) -> np.ndarray:
    """Apply an isotropic low-pass filter in Fourier space.

    Parameters
    ----------
    field:
        2D array to filter.
    frac:
        Fraction of the Nyquist radius to keep (0 < frac <= 1).
    """
    if frac <= 0.0 or frac >= 1.0:
        return field
    nx, ny = field.shape
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    k2 = kxg**2 + kyg**2
    k2_max = float(np.max(k2))
    if k2_max <= 0.0:
        return field
    mask = k2 <= (frac**2) * k2_max
    fhat = np.fft.fftn(field)
    fhat *= mask
    return np.fft.ifftn(fhat).real


def maybe_lowpass(field: np.ndarray, frac: float | None) -> np.ndarray:
    if frac is None:
        return field
    frac = float(frac)
    if frac <= 0.0:
        return field
    return lowpass_2d(field, frac)
