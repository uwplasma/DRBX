from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Spectrum2D:
    kx: np.ndarray
    ky: np.ndarray
    power: np.ndarray


@dataclass(frozen=True)
class Spectrum1D:
    k: np.ndarray
    power: np.ndarray


def _as_numpy(field) -> np.ndarray:
    return np.asarray(field)


def _window_2d(nx: int, ny: int, kind: str | None) -> np.ndarray:
    if kind is None or kind.lower() in ("none", "off"):
        return np.ones((nx, ny))
    kind = kind.lower()
    if kind in ("hann", "hanning"):
        wx = np.hanning(nx)
        wy = np.hanning(ny)
        return np.outer(wx, wy)
    if kind in ("hamming",):
        wx = np.hamming(nx)
        wy = np.hamming(ny)
        return np.outer(wx, wy)
    raise ValueError(f"Unsupported window '{kind}'.")


def power_spectrum_2d(
    field,
    *,
    dx: float,
    dy: float,
    detrend: str | None = "mean",
    window: str | None = "hann",
    normalize: bool = True,
) -> Spectrum2D:
    """Return the 2D power spectrum of a real field.

    Parameters
    ----------
    field:
        2D array-like.
    dx, dy:
        Grid spacing.
    detrend:
        "mean" subtracts the global mean before FFT.
    window:
        Optional window (hann/hamming) to reduce spectral leakage.
    normalize:
        When True, normalize by (nx*ny)^2 so Parseval holds approximately.
    """
    arr = _as_numpy(field)
    if arr.ndim != 2:
        raise ValueError("power_spectrum_2d expects a 2D field.")
    nx, ny = arr.shape
    if detrend and detrend.lower() == "mean":
        arr = arr - np.mean(arr)
    win = _window_2d(nx, ny, window)
    arr = arr * win
    fft = np.fft.fftn(arr)
    power = np.abs(fft) ** 2
    if normalize:
        power = power / float(nx * ny) ** 2
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    return Spectrum2D(kx=kx, ky=ky, power=power)


def kxky_spectrum(field, *, dx: float, dy: float) -> Spectrum2D:
    """Convenience wrapper for the 2D spectrum without detrending/window."""
    return power_spectrum_2d(field, dx=dx, dy=dy, detrend=None, window=None)


def isotropic_spectrum(
    field,
    *,
    dx: float,
    dy: float,
    nbins: int | None = None,
    detrend: str | None = "mean",
    window: str | None = "hann",
) -> Spectrum1D:
    """Return isotropic (shell-averaged) spectrum."""
    spec = power_spectrum_2d(field, dx=dx, dy=dy, detrend=detrend, window=window)
    kx, ky = np.meshgrid(spec.kx, spec.ky, indexing="ij")
    k = np.sqrt(kx**2 + ky**2)
    kmax = float(np.max(k))
    if nbins is None:
        nbins = max(spec.power.shape) // 2
    edges = np.linspace(0.0, kmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    power = np.zeros_like(centers)
    counts = np.zeros_like(centers)
    flat_k = k.ravel()
    flat_p = spec.power.ravel()
    idx = np.digitize(flat_k, edges) - 1
    for i, (bin_idx, pval) in enumerate(zip(idx, flat_p, strict=False)):
        if 0 <= bin_idx < nbins:
            power[bin_idx] += pval
            counts[bin_idx] += 1.0
    counts = np.maximum(counts, 1.0)
    power = power / counts
    return Spectrum1D(k=centers, power=power)
