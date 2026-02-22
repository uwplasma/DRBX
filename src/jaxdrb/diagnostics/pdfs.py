from __future__ import annotations

import numpy as np


def pdf_1d(
    field,
    *,
    bins: int = 80,
    range: tuple[float, float] | None = None,
    density: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return histogram centers and PDF for a field."""
    arr = np.asarray(field).ravel()
    hist, edges = np.histogram(arr, bins=bins, range=range, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def joint_pdf(
    field_x,
    field_y,
    *,
    bins: int = 64,
    range: tuple[tuple[float, float], tuple[float, float]] | None = None,
    density: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 2D joint PDF for two fields."""
    x = np.asarray(field_x).ravel()
    y = np.asarray(field_y).ravel()
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range, density=density)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    return xc, yc, hist
