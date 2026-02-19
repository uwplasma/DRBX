from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from .gbs_io import load_gbs_field, infer_gbs_grid, parse_gbs_input_text, read_gbs_stdin_text


def _default_index(size: int) -> int:
    return max(size // 2, 0)


def slice_2d(data_zxy: np.ndarray, cut: Literal["pol", "tor", "rad"], index: int | None = None) -> np.ndarray:
    # data_zxy shape (z, x, y)
    nz, nx, ny = data_zxy.shape
    if cut == "pol":
        iz = _default_index(nz) if index is None else int(index)
        return data_zxy[iz, :, :].T  # (y, x)
    if cut == "tor":
        ix = _default_index(nx) if index is None else int(index)
        return data_zxy[:, ix, :].T  # (y, z)
    if cut == "rad":
        iy = _default_index(ny) if index is None else int(index)
        return data_zxy[:, :, iy].T  # (x, z)
    raise ValueError(f"Unknown cut '{cut}'")


def plot_snapshot(
    h5_path: str | Path,
    field: str,
    *,
    step: int | str | None = None,
    cut: Literal["pol", "tor", "rad"] = "pol",
    index: int | None = None,
    axes: str = "zxy",
    title: str | None = None,
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    output: str | Path | None = None,
) -> plt.Figure:
    data = load_gbs_field(h5_path, field, step=step, axes=axes)
    data2d = slice_2d(data[0], cut=cut, index=index)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    im = ax.imshow(
        data2d,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title or f"{field} ({cut})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    if output is not None:
        fig.savefig(str(output), dpi=150)
    return fig


def _poloidal_coords(ny: int, nx: int, Lx: float, Ly: float, width_factor: float = 0.9, theta_window: float = 0.0):
    a = Ly / (2.0 * np.pi)
    width = width_factor * Lx
    theta = np.linspace(theta_window / 2.0, 2.0 * np.pi - theta_window / 2.0, ny)
    x = np.linspace(0.0, width, nx)
    xp = np.zeros((ny, nx))
    yp = np.zeros((ny, nx))
    for jj in range(ny):
        xp[jj, :] = (-x - a) * np.cos(theta[jj])
        yp[jj, :] = -(x + a) * np.sin(theta[jj])
    return xp, yp


def plot_poloidal(
    h5_path: str | Path,
    field: str,
    *,
    step: int | str | None = None,
    axes: str = "zxy",
    title: str | None = None,
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    output: str | Path | None = None,
    width_factor: float = 0.9,
    theta_window: float = 0.0,
) -> plt.Figure:
    data = load_gbs_field(h5_path, field, step=step, axes=axes)
    data2d = slice_2d(data[0], cut="pol", index=None)  # (y,x)

    text = read_gbs_stdin_text(h5_path) or ""
    params = parse_gbs_input_text(text)
    grid = infer_gbs_grid(params)
    if grid is None:
        Lx = 1.0
        Ly = 1.0
    else:
        Lx = grid.Lx
        Ly = grid.Ly

    xp, yp = _poloidal_coords(data2d.shape[0], data2d.shape[1], Lx=Lx, Ly=Ly, width_factor=width_factor, theta_window=theta_window)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    surf = ax.pcolormesh(xp, yp, data2d, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"{field} (poloidal)")
    fig.colorbar(surf, ax=ax, shrink=0.8)
    fig.tight_layout()
    if output is not None:
        fig.savefig(str(output), dpi=150)
    return fig
