from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

from .gbs_io import load_gbs_field, list_gbs_steps, infer_gbs_grid, parse_gbs_input_text, read_gbs_stdin_text
from .gbs_plot import slice_2d, _poloidal_coords


def _get_writer(output: Path, fps: int = 15):
    if output.suffix.lower() == ".mp4" and FFMpegWriter.isAvailable():
        return FFMpegWriter(fps=fps)
    if output.suffix.lower() not in (".gif", ".mp4"):
        output = output.with_suffix(".gif")
    return PillowWriter(fps=fps)


def make_movie_rect(
    h5_path: str | Path,
    field: str,
    *,
    steps: Iterable[int | str] | None = None,
    cut: Literal["pol", "tor", "rad"] = "pol",
    index: int | None = None,
    axes: str = "zxy",
    output: str | Path = "movie.gif",
    fps: int = 15,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "jet",
) -> Path:
    h5_path = Path(h5_path)
    output = Path(output)
    if steps is None:
        steps = list_gbs_steps(h5_path, var="theta")
    steps = list(steps)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    writer = _get_writer(output, fps=fps)

    with writer.saving(fig, str(output), dpi=150):
        for step in steps:
            data = load_gbs_field(h5_path, field, step=step, axes=axes)
            frame = slice_2d(data[0], cut=cut, index=index)
            ax.clear()
            im = ax.imshow(frame, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{field} {cut} step {step}")
            fig.colorbar(im, ax=ax, shrink=0.8)
            writer.grab_frame()
    plt.close(fig)
    return output


def make_movie_poloidal(
    h5_path: str | Path,
    field: str,
    *,
    steps: Iterable[int | str] | None = None,
    axes: str = "zxy",
    output: str | Path = "movie_poloidal.gif",
    fps: int = 15,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "jet",
    width_factor: float = 0.9,
    theta_window: float = 0.0,
) -> Path:
    h5_path = Path(h5_path)
    output = Path(output)
    if steps is None:
        steps = list_gbs_steps(h5_path, var="theta")
    steps = list(steps)

    text = read_gbs_stdin_text(h5_path) or ""
    params = parse_gbs_input_text(text)
    grid = infer_gbs_grid(params)
    Lx = grid.Lx if grid is not None else 1.0
    Ly = grid.Ly if grid is not None else 1.0

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    writer = _get_writer(output, fps=fps)

    with writer.saving(fig, str(output), dpi=150):
        for step in steps:
            data = load_gbs_field(h5_path, field, step=step, axes=axes)
            frame = slice_2d(data[0], cut="pol", index=None)
            xp, yp = _poloidal_coords(frame.shape[0], frame.shape[1], Lx=Lx, Ly=Ly, width_factor=width_factor, theta_window=theta_window)
            ax.clear()
            surf = ax.pcolormesh(xp, yp, frame, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{field} poloidal step {step}")
            fig.colorbar(surf, ax=ax, shrink=0.8)
            writer.grab_frame()
    plt.close(fig)
    return output
