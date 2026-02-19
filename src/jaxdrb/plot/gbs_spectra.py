from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional
    jax = None
    jnp = None

from .gbs_io import load_gbs_field, infer_gbs_grid, parse_gbs_input_text, read_gbs_stdin_text


def _fft_backend(name: str):
    if name == "jax" and jnp is not None:
        return jnp
    return np


def power_spectrum_1d(
    data: np.ndarray,
    *,
    axis: int,
    length: float,
    backend: Literal["numpy", "jax"] = "numpy",
) -> tuple[np.ndarray, np.ndarray]:
    xp = _fft_backend(backend)
    arr = xp.asarray(data)
    spec = xp.abs(xp.fft.rfft(arr, axis=axis)) ** 2
    # average over other axes
    axes = tuple(i for i in range(spec.ndim) if i != axis)
    spec = spec.mean(axis=axes)
    n = arr.shape[axis]
    k = 2.0 * np.pi * np.fft.rfftfreq(n, d=length / n)
    spec = np.asarray(spec)
    return k, spec


def plot_power_spectrum(
    h5_path: str | Path,
    field: str,
    *,
    step: int | str | None = None,
    axis: Literal["x", "y", "z"] = "y",
    axes: str = "zxy",
    backend: Literal["numpy", "jax"] = "numpy",
    output: str | Path | None = None,
) -> plt.Figure:
    data = load_gbs_field(h5_path, field, step=step, axes=axes)
    snapshot = data[0]  # z,x,y

    text = read_gbs_stdin_text(h5_path) or ""
    params = parse_gbs_input_text(text)
    grid = infer_gbs_grid(params)
    if grid is None:
        Lx = Ly = Lz = 1.0
    else:
        Lx, Ly, Lz = grid.Lx, grid.Ly, grid.Lz

    axis_map = {"z": 0, "x": 1, "y": 2}
    if axis not in axis_map:
        raise ValueError("axis must be x, y, or z")
    ax = axis_map[axis]
    length = {"x": Lx, "y": Ly, "z": Lz}[axis]

    k, spec = power_spectrum_1d(snapshot, axis=ax, length=length, backend=backend)

    fig, axp = plt.subplots(figsize=(5, 4), dpi=150)
    axp.loglog(k[1:], spec[1:] + 1e-30)
    axp.set_xlabel(f"k{axis}")
    axp.set_ylabel(f"FFT({field})")
    axp.set_title(f"Power spectrum {field} vs k{axis}")
    fig.tight_layout()
    if output is not None:
        fig.savefig(str(output), dpi=150)
    return fig
