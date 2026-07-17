from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np


PUBLICATION_DPI = 320


def style_axis(
    axis,
    *,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    yscale: str | None = None,
    xscale: str | None = None,
    grid: str = "y",
) -> None:
    axis.set_title(title, fontsize=13.5, fontweight="semibold", pad=8.0)
    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=12.0)
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=12.0)
    if yscale is not None:
        axis.set_yscale(yscale)
    if xscale is not None:
        axis.set_xscale(xscale)
    axis.grid(alpha=0.22, axis=grid, linewidth=0.8)
    axis.tick_params(labelsize=11.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def annotate_bars(
    axis,
    x: np.ndarray,
    values: np.ndarray,
    *,
    fmt: str = "{:.2e}",
    fontsize: float = 9.0,
    rotation: float = 0.0,
) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return
    positive = np.abs(values[np.nonzero(values)])
    scale = float(np.max(positive)) if positive.size else 1.0
    offset = 0.03 * scale
    for xi, value in zip(np.asarray(x, dtype=np.float64), values, strict=True):
        va = "bottom" if value >= 0.0 else "top"
        delta = offset if value >= 0.0 else -offset
        axis.text(
            float(xi),
            float(value + delta),
            fmt.format(float(value)),
            ha="center",
            va=va,
            fontsize=fontsize,
            rotation=rotation,
        )


def support_window_slice(
    *arrays: np.ndarray,
    padding: int = 4,
    threshold: float = 1.0e-12,
) -> slice:
    support = np.zeros_like(np.asarray(arrays[0], dtype=np.float64), dtype=bool)
    for array in arrays:
        support |= np.abs(np.asarray(array, dtype=np.float64)) > threshold
    indices = np.flatnonzero(support)
    if indices.size == 0:
        return slice(0, support.size)
    start = max(0, int(indices[0]) - padding)
    stop = min(support.size, int(indices[-1]) + padding + 1)
    return slice(start, stop)


def save_publication_figure(figure, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=PUBLICATION_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
