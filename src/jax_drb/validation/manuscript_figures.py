from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import image as mpimg
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ManuscriptFigureArtifacts:
    manifest_json_path: Path
    architecture_png_path: Path
    equations_geometry_png_path: Path
    transient_panel_png_path: Path


def create_manuscript_figure_package(
    *,
    output_root: str | Path,
    case_label: str = "manuscript_figures",
    neutral_diagnostics_png: str | Path | None = None,
) -> ManuscriptFigureArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    neutral_png = _resolve_or_default(
        neutral_diagnostics_png,
        "docs/images/neutral_mixed_short_window_diagnostics.png",
    )
    architecture_png_path = images_dir / f"{case_label}_architecture.png"
    equations_geometry_png_path = images_dir / f"{case_label}_equations_geometry.png"
    transient_panel_png_path = images_dir / f"{case_label}_transient_panel.png"
    _save_architecture_schematic(architecture_png_path)
    _save_equations_geometry_schematic(equations_geometry_png_path)
    _save_transient_validation_panel(transient_panel_png_path, neutral_diagnostics_png=neutral_png)

    manifest = {
        "case": "manuscript_figures",
        "figures": [
            {
                "name": "architecture_validation_ladder",
                "path": str(architecture_png_path.relative_to(root)),
                "purpose": "Solver architecture, parity ladder, and selected-lane claim boundary.",
            },
            {
                "name": "equations_geometry_summary",
                "path": str(equations_geometry_png_path.relative_to(root)),
                "purpose": "Governing model blocks, closures, and supported geometry families.",
            },
            {
                "name": "transient_validation_panel",
                "path": str(transient_panel_png_path.relative_to(root)),
                "purpose": "Neutral short-window diagnostics plus direct tokamak recycling-window summary.",
            },
        ],
    }
    manifest_json_path = data_dir / f"{case_label}_manifest.json"
    manifest_json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return ManuscriptFigureArtifacts(
        manifest_json_path=manifest_json_path,
        architecture_png_path=architecture_png_path,
        equations_geometry_png_path=equations_geometry_png_path,
        transient_panel_png_path=transient_panel_png_path,
    )


def _save_architecture_schematic(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.5, 7.0), constrained_layout=True)
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="#1b1b1b",
            facecolor=color,
        )
        axis.add_patch(rect)
        axis.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=11, wrap=True)

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.8, color="#1b1b1b"))

    box(0.05, 0.72, 0.22, 0.16, "Curated inputs\nBOUT.inp, grids,\nreference cases", "#dbe9f6")
    box(0.38, 0.72, 0.22, 0.16, "Config + runtime layer\nCLI, Python driver,\nrestart, provenance", "#d9f0d3")
    box(0.71, 0.72, 0.22, 0.16, "Native kernels\noperators, steppers,\ngeometry, diagnostics", "#fde0c5")
    arrow((0.27, 0.80), (0.38, 0.80))
    arrow((0.60, 0.80), (0.71, 0.80))

    box(0.08, 0.42, 0.17, 0.12, "Unit/operator\ntests", "#f3e5f5")
    box(0.29, 0.42, 0.17, 0.12, "MMS +\nconvergence", "#f3e5f5")
    box(0.50, 0.42, 0.17, 0.12, "Reference parity\none-RHS / one-step /\nshort-window", "#f3e5f5")
    box(0.71, 0.42, 0.17, 0.12, "Physics-facing\nbenchmarks +\nreview figures", "#f3e5f5")
    arrow((0.25, 0.48), (0.29, 0.48))
    arrow((0.46, 0.48), (0.50, 0.48))
    arrow((0.67, 0.48), (0.71, 0.48))

    box(0.10, 0.12, 0.33, 0.14, "Selected-lane claim boundary\npromoted native 1D/2D lanes + general 3D infrastructure", "#d9f0d3")
    box(0.57, 0.12, 0.28, 0.14, "Explicit non-claim boundary\nnot a broad parity-complete standalone replacement", "#fde0c5")
    axis.text(0.50, 0.95, "jax_drb manuscript architecture and validation ladder", ha="center", va="center", fontsize=15, weight="bold")
    axis.text(0.50, 0.63, "Verification and validation spine", ha="center", va="center", fontsize=13, weight="bold")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _save_equations_geometry_schematic(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.5, 7.5), constrained_layout=True)
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.4,
            edgecolor="#1b1b1b",
            facecolor=color,
        )
        axis.add_patch(rect)
        axis.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=10.5, wrap=True)

    axis.text(0.5, 0.95, "Governing model and supported geometry families", ha="center", va="center", fontsize=15, weight="bold")
    box(0.05, 0.70, 0.25, 0.16, "Core DRB state\ncontinuity, parallel momentum,\npressure/energy,\nphi / vorticity, selected Apar", "#dbe9f6")
    box(0.37, 0.70, 0.25, 0.16, "Closures and sources\nsheath, recycling,\nreactions, radiation,\nanomalous transport", "#d9f0d3")
    box(0.69, 0.70, 0.25, 0.16, "Numerical realization\nFV operators, guard handling,\nimplicit or matrix-free transients,\nrestart + provenance", "#fde0c5")

    box(0.08, 0.38, 0.18, 0.14, "1D open-field\nand recycling", "#f3e5f5")
    box(0.31, 0.38, 0.18, 0.14, "2D slab / blob /\ndrift-wave", "#f3e5f5")
    box(0.54, 0.38, 0.18, 0.14, "2D diverted tokamak\ntransport / turbulence /\nrecycling", "#f3e5f5")
    box(0.77, 0.38, 0.18, 0.14, "3D adapters\ntokamak, traced-field-line,\nVMEC stellarator", "#f3e5f5")

    axis.text(0.18, 0.30, "Promoted native lanes", ha="center", fontsize=11, weight="bold")
    axis.text(0.86, 0.30, "Generalized infrastructure", ha="center", fontsize=11, weight="bold")

    box(0.10, 0.10, 0.35, 0.12, "Differentiable lane\nPython-driven native kernels, sensitivity, UQ,\ninverse design, scaling", "#e8f1ff")
    box(0.55, 0.10, 0.30, 0.12, "CLI/runtime lane\nSciPy/NumPy allowed where useful,\nno differentiability requirement", "#fff2cc")

    figure.savefig(path, dpi=220)
    plt.close(figure)


def _save_transient_validation_panel(path: Path, *, neutral_diagnostics_png: Path) -> None:
    figure = plt.figure(figsize=(15.5, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    left = figure.add_subplot(grid[0, 0])
    right = figure.add_subplot(grid[0, 1])

    neutral_image = mpimg.imread(neutral_diagnostics_png)
    left.imshow(neutral_image)
    left.set_title("Neutral mixed short-window diagnostics")
    left.axis("off")

    windows = ["D/T one-step", "D/T nout=2", "D/T/He/Ne nout=3", "D/T/He/Ne nout=5"]
    worst_relative = np.asarray([5.0e-2, 2.74e-2, 3.44e-3, 2.08e-3], dtype=np.float64)
    worst_near_zero = np.asarray([2.0e-5, 2.67e-5, 2.54e-4, 4.29e-4], dtype=np.float64)
    x = np.arange(len(windows))
    width = 0.36
    right.bar(x - width / 2.0, worst_relative, width=width, color="#005f73", label="worst scaled diff")
    right.bar(x + width / 2.0, worst_near_zero, width=width, color="#ca6702", label="worst near-zero |Δ|")
    right.set_xticks(x, windows, rotation=15, ha="right")
    right.set_ylabel("bounded parity metric")
    right.set_title("Direct tokamak recycling bounded windows")
    right.grid(axis="y", alpha=0.25)
    right.legend(frameon=False)
    right.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
    right.text(
        0.02,
        0.96,
        "Values summarize the current\nbounded live Hermes-backed windows\nrecorded in the parity harness docs.",
        transform=right.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#999999"),
    )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _resolve_or_default(path: str | Path | None, default_relative: str) -> Path:
    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parents[3] / default_relative
