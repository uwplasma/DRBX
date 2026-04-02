from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .blob2d import (
    Blob2DAnalysisResult,
    Blob2DParityResult,
    analyze_blob2d_array_payload,
    compare_blob2d_artifacts,
    save_blob2d_parity_plot,
    write_blob2d_analysis_json,
    write_blob2d_parity_json,
)


@dataclass(frozen=True)
class Blob2DMeetingArtifacts:
    native_arrays_path: Path
    analysis_json_path: Path
    parity_json_path: Path
    snapshots_png_path: Path
    parity_png_path: Path
    poster_png_path: Path
    movie_2d_path: Path
    movie_3d_path: Path


def create_blob2d_meeting_package(
    payload: Mapping[str, Any],
    *,
    output_root: str | Path,
    native_arrays_path: str | Path | None = None,
    reference_metrics_path: str | Path,
    density_variable: str = "Ne",
    background_density: float = 1.0,
    case_label: str = "blob2d_short_window",
    fps: int = 10,
) -> Blob2DMeetingArtifacts:
    root = Path(output_root)
    images_dir = root / "images"
    movies_dir = root / "movies"
    data_dir = root / "data"
    images_dir.mkdir(parents=True, exist_ok=True)
    movies_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    arrays_path = Path(native_arrays_path) if native_arrays_path is not None else data_dir / f"{case_label}_native.npz"
    analysis_json_path = data_dir / f"{case_label}_analysis.json"
    parity_json_path = data_dir / f"{case_label}_parity.json"
    snapshots_png_path = images_dir / f"{case_label}_snapshots.png"
    parity_png_path = images_dir / f"{case_label}_parity.png"
    poster_png_path = images_dir / f"{case_label}_movie_poster.png"
    movie_2d_path = movies_dir / f"{case_label}_2d.mp4"
    movie_3d_path = movies_dir / f"{case_label}_3d.mp4"

    analysis = analyze_blob2d_array_payload(
        payload,
        density_variable=density_variable,
        background_density=background_density,
    )
    write_blob2d_analysis_json(analysis, analysis_json_path)

    parity = compare_blob2d_artifacts(
        reference_metrics_path,
        arrays_path if arrays_path.exists() else _write_temporary_array_payload(payload, arrays_path),
        density_variable=density_variable,
        background_density=background_density,
    )
    write_blob2d_parity_json(parity, parity_json_path)
    save_blob2d_parity_plot(parity, parity_png_path)

    save_blob2d_snapshot_panel(
        payload,
        analysis=analysis,
        path=snapshots_png_path,
        density_variable=density_variable,
        background_density=background_density,
    )
    save_blob2d_poster_frame(
        payload,
        analysis=analysis,
        path=poster_png_path,
        density_variable=density_variable,
        background_density=background_density,
    )
    save_blob2d_heatmap_movie(
        payload,
        analysis=analysis,
        path=movie_2d_path,
        density_variable=density_variable,
        background_density=background_density,
        fps=fps,
    )
    save_blob2d_surface_movie(
        payload,
        analysis=analysis,
        path=movie_3d_path,
        density_variable=density_variable,
        background_density=background_density,
        fps=fps,
    )

    return Blob2DMeetingArtifacts(
        native_arrays_path=arrays_path,
        analysis_json_path=analysis_json_path,
        parity_json_path=parity_json_path,
        snapshots_png_path=snapshots_png_path,
        parity_png_path=parity_png_path,
        poster_png_path=poster_png_path,
        movie_2d_path=movie_2d_path,
        movie_3d_path=movie_3d_path,
    )


def save_blob2d_snapshot_panel(
    payload: Mapping[str, Any],
    *,
    analysis: Blob2DAnalysisResult,
    path: str | Path,
    density_variable: str = "Ne",
    background_density: float = 1.0,
) -> Path:
    plt = _plt()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    density_excess, _, x_coords, z_coords = _blob2d_density_excess(payload, density_variable, background_density)
    frame_indices = _representative_frame_indices(density_excess.shape[0])
    color_limit = float(np.max(np.abs(density_excess)))

    figure, axes = plt.subplots(1, len(frame_indices), figsize=(12.0, 4.2), constrained_layout=True)
    if len(frame_indices) == 1:
        axes = [axes]

    for axis, frame_index, label in zip(axes, frame_indices, ("Initial", "Mid", "Final"), strict=True):
        image = axis.imshow(
            density_excess[frame_index].T,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=color_limit,
            extent=(x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]),
            aspect="auto",
        )
        axis.scatter(
            analysis.center_of_mass_x_history[frame_index],
            analysis.center_of_mass_z_history[frame_index],
            s=32,
            marker="o",
            color="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        axis.set_title(f"{label}\n$t={analysis.time_points[frame_index]:.4g}$")
        axis.set_xlabel("x index")
        axis.set_ylabel("z index")

    colorbar = figure.colorbar(image, ax=axes, shrink=0.92, pad=0.02)
    colorbar.set_label("Density excess")
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def save_blob2d_poster_frame(
    payload: Mapping[str, Any],
    *,
    analysis: Blob2DAnalysisResult,
    path: str | Path,
    density_variable: str = "Ne",
    background_density: float = 1.0,
) -> Path:
    plt = _plt()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    density_excess, _, x_coords, z_coords = _blob2d_density_excess(payload, density_variable, background_density)
    frame_index = density_excess.shape[0] - 1
    xx, zz = np.meshgrid(x_coords, z_coords, indexing="ij")
    color_limit = float(np.max(np.abs(density_excess)))

    figure = plt.figure(figsize=(7.5, 5.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        xx,
        zz,
        density_excess[frame_index],
        cmap="magma",
        linewidth=0.0,
        antialiased=True,
        vmin=0.0,
        vmax=color_limit,
    )
    axis.view_init(elev=30.0, azim=-55.0)
    axis.set_title(f"Blob2D final frame\n$t={analysis.time_points[frame_index]:.4g}$")
    axis.set_xlabel("x index")
    axis.set_ylabel("z index")
    axis.set_zlabel("Density excess")
    figure.colorbar(surface, ax=axis, shrink=0.72, pad=0.08, label="Density excess")
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def save_blob2d_heatmap_movie(
    payload: Mapping[str, Any],
    *,
    analysis: Blob2DAnalysisResult,
    path: str | Path,
    density_variable: str = "Ne",
    background_density: float = 1.0,
    fps: int = 10,
) -> Path:
    plt = _plt()
    animation = _animation()
    writer = _ffmpeg_writer(fps=fps)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    density_excess, time_points, x_coords, z_coords = _blob2d_density_excess(payload, density_variable, background_density)
    color_limit = float(np.max(np.abs(density_excess)))

    figure, axis = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    image = axis.imshow(
        density_excess[0].T,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=color_limit,
        extent=(x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]),
        aspect="auto",
    )
    marker = axis.scatter(
        analysis.center_of_mass_x_history[0],
        analysis.center_of_mass_z_history[0],
        s=30,
        marker="o",
        color="white",
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    title = axis.set_title("")
    axis.set_xlabel("x index")
    axis.set_ylabel("z index")
    figure.colorbar(image, ax=axis, shrink=0.88, pad=0.02, label="Density excess")

    def update(frame_index: int):
        image.set_data(density_excess[frame_index].T)
        marker.set_offsets(
            np.array(
                [[analysis.center_of_mass_x_history[frame_index], analysis.center_of_mass_z_history[frame_index]]],
                dtype=np.float64,
            )
        )
        title.set_text(
            f"Blob2D density excess\n"
            f"t={time_points[frame_index]:.4g}, peak={analysis.peak_excess_history[frame_index]:.4g}"
        )
        return image, marker, title

    movie = animation.FuncAnimation(figure, update, frames=density_excess.shape[0], interval=1000 / max(fps, 1), blit=False)
    movie.save(target, writer=writer, dpi=160)
    plt.close(figure)
    return target


def save_blob2d_surface_movie(
    payload: Mapping[str, Any],
    *,
    analysis: Blob2DAnalysisResult,
    path: str | Path,
    density_variable: str = "Ne",
    background_density: float = 1.0,
    fps: int = 10,
) -> Path:
    plt = _plt()
    animation = _animation()
    writer = _ffmpeg_writer(fps=fps)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    density_excess, time_points, x_coords, z_coords = _blob2d_density_excess(payload, density_variable, background_density)
    xx, zz = np.meshgrid(x_coords, z_coords, indexing="ij")
    color_limit = float(np.max(np.abs(density_excess)))

    figure = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")

    def draw(frame_index: int):
        axis.clear()
        surface = axis.plot_surface(
            xx,
            zz,
            density_excess[frame_index],
            cmap="magma",
            linewidth=0.0,
            antialiased=True,
            vmin=0.0,
            vmax=color_limit,
        )
        axis.view_init(elev=30.0, azim=-55.0)
        axis.set_zlim(0.0, color_limit if color_limit > 0.0 else 1.0)
        axis.set_xlabel("x index")
        axis.set_ylabel("z index")
        axis.set_zlabel("Density excess")
        axis.set_title(
            f"Blob2D density surface\n"
            f"t={time_points[frame_index]:.4g}, peak={analysis.peak_excess_history[frame_index]:.4g}"
        )
        return (surface,)

    movie = animation.FuncAnimation(figure, draw, frames=density_excess.shape[0], interval=1000 / max(fps, 1), blit=False)
    movie.save(target, writer=writer, dpi=160)
    plt.close(figure)
    return target


def _blob2d_density_excess(
    payload: Mapping[str, Any],
    density_variable: str,
    background_density: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    variables = payload.get("variables", {})
    if density_variable not in variables:
        available = ", ".join(sorted(variables))
        raise KeyError(f"Missing density variable {density_variable!r}. Available variables: {available}")
    density_history = np.asarray(variables[density_variable], dtype=np.float64)
    if density_history.ndim != 4 or density_history.shape[2] != 1:
        raise ValueError("Blob2D meeting visuals require a (t, x, 1, z) density history.")

    density_excess = np.maximum(density_history[:, :, 0, :] - background_density, 0.0)
    time_points = np.asarray(payload.get("time_points", []), dtype=np.float64)
    if time_points.size != density_excess.shape[0]:
        raise ValueError("Blob2D meeting visuals require one time point per stored output.")
    x_coords = np.arange(density_excess.shape[1], dtype=np.float64)
    z_coords = np.arange(density_excess.shape[2], dtype=np.float64)
    return density_excess, time_points, x_coords, z_coords


def _representative_frame_indices(frame_count: int) -> tuple[int, int, int]:
    if frame_count < 3:
        return (0, max(frame_count - 1, 0), max(frame_count - 1, 0))
    return (0, frame_count // 2, frame_count - 1)


def _write_temporary_array_payload(payload: Mapping[str, Any], target: Path) -> Path:
    from ..parity.arrays import write_portable_array_payload

    write_portable_array_payload(payload, target)
    return target


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
    return plt


def _animation():
    import matplotlib.animation as animation

    return animation


def _ffmpeg_writer(*, fps: int):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write Blob2D meeting movies.")
    animation = _animation()
    return animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=2200)
