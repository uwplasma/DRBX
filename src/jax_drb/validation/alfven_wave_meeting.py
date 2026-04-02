from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config.boutinp import load_bout_input
from ..native.units import resolved_dataset_scalars
from ..runtime.run_config import RunConfiguration
from .alfven_wave import (
    AlfvenWaveAnalysisResult,
    analyze_alfven_wave_array_payload,
    compare_alfven_wave_npz,
    save_alfven_wave_diagnostic_plot,
    save_alfven_wave_parity_plot,
    write_alfven_wave_analysis_json,
    write_alfven_wave_parity_json,
)


@dataclass(frozen=True)
class AlfvenWaveMeetingArtifacts:
    native_arrays_path: Path
    analysis_json_path: Path
    parity_json_path: Path
    snapshots_png_path: Path
    diagnostics_png_path: Path
    parity_png_path: Path
    poster_png_path: Path
    movie_2d_path: Path
    movie_3d_path: Path


def create_alfven_wave_meeting_package(
    payload: Mapping[str, Any],
    *,
    input_file: str | Path,
    expected_arrays_path: str | Path,
    native_arrays_path: str | Path,
    output_root: str | Path,
    field_variable: str = "phi",
    x_index: int = 2,
    case_label: str = "alfven_wave_meeting",
    fps: int = 10,
) -> AlfvenWaveMeetingArtifacts:
    root = Path(output_root)
    images_dir = root / "images"
    movies_dir = root / "movies"
    data_dir = root / "data"
    images_dir.mkdir(parents=True, exist_ok=True)
    movies_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    config = load_bout_input(input_file)
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    analysis = analyze_alfven_wave_array_payload(
        payload,
        config=config,
        dataset_scalars=dataset_scalars,
        field_variable=field_variable,
        x_index=x_index,
    )
    parity = compare_alfven_wave_npz(
        expected_arrays_path,
        native_arrays_path,
        input_file=input_file,
        field_variable=field_variable,
        x_index=x_index,
    )

    analysis_json_path = data_dir / f"{case_label}_analysis.json"
    parity_json_path = data_dir / f"{case_label}_parity.json"
    snapshots_png_path = images_dir / f"{case_label}_snapshots.png"
    diagnostics_png_path = images_dir / f"{case_label}_diagnostics.png"
    parity_png_path = images_dir / f"{case_label}_parity.png"
    poster_png_path = images_dir / f"{case_label}_movie_poster.png"
    movie_2d_path = movies_dir / f"{case_label}_2d.mp4"
    movie_3d_path = movies_dir / f"{case_label}_3d.mp4"

    write_alfven_wave_analysis_json(analysis, analysis_json_path)
    write_alfven_wave_parity_json(parity, parity_json_path)
    save_alfven_wave_diagnostic_plot(analysis, diagnostics_png_path)
    save_alfven_wave_parity_plot(parity, parity_png_path)
    save_alfven_wave_snapshot_panel(payload, path=snapshots_png_path, field_variable=field_variable, x_index=x_index)
    save_alfven_wave_poster_frame(payload, path=poster_png_path, field_variable=field_variable, x_index=x_index)
    save_alfven_wave_heatmap_movie(payload, path=movie_2d_path, field_variable=field_variable, x_index=x_index, fps=fps)
    save_alfven_wave_surface_movie(payload, path=movie_3d_path, field_variable=field_variable, x_index=x_index, fps=fps)

    return AlfvenWaveMeetingArtifacts(
        native_arrays_path=Path(native_arrays_path),
        analysis_json_path=analysis_json_path,
        parity_json_path=parity_json_path,
        snapshots_png_path=snapshots_png_path,
        diagnostics_png_path=diagnostics_png_path,
        parity_png_path=parity_png_path,
        poster_png_path=poster_png_path,
        movie_2d_path=movie_2d_path,
        movie_3d_path=movie_3d_path,
    )


def save_alfven_wave_snapshot_panel(
    payload: Mapping[str, Any],
    *,
    path: str | Path,
    field_variable: str = "phi",
    x_index: int = 2,
) -> Path:
    plt = _plt()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    plane_history, time_points, y_coords, z_coords = _plane_history(payload, field_variable=field_variable, x_index=x_index)
    frame_indices = _frame_indices(plane_history.shape[0])
    color_limit = float(np.max(np.abs(plane_history)))

    figure, axes = plt.subplots(1, len(frame_indices), figsize=(12.0, 4.2), constrained_layout=True)
    if len(frame_indices) == 1:
        axes = [axes]
    for axis, frame_index, label in zip(axes, frame_indices, ("Initial", "Mid", "Final"), strict=True):
        image = axis.imshow(
            plane_history[frame_index].T,
            origin="lower",
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
            extent=(y_coords[0], y_coords[-1], z_coords[0], z_coords[-1]),
            aspect="auto",
        )
        axis.set_title(f"{label}\n$t={time_points[frame_index]:.4g}$")
        axis.set_xlabel("y index")
        axis.set_ylabel("z index")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.92, pad=0.02)
    colorbar.set_label(field_variable)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def save_alfven_wave_poster_frame(
    payload: Mapping[str, Any],
    *,
    path: str | Path,
    field_variable: str = "phi",
    x_index: int = 2,
) -> Path:
    plt = _plt()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    plane_history, time_points, y_coords, z_coords = _plane_history(payload, field_variable=field_variable, x_index=x_index)
    yy, zz = np.meshgrid(y_coords, z_coords, indexing="ij")
    frame_index = plane_history.shape[0] - 1
    color_limit = float(np.max(np.abs(plane_history)))

    figure = plt.figure(figsize=(7.5, 5.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        yy,
        zz,
        plane_history[frame_index],
        cmap="coolwarm",
        linewidth=0.0,
        antialiased=True,
        vmin=-color_limit,
        vmax=color_limit,
    )
    axis.view_init(elev=28.0, azim=-50.0)
    axis.set_title(f"Alfven-wave final frame\n$t={time_points[frame_index]:.4g}$")
    axis.set_xlabel("y index")
    axis.set_ylabel("z index")
    axis.set_zlabel(field_variable)
    figure.colorbar(surface, ax=axis, shrink=0.72, pad=0.08, label=field_variable)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def save_alfven_wave_heatmap_movie(
    payload: Mapping[str, Any],
    *,
    path: str | Path,
    field_variable: str = "phi",
    x_index: int = 2,
    fps: int = 10,
) -> Path:
    plt = _plt()
    animation = _animation()
    writer = _ffmpeg_writer(fps=fps)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    plane_history, time_points, y_coords, z_coords = _plane_history(payload, field_variable=field_variable, x_index=x_index)
    color_limit = float(np.max(np.abs(plane_history)))

    figure, axis = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    image = axis.imshow(
        plane_history[0].T,
        origin="lower",
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
        extent=(y_coords[0], y_coords[-1], z_coords[0], z_coords[-1]),
        aspect="auto",
    )
    title = axis.set_title("")
    axis.set_xlabel("y index")
    axis.set_ylabel("z index")
    figure.colorbar(image, ax=axis, shrink=0.88, pad=0.02, label=field_variable)

    def update(frame_index: int):
        image.set_data(plane_history[frame_index].T)
        title.set_text(f"Alfven-wave {field_variable}\nt={time_points[frame_index]:.4g}")
        return image, title

    movie = animation.FuncAnimation(figure, update, frames=plane_history.shape[0], interval=1000 / max(fps, 1), blit=False)
    movie.save(target, writer=writer, dpi=160)
    plt.close(figure)
    return target


def save_alfven_wave_surface_movie(
    payload: Mapping[str, Any],
    *,
    path: str | Path,
    field_variable: str = "phi",
    x_index: int = 2,
    fps: int = 10,
) -> Path:
    plt = _plt()
    animation = _animation()
    writer = _ffmpeg_writer(fps=fps)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    plane_history, time_points, y_coords, z_coords = _plane_history(payload, field_variable=field_variable, x_index=x_index)
    yy, zz = np.meshgrid(y_coords, z_coords, indexing="ij")
    color_limit = float(np.max(np.abs(plane_history)))

    figure = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")

    def draw(frame_index: int):
        axis.clear()
        surface = axis.plot_surface(
            yy,
            zz,
            plane_history[frame_index],
            cmap="coolwarm",
            linewidth=0.0,
            antialiased=True,
            vmin=-color_limit,
            vmax=color_limit,
        )
        axis.view_init(elev=28.0, azim=-50.0)
        axis.set_zlim(-color_limit, color_limit if color_limit > 0.0 else 1.0)
        axis.set_xlabel("y index")
        axis.set_ylabel("z index")
        axis.set_zlabel(field_variable)
        axis.set_title(f"Alfven-wave {field_variable} surface\nt={time_points[frame_index]:.4g}")
        return (surface,)

    movie = animation.FuncAnimation(figure, draw, frames=plane_history.shape[0], interval=1000 / max(fps, 1), blit=False)
    movie.save(target, writer=writer, dpi=160)
    plt.close(figure)
    return target


def _plane_history(
    payload: Mapping[str, Any],
    *,
    field_variable: str,
    x_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    variables = payload.get("variables", {})
    if field_variable not in variables:
        available = ", ".join(sorted(variables))
        raise KeyError(f"Missing field variable {field_variable!r}. Available variables: {available}")
    history = np.asarray(variables[field_variable], dtype=np.float64)
    if history.ndim != 4:
        raise ValueError(f"{field_variable} must have shape (t, x, y, z).")
    if not 0 <= x_index < history.shape[1]:
        raise IndexError(f"x_index {x_index} is out of bounds for shape {history.shape}.")
    plane_history = history[:, x_index, :, :]
    time_points = np.asarray(payload.get("time_points", []), dtype=np.float64)
    if time_points.size != plane_history.shape[0]:
        raise ValueError("Need one time point per stored frame.")
    y_coords = np.arange(plane_history.shape[1], dtype=np.float64)
    z_coords = np.arange(plane_history.shape[2], dtype=np.float64)
    return plane_history, time_points, y_coords, z_coords


def _frame_indices(frame_count: int) -> tuple[int, int, int]:
    if frame_count < 3:
        return (0, max(frame_count - 1, 0), max(frame_count - 1, 0))
    return (0, frame_count // 2, frame_count - 1)


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
        raise RuntimeError("ffmpeg is required to write Alfven-wave meeting movies.")
    animation = _animation()
    return animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=2200)
