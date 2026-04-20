from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from matplotlib import cm
from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np


@dataclass(frozen=True)
class TcvX21ToroidalMovieArtifacts:
    arrays_npz_path: Path
    summary_json_path: Path
    poster_png_path: Path
    movie_gif_path: Path


def create_tcv_x21_toroidal_movie_package(
    *,
    arrays_npz_path: str | Path,
    output_root: str | Path,
    case_label: str = "tokamak_tcv_x21_toroidal",
    major_radius: float = 1.35,
    minor_radius_min: float = 0.18,
    minor_radius_max: float = 0.48,
    elongation: float = 1.55,
    toroidal_samples: int = 48,
    radial_stride: int = 4,
    poloidal_stride: int = 4,
    interpolation_substeps: int = 4,
    fps: int = 8,
    toroidal_opening_degrees: float = 95.0,
) -> TcvX21ToroidalMovieArtifacts:
    source = Path(arrays_npz_path)
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    movies_dir = root / "movies"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    movies_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(source)
    field_history = np.asarray(payload["field_history"], dtype=np.float64)
    time_points = np.asarray(payload["time_points"], dtype=np.float64)
    field_name = str(np.asarray(payload["field_name"]).item())

    reduced_history = field_history[:, :: max(radial_stride, 1), :: max(poloidal_stride, 1)]
    finite_history = np.where(np.isfinite(reduced_history), reduced_history, 0.0)
    finite_counts = np.sum(np.isfinite(reduced_history), axis=0, keepdims=True)
    mean_history = np.divide(
        np.sum(finite_history, axis=0, keepdims=True),
        np.maximum(finite_counts, 1),
        dtype=np.float64,
    )
    fluctuations = np.nan_to_num(reduced_history - mean_history, nan=0.0, posinf=0.0, neginf=0.0)
    interpolated_history, interpolated_times = _interpolate_history(
        fluctuations,
        time_points,
        substeps=max(interpolation_substeps, 1),
    )
    shell_history = np.nanmean(interpolated_history[:, int(0.7 * interpolated_history.shape[1]) :, :], axis=1)
    shell_history = np.nan_to_num(shell_history, nan=0.0, posinf=0.0, neginf=0.0)

    theta = np.linspace(0.0, 2.0 * np.pi, interpolated_history.shape[2], endpoint=False)
    opening_radians = np.deg2rad(float(toroidal_opening_degrees))
    phi = np.linspace(
        0.5 * opening_radians,
        2.0 * np.pi - 0.5 * opening_radians,
        max(toroidal_samples, 12),
        endpoint=True,
    )
    radial = np.linspace(minor_radius_min, minor_radius_max, interpolated_history.shape[1], endpoint=True)
    color_limit = float(np.nanpercentile(np.abs(interpolated_history), 98.0))
    color_limit = max(color_limit, 1.0e-9)

    arrays_out = data_dir / f"{case_label}_arrays.npz"
    np.savez_compressed(
        arrays_out,
        time_points=np.asarray(interpolated_times, dtype=np.float64),
        fluctuation_history=np.asarray(interpolated_history, dtype=np.float64),
        shell_history=np.asarray(shell_history, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
        phi=np.asarray(phi, dtype=np.float64),
        radial=np.asarray(radial, dtype=np.float64),
        field_name=np.asarray(field_name),
    )

    summary = {
        "available": True,
        "parse_status": "ok",
        "case": case_label,
        "field_name": field_name,
        "frame_count": int(interpolated_history.shape[0]),
        "radial_points": int(interpolated_history.shape[1]),
        "poloidal_points": int(interpolated_history.shape[2]),
        "toroidal_samples": int(phi.size),
        "major_radius": float(major_radius),
        "minor_radius_range": [float(minor_radius_min), float(minor_radius_max)],
        "elongation": float(elongation),
        "toroidal_opening_degrees": float(toroidal_opening_degrees),
        "time_points": interpolated_times.tolist(),
        "color_limit": float(color_limit),
        "source_arrays": str(source.relative_to(source.parents[3])) if len(source.parents) >= 4 else source.name,
    }
    summary_json = data_dir / f"{case_label}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    poster_png = images_dir / f"{case_label}_poster.png"
    _save_frame(
        interpolated_history[-1],
        interpolated_times[-1],
        shell_history[-1],
        theta=theta,
        phi=phi,
        radial=radial,
        field_name=field_name,
        path=poster_png,
        color_limit=color_limit,
        major_radius=major_radius,
        elongation=elongation,
    )

    movie_gif = movies_dir / f"{case_label}.gif"
    _save_movie(
        interpolated_history,
        interpolated_times,
        shell_history,
        theta=theta,
        phi=phi,
        radial=radial,
        field_name=field_name,
        path=movie_gif,
        color_limit=color_limit,
        major_radius=major_radius,
        elongation=elongation,
        fps=fps,
    )

    return TcvX21ToroidalMovieArtifacts(
        arrays_npz_path=arrays_out,
        summary_json_path=summary_json,
        poster_png_path=poster_png,
        movie_gif_path=movie_gif,
    )


def _interpolate_history(history: np.ndarray, time_points: np.ndarray, *, substeps: int) -> tuple[np.ndarray, np.ndarray]:
    if history.shape[0] == 1 or substeps <= 1:
        return np.asarray(history, dtype=np.float64), np.asarray(time_points, dtype=np.float64)
    frames = []
    times = []
    for index in range(history.shape[0] - 1):
        start = history[index]
        end = history[index + 1]
        t0 = float(time_points[index])
        t1 = float(time_points[index + 1])
        for substep in range(substeps):
            weight = substep / float(substeps)
            frames.append((1.0 - weight) * start + weight * end)
            times.append((1.0 - weight) * t0 + weight * t1)
    frames.append(history[-1])
    times.append(float(time_points[-1]))
    return np.asarray(frames, dtype=np.float64), np.asarray(times, dtype=np.float64)


def _save_frame(
    plane: np.ndarray,
    time_point: float,
    shell_values: np.ndarray,
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    radial: np.ndarray,
    field_name: str,
    path: Path,
    color_limit: float,
    major_radius: float,
    elongation: float,
    azimuth: float = 42.0,
) -> None:
    figure, axis, colorbar = _prepare_figure_with_colorbar(
        field_name=field_name,
        color_limit=color_limit,
    )
    _draw_toroidal_frame(
        axis,
        plane,
        shell_values,
        theta=theta,
        phi=phi,
        radial=radial,
        field_name=field_name,
        time_point=time_point,
        color_limit=color_limit,
        major_radius=major_radius,
        elongation=elongation,
        azimuth=azimuth,
        colorbar=colorbar,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_movie(
    history: np.ndarray,
    time_points: np.ndarray,
    shell_history: np.ndarray,
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    radial: np.ndarray,
    field_name: str,
    path: Path,
    color_limit: float,
    major_radius: float,
    elongation: float,
    fps: int,
) -> None:
    figure, axis, colorbar = _prepare_figure_with_colorbar(
        field_name=field_name,
        color_limit=color_limit,
    )

    def _update(frame_index: int):
        axis.cla()
        _draw_toroidal_frame(
            axis,
            history[frame_index],
            shell_history[frame_index],
            theta=theta,
            phi=phi,
            radial=radial,
            field_name=field_name,
            time_point=float(time_points[frame_index]),
            color_limit=color_limit,
            major_radius=major_radius,
            elongation=elongation,
            azimuth=42.0 + 1.2 * frame_index,
            colorbar=colorbar,
        )
        return ()

    animation.FuncAnimation(
        figure,
        _update,
        frames=history.shape[0],
        interval=max(1, int(1000 / max(fps, 1))),
        blit=False,
    ).save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(figure)


def _prepare_figure_with_colorbar(*, field_name: str, color_limit: float):
    figure = plt.figure(figsize=(10.8, 7.6), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    scalar_mappable = cm.ScalarMappable(
        norm=plt.Normalize(vmin=-color_limit, vmax=color_limit),
        cmap=plt.get_cmap("coolwarm"),
    )
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(
        scalar_mappable,
        ax=axis,
        shrink=0.72,
        pad=0.06,
        fraction=0.05,
    )
    colorbar.set_label(f"{field_name} fluctuation amplitude")
    return figure, axis, colorbar


def _draw_toroidal_frame(
    axis,
    plane: np.ndarray,
    shell_values: np.ndarray,
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    radial: np.ndarray,
    field_name: str,
    time_point: float,
    color_limit: float,
    major_radius: float,
    elongation: float,
    azimuth: float,
    colorbar,
) -> None:
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-color_limit, vmax=color_limit)

    theta_grid, radial_grid = np.meshgrid(theta, radial, indexing="xy")
    x_plane_0 = major_radius + radial_grid * np.cos(theta_grid)
    y_plane_0 = np.zeros_like(x_plane_0)
    z_plane_0 = elongation * radial_grid * np.sin(theta_grid)

    x_plane_1 = np.zeros_like(x_plane_0)
    y_plane_1 = major_radius + radial_grid * np.cos(theta_grid)
    z_plane_1 = z_plane_0

    theta_shell, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    shell_radius = radial[-1]
    r_surface = major_radius + shell_radius * np.cos(theta_shell)
    x_shell = r_surface * np.cos(phi_grid)
    y_shell = r_surface * np.sin(phi_grid)
    z_shell = elongation * shell_radius * np.sin(theta_shell)
    shell_colors = np.repeat(shell_values[:, None], phi.size, axis=1)

    axis.plot_surface(
        x_shell,
        y_shell,
        z_shell,
        facecolors=cmap(norm(shell_colors)),
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.95,
    )
    axis.plot(
        x_shell[:, 0],
        y_shell[:, 0],
        z_shell[:, 0],
        color="black",
        linewidth=0.9,
        alpha=0.75,
    )
    axis.plot(
        x_shell[:, -1],
        y_shell[:, -1],
        z_shell[:, -1],
        color="black",
        linewidth=0.9,
        alpha=0.75,
    )
    axis.plot_surface(
        x_plane_0,
        y_plane_0,
        z_plane_0,
        facecolors=cmap(norm(plane)),
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.92,
    )
    axis.plot_surface(
        x_plane_1,
        y_plane_1,
        z_plane_1,
        facecolors=cmap(norm(plane)),
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.70,
    )

    lcfs_theta = np.linspace(0.0, 2.0 * np.pi, 240)
    lcfs_r = radial[-1]
    axis.plot(
        major_radius + lcfs_r * np.cos(lcfs_theta),
        np.zeros_like(lcfs_theta),
        elongation * lcfs_r * np.sin(lcfs_theta),
        color="black",
        linewidth=1.0,
        alpha=0.85,
    )
    axis.plot(
        np.zeros_like(lcfs_theta),
        major_radius + lcfs_r * np.cos(lcfs_theta),
        elongation * lcfs_r * np.sin(lcfs_theta),
        color="black",
        linewidth=1.0,
        alpha=0.65,
    )

    axis.set_title(f"Toroidal {field_name} fluctuation field · t = {time_point:.3f}")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    limit = major_radius + radial[-1] + 0.12
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-elongation * radial[-1] - 0.08, elongation * radial[-1] + 0.08)
    axis.set_box_aspect((1.0, 1.0, 0.6))
    axis.view_init(elev=24, azim=azimuth)
    axis.xaxis.pane.set_alpha(0.0)
    axis.yaxis.pane.set_alpha(0.0)
    axis.zaxis.pane.set_alpha(0.0)
    axis.grid(False)
    axis.text2D(
        0.03,
        0.96,
        "Cutaway toroidal shell with poloidal cross-sections showing interior turbulence.",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    colorbar.ax.set_title("cutaway", fontsize=8, pad=8)
