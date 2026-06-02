from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from netCDF4 import Dataset


@dataclass(frozen=True)
class DivertedTokamakGeometry:
    rxy: np.ndarray
    zxy: np.ndarray
    psixy: np.ndarray
    wall_r: np.ndarray
    wall_z: np.ndarray
    lower_target_r: np.ndarray
    lower_target_z: np.ndarray
    upper_target_r: np.ndarray
    upper_target_z: np.ndarray


@dataclass(frozen=True)
class DivertedTokamakFieldHistory:
    field_name: str
    time_points: np.ndarray
    history_4d: np.ndarray


@dataclass(frozen=True)
class DivertedTokamakMovieArtifacts:
    arrays_npz_path: Path
    analysis_json_path: Path
    snapshots_png_path: Path
    poster_png_path: Path
    movie_gif_path: Path


def assemble_tokamak_rank_history(
    workdir: str | Path, *, field_name: str
) -> DivertedTokamakFieldHistory:
    root = Path(workdir)
    dump_paths = sorted(root.glob("BOUT.dmp.*.nc"))
    if not dump_paths:
        raise FileNotFoundError(f"No BOUT dumps found in {root}")

    with Dataset(dump_paths[0]) as dataset:
        mxsub = int(np.asarray(dataset.variables["MXSUB"][:]).item())
        mysub = int(np.asarray(dataset.variables["MYSUB"][:]).item())
        mxg = int(np.asarray(dataset.variables["MXG"][:]).item())
        myg = int(np.asarray(dataset.variables["MYG"][:]).item())
        nxpe = int(np.asarray(dataset.variables["NXPE"][:]).item())
        nype = int(np.asarray(dataset.variables["NYPE"][:]).item())
        time_points = np.asarray(dataset.variables["t_array"][:], dtype=np.float64)
        nz = dataset.dimensions["z"].size

    history = np.zeros((time_points.size, mxsub * nxpe, mysub * nype, nz), dtype=np.float64)
    for dump_path in dump_paths:
        with Dataset(dump_path) as dataset:
            pe_xind = int(np.asarray(dataset.variables["PE_XIND"][:]).item())
            pe_yind = int(np.asarray(dataset.variables["PE_YIND"][:]).item())
            field = np.asarray(dataset.variables[field_name][:], dtype=np.float64)
            active = field[:, mxg : mxg + mxsub, myg : myg + mysub, :]
            x_slice = slice(pe_xind * mxsub, (pe_xind + 1) * mxsub)
            y_slice = slice(pe_yind * mysub, (pe_yind + 1) * mysub)
            history[:, x_slice, y_slice, :] = active

    return DivertedTokamakFieldHistory(
        field_name=field_name,
        time_points=time_points,
        history_4d=history,
    )


def load_diverted_tokamak_geometry(
    mesh_path: str | Path,
    *,
    active_nx: int,
) -> DivertedTokamakGeometry:
    mesh = Path(mesh_path)
    with Dataset(mesh) as dataset:
        rxy_full = np.asarray(dataset.variables["Rxy"][:], dtype=np.float64)
        zxy_full = np.asarray(dataset.variables["Zxy"][:], dtype=np.float64)
        psixy_full = np.asarray(dataset.variables["psixy"][:], dtype=np.float64)
    if active_nx > rxy_full.shape[0]:
        raise ValueError(
            f"Active nx {active_nx} exceeds mesh x extent {rxy_full.shape[0]}"
        )
    mxg = (rxy_full.shape[0] - active_nx) // 2
    x_slice = slice(mxg, mxg + active_nx)
    rxy = rxy_full[x_slice, :]
    zxy = zxy_full[x_slice, :]
    psixy = psixy_full[x_slice, :]
    return DivertedTokamakGeometry(
        rxy=rxy,
        zxy=zxy,
        psixy=psixy,
        wall_r=np.asarray(rxy[-1, :], dtype=np.float64),
        wall_z=np.asarray(zxy[-1, :], dtype=np.float64),
        lower_target_r=np.asarray(rxy[:, 0], dtype=np.float64),
        lower_target_z=np.asarray(zxy[:, 0], dtype=np.float64),
        upper_target_r=np.asarray(rxy[:, -1], dtype=np.float64),
        upper_target_z=np.asarray(zxy[:, -1], dtype=np.float64),
    )


def toroidal_mean_fluctuation(history: DivertedTokamakFieldHistory) -> np.ndarray:
    mean_history = np.asarray(history.history_4d.mean(axis=-1), dtype=np.float64)
    return mean_history - mean_history[0]


def build_diverted_tokamak_analysis(
    geometry: DivertedTokamakGeometry,
    history: DivertedTokamakFieldHistory,
    *,
    field_history_2d: np.ndarray,
) -> dict[str, Any]:
    return {
        "field_name": history.field_name,
        "time_points": [float(value) for value in history.time_points],
        "frame_minima": [float(np.min(frame)) for frame in field_history_2d],
        "frame_maxima": [float(np.max(frame)) for frame in field_history_2d],
        "global_min": float(np.min(field_history_2d)),
        "global_max": float(np.max(field_history_2d)),
        "lcfs_level": 0.0,
        "geometry_shape": {
            "x": int(geometry.rxy.shape[0]),
            "y": int(geometry.rxy.shape[1]),
        },
    }


def write_diverted_tokamak_analysis_json(
    analysis: Mapping[str, Any], path: str | Path
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(analysis), indent=2, sort_keys=True), encoding="utf-8"
    )
    return target


def write_diverted_tokamak_arrays_npz(
    geometry: DivertedTokamakGeometry,
    history: DivertedTokamakFieldHistory,
    *,
    field_history_2d: np.ndarray,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        field_name=np.asarray(history.field_name),
        time_points=np.asarray(history.time_points, dtype=np.float64),
        field_history_2d=np.asarray(field_history_2d, dtype=np.float64),
        rxy=np.asarray(geometry.rxy, dtype=np.float64),
        zxy=np.asarray(geometry.zxy, dtype=np.float64),
        psixy=np.asarray(geometry.psixy, dtype=np.float64),
    )
    return target


def load_diverted_tokamak_arrays_npz(
    path: str | Path,
) -> tuple[DivertedTokamakGeometry, str, np.ndarray, np.ndarray]:
    source = Path(path)
    with np.load(source) as payload:
        field_name = str(np.asarray(payload["field_name"]).item())
        time_points = np.asarray(payload["time_points"], dtype=np.float64)
        field_history_2d = np.asarray(payload["field_history_2d"], dtype=np.float64)
        rxy = np.asarray(payload["rxy"], dtype=np.float64)
        zxy = np.asarray(payload["zxy"], dtype=np.float64)
        psixy = np.asarray(payload["psixy"], dtype=np.float64)
    geometry = DivertedTokamakGeometry(
        rxy=rxy,
        zxy=zxy,
        psixy=psixy,
        wall_r=np.asarray(rxy[-1, :], dtype=np.float64),
        wall_z=np.asarray(zxy[-1, :], dtype=np.float64),
        lower_target_r=np.asarray(rxy[:, 0], dtype=np.float64),
        lower_target_z=np.asarray(zxy[:, 0], dtype=np.float64),
        upper_target_r=np.asarray(rxy[:, -1], dtype=np.float64),
        upper_target_z=np.asarray(zxy[:, -1], dtype=np.float64),
    )
    return geometry, field_name, time_points, field_history_2d


def create_diverted_tokamak_movie_package(
    *,
    workdir: str | Path,
    mesh_path: str | Path,
    output_root: str | Path,
    field_name: str = "Nd+",
    case_label: str = "diverted_tokamak_turbulence",
    fps: int = 10,
    frames_per_interval: int = 8,
) -> DivertedTokamakMovieArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    movies_dir = root / "movies"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    movies_dir.mkdir(parents=True, exist_ok=True)

    history = assemble_tokamak_rank_history(workdir, field_name=field_name)
    geometry = load_diverted_tokamak_geometry(
        mesh_path, active_nx=history.history_4d.shape[1]
    )
    field_history_2d = toroidal_mean_fluctuation(history)
    analysis = build_diverted_tokamak_analysis(
        geometry, history, field_history_2d=field_history_2d
    )

    arrays_npz_path = write_diverted_tokamak_arrays_npz(
        geometry,
        history,
        field_history_2d=field_history_2d,
        path=data_dir / f"{case_label}_arrays.npz",
    )
    analysis_json_path = write_diverted_tokamak_analysis_json(
        analysis,
        data_dir / f"{case_label}_analysis.json",
    )
    snapshots_png_path = save_diverted_tokamak_snapshot_panel(
        geometry,
        time_points=history.time_points,
        field_history_2d=field_history_2d,
        field_name=field_name,
        path=images_dir / f"{case_label}_snapshots.png",
    )
    poster_png_path = save_diverted_tokamak_poster_frame(
        geometry,
        time_points=history.time_points,
        field_history_2d=field_history_2d,
        field_name=field_name,
        path=images_dir / f"{case_label}_poster.png",
    )
    movie_gif_path = save_diverted_tokamak_gif(
        geometry,
        time_points=history.time_points,
        field_history_2d=field_history_2d,
        field_name=field_name,
        path=movies_dir / f"{case_label}.gif",
        fps=fps,
        frames_per_interval=frames_per_interval,
    )
    return DivertedTokamakMovieArtifacts(
        arrays_npz_path=arrays_npz_path,
        analysis_json_path=analysis_json_path,
        snapshots_png_path=snapshots_png_path,
        poster_png_path=poster_png_path,
        movie_gif_path=movie_gif_path,
    )


def create_diverted_tokamak_movie_package_from_arrays(
    *,
    arrays_npz_path: str | Path,
    output_root: str | Path,
    case_label: str = "diverted_tokamak_turbulence",
    field_name: str | None = None,
    fps: int = 10,
    frames_per_interval: int = 8,
) -> DivertedTokamakMovieArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    movies_dir = root / "movies"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    movies_dir.mkdir(parents=True, exist_ok=True)

    source_npz = Path(arrays_npz_path)
    (
        geometry,
        stored_field_name,
        time_points,
        field_history_2d,
    ) = load_diverted_tokamak_arrays_npz(source_npz)
    resolved_field_name = field_name or stored_field_name
    arrays_target = data_dir / f"{case_label}_arrays.npz"
    if source_npz.resolve() != arrays_target.resolve():
        shutil.copyfile(source_npz, arrays_target)
    else:
        arrays_target = source_npz
    history = DivertedTokamakFieldHistory(
        field_name=resolved_field_name,
        time_points=time_points,
        history_4d=np.empty((time_points.size, 0, 0, 0), dtype=np.float64),
    )
    analysis = build_diverted_tokamak_analysis(
        geometry,
        history,
        field_history_2d=field_history_2d,
    )
    analysis_json_path = write_diverted_tokamak_analysis_json(
        analysis,
        data_dir / f"{case_label}_analysis.json",
    )
    snapshots_png_path = save_diverted_tokamak_snapshot_panel(
        geometry,
        time_points=time_points,
        field_history_2d=field_history_2d,
        field_name=resolved_field_name,
        path=images_dir / f"{case_label}_snapshots.png",
    )
    poster_png_path = save_diverted_tokamak_poster_frame(
        geometry,
        time_points=time_points,
        field_history_2d=field_history_2d,
        field_name=resolved_field_name,
        path=images_dir / f"{case_label}_poster.png",
    )
    movie_gif_path = save_diverted_tokamak_gif(
        geometry,
        time_points=time_points,
        field_history_2d=field_history_2d,
        field_name=resolved_field_name,
        path=movies_dir / f"{case_label}.gif",
        fps=fps,
        frames_per_interval=frames_per_interval,
    )
    return DivertedTokamakMovieArtifacts(
        arrays_npz_path=arrays_target,
        analysis_json_path=analysis_json_path,
        snapshots_png_path=snapshots_png_path,
        poster_png_path=poster_png_path,
        movie_gif_path=movie_gif_path,
    )


def save_diverted_tokamak_snapshot_panel(
    geometry: DivertedTokamakGeometry,
    *,
    time_points: np.ndarray,
    field_history_2d: np.ndarray,
    field_name: str,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = _representative_frame_indices(field_history_2d.shape[0])
    figure, axes = plt.subplots(
        1, len(frame_indices), figsize=(14.0, 5.2), constrained_layout=True
    )
    if len(frame_indices) == 1:
        axes = [axes]
    vlim = _symmetric_color_limit(field_history_2d)
    image = None
    for axis, frame_index, label in zip(
        axes, frame_indices, ("Initial", "Mid", "Final"), strict=True
    ):
        image = _draw_diverted_tokamak_frame(
            axis,
            geometry=geometry,
            field_frame=field_history_2d[frame_index],
            field_name=field_name,
            time_value=float(time_points[frame_index]),
            color_limit=vlim,
            title_prefix=label,
        )
    figure.colorbar(
        image,
        ax=axes,
        shrink=0.88,
        pad=0.02,
        label=f"{field_name} toroidal-mean fluctuation",
    )
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def save_diverted_tokamak_poster_frame(
    geometry: DivertedTokamakGeometry,
    *,
    time_points: np.ndarray,
    field_history_2d: np.ndarray,
    field_name: str,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    vlim = _symmetric_color_limit(field_history_2d)
    frame_index = field_history_2d.shape[0] - 1
    figure, axis = plt.subplots(figsize=(7.2, 8.0), constrained_layout=True)
    image = _draw_diverted_tokamak_frame(
        axis,
        geometry=geometry,
        field_frame=field_history_2d[frame_index],
        field_name=field_name,
        time_value=float(time_points[frame_index]),
        color_limit=vlim,
        title_prefix="Final",
    )
    figure.colorbar(
        image,
        ax=axis,
        shrink=0.88,
        pad=0.02,
        label=f"{field_name} toroidal-mean fluctuation",
    )
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def save_diverted_tokamak_gif(
    geometry: DivertedTokamakGeometry,
    *,
    time_points: np.ndarray,
    field_history_2d: np.ndarray,
    field_name: str,
    path: str | Path,
    fps: int,
    frames_per_interval: int,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dense_time_points, dense_history = _interpolate_history(
        np.asarray(time_points, dtype=np.float64),
        np.asarray(field_history_2d, dtype=np.float64),
        frames_per_interval=frames_per_interval,
    )
    vlim = _symmetric_color_limit(dense_history)
    figure, axis = plt.subplots(figsize=(7.2, 8.0), constrained_layout=True)
    image = _draw_diverted_tokamak_frame(
        axis,
        geometry=geometry,
        field_frame=dense_history[0],
        field_name=field_name,
        time_value=float(dense_time_points[0]),
        color_limit=vlim,
        title_prefix="Toroidal Mean",
    )
    figure.colorbar(image, ax=axis, shrink=0.88, pad=0.02, label=f"{field_name} toroidal-mean fluctuation")

    def update(frame_index: int):
        axis.clear()
        return (
            _draw_diverted_tokamak_frame(
                axis,
                geometry=geometry,
                field_frame=dense_history[frame_index],
                field_name=field_name,
                time_value=float(dense_time_points[frame_index]),
                color_limit=vlim,
                title_prefix="Toroidal Mean",
            ),
        )

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=dense_history.shape[0],
        interval=1000 / max(fps, 1),
        blit=False,
    )
    movie.save(target, writer=animation.PillowWriter(fps=fps), dpi=140)
    plt.close(figure)
    return target


def _draw_diverted_tokamak_frame(
    axis,
    *,
    geometry: DivertedTokamakGeometry,
    field_frame: np.ndarray,
    field_name: str,
    time_value: float,
    color_limit: float,
    title_prefix: str,
):
    image = axis.tripcolor(
        geometry.rxy.ravel(),
        geometry.zxy.ravel(),
        field_frame.ravel(),
        shading="gouraud",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        rasterized=True,
    )
    axis.contour(
        geometry.rxy,
        geometry.zxy,
        geometry.psixy,
        levels=[0.0],
        colors="white",
        linewidths=1.5,
        linestyles="--",
    )
    axis.plot(geometry.wall_r, geometry.wall_z, color="black", linewidth=2.4, label="Wall")
    axis.plot(
        geometry.lower_target_r,
        geometry.lower_target_z,
        color="#ff9f1c",
        linewidth=2.2,
        label="Lower divertor",
    )
    axis.plot(
        geometry.upper_target_r,
        geometry.upper_target_z,
        color="#2ec4b6",
        linewidth=2.2,
        label="Upper divertor",
    )
    axis.set_aspect("equal")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_title(f"{title_prefix}: {field_name}  |  t = {time_value:.3f}")
    axis.legend(loc="upper right", frameon=True)
    return image


def _representative_frame_indices(frame_count: int) -> tuple[int, int, int]:
    if frame_count <= 1:
        return (0, 0, 0)
    if frame_count == 2:
        return (0, 1, 1)
    return (0, frame_count // 2, frame_count - 1)


def _symmetric_color_limit(field_history_2d: np.ndarray) -> float:
    limit = float(np.percentile(np.abs(field_history_2d), 99.0))
    return max(limit, 1.0e-12)


def _interpolate_history(
    time_points: np.ndarray,
    field_history_2d: np.ndarray,
    *,
    frames_per_interval: int,
) -> tuple[np.ndarray, np.ndarray]:
    if field_history_2d.shape[0] <= 1 or frames_per_interval <= 1:
        return np.asarray(time_points, dtype=np.float64), np.asarray(field_history_2d, dtype=np.float64)
    dense_times: list[float] = []
    dense_frames: list[np.ndarray] = []
    for index in range(field_history_2d.shape[0] - 1):
        start_time = float(time_points[index])
        end_time = float(time_points[index + 1])
        start_frame = np.asarray(field_history_2d[index], dtype=np.float64)
        end_frame = np.asarray(field_history_2d[index + 1], dtype=np.float64)
        for substep in range(frames_per_interval):
            alpha = substep / float(frames_per_interval)
            dense_times.append((1.0 - alpha) * start_time + alpha * end_time)
            dense_frames.append((1.0 - alpha) * start_frame + alpha * end_frame)
    dense_times.append(float(time_points[-1]))
    dense_frames.append(np.asarray(field_history_2d[-1], dtype=np.float64))
    return np.asarray(dense_times, dtype=np.float64), np.asarray(dense_frames, dtype=np.float64)
