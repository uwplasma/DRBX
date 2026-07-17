from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence
import sys

_CACHE_ROOT = Path("/tmp") / f"drbx_plot_density_3d_{os.getuid()}_{os.getpid()}"
_MPLCONFIGDIR = _CACHE_ROOT / "mplconfig"
_XDG_CACHE_HOME = _CACHE_ROOT / "xdg_cache"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from matplotlib import animation, cm, colors, pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from analyze_EB_density import _load_eb_blob_step_history, _resolve_step_dump_dir  # noqa: E402
from test_mms_shifted_torus_4_field import alpha_value as _ALPHA_VALUE, r0 as _R0, sigma as _SIGMA  # noqa: E402
from test_shifted_torus_EB_blob import _build_eb_blob_geometry, _eb_blob_artifact_stem, _load_eb_blob_history  # noqa: E402


DEFAULT_FRAME_STRIDE = 4
TARGET_RENDER_CELLS_PER_AXIS = 18
FIXED_ELEVATION_DEG = 24.0
FIXED_AZIMUTH_DEG = 35.0
VOLUME_CMAP = plt.get_cmap("magma")


def _robust_limits(values: np.ndarray, *, qlo: float = 2.0, qhi: float = 98.0) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot compute limits from non-finite values")
    lo = float(np.percentile(finite, qlo))
    hi = float(np.percentile(finite, qhi))
    if np.isclose(lo, hi):
        spread = max(abs(lo), abs(hi), 1.0)
        lo -= 0.5 * spread
        hi += 0.5 * spread
    if hi <= lo:
        hi = lo + max(abs(lo), 1.0)
    return lo, hi


def _coarse_axis_edges(num_cells: int, target_cells: int = TARGET_RENDER_CELLS_PER_AXIS) -> np.ndarray:
    if num_cells < 1:
        raise ValueError(f"num_cells must be positive, got {num_cells}")
    target_cells = max(1, int(target_cells))
    step = max(1, int(np.ceil(float(num_cells) / float(target_cells))))
    edges = np.arange(0, num_cells + 1, step, dtype=np.int64)
    if edges[0] != 0:
        edges = np.insert(edges, 0, 0)
    if edges[-1] != num_cells:
        edges = np.append(edges, num_cells)
    return edges


def _half_torus_coarse_mask(geometry, z_edges: np.ndarray) -> np.ndarray:
    zeta_centers = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    keep = np.zeros(z_edges.size - 1, dtype=bool)
    for iz, (z0, z1) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        zeta_center = float(np.mean(zeta_centers[z0:z1]))
        keep[iz] = zeta_center <= np.pi
    return keep


def _density_facecolors(values: np.ndarray, density_limits: tuple[float, float]) -> np.ndarray:
    lo, hi = density_limits
    span = max(hi - lo, 1.0e-12)
    finite = np.isfinite(values)
    normalized = np.clip((np.where(finite, values, lo) - lo) / span, 0.0, 1.0)
    rgba = VOLUME_CMAP(normalized)
    rgba[..., 3] = np.where(finite, 0.04 + 0.78 * normalized**1.45, 0.0)
    return rgba


def _shifted_torus_face_xyz(geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_faces = np.asarray(geometry.grid.x.faces, dtype=np.float64)
    theta_faces = np.asarray(geometry.grid.y.faces, dtype=np.float64)
    zeta_faces = np.asarray(geometry.grid.z.faces, dtype=np.float64)
    rho, theta, zeta = np.meshgrid(rho_faces, theta_faces, zeta_faces, indexing="ij")
    rho_mid = 0.5 * (float(rho_faces[0]) + float(rho_faces[-1]))
    theta_shift = theta + float(_SIGMA) * (rho - rho_mid)
    major_radius = float(_R0) + float(_ALPHA_VALUE) * rho + rho * np.cos(theta_shift)
    x = major_radius * np.cos(zeta)
    y = major_radius * np.sin(zeta)
    z = rho * np.sin(theta_shift)
    return x, y, z


def _resolve_movie_paths(run_name: str, output_path: Path | None) -> tuple[Path, Path, Path | None]:
    artifact_stem = _eb_blob_artifact_stem(run_name)
    movie_name = f"{artifact_stem}_density_volume.gif"

    history_path: Path | None = None
    resolution_root: Path | None

    if output_path is None:
        resolution_root = None
        movie_path = Path(movie_name)
    else:
        candidate = Path(output_path)
        if candidate.suffix.lower() == ".gif":
            resolution_root = candidate.parent
            movie_path = candidate
        elif candidate.suffix.lower() == ".npz" and candidate.name.endswith("_histories.npz"):
            resolution_root = candidate.parent
            history_path = candidate
            movie_path = candidate.parent / movie_name
        else:
            resolution_root = candidate
            movie_path = candidate / movie_name

    step_dump_dir = _resolve_step_dump_dir(run_name, resolution_root)
    if output_path is None:
        movie_dir = step_dump_dir.parent if step_dump_dir.name == "step_dumps" else step_dump_dir
        movie_path = movie_dir / movie_name
    return step_dump_dir, movie_path, history_path


def _load_density_history(
    step_dump_dir: Path,
    *,
    run_name: str,
    history_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if history_path is not None:
        (
            times,
            density_history,
            phi_history,
            te_history,
            ti_history,
            vi_history,
            ve_history,
            vorticity_history,
            _metadata,
        ) = _load_eb_blob_history(history_path)
        return (
            np.asarray(times, dtype=np.float64),
            np.asarray(density_history, dtype=np.float64),
            np.asarray(phi_history, dtype=np.float64),
            np.asarray(te_history, dtype=np.float64),
            np.asarray(ti_history, dtype=np.float64),
            np.asarray(vi_history, dtype=np.float64),
            np.asarray(ve_history, dtype=np.float64),
            np.asarray(vorticity_history, dtype=np.float64),
        )

    step_files = sorted(step_dump_dir.glob("step_*.npz"))
    if step_files:
        (
            times,
            density_history,
            phi_history,
            te_history,
            ti_history,
            vi_history,
            ve_history,
            vorticity_history,
        ) = _load_eb_blob_step_history(step_dump_dir)
        return (
            np.asarray(times, dtype=np.float64),
            np.asarray(density_history, dtype=np.float64),
            np.asarray(phi_history, dtype=np.float64),
            np.asarray(te_history, dtype=np.float64),
            np.asarray(ti_history, dtype=np.float64),
            np.asarray(vi_history, dtype=np.float64),
            np.asarray(ve_history, dtype=np.float64),
            np.asarray(vorticity_history, dtype=np.float64),
        )

    artifact_stem = _eb_blob_artifact_stem(run_name)
    candidate_history_paths = (
        step_dump_dir / f"{artifact_stem}_histories.npz",
        step_dump_dir.parent / f"{artifact_stem}_histories.npz",
    )
    for candidate_history_path in candidate_history_paths:
        if candidate_history_path.exists():
            return _load_density_history(step_dump_dir, run_name=run_name, history_path=candidate_history_path)

    raise FileNotFoundError(
        f"no step_*.npz files or {artifact_stem}_histories.npz found in {step_dump_dir}"
    )


def _coarsen_density_frame(
    density_frame: np.ndarray,
    weights: np.ndarray | None,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
) -> np.ndarray:
    density_np = np.asarray(density_frame, dtype=np.float64)
    coarse = np.empty((x_edges.size - 1, y_edges.size - 1, z_edges.size - 1), dtype=np.float64)

    for ix, (x0, x1) in enumerate(zip(x_edges[:-1], x_edges[1:])):
        for iy, (y0, y1) in enumerate(zip(y_edges[:-1], y_edges[1:])):
            for iz, (z0, z1) in enumerate(zip(z_edges[:-1], z_edges[1:])):
                block = density_np[x0:x1, y0:y1, z0:z1]
                finite_block = np.isfinite(block)
                if not np.any(finite_block):
                    coarse[ix, iy, iz] = np.nan
                    continue

                if weights is None:
                    coarse[ix, iy, iz] = float(np.mean(block[finite_block]))
                    continue

                block_weights = np.asarray(weights[x0:x1, y0:y1, z0:z1], dtype=np.float64)
                valid = finite_block & np.isfinite(block_weights)
                if not np.any(valid):
                    coarse[ix, iy, iz] = float(np.mean(block[finite_block]))
                    continue

                valid_weights = block_weights[valid]
                weight_sum = float(np.sum(valid_weights))
                if weight_sum <= 0.0:
                    coarse[ix, iy, iz] = float(np.mean(block[valid]))
                else:
                    coarse[ix, iy, iz] = float(np.average(block[valid], weights=valid_weights))

    return coarse


def _save_density_volume_movie(
    times: np.ndarray,
    density_history: np.ndarray,
    geometry,
    *,
    output_path: Path,
    title: str,
    frame_stride: int = 4,
    fps: int = 8,
) -> None:
    times_np = np.asarray(times, dtype=np.float64)
    density_np = np.asarray(density_history, dtype=np.float64)
    if density_np.ndim != 4:
        raise ValueError(f"density_history must have shape (time, nx, ny, nz), got {density_np.shape}")
    if times_np.shape[0] != density_np.shape[0]:
        raise ValueError(
            f"times and density_history disagree: {times_np.shape[0]} times for {density_np.shape[0]} frames"
        )

    geometry_shape = tuple(int(v) for v in geometry.shape)
    if density_np.shape[1:] != geometry_shape:
        raise ValueError(f"density shape {density_np.shape[1:]} does not match geometry shape {geometry_shape}")

    x_faces, y_faces, z_faces = _shifted_torus_face_xyz(geometry)
    x_edges = _coarse_axis_edges(density_np.shape[1])
    y_edges = _coarse_axis_edges(density_np.shape[2])
    z_edges = _coarse_axis_edges(density_np.shape[3])
    coarse_x = np.asarray(x_faces[np.ix_(x_edges, y_edges, z_edges)], dtype=np.float64)
    coarse_y = np.asarray(y_faces[np.ix_(x_edges, y_edges, z_edges)], dtype=np.float64)
    coarse_z = np.asarray(z_faces[np.ix_(x_edges, y_edges, z_edges)], dtype=np.float64)
    half_torus_keep = _half_torus_coarse_mask(geometry, z_edges)

    weights = np.asarray(geometry.cell_metric.J, dtype=np.float64)
    if weights.shape != geometry_shape:
        raise ValueError(f"geometry cell_metric has shape {weights.shape}, expected {geometry_shape}")

    density_limits = _robust_limits(density_np)
    frame_stride = max(1, int(frame_stride))
    fps = max(1, int(fps))
    frame_indices = np.arange(0, int(times_np.shape[0]), frame_stride, dtype=np.int64)
    if frame_indices.size == 0 or frame_indices[-1] != int(times_np.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times_np.shape[0]) - 1)

    finite_x = coarse_x[np.isfinite(coarse_x)]
    finite_y = coarse_y[np.isfinite(coarse_y)]
    finite_z = coarse_z[np.isfinite(coarse_z)]
    x_limits = (float(np.min(finite_x)), float(np.max(finite_x)))
    y_limits = (float(np.min(finite_y)), float(np.max(finite_y)))
    z_limits = (float(np.min(finite_z)), float(np.max(finite_z)))
    x_span = max(x_limits[1] - x_limits[0], 1.0e-12)
    y_span = max(y_limits[1] - y_limits[0], 1.0e-12)
    z_span = max(z_limits[1] - z_limits[0], 1.0e-12)
    aspect = (x_span, y_span, z_span)

    figure = plt.figure(figsize=(8.6, 7.4), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    scalar = cm.ScalarMappable(norm=colors.Normalize(vmin=density_limits[0], vmax=density_limits[1]), cmap=VOLUME_CMAP)
    scalar.set_array([])
    figure.colorbar(scalar, ax=axis, shrink=0.72, pad=0.03, label="density")
    title_artist = figure.suptitle("", fontsize=11)

    def update(frame_index: int):
        actual_index = int(frame_indices[frame_index])
        coarse_density = _coarsen_density_frame(
            density_np[actual_index],
            weights,
            x_edges,
            y_edges,
            z_edges,
        )
        filled = np.isfinite(coarse_density) & half_torus_keep[None, None, :]
        facecolors = _density_facecolors(coarse_density, density_limits)
        facecolors[~filled] = (0.0, 0.0, 0.0, 0.0)

        axis.cla()
        axis.voxels(coarse_x, coarse_y, coarse_z, filled, facecolors=facecolors)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_zlim(*z_limits)
        axis.set_box_aspect(aspect)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.grid(False)
        axis.view_init(elev=FIXED_ELEVATION_DEG, azim=FIXED_AZIMUTH_DEG)
        title_artist.set_text(f"{title}\nstep={actual_index}, t={float(times_np[actual_index]):.6e}")
        return ()

    animator = animation.FuncAnimation(
        figure,
        update,
        frames=int(frame_indices.shape[0]),
        interval=max(1, int(round(1000.0 / float(fps)))),
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    animator.save(output_path, writer=writer, dpi=140)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Render a 3D density volume movie from saved EB blob steps.")
    parser.add_argument(
        "--run-name",
        "--run_name",
        dest="run_name",
        default="eb_blob",
        help="Run name used to locate the saved output directory or step dumps.",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        dest="output_path",
        type=Path,
        default=None,
        help=(
            "Output directory or GIF path. If a directory is provided, the movie is saved as "
            "<run-name>_density_volume.gif inside it."
        ),
    )
    parser.add_argument(
        "--frame-stride",
        "--frame_stride",
        dest="frame_stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        help="Stride between rendered frames in the density movie.",
    )
    args = parser.parse_args(argv)

    step_dump_dir, movie_path, history_path = _resolve_movie_paths(args.run_name, args.output_path)
    if history_path is None and not any(step_dump_dir.glob("step_*.npz")):
        fallback_step_dump_dir = _resolve_step_dump_dir(args.run_name, None)
        if fallback_step_dump_dir != step_dump_dir and (
            any(fallback_step_dump_dir.glob("step_*.npz"))
            or any((fallback_step_dump_dir / "step_dumps").glob("step_*.npz"))
        ):
            step_dump_dir = fallback_step_dump_dir
    if not step_dump_dir.exists() and history_path is None:
        raise FileNotFoundError(f"missing EB blob step dump directory: {step_dump_dir}")

    (
        times,
        density_history,
        _phi_history,
        _te_history,
        _ti_history,
        _vi_history,
        _ve_history,
        _vorticity_history,
    ) = _load_density_history(step_dump_dir, run_name=args.run_name, history_path=history_path)

    density_shape = tuple(int(size) for size in np.asarray(density_history).shape[1:])
    geometry = _build_eb_blob_geometry(density_shape, construct_fci_maps=False)
    _save_density_volume_movie(
        np.asarray(times, dtype=np.float64),
        np.asarray(density_history, dtype=np.float64),
        geometry,
        output_path=movie_path,
        title="EB blob density volume render",
        frame_stride=int(args.frame_stride),
    )

    print(f"saved 3D density volume movie to {movie_path}", flush=True)
    return movie_path


if __name__ == "__main__":
    main()
