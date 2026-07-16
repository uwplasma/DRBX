from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import jax.numpy as jnp
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from analyze_EB_density import _load_eb_blob_step_history, _resolve_step_dump_dir  # noqa: E402
from jax_drb.native.fci_drb_EB_rhs import FciDrbEBState, _multiply_local_stencils  # noqa: E402
from jax_drb.native.fci_operators import grad_parallel_op_direct  # noqa: E402
from test_shifted_torus_EB_blob import (  # noqa: E402
    PERIODIC_AXES,
    _eb_blob_artifact_stem,
    _eb_blob_z_indices,
    _load_eb_blob_history,
    _resolve_eb_blob_history_path,
    local_stencil_builder,
    z0,
)
from analyze_rhs_common import build_eb_blob_context  # noqa: E402


DEFAULT_FRAME_STRIDE = 2
DEFAULT_MOVIE_FPS = 10
OUTPUT_SUFFIX = "parallel_operators"


def _resolve_output_dir(run_name: str, output_path: Path | None) -> Path:
    if output_path is None:
        return Path(f"{_eb_blob_artifact_stem(run_name)}_outputs")
    candidate = Path(output_path)
    if candidate.is_file():
        return candidate.parent
    if candidate.name.endswith("_histories.npz"):
        return candidate.parent
    return candidate


def _load_movie_history(
    *,
    run_name: str,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    history_path = _resolve_eb_blob_history_path(run_name, output_dir)
    if history_path.exists():
        print(f"loading EB blob histories from {history_path}", flush=True)
        times, density_history, phi_history, te_history, ti_history, vi_history, ve_history, vorticity_history, _ = _load_eb_blob_history(
            history_path
        )
        return (
            np.asarray(times, dtype=np.float64),
            np.asarray(density_history, dtype=np.float64),
            np.asarray(phi_history, dtype=np.float64),
            np.asarray(te_history, dtype=np.float64),
            np.asarray(ti_history, dtype=np.float64),
            np.asarray(vi_history, dtype=np.float64),
            np.asarray(ve_history, dtype=np.float64),
            np.asarray(vorticity_history, dtype=np.float64),
            history_path,
        )

    step_dump_dir = _resolve_step_dump_dir(run_name, output_dir)
    if not step_dump_dir.exists():
        raise FileNotFoundError(
            f"missing EB blob history file {history_path} and missing step dump directory {step_dump_dir}"
        )
    print(f"loading EB blob step dumps from {step_dump_dir}", flush=True)
    times, density_history, phi_history, te_history, ti_history, vi_history, ve_history, vorticity_history = _load_eb_blob_step_history(
        step_dump_dir
    )
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(density_history, dtype=np.float64),
        np.asarray(phi_history, dtype=np.float64),
        np.asarray(te_history, dtype=np.float64),
        np.asarray(ti_history, dtype=np.float64),
        np.asarray(vi_history, dtype=np.float64),
        np.asarray(ve_history, dtype=np.float64),
        np.asarray(vorticity_history, dtype=np.float64),
        step_dump_dir,
    )


def _robust_signed_norm(values: np.ndarray, *, qlo: float = 2.0, qhi: float = 98.0):
    import matplotlib.colors as colors

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot normalize all-nonfinite values")
    lo = float(np.percentile(finite, qlo))
    hi = float(np.percentile(finite, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("cannot normalize non-finite percentile bounds")
    if np.isclose(lo, hi):
        spread = max(abs(lo), abs(hi), 1.0)
        lo, hi = -spread, spread
    if lo < 0.0 < hi:
        bound = max(abs(lo), abs(hi))
        return colors.TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
    return colors.Normalize(vmin=lo, vmax=hi)


def _compute_parallel_operator_slices(
    state: FciDrbEBState,
    *,
    geometry,
    parameters,
    boundary_condition_builder,
    cut_wall_geometry,
    cut_wall_bc,
    z_indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    boundary_conditions = boundary_condition_builder(
        state,
        geometry,
        PERIODIC_AXES,
        cut_wall_geometry,
        cut_wall_bc,
    )

    density = jnp.asarray(state.density, dtype=jnp.float64)
    Ve = jnp.asarray(state.Ve, dtype=jnp.float64)
    Vi = jnp.asarray(state.Vi, dtype=jnp.float64)

    density_stencil = local_stencil_builder(
        density,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.density_face_bc,
        cut_wall_geometry,
        boundary_conditions.density_cut_wall_bc,
    )
    Ve_stencil = local_stencil_builder(
        Ve,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Ve_face_bc,
        cut_wall_geometry,
        boundary_conditions.Ve_cut_wall_bc,
    )
    Vi_stencil = local_stencil_builder(
        Vi,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Vi_face_bc,
        cut_wall_geometry,
        boundary_conditions.Vi_cut_wall_bc,
    )

    nve_stencil = _multiply_local_stencils(density_stencil, Ve_stencil)
    j_parallel = density * (Vi - Ve)
    j_parallel_stencil = local_stencil_builder(
        j_parallel,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Ve_face_bc,
        cut_wall_geometry,
        boundary_conditions.Ve_cut_wall_bc,
    )

    neg_grad_nve = -grad_parallel_op_direct(nve_stencil, geometry)
    grad_Ve = grad_parallel_op_direct(Ve_stencil, geometry)
    grad_Vi = grad_parallel_op_direct(Vi_stencil, geometry)
    grad_j = grad_parallel_op_direct(j_parallel_stencil, geometry)

    return {
        "-grad_parallel(nVe)": np.asarray(neg_grad_nve[:, :, z_indices], dtype=np.float64).transpose(2, 0, 1),
        "grad_parallel(Ve)": np.asarray(grad_Ve[:, :, z_indices], dtype=np.float64).transpose(2, 0, 1),
        "grad_parallel(Vi)": np.asarray(grad_Vi[:, :, z_indices], dtype=np.float64).transpose(2, 0, 1),
        "grad_parallel(j_parallel)": np.asarray(grad_j[:, :, z_indices], dtype=np.float64).transpose(2, 0, 1),
    }


def _collect_operator_histories(
    times: np.ndarray,
    density_history: np.ndarray,
    phi_history: np.ndarray,
    te_history: np.ndarray,
    ti_history: np.ndarray,
    vi_history: np.ndarray,
    ve_history: np.ndarray,
    vorticity_history: np.ndarray,
    *,
    geometry,
    parameters,
    boundary_condition_builder,
    cut_wall_geometry,
    cut_wall_bc,
    z_indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    total_steps = int(np.asarray(times).shape[0])
    report_interval = max(1, total_steps // 10)
    operator_histories: dict[str, list[np.ndarray]] = {
        "-grad_parallel(nVe)": [],
        "grad_parallel(Ve)": [],
        "grad_parallel(Vi)": [],
        "grad_parallel(j_parallel)": [],
    }
    print(
        f"building parallel operator movie from {total_steps} saved steps and {len(z_indices)} toroidal slices",
        flush=True,
    )

    for index in range(total_steps):
        state = FciDrbEBState(
            density=jnp.asarray(density_history[index], dtype=jnp.float64),
            phi=jnp.asarray(phi_history[index], dtype=jnp.float64),
            Te=jnp.asarray(te_history[index], dtype=jnp.float64),
            Ti=jnp.asarray(ti_history[index], dtype=jnp.float64),
            Vi=jnp.asarray(vi_history[index], dtype=jnp.float64),
            Ve=jnp.asarray(ve_history[index], dtype=jnp.float64),
            vorticity=jnp.asarray(vorticity_history[index], dtype=jnp.float64),
        )

        slices = _compute_parallel_operator_slices(
            state,
            geometry=geometry,
            parameters=parameters,
            boundary_condition_builder=boundary_condition_builder,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=cut_wall_bc,
            z_indices=z_indices,
        )
        for key, value in slices.items():
            operator_histories[key].append(value)

        if index == 0 or (index + 1) % report_interval == 0 or index + 1 == total_steps:
            print(
                f"processed parallel-operator frame {index + 1}/{total_steps} "
                f"at t={float(np.asarray(times[index], dtype=np.float64)):.6e}",
                flush=True,
            )

    return {key: np.asarray(value, dtype=np.float64) for key, value in operator_histories.items()}


def _save_parallel_operator_movie(
    times: np.ndarray,
    operator_histories: dict[str, np.ndarray],
    geometry,
    z_indices: tuple[int, ...],
    *,
    output_path: Path,
    frame_stride: int,
    movie_fps: int,
    title: str,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    y_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    theta_grid, radius_grid = np.meshgrid(y_values, x_values)

    frame_stride = max(1, int(frame_stride))
    movie_fps = max(1, int(movie_fps))
    frame_indices = np.arange(0, int(times.shape[0]), frame_stride, dtype=np.int64)
    if frame_indices[-1] != int(times.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times.shape[0]) - 1)

    operator_names = (
        "-grad_parallel(nVe)",
        "grad_parallel(Ve)",
        "grad_parallel(Vi)",
        "grad_parallel(j_parallel)",
    )
    norms = tuple(_robust_signed_norm(operator_histories[name]) for name in operator_names)

    fig, axes = plt.subplots(
        nrows=len(operator_names),
        ncols=len(z_indices),
        figsize=(18.5, 14.0),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    if len(operator_names) == 1:
        axes = np.asarray([axes])

    row_mappables = [
        cm.ScalarMappable(norm=norm, cmap="coolwarm")
        for norm in norms
    ]
    for row, (name, mappable) in enumerate(zip(operator_names, row_mappables, strict=True)):
        fig.colorbar(
            mappable,
            ax=list(axes[row, :]),
            location="right",
            pad=0.02,
            shrink=0.88,
            label=name,
        )

    images: list[list[object]] = [[None for _ in z_indices] for _ in operator_names]  # type: ignore[list-item]
    for row, name in enumerate(operator_names):
        row_history = operator_histories[name]
        for col, z_index in enumerate(z_indices):
            ax = axes[row, col]
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(-1)
            ax.set_ylim(0.0, float(x_values[-1]))
            ax.set_yticklabels([])
            ax.set_title(f"{name}, z={z_values[z_index]:.3f}")
            image = ax.pcolormesh(
                theta_grid,
                radius_grid,
                row_history[0, col],
                shading="auto",
                cmap="coolwarm",
                norm=norms[row],
            )
            images[row][col] = image

    suptitle = fig.suptitle(title)

    def update(animation_index: int):
        actual_index = int(frame_indices[animation_index])
        time_value = float(times[actual_index])
        for row, name in enumerate(operator_names):
            row_history = operator_histories[name]
            for col, z_index in enumerate(z_indices):
                ax = axes[row, col]
                ax.clear()
                ax.set_theta_zero_location("E")
                ax.set_theta_direction(-1)
                ax.set_ylim(0.0, float(x_values[-1]))
                ax.set_yticklabels([])
                ax.set_title(f"{name}, z={z_values[z_index]:.3f}, t={time_value:.3e}")
                field_slice = np.asarray(row_history[actual_index, col], dtype=np.float64)
                finite = np.isfinite(field_slice)
                if not np.any(finite):
                    raise ValueError(
                        f"all values are non-finite for {name} at frame {actual_index}, z index {z_index}"
                    )
                ax.pcolormesh(
                    theta_grid,
                    radius_grid,
                    np.where(np.isfinite(field_slice), field_slice, np.nan),
                    shading="auto",
                    cmap="coolwarm",
                    norm=norms[row],
                )
        suptitle.set_text(f"{title}, t={time_value:.3e}")
        return []

    animator = animation.FuncAnimation(fig, update, frames=int(frame_indices.shape[0]), interval=100, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=movie_fps)
    animator.save(str(output_path), writer=writer)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Render a movie of parallel operator slices for the EB blob run.")
    parser.add_argument(
        "--run-name",
        "--run_name",
        dest="run_name",
        default="EB_perp_diffusion",
        help="Run name used to locate the saved EB blob output directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Run output directory, step-dump directory, or history file path.",
    )
    parser.add_argument(
        "--perp-diffusion",
        type=float,
        default=1.0e-5,
        help="Perpendicular diffusion coefficient used when rebuilding the geometry context.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        help="Use every Nth saved timestep when rendering the movie.",
    )
    parser.add_argument(
        "--movie-fps",
        type=int,
        default=DEFAULT_MOVIE_FPS,
        help="Frames per second for the output GIF.",
    )
    args = parser.parse_args(argv)

    output_dir = _resolve_output_dir(str(args.run_name), args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = _eb_blob_artifact_stem(str(args.run_name))
    movie_path = output_dir / f"{artifact_stem}_{OUTPUT_SUFFIX}.gif"

    (
        times,
        density_history,
        phi_history,
        te_history,
        ti_history,
        vi_history,
        ve_history,
        vorticity_history,
        source_path,
    ) = _load_movie_history(run_name=str(args.run_name), output_dir=output_dir)
    context = build_eb_blob_context(np.asarray(density_history, dtype=np.float64), float(args.perp_diffusion))
    geometry = context.geometry
    z_indices = _eb_blob_z_indices(geometry, z0, count=4)

    print(
        f"building parallel operator movie from {source_path} with {int(times.shape[0])} snapshots "
        f"and z indices {z_indices}",
        flush=True,
    )

    operator_histories = _collect_operator_histories(
        np.asarray(times, dtype=np.float64),
        np.asarray(density_history, dtype=np.float64),
        np.asarray(phi_history, dtype=np.float64),
        np.asarray(te_history, dtype=np.float64),
        np.asarray(ti_history, dtype=np.float64),
        np.asarray(vi_history, dtype=np.float64),
        np.asarray(ve_history, dtype=np.float64),
        np.asarray(vorticity_history, dtype=np.float64),
        geometry=context.geometry,
        parameters=context.parameters,
        boundary_condition_builder=context.boundary_condition_builder,
        cut_wall_geometry=context.cut_wall_geometry,
        cut_wall_bc=context.cut_wall_bc,
        z_indices=z_indices,
    )

    _save_parallel_operator_movie(
        np.asarray(times, dtype=np.float64),
        operator_histories,
        context.geometry,
        z_indices,
        output_path=movie_path,
        frame_stride=int(args.frame_stride),
        movie_fps=int(args.movie_fps),
        title=f"EB blob parallel operator diagnostics, D_perp={float(args.perp_diffusion):.3e}",
    )
    print(f"saved parallel operator movie to {movie_path}", flush=True)
    return movie_path


if __name__ == "__main__":
    main()
