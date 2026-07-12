from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from jax_drb.geometry import RegularFaceGeometry3D  # noqa: E402
from jax_drb.native.fci_boundaries import BoundaryFaceBC3D, CutWallBC3D, CutWallGeometry3D  # noqa: E402
from jax_drb.native.fci_operators import grad_parallel_op_direct, grad_perp_op  # noqa: E402
from test_shifted_torus_EB_blob import (  # noqa: E402
    DEFAULT_RESOLUTION,
    PERIODIC_AXES,
    _build_eb_blob_geometry,
    _eb_blob_artifact_stem,
    _eb_blob_z_indices,
    local_stencil_builder,
    radial_b_fraction,
    z0,
)


def _resolve_output_dir(run_name: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return Path(f"{_eb_blob_artifact_stem(run_name)}_outputs")


def _load_metadata(history_path: Path) -> dict[str, object]:
    if not history_path.exists():
        return {}
    metadata: dict[str, object] = {}
    field_keys = {"times", "density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity"}
    with np.load(history_path, allow_pickle=False) as history:
        for key in history.files:
            if key in field_keys:
                continue
            value = history[key]
            metadata[key] = value.item() if getattr(value, "shape", ()) == () else value
    return metadata


def _resolve_step_dump_dir(output_dir: Path) -> Path:
    nested = output_dir / "step_dumps"
    if any(nested.glob("step_*.npz")):
        return nested
    if any(output_dir.glob("step_*.npz")):
        return output_dir
    return nested


def _load_step_history(step_dump_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step_files = sorted(step_dump_dir.glob("step_*.npz"))
    if not step_files:
        raise FileNotFoundError(f"no step_*.npz files found in {step_dump_dir}")

    snapshots: list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for fallback_index, step_file in enumerate(step_files):
        with np.load(step_file, allow_pickle=False) as step:
            step_index = int(step["step_index"]) if "step_index" in step.files else fallback_index
            time_value = float(step["time"]) if "time" in step.files else float(fallback_index)
            snapshots.append(
                (
                    step_index,
                    time_value,
                    np.asarray(step["phi"], dtype=np.float64),
                    np.asarray(step["Te"], dtype=np.float64),
                    np.asarray(step["Ti"], dtype=np.float64),
                    np.asarray(step["Ve"], dtype=np.float64),
                    np.asarray(step["Vi"], dtype=np.float64),
                )
            )

    snapshots.sort(key=lambda item: item[0])
    times = np.asarray([item[1] for item in snapshots], dtype=np.float64)
    phi_history = np.asarray([item[2] for item in snapshots], dtype=np.float64)
    te_history = np.asarray([item[3] for item in snapshots], dtype=np.float64)
    ti_history = np.asarray([item[4] for item in snapshots], dtype=np.float64)
    ve_history = np.asarray([item[5] for item in snapshots], dtype=np.float64)
    vi_history = np.asarray([item[6] for item in snapshots], dtype=np.float64)
    return times, phi_history, te_history, ti_history, ve_history, vi_history


def _robust_norm(values: np.ndarray, *, qlo: float = 2.0, qhi: float = 98.0):
    import matplotlib.colors as colors

    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
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


def _positive_norm(values: np.ndarray, *, qhi: float = 98.0):
    import matplotlib.colors as colors

    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot normalize all-nonfinite values")
    vmax = float(np.percentile(finite, qhi))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = max(float(np.nanmax(finite)), 1.0)
    return colors.Normalize(vmin=0.0, vmax=vmax)


def _compute_exb_from_phi(phi: np.ndarray, geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    phi_jnp = jnp.asarray(phi, dtype=jnp.float64)
    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()
    potential_stencil = local_stencil_builder(
        phi_jnp,
        geometry,
        PERIODIC_AXES,
        face_bc,
        cut_wall_geometry,
        cut_wall_bc,
    )

    grad_phi_perp = grad_perp_op(potential_stencil, geometry)
    grad_phi_parallel = grad_parallel_op_direct(potential_stencil, geometry)

    metric = geometry.cell_metric
    g_cov = jnp.asarray(metric.g_cov, dtype=jnp.float64)
    e_cov = jnp.einsum("...ij,...j->...i", g_cov, grad_phi_perp)
    b_contra = jnp.asarray(geometry.cell_bfield.B_contra, dtype=jnp.float64)
    b_cov = jnp.einsum("...ij,...j->...i", g_cov, b_contra)
    bmag_sq = jnp.maximum(jnp.square(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64)), 1.0e-30)
    jacobian = jnp.maximum(jnp.asarray(metric.J, dtype=jnp.float64), 1.0e-30)

    exb = jnp.cross(e_cov, b_cov, axis=-1) / (jacobian[..., None] * bmag_sq[..., None])
    speed_xy = jnp.sqrt(jnp.square(exb[..., 0]) + jnp.square(exb[..., 1]))
    parallel_linf = float(jnp.max(jnp.abs(grad_phi_parallel)))
    return (
        np.asarray(exb, dtype=np.float64),
        np.asarray(speed_xy, dtype=np.float64),
        np.asarray(grad_phi_parallel, dtype=np.float64),
        parallel_linf,
    )


def _exb_dot_grad_temperature(temperature: np.ndarray, exb: np.ndarray, geometry) -> np.ndarray:
    temperature_jnp = jnp.asarray(temperature, dtype=jnp.float64)
    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()
    temperature_stencil = local_stencil_builder(
        temperature_jnp,
        geometry,
        PERIODIC_AXES,
        face_bc,
        cut_wall_geometry,
        cut_wall_bc,
    )
    grad_temperature_perp = grad_perp_op(temperature_stencil, geometry)
    exb_jnp = jnp.asarray(exb, dtype=jnp.float64)
    g_cov = jnp.asarray(geometry.cell_metric.g_cov, dtype=jnp.float64)
    return np.asarray(jnp.einsum("...i,...ij,...j->...", exb_jnp, g_cov, grad_temperature_perp), dtype=np.float64)


def _parallel_temperature_terms(
    temperature: np.ndarray,
    velocity: np.ndarray,
    geometry,
) -> tuple[np.ndarray, np.ndarray]:
    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()
    temperature_jnp = jnp.asarray(temperature, dtype=jnp.float64)
    velocity_jnp = jnp.asarray(velocity, dtype=jnp.float64)
    temperature_stencil = local_stencil_builder(
        temperature_jnp,
        geometry,
        PERIODIC_AXES,
        face_bc,
        cut_wall_geometry,
        cut_wall_bc,
    )
    velocity_stencil = local_stencil_builder(
        velocity_jnp,
        geometry,
        PERIODIC_AXES,
        face_bc,
        cut_wall_geometry,
        cut_wall_bc,
    )
    grad_parallel_temperature = grad_parallel_op_direct(temperature_stencil, geometry)
    grad_parallel_velocity = grad_parallel_op_direct(velocity_stencil, geometry)
    parallel_advection = velocity_jnp * grad_parallel_temperature
    parallel_compression = temperature_jnp * grad_parallel_velocity
    return (
        np.asarray(parallel_advection, dtype=np.float64),
        np.asarray(parallel_compression, dtype=np.float64),
    )


def _compute_exb_slice_history(
    phi_history: np.ndarray,
    te_history: np.ndarray,
    ti_history: np.ndarray,
    ve_history: np.ndarray,
    vi_history: np.ndarray,
    geometry,
    z_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    exb_slice_history: list[np.ndarray] = []
    speed_slice_history: list[np.ndarray] = []
    te_advection_slice_history: list[np.ndarray] = []
    ti_advection_slice_history: list[np.ndarray] = []
    te_parallel_advection_slice_history: list[np.ndarray] = []
    ti_parallel_advection_slice_history: list[np.ndarray] = []
    te_parallel_compression_slice_history: list[np.ndarray] = []
    ti_parallel_compression_slice_history: list[np.ndarray] = []
    parallel_linf_history: list[float] = []
    for frame_index, (phi, te, ti, ve, vi) in enumerate(
        zip(
            np.asarray(phi_history, dtype=np.float64),
            np.asarray(te_history, dtype=np.float64),
            np.asarray(ti_history, dtype=np.float64),
            np.asarray(ve_history, dtype=np.float64),
            np.asarray(vi_history, dtype=np.float64),
        )
    ):
        exb, speed, _, parallel_linf = _compute_exb_from_phi(phi, geometry)
        if not np.any(np.isfinite(speed)):
            raise ValueError(f"ExB speed is all non-finite at frame {frame_index}")
        te_advection = _exb_dot_grad_temperature(te, exb, geometry)
        ti_advection = _exb_dot_grad_temperature(ti, exb, geometry)
        te_parallel_advection, te_parallel_compression = _parallel_temperature_terms(te, ve, geometry)
        ti_parallel_advection, ti_parallel_compression = _parallel_temperature_terms(ti, vi, geometry)
        exb_slice_history.append(exb[:, :, z_indices, :])
        speed_slice_history.append(speed[:, :, z_indices])
        te_advection_slice_history.append(te_advection[:, :, z_indices])
        ti_advection_slice_history.append(ti_advection[:, :, z_indices])
        te_parallel_advection_slice_history.append(te_parallel_advection[:, :, z_indices])
        ti_parallel_advection_slice_history.append(ti_parallel_advection[:, :, z_indices])
        te_parallel_compression_slice_history.append(te_parallel_compression[:, :, z_indices])
        ti_parallel_compression_slice_history.append(ti_parallel_compression[:, :, z_indices])
        parallel_linf_history.append(parallel_linf)
    return (
        np.asarray(exb_slice_history, dtype=np.float64),
        np.asarray(speed_slice_history, dtype=np.float64),
        np.asarray(te_advection_slice_history, dtype=np.float64),
        np.asarray(ti_advection_slice_history, dtype=np.float64),
        np.asarray(te_parallel_advection_slice_history, dtype=np.float64),
        np.asarray(ti_parallel_advection_slice_history, dtype=np.float64),
        np.asarray(te_parallel_compression_slice_history, dtype=np.float64),
        np.asarray(ti_parallel_compression_slice_history, dtype=np.float64),
        np.asarray(parallel_linf_history, dtype=np.float64),
    )


def _save_exb_streamline_gif(
    times: np.ndarray,
    phi_history: np.ndarray,
    exb_slice_history: np.ndarray,
    speed_slice_history: np.ndarray,
    parallel_linf_history: np.ndarray,
    geometry,
    *,
    output_path: Path,
    frame_stride: int,
    stream_density: float,
    z_indices: tuple[int, ...],
    title: str = "EB blob ExB drift streamlines",
) -> None:
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    y_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    theta_grid, radius_grid = np.meshgrid(y_values, x_values)

    frame_indices = np.arange(0, int(times.shape[0]), max(1, int(frame_stride)), dtype=np.int64)
    if frame_indices[-1] != int(times.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times.shape[0]) - 1)

    phi_norm = _robust_norm(phi_history[:, :, :, z_indices])
    speed_norm = _positive_norm(speed_slice_history)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(18.0, 4.8),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    phi_mappable = cm.ScalarMappable(norm=phi_norm, cmap="coolwarm")
    speed_mappable = cm.ScalarMappable(norm=speed_norm, cmap="viridis")
    fig.colorbar(phi_mappable, ax=list(axes), location="right", pad=0.02, shrink=0.85, label="phi")
    fig.colorbar(speed_mappable, ax=list(axes), location="bottom", pad=0.08, shrink=0.85, label="|v_E| in x-y plane")
    suptitle = fig.suptitle(title)

    def update(animation_index: int):
        actual_index = int(frame_indices[animation_index])
        time_value = float(times[actual_index])
        returned_artists = []
        for col, z_index in enumerate(z_indices):
            ax = axes[col]
            ax.clear()
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(-1)
            ax.set_ylim(0.0, float(x_values[-1]))
            ax.set_yticklabels([])

            phi_slice = np.asarray(phi_history[actual_index, :, :, z_index], dtype=np.float64)
            exb_slice = np.asarray(exb_slice_history[actual_index, :, :, col, :], dtype=np.float64)
            speed_slice = np.asarray(speed_slice_history[actual_index, :, :, col], dtype=np.float64)
            finite = np.isfinite(phi_slice) & np.isfinite(exb_slice[..., 0]) & np.isfinite(exb_slice[..., 1]) & np.isfinite(speed_slice)
            if not np.any(finite):
                raise ValueError(f"all ExB values are non-finite at frame {actual_index}, z index {z_index}")
            phi_slice = np.where(np.isfinite(phi_slice), phi_slice, np.nan)
            radial_velocity = np.where(finite, exb_slice[..., 0], 0.0)
            angular_velocity = np.where(finite, exb_slice[..., 1], 0.0)
            speed_slice = np.where(np.isfinite(speed_slice), speed_slice, 0.0)

            ax.pcolormesh(
                theta_grid,
                radius_grid,
                phi_slice,
                shading="auto",
                cmap="coolwarm",
                norm=phi_norm,
                alpha=0.28,
            )
            ax.contour(
                theta_grid,
                radius_grid,
                phi_slice,
                levels=7,
                colors="0.25",
                linewidths=0.45,
                alpha=0.35,
            )
            ax.streamplot(
                y_values,
                x_values,
                angular_velocity,
                radial_velocity,
                color=speed_slice,
                cmap="viridis",
                norm=speed_norm,
                density=float(stream_density),
                linewidth=0.8,
                arrowsize=0.8,
            )
            ax.set_title(f"z={z_values[z_index]:.3f}, t={time_value:.3e}")

        suptitle.set_text(
            f"{title}, t={time_value:.3e}, max |grad_parallel phi|={parallel_linf_history[actual_index]:.3e}"
        )
        return returned_artists

    animator = animation.FuncAnimation(fig, update, frames=int(frame_indices.shape[0]), interval=100, blit=False)
    writer = animation.PillowWriter(fps=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animator.save(str(output_path), writer=writer)
    plt.close(fig)


def _save_temperature_advection_gif(
    times: np.ndarray,
    phi_history: np.ndarray,
    term_specs: tuple[tuple[str, np.ndarray], ...],
    geometry,
    *,
    output_path: Path,
    frame_stride: int,
    z_indices: tuple[int, ...],
    title: str = "EB blob temperature advection diagnostics",
) -> None:
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    y_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    theta_grid, radius_grid = np.meshgrid(y_values, x_values)
    frame_indices = np.arange(0, int(times.shape[0]), max(1, int(frame_stride)), dtype=np.int64)
    if frame_indices[-1] != int(times.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times.shape[0]) - 1)

    if not term_specs:
        raise ValueError("term_specs must contain at least one diagnostic field")
    advection_norms = tuple(_robust_norm(field_history, qlo=1.0, qhi=99.0) for _, field_history in term_specs)
    phi_norm = _robust_norm(phi_history[:, :, :, z_indices])

    fig, axes = plt.subplots(
        nrows=len(term_specs),
        ncols=4,
        figsize=(18.0, max(4.8, 4.0 * len(term_specs))),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    if len(term_specs) == 1:
        axes = np.asarray([axes])
    phi_mappable = cm.ScalarMappable(norm=phi_norm, cmap="Greys")
    for row, ((row_label, _), advection_norm) in enumerate(zip(term_specs, advection_norms)):
        advection_mappable = cm.ScalarMappable(norm=advection_norm, cmap="coolwarm")
        fig.colorbar(
            advection_mappable,
            ax=list(axes[row, :]),
            location="right",
            pad=0.02,
            shrink=0.88,
            label=row_label,
        )
    fig.colorbar(
        phi_mappable,
        ax=list(axes.ravel()),
        location="bottom",
        pad=0.07,
        shrink=0.88,
        label="phi contour/background",
    )
    suptitle = fig.suptitle(title)

    def update(animation_index: int):
        actual_index = int(frame_indices[animation_index])
        time_value = float(times[actual_index])
        for row, ((row_label, field_history), advection_norm) in enumerate(zip(term_specs, advection_norms)):
            for col, z_index in enumerate(z_indices):
                ax = axes[row, col]
                ax.clear()
                ax.set_theta_zero_location("E")
                ax.set_theta_direction(-1)
                ax.set_ylim(0.0, float(x_values[-1]))
                ax.set_yticklabels([])

                phi_slice = np.asarray(phi_history[actual_index, :, :, z_index], dtype=np.float64)
                adv_slice = np.asarray(field_history[actual_index, :, :, col], dtype=np.float64)
                finite = np.isfinite(phi_slice) & np.isfinite(adv_slice)
                if not np.any(finite):
                    raise ValueError(
                        f"all temperature-advection values are non-finite at frame {actual_index}, z index {z_index}"
                    )

                phi_slice = np.where(np.isfinite(phi_slice), phi_slice, np.nan)
                adv_slice = np.where(np.isfinite(adv_slice), adv_slice, np.nan)
                ax.pcolormesh(
                    theta_grid,
                    radius_grid,
                    phi_slice,
                    shading="auto",
                    cmap="Greys",
                    norm=phi_norm,
                    alpha=0.18,
                )
                ax.pcolormesh(
                    theta_grid,
                    radius_grid,
                    adv_slice,
                    shading="auto",
                    cmap="coolwarm",
                    norm=advection_norm,
                    alpha=0.82,
                )
                ax.contour(
                    theta_grid,
                    radius_grid,
                    phi_slice,
                    levels=7,
                    colors="0.20",
                    linewidths=0.45,
                    alpha=0.45,
                )
                ax.set_title(f"{row_label}, z={z_values[z_index]:.3f}, t={time_value:.3e}")

        suptitle.set_text(f"{title}, t={time_value:.3e}")
        return []

    animator = animation.FuncAnimation(fig, update, frames=int(frame_indices.shape[0]), interval=100, blit=False)
    writer = animation.PillowWriter(fps=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animator.save(str(output_path), writer=writer)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze EB blob ExB drift and save streamline GIFs.")
    parser.add_argument("--run-name", default="EB_test", help="Run name used for output filenames.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory containing <run_name>_histories.npz and step_dumps/. Defaults to <run_name>_outputs.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Fallback cubic grid resolution when the histories NPZ is unavailable.",
    )
    parser.add_argument(
        "--radial-b-fraction",
        type=float,
        default=radial_b_fraction,
        help="Fallback radial B fraction when the histories NPZ is unavailable.",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Stride through saved steps for GIF frames.")
    parser.add_argument("--stream-density", type=float, default=1.2, help="Matplotlib streamplot density.")
    args = parser.parse_args()

    artifact_stem = _eb_blob_artifact_stem(args.run_name)
    output_dir = _resolve_output_dir(args.run_name, args.output_dir)
    history_path = output_dir / f"{artifact_stem}_histories.npz"
    step_dump_dir = _resolve_step_dump_dir(output_dir)
    metadata = _load_metadata(history_path)
    resolution = int(metadata.get("resolution", args.resolution))
    radial_fraction_value = float(metadata.get("radial_b_fraction", args.radial_b_fraction))

    print(f"loading EB step dumps from {step_dump_dir}", flush=True)
    times, phi_history, te_history, ti_history, ve_history, vi_history = _load_step_history(step_dump_dir)
    geometry = _build_eb_blob_geometry((resolution, resolution, resolution), radial_fraction=radial_fraction_value)
    if tuple(int(value) for value in phi_history.shape[1:]) != tuple(int(value) for value in geometry.shape):
        raise ValueError(
            f"step phi shape {phi_history.shape[1:]} does not match reconstructed geometry shape {geometry.shape}"
        )

    print("computing ExB drift history", flush=True)
    z_indices = _eb_blob_z_indices(geometry, z0)
    (
        exb_slice_history,
        speed_slice_history,
        te_advection_slice_history,
        ti_advection_slice_history,
        te_parallel_advection_slice_history,
        ti_parallel_advection_slice_history,
        te_parallel_compression_slice_history,
        ti_parallel_compression_slice_history,
        parallel_linf_history,
    ) = _compute_exb_slice_history(
        phi_history,
        te_history,
        ti_history,
        ve_history,
        vi_history,
        geometry,
        z_indices,
    )
    output_path = output_dir / f"{artifact_stem}_exb_streamlines.gif"
    print(f"saving ExB streamline GIF to {output_path}", flush=True)
    _save_exb_streamline_gif(
        times,
        phi_history,
        exb_slice_history,
        speed_slice_history,
        parallel_linf_history,
        geometry,
        output_path=output_path,
        frame_stride=int(args.frame_stride),
        stream_density=float(args.stream_density),
        z_indices=z_indices,
    )
    print(f"saved ExB streamline GIF to {output_path}", flush=True)

    te_advection_path = output_dir / f"{artifact_stem}_te_advection_diagnostics.gif"
    print(f"saving Te advection diagnostic GIF to {te_advection_path}", flush=True)
    _save_temperature_advection_gif(
        times,
        phi_history,
        (
            ("-v_E dot grad(Te)", -te_advection_slice_history),
            ("-Ve grad_parallel Te", -te_parallel_advection_slice_history),
            ("-Te grad_parallel Ve", -te_parallel_compression_slice_history),
        ),
        geometry,
        output_path=te_advection_path,
        frame_stride=int(args.frame_stride),
        z_indices=z_indices,
        title="EB blob Te advection diagnostics",
    )
    print(f"saved Te advection diagnostic GIF to {te_advection_path}", flush=True)

    ti_advection_path = output_dir / f"{artifact_stem}_ti_advection_diagnostics.gif"
    print(f"saving Ti advection diagnostic GIF to {ti_advection_path}", flush=True)
    _save_temperature_advection_gif(
        times,
        phi_history,
        (
            ("-v_E dot grad(Ti)", -ti_advection_slice_history),
            ("-Vi grad_parallel Ti", -ti_parallel_advection_slice_history),
            ("-Ti grad_parallel Vi", -ti_parallel_compression_slice_history),
        ),
        geometry,
        output_path=ti_advection_path,
        frame_stride=int(args.frame_stride),
        z_indices=z_indices,
        title="EB blob Ti advection diagnostics",
    )
    print(f"saved Ti advection diagnostic GIF to {ti_advection_path}", flush=True)


if __name__ == "__main__":
    main()
