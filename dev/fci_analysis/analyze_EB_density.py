from __future__ import annotations

import argparse
from functools import partial
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

from jax_drb.geometry import build_curvature_coefficients  # noqa: E402
from jax_drb.native.fci_boundaries import (  # noqa: E402
    BoundaryConditionBuilder,
    CoordinateFaceValueReconstructor3D,
    CoordinateNormalDerivativeConstructor3D,
    CutWallBC3D,
    CutWallGeometry3D,
)
from jax_drb.native.fci_drb_EB_rhs import FciDrbEBState, _multiply_local_stencils  # noqa: E402
from jax_drb.native.fci_operators import (  # noqa: E402
    curvature_op,
    grad_parallel_op_direct,
    parallel_laplacian_direct_op,
    perp_laplacian_conservative_op,
    poisson_bracket_op,
)
from test_shifted_torus_EB_blob import (  # noqa: E402
    DEFAULT_PERP_DIFFUSION,
    AXIS_REGULAR_AXES,
    PERIODIC_AXES,
    _build_eb_blob_geometry,
    _build_eb_blob_parameters,
    _build_eb_boundary_conditions,
    _eb_blob_artifact_stem,
    conservative_stencil_builder,
    local_stencil_builder,
)
from test_mms_shifted_torus_4_field import alpha_value, r0, sigma  # noqa: E402


DENSITY_TERM_NAMES = (
    "ExB bracket",
    "parallel Ve compression",
    "n * parallel Ve compression",
    "Ve * parallel density gradient",
    "parallel particle flux divergence",
    "curvature",
    "perp diffusion",
    "parallel diffusion",
    "total RHS",
)


def _resolve_step_dump_dir(run_name: str, output_path: Path | None) -> Path:
    if output_path is None:
        candidate = _THIS_DIR / run_name
        nested = candidate / "step_dumps"
        if any(nested.glob("step_*.npz")):
            return nested
        if any(candidate.glob("step_*.npz")):
            return candidate
        if candidate.exists():
            return candidate
        return Path(f"{_eb_blob_artifact_stem(run_name)}_step_dumps")

    candidate = Path(output_path)
    if candidate.is_file():
        if candidate.name.startswith("step_") and candidate.suffix == ".npz":
            return candidate.parent
        if candidate.name.endswith("_histories.npz"):
            nested = candidate.parent / "step_dumps"
            if any(nested.glob("step_*.npz")):
                return nested
            if any(candidate.parent.glob("step_*.npz")):
                return candidate.parent
            return nested
        return candidate.parent

    nested = candidate / "step_dumps"
    if any(nested.glob("step_*.npz")):
        return nested
    if any(candidate.glob("step_*.npz")):
        return candidate
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return _THIS_DIR / candidate


def _load_eb_blob_step_history(
    step_dump_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    step_files = sorted(step_dump_dir.glob("step_*.npz"))
    if not step_files:
        raise FileNotFoundError(f"no step_*.npz files found in {step_dump_dir}")

    snapshots: list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for fallback_index, step_file in enumerate(step_files):
        with np.load(step_file, allow_pickle=False) as step:
            step_index = int(step["step_index"]) if "step_index" in step.files else fallback_index
            time_value = float(step["time"]) if "time" in step.files else float(fallback_index)
            snapshots.append(
                (
                    step_index,
                    time_value,
                    np.asarray(step["density"], dtype=np.float64),
                    np.asarray(step["phi"], dtype=np.float64),
                    np.asarray(step["Te"], dtype=np.float64),
                    np.asarray(step["Ti"], dtype=np.float64),
                    np.asarray(step["Vi"], dtype=np.float64),
                    np.asarray(step["Ve"], dtype=np.float64),
                    np.asarray(step["vorticity"], dtype=np.float64),
                )
            )

    snapshots.sort(key=lambda item: item[0])
    times = np.asarray([item[1] for item in snapshots], dtype=np.float64)
    density_history = np.asarray([item[2] for item in snapshots], dtype=np.float64)
    phi_history = np.asarray([item[3] for item in snapshots], dtype=np.float64)
    te_history = np.asarray([item[4] for item in snapshots], dtype=np.float64)
    ti_history = np.asarray([item[5] for item in snapshots], dtype=np.float64)
    vi_history = np.asarray([item[6] for item in snapshots], dtype=np.float64)
    ve_history = np.asarray([item[7] for item in snapshots], dtype=np.float64)
    vorticity_history = np.asarray([item[8] for item in snapshots], dtype=np.float64)
    return times, density_history, phi_history, te_history, ti_history, vi_history, ve_history, vorticity_history


def _cell_volume_weights(geometry) -> np.ndarray:
    return np.asarray(geometry.cell_metric.J, dtype=np.float64) * np.asarray(
        geometry.spacing.dx * geometry.spacing.dy * geometry.spacing.dz,
        dtype=np.float64,
    )


def _density_rhs_terms(
    state: FciDrbEBState,
    *,
    geometry,
    parameters,
    boundary_condition_builder,
    cut_wall_geometry,
    cut_wall_bc,
    curvature_coefficients,
) -> dict[str, np.ndarray]:
    boundary_conditions = boundary_condition_builder(
        state,
        geometry,
        PERIODIC_AXES,
        cut_wall_geometry,
        cut_wall_bc,
    )

    density = jnp.asarray(state.density, dtype=jnp.float64)
    phi = jnp.asarray(state.phi, dtype=jnp.float64)
    Te = jnp.asarray(state.Te, dtype=jnp.float64)
    Ve = jnp.asarray(state.Ve, dtype=jnp.float64)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    density_D_perp = jnp.asarray(parameters.density_D_perp, dtype=jnp.float64)
    density_D_parallel = jnp.asarray(parameters.density_D_parallel, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), 1.0e-30)

    density_stencil = local_stencil_builder(
        density,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.density_face_bc,
        cut_wall_geometry,
        boundary_conditions.density_cut_wall_bc,
    )
    potential_stencil = local_stencil_builder(
        phi,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.potential_face_bc,
        cut_wall_geometry,
        boundary_conditions.potential_cut_wall_bc,
    )
    Pe = density * Te
    Pe_stencil = local_stencil_builder(
        Pe,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Te_face_bc,
        cut_wall_geometry,
        boundary_conditions.Te_cut_wall_bc,
    )
    Ve_stencil = local_stencil_builder(
        Ve,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Ve_face_bc,
        cut_wall_geometry,
        boundary_conditions.Ve_cut_wall_bc,
    )

    poisson_bracket_density = poisson_bracket_op(potential_stencil, density_stencil, geometry)
    parallel_density_flux = grad_parallel_op_direct(Ve_stencil, geometry)
    parallel_particle_flux = grad_parallel_op_direct(_multiply_local_stencils(density_stencil, Ve_stencil), geometry)
    parallel_density_gradient = grad_parallel_op_direct(density_stencil, geometry)
    curvature_Pe = curvature_op(Pe_stencil, geometry, curvature_coefficients=curvature_coefficients)
    curvature_potential = curvature_op(potential_stencil, geometry, curvature_coefficients=curvature_coefficients)

    exb_bracket = -(poisson_bracket_density / (rho_star * bmag))
    parallel_ve_compression = -parallel_density_flux
    parallel_ve_density_weighted_compression = -density * parallel_density_flux
    ve_parallel_density_gradient = -Ve * parallel_density_gradient
    parallel_particle_flux_divergence = -parallel_particle_flux
    curvature = (2.0 / bmag) * (curvature_Pe - density * curvature_potential)

    perp_diffusion = jnp.zeros_like(density)
    if float(density_D_perp) != 0.0:
        density_conservative_stencil = conservative_stencil_builder(
            density,
            geometry,
            PERIODIC_AXES,
            boundary_conditions.density_face_bc,
        )
        perp_diffusion = density_D_perp * perp_laplacian_conservative_op(
            density_conservative_stencil,
            geometry,
            face_bc=boundary_conditions.density_face_bc,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=boundary_conditions.density_cut_wall_bc,
            periodic_axes=PERIODIC_AXES,
        )

    parallel_diffusion = jnp.zeros_like(density)
    if float(density_D_parallel) != 0.0:
        parallel_diffusion = density_D_parallel * parallel_laplacian_direct_op(
            density,
            geometry,
            face_bc=boundary_conditions.density_face_bc,
            periodic_axes=PERIODIC_AXES,
        )

    total = exb_bracket + parallel_particle_flux_divergence + curvature + perp_diffusion + parallel_diffusion
    return {
        "ExB bracket": np.asarray(exb_bracket, dtype=np.float64),
        "parallel Ve compression": np.asarray(parallel_ve_compression, dtype=np.float64),
        "n * parallel Ve compression": np.asarray(parallel_ve_density_weighted_compression, dtype=np.float64),
        "Ve * parallel density gradient": np.asarray(ve_parallel_density_gradient, dtype=np.float64),
        "parallel particle flux divergence": np.asarray(parallel_particle_flux_divergence, dtype=np.float64),
        "curvature": np.asarray(curvature, dtype=np.float64),
        "perp diffusion": np.asarray(perp_diffusion, dtype=np.float64),
        "parallel diffusion": np.asarray(parallel_diffusion, dtype=np.float64),
        "total_rhs": np.asarray(total, dtype=np.float64),
    }


def _minimum_density_index(density: np.ndarray) -> tuple[int, int, int] | None:
    density_np = np.asarray(density, dtype=np.float64)
    finite = np.isfinite(density_np)
    if not np.any(finite):
        return None
    masked = np.where(finite, density_np, np.inf)
    return tuple(int(index) for index in np.unravel_index(int(np.argmin(masked)), density_np.shape))


def _lowest_density_indices(density: np.ndarray, count: int = 10) -> list[tuple[int, int, int]]:
    density_np = np.asarray(density, dtype=np.float64)
    finite_mask = np.isfinite(density_np)
    if not np.any(finite_mask):
        return []

    flat_values = density_np[finite_mask]
    flat_indices = np.flatnonzero(finite_mask)
    count = min(max(0, int(count)), int(flat_values.size))
    if count == 0:
        return []

    selected = np.argpartition(flat_values, count - 1)[:count]
    selected = selected[np.argsort(flat_values[selected])]
    return [
        tuple(int(index) for index in np.unravel_index(int(flat_indices[item]), density_np.shape))
        for item in selected
    ]


def _positivity_timescale(
    field: np.ndarray,
    rhs: np.ndarray,
    *,
    eps: float = 1.0e-300,
) -> tuple[float, tuple[int, int, int] | None]:
    field_np = np.asarray(field, dtype=np.float64)
    rhs_np = np.asarray(rhs, dtype=np.float64)
    finite_mask = np.isfinite(field_np) & np.isfinite(rhs_np)
    valid_mask = finite_mask & (rhs_np < 0.0) & (field_np > 0.0)
    if not np.any(valid_mask):
        return float(np.inf), None

    tau = np.full_like(field_np, np.inf, dtype=np.float64)
    tau[valid_mask] = field_np[valid_mask] / (-rhs_np[valid_mask] + float(eps))
    tau = np.where(np.isfinite(tau), tau, np.inf)
    min_flat_index = int(np.argmin(tau))
    min_index = tuple(int(index) for index in np.unravel_index(min_flat_index, tau.shape))
    return float(np.min(tau)), min_index


def _summarize_term(
    field: np.ndarray,
    term: np.ndarray,
    weights: np.ndarray,
    min_density_index: tuple[int, int, int] | None,
) -> dict[str, float]:
    field_np = np.asarray(field, dtype=np.float64)
    values = np.asarray(term, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    weight_sum = float(np.sum(weights_np))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("cell-volume weights must have a positive finite sum")

    positive = np.maximum(values, 0.0)
    negative = np.minimum(values, 0.0)
    min_density_value = np.nan
    if min_density_index is not None:
        min_density_value = float(values[min_density_index])
    tau_pos, _ = _positivity_timescale(field_np, values)
    return {
        "weighted_l2": float(np.sqrt(np.sum(weights_np * values * values) / weight_sum)),
        "weighted_mean": float(np.sum(weights_np * values) / weight_sum),
        "linf": float(np.nanmax(np.abs(values))),
        "positive_weighted_mean": float(np.sum(weights_np * positive) / weight_sum),
        "negative_weighted_mean": float(np.sum(weights_np * negative) / weight_sum),
        "at_min_density": min_density_value,
        "positivity_timescale": tau_pos,
        "nonfinite_count": float(np.size(values) - np.count_nonzero(np.isfinite(values))),
    }


def _maybe_symlog_axis(ax, values: np.ndarray) -> None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    finite = finite[np.abs(finite) > 0.0]
    if finite.size == 0:
        return
    vmax = float(np.max(np.abs(finite)))
    vmin = float(np.min(np.abs(finite)))
    if vmax / max(vmin, 1.0e-300) < 1.0e3:
        return
    ax.set_yscale("symlog", linthresh=max(vmin, vmax * 1.0e-8, 1.0e-30))


def _robust_positive_norm(values: np.ndarray, *, qlo: float = 2.0, qhi: float = 98.0):
    import matplotlib.colors as colors

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot build a norm from non-finite values")
    lo = float(np.percentile(finite, qlo))
    hi = float(np.percentile(finite, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("cannot build a norm from non-finite values")
    if np.isclose(lo, hi):
        spread = max(abs(lo), 1.0)
        lo, hi = 0.0, lo + spread
    if hi <= lo:
        hi = lo + max(abs(lo), 1.0)
    return colors.Normalize(vmin=lo, vmax=hi)


def _robust_signed_norm(values: np.ndarray, *, qlo: float = 2.0, qhi: float = 98.0):
    import matplotlib.colors as colors

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot build a norm from non-finite values")
    lo = float(np.percentile(finite, qlo))
    hi = float(np.percentile(finite, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("cannot build a norm from non-finite values")
    if np.isclose(lo, hi):
        spread = max(abs(lo), 1.0)
        lo, hi = -spread, spread
    if lo < 0.0 < hi:
        bound = max(abs(lo), abs(hi))
        return colors.TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
    return colors.Normalize(vmin=lo, vmax=hi)


def _save_density_rhs_term_plot(
    times: np.ndarray,
    metrics: dict[str, dict[str, np.ndarray]],
    *,
    output_path: Path,
    title: str,
    reference_dt: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    columns = (
        ("weighted_l2", "J-weighted L2"),
        ("weighted_mean", "J-weighted mean"),
        ("positive_weighted_mean", "J-mean max(term, 0)"),
        ("negative_weighted_mean", "J-mean min(term, 0)"),
        ("positivity_timescale", "positivity timescale"),
    )
    fig, axes = plt.subplots(
        len(DENSITY_TERM_NAMES),
        len(columns),
        figsize=(24.0, 2.6 * len(DENSITY_TERM_NAMES)),
        sharex=True,
        constrained_layout=True,
    )
    times_np = np.asarray(times, dtype=np.float64)
    for row, term_name in enumerate(DENSITY_TERM_NAMES):
        for col, (metric_name, column_title) in enumerate(columns):
            ax = axes[row, col]
            values = np.asarray(metrics[term_name][metric_name], dtype=np.float64)
            plot_values = values
            if metric_name == "positivity_timescale":
                plot_values = np.where(np.isfinite(values) & (values > 0.0), values, np.nan)
                finite_positive = plot_values[np.isfinite(plot_values)]
                if finite_positive.size > 0:
                    ax.set_yscale("log")
                    if reference_dt is not None and np.isfinite(reference_dt) and reference_dt > 0.0:
                        ax.axhline(reference_dt / 2.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7)
                        ax.axhline(reference_dt, color="0.35", linestyle=":", linewidth=1.0, alpha=0.7)
                        if row == 0:
                            ax.legend(["dt/2", "dt"], loc="best", fontsize=8)
            ax.plot(times_np, plot_values, linewidth=1.8)
            ax.grid(True, alpha=0.3)
            if metric_name != "positivity_timescale":
                _maybe_symlog_axis(ax, values)
            if row == 0:
                ax.set_title(column_title)
            if col == 0:
                ax.set_ylabel(term_name)
            if row == len(DENSITY_TERM_NAMES) - 1:
                ax.set_xlabel("t")
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _save_density_rhs_term_data(
    times: np.ndarray,
    metrics: dict[str, dict[str, np.ndarray]],
    *,
    output_path: Path,
) -> None:
    payload: dict[str, np.ndarray] = {
        "times": np.asarray(times, dtype=np.float64),
        "term_names": np.asarray(DENSITY_TERM_NAMES),
    }
    for term_name in DENSITY_TERM_NAMES:
        key_prefix = term_name.lower().replace(" ", "_").replace("-", "_")
        for metric_name, values in metrics[term_name].items():
            payload[f"{key_prefix}_{metric_name}"] = np.asarray(values, dtype=np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def _shifted_torus_cell_center_xyz(geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    zeta_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    rho, theta, zeta = np.meshgrid(rho_values, theta_values, zeta_values, indexing="ij")
    rho_mid = 0.5 * (float(rho_values[0]) + float(rho_values[-1]))
    theta_shift = theta + float(sigma) * (rho - rho_mid)
    major_radius = float(r0) + float(alpha_value) * rho + rho * np.cos(theta_shift)
    x = major_radius * np.cos(zeta)
    y = major_radius * np.sin(zeta)
    z = rho * np.sin(theta_shift)
    return x, y, z


def _set_3d_axes_limits(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    finite_z = z[np.isfinite(z)]
    if finite_x.size == 0 or finite_y.size == 0 or finite_z.size == 0:
        return

    xmin, xmax = float(np.min(finite_x)), float(np.max(finite_x))
    ymin, ymax = float(np.min(finite_y)), float(np.max(finite_y))
    zmin, zmax = float(np.min(finite_z)), float(np.max(finite_z))
    xspan = max(xmax - xmin, 1.0e-12)
    yspan = max(ymax - ymin, 1.0e-12)
    zspan = max(zmax - zmin, 1.0e-12)
    span = max(xspan, yspan, zspan)
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    zmid = 0.5 * (zmin + zmax)
    pad = 0.04 * span
    ax.set_xlim(xmid - 0.5 * span - pad, xmid + 0.5 * span + pad)
    ax.set_ylim(ymid - 0.5 * span - pad, ymid + 0.5 * span + pad)
    ax.set_zlim(zmid - 0.5 * span - pad, zmid + 0.5 * span + pad)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def _save_min_density_cell_movie(
    times: np.ndarray,
    density_history: np.ndarray,
    geometry,
    *,
    output_path: Path,
    title: str,
    frame_stride: int = 1,
    point_stride: int = 3,
    fps: int = 8,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    times_np = np.asarray(times, dtype=np.float64)
    density_np = np.asarray(density_history, dtype=np.float64)
    if density_np.ndim != 4:
        raise ValueError(f"density_history must have shape (time, nx, ny, nz), got {density_np.shape}")
    if times_np.shape[0] != density_np.shape[0]:
        raise ValueError(
            f"times and density_history disagree: {times_np.shape[0]} times for {density_np.shape[0]} frames"
        )

    x, y, z = _shifted_torus_cell_center_xyz(geometry)
    frame_stride = max(1, int(frame_stride))
    point_stride = max(1, int(point_stride))
    fps = max(1, int(fps))
    frame_indices = np.arange(0, int(times_np.shape[0]), frame_stride, dtype=np.int64)
    if frame_indices[-1] != int(times_np.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times_np.shape[0]) - 1)

    top_count = 10
    min_indices: list[tuple[int, int, int] | None] = []
    min_values = np.full(int(times_np.shape[0]), np.nan, dtype=np.float64)
    min_xyz = np.full((int(times_np.shape[0]), 3), np.nan, dtype=np.float64)
    lowest_xyz = np.full((int(times_np.shape[0]), top_count, 3), np.nan, dtype=np.float64)
    lowest_values = np.full((int(times_np.shape[0]), top_count), np.nan, dtype=np.float64)
    for frame_index, density in enumerate(density_np):
        lowest_indices = _lowest_density_indices(density, count=top_count)
        min_index = lowest_indices[0] if lowest_indices else None
        min_indices.append(min_index)
        if min_index is None:
            continue
        for rank, cell_index in enumerate(lowest_indices[:top_count]):
            lowest_xyz[frame_index, rank] = (float(x[cell_index]), float(y[cell_index]), float(z[cell_index]))
            lowest_values[frame_index, rank] = float(density[cell_index])
        min_values[frame_index] = float(density[min_index])
        min_xyz[frame_index] = (float(x[min_index]), float(y[min_index]), float(z[min_index]))

    if not np.any(np.isfinite(min_values)):
        raise ValueError("cannot build min-density movie because no finite density values were found")

    sample = (
        slice(None, None, point_stride),
        slice(None, None, point_stride),
        slice(None, None, point_stride),
    )
    sample_x = x[sample].ravel()
    sample_y = y[sample].ravel()
    sample_z = z[sample].ravel()

    fig = plt.figure(figsize=(9.0, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        sample_x,
        sample_y,
        sample_z,
        s=2.0,
        c="0.72",
        alpha=0.12,
        linewidths=0.0,
        depthshade=False,
    )
    first_valid_frames = [int(index) for index in frame_indices if np.all(np.isfinite(min_xyz[int(index)]))]
    first_frame = first_valid_frames[0] if first_valid_frames else int(np.flatnonzero(np.isfinite(min_values))[0])
    rank_colors = plt.get_cmap("plasma_r")(np.linspace(0.05, 0.95, top_count))
    rank_sizes = np.linspace(180.0, 70.0, top_count, dtype=np.float64)
    first_lowest = lowest_xyz[first_frame]
    first_lowest_mask = np.all(np.isfinite(first_lowest), axis=1)
    marker = ax.scatter(
        first_lowest[first_lowest_mask, 0],
        first_lowest[first_lowest_mask, 1],
        first_lowest[first_lowest_mask, 2],
        s=rank_sizes[first_lowest_mask],
        c=rank_colors[first_lowest_mask],
        edgecolors="black",
        linewidths=0.9,
        depthshade=False,
        label="lowest 10 density cells",
    )
    trail, = ax.plot([], [], [], color="crimson", linewidth=2.0, alpha=0.75)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=24.0, azim=38.0)
    ax.legend(loc="upper left")
    _set_3d_axes_limits(ax, x, y, z)
    title_artist = ax.set_title("")

    def update(movie_frame_index: int):
        actual_index = int(frame_indices[movie_frame_index])
        current_lowest = lowest_xyz[actual_index]
        current_mask = np.all(np.isfinite(current_lowest), axis=1)
        if np.any(current_mask):
            current_xyz = current_lowest[current_mask]
            marker._offsets3d = (current_xyz[:, 0], current_xyz[:, 1], current_xyz[:, 2])
            marker.set_facecolor(rank_colors[current_mask])
            marker.set_edgecolor("black")
            marker.set_sizes(rank_sizes[current_mask])
        else:
            marker._offsets3d = ([], [], [])
            marker.set_facecolor(np.empty((0, 4), dtype=np.float64))
            marker.set_sizes([])

        trail_points = min_xyz[: actual_index + 1]
        finite_trail = np.all(np.isfinite(trail_points), axis=1)
        trail.set_data(trail_points[finite_trail, 0], trail_points[finite_trail, 1])
        trail.set_3d_properties(trail_points[finite_trail, 2])
        min_index = min_indices[actual_index]
        index_text = "none" if min_index is None else f"({min_index[0]}, {min_index[1]}, {min_index[2]})"
        min_rank_value = float(min_values[actual_index])
        tenth_rank_value = float(lowest_values[actual_index, np.count_nonzero(current_mask) - 1]) if np.any(current_mask) else float("nan")
        title_artist.set_text(
            f"{title}\n"
            f"step={actual_index}, t={float(times_np[actual_index]):.6e}, "
            f"min density={min_rank_value:.6e}, 10th lowest={tenth_rank_value:.6e}, index={index_text}"
        )
        return marker, trail, title_artist

    animator = animation.FuncAnimation(
        fig,
        update,
        frames=int(frame_indices.shape[0]),
        interval=max(1, int(round(1000.0 / float(fps)))),
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    animator.save(output_path, writer=writer)
    plt.close(fig)


def _save_outer_wall_slice_physics_movie(
    times: np.ndarray,
    density_history: np.ndarray,
    phi_history: np.ndarray,
    te_history: np.ndarray,
    ti_history: np.ndarray,
    vi_history: np.ndarray,
    ve_history: np.ndarray,
    vorticity_history: np.ndarray,
    geometry,
    boundary_condition_builder,
    cut_wall_geometry,
    cut_wall_bc,
    *,
    output_path: Path,
    title: str = "Outer wall slice physics",
    z_index: int | None = None,
    radial_cells: int = 5,
    fps: int = 8,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    times_np = np.asarray(times, dtype=np.float64)
    density_np = np.asarray(density_history, dtype=np.float64)
    phi_np = np.asarray(phi_history, dtype=np.float64)
    te_np = np.asarray(te_history, dtype=np.float64)
    ti_np = np.asarray(ti_history, dtype=np.float64)
    vi_np = np.asarray(vi_history, dtype=np.float64)
    ve_np = np.asarray(ve_history, dtype=np.float64)
    vorticity_np = np.asarray(vorticity_history, dtype=np.float64)
    histories = {
        "density": density_np,
        "phi": phi_np,
        "Te": te_np,
        "Ti": ti_np,
        "Vi": vi_np,
        "Ve": ve_np,
        "vorticity": vorticity_np,
    }
    num_frames = int(times_np.shape[0])
    if num_frames == 0:
        raise ValueError("cannot build a wall physics movie from an empty history")
    for name, history in histories.items():
        if history.shape[0] != num_frames:
            raise ValueError(
                f"{name}_history disagrees with times: {history.shape[0]} frames for {num_frames} times"
            )

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    y_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    if x_values.size == 0 or y_values.size == 0 or z_values.size == 0:
        raise ValueError("geometry must have non-empty x, y, and z grids")

    if z_index is None:
        z_index = int(z_values.shape[0] // 2)
    z_index = int(z_index)
    if z_index < 0 or z_index >= int(z_values.shape[0]):
        raise ValueError(f"wall z_index must be in [0, {int(z_values.shape[0]) - 1}], got {z_index}")

    radial_cells = max(1, int(radial_cells))
    radial_cells = min(radial_cells, int(x_values.shape[0]))
    radial_start = int(x_values.shape[0]) - radial_cells
    wall_x_values = x_values[radial_start:]
    wall_x_widths = np.asarray(geometry.grid.x.widths, dtype=np.float64)[radial_start:]
    wall_radius_min = float(wall_x_values[0] - 0.5 * wall_x_widths[0])
    wall_radius_max = float(wall_x_values[-1] + 0.5 * wall_x_widths[-1])

    fps = max(1, int(fps))
    theta_grid, radius_grid = np.meshgrid(y_values, wall_x_values)

    outer_ring_template = np.full((int(wall_x_values.shape[0]), int(y_values.shape[0])), np.nan, dtype=np.float64)

    bn_outer = np.asarray(geometry.face_bfield.x.B_contra[-1, :, z_index, 0], dtype=np.float64)
    bn_abs_outer = np.abs(bn_outer)
    sign_outer = np.sign(bn_outer)
    sign_outer = np.where(np.isfinite(sign_outer), sign_outer, 0.0)
    sign_flip_outer = (sign_outer * np.roll(sign_outer, 1) < 0.0).astype(np.float64)

    ve_wall_history = np.full((num_frames, int(y_values.shape[0])), np.nan, dtype=np.float64)
    grad_parallel_ve_history = np.full((num_frames, int(y_values.shape[0])), np.nan, dtype=np.float64)
    minus_n_grad_parallel_ve_history = np.full((num_frames, int(y_values.shape[0])), np.nan, dtype=np.float64)
    for frame_index in range(num_frames):
        state = FciDrbEBState(
            density=jnp.asarray(density_np[frame_index], dtype=jnp.float64),
            phi=jnp.asarray(phi_np[frame_index], dtype=jnp.float64),
            Te=jnp.asarray(te_np[frame_index], dtype=jnp.float64),
            Ti=jnp.asarray(ti_np[frame_index], dtype=jnp.float64),
            Vi=jnp.asarray(vi_np[frame_index], dtype=jnp.float64),
            Ve=jnp.asarray(ve_np[frame_index], dtype=jnp.float64),
            vorticity=jnp.asarray(vorticity_np[frame_index], dtype=jnp.float64),
        )
        boundary_conditions = boundary_condition_builder(
            state,
            geometry,
            PERIODIC_AXES,
            cut_wall_geometry,
            cut_wall_bc,
        )
        ve_wall_history[frame_index] = np.asarray(
            boundary_conditions.Ve_face_bc.value_x[-1, :, z_index],
            dtype=np.float64,
        )
        ve_stencil = local_stencil_builder(
            state.Ve,
            geometry,
            PERIODIC_AXES,
            boundary_conditions.Ve_face_bc,
            cut_wall_geometry,
            boundary_conditions.Ve_cut_wall_bc,
        )
        parallel_ve = np.asarray(grad_parallel_op_direct(ve_stencil, geometry), dtype=np.float64)
        grad_parallel_ve_history[frame_index] = parallel_ve[-1, :, z_index]
        minus_n_grad_parallel_ve_history[frame_index] = -density_np[frame_index, -1, :, z_index] * grad_parallel_ve_history[
            frame_index
        ]

    panel_names = (
        "b·n",
        "|b·n|",
        "sign-flip mask",
        "Ve_wall",
        "∇_parallel Ve",
        "-n ∇_parallel Ve",
    )
    panel_data = (
        np.broadcast_to(bn_outer[None, :], (num_frames, bn_outer.shape[0])),
        np.broadcast_to(bn_abs_outer[None, :], (num_frames, bn_abs_outer.shape[0])),
        np.broadcast_to(sign_flip_outer[None, :], (num_frames, sign_flip_outer.shape[0])),
        ve_wall_history,
        grad_parallel_ve_history,
        minus_n_grad_parallel_ve_history,
    )
    panel_norms = (
        _robust_signed_norm(bn_outer),
        _robust_positive_norm(bn_abs_outer),
        colors.Normalize(vmin=0.0, vmax=1.0),
        _robust_positive_norm(ve_wall_history),
        _robust_signed_norm(grad_parallel_ve_history),
        _robust_signed_norm(minus_n_grad_parallel_ve_history),
    )
    panel_cmaps = (
        plt.get_cmap("coolwarm").copy(),
        plt.get_cmap("viridis").copy(),
        plt.get_cmap("Reds").copy(),
        plt.get_cmap("viridis").copy(),
        plt.get_cmap("coolwarm").copy(),
        plt.get_cmap("coolwarm").copy(),
    )
    for cmap in panel_cmaps:
        cmap.set_bad(alpha=0.0)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17.0, 11.5),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes).ravel()
    images = []
    for ax, panel_name, cmap, norm in zip(axes_flat, panel_names, panel_cmaps, panel_norms):
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_ylim(wall_radius_min, wall_radius_max)
        ax.set_yticklabels([])
        ax.set_title(panel_name)
        image = ax.pcolormesh(
            theta_grid,
            radius_grid,
            outer_ring_template,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        fig.colorbar(image, ax=ax, location="right", pad=0.02, shrink=0.88)
        images.append(image)

    title_artist = fig.suptitle("")

    def update(frame_index: int):
        time_value = float(times_np[frame_index])
        for image, data_history in zip(images, panel_data):
            frame_data = outer_ring_template.copy()
            frame_data[-1, :] = data_history[frame_index]
            image.set_array(np.ma.masked_invalid(frame_data).ravel())
        title_artist.set_text(
            f"{title}\n"
            f"step={frame_index}, t={time_value:.6e}, z-index={z_index}, z={float(z_values[z_index]):.6e}, "
            f"outer radial cells={radial_cells}"
        )
        return (*images, title_artist)

    animator = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=max(1, int(round(1000.0 / float(fps)))),
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    animator.save(output_path, writer=writer)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved EB blob density RHS terms.")
    parser.add_argument(
        "--run-name",
        "--run_name",
        dest="run_name",
        default="EB_perp_diffusion",
        help="Name of the step-dump directory under tests/, e.g. EB_perp_diffusion.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Step-dump directory. Defaults to tests/<run_name>, falling back to <run_name>_step_dumps.",
    )
    parser.add_argument(
        "--perp-diffusion",
        type=float,
        default=DEFAULT_PERP_DIFFUSION,
        help="Perpendicular diffusion coefficient used for the density diffusion term.",
    )
    parser.add_argument(
        "--movie-frame-stride",
        type=int,
        default=1,
        help="Use every Nth saved timestep in the min-density 3D movie.",
    )
    parser.add_argument(
        "--movie-point-stride",
        type=int,
        default=3,
        help="Use every Nth grid point along each axis for the static 3D torus context cloud.",
    )
    parser.add_argument(
        "--movie-fps",
        type=int,
        default=8,
        help="Frames per second for the min-density 3D GIF.",
    )
    parser.add_argument(
        "--wall-z-index",
        type=int,
        default=None,
        help="Z slice to visualize in the outer-wall physics GIF; defaults to the middle slice.",
    )
    parser.add_argument(
        "--wall-movie-fps",
        type=int,
        default=8,
        help="Frames per second for the outer-wall physics GIF.",
    )
    parser.add_argument(
        "--skip-min-density-movie",
        action="store_true",
        help="Skip writing the 3D movie that tracks the minimum-density cell.",
    )
    args = parser.parse_args()

    step_dump_dir = _resolve_step_dump_dir(args.run_name, args.output_path)
    if not step_dump_dir.exists():
        raise FileNotFoundError(f"missing EB blob step dump directory: {step_dump_dir}")

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

    shape = tuple(int(size) for size in density_history.shape[1:])
    geometry = _build_eb_blob_geometry(shape, construct_fci_maps=False)
    parameters = _build_eb_blob_parameters(float(args.perp_diffusion))
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()
    coordinate_face_reconstructor = CoordinateFaceValueReconstructor3D()
    coordinate_normal_derivative_constructor = CoordinateNormalDerivativeConstructor3D.from_geometry(geometry)
    boundary_condition_builder = BoundaryConditionBuilder(
        partial(
            _build_eb_boundary_conditions,
            face_reconstructor=coordinate_face_reconstructor,
            normal_derivative_constructor=coordinate_normal_derivative_constructor,
        )
    )
    curvature_coefficients = build_curvature_coefficients(
        geometry,
        periodic_axes=PERIODIC_AXES,
        axis_regular_axes=AXIS_REGULAR_AXES,
    )
    weights = _cell_volume_weights(geometry)

    metric_names = (
        "weighted_l2",
        "weighted_mean",
        "linf",
        "positive_weighted_mean",
        "negative_weighted_mean",
        "at_min_density",
        "positivity_timescale",
        "nonfinite_count",
    )
    metrics: dict[str, dict[str, list[float]]] = {
        term_name: {metric_name: [] for metric_name in metric_names} for term_name in DENSITY_TERM_NAMES
    }

    print(f"analyzing density RHS terms from {step_dump_dir}", flush=True)
    for index, time_value in enumerate(times):
        state = FciDrbEBState(
            density=jnp.asarray(density_history[index], dtype=jnp.float64),
            phi=jnp.asarray(phi_history[index], dtype=jnp.float64),
            Te=jnp.asarray(te_history[index], dtype=jnp.float64),
            Ti=jnp.asarray(ti_history[index], dtype=jnp.float64),
            Vi=jnp.asarray(vi_history[index], dtype=jnp.float64),
            Ve=jnp.asarray(ve_history[index], dtype=jnp.float64),
            vorticity=jnp.asarray(vorticity_history[index], dtype=jnp.float64),
        )
        min_density_index = _minimum_density_index(density_history[index])
        terms = _density_rhs_terms(
            state,
            geometry=geometry,
            parameters=parameters,
            boundary_condition_builder=boundary_condition_builder,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=cut_wall_bc,
            curvature_coefficients=curvature_coefficients,
        )
        for term_name in DENSITY_TERM_NAMES:
            summary = _summarize_term(density_history[index], terms[term_name], weights, min_density_index)
            for metric_name in metric_names:
                metrics[term_name][metric_name].append(summary[metric_name])
        if index == 0 or index == len(times) - 1 or (index + 1) % 10 == 0:
            print(f"processed density RHS terms for step {index + 1}/{len(times)} at t={time_value:.6e}", flush=True)

    metrics_np: dict[str, dict[str, np.ndarray]] = {
        term_name: {
            metric_name: np.asarray(values, dtype=np.float64)
            for metric_name, values in term_metrics.items()
        }
        for term_name, term_metrics in metrics.items()
    }
    nonfinite_totals = {
        term_name: float(np.nansum(term_metrics["nonfinite_count"]))
        for term_name, term_metrics in metrics_np.items()
    }
    nonfinite_report = ", ".join(f"{name}={count:.0f}" for name, count in nonfinite_totals.items())
    print(f"density RHS nonfinite sample counts: {nonfinite_report}", flush=True)

    artifact_stem = _eb_blob_artifact_stem(args.run_name)
    plot_path = step_dump_dir / f"{artifact_stem}_density_rhs_terms.png"
    data_path = step_dump_dir / f"{artifact_stem}_density_rhs_terms.npz"
    movie_path = step_dump_dir / f"{artifact_stem}_min_density_cell_3d.gif"
    wall_physics_movie_path = step_dump_dir / f"{artifact_stem}_outer_wall_physics_slice.gif"
    dt_estimate = float(np.median(np.diff(np.asarray(times, dtype=np.float64)))) if len(times) > 1 else None
    _save_density_rhs_term_plot(
        times,
        metrics_np,
        output_path=plot_path,
        title=f"EB blob density RHS term diagnostics, D_perp={float(args.perp_diffusion):.3e}",
        reference_dt=dt_estimate,
    )
    _save_density_rhs_term_data(times, metrics_np, output_path=data_path)
    print(f"saved density RHS term plot to {plot_path}", flush=True)
    print(f"saved density RHS term data to {data_path}", flush=True)
    if not args.skip_min_density_movie:
        print(f"saving min-density 3D movie to {movie_path}", flush=True)
        _save_min_density_cell_movie(
            times,
            density_history,
            geometry,
            output_path=movie_path,
            title="EB blob minimum-density cell trajectory",
            frame_stride=int(args.movie_frame_stride),
            point_stride=int(args.movie_point_stride),
            fps=int(args.movie_fps),
        )
        print(f"saved min-density 3D movie to {movie_path}", flush=True)

    print(f"saving outer-wall physics movie to {wall_physics_movie_path}", flush=True)
    _save_outer_wall_slice_physics_movie(
        times,
        density_history,
        phi_history,
        te_history,
        ti_history,
        vi_history,
        ve_history,
        vorticity_history,
        geometry,
        boundary_condition_builder,
        cut_wall_geometry,
        cut_wall_bc,
        output_path=wall_physics_movie_path,
        title="Outer wall slice physics at fixed z",
        z_index=args.wall_z_index,
        fps=int(args.wall_movie_fps),
    )
    print(f"saved outer-wall physics movie to {wall_physics_movie_path}", flush=True)


if __name__ == "__main__":
    main()
