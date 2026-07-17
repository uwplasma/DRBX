from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Sequence
import sys

import jax.numpy as jnp
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from analyze_EB_density import _load_eb_blob_step_history, _resolve_step_dump_dir  # noqa: E402
from drbx.geometry import build_curvature_coefficients  # noqa: E402
from drbx.native.fci_boundaries import (  # noqa: E402
    BoundaryConditionBuilder,
    CoordinateFaceValueReconstructor3D,
    CoordinateNormalDerivativeConstructor3D,
    CutWallBC3D,
    CutWallGeometry3D,
)
from drbx.native.fci_drb_EB_rhs import FciDrbEBState  # noqa: E402
from test_shifted_torus_EB_blob import (  # noqa: E402
    AXIS_REGULAR_AXES,
    DEFAULT_PERP_DIFFUSION,
    PERIODIC_AXES,
    _build_eb_blob_geometry,
    _build_eb_blob_parameters,
    _build_eb_boundary_conditions,
    _eb_blob_artifact_stem,
    conservative_stencil_builder,
    local_stencil_builder,
    z0,
)

load_eb_blob_step_history = _load_eb_blob_step_history
resolve_step_dump_dir = _resolve_step_dump_dir
eb_blob_artifact_stem = _eb_blob_artifact_stem


@dataclass(frozen=True)
class RhsTermSpec:
    key: str
    label: str
    color: str


@dataclass(frozen=True)
class CellTarget:
    name: str
    label: str
    theta_target: float


@dataclass(frozen=True)
class CellSelection:
    name: str
    label: str
    x_target: float
    theta_target: float
    z_target: float
    x_index: int
    theta_index: int
    z_index: int
    x_value: float
    theta_value: float
    z_value: float


@dataclass(frozen=True)
class EBBlobContext:
    geometry: object
    parameters: object
    boundary_condition_builder: BoundaryConditionBuilder
    cut_wall_geometry: CutWallGeometry3D
    cut_wall_bc: CutWallBC3D
    curvature_coefficients: np.ndarray


@dataclass(frozen=True)
class FieldAnalyzerSpec:
    display_name: str
    output_suffix: str
    term_specs: tuple[RhsTermSpec, ...]
    sum_keys: tuple[str, ...]
    evaluate_terms: Callable[..., dict[str, np.ndarray]]
    legend_ncol: int = 4


TERM_COLOR_CYCLE: tuple[str, ...] = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
)

DEFAULT_CELL_TARGETS: tuple[CellTarget, ...] = (
    CellTarget("midplane", "theta ~= pi", np.pi),
    CellTarget("plus_45", "theta ~= pi + 45 deg", np.pi + 0.25 * np.pi),
    CellTarget("minus_45", "theta ~= pi - 45 deg", np.pi - 0.25 * np.pi),
    CellTarget("plus_90", "theta ~= pi + 90 deg", np.pi + 0.50 * np.pi),
    CellTarget("minus_90", "theta ~= pi - 90 deg", np.pi - 0.50 * np.pi),
)


def build_term_specs(
    items: Sequence[tuple[str, str]],
    *,
    colors: Sequence[str] = TERM_COLOR_CYCLE,
) -> tuple[RhsTermSpec, ...]:
    return tuple(
        RhsTermSpec(key=key, label=label, color=colors[index % len(colors)])
        for index, (key, label) in enumerate(items)
    )


def nearest_index(values: np.ndarray, target: float) -> int:
    values_np = np.asarray(values, dtype=np.float64)
    if values_np.ndim != 1 or values_np.size == 0:
        raise ValueError(f"values must be a non-empty 1D array, got shape {values_np.shape}")
    return int(np.argmin(np.abs(values_np - float(target))))


def nearest_periodic_index(values: np.ndarray, target: float) -> int:
    values_np = np.asarray(values, dtype=np.float64)
    if values_np.ndim != 1 or values_np.size == 0:
        raise ValueError(f"values must be a non-empty 1D array, got shape {values_np.shape}")
    deltas = np.arctan2(np.sin(values_np - float(target)), np.cos(values_np - float(target)))
    return int(np.argmin(np.abs(deltas)))


def build_cell_selections(
    geometry,
    *,
    cell_targets: Sequence[CellTarget] = DEFAULT_CELL_TARGETS,
    x_target: float = 0.5,
    z_target: float = z0,
) -> tuple[CellSelection, ...]:
    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    x_index = nearest_index(x_values, float(x_target))
    z_index = nearest_periodic_index(z_values, float(z_target))

    selections: list[CellSelection] = []
    for target in cell_targets:
        theta_index = nearest_periodic_index(theta_values, float(target.theta_target))
        selections.append(
            CellSelection(
                name=target.name,
                label=target.label,
                x_target=float(x_target),
                theta_target=float(target.theta_target),
                z_target=float(z_target),
                x_index=x_index,
                theta_index=theta_index,
                z_index=z_index,
                x_value=float(x_values[x_index]),
                theta_value=float(theta_values[theta_index]),
                z_value=float(z_values[z_index]),
            )
        )
    return tuple(selections)


def build_boundary_condition_builder(geometry) -> BoundaryConditionBuilder:
    coordinate_face_reconstructor = CoordinateFaceValueReconstructor3D()
    coordinate_normal_derivative_constructor = CoordinateNormalDerivativeConstructor3D.from_geometry(geometry)
    return BoundaryConditionBuilder(
        partial(
            _build_eb_boundary_conditions,
            face_reconstructor=coordinate_face_reconstructor,
            normal_derivative_constructor=coordinate_normal_derivative_constructor,
        )
    )


def build_eb_blob_context(density_history: np.ndarray, perp_diffusion: float) -> EBBlobContext:
    shape = tuple(int(size) for size in np.asarray(density_history).shape[1:])
    geometry = _build_eb_blob_geometry(shape, construct_fci_maps=False)
    parameters = _build_eb_blob_parameters(float(perp_diffusion))
    boundary_condition_builder = build_boundary_condition_builder(geometry)
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()
    curvature_coefficients = build_curvature_coefficients(
        geometry,
        periodic_axes=PERIODIC_AXES,
        axis_regular_axes=AXIS_REGULAR_AXES,
    )
    return EBBlobContext(
        geometry=geometry,
        parameters=parameters,
        boundary_condition_builder=boundary_condition_builder,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        curvature_coefficients=curvature_coefficients,
    )


def _robust_positive_norm(values: np.ndarray, *, qlo: float = 2.0, qhi: float = 98.0):
    import matplotlib.colors as colors

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot normalize non-finite values")
    lo = float(np.percentile(finite, qlo))
    hi = float(np.percentile(finite, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("cannot normalize non-finite percentile bounds")
    if np.isclose(lo, hi):
        spread = max(abs(lo), abs(hi), 1.0)
        lo, hi = 0.0, lo + spread
    if hi <= lo:
        hi = lo + max(abs(lo), 1.0)
    return colors.Normalize(vmin=lo, vmax=hi)


def collect_rhs_histories(
    times: np.ndarray,
    density_history: np.ndarray,
    phi_history: np.ndarray,
    te_history: np.ndarray,
    ti_history: np.ndarray,
    vi_history: np.ndarray,
    ve_history: np.ndarray,
    vorticity_history: np.ndarray,
    *,
    field_spec: FieldAnalyzerSpec,
    geometry,
    parameters,
    boundary_condition_builder,
    cut_wall_geometry,
    cut_wall_bc,
    curvature_coefficients,
    cell_selections: Sequence[CellSelection],
) -> dict[str, dict[str, np.ndarray]]:
    total_steps = int(np.asarray(density_history).shape[0])
    report_interval = max(1, total_steps // 10)
    cell_histories: dict[str, dict[str, list[float]]] = {
        selection.name: {spec.key: [] for spec in field_spec.term_specs}
        for selection in cell_selections
    }
    print(
        f"analyzing {field_spec.display_name} RHS terms for {total_steps} saved steps "
        f"and {len(cell_selections)} selected cells",
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
        term_fields = field_spec.evaluate_terms(
            state,
            geometry=geometry,
            parameters=parameters,
            boundary_condition_builder=boundary_condition_builder,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=cut_wall_bc,
            curvature_coefficients=curvature_coefficients,
        )
        if "total_rhs" not in term_fields:
            raise KeyError(f"{field_spec.display_name} evaluator must return a total_rhs entry")
        total_from_terms = np.sum([np.asarray(term_fields[key], dtype=np.float64) for key in field_spec.sum_keys], axis=0)
        np.testing.assert_allclose(
            total_from_terms,
            np.asarray(term_fields["total_rhs"], dtype=np.float64),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        for selection in cell_selections:
            cell_index = (selection.x_index, selection.theta_index, selection.z_index)
            for spec in field_spec.term_specs:
                cell_histories[selection.name][spec.key].append(float(term_fields[spec.key][cell_index]))
        if index == 0 or (index + 1) % report_interval == 0 or index + 1 == total_steps:
            time_value = float(np.asarray(times[index], dtype=np.float64))
            print(
                f"processed {field_spec.display_name} RHS terms for step {index + 1}/{total_steps} "
                f"at t={time_value:.6e}",
                flush=True,
            )

    return {
        cell_name: {
            term_name: np.asarray(term_values, dtype=np.float64)
            for term_name, term_values in term_history.items()
        }
        for cell_name, term_history in cell_histories.items()
    }


def build_rhs_terms_figure(
    times: np.ndarray,
    cell_histories: dict[str, dict[str, np.ndarray]],
    density_slice: np.ndarray,
    geometry,
    cell_selections: Sequence[CellSelection],
    z_index: int,
    *,
    field_spec: FieldAnalyzerSpec,
    title: str,
):
    import matplotlib.pyplot as plt

    times_np = np.asarray(times, dtype=np.float64)
    density_slice_np = np.asarray(density_slice, dtype=np.float64)
    if density_slice_np.ndim != 2:
        raise ValueError(f"density_slice must be a 2D array, got shape {density_slice_np.shape}")

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    theta_grid, radius_grid = np.meshgrid(theta_values, x_values)

    fig = plt.figure(figsize=(25.0, 11.5), constrained_layout=True)
    grid_spec = fig.add_gridspec(
        nrows=3,
        ncols=3,
        width_ratios=(3.1, 3.1, 2.1),
        wspace=0.16,
        hspace=0.10,
    )
    trace_axes = [fig.add_subplot(grid_spec[0, 0])]
    trace_axes.append(fig.add_subplot(grid_spec[1, 0], sharex=trace_axes[0]))
    trace_axes.append(fig.add_subplot(grid_spec[2, 0], sharex=trace_axes[0]))
    trace_axes.append(fig.add_subplot(grid_spec[0, 1], sharex=trace_axes[0]))
    trace_axes.append(fig.add_subplot(grid_spec[1, 1], sharex=trace_axes[0]))
    cross_ax = fig.add_subplot(grid_spec[:, 2], projection="polar")

    density_norm = _robust_positive_norm(density_slice_np)
    density_image = cross_ax.pcolormesh(
        theta_grid,
        radius_grid,
        density_slice_np,
        shading="auto",
        cmap="viridis",
        norm=density_norm,
    )
    density_colorbar = fig.colorbar(
        density_image,
        ax=cross_ax,
        location="right",
        pad=0.03,
        shrink=0.88,
    )

    for selection, row_ax in zip(cell_selections, trace_axes):
        for spec in field_spec.term_specs:
            row_ax.plot(
                times_np,
                np.asarray(cell_histories[selection.name][spec.key], dtype=np.float64),
                color=spec.color,
                linewidth=1.8,
                label=spec.label,
            )
        row_ax.axhline(0.0, color="0.55", linewidth=0.8, alpha=0.55)
        row_ax.grid(True, alpha=0.25)
        row_ax.tick_params(axis="both", labelsize=9)
        row_ax.set_yscale("linear")
        row_ax.set_title(
            f"{selection.label} | x ~= {selection.x_value:.3f}, z ~= {selection.z_value:.3f}",
            fontsize=11,
        )
        row_ax.set_ylabel(f"{field_spec.display_name} RHS", fontsize=11, labelpad=12)
        row_ax.legend(
            loc="upper left",
            ncol=field_spec.legend_ncol,
            fontsize=6.3,
            frameon=False,
            handlelength=1.4,
            columnspacing=0.8,
            borderaxespad=0.2,
        )

    for row_ax in trace_axes:
        row_ax.tick_params(labelbottom=False)
    trace_axes[2].set_xlabel("time")
    trace_axes[2].tick_params(labelbottom=True)
    trace_axes[4].set_xlabel("time")
    trace_axes[4].tick_params(labelbottom=True)
    cross_ax.set_theta_zero_location("E")
    cross_ax.set_theta_direction(-1)
    cross_ax.set_ylim(0.0, float(x_values[-1]))
    cross_ax.set_yticklabels([])
    cross_ax.tick_params(labelsize=9)
    cross_ax.set_title(f"initial density at z ~= {float(z_values[z_index]):.3f}", fontsize=12)
    for selection in cell_selections:
        cross_ax.scatter(
            selection.theta_value,
            selection.x_value,
            marker="x",
            s=130.0,
            linewidths=2.4,
            color="red",
            zorder=3,
        )
    fig.suptitle(title)
    density_colorbar.ax.tick_params(labelsize=9)
    density_colorbar.set_label("initial density", fontsize=10, labelpad=10)
    return fig, trace_axes, cross_ax, density_colorbar.ax


def save_rhs_terms_plot(
    times: np.ndarray,
    cell_histories: dict[str, dict[str, np.ndarray]],
    density_slice: np.ndarray,
    geometry,
    cell_selections: Sequence[CellSelection],
    z_index: int,
    *,
    field_spec: FieldAnalyzerSpec,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, _, _, _ = build_rhs_terms_figure(
        times,
        cell_histories,
        density_slice,
        geometry,
        cell_selections,
        z_index,
        field_spec=field_spec,
        title=title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run_field_analyzer(field_spec: FieldAnalyzerSpec, argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=f"Analyze saved EB blob {field_spec.display_name} RHS terms.")
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
        help=(
            "Run output directory or step-dump directory. Defaults to tests/<run_name>, "
            "preferring nested step_dumps/ when present and falling back to <run_name>_step_dumps."
        ),
    )
    parser.add_argument(
        "--perp-diffusion",
        type=float,
        default=DEFAULT_PERP_DIFFUSION,
        help="Perpendicular diffusion coefficient used for the field diffusion terms.",
    )
    args = parser.parse_args(argv)

    step_dump_dir = _resolve_step_dump_dir(args.run_name, args.output_path)
    if not step_dump_dir.exists():
        raise FileNotFoundError(f"missing EB blob step dump directory: {step_dump_dir}")
    print(f"analyzing {field_spec.display_name} RHS terms from {step_dump_dir}", flush=True)

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

    context = build_eb_blob_context(density_history, float(args.perp_diffusion))
    cell_selections = build_cell_selections(context.geometry)
    cell_histories = collect_rhs_histories(
        np.asarray(times, dtype=np.float64),
        np.asarray(density_history, dtype=np.float64),
        np.asarray(phi_history, dtype=np.float64),
        np.asarray(te_history, dtype=np.float64),
        np.asarray(ti_history, dtype=np.float64),
        np.asarray(vi_history, dtype=np.float64),
        np.asarray(ve_history, dtype=np.float64),
        np.asarray(vorticity_history, dtype=np.float64),
        field_spec=field_spec,
        geometry=context.geometry,
        parameters=context.parameters,
        boundary_condition_builder=context.boundary_condition_builder,
        cut_wall_geometry=context.cut_wall_geometry,
        cut_wall_bc=context.cut_wall_bc,
        curvature_coefficients=context.curvature_coefficients,
        cell_selections=cell_selections,
    )

    artifact_stem = _eb_blob_artifact_stem(args.run_name)
    output_path = step_dump_dir / f"{artifact_stem}_{field_spec.output_suffix}_rhs_terms.png"
    density_slice = np.asarray(density_history[0, :, :, cell_selections[0].z_index], dtype=np.float64)
    save_rhs_terms_plot(
        np.asarray(times, dtype=np.float64),
        cell_histories,
        density_slice,
        context.geometry,
        cell_selections,
        cell_selections[0].z_index,
        field_spec=field_spec,
        output_path=output_path,
        title=f"EB blob {field_spec.display_name} RHS term diagnostics, D_perp={float(args.perp_diffusion):.3e}",
    )

    print(f"saved {field_spec.display_name} RHS term plot to {output_path}", flush=True)
    for selection in cell_selections:
        print(
            f"selected cell {selection.name}: x_index={selection.x_index}, theta_index={selection.theta_index}, "
            f"z_index={selection.z_index}",
            flush=True,
        )
    return output_path
