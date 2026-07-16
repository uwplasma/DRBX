from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
import sys

import jax.numpy as jnp
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from analyze_rhs_common import (  # noqa: E402
    AXIS_REGULAR_AXES,
    DEFAULT_CELL_TARGETS,
    DEFAULT_PERP_DIFFUSION,
    PERIODIC_AXES,
    CellSelection,
    FieldAnalyzerSpec,
    build_cell_selections,
    build_rhs_terms_figure,
    build_term_specs,
    collect_rhs_histories,
    eb_blob_artifact_stem,
    load_eb_blob_step_history,
    resolve_step_dump_dir,
    save_rhs_terms_plot,
)
from jax_drb.geometry import build_curvature_coefficients  # noqa: E402
from jax_drb.native.fci_boundaries import (  # noqa: E402
    BoundaryConditionBuilder,
    CoordinateFaceValueReconstructor3D,
    CoordinateNormalDerivativeConstructor3D,
    CutWallBC3D,
    CutWallGeometry3D,
)
from jax_drb.native.fci_drb_EB_rhs import FciDrbEBState, _Ve_rhs  # noqa: E402
from jax_drb.native.fci_operators import (  # noqa: E402
    grad_parallel_op_direct,
    parallel_laplacian_direct_op,
    poisson_bracket_op,
    perp_laplacian_conservative_op,
)
from test_shifted_torus_EB_blob import (  # noqa: E402
    _build_eb_blob_geometry,
    _build_eb_blob_parameters,
    _build_eb_boundary_conditions,
    conservative_stencil_builder,
    local_stencil_builder,
    z0,
)


VE_RHS_TERM_SPECS = build_term_specs(
    (
        ("exb", "ExB"),
        ("parallel_advection", "-Ve d_parallel Ve"),
        ("ve_nu_j_parallel", "mi_over_me * Ve_nu j_parallel"),
        ("parallel_phi", "mi_over_me * d_parallel phi"),
        ("parallel_pe_over_density", "mi_over_me * -(d_parallel Pe)/n"),
        ("parallel_te", "mi_over_me * -0.71 d_parallel Te"),
        ("parallel_diffusion", "parallel diffusion"),
        ("perp_diffusion", "perp diffusion"),
    )
)

VE_CELL_TARGETS = DEFAULT_CELL_TARGETS
VE_CELL_COLORS: tuple[str, ...] = ("tab:blue", "tab:orange", "tab:green")


def _ve_rhs_term_fields(
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
    Vi = jnp.asarray(state.Vi, dtype=jnp.float64)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    mi_over_me = jnp.asarray(parameters.mi_over_me, dtype=jnp.float64)
    Ve_nu = jnp.asarray(parameters.Ve_nu, dtype=jnp.float64)
    Ve_D_perp = jnp.asarray(parameters.Ve_D_perp, dtype=jnp.float64)
    Ve_parallel_viscosity = jnp.asarray(parameters.Ve_parallel_viscosity, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), 1.0e-30)
    density_safe = jnp.maximum(density, 1.0e-30)

    Ve_stencil = local_stencil_builder(
        Ve,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Ve_face_bc,
        cut_wall_geometry,
        boundary_conditions.Ve_cut_wall_bc,
    )
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
    Te_stencil = local_stencil_builder(
        Te,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Te_face_bc,
        cut_wall_geometry,
        boundary_conditions.Te_cut_wall_bc,
    )
    current_density = density * (Vi - Ve)
    current_density_stencil = local_stencil_builder(
        current_density,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Ve_face_bc,
        cut_wall_geometry,
        boundary_conditions.Ve_cut_wall_bc,
    )

    ve_conservative_stencil = (
        conservative_stencil_builder(
            Ve,
            geometry,
            PERIODIC_AXES,
            boundary_conditions.Ve_face_bc,
        )
        if float(Ve_D_perp) != 0.0
        else None
    )

    exb = -(poisson_bracket_op(potential_stencil, Ve_stencil, geometry) / (rho_star * bmag))
    parallel_advection = -(Ve * grad_parallel_op_direct(Ve_stencil, geometry))
    ve_nu_j_parallel = mi_over_me * (Ve_nu * current_density)
    parallel_phi = mi_over_me * grad_parallel_op_direct(potential_stencil, geometry)
    parallel_pe_over_density = mi_over_me * (
        -grad_parallel_op_direct(Pe_stencil, geometry) / density_safe
    )
    parallel_te = mi_over_me * (-0.71 * grad_parallel_op_direct(Te_stencil, geometry))

    parallel_diffusion = jnp.zeros_like(Ve)
    if float(Ve_parallel_viscosity) != 0.0:
        parallel_diffusion = Ve_parallel_viscosity * parallel_laplacian_direct_op(
            Ve,
            geometry,
            face_bc=boundary_conditions.Ve_face_bc,
            periodic_axes=PERIODIC_AXES,
        )

    perp_diffusion = jnp.zeros_like(Ve)
    if float(Ve_D_perp) != 0.0:
        if ve_conservative_stencil is None:
            raise ValueError("Ve_conservative_stencil is required when Ve_D_perp is nonzero")
        perp_diffusion = Ve_D_perp * perp_laplacian_conservative_op(
            ve_conservative_stencil,
            geometry,
            face_bc=boundary_conditions.Ve_face_bc,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=boundary_conditions.Ve_cut_wall_bc,
            periodic_axes=PERIODIC_AXES,
        )

    total = (
        exb
        + parallel_advection
        + ve_nu_j_parallel
        + parallel_phi
        + parallel_pe_over_density
        + parallel_te
        + parallel_diffusion
        + perp_diffusion
    )

    return {
        "exb": np.asarray(exb, dtype=np.float64),
        "parallel_advection": np.asarray(parallel_advection, dtype=np.float64),
        "ve_nu_j_parallel": np.asarray(ve_nu_j_parallel, dtype=np.float64),
        "parallel_phi": np.asarray(parallel_phi, dtype=np.float64),
        "parallel_pe_over_density": np.asarray(parallel_pe_over_density, dtype=np.float64),
        "parallel_te": np.asarray(parallel_te, dtype=np.float64),
        "parallel_diffusion": np.asarray(parallel_diffusion, dtype=np.float64),
        "perp_diffusion": np.asarray(perp_diffusion, dtype=np.float64),
        "total_rhs": np.asarray(total, dtype=np.float64),
    }


FIELD_SPEC = FieldAnalyzerSpec(
    display_name="Ve",
    output_suffix="ve",
    term_specs=VE_RHS_TERM_SPECS,
    sum_keys=(
        "exb",
        "parallel_advection",
        "ve_nu_j_parallel",
        "parallel_phi",
        "parallel_pe_over_density",
        "parallel_te",
        "parallel_diffusion",
        "perp_diffusion",
    ),
    evaluate_terms=_ve_rhs_term_fields,
    legend_ncol=4,
)


def _ve_profile_history(ve_history: np.ndarray, geometry) -> np.ndarray:
    ve = jnp.asarray(ve_history, dtype=jnp.float64)
    if ve.ndim != 4:
        raise ValueError(f"ve_history must have shape (time, nx, ny, nz), got {ve.shape}")

    jacobian = jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64)
    if jacobian.shape != ve.shape[1:]:
        raise ValueError(f"geometry jacobian shape {jacobian.shape} does not match ve cell shape {ve.shape[1:]}")

    radial_weight = jnp.sum(jacobian, axis=(1, 2))
    if np.any(np.asarray(radial_weight == 0.0, dtype=bool)):
        raise ValueError("cannot build ve profile with a zero radial jacobian weight")

    profile = jnp.sum(ve * jacobian[None, :, :, :], axis=(2, 3)) / radial_weight[None, :]
    return np.asarray(profile, dtype=np.float64)


def _save_ve_profile_movie(
    times: np.ndarray,
    ve_history: np.ndarray,
    geometry,
    *,
    output_path: Path,
    title: str,
    frame_stride: int = 1,
    fps: int = 10,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    times_np = np.asarray(times, dtype=np.float64)
    profiles = _ve_profile_history(ve_history, geometry)
    if times_np.shape[0] != profiles.shape[0]:
        raise ValueError(f"times and profiles disagree: {times_np.shape[0]} times for {profiles.shape[0]} frames")

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    if x_values.shape[0] != profiles.shape[1]:
        raise ValueError(f"x grid has {x_values.shape[0]} cells but profiles have {profiles.shape[1]} radial values")

    finite_profiles = profiles[np.isfinite(profiles)]
    if finite_profiles.size == 0:
        raise ValueError("cannot build ve profile movie from non-finite profile values")
    y_min = float(np.min(finite_profiles))
    y_max = float(np.max(finite_profiles))
    y_span = max(y_max - y_min, 1.0e-12)
    y_pad = 0.08 * y_span

    frame_stride = max(1, int(frame_stride))
    fps = max(1, int(fps))
    frame_indices = np.arange(0, int(times_np.shape[0]), frame_stride, dtype=np.int64)
    if frame_indices[-1] != int(times_np.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times_np.shape[0]) - 1)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    line, = ax.plot(x_values, profiles[int(frame_indices[0])], color="tab:purple", linewidth=2.4)
    ax.set_xlabel("radial coordinate x")
    ax.set_ylabel("J-weighted Ve profile")
    ax.set_xlim(float(np.min(x_values)), float(np.max(x_values)))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(True, alpha=0.28)
    title_artist = ax.set_title("")

    def update(movie_frame_index: int):
        actual_index = int(frame_indices[movie_frame_index])
        line.set_ydata(profiles[actual_index])
        title_artist.set_text(f"{title}\nstep={actual_index}, t={float(times_np[actual_index]):.6e}")
        return line, title_artist

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


def _ve_profile_movie_path(rhs_plot_path: Path) -> Path:
    suffix = f"_{FIELD_SPEC.output_suffix}_rhs_terms.png"
    filename = rhs_plot_path.name
    artifact_stem = filename[: -len(suffix)] if filename.endswith(suffix) else eb_blob_artifact_stem(rhs_plot_path.stem)
    return rhs_plot_path.parent / f"{artifact_stem}_ve_profile.gif"


_build_ve_cell_selections = build_cell_selections


def _collect_ve_rhs_histories(
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
    curvature_coefficients,
    cell_selections: Sequence[CellSelection],
) -> dict[str, dict[str, np.ndarray]]:
    return collect_rhs_histories(
        times,
        density_history,
        phi_history,
        te_history,
        ti_history,
        vi_history,
        ve_history,
        vorticity_history,
        field_spec=FIELD_SPEC,
        geometry=geometry,
        parameters=parameters,
        boundary_condition_builder=boundary_condition_builder,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        curvature_coefficients=curvature_coefficients,
        cell_selections=cell_selections,
    )


def _build_ve_rhs_terms_figure(
    times: np.ndarray,
    cell_histories: dict[str, dict[str, np.ndarray]],
    density_slice: np.ndarray,
    geometry,
    cell_selections: Sequence[CellSelection],
    z_index: int,
    *,
    title: str,
):
    return build_rhs_terms_figure(
        times,
        cell_histories,
        density_slice,
        geometry,
        cell_selections,
        z_index,
        field_spec=FIELD_SPEC,
        title=title,
    )


def _save_ve_rhs_terms_plot(
    times: np.ndarray,
    cell_histories: dict[str, dict[str, np.ndarray]],
    density_slice: np.ndarray,
    geometry,
    cell_selections: Sequence[CellSelection],
    z_index: int,
    *,
    output_path: Path,
    title: str,
) -> None:
    save_rhs_terms_plot(
        times,
        cell_histories,
        density_slice,
        geometry,
        cell_selections,
        z_index,
        field_spec=FIELD_SPEC,
        output_path=output_path,
        title=title,
    )


def main(argv: Sequence[str] | None = None) -> Path:
    from analyze_rhs_common import run_field_analyzer

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Frame stride to use when rendering the Ve profile GIF.",
    )
    ve_args, remaining_argv = parser.parse_known_args(argv)

    rhs_plot_path = run_field_analyzer(FIELD_SPEC, remaining_argv)
    (
        times,
        _density_history,
        _phi_history,
        _te_history,
        _ti_history,
        _vi_history,
        ve_history,
        _vorticity_history,
    ) = load_eb_blob_step_history(rhs_plot_path.parent)
    ve_shape = tuple(int(size) for size in np.asarray(ve_history).shape[1:])
    geometry = _build_eb_blob_geometry(ve_shape, construct_fci_maps=False)
    movie_path = _ve_profile_movie_path(rhs_plot_path)
    _save_ve_profile_movie(
        np.asarray(times, dtype=np.float64),
        np.asarray(ve_history, dtype=np.float64),
        geometry,
        output_path=movie_path,
        title="EB blob J-weighted Ve profile",
        frame_stride=int(ve_args.frame_stride),
    )
    print(f"saved Ve profile movie to {movie_path}", flush=True)
    return rhs_plot_path


_load_eb_blob_step_history = load_eb_blob_step_history
_resolve_step_dump_dir = resolve_step_dump_dir
_eb_blob_artifact_stem = eb_blob_artifact_stem


if __name__ == "__main__":
    main()
