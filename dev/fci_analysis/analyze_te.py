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
    FieldAnalyzerSpec,
    build_term_specs,
    eb_blob_artifact_stem,
    load_eb_blob_step_history,
    run_field_analyzer,
)
from dkx.native.fci_operators import (  # noqa: E402
    curvature_op,
    grad_parallel_op_direct,
    parallel_laplacian_direct_op,
    poisson_bracket_op,
    perp_laplacian_conservative_op,
)
from test_shifted_torus_EB_blob import (  # noqa: E402
    _build_eb_blob_geometry,
    PERIODIC_AXES,
    conservative_stencil_builder,
    local_stencil_builder,
)


TE_TERM_SPECS = build_term_specs(
    (
        ("exb", "ExB"),
        ("parallel_advection", "-Ve d_parallel Te"),
        ("curvature_pressure", "curvature pressure"),
        ("curvature_temperature", "curvature temperature"),
        ("curvature_potential", "curvature potential"),
        ("parallel_current_density", "0.71 parallel current"),
        ("parallel_ve", "-n * parallel Ve"),
        ("parallel_diffusion", "parallel diffusion"),
        ("perp_diffusion", "perp diffusion"),
    )
)


def _te_rhs_term_fields(
    state,
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
    Vi = jnp.asarray(state.Vi, dtype=jnp.float64)
    Ve = jnp.asarray(state.Ve, dtype=jnp.float64)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    D_perp = jnp.asarray(parameters.electron_temperature_D_perp, dtype=jnp.float64)
    chi_parallel = jnp.asarray(parameters.electron_temperature_chi_parallel, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), 1.0e-30)
    density_safe = jnp.maximum(density, 1.0e-30)

    temperature_stencil = local_stencil_builder(
        Te,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Te_face_bc,
        cut_wall_geometry,
        boundary_conditions.Te_cut_wall_bc,
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
    current_density = density * (Vi - Ve)
    current_density_stencil = local_stencil_builder(
        current_density,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Ve_face_bc,
        cut_wall_geometry,
        boundary_conditions.Ve_cut_wall_bc,
    )
    Ve_stencil = local_stencil_builder(
        Ve,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Ve_face_bc,
        cut_wall_geometry,
        boundary_conditions.Ve_cut_wall_bc,
    )

    temperature_conservative_stencil = (
        conservative_stencil_builder(
            Te,
            geometry,
            PERIODIC_AXES,
            boundary_conditions.Te_face_bc,
        )
        if float(D_perp) != 0.0
        else None
    )
    exb = -(poisson_bracket_op(potential_stencil, temperature_stencil, geometry) / (rho_star * bmag))
    parallel_advection = -(Ve * grad_parallel_op_direct(temperature_stencil, geometry))
    curvature_Pe = curvature_op(Pe_stencil, geometry, curvature_coefficients=curvature_coefficients)
    curvature_temperature_op = curvature_op(
        temperature_stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )
    curvature_potential_op = curvature_op(
        potential_stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )
    parallel_current_density = grad_parallel_op_direct(current_density_stencil, geometry)
    parallel_Ve = grad_parallel_op_direct(Ve_stencil, geometry)

    curvature_pressure = (4.0 * Te / (3.0 * bmag)) * (curvature_Pe / density_safe)
    curvature_temperature = (4.0 * Te / (3.0 * bmag)) * (2.5 * curvature_temperature_op)
    curvature_potential = (4.0 * Te / (3.0 * bmag)) * (-curvature_potential_op)
    parallel_current_term = (2.0 * Te / (3.0 * density_safe)) * (0.71 * parallel_current_density)
    parallel_ve_term = (2.0 * Te / (3.0 * density_safe)) * (-density_safe * parallel_Ve)

    parallel_diffusion = jnp.zeros_like(Te)
    if float(chi_parallel) != 0.0:
        parallel_diffusion = chi_parallel * parallel_laplacian_direct_op(
            Te,
            geometry,
            face_bc=boundary_conditions.Te_face_bc,
            periodic_axes=PERIODIC_AXES,
        )

    perp_diffusion = jnp.zeros_like(Te)
    if float(D_perp) != 0.0:
        if temperature_conservative_stencil is None:
            raise ValueError("temperature conservative stencil is required when electron_temperature_D_perp is nonzero")
        perp_diffusion = D_perp * perp_laplacian_conservative_op(
            temperature_conservative_stencil,
            geometry,
            face_bc=boundary_conditions.Te_face_bc,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=boundary_conditions.Te_cut_wall_bc,
            periodic_axes=PERIODIC_AXES,
        )

    total = (
        exb
        + parallel_advection
        + curvature_pressure
        + curvature_temperature
        + curvature_potential
        + parallel_current_term
        + parallel_ve_term
        + parallel_diffusion
        + perp_diffusion
    )

    return {
        "exb": np.asarray(exb, dtype=np.float64),
        "parallel_advection": np.asarray(parallel_advection, dtype=np.float64),
        "curvature_pressure": np.asarray(curvature_pressure, dtype=np.float64),
        "curvature_temperature": np.asarray(curvature_temperature, dtype=np.float64),
        "curvature_potential": np.asarray(curvature_potential, dtype=np.float64),
        "parallel_current_density": np.asarray(parallel_current_term, dtype=np.float64),
        "parallel_ve": np.asarray(parallel_ve_term, dtype=np.float64),
        "parallel_diffusion": np.asarray(parallel_diffusion, dtype=np.float64),
        "perp_diffusion": np.asarray(perp_diffusion, dtype=np.float64),
        "total_rhs": np.asarray(total, dtype=np.float64),
    }


FIELD_SPEC = FieldAnalyzerSpec(
    display_name="Te",
    output_suffix="te",
    term_specs=TE_TERM_SPECS,
    sum_keys=(
        "exb",
        "parallel_advection",
        "curvature_pressure",
        "curvature_temperature",
        "curvature_potential",
        "parallel_current_density",
        "parallel_ve",
        "parallel_diffusion",
        "perp_diffusion",
    ),
    evaluate_terms=_te_rhs_term_fields,
    legend_ncol=4,
)


def _te_profile_history(te_history: np.ndarray, geometry) -> np.ndarray:
    te = jnp.asarray(te_history, dtype=jnp.float64)
    if te.ndim != 4:
        raise ValueError(f"te_history must have shape (time, nx, ny, nz), got {te.shape}")

    jacobian = jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64)
    if jacobian.shape != te.shape[1:]:
        raise ValueError(f"geometry jacobian shape {jacobian.shape} does not match Te cell shape {te.shape[1:]}")

    radial_weight = jnp.sum(jacobian, axis=(1, 2))
    if np.any(np.asarray(radial_weight == 0.0, dtype=bool)):
        raise ValueError("cannot build Te profile with a zero radial jacobian weight")

    profile = jnp.sum(te * jacobian[None, :, :, :], axis=(2, 3)) / radial_weight[None, :]
    return np.asarray(profile, dtype=np.float64)


def _save_te_profile_movie(
    times: np.ndarray,
    te_history: np.ndarray,
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
    profiles = _te_profile_history(te_history, geometry)
    if times_np.shape[0] != profiles.shape[0]:
        raise ValueError(f"times and profiles disagree: {times_np.shape[0]} times for {profiles.shape[0]} frames")

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    if x_values.shape[0] != profiles.shape[1]:
        raise ValueError(f"x grid has {x_values.shape[0]} cells but profiles have {profiles.shape[1]} radial values")

    finite_profiles = profiles[np.isfinite(profiles)]
    if finite_profiles.size == 0:
        raise ValueError("cannot build Te profile movie from non-finite profile values")
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
    line, = ax.plot(x_values, profiles[int(frame_indices[0])], color="tab:blue", linewidth=2.4)
    ax.set_xlabel("radial coordinate x")
    ax.set_ylabel("J-weighted Te profile")
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


def _te_profile_movie_path(rhs_plot_path: Path) -> Path:
    suffix = f"_{FIELD_SPEC.output_suffix}_rhs_terms.png"
    filename = rhs_plot_path.name
    artifact_stem = filename[: -len(suffix)] if filename.endswith(suffix) else eb_blob_artifact_stem(rhs_plot_path.stem)
    return rhs_plot_path.parent / f"{artifact_stem}_te_profile.gif"


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Frame stride to use when rendering the Te profile GIF.",
    )
    te_args, remaining_argv = parser.parse_known_args(argv)

    rhs_plot_path = run_field_analyzer(FIELD_SPEC, remaining_argv)
    (
        times,
        _density_history,
        _phi_history,
        te_history,
        _ti_history,
        _vi_history,
        _ve_history,
        _vorticity_history,
    ) = load_eb_blob_step_history(rhs_plot_path.parent)
    te_shape = tuple(int(size) for size in np.asarray(te_history).shape[1:])
    geometry = _build_eb_blob_geometry(te_shape, construct_fci_maps=False)
    movie_path = _te_profile_movie_path(rhs_plot_path)
    _save_te_profile_movie(
        np.asarray(times, dtype=np.float64),
        np.asarray(te_history, dtype=np.float64),
        geometry,
        output_path=movie_path,
        title="EB blob J-weighted Te profile",
        frame_stride=int(te_args.frame_stride),
    )
    print(f"saved Te profile movie to {movie_path}", flush=True)
    return rhs_plot_path


if __name__ == "__main__":
    main()
