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
    grad_parallel_op_direct,
    parallel_laplacian_direct_op,
    poisson_bracket_op,
    perp_laplacian_conservative_op,
)
from test_shifted_torus_EB_blob import (  # noqa: E402
    PERIODIC_AXES,
    _build_eb_blob_geometry,
    conservative_stencil_builder,
    local_stencil_builder,
)


VI_TERM_SPECS = build_term_specs(
    (
        ("exb", "ExB"),
        ("parallel_advection", "-Vi d_parallel Vi"),
        ("parallel_pressure", "pressure gradient"),
        ("parallel_diffusion", "parallel diffusion"),
        ("perp_diffusion", "perp diffusion"),
    )
)


def _vi_rhs_term_fields(
    state,
    *,
    geometry,
    parameters,
    boundary_condition_builder,
    cut_wall_geometry,
    cut_wall_bc,
    curvature_coefficients,
) -> dict[str, np.ndarray]:
    del curvature_coefficients
    boundary_conditions = boundary_condition_builder(
        state,
        geometry,
        PERIODIC_AXES,
        cut_wall_geometry,
        cut_wall_bc,
    )

    density = jnp.asarray(state.density, dtype=jnp.float64)
    phi = jnp.asarray(state.phi, dtype=jnp.float64)
    Ti = jnp.asarray(state.Ti, dtype=jnp.float64)
    Te = jnp.asarray(state.Te, dtype=jnp.float64)
    Vi = jnp.asarray(state.Vi, dtype=jnp.float64)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    Vi_D_perp = jnp.asarray(parameters.Vi_D_perp, dtype=jnp.float64)
    Vi_D_parallel = jnp.asarray(parameters.Vi_D_parallel, dtype=jnp.float64)
    tau = jnp.asarray(parameters.tau, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), 1.0e-30)
    density_safe = jnp.maximum(density, 1.0e-30)

    Vi_stencil = local_stencil_builder(
        Vi,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Vi_face_bc,
        cut_wall_geometry,
        boundary_conditions.Vi_cut_wall_bc,
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
    pressure = Pe + tau * (density * Ti)
    pressure_stencil = local_stencil_builder(
        pressure,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.Te_face_bc,
        cut_wall_geometry,
        boundary_conditions.Te_cut_wall_bc,
    )

    Vi_conservative_stencil = (
        conservative_stencil_builder(
            Vi,
            geometry,
            PERIODIC_AXES,
            boundary_conditions.Vi_face_bc,
        )
        if float(Vi_D_perp) != 0.0
        else None
    )

    exb = -(poisson_bracket_op(potential_stencil, Vi_stencil, geometry) / (rho_star * bmag))
    parallel_advection = -(Vi * grad_parallel_op_direct(Vi_stencil, geometry))
    parallel_pressure = -grad_parallel_op_direct(pressure_stencil, geometry) / density_safe

    parallel_diffusion = jnp.zeros_like(Vi)
    if float(Vi_D_parallel) != 0.0:
        parallel_diffusion = Vi_D_parallel * parallel_laplacian_direct_op(
            Vi,
            geometry,
            face_bc=boundary_conditions.Vi_face_bc,
            periodic_axes=PERIODIC_AXES,
        )

    perp_diffusion = jnp.zeros_like(Vi)
    if float(Vi_D_perp) != 0.0:
        if Vi_conservative_stencil is None:
            raise ValueError("Vi conservative stencil is required when Vi_D_perp is nonzero")
        perp_diffusion = Vi_D_perp * perp_laplacian_conservative_op(
            Vi_conservative_stencil,
            geometry,
            face_bc=boundary_conditions.Vi_face_bc,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=boundary_conditions.Vi_cut_wall_bc,
            periodic_axes=PERIODIC_AXES,
        )

    total = exb + parallel_advection + parallel_pressure + parallel_diffusion + perp_diffusion

    return {
        "exb": np.asarray(exb, dtype=np.float64),
        "parallel_advection": np.asarray(parallel_advection, dtype=np.float64),
        "parallel_pressure": np.asarray(parallel_pressure, dtype=np.float64),
        "parallel_diffusion": np.asarray(parallel_diffusion, dtype=np.float64),
        "perp_diffusion": np.asarray(perp_diffusion, dtype=np.float64),
        "total_rhs": np.asarray(total, dtype=np.float64),
    }


FIELD_SPEC = FieldAnalyzerSpec(
    display_name="Vi",
    output_suffix="vi",
    term_specs=VI_TERM_SPECS,
    sum_keys=("exb", "parallel_advection", "parallel_pressure", "parallel_diffusion", "perp_diffusion"),
    evaluate_terms=_vi_rhs_term_fields,
    legend_ncol=3,
)


def _vi_profile_history(vi_history: np.ndarray, geometry) -> np.ndarray:
    vi = jnp.asarray(vi_history, dtype=jnp.float64)
    if vi.ndim != 4:
        raise ValueError(f"vi_history must have shape (time, nx, ny, nz), got {vi.shape}")

    jacobian = jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64)
    if jacobian.shape != vi.shape[1:]:
        raise ValueError(f"geometry jacobian shape {jacobian.shape} does not match Vi cell shape {vi.shape[1:]}")

    radial_weight = jnp.sum(jacobian, axis=(1, 2))
    if np.any(np.asarray(radial_weight == 0.0, dtype=bool)):
        raise ValueError("cannot build Vi profile with a zero radial jacobian weight")

    profile = jnp.sum(vi * jacobian[None, :, :, :], axis=(2, 3)) / radial_weight[None, :]
    return np.asarray(profile, dtype=np.float64)


def _save_vi_profile_movie(
    times: np.ndarray,
    vi_history: np.ndarray,
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
    profiles = _vi_profile_history(vi_history, geometry)
    if times_np.shape[0] != profiles.shape[0]:
        raise ValueError(f"times and profiles disagree: {times_np.shape[0]} times for {profiles.shape[0]} frames")

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    if x_values.shape[0] != profiles.shape[1]:
        raise ValueError(f"x grid has {x_values.shape[0]} cells but profiles have {profiles.shape[1]} radial values")

    finite_profiles = profiles[np.isfinite(profiles)]
    if finite_profiles.size == 0:
        raise ValueError("cannot build Vi profile movie from non-finite profile values")
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
    ax.set_ylabel("J-weighted Vi profile")
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


def _vi_profile_movie_path(rhs_plot_path: Path) -> Path:
    suffix = f"_{FIELD_SPEC.output_suffix}_rhs_terms.png"
    filename = rhs_plot_path.name
    artifact_stem = filename[: -len(suffix)] if filename.endswith(suffix) else eb_blob_artifact_stem(rhs_plot_path.stem)
    return rhs_plot_path.parent / f"{artifact_stem}_vi_profile.gif"


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Frame stride to use when rendering the Vi profile GIF.",
    )
    vi_args, remaining_argv = parser.parse_known_args(argv)

    rhs_plot_path = run_field_analyzer(FIELD_SPEC, remaining_argv)
    (
        times,
        _density_history,
        _phi_history,
        _te_history,
        _ti_history,
        vi_history,
        _ve_history,
        _vorticity_history,
    ) = load_eb_blob_step_history(rhs_plot_path.parent)
    vi_shape = tuple(int(size) for size in np.asarray(vi_history).shape[1:])
    geometry = _build_eb_blob_geometry(vi_shape, construct_fci_maps=False)
    movie_path = _vi_profile_movie_path(rhs_plot_path)
    _save_vi_profile_movie(
        np.asarray(times, dtype=np.float64),
        np.asarray(vi_history, dtype=np.float64),
        geometry,
        output_path=movie_path,
        title="EB blob J-weighted Vi profile",
        frame_stride=int(vi_args.frame_stride),
    )
    print(f"saved Vi profile movie to {movie_path}", flush=True)
    return rhs_plot_path


if __name__ == "__main__":
    main()
