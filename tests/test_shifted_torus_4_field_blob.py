from __future__ import annotations

import time as time_module
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from jax_drb.geometry import (
    ConservativeStencilBuilder,
    FciGeometry3D,
    LocalStencilBuilder,
    RegularFaceGeometry3D,
    build_conservative_stencil_from_field,
    build_curvature_coefficients,
    build_local_stencil_from_field,
)
from jax_drb.native import (
    Fci4FieldBlobParameters,
    Fci4FieldState,
    build_perp_laplacian_face_projectors,
    compute_4field_blob_rhs,
)
from jax_drb.native.fci_boundaries import CutWallBC3D, CutWallGeometry3D
from jax_drb.native.fci_operators import PerpLaplacianInverseSolver

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from test_mms_shifted_torus_4_field import build_shifted_torus_4field_geometry, x_max, x_min
from test_shifted_torus_4_field_free_decay import (
    FreeDecayBoundaryConditions,
    _add_state,
    _build_free_decay_boundary_conditions,
    _format_progress_bar,
    _max_abs_timing,
    _max_timing,
    _save_shifted_torus_free_decay_movie,
    _save_shifted_torus_free_decay_time_traces,
)


tf = 0.6
num_steps = 300
resolution = 60
rho_star = 1.0
Te = 1.0
mi_over_me = 1836.0
phi_inversion_tol = 5.0e-5
phi_inversion_restart = 200
phi_inversion_maxiter = 100

n_bg = 1.0
A_blob = 0.1
x0 = 0.5 * (float(x_min) + float(x_max))
theta0 = np.pi
zeta0 = np.pi
sigma_x = 0.10 * (float(x_max) - float(x_min))
sigma_theta = 0.25
sigma_zeta = 0.25


def _blob_artifact_stem() -> str:
    return "blob_4field"


def _logical_coordinates(geometry: FciGeometry3D) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)[:, None, None]
    theta = jnp.asarray(geometry.grid.y.centers, dtype=jnp.float64)[None, :, None]
    zeta = jnp.asarray(geometry.grid.z.centers, dtype=jnp.float64)[None, None, :]
    return (
        jnp.broadcast_to(x, geometry.shape),
        jnp.broadcast_to(theta, geometry.shape),
        jnp.broadcast_to(zeta, geometry.shape),
    )


def _periodic_angle_distance(angle: jnp.ndarray, center: float) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(angle - float(center)), jnp.cos(angle - float(center)))


def _blob_z_indices(geometry: FciGeometry3D, center: float, count: int = 4) -> tuple[int, ...]:
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    center_index = int(np.argmin(np.abs(z_values - float(center))))
    offsets = np.arange(-(count // 2), -(count // 2) + count, dtype=np.int64)
    return tuple(int((center_index + offset) % int(z_values.shape[0])) for offset in offsets)


def _build_blob_initial_state(geometry: FciGeometry3D) -> Fci4FieldState:
    x, theta, zeta = _logical_coordinates(geometry)
    d_theta = _periodic_angle_distance(theta, theta0)
    d_zeta = _periodic_angle_distance(zeta, zeta0)
    blob = (
        jnp.exp(-((x - float(x0)) ** 2) / (float(sigma_x) ** 2))
        * jnp.exp(-(d_theta**2) / (float(sigma_theta) ** 2))
        * jnp.exp(-(d_zeta**2) / (float(sigma_zeta) ** 2))
    )
    density = float(n_bg) * (1.0 + float(A_blob) * blob)
    zeros = jnp.zeros_like(density, dtype=jnp.float64)
    return Fci4FieldState(
        density=density,
        omega=zeros,
        v_ion_parallel=zeros,
        v_electron_parallel=zeros,
    )


def shifted_torus_4field_blob_rk4(
    state: Fci4FieldState,
    *,
    geometry: FciGeometry3D,
    timestep: float,
    parameters: Fci4FieldBlobParameters,
    curvature_coefficients: jnp.ndarray,
    stencil_builder: LocalStencilBuilder,
    conservative_stencil_builder: ConservativeStencilBuilder,
    boundary_conditions: FreeDecayBoundaryConditions,
    phi_face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None,
    phi_inverse_solver: PerpLaplacianInverseSolver,
    gmres_debug: bool = False,
    phi_guess: jnp.ndarray | None = None,
) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray]:
    common_kwargs = dict(
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi_face_bc=boundary_conditions.phi_face_bc,
        density_face_bc=boundary_conditions.density_face_bc,
        omega_face_bc=boundary_conditions.omega_face_bc,
        v_ion_parallel_face_bc=boundary_conditions.v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=boundary_conditions.v_electron_parallel_face_bc,
        phi_cut_wall_geometry=boundary_conditions.phi_cut_wall_geometry,
        phi_cut_wall_bc=boundary_conditions.phi_cut_wall_bc,
        density_cut_wall_geometry=boundary_conditions.density_cut_wall_geometry,
        density_cut_wall_bc=boundary_conditions.density_cut_wall_bc,
        omega_cut_wall_geometry=boundary_conditions.omega_cut_wall_geometry,
        omega_cut_wall_bc=boundary_conditions.omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=boundary_conditions.v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=boundary_conditions.v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=boundary_conditions.v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=boundary_conditions.v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        phi_inverse_solver=phi_inverse_solver,
        gmres_debug=gmres_debug,
        return_phi=True,
        with_diagnostics=True,
    )

    rhs_1, timings_1, phi_1 = compute_4field_blob_rhs(state, phi_guess=phi_guess, **common_kwargs)
    k1 = rhs_1.rhs
    stage_1 = _add_state(state, k1, scale=0.5 * timestep)
    jax.block_until_ready(stage_1.density)

    rhs_2, timings_2, phi_2 = compute_4field_blob_rhs(stage_1, phi_guess=phi_1, **common_kwargs)
    k2 = rhs_2.rhs
    stage_2 = _add_state(state, k2, scale=0.5 * timestep)
    jax.block_until_ready(stage_2.density)

    rhs_3, timings_3, phi_3 = compute_4field_blob_rhs(stage_2, phi_guess=phi_2, **common_kwargs)
    k3 = rhs_3.rhs
    stage_3 = _add_state(state, k3, scale=timestep)
    jax.block_until_ready(stage_3.density)

    rhs_4, timings_4, phi_4 = compute_4field_blob_rhs(stage_3, phi_guess=phi_3, **common_kwargs)
    k4 = rhs_4.rhs

    next_state = _add_state(
        state,
        Fci4FieldState(
            density=(k1.density + 2.0 * k2.density + 2.0 * k3.density + k4.density) / 6.0,
            omega=(k1.omega + 2.0 * k2.omega + 2.0 * k3.omega + k4.omega) / 6.0,
            v_ion_parallel=(k1.v_ion_parallel + 2.0 * k2.v_ion_parallel + 2.0 * k3.v_ion_parallel + k4.v_ion_parallel)
            / 6.0,
            v_electron_parallel=(
                k1.v_electron_parallel
                + 2.0 * k2.v_electron_parallel
                + 2.0 * k3.v_electron_parallel
                + k4.v_electron_parallel
            )
            / 6.0,
        ),
        scale=timestep,
    )
    jax.block_until_ready(next_state.density)
    return next_state, jnp.asarray(
        [
            float(timings_1[0]) + float(timings_2[0]) + float(timings_3[0]) + float(timings_4[0]),
            float(timings_1[1]) + float(timings_2[1]) + float(timings_3[1]) + float(timings_4[1]),
            float(timings_1[2]) + float(timings_2[2]) + float(timings_3[2]) + float(timings_4[2]),
            float(timings_1[3]) + float(timings_2[3]) + float(timings_3[3]) + float(timings_4[3]),
            _max_abs_timing(timings_1, timings_2, timings_3, timings_4, index=4),
            _max_timing(timings_1, timings_2, timings_3, timings_4, index=5),
            _max_timing(timings_1, timings_2, timings_3, timings_4, index=6),
            _max_abs_timing(timings_1, timings_2, timings_3, timings_4, index=7),
            _max_timing(timings_1, timings_2, timings_3, timings_4, index=8),
            _max_timing(timings_1, timings_2, timings_3, timings_4, index=9),
            _max_timing(timings_1, timings_2, timings_3, timings_4, index=10),
        ],
        dtype=jnp.float64,
    ), phi_4


def simulate_shifted_torus_4field_blob(
    geometry: FciGeometry3D,
    initial_state: Fci4FieldState,
    boundary_conditions: FreeDecayBoundaryConditions,
    *,
    parameters: Fci4FieldBlobParameters,
    timestep: float | None = None,
    final_time: float = tf,
    show_progress: bool = False,
    gmres_debug: bool = False,
) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    stencil_builder = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
    conservative_stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    dt = float(final_time) / float(num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)

    curvature_start = time_module.perf_counter()
    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    curvature_build_time = time_module.perf_counter() - curvature_start
    face_projectors = build_perp_laplacian_face_projectors(geometry)
    phi_inverse_solver = PerpLaplacianInverseSolver(
        geometry,
        conservative_stencil_builder,
        tol=float(parameters.phi_inversion_tol),
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        face_projectors=face_projectors,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=CutWallGeometry3D.empty(),
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=(False, True, True),
        pin_point=None,
        pin_value=0.0,
        project_mean_zero=False,
        target_mean_phi=None,
        regularization_epsilon=0.0,
        gmres_debug=gmres_debug,
    )

    state = initial_state
    time_value = 0.0
    current_phi_guess = None
    times: list[float] = [0.0]
    density_history: list[jnp.ndarray] = [jnp.asarray(initial_state.density, dtype=jnp.float32)]
    omega_history: list[jnp.ndarray] = [jnp.asarray(initial_state.omega, dtype=jnp.float32)]
    v_ion_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_ion_parallel, dtype=jnp.float32)]
    v_electron_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_electron_parallel, dtype=jnp.float32)]
    timing_history: list[jnp.ndarray] = []
    simulation_start = time_module.perf_counter()
    progress_start = time_module.perf_counter()
    if show_progress:
        print(f"blob RK4 progress: {_format_progress_bar(0, steps, start_time=progress_start)}", end="", flush=True)

    for step_index in range(steps):
        try:
            state, step_timings, current_phi_guess = shifted_torus_4field_blob_rk4(
                state,
                geometry=geometry,
                timestep=dt,
                parameters=parameters,
                curvature_coefficients=curvature_coefficients,
                stencil_builder=stencil_builder,
                conservative_stencil_builder=conservative_stencil_builder,
                boundary_conditions=boundary_conditions,
                phi_face_projectors=face_projectors,
                phi_inverse_solver=phi_inverse_solver,
                gmres_debug=gmres_debug,
                phi_guess=current_phi_guess,
            )
        except RuntimeError as error:
            print(f"shifted_torus_4field_blob RK step failed: step={step_index}, time={time_value:.6e}, dt={dt:.6e}")
            for field_name, field_values in {
                "density": state.density,
                "omega": state.omega,
                "v_ion_parallel": state.v_ion_parallel,
                "v_electron_parallel": state.v_electron_parallel,
            }.items():
                values = jnp.asarray(field_values, dtype=jnp.float64)
                print(
                    f"  state {field_name}: finite={bool(jnp.all(jnp.isfinite(values)))}, "
                    f"min={float(jnp.nanmin(values)):.6e}, max={float(jnp.nanmax(values)):.6e}, "
                    f"l2={float(jnp.linalg.norm(jnp.nan_to_num(values))):.6e}"
                )
            raise error

        time_value += dt
        times.append(time_value)
        density_history.append(jnp.asarray(state.density, dtype=jnp.float32))
        omega_history.append(jnp.asarray(state.omega, dtype=jnp.float32))
        v_ion_history.append(jnp.asarray(state.v_ion_parallel, dtype=jnp.float32))
        v_electron_history.append(jnp.asarray(state.v_electron_parallel, dtype=jnp.float32))
        timing_history.append(step_timings)
        if show_progress:
            print(
                "\r\033[K"
                f"blob RK4 progress: "
                f"{_format_progress_bar(step_index + 1, steps, start_time=progress_start, time_value=time_value, gmres_steps_per_solve=float(step_timings[3]) / 4.0, gmres_rel_res=float(step_timings[10]))}",
                end="",
                flush=True,
            )

    if show_progress:
        print()

    if timing_history:
        timing_array = np.asarray(timing_history, dtype=np.float64)
        total_time = time_module.perf_counter() - simulation_start
        print(f"shifted_torus_4field_blob curvature coefficient build time: {curvature_build_time:.6e} s")
        print("shifted_torus_4field_blob multigrid preconditioner: disabled")
        print(
            "shifted_torus_4field_blob mean timings per RK step: "
            f"phi_inverse={float(np.mean(timing_array[:, 0])):.6e} s, "
            f"local_stencil={float(np.mean(timing_array[:, 1])):.6e} s, "
            f"operator={float(np.mean(timing_array[:, 2])):.6e} s, "
            f"phi_gmres_steps_per_rk={float(np.mean(timing_array[:, 3])):.2f}, "
            f"phi_gmres_steps_per_solve={float(np.mean(timing_array[:, 3]) / 4.0):.2f}"
        )
        print(
            "shifted_torus_4field_blob RK4 timing: "
            f"steps={steps}, total_time={total_time:.6e} s, "
            f"avg_step_time={total_time / float(max(steps, 1)):.6e} s"
        )

    return (
        state,
        jnp.asarray(times, dtype=jnp.float64),
        jnp.stack(density_history, axis=0),
        jnp.stack(omega_history, axis=0),
        jnp.stack(v_ion_history, axis=0),
        jnp.stack(v_electron_history, axis=0),
    )


def main() -> None:
    geometry = build_shifted_torus_4field_geometry((resolution, resolution, resolution))
    boundary_conditions = _build_free_decay_boundary_conditions(geometry, 0.0)
    parameters = Fci4FieldBlobParameters(
        rho_star=rho_star,
        Te=Te,
        mi_over_me=mi_over_me,
        phi_inversion_tol=phi_inversion_tol,
        phi_inversion_maxiter=phi_inversion_maxiter,
        phi_inversion_restart=phi_inversion_restart,
        density_perp_diffusion=1.0e-3,
        omega_perp_diffusion=1.0e-3,
        v_ion_parallel_perp_diffusion=1.0e-4,
        v_electron_parallel_perp_diffusion=1.0e-3,
    )
    artifact_stem = _blob_artifact_stem()
    history_path = Path(f"{artifact_stem}_histories.npz")

    run_simulation = True
    if history_path.exists():
        with np.load(history_path, allow_pickle=False) as history:
            saved_resolution = int(history["resolution"]) if "resolution" in history else None
            saved_num_steps = int(history["num_steps"]) if "num_steps" in history else None
            saved_a_blob = float(history["A_blob"]) if "A_blob" in history else None
            if saved_resolution == resolution and saved_num_steps == num_steps and np.isclose(saved_a_blob, A_blob):
                times = jnp.asarray(history["times"], dtype=jnp.float64)
                density_history = jnp.asarray(history["density"], dtype=jnp.float64)
                omega_history = jnp.asarray(history["omega"], dtype=jnp.float64)
                v_ion_history = jnp.asarray(history["v_ion_parallel"], dtype=jnp.float64)
                v_electron_history = jnp.asarray(history["v_electron_parallel"], dtype=jnp.float64)
                run_simulation = False
            else:
                print("shifted_torus_4field_blob history settings mismatch; rerunning")

    if run_simulation:
        initial_state = _build_blob_initial_state(geometry)
        _, times, density_history, omega_history, v_ion_history, v_electron_history = simulate_shifted_torus_4field_blob(
            geometry,
            initial_state,
            boundary_conditions,
            final_time=tf,
            timestep=tf / float(num_steps),
            show_progress=True,
            parameters=parameters,
        )
        np.savez(
            history_path,
            times=np.asarray(times, dtype=np.float64),
            density=np.asarray(density_history, dtype=np.float64),
            omega=np.asarray(omega_history, dtype=np.float64),
            v_ion_parallel=np.asarray(v_ion_history, dtype=np.float64),
            v_electron_parallel=np.asarray(v_electron_history, dtype=np.float64),
            resolution=np.asarray(resolution, dtype=np.int64),
            num_steps=np.asarray(num_steps, dtype=np.int64),
            A_blob=np.asarray(A_blob, dtype=np.float64),
            x0=np.asarray(x0, dtype=np.float64),
            theta0=np.asarray(theta0, dtype=np.float64),
            zeta0=np.asarray(zeta0, dtype=np.float64),
            sigma_x=np.asarray(sigma_x, dtype=np.float64),
            sigma_theta=np.asarray(sigma_theta, dtype=np.float64),
            sigma_zeta=np.asarray(sigma_zeta, dtype=np.float64),
        )

    _save_shifted_torus_free_decay_time_traces(
        times,
        density_history,
        omega_history,
        v_ion_history,
        v_electron_history,
        geometry,
        boundary_conditions,
        parameters,
        output_path=f"{artifact_stem}_time_traces.png",
        title="Shifted-torus 4-field blob time traces",
    )
    _save_shifted_torus_free_decay_movie(
        times,
        density_history,
        omega_history,
        v_ion_history,
        v_electron_history,
        geometry,
        output_path=f"{artifact_stem}.gif",
        frame_stride=2,
        title="Shifted-torus 4-field blob evolution",
        z_indices=_blob_z_indices(geometry, zeta0),
    )


if __name__ == "__main__":
    main()
