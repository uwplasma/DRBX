from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np

from jax_drb.geometry import FciGeometry3D, logical_grid_from_axis_vectors
from jax_drb.native.fci_2_field_rhs import Fci2FieldRhsParameters, Fci2FieldState, compute_2field_rhs

A = 0.1
B = 0.1
B0 = 1.0
alpha = 0.2
omega = 2.0 * jnp.pi
rho_star = 1
tf = 0.1
num_steps = 100


def _resolution_step_count(resolution: int, *, base_resolution: int = 20, base_steps: int = num_steps) -> int:
    """Scale the timestep count like ``sqrt(resolution)`` relative to the base grid."""

    scale = np.sqrt(float(resolution) / float(base_resolution))
    return max(1, int(round(float(base_steps) * scale)))


def build_slab_2field_geometry(nx: int, ny: int, nz: int) -> FciGeometry3D:
    """Build a simple Cartesian slab geometry with constant unit ``B`` along ``z``.

    Field-line tracing is intentionally bypassed for this initial manufactured-solution
    scaffold. The geometry uses identity metric tensors on a logical grid spanning
    ``[0, 1]`` in each direction.
    """

    x_axis = jnp.linspace(0.0, 1.0, nx, dtype=jnp.float64)
    y_axis = jnp.linspace(0.0, 1.0, ny, dtype=jnp.float64)
    z_axis = jnp.linspace(0.0, 1.0, nz, dtype=jnp.float64)
    logical_grid = logical_grid_from_axis_vectors(x_axis, y_axis, z_axis)
    target_shape = (nx, ny, nz)
    ones = jnp.ones(target_shape, dtype=jnp.float64)
    zeros = jnp.zeros(target_shape, dtype=jnp.float64)
    b_contravariant = jnp.stack((zeros, zeros, ones), axis=-1)

    return FciGeometry3D(
        logical_grid=logical_grid,
        forward_x=jnp.broadcast_to(x_axis[:, None, None], target_shape),
        forward_y=jnp.broadcast_to(y_axis[None, :, None], target_shape),
        backward_x=jnp.broadcast_to(x_axis[:, None, None], target_shape),
        backward_y=jnp.broadcast_to(y_axis[None, :, None], target_shape),
        forward_length=ones,
        backward_length=ones,
        forward_boundary=jnp.zeros(target_shape, dtype=bool),
        backward_boundary=jnp.zeros(target_shape, dtype=bool),
        dx=ones * (1.0 / float(max(nx - 1, 1))),
        dy=ones * (1.0 / float(max(ny - 1, 1))),
        dz=ones * (1.0 / float(max(nz - 1, 1))),
        J=ones,
        B_contravariant=b_contravariant,
        g11=ones,
        g22=ones,
        g33=ones,
        g12=zeros,
        g13=zeros,
        g23=zeros,
        g_11=ones,
        g_22=ones,
        g_33=ones,
        g_12=zeros,
        g_13=zeros,
        g_23=zeros,
    )


def _add_state(state: Fci2FieldState, rhs: Fci2FieldState, *, scale: float) -> Fci2FieldState:
    return Fci2FieldState(
        density=state.density + scale * rhs.density,
        v_parallel=state.v_parallel + scale * rhs.v_parallel,
        density_background=state.density_background + scale * rhs.density_background,
    )


def _raise_if_nonfinite_state(state: Fci2FieldState, *, label: str) -> None:
    if not np.isfinite(np.asarray(state.density, dtype=np.float64)).all():
        raise FloatingPointError(f"non-finite density encountered in {label}")
    if not np.isfinite(np.asarray(state.v_parallel, dtype=np.float64)).all():
        raise FloatingPointError(f"non-finite v_parallel encountered in {label}")
    if not np.isfinite(np.asarray(state.density_background, dtype=np.float64)).all():
        raise FloatingPointError(f"non-finite density_background encountered in {label}")


def _mms_coordinates(geometry: FciGeometry3D) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return geometry.logical_grid[..., 0], geometry.logical_grid[..., 1], geometry.logical_grid[..., 2]


def _mms_background_density(geometry: FciGeometry3D) -> jnp.ndarray:
    x, _, _ = _mms_coordinates(geometry)
    return 1.0 + alpha * jnp.cos(jnp.pi * x)


def _mms_phi(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    x, y, z = _mms_coordinates(geometry)
    return A * jnp.sin(jnp.pi * x) * jnp.cos(2.0 * jnp.pi * y) * jnp.sin(jnp.pi * z) * jnp.cos(omega * time)


def _mms_density(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    density_background = _mms_background_density(geometry)
    return density_background * jnp.exp(_mms_phi(geometry, time))


def _mms_v_parallel(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    x, y, z = _mms_coordinates(geometry)
    return B * jnp.cos(jnp.pi * x) * jnp.sin(2.0 * jnp.pi * y) * jnp.cos(jnp.pi * z) * jnp.sin(omega * time)


def _mms_density_source(geometry: FciGeometry3D, time: float, *, rho_star_value: float) -> jnp.ndarray:
    x, y, z = _mms_coordinates(geometry)
    sx = jnp.sin(jnp.pi * x)
    cx = jnp.cos(jnp.pi * x)
    sy = jnp.sin(2.0 * jnp.pi * y)
    cy = jnp.cos(2.0 * jnp.pi * y)
    sz = jnp.sin(jnp.pi * z)
    st = jnp.sin(omega * time)
    ct = jnp.cos(omega * time)
    density_background = _mms_background_density(geometry)
    density = _mms_density(geometry, time)
    return density * (
        -A * omega * sx * cy * sz * st
        - (2.0 * alpha * (jnp.pi**2) * A / (rho_star_value * B0 * density_background)) * sx**2 * sy * sz * ct
        - jnp.pi * B * cx * sy * sz * st
    )


def _mms_v_parallel_source(geometry: FciGeometry3D, time: float, *, rho_star_value: float) -> jnp.ndarray:
    x, y, z = _mms_coordinates(geometry)
    sx = jnp.sin(jnp.pi * x)
    cx = jnp.cos(jnp.pi * x)
    sy = jnp.sin(2.0 * jnp.pi * y)
    cy = jnp.cos(2.0 * jnp.pi * y)
    sz = jnp.sin(jnp.pi * z)
    st = jnp.sin(omega * time)
    ct = jnp.cos(omega * time)
    return B * omega * cx * sy * jnp.cos(jnp.pi * z) * ct + (
        2.0 * (jnp.pi**2) * A * B / (rho_star_value * B0)
    ) * sz * jnp.cos(jnp.pi * z) * ct * st * (cx**2 * cy**2 - sx**2 * sy**2)


def _mms_exact_state(geometry: FciGeometry3D, time: float) -> Fci2FieldState:
    return Fci2FieldState(
        density=_mms_density(geometry, time),
        v_parallel=_mms_v_parallel(geometry, time),
        density_background=_mms_background_density(geometry),
    )


def _apply_mms_dirichlet_boundaries(state: Fci2FieldState, geometry: FciGeometry3D, time: float) -> Fci2FieldState:
    exact = _mms_exact_state(geometry, time)
    density = jnp.asarray(state.density, dtype=jnp.float64)
    v_parallel = jnp.asarray(state.v_parallel, dtype=jnp.float64)
    density = density.at[0, :, :].set(exact.density[0, :, :])
    density = density.at[-1, :, :].set(exact.density[-1, :, :])
    density = density.at[:, 0, :].set(exact.density[:, 0, :])
    density = density.at[:, -1, :].set(exact.density[:, -1, :])
    density = density.at[:, :, 0].set(exact.density[:, :, 0])
    density = density.at[:, :, -1].set(exact.density[:, :, -1])
    v_parallel = v_parallel.at[0, :, :].set(exact.v_parallel[0, :, :])
    v_parallel = v_parallel.at[-1, :, :].set(exact.v_parallel[-1, :, :])
    v_parallel = v_parallel.at[:, 0, :].set(exact.v_parallel[:, 0, :])
    v_parallel = v_parallel.at[:, -1, :].set(exact.v_parallel[:, -1, :])
    v_parallel = v_parallel.at[:, :, 0].set(exact.v_parallel[:, :, 0])
    v_parallel = v_parallel.at[:, :, -1].set(exact.v_parallel[:, :, -1])
    return Fci2FieldState(
        density=density,
        v_parallel=v_parallel,
        density_background=exact.density_background,
    )


def _mms_z_cut_indices(geometry: FciGeometry3D) -> tuple[int, int, int, int]:
    z_values = np.asarray(geometry.logical_grid[0, 0, :, 2], dtype=np.float64)
    z_cuts = np.linspace(0.1, 0.9, 4)
    return tuple(int(np.argmin(np.abs(z_values - cut))) for cut in z_cuts)


def _mms_slice_trace(state: Fci2FieldState, z_indices: tuple[int, int, int, int]) -> tuple[jnp.ndarray, jnp.ndarray]:
    density_slices = jnp.stack([state.density[:, :, z_index] for z_index in z_indices], axis=0)
    v_parallel_slices = jnp.stack([state.v_parallel[:, :, z_index] for z_index in z_indices], axis=0)
    return density_slices, v_parallel_slices


def fci_2field_rk4(
    state: Fci2FieldState,
    *,
    geometry: FciGeometry3D,
    time: float,
    timestep: float,
    parameters: Fci2FieldRhsParameters,
) -> Fci2FieldState:
    """Advance the two-field state by one classical RK4 step."""

    stage_0 = _apply_mms_dirichlet_boundaries(state, geometry, time)
    _raise_if_nonfinite_state(stage_0, label="rk4 stage_0")
    k1 = compute_2field_rhs(
        stage_0,
        geometry=geometry,
        parameters=parameters,
        periodic_axes=(False, False, False),
        density_source=_mms_density_source(geometry, time, rho_star_value=parameters.rho_star),
        v_parallel_source=_mms_v_parallel_source(geometry, time, rho_star_value=parameters.rho_star),
    ).rhs
    _raise_if_nonfinite_state(k1, label="rk4 k1")
    stage_1 = _apply_mms_dirichlet_boundaries(_add_state(stage_0, k1, scale=0.5 * timestep), geometry, time + 0.5 * timestep)
    _raise_if_nonfinite_state(stage_1, label="rk4 stage_1")
    k2 = compute_2field_rhs(
        stage_1,
        geometry=geometry,
        parameters=parameters,
        periodic_axes=(False, False, False),
        density_source=_mms_density_source(geometry, time + 0.5 * timestep, rho_star_value=parameters.rho_star),
        v_parallel_source=_mms_v_parallel_source(geometry, time + 0.5 * timestep, rho_star_value=parameters.rho_star),
    ).rhs
    _raise_if_nonfinite_state(k2, label="rk4 k2")
    stage_2 = _apply_mms_dirichlet_boundaries(_add_state(stage_0, k2, scale=0.5 * timestep), geometry, time + 0.5 * timestep)
    _raise_if_nonfinite_state(stage_2, label="rk4 stage_2")
    k3 = compute_2field_rhs(
        stage_2,
        geometry=geometry,
        parameters=parameters,
        periodic_axes=(False, False, False),
        density_source=_mms_density_source(geometry, time + 0.5 * timestep, rho_star_value=parameters.rho_star),
        v_parallel_source=_mms_v_parallel_source(geometry, time + 0.5 * timestep, rho_star_value=parameters.rho_star),
    ).rhs
    _raise_if_nonfinite_state(k3, label="rk4 k3")
    stage_3 = _apply_mms_dirichlet_boundaries(_add_state(stage_0, k3, scale=timestep), geometry, time + timestep)
    _raise_if_nonfinite_state(stage_3, label="rk4 stage_3")
    k4 = compute_2field_rhs(
        stage_3,
        geometry=geometry,
        parameters=parameters,
        periodic_axes=(False, False, False),
        density_source=_mms_density_source(geometry, time + timestep, rho_star_value=parameters.rho_star),
        v_parallel_source=_mms_v_parallel_source(geometry, time + timestep, rho_star_value=parameters.rho_star),
    ).rhs
    _raise_if_nonfinite_state(k4, label="rk4 k4")
    next_state = _apply_mms_dirichlet_boundaries(
        Fci2FieldState(
            density=state.density + (timestep / 6.0) * (k1.density + 2.0 * k2.density + 2.0 * k3.density + k4.density),
            v_parallel=state.v_parallel
            + (timestep / 6.0) * (k1.v_parallel + 2.0 * k2.v_parallel + 2.0 * k3.v_parallel + k4.v_parallel),
            density_background=state.density_background,
        ),
        geometry,
        time + timestep,
    )
    _raise_if_nonfinite_state(next_state, label="rk4 next_state")
    return next_state


def _assert_mms_slab_geometry(geometry: FciGeometry3D) -> None:
    assert geometry.shape == geometry.logical_grid.shape[:3]
    assert jnp.allclose(geometry.logical_grid[0, 0, 0], jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64))
    assert jnp.allclose(geometry.logical_grid[-1, -1, -1], jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64))
    assert jnp.allclose(geometry.Bmag, B0)
    assert jnp.allclose(geometry.B_contravariant[..., 0], 0.0)
    assert jnp.allclose(geometry.B_contravariant[..., 1], 0.0)
    assert jnp.allclose(geometry.B_contravariant[..., 2], 1.0)
    assert jnp.allclose(geometry.g11, 1.0)
    assert jnp.allclose(geometry.g22, 1.0)
    assert jnp.allclose(geometry.g33, 1.0)
    assert jnp.allclose(geometry.forward_boundary, False)
    assert jnp.allclose(geometry.backward_boundary, False)


def simulate_mms_2field_slab(
    geometry: FciGeometry3D,
    *,
    timestep: float | None = None,
    final_time: float = tf,
    rho_star_value: float = rho_star,
) -> tuple[Fci2FieldState, jnp.ndarray, list[Fci2FieldState]]:
    _assert_mms_slab_geometry(geometry)

    parameters = Fci2FieldRhsParameters(rho_star=rho_star_value)
    dt = float(final_time) / float(num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)

    initial_state = _apply_mms_dirichlet_boundaries(_mms_exact_state(geometry, 0.0), geometry, 0.0)
    state = initial_state
    time_value = 0.0
    history: list[jnp.ndarray] = []
    state_history: list[Fci2FieldState] = [initial_state]
    for _ in range(steps):
        state = fci_2field_rk4(
            state,
            geometry=geometry,
            time=jnp.asarray(time_value, dtype=jnp.float64),
            timestep=dt,
            parameters=parameters,
        )
        time_value += dt
        state_history.append(state)
        exact = _mms_exact_state(geometry, time_value)
        diagnostics = jnp.asarray(
            [
                time_value,
                jnp.sqrt(jnp.mean(jnp.square(state.density - exact.density))),
                jnp.sqrt(jnp.mean(jnp.square(state.v_parallel - exact.v_parallel))),
            ],
            dtype=jnp.float64,
        )
        history.append(diagnostics)
    return state, jnp.stack(history, axis=0), state_history


def _state_history_to_slice_history(
    geometry: FciGeometry3D,
    state_history: list[Fci2FieldState],
    *,
    timestep: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    z_indices = _mms_z_cut_indices(geometry)
    times = jnp.asarray([float(index) * float(timestep) for index in range(len(state_history))], dtype=jnp.float64)
    density_history: list[jnp.ndarray] = []
    v_parallel_history: list[jnp.ndarray] = []
    for state in state_history:
        density_slices, v_parallel_slices = _mms_slice_trace(state, z_indices)
        density_history.append(density_slices.astype(jnp.float32))
        v_parallel_history.append(v_parallel_slices.astype(jnp.float32))
    return times, jnp.stack(density_history, axis=0), jnp.stack(v_parallel_history, axis=0)


def _combined_error_statistics(final_state: Fci2FieldState, geometry: FciGeometry3D, time: float) -> tuple[float, float, float]:
    exact = _mms_exact_state(geometry, time)
    error = jnp.concatenate(
        [
            jnp.ravel(jnp.abs(final_state.density - exact.density)),
            jnp.ravel(jnp.abs(final_state.v_parallel - exact.v_parallel)),
        ]
    )
    return float(jnp.sqrt(jnp.mean(error**2))), float(jnp.median(error)), float(jnp.max(error))


def _plot_high_resolution_slices(
    state: Fci2FieldState,
    exact_state: Fci2FieldState,
    geometry: FciGeometry3D,
    resolution: int,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.logical_grid[:, 0, 0, 0], dtype=np.float64)
    y_values = np.asarray(geometry.logical_grid[0, :, 0, 1], dtype=np.float64)
    z_values = np.asarray(geometry.logical_grid[0, 0, :, 2], dtype=np.float64)
    z_cuts = np.linspace(0.1, 0.9, 2)
    z_indices = [int(np.argmin(np.abs(z_values - cut))) for cut in z_cuts]

    density = np.asarray(state.density, dtype=np.float64)
    v_parallel = np.asarray(state.v_parallel, dtype=np.float64)
    exact_density = np.asarray(exact_state.density, dtype=np.float64)
    exact_v_parallel = np.asarray(exact_state.v_parallel, dtype=np.float64)

    density_vmax = float(np.max(np.abs(np.stack([density[:, :, z_indices], exact_density[:, :, z_indices]], axis=0))))
    v_parallel_vmax = float(np.max(np.abs(np.stack([v_parallel[:, :, z_indices], exact_v_parallel[:, :, z_indices]], axis=0))))

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.5), constrained_layout=True)
    density_im = None
    v_parallel_im = None

    panel_labels = [("sim", state), ("exact", exact_state)]
    for cut_index, (cut, z_index) in enumerate(zip(z_cuts, z_indices)):
        for panel_index, (panel_label, panel_state) in enumerate(panel_labels):
            density_ax = axes[0, 2 * cut_index + panel_index]
            v_parallel_ax = axes[1, 2 * cut_index + panel_index]
            density_slice = np.asarray(panel_state.density[:, :, z_index], dtype=np.float64).T
            v_parallel_slice = np.asarray(panel_state.v_parallel[:, :, z_index], dtype=np.float64).T
            density_im = density_ax.imshow(
                density_slice,
                origin="lower",
                extent=(float(x_values[0]), float(x_values[-1]), float(y_values[0]), float(y_values[-1])),
                aspect="auto",
                cmap="viridis",
                vmin=-density_vmax,
                vmax=density_vmax,
            )
            v_parallel_im = v_parallel_ax.imshow(
                v_parallel_slice,
                origin="lower",
                extent=(float(x_values[0]), float(x_values[-1]), float(y_values[0]), float(y_values[-1])),
                aspect="auto",
                cmap="coolwarm",
                vmin=-v_parallel_vmax,
                vmax=v_parallel_vmax,
            )
            density_ax.set_title(f"{panel_label}, z={z_values[z_index]:.3f}")
            v_parallel_ax.set_title(f"{panel_label}, z={z_values[z_index]:.3f}")
            density_ax.set_xlabel("x")
            density_ax.set_ylabel("y")
            v_parallel_ax.set_xlabel("x")
            v_parallel_ax.set_ylabel("y")

    if density_im is not None:
        fig.colorbar(density_im, ax=axes[0, :].ravel().tolist(), shrink=0.88, pad=0.02)
    if v_parallel_im is not None:
        fig.colorbar(v_parallel_im, ax=axes[1, :].ravel().tolist(), shrink=0.88, pad=0.02)

    fig.suptitle(f"2-field MMS fields at resolution {int(resolution)}")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_high_resolution_slice_movie(
    times: jnp.ndarray,
    density_history: jnp.ndarray,
    v_parallel_history: jnp.ndarray,
    geometry: FciGeometry3D,
    resolution: int,
    output_path: str,
    frame_stride: int = 5,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    z_values = np.asarray(geometry.logical_grid[0, 0, :, 2], dtype=np.float64)
    z_cuts = np.linspace(0.1, 0.9, 4)
    z_indices = [int(np.argmin(np.abs(z_values - cut))) for cut in z_cuts]
    x_values = np.asarray(geometry.logical_grid[:, 0, 0, 0], dtype=np.float64)
    y_values = np.asarray(geometry.logical_grid[0, :, 0, 1], dtype=np.float64)

    density_data = np.asarray(density_history, dtype=np.float64)
    v_parallel_data = np.asarray(v_parallel_history, dtype=np.float64)
    frame_indices = np.arange(0, int(times.shape[0]), max(1, int(frame_stride)), dtype=np.int64)
    if frame_indices[-1] != int(times.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times.shape[0]) - 1)
    density_vmax = float(np.max(np.abs(density_data)))
    v_parallel_vmax = float(np.max(np.abs(v_parallel_data)))

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.5), constrained_layout=True)
    images = []
    for row in range(2):
        for col in range(4):
            ax = axes[row, col]
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if row == 0:
                ax.set_title(f"density, z={z_values[z_indices[col]]:.3f}")
                image = ax.imshow(
                    density_data[0, col].T,
                    origin="lower",
                    extent=(float(x_values[0]), float(x_values[-1]), float(y_values[0]), float(y_values[-1])),
                    aspect="auto",
                    cmap="viridis",
                    vmin=-density_vmax,
                    vmax=density_vmax,
                )
            else:
                ax.set_title(f"v_parallel, z={z_values[z_indices[col]]:.3f}")
                image = ax.imshow(
                    v_parallel_data[0, col].T,
                    origin="lower",
                    extent=(float(x_values[0]), float(x_values[-1]), float(y_values[0]), float(y_values[-1])),
                    aspect="auto",
                    cmap="coolwarm",
                    vmin=-v_parallel_vmax,
                    vmax=v_parallel_vmax,
                )
            images.append(image)

    suptitle = fig.suptitle(f"2-field MMS fields at resolution {int(resolution)}")

    def update(frame_index: int):
        actual_index = int(frame_indices[frame_index])
        time_value = float(times[actual_index])
        for col in range(4):
            images[col].set_data(density_data[actual_index, col].T)
            images[4 + col].set_data(v_parallel_data[actual_index, col].T)
            axes[0, col].set_title(f"density, z={z_values[z_indices[col]]:.3f}, t={time_value:.3f}")
            axes[1, col].set_title(f"v_parallel, z={z_values[z_indices[col]]:.3f}, t={time_value:.3f}")
        suptitle.set_text(f"2-field MMS fields at resolution {int(resolution)}, t={time_value:.3f}")
        return images

    animator = animation.FuncAnimation(fig, update, frames=int(frame_indices.shape[0]), interval=100, blit=False)
    writer = animation.PillowWriter(fps=10)
    animator.save(output_path, writer=writer)
    plt.close(fig)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    resolutions = np.asarray([15, 30, 60,120], dtype=np.int64)
    successful_resolutions: list[int] = []
    mean_errors: list[float] = []
    max_errors: list[float] = []
    highest_resolution_state: Fci2FieldState | None = None
    highest_resolution_geometry: FciGeometry3D | None = None
    highest_resolution_resolution: int | None = None
    highest_resolution_times: jnp.ndarray | None = None
    highest_resolution_density_history: jnp.ndarray | None = None
    highest_resolution_v_parallel_history: jnp.ndarray | None = None

    for resolution in resolutions:
        geometry = build_slab_2field_geometry(int(resolution), int(resolution), int(resolution))
        steps = _resolution_step_count(int(resolution))
        dt = float(tf) / float(steps)
        start = time.perf_counter()
        try:
            final_state, _history, state_history = simulate_mms_2field_slab(geometry, final_time=tf, timestep=dt, rho_star_value=rho_star)
            elapsed = time.perf_counter() - start
            mean_error, median_error, max_error = _combined_error_statistics(final_state, geometry, tf)
        except FloatingPointError as exc:
            elapsed = time.perf_counter() - start
            print(f"WARNING: res={int(resolution)} failed with non-finite values after {elapsed:.6e} s: {exc}")
            continue

        mean_errors.append(mean_error)
        max_errors.append(max_error)
        successful_resolutions.append(int(resolution))
        print(
            f"res={int(resolution)}: steps={steps}, total_time={elapsed:.6e} s, "
            f"avg_step_time={elapsed / float(steps):.6e} s, "
            f"mean_error={mean_error:.6e}, median_error={median_error:.6e}, max_error={max_error:.6e}"
        )

        highest_resolution_state = final_state
        highest_resolution_geometry = geometry
        highest_resolution_resolution = int(resolution)
        highest_resolution_times, highest_resolution_density_history, highest_resolution_v_parallel_history = _state_history_to_slice_history(
            geometry,
            state_history,
            timestep=dt,
        )

    if mean_errors and max_errors:
        plotted_resolutions = np.asarray(successful_resolutions, dtype=np.int64)
        log_resolutions = np.log(plotted_resolutions.astype(np.float64))
        mean_log_errors = np.log(np.asarray(mean_errors, dtype=np.float64))
        max_log_errors = np.log(np.asarray(max_errors, dtype=np.float64))

        mean_slope, mean_intercept = np.polyfit(log_resolutions, mean_log_errors, 1)
        max_slope, max_intercept = np.polyfit(log_resolutions, max_log_errors, 1)

        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.loglog(plotted_resolutions, mean_errors, "o-", label=f"mean, order {mean_slope:.2f}")
        ax.loglog(plotted_resolutions, max_errors, "^-", label=f"max, order {max_slope:.2f}")
        ax.loglog(
            plotted_resolutions,
            np.exp(mean_intercept) * plotted_resolutions.astype(np.float64) ** mean_slope,
            "--",
            color=ax.lines[0].get_color(),
        )
        ax.loglog(
            plotted_resolutions,
            np.exp(max_intercept) * plotted_resolutions.astype(np.float64) ** max_slope,
            "--",
            color=ax.lines[1].get_color(),
        )
        ax.set_xlabel("resolution")
        ax.set_ylabel("absolute error")
        ax.set_title("2-field slab MMS convergence")
        ax.grid(True, which="both", linestyle=":", alpha=0.45)
        ax.legend()
        fig.tight_layout()
        fig.savefig("slab_2field_convergence.png", dpi=200)
        plt.close(fig)
    else:
        print("WARNING: no valid resolutions completed, skipping convergence plot.")

    if highest_resolution_state is not None and highest_resolution_geometry is not None and highest_resolution_resolution is not None:
        highest_resolution_exact_state = _mms_exact_state(highest_resolution_geometry, tf)
        _plot_high_resolution_slices(
            highest_resolution_state,
            highest_resolution_exact_state,
            highest_resolution_geometry,
            highest_resolution_resolution,
            "slab_2field_slices.png",
        )
    if (
        highest_resolution_times is not None
        and highest_resolution_density_history is not None
        and highest_resolution_v_parallel_history is not None
        and highest_resolution_geometry is not None
        and highest_resolution_resolution is not None
    ):
        _save_high_resolution_slice_movie(
            highest_resolution_times,
            highest_resolution_density_history,
            highest_resolution_v_parallel_history,
            highest_resolution_geometry,
            highest_resolution_resolution,
            "slab_2field_slices.gif",
            frame_stride=5,
        )
