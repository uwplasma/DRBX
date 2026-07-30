from __future__ import annotations

import time as time_module

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry import (
    FciGeometry3D,
    StencilBuilderContext,
    build_local_curvature_coefficients,
    build_local_direct_stencil_one_sided_physical_from_halo,
    build_shifted_torus_geometry,
    logical_grid_from_axis_vectors,
)
from drbx.native import (
    Fci2FieldRhsParameters,
    Fci2FieldState,
    HaloExchange3D,
    LocalPeriodicTopologyRule3D,
    TopologyHaloFiller3D,
    assemble_local_fci_geometry,
    build_local_fci_geometries,
    compute_local_2field_rhs,
    inject_owned_field_to_halo,
    make_shard_mesh,
)
from jax.sharding import NamedSharding, PartitionSpec as P


A = 0.1
Bv = 0.1
alpha = 0.2
omega = 2.0 * jnp.pi
rho_star = 1.0
M_phi = 2
N_phi = 3
M_v = 3
N_v = 4
sigma = 0.0
r0 = 3.0
alpha_value = 0.25
iota = 1.1
c_phi = 3.0
x_min = 0.15
x_max = 1.0
tf = 0.1
num_steps = 100


def _resolution_step_count(resolution: int, *, base_resolution: int = 20, base_steps: int = num_steps) -> int:
    scale = np.sqrt(float(resolution) / float(base_resolution))
    return max(1, int(round(float(base_steps) * scale)))


# The verified shifted-torus geometry constructor now lives in the package
# (``drbx.geometry.build_shifted_torus_geometry``). Keep the historical name
# as a thin alias so the MMS harness below is unchanged; the package defaults
# match the module-level constants above.
build_shifted_torus_2field_geometry = build_shifted_torus_geometry


def _shifted_torus_coordinates(geometry: FciGeometry3D) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    logical_grid = logical_grid_from_axis_vectors(*geometry.grid.logical_axis_vectors)
    x = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    zeta = logical_grid[..., 2]
    x_mid = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_mid)
    return x, theta_shift, theta, zeta


def _shifted_torus_background_density(geometry: FciGeometry3D) -> jnp.ndarray:
    return jnp.ones(geometry.shape, dtype=jnp.float64)


def _shifted_torus_phi(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    x, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    radial_envelope = jnp.sin(jnp.pi * x)
    return float(A) * radial_envelope * jnp.cos(float(M_phi) * theta_shift) * jnp.sin(float(N_phi) * zeta) * jnp.cos(float(omega) * time)


def _shifted_torus_density(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    density_background = _shifted_torus_background_density(geometry)
    return density_background * jnp.exp(_shifted_torus_phi(geometry, time))


def _shifted_torus_v_parallel(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    x, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    radial_envelope = jnp.cos(jnp.pi * x)
    return float(Bv) * radial_envelope * jnp.sin(float(M_v) * theta_shift) * jnp.cos(float(N_v) * zeta) * jnp.sin(float(omega) * time)


def _shifted_torus_exact_state(geometry: FciGeometry3D, time: float) -> Fci2FieldState:
    return Fci2FieldState(
        density=_shifted_torus_density(geometry, time),
        v_parallel=_shifted_torus_v_parallel(geometry, time),
        density_background=_shifted_torus_background_density(geometry),
    )


def _shifted_torus_geometry_quantities(geometry: FciGeometry3D) -> tuple[jnp.ndarray, ...]:
    logical_grid = logical_grid_from_axis_vectors(*geometry.grid.logical_axis_vectors)
    x = jnp.asarray(logical_grid[..., 0], dtype=jnp.float64)
    theta = jnp.asarray(logical_grid[..., 1], dtype=jnp.float64)
    zeta = jnp.asarray(logical_grid[..., 2], dtype=jnp.float64)
    x_mid = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_mid)
    cos_shift = jnp.cos(theta_shift)
    sin_shift = jnp.sin(theta_shift)
    R = float(r0) + float(alpha_value) * x + x * cos_shift
    Q = 1.0 + float(alpha_value) * cos_shift
    J = x * R * Q
    D2 = (float(iota) ** 2) * x**2 + R**2
    D = jnp.sqrt(D2)
    P = float(alpha_value) + cos_shift
    E = x * Q + float(alpha_value) * R
    A_term = (float(iota) ** 2) * x + R * P
    return x, theta_shift, zeta, cos_shift, sin_shift, R, Q, J, D, P, E, A_term


def _shifted_torus_phi_derivatives(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, zeta, cos_shift, sin_shift, _, _, _, _, _, _, _ = _shifted_torus_geometry_quantities(geometry)
    sin_u = jnp.sin(jnp.pi * x)
    cos_u = jnp.cos(jnp.pi * x)
    sin_mphi = jnp.sin(float(M_phi) * theta_shift)
    cos_mphi = jnp.cos(float(M_phi) * theta_shift)
    sin_nphi = jnp.sin(float(N_phi) * zeta)
    cos_nphi = jnp.cos(float(N_phi) * zeta)
    cos_omega_t = jnp.cos(float(omega) * time)
    sin_omega_t = jnp.sin(float(omega) * time)

    phi = float(A) * sin_u * cos_mphi * sin_nphi * cos_omega_t
    phi_u = float(A) * (
        jnp.pi * cos_u * cos_mphi - float(sigma) * float(M_phi) * sin_u * sin_mphi
    ) * sin_nphi * cos_omega_t
    phi_theta = -float(A) * float(M_phi) * sin_u * sin_mphi * sin_nphi * cos_omega_t
    phi_zeta = float(A) * float(N_phi) * sin_u * cos_mphi * cos_nphi * cos_omega_t
    phi_t = -float(A) * float(omega) * sin_u * cos_mphi * sin_nphi * sin_omega_t
    return phi, phi_u, phi_theta, phi_zeta, phi_t


def _shifted_torus_density_derivatives(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, zeta, _, _, _, _, _, _, _, _, _ = _shifted_torus_geometry_quantities(geometry)
    phi, phi_u, phi_theta, phi_zeta, phi_t = _shifted_torus_phi_derivatives(geometry, time)
    n0 = jnp.ones_like(x)
    n0_u = jnp.zeros_like(x)
    exp_phi = jnp.exp(phi)
    density = n0 * exp_phi
    density_u = n0_u * exp_phi + n0 * exp_phi * phi_u
    density_theta = density * phi_theta
    density_zeta = density * phi_zeta
    density_t = density * phi_t
    return density, density_u, density_theta, density_zeta, density_t


def _shifted_torus_v_parallel_derivatives(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, zeta, _, _, _, _, _, _, _, _, _ = _shifted_torus_geometry_quantities(geometry)
    sin_u = jnp.sin(jnp.pi * x)
    cos_u = jnp.cos(jnp.pi * x)
    sin_mv = jnp.sin(float(M_v) * theta_shift)
    cos_mv = jnp.cos(float(M_v) * theta_shift)
    sin_nv = jnp.sin(float(N_v) * zeta)
    cos_nv = jnp.cos(float(N_v) * zeta)
    sin_omega_t = jnp.sin(float(omega) * time)
    cos_omega_t = jnp.cos(float(omega) * time)

    v_parallel = float(Bv) * cos_u * sin_mv * cos_nv * sin_omega_t
    v_parallel_u = float(Bv) * (
        -jnp.pi * sin_u * sin_mv + float(sigma) * float(M_v) * cos_u * cos_mv
    ) * cos_nv * sin_omega_t
    v_parallel_theta = float(Bv) * cos_u * float(M_v) * cos_mv * cos_nv * sin_omega_t
    v_parallel_zeta = -float(Bv) * cos_u * sin_mv * float(N_v) * sin_nv * sin_omega_t
    v_parallel_t = float(Bv) * cos_u * sin_mv * cos_nv * float(omega) * cos_omega_t
    return v_parallel, v_parallel_u, v_parallel_theta, v_parallel_zeta, v_parallel_t


def _shifted_torus_poisson_bracket(
    f_u: jnp.ndarray,
    f_theta: jnp.ndarray,
    f_zeta: jnp.ndarray,
    g_u: jnp.ndarray,
    g_theta: jnp.ndarray,
    g_zeta: jnp.ndarray,
    geometry: FciGeometry3D,
) -> jnp.ndarray:
    x, theta_shift, zeta, cos_shift, sin_shift, R, Q, J, D, _, _, _ = _shifted_torus_geometry_quantities(geometry)
    return (
        1.0
        / (J * D)
        * (
            -float(alpha_value) * float(iota) * x * sin_shift * (f_theta * g_zeta - f_zeta * g_theta)
            + float(iota) * x**2 * (f_zeta * g_u - f_u * g_zeta)
            + R**2 * (f_u * g_theta - f_theta * g_u)
        )
    )


def _shifted_torus_curvature(field_u: jnp.ndarray, field_theta: jnp.ndarray, field_zeta: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    x, theta_shift, zeta, cos_shift, sin_shift, R, Q, J, D, P, E, A_term = _shifted_torus_geometry_quantities(geometry)
    K_u = (
        1.0
        / (2.0 * J)
        * (
            -2.0 * x * R * sin_shift / D
            + 2.0 * x * R**3 * sin_shift / D**3
            - x * R**2 * sin_shift * E / (D * J)
        )
    )
    K_theta = (
        -1.0
        / (2.0 * J)
        * (
            2.0 * R * P / D
            - 2.0 * R**2 * A_term / D**3
            + R**2 * Q * (R + x * P) / (D * J)
        )
    )
    K_zeta = (
        float(iota)
        / (2.0 * J)
        * (
            x * (2.0 + float(alpha_value) * cos_shift) / D
            - 2.0 * x**2 * A_term / D**3
            + 2.0 * float(alpha_value) * x**2 * R * sin_shift**2 / D**3
            + (x**2 * Q * (R + x * P) - float(alpha_value) * x**2 * sin_shift**2 * E) / (D * J)
        )
    )
    return K_u * field_u + K_theta * field_theta + K_zeta * field_zeta


def _shifted_torus_grad_parallel(field_theta: jnp.ndarray, field_zeta: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    _, _, _, _, _, _, _, _, D, _, _, _ = _shifted_torus_geometry_quantities(geometry)
    return (float(iota) * field_theta + field_zeta) / D


def _shifted_torus_density_source(geometry: FciGeometry3D, time: float, *, parameters: Fci2FieldRhsParameters) -> jnp.ndarray:
    phi, phi_u, phi_theta, phi_zeta, phi_t = _shifted_torus_phi_derivatives(geometry, time)
    density, density_u, density_theta, density_zeta, density_t = _shifted_torus_density_derivatives(geometry, time)
    v_parallel, _, v_parallel_theta, v_parallel_zeta, _ = _shifted_torus_v_parallel_derivatives(geometry, time)
    bmag = geometry.cell_bfield.Bmag
    poisson = _shifted_torus_poisson_bracket(
        phi_u,
        phi_theta,
        phi_zeta,
        density_u,
        density_theta,
        density_zeta,
        geometry,
    )
    curvature_density = _shifted_torus_curvature(density_u, density_theta, density_zeta, geometry)
    curvature_phi = _shifted_torus_curvature(phi_u, phi_theta, phi_zeta, geometry)
    grad_parallel_v = _shifted_torus_grad_parallel(v_parallel_theta, v_parallel_zeta, geometry)
    rho_star_value = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    return density_t + (1.0 / (rho_star_value * bmag)) * poisson - (2.0 / bmag) * curvature_density + (2.0 * density / bmag) * curvature_phi + density * grad_parallel_v


def _shifted_torus_v_parallel_source(geometry: FciGeometry3D, time: float, *, parameters: Fci2FieldRhsParameters) -> jnp.ndarray:
    phi, phi_u, phi_theta, phi_zeta, _ = _shifted_torus_phi_derivatives(geometry, time)
    v_parallel, v_parallel_u, v_parallel_theta, v_parallel_zeta, v_parallel_t = _shifted_torus_v_parallel_derivatives(geometry, time)
    bmag = geometry.cell_bfield.Bmag
    poisson = _shifted_torus_poisson_bracket(
        phi_u,
        phi_theta,
        phi_zeta,
        v_parallel_u,
        v_parallel_theta,
        v_parallel_zeta,
        geometry,
    )
    rho_star_value = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    return v_parallel_t + (1.0 / (rho_star_value * bmag)) * poisson


def simulate_mms_2field_shifted_torus(
    geometry: FciGeometry3D,
    *,
    timestep: float | None = None,
    final_time: float = tf,
    rho_star_value: float = rho_star,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
) -> tuple[Fci2FieldState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evolve the shifted-torus MMS system entirely through local ``shard_map``."""

    parameters = Fci2FieldRhsParameters(rho_star=rho_star_value)
    dt = float(final_time) / float(num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)
    mesh = make_shard_mesh(shard_counts)
    local_bundle = build_local_fci_geometries(
        geometry, shard_counts, halo_width=1, periodic_axes=(False, True, True)
    )
    partition = P("x", "y", "z")
    sharding = NamedSharding(mesh, partition)
    cell_fields = jax.device_put(local_bundle.cell_fields, sharding)

    def _curvature(cell_fields_owned):
        local_geometry = assemble_local_fci_geometry(local_bundle, cell_fields_owned)
        return build_local_curvature_coefficients(
            local_geometry, local_bundle.domain,
            periodic_axes=(False, True, True),
            axis_regular_axes=(False, False, False),
        )

    curvature = jax.jit(jax.shard_map(
        _curvature, mesh=mesh, in_specs=partition, out_specs=partition, check_vma=False
    ))(cell_fields)

    def _step_kernel(density, velocity, background, c0, v0, c1, v1, c2, v2, c3, v3, cell_fields_owned, curvature_owned):
        local_geometry = assemble_local_fci_geometry(local_bundle, cell_fields_owned)
        context = StencilBuilderContext(layout=local_bundle.domain.layout, domain=local_bundle.domain)
        exchange = HaloExchange3D()
        topology = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))

        def stencil(field, _geometry):
            halo = inject_owned_field_to_halo(jnp.asarray(field, dtype=jnp.float64), local_bundle.domain.layout)
            halo = topology(exchange(halo, local_bundle.domain), local_bundle.domain)
            return build_local_direct_stencil_one_sided_physical_from_halo(halo, local_geometry, context)

        def rhs(current, density_source, velocity_source):
            result = compute_local_2field_rhs(
                current, geometry=local_geometry, stencil_builder=stencil,
                parameters=parameters, curvature_coefficients=curvature_owned,
                density_source=density_source, v_parallel_source=velocity_source,
            )
            return result.rhs

        current = Fci2FieldState(density, velocity, background)
        k1 = rhs(current, c0, v0)
        k2 = rhs(Fci2FieldState(density + 0.5 * dt * k1.density, velocity + 0.5 * dt * k1.v_parallel, background), c1, v1)
        k3 = rhs(Fci2FieldState(density + 0.5 * dt * k2.density, velocity + 0.5 * dt * k2.v_parallel, background), c2, v2)
        k4 = rhs(Fci2FieldState(density + dt * k3.density, velocity + dt * k3.v_parallel, background), c3, v3)
        return (
            density + dt * (k1.density + 2.0 * k2.density + 2.0 * k3.density + k4.density) / 6.0,
            velocity + dt * (k1.v_parallel + 2.0 * k2.v_parallel + 2.0 * k3.v_parallel + k4.v_parallel) / 6.0,
            background,
        )

    kernel = jax.jit(jax.shard_map(
        _step_kernel, mesh=mesh, in_specs=(partition,) * 13, out_specs=(partition,) * 3, check_vma=False
    ))

    def step(state, time_value):
        source_times = (time_value, time_value + 0.5 * dt, time_value + 0.5 * dt, time_value + dt)
        sources = []
        for stage_time in source_times:
            sources.extend((
                jax.device_put(_shifted_torus_density_source(geometry, stage_time, parameters=parameters), sharding),
                jax.device_put(_shifted_torus_v_parallel_source(geometry, stage_time, parameters=parameters), sharding),
            ))
        fields = tuple(jax.device_put(jnp.asarray(getattr(state, name), dtype=jnp.float64), sharding) for name in ("density", "v_parallel", "density_background"))
        result = kernel(*fields, *sources, cell_fields, curvature)
        return Fci2FieldState(*[jax.block_until_ready(value) for value in result])

    initial_exact = _shifted_torus_exact_state(geometry, 0.0)
    state = initial_exact
    time_value = 0.0
    times: list[float] = [0.0]
    density_history: list[jnp.ndarray] = [jnp.asarray(state.density, dtype=jnp.float32)]
    v_parallel_history: list[jnp.ndarray] = [jnp.asarray(state.v_parallel, dtype=jnp.float32)]

    for _ in range(steps):
        state = step(state, time_value)
        time_value += dt
        times.append(time_value)
        density_history.append(jnp.asarray(state.density, dtype=jnp.float32))
        v_parallel_history.append(jnp.asarray(state.v_parallel, dtype=jnp.float32))
    print(f"shifted_torus_2field local shard_map complete: shards={tuple(shard_counts)}, steps={steps}")

    return (
        state,
        jnp.asarray(times, dtype=jnp.float64),
        jnp.stack(density_history, axis=0),
        jnp.stack(v_parallel_history, axis=0),
    )


def _shifted_torus_z_cut_indices(geometry: FciGeometry3D, count: int) -> tuple[int, ...]:
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    z_cuts = np.linspace(0.1, 0.9, count)
    return tuple(int(np.argmin(np.abs(z_values - cut))) for cut in z_cuts)


def _shifted_torus_field_slices(field: jnp.ndarray, z_indices: tuple[int, ...]) -> jnp.ndarray:
    return jnp.stack([field[:, :, z_index] for z_index in z_indices], axis=0)


def _combined_error_statistics(final_state: Fci2FieldState, geometry: FciGeometry3D, time: float) -> tuple[float, float, float]:
    exact = _shifted_torus_exact_state(geometry, time)
    density_error = jnp.abs(final_state.density - exact.density)[1:-1, :, :]
    v_parallel_error = jnp.abs(final_state.v_parallel - exact.v_parallel)[1:-1, :, :]
    error = jnp.concatenate(
        [
            jnp.ravel(density_error),
            jnp.ravel(v_parallel_error),
        ]
    )
    return float(jnp.sqrt(jnp.mean(error**2))), float(jnp.median(error)), float(jnp.max(error))


def _plot_final_slices(
    state: Fci2FieldState,
    exact_state: Fci2FieldState,
    geometry: FciGeometry3D,
    resolution: int,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    z_indices = _shifted_torus_z_cut_indices(geometry, 2)

    density = np.asarray(state.density, dtype=np.float64)
    v_parallel = np.asarray(state.v_parallel, dtype=np.float64)
    exact_density = np.asarray(exact_state.density, dtype=np.float64)
    exact_v_parallel = np.asarray(exact_state.v_parallel, dtype=np.float64)

    density_slices = np.asarray(_shifted_torus_field_slices(jnp.asarray(density), z_indices), dtype=np.float64)
    exact_density_slices = np.asarray(_shifted_torus_field_slices(jnp.asarray(exact_density), z_indices), dtype=np.float64)
    v_parallel_slices = np.asarray(_shifted_torus_field_slices(jnp.asarray(v_parallel), z_indices), dtype=np.float64)
    exact_v_parallel_slices = np.asarray(_shifted_torus_field_slices(jnp.asarray(exact_v_parallel), z_indices), dtype=np.float64)

    density_vmax = float(np.max(np.abs(np.stack([density_slices, exact_density_slices], axis=0))))
    v_parallel_vmax = float(np.max(np.abs(np.stack([v_parallel_slices, exact_v_parallel_slices], axis=0))))

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.5), subplot_kw={"projection": "polar"}, constrained_layout=True)
    density_im = None
    v_parallel_im = None
    theta_grid, radius_grid = np.meshgrid(theta_values, x_values)

    for cut_index, z_index in enumerate(z_indices):
        density_slice = density_slices[cut_index]
        v_parallel_slice = v_parallel_slices[cut_index]
        exact_density_slice = exact_density_slices[cut_index]
        exact_v_parallel_slice = exact_v_parallel_slices[cut_index]

        density_im = axes[0, cut_index].pcolormesh(theta_grid, radius_grid, density_slice, shading="auto", cmap="viridis", vmin=-density_vmax, vmax=density_vmax)
        axes[0, cut_index].set_theta_zero_location("E")
        axes[0, cut_index].set_theta_direction(-1)
        axes[0, cut_index].set_ylim(0.0, float(x_values[-1]))
        axes[0, cut_index].set_title(f"sim, zeta={z_values[z_index]:.3f}")
        axes[0, cut_index].set_yticklabels([])

        density_im = axes[0, 2 + cut_index].pcolormesh(theta_grid, radius_grid, exact_density_slice, shading="auto", cmap="viridis", vmin=-density_vmax, vmax=density_vmax)
        axes[0, 2 + cut_index].set_theta_zero_location("E")
        axes[0, 2 + cut_index].set_theta_direction(-1)
        axes[0, 2 + cut_index].set_ylim(0.0, float(x_values[-1]))
        axes[0, 2 + cut_index].set_title(f"exact, zeta={z_values[z_index]:.3f}")
        axes[0, 2 + cut_index].set_yticklabels([])

        v_parallel_im = axes[1, cut_index].pcolormesh(theta_grid, radius_grid, v_parallel_slice, shading="auto", cmap="coolwarm", vmin=-v_parallel_vmax, vmax=v_parallel_vmax)
        axes[1, cut_index].set_theta_zero_location("E")
        axes[1, cut_index].set_theta_direction(-1)
        axes[1, cut_index].set_ylim(0.0, float(x_values[-1]))
        axes[1, cut_index].set_title(f"sim, zeta={z_values[z_index]:.3f}")
        axes[1, cut_index].set_yticklabels([])

        v_parallel_im = axes[1, 2 + cut_index].pcolormesh(theta_grid, radius_grid, exact_v_parallel_slice, shading="auto", cmap="coolwarm", vmin=-v_parallel_vmax, vmax=v_parallel_vmax)
        axes[1, 2 + cut_index].set_theta_zero_location("E")
        axes[1, 2 + cut_index].set_theta_direction(-1)
        axes[1, 2 + cut_index].set_ylim(0.0, float(x_values[-1]))
        axes[1, 2 + cut_index].set_title(f"exact, zeta={z_values[z_index]:.3f}")
        axes[1, 2 + cut_index].set_yticklabels([])

    if density_im is not None:
        fig.colorbar(density_im, ax=axes[0, :].ravel().tolist(), shrink=0.88, pad=0.02)
    if v_parallel_im is not None:
        fig.colorbar(v_parallel_im, ax=axes[1, :].ravel().tolist(), shrink=0.88, pad=0.02)

    fig.suptitle(f"Shifted-torus 2-field MMS fields at resolution {int(resolution)}")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_shifted_torus_movie(
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

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    z_indices = _shifted_torus_z_cut_indices(geometry, 4)

    density_data = np.asarray(density_history, dtype=np.float64)
    v_parallel_data = np.asarray(v_parallel_history, dtype=np.float64)
    frame_indices = np.arange(0, int(times.shape[0]), max(1, int(frame_stride)), dtype=np.int64)
    if frame_indices[-1] != int(times.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times.shape[0]) - 1)
    density_vmax = float(np.max(np.abs(density_data)))
    v_parallel_vmax = float(np.max(np.abs(v_parallel_data)))

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.5), subplot_kw={"projection": "polar"}, constrained_layout=True)
    images = []
    theta_grid, radius_grid = np.meshgrid(theta_values, x_values)
    for row in range(2):
        for col in range(4):
            ax = axes[row, col]
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(-1)
            ax.set_ylim(0.0, float(x_values[-1]))
            ax.set_yticklabels([])
            if row == 0:
                ax.set_title(f"density, zeta={z_values[z_indices[col]]:.3f}")
                image = ax.pcolormesh(theta_grid, radius_grid, density_data[0, :, :, z_indices[col]], shading="auto", cmap="viridis", vmin=-density_vmax, vmax=density_vmax)
            else:
                ax.set_title(f"v_parallel, zeta={z_values[z_indices[col]]:.3f}")
                image = ax.pcolormesh(theta_grid, radius_grid, v_parallel_data[0, :, :, z_indices[col]], shading="auto", cmap="coolwarm", vmin=-v_parallel_vmax, vmax=v_parallel_vmax)
            images.append(image)

    suptitle = fig.suptitle(f"Shifted-torus 2-field MMS fields at resolution {int(resolution)}")

    def update(frame_index: int):
        actual_index = int(frame_indices[frame_index])
        time_value = float(times[actual_index])
        for col in range(4):
            images[col].set_array(density_data[actual_index, :, :, z_indices[col]].ravel())
            images[4 + col].set_array(v_parallel_data[actual_index, :, :, z_indices[col]].ravel())
            axes[0, col].set_title(f"density, zeta={z_values[z_indices[col]]:.3f}, t={time_value:.3f}")
            axes[1, col].set_title(f"v_parallel, zeta={z_values[z_indices[col]]:.3f}, t={time_value:.3f}")
        suptitle.set_text(f"Shifted-torus 2-field MMS fields at resolution {int(resolution)}, t={time_value:.3f}")
        return images

    animator = animation.FuncAnimation(fig, update, frames=int(frame_indices.shape[0]), interval=100, blit=False)
    writer = animation.PillowWriter(fps=10)
    animator.save(output_path, writer=writer)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Shifted-torus two-field local-shard MMS convergence")
    parser.add_argument("--shard-counts", type=int, nargs=3, default=(1, 1, 1))
    args = parser.parse_args()
    shard_counts = tuple(int(value) for value in args.shard_counts)
    resolutions = np.asarray([30, 60,120], dtype=np.int64)
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    max_errors: list[float] = []
    final_resolution_state: Fci2FieldState | None = None
    final_resolution_geometry: FciGeometry3D | None = None
    final_resolution: int | None = None
    final_resolution_times: jnp.ndarray | None = None
    final_resolution_density_history: jnp.ndarray | None = None
    final_resolution_v_parallel_history: jnp.ndarray | None = None

    for resolution in resolutions:
        geometry = build_shifted_torus_2field_geometry((int(resolution), int(resolution), int(resolution)))
        steps = _resolution_step_count(int(resolution))
        dt = float(tf) / float(steps)
        print(f"Starting simulation for resolution={int(resolution)}, steps={steps}, dt={dt:.6e}")
        start = time_module.perf_counter()
        try:
            final_state, times, density_history, v_parallel_history = simulate_mms_2field_shifted_torus(
                geometry,
                final_time=tf,
                timestep=dt,
                rho_star_value=rho_star,
                shard_counts=shard_counts,
            )
            elapsed = time_module.perf_counter() - start
            mean_error, _, max_error = _combined_error_statistics(final_state, geometry, tf)
        except FloatingPointError as exc:
            elapsed = time_module.perf_counter() - start
            print(f"WARNING: res={int(resolution)} failed with non-finite values after {elapsed:.6e} s: {exc}")
            continue

        successful_resolutions.append(int(resolution))
        l2_errors.append(mean_error)
        max_errors.append(max_error)
        print(
            f"res={int(resolution)}: steps={steps}, total_time={elapsed:.6e} s, "
            f"avg_step_time={elapsed / float(steps):.6e} s, "
            f"l2_error={mean_error:.6e}, max_error={max_error:.6e}"
        )

        final_resolution_state = final_state
        final_resolution_geometry = geometry
        final_resolution = int(resolution)
        final_resolution_times = times
        final_resolution_density_history = density_history
        final_resolution_v_parallel_history = v_parallel_history

    if successful_resolutions:
        plotted_resolutions = np.asarray(successful_resolutions, dtype=np.int64)
        log_resolutions = np.log(plotted_resolutions.astype(np.float64))
        l2_log_errors = np.log(np.asarray(l2_errors, dtype=np.float64))
        max_log_errors = np.log(np.asarray(max_errors, dtype=np.float64))
        l2_slope, l2_intercept = np.polyfit(log_resolutions, l2_log_errors, 1)
        max_slope, max_intercept = np.polyfit(log_resolutions, max_log_errors, 1)
        print(f"shifted_torus_2field l2 convergence order: {-l2_slope:.6f}")
        print(f"shifted_torus_2field max convergence order: {-max_slope:.6f}")

        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.loglog(plotted_resolutions, l2_errors, "o-", label=f"l2, order {l2_slope:.2f}")
        ax.loglog(plotted_resolutions, max_errors, "^-", label=f"max, order {max_slope:.2f}")
        ax.loglog(
            plotted_resolutions,
            np.exp(l2_intercept) * plotted_resolutions.astype(np.float64) ** l2_slope,
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
        ax.set_title("Shifted-torus 2-field MMS convergence")
        ax.grid(True, which="both", linestyle=":", alpha=0.45)
        ax.legend()
        fig.tight_layout()
        fig.savefig("shifted_torus_2field_convergence.png", dpi=200)
        plt.close(fig)
    else:
        print("WARNING: no valid resolutions completed, skipping convergence plot.")

    if final_resolution_state is not None and final_resolution_geometry is not None and final_resolution is not None:
        final_exact_state = _shifted_torus_exact_state(final_resolution_geometry, tf)
        _plot_final_slices(
            final_resolution_state,
            final_exact_state,
            final_resolution_geometry,
            final_resolution,
            "shifted_torus_2field_slices.png",
        )

    if (
        final_resolution_times is not None
        and final_resolution_density_history is not None
        and final_resolution_v_parallel_history is not None
        and final_resolution_geometry is not None
        and final_resolution is not None
    ):
        _save_shifted_torus_movie(
            final_resolution_times,
            final_resolution_density_history,
            final_resolution_v_parallel_history,
            final_resolution_geometry,
            final_resolution,
            "shifted_torus_2field_slices.gif",
            frame_stride=5,
        )
