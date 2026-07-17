from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..config.boutinp import parse_bout_input
from ..native.mesh import StructuredMesh, build_structured_mesh
from ..native.metrics import StructuredMetrics, build_structured_metrics
from ..native.transport import advance_anomalous_diffusion_history
from ..runtime.run_config import RunConfiguration


@dataclass(frozen=True)
class DiffusionAutodiffSetup:
    config_text: str
    run_config: RunConfiguration
    mesh: StructuredMesh
    metrics: StructuredMetrics
    timestep: float
    steps: int
    density_boundary: str
    pressure_boundary: str
    background_density: float


@dataclass(frozen=True)
class StrongScalingPoint:
    backend: str
    device_count: int
    elapsed_seconds: float
    speedup: float
    efficiency: float


def build_diffusion_autodiff_setup(
    *,
    nx: int = 128,
    ny: int = 24,
    timestep: float = 5.0,
    steps: int = 6,
    background_density: float = 1.0,
) -> DiffusionAutodiffSetup:
    config_text = f"""
nout = {steps}
timestep = {timestep}

[mesh]
nx = {nx}
ny = {ny}
nz = 1
dx = {1.0 / nx:.12f}
dy = {1.0 / ny:.12f}
dz = 1.0
J = 1
g11 = 1
g_22 = 1
g33 = 1

[solver]
type = native

[model]
components = h

[h]
type = evolve_density, evolve_pressure, anomalous_diffusion
AA = 1
charge = 1
anomalous_D = 1.0
thermal_conduction = false

[Nh]
bndry_all = neumann

[Ph]
bndry_all = neumann
"""
    config = parse_bout_input(config_text)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    return DiffusionAutodiffSetup(
        config_text=config_text,
        run_config=run_config,
        mesh=mesh,
        metrics=metrics,
        timestep=timestep,
        steps=steps,
        density_boundary="neumann",
        pressure_boundary="neumann",
        background_density=background_density,
    )


def theta_to_physical(theta: jnp.ndarray) -> dict[str, jnp.ndarray]:
    theta = jnp.asarray(theta, dtype=jnp.float64)
    diffusivity = 0.05 + jax.nn.softplus(theta[0])
    amplitude = 0.02 + jax.nn.softplus(theta[1])
    center = jax.nn.sigmoid(theta[2])
    width = 0.04 + 0.26 * jax.nn.sigmoid(theta[3])
    return {
        "anomalous_D": diffusivity,
        "amplitude": amplitude,
        "center": center,
        "width": width,
    }


def physical_to_theta(
    *,
    anomalous_D: float,
    amplitude: float,
    center: float,
    width: float,
) -> jnp.ndarray:
    center = float(np.clip(center, 1.0e-6, 1.0 - 1.0e-6))
    width_scaled = (float(width) - 0.04) / 0.26
    width_scaled = float(np.clip(width_scaled, 1.0e-6, 1.0 - 1.0e-6))

    def inverse_softplus(value: float) -> float:
        shifted = max(value - 0.05, 1.0e-6)
        return float(np.log(np.expm1(shifted)))

    def inverse_softplus_amp(value: float) -> float:
        shifted = max(value - 0.02, 1.0e-6)
        return float(np.log(np.expm1(shifted)))

    return jnp.asarray(
        [
            inverse_softplus(anomalous_D),
            inverse_softplus_amp(amplitude),
            np.log(center / (1.0 - center)),
            np.log(width_scaled / (1.0 - width_scaled)),
        ],
        dtype=jnp.float64,
    )


def design_field(setup: DiffusionAutodiffSetup, theta: jnp.ndarray) -> jnp.ndarray:
    physical = theta_to_physical(theta)
    return design_field_from_physical(
        setup,
        anomalous_D=physical["anomalous_D"],
        amplitude=physical["amplitude"],
        center=physical["center"],
        width=physical["width"],
    )


def design_field_from_physical(
    setup: DiffusionAutodiffSetup,
    *,
    anomalous_D: jnp.ndarray,
    amplitude: jnp.ndarray,
    center: jnp.ndarray,
    width: jnp.ndarray,
) -> jnp.ndarray:
    x = jnp.linspace(0.0, 1.0, setup.mesh.nx, dtype=jnp.float64)[:, None, None]
    y = jnp.linspace(0.0, 1.0, setup.mesh.local_ny, dtype=jnp.float64)[None, :, None]
    profile = jnp.exp(-0.5 * ((x - center) / width) ** 2)
    y_modulation = 1.0 + 0.15 * jnp.cos(2.0 * jnp.pi * y)
    return setup.background_density + amplitude * profile * y_modulation


def simulate_density_history(setup: DiffusionAutodiffSetup, theta: jnp.ndarray) -> jnp.ndarray:
    physical = theta_to_physical(theta)
    return simulate_density_history_from_physical(
        setup,
        anomalous_D=physical["anomalous_D"],
        amplitude=physical["amplitude"],
        center=physical["center"],
        width=physical["width"],
    )


def simulate_density_history_from_physical(
    setup: DiffusionAutodiffSetup,
    *,
    anomalous_D: jnp.ndarray,
    amplitude: jnp.ndarray,
    center: jnp.ndarray,
    width: jnp.ndarray,
) -> jnp.ndarray:
    density0 = design_field_from_physical(
        setup,
        anomalous_D=anomalous_D,
        amplitude=amplitude,
        center=center,
        width=width,
    )
    history = advance_anomalous_diffusion_history(
        density0,
        density0,
        mesh=setup.mesh,
        metrics=setup.metrics,
        anomalous_D=anomalous_D,
        density_boundary=setup.density_boundary,
        pressure_boundary=setup.pressure_boundary,
        timestep=setup.timestep,
        steps=setup.steps,
    )
    return history.density_history


def active_density_slice(setup: DiffusionAutodiffSetup, field: jnp.ndarray) -> jnp.ndarray:
    return field[setup.mesh.xstart : setup.mesh.xend + 1, setup.mesh.ystart : setup.mesh.yend + 1, 0]


def objective_for_parameter_vector(
    theta: jnp.ndarray,
    setup: DiffusionAutodiffSetup,
    *,
    target_final_density: jnp.ndarray | None = None,
    objective_kind: str = "target_misfit",
) -> jnp.ndarray:
    history = simulate_density_history(setup, theta)
    final_density = active_density_slice(setup, history[-1])
    if objective_kind == "variance":
        centered = final_density - jnp.mean(final_density)
        return jnp.mean(centered * centered)
    if target_final_density is None:
        raise ValueError("target_final_density is required for target_misfit objectives")
    target = jnp.asarray(target_final_density, dtype=jnp.float64)
    mismatch = final_density - target
    return 0.5 * jnp.mean(mismatch * mismatch)


def objective_for_physical_parameters(
    parameters: jnp.ndarray,
    setup: DiffusionAutodiffSetup,
    *,
    target_final_density: jnp.ndarray | None = None,
    objective_kind: str = "target_misfit",
) -> jnp.ndarray:
    parameters = jnp.asarray(parameters, dtype=jnp.float64)
    history = simulate_density_history_from_physical(
        setup,
        anomalous_D=parameters[0],
        amplitude=parameters[1],
        center=parameters[2],
        width=parameters[3],
    )
    final_density = active_density_slice(setup, history[-1])
    if objective_kind == "variance":
        centered = final_density - jnp.mean(final_density)
        return jnp.mean(centered * centered)
    if target_final_density is None:
        raise ValueError("target_final_density is required for target_misfit objectives")
    target = jnp.asarray(target_final_density, dtype=jnp.float64)
    mismatch = final_density - target
    return 0.5 * jnp.mean(mismatch * mismatch)


def finite_difference_gradient(
    objective_fn,
    theta: jnp.ndarray,
    *,
    epsilon: float = 1.0e-4,
) -> jnp.ndarray:
    theta = jnp.asarray(theta, dtype=jnp.float64)
    basis = jnp.eye(theta.size, dtype=jnp.float64)
    values = []
    for direction in basis:
        forward = objective_fn(theta + epsilon * direction)
        backward = objective_fn(theta - epsilon * direction)
        values.append((forward - backward) / (2.0 * epsilon))
    return jnp.asarray(values, dtype=jnp.float64)


def optimize_inverse_design(
    objective_fn,
    theta0: jnp.ndarray,
    *,
    iterations: int = 80,
    learning_rate: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
) -> dict[str, Any]:
    theta = jnp.asarray(theta0, dtype=jnp.float64)
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    loss_history: list[float] = []
    theta_history: list[np.ndarray] = [np.asarray(theta)]
    value_and_grad_fn = jax.jit(jax.value_and_grad(objective_fn))
    objective_only_fn = jax.jit(objective_fn)

    for iteration in range(1, iterations + 1):
        loss_value, gradient = value_and_grad_fn(theta)
        m = beta1 * m + (1.0 - beta1) * gradient
        v = beta2 * v + (1.0 - beta2) * (gradient * gradient)
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        theta = theta - learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        loss_history.append(float(loss_value))
        theta_history.append(np.asarray(theta))

    final_loss = float(objective_only_fn(theta))
    return {
        "theta": theta,
        "loss_history": np.asarray(loss_history, dtype=np.float64),
        "theta_history": np.asarray(theta_history, dtype=np.float64),
        "final_loss": final_loss,
    }


def compute_strong_scaling_points(points: list[tuple[int, float]], *, backend: str) -> list[StrongScalingPoint]:
    if not points:
        return []
    ordered = sorted(points, key=lambda item: item[0])
    baseline_devices, baseline_time = ordered[0]
    result: list[StrongScalingPoint] = []
    for device_count, elapsed_seconds in ordered:
        speedup = baseline_time / elapsed_seconds
        efficiency = speedup / float(device_count / baseline_devices)
        result.append(
            StrongScalingPoint(
                backend=backend,
                device_count=device_count,
                elapsed_seconds=elapsed_seconds,
                speedup=speedup,
                efficiency=efficiency,
            )
        )
    return result
