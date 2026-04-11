from __future__ import annotations

import numpy as np
from jax import grad
import jax.numpy as jnp

from jax_drb.validation.autodiff_diffusion import (
    build_diffusion_autodiff_setup,
    compute_strong_scaling_points,
    finite_difference_gradient,
    objective_for_physical_parameters,
    optimize_inverse_design,
    simulate_density_history_from_physical,
)


def test_autodiff_diffusion_gradient_matches_finite_difference() -> None:
    setup = build_diffusion_autodiff_setup(nx=48, ny=10, timestep=2.0, steps=3)
    target_history = simulate_density_history_from_physical(
        setup,
        anomalous_D=0.42,
        amplitude=0.18,
        center=0.55,
        width=0.11,
    )
    target_final = target_history[-1][setup.mesh.xstart : setup.mesh.xend + 1, setup.mesh.ystart : setup.mesh.yend + 1, 0]
    objective = lambda parameters: objective_for_physical_parameters(parameters, setup, target_final_density=target_final)
    nominal = np.asarray([0.31, 0.14, 0.46, 0.16], dtype=np.float64)

    autodiff = np.asarray(grad(lambda parameters: objective(jnp.asarray(parameters, dtype=jnp.float64)))(jnp.asarray(nominal)), dtype=np.float64)
    reference = np.asarray(finite_difference_gradient(objective, nominal, epsilon=5.0e-4), dtype=np.float64)

    np.testing.assert_allclose(autodiff, reference, rtol=2.0e-2, atol=5.0e-4)


def test_autodiff_inverse_design_reduces_objective() -> None:
    setup = build_diffusion_autodiff_setup(nx=48, ny=10, timestep=2.0, steps=3)
    target_history = simulate_density_history_from_physical(
        setup,
        anomalous_D=0.45,
        amplitude=0.20,
        center=0.58,
        width=0.10,
    )
    target_final = target_history[-1][setup.mesh.xstart : setup.mesh.xend + 1, setup.mesh.ystart : setup.mesh.yend + 1, 0]
    objective = lambda parameters: objective_for_physical_parameters(parameters, setup, target_final_density=target_final, objective_kind="target_misfit")

    initial = np.asarray([0.24, 0.08, 0.36, 0.18], dtype=np.float64)
    result = optimize_inverse_design(objective, initial, iterations=12, learning_rate=0.04)

    assert result["loss_history"][0] > result["final_loss"]


def test_compute_strong_scaling_points_reports_speedup_and_efficiency() -> None:
    points = compute_strong_scaling_points([(1, 8.0), (2, 4.5), (4, 2.6)], backend="cpu")

    assert [point.device_count for point in points] == [1, 2, 4]
    assert points[0].speedup == 1.0
    assert points[0].efficiency == 1.0
    assert points[-1].speedup > 3.0
    assert 0.7 < points[-1].efficiency < 1.0
