from __future__ import annotations

import math

from jax import grad, jit
from jax import numpy as jnp

from drbx.config.boutinp import parse_bout_input
from drbx.native.mesh import build_structured_mesh
from drbx.native.metrics import build_structured_metrics
from drbx.native import run_config_case
from drbx.native.transport import advance_anomalous_diffusion_history
from drbx.runtime.run_config import RunConfiguration


_DIFFUSION_INPUT = """
nout = 5
timestep = 1000

[mesh]
nx = 10
ny = 10
nz = 1
dx = 0.0075 + 0.005*x
dy = 0.01
dz = 0.01
J = 1

[model]
components = h

[h]
type = evolve_density, evolve_pressure, anomalous_diffusion
AA = 1
charge = 1
anomalous_D = 2
thermal_conduction = false

[Nh]
function = 1 + H(x - 0.25) * H(0.75-x) * exp(-(y-π)^2)
bndry_all = neumann

[Ph]
function = Nh:function
bndry_all = neumann
"""


def test_anomalous_diffusion_history_supports_jit_and_grad() -> None:
    config = parse_bout_input(_DIFFUSION_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    result = run_config_case(
        config,
        case_name="diffusion_short_window",
        parity_mode="short_window",
        compare_variables=("Nh", "Ph"),
    )
    initial_density = jnp.asarray(result.variables["Nh"][0], dtype=jnp.float64)
    initial_pressure = jnp.asarray(result.variables["Ph"][0], dtype=jnp.float64)

    @jit
    def loss_fn(anomalous_D: float) -> jnp.ndarray:
        history = advance_anomalous_diffusion_history(
            initial_density,
            initial_pressure,
            mesh=mesh,
            metrics=metrics,
            anomalous_D=anomalous_D,
            density_boundary="neumann",
            pressure_boundary="neumann",
            timestep=run_config.time.timestep,
            steps=2,
        )
        return jnp.sum(history.density_history[-1])

    value = float(loss_fn(0.02))
    derivative = float(grad(loss_fn)(0.02))

    assert math.isfinite(value)
    assert math.isfinite(derivative)
    assert abs(derivative) > 0.0
