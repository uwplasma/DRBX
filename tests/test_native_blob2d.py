from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import grad, jit

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.blob2d import (
    build_blob2d_benchmark,
    build_blob2d_potential_operator,
    compute_blob2d_rhs,
    initialize_blob2d_state,
)
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.runtime.run_config import RunConfiguration


_BLOB_INPUT = """
nout = 1
timestep = 2

[mesh]
nx = 8
ny = 1
nz = 16
Lx = 0.02
Lz = 0.04
dx = 0.001
dz = Lz / nz
J = 1
g11 = 1
g33 = 1
bxcvz = 0.1

[model]
components = (e, vorticity, sheath_closure)

[e]
type = evolve_density, fixed_temperature
temperature = 50
bndry_flux = true

[Ne]
function = 1 + 0.1 * exp(-((x - 0.5)^2 + (z/(2*pi) - 0.5)^2) / 0.05)

[vorticity]
average_atomic_mass = 2
bndry_flux = false

[sheath_closure]
connection_length = 10
"""


def _build_blob_case():
    config = parse_bout_input(_BLOB_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    benchmark = build_blob2d_benchmark(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    state = initialize_blob2d_state(config, mesh=mesh)
    operator = build_blob2d_potential_operator(mesh=mesh, metrics=metrics, average_atomic_mass=2.0)
    return mesh, benchmark, state, operator


def test_blob2d_rhs_supports_jit_and_grad() -> None:
    mesh, benchmark, state, operator = _build_blob_case()
    base_density = jnp.asarray(state.electron_density, dtype=jnp.float64)
    base_vorticity = jnp.asarray(state.vorticity, dtype=jnp.float64)

    @jit
    def loss_fn(scale: jnp.ndarray) -> jnp.ndarray:
        rhs = compute_blob2d_rhs(
            type(state)(
                electron_density=scale * base_density,
                vorticity=base_vorticity,
            ),
            mesh=mesh,
            benchmark=benchmark,
            operator=operator,
        )
        return jnp.sum(rhs.potential * rhs.potential) + jnp.sum(rhs.vorticity_rhs * rhs.vorticity_rhs)

    value = loss_fn(jnp.array(1.0, dtype=jnp.float64))
    derivative = grad(loss_fn)(jnp.array(1.0, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))
