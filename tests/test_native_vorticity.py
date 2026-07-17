from __future__ import annotations

import numpy as np
from jax import grad, jit

from drbx.config.boutinp import parse_bout_input
from drbx.native.expression import ArrayExpressionEvaluator
from drbx.native.mesh import broadcast_to_field_shape, build_structured_mesh
from drbx.native.metrics import build_structured_metrics
from drbx.native.vorticity import (
    apply_vorticity_boundaries,
    build_vorticity_operator,
    compute_vorticity_rhs,
    solve_potential,
)
from drbx.runtime.run_config import RunConfiguration


_VORTICITY_INPUT = """
nout = 10
timestep = 20
MYG = 0

[mesh]
nx = 10
ny = 1
nz = 10
zn = z / (2π)
J = 1

[mesh:paralleltransform]
type = identity

[model]
components = vorticity

[vorticity]
diamagnetic = false
diamagnetic_polarisation = false
average_atomic_mass = 2
bndry_flux = false
poloidal_flows = false
split_n0 = false
phi_dissipation = false

[Vort]
function = exp(-((x-0.5)^2 + (mesh:zn - 0.5)^2)/(0.2^2))
"""


def _initial_vorticity():
    config = parse_bout_input(_VORTICITY_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    field = broadcast_to_field_shape(evaluator.resolve_option("Vort", "function"), mesh)
    field = apply_vorticity_boundaries(field, mesh)
    metrics = build_structured_metrics(config, run_config, mesh)
    operator = build_vorticity_operator(mesh=mesh, metrics=metrics, average_atomic_mass=2.0)
    return mesh, metrics, operator, np.asarray(field, dtype=np.float64)


def test_solve_potential_matches_expected_boundary_pattern() -> None:
    mesh, metrics, operator, vorticity = _initial_vorticity()
    potential = np.asarray(solve_potential(vorticity, mesh=mesh, operator=operator), dtype=np.float64)

    np.testing.assert_allclose(potential[0, 0, :], potential[1, 0, :], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(potential[1, 0, :], -potential[2, 0, :], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(potential[8, 0, :], -potential[7, 0, :], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(potential[9, 0, :], potential[8, 0, :], rtol=1e-12, atol=1e-12)


def test_vorticity_rhs_matches_reference_regression_samples() -> None:
    mesh, metrics, operator, vorticity = _initial_vorticity()
    rhs = compute_vorticity_rhs(vorticity, mesh=mesh, metrics=metrics, operator=operator)

    expected_row4 = np.array(
        [
            4.06575815e-20,
            -2.23030000e-03,
            -1.34739415e-02,
            -3.73836755e-02,
            -5.25641459e-02,
            -9.08698274e-03,
            2.78164677e-02,
            3.15887512e-02,
            1.16374073e-02,
            6.83401216e-04,
        ]
    )
    expected_row5 = np.array(
        [
            0.0,
            6.83401216e-04,
            1.16374073e-02,
            3.15887512e-02,
            2.78164677e-02,
            -9.08698274e-03,
            -5.25641459e-02,
            -3.73836755e-02,
            -1.34739415e-02,
            -2.23030000e-03,
        ]
    )

    np.testing.assert_allclose(np.asarray(rhs.vorticity[4, 0, :], dtype=np.float64), expected_row4, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rhs.vorticity[5, 0, :], dtype=np.float64), expected_row5, rtol=1e-8, atol=1e-12)


def test_vorticity_potential_supports_jit_and_grad() -> None:
    mesh, metrics, operator, vorticity = _initial_vorticity()

    @jit
    def loss_fn(scale):
        potential = solve_potential(scale * vorticity, mesh=mesh, operator=operator)
        return (potential * potential).sum()

    value = loss_fn(1.0)
    derivative = grad(loss_fn)(1.0)

    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))
