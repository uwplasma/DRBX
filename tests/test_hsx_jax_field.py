"""Parity tests for the compiled continuous HSX field transform."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry.Bfield_evaluator import ComponentSplineBFieldEvaluator
from drbx.geometry.MetricEvaluator import MetricEvaluator
from drbx.geometry.hsx_jax_field import JaxHsxMagneticField
from drbx.geometry.fci_geometry import (
    CellCenteredGrid3D,
    Grid1D,
    build_fci_maps_from_callbacks,
)
from drbx.geometry.fci_trace_executor import FciTraceExecutor


jax.config.update("jax_enable_x64", True)


def _evaluators():
    u = np.linspace(0.0, 1.0, 8)
    theta = 2.0 * np.pi * np.arange(12) / 12.0
    eta = 2.0 * np.pi * np.arange(14) / 14.0
    ug, tg, eg = np.meshgrid(u, theta, eta, indexing="ij")
    angle = tg + 0.08 * np.sin(eg)
    radius = 2.7 + 0.55 * ug * np.cos(angle)
    positions = np.stack(
        (radius * np.cos(eg), radius * np.sin(eg), -0.55 * ug * np.sin(angle)),
        axis=-1,
    )
    metric = MetricEvaluator(
        u,
        theta,
        eta,
        positions,
        period=2.0 * np.pi,
        topology="toroidal",
        radial_degree=6,
        poloidal_modes=3,
        toroidal_modes=2,
    )
    R = np.linspace(1.8, 3.6, 18)
    phi = 2.0 * np.pi * np.arange(20) / 20.0
    Z = np.linspace(-0.8, 0.8, 17)
    pp, zz, rr = np.meshgrid(phi, Z, R, indexing="ij")
    field = np.stack(
        (
            0.03 * np.sin(pp) + 0.01 * zz,
            1.2 + 0.08 * np.cos(pp) + 0.02 * (rr - 2.7),
            0.04 * np.sin(2.0 * pp) - 0.01 * zz,
        ),
        axis=-1,
    )
    bfield = ComponentSplineBFieldEvaluator(
        R, phi, Z, field, method="cubic", extrapolate=False
    )
    return metric, bfield


def test_compiled_field_matches_producer_transform_eager_and_jit():
    metric, bfield = _evaluators()
    runtime = JaxHsxMagneticField.from_evaluators(metric, bfield)
    points = np.asarray(
        [
            [1.0e-6, 0.0, 0.0],
            [0.2, 1.3, 2.1],
            [0.75, 5.8, 2.0 * np.pi - 1.0e-10],
        ]
    )
    expected = metric.evaluate_magnetic_field(points, bfield)
    for result in (runtime(points), jax.jit(lambda state, q: state(q))(runtime, jnp.asarray(points))):
        np.testing.assert_allclose(
            np.asarray(result[0]), expected.B_contravariant, rtol=5.0e-11, atol=5.0e-11
        )
        np.testing.assert_allclose(
            np.asarray(result[1]), expected.magnitude, rtol=5.0e-12, atol=5.0e-12
        )


def test_field_state_can_be_replicated_with_named_sharding():
    metric, bfield = _evaluators()
    runtime = JaxHsxMagneticField.from_evaluators(metric, bfield)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("trace",))
    replicated = jax.device_put(
        runtime,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )
    points = jnp.asarray([[0.3, 0.4, 0.7], [0.6, 2.0, 1.2]])
    result = jax.jit(lambda state, q: state(q))(replicated, points)
    assert result[0].shape == (2, 3)
    assert result[1].shape == (2,)
    assert np.all(np.isfinite(np.asarray(result[0])))


def test_executor_reuses_already_replicated_field_state():
    metric, bfield = _evaluators()
    runtime = JaxHsxMagneticField.from_evaluators(metric, bfield)
    bounds = np.asarray([[0.0, 1.0], [0.0, 2.0 * np.pi], [0.0, 2.0 * np.pi]])
    first = FciTraceExecutor(
        runtime,
        4,
        device_count=1,
        grid_bounds=bounds,
        substeps=1,
        axis_regular_axes=(True, False, False),
        axis_epsilon=1.0e-9,
    )
    first.trace(np.asarray([[0.2, 0.4, 0.7]]), 0.1)
    placed = first.device_evaluator
    second = FciTraceExecutor(
        placed,
        4,
        device_count=1,
        grid_bounds=bounds,
        substeps=2,
        axis_regular_axes=(True, False, False),
        axis_epsilon=1.0e-9,
    )
    assert second.device_evaluator is placed


def test_compiled_batched_tracing_preserves_center_map_lowering():
    metric, bfield = _evaluators()
    runtime = JaxHsxMagneticField.from_evaluators(metric, bfield)
    grid = CellCenteredGrid3D(
        Grid1D(jnp.asarray([0.125, 0.375, 0.625, 0.875]), jnp.linspace(0.0, 1.0, 5)),
        Grid1D(
            jnp.asarray(np.pi * (np.arange(6) + 0.5) / 3.0),
            jnp.linspace(0.0, 2.0 * np.pi, 7),
        ),
        Grid1D(
            jnp.asarray(np.pi * (np.arange(4) + 0.5) / 2.0),
            jnp.linspace(0.0, 2.0 * np.pi, 5),
        ),
    )

    def host_field(points):
        return metric.evaluate_magnetic_field(points, bfield)

    reference = build_fci_maps_from_callbacks(
        grid,
        host_field,
        substeps=3,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
        max_trace_batch_seeds=16,
    )
    bounds = np.asarray(
        [[axis.faces[0], axis.faces[-1]] for axis in (grid.x, grid.y, grid.z)]
    )
    executor = FciTraceExecutor(
        runtime,
        16,
        grid_bounds=bounds,
        substeps=3,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
        axis_epsilon=max(1.0e-12, 1.0e-8 * float(grid.x.widths[0])),
    )
    compiled = build_fci_maps_from_callbacks(
        grid,
        host_field,
        substeps=3,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
        trace_batch=lambda seeds, step: executor.trace(seeds, step).outputs,
        trace_backend_identity="test-jax",
        max_trace_batch_seeds=16,
    )
    assert set(compiled) == set(reference)
    for name in reference:
        if name.endswith("boundary"):
            np.testing.assert_array_equal(compiled[name], reference[name])
        else:
            np.testing.assert_allclose(
                compiled[name], reference[name], rtol=2.0e-9, atol=2.0e-9
            )
