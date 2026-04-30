from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import (
    advance_recycling_1d_backward_euler_step,
    _build_recycling_runtime_model,
    _build_recycling_state_fields,
    _compute_recycling_1d_packed_rhs,
)
from jax_drb.native.recycling_fixed_residual import (
    build_fixed_backward_euler_residual,
    build_fixed_host_rhs_bridge,
    fixed_state_from_fields,
    fixed_state_to_feedback_integrals,
    fixed_state_to_full_fields,
    pack_fixed_state,
    unpack_fixed_state,
)
from jax_drb.native.recycling_layout import (
    build_recycling_packed_state_layout,
    pack_recycling_active_state,
)
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.runtime.run_config import RunConfiguration


_DTHE_INPUT = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling-dthe/data/BOUT.inp")
_HYDROGEN_INPUT = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")


def _dthe_context():
    if not _DTHE_INPUT.exists():
        pytest.skip("Hermès DTHE recycling reference deck is not available.")
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    fields = _build_recycling_state_fields(runtime_model)
    feedback_integrals = {name: 0.0 for name in runtime_model.feedback_names}
    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=runtime_model.field_names,
        feedback_names=runtime_model.feedback_names,
        mesh=mesh,
    )
    return config, mesh, metrics, scalars, runtime_model, fields, feedback_integrals, layout


def test_fixed_state_round_trips_actual_dthe_recycling_deck() -> None:
    _, mesh, _, _, runtime_model, fields, feedback_integrals, layout = _dthe_context()

    fixed_state = fixed_state_from_fields(fields, feedback_integrals=feedback_integrals, layout=layout)
    packed_fixed = np.asarray(pack_fixed_state(fixed_state), dtype=np.float64)
    packed_legacy = pack_recycling_active_state(
        fields,
        feedback_integrals=feedback_integrals,
        field_names=runtime_model.field_names,
        feedback_names=runtime_model.feedback_names,
        mesh=mesh,
        layout=layout,
    )
    restored_fields = fixed_state_to_full_fields(fixed_state, layout=layout)
    restored_integrals = fixed_state_to_feedback_integrals(
        fixed_state,
        layout=layout,
        base_feedback_integrals=feedback_integrals,
    )

    np.testing.assert_allclose(packed_fixed, packed_legacy, rtol=0.0, atol=0.0)
    for name in runtime_model.field_names:
        np.testing.assert_allclose(np.asarray(restored_fields[name]), fields[name], rtol=0.0, atol=0.0)
    assert set(restored_integrals) == set(feedback_integrals)


def test_unpack_fixed_state_preserves_host_arrays_for_scipy_bridge() -> None:
    _, mesh, _, _, runtime_model, fields, feedback_integrals, layout = _dthe_context()
    packed = pack_recycling_active_state(
        fields,
        feedback_integrals=feedback_integrals,
        field_names=runtime_model.field_names,
        feedback_names=runtime_model.feedback_names,
        mesh=mesh,
        layout=layout,
    )

    fixed_state = unpack_fixed_state(np.asarray(packed, dtype=np.float64), layout=layout)

    assert all(isinstance(value, np.ndarray) for value in fixed_state.field_values)
    assert isinstance(fixed_state.feedback_values, np.ndarray)


def test_fixed_host_rhs_bridge_matches_dthe_packed_rhs_oracle() -> None:
    config, mesh, metrics, scalars, runtime_model, fields, feedback_integrals, layout = _dthe_context()
    fixed_state = fixed_state_from_fields(fields, feedback_integrals=feedback_integrals, layout=layout)

    def packed_rhs(state_fields: dict[str, object], state_integrals: dict[str, object]):
        return _compute_recycling_1d_packed_rhs(
            config,
            state_fields,
            runtime_model=runtime_model,
            sanitize_fields=True,
            feedback_integrals=state_integrals,
            field_names=runtime_model.field_names,
            feedback_names=runtime_model.feedback_names,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=scalars,
            layout=layout,
        )

    direct_rhs = packed_rhs(fields, feedback_integrals)
    bridge = build_fixed_host_rhs_bridge(
        packed_rhs,
        layout=layout,
        base_feedback_integrals=feedback_integrals,
    )
    bridged_rhs = np.asarray(pack_fixed_state(bridge(fixed_state)), dtype=np.float64)

    np.testing.assert_allclose(bridged_rhs, direct_rhs, rtol=1.0e-12, atol=1.0e-12)

    timestep = 1.0e-6
    previous = np.asarray(pack_fixed_state(fixed_state), dtype=np.float64)
    residual = build_fixed_backward_euler_residual(
        bridge,
        layout=layout,
        previous_packed_state=previous,
        timestep=timestep,
    )

    np.testing.assert_allclose(
        np.asarray(residual(previous), dtype=np.float64),
        -timestep * np.asarray(direct_rhs, dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_jax_linearized_recycling_step_reaches_full_fixed_residual_without_host_barrier() -> None:
    pytest.importorskip("jax")
    if not _HYDROGEN_INPUT.exists():
        pytest.skip("Hermès hydrogen recycling reference deck is not available.")
    config = load_bout_input(_HYDROGEN_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    fields = _build_recycling_state_fields(runtime_model)
    feedback_integrals = {name: 0.0 for name in runtime_model.feedback_names}

    _, _, info = advance_recycling_1d_backward_euler_step(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=1.0e-6,
        solver_mode="jax_linearized",
        residual_tolerance=1.0e-6,
        max_nonlinear_iterations=1,
    )

    assert info.residual_inf_norm < 1.0e-8
    assert info.diagnostics["jacobian_refresh_count"] == 1
    assert info.diagnostics["jacobian_assembly_seconds"] >= 0.0
    assert info.diagnostics["jacobian_mode"] == "jax_linearized:jax_gmres"


def test_jax_linearized_recycling_step_supports_dthe_fixed_residual() -> None:
    pytest.importorskip("jax")
    config, mesh, metrics, scalars, runtime_model, fields, feedback_integrals, _ = _dthe_context()

    _, _, info = advance_recycling_1d_backward_euler_step(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=1.0e-6,
        solver_mode="jax_linearized",
        residual_tolerance=1.0e-6,
        max_nonlinear_iterations=1,
    )

    assert info.residual_inf_norm < 1.0e-8
    assert info.diagnostics["jacobian_refresh_count"] == 1
    assert info.diagnostics["jacobian_mode"] == "jax_linearized:jax_gmres"
