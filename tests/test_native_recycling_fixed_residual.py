from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import (
    advance_recycling_1d_backward_euler_step,
    build_recycling_1d_bdf2_residual_context,
    build_recycling_1d_backward_euler_residual_context,
    _build_recycling_runtime_model,
    _build_recycling_state_fields,
    _compute_recycling_1d_packed_rhs,
)
from jax_drb.native.recycling_fixed_residual import (
    RecyclingFixedState,
    build_fixed_array_rhs,
    build_fixed_array_state_rhs,
    build_fixed_backward_euler_residual,
    build_fixed_residual_linearized_action,
    build_fixed_host_rhs_bridge,
    fixed_residual_jvp_batch_action,
    fixed_state_from_fields,
    fixed_state_to_feedback_integrals,
    fixed_state_to_full_fields,
    pack_fixed_state,
    solve_fixed_residual_linearized_action_update,
    solve_fixed_residual_linearized_update,
    unpack_fixed_state,
)
import jax_drb.native.recycling_fixed_residual as fixed_residual_mod
from jax_drb.native.recycling_layout import (
    build_recycling_packed_state_layout,
    pack_recycling_active_state,
)
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.runtime.run_config import RunConfiguration


_DTHE_INPUT = Path(
    "/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling-dthe/data/BOUT.inp"
)
_DTHE_FIXTURE_INPUT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "reference-root"
    / "tests"
    / "integrated"
    / "1D-recycling-dthe"
    / "data"
    / "BOUT.inp"
)
_HYDROGEN_INPUT = Path(
    "/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp"
)


def _build_dthe_context(input_path: Path):
    config = load_bout_input(input_path)
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
    return (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        fields,
        feedback_integrals,
        layout,
    )


def _dthe_context():
    if not _DTHE_INPUT.exists():
        pytest.skip("Hermès DTHE recycling reference deck is not available.")
    return _build_dthe_context(_DTHE_INPUT)


def _dthe_fixture_context():
    if not _DTHE_FIXTURE_INPUT.exists():
        raise AssertionError("committed D/T/He recycling fixture deck is missing")
    return _build_dthe_context(_DTHE_FIXTURE_INPUT)


def test_fixed_state_round_trips_actual_dthe_recycling_deck() -> None:
    _, mesh, _, _, runtime_model, fields, feedback_integrals, layout = _dthe_context()

    fixed_state = fixed_state_from_fields(
        fields, feedback_integrals=feedback_integrals, layout=layout
    )
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
        np.testing.assert_allclose(
            np.asarray(restored_fields[name]), fields[name], rtol=0.0, atol=0.0
        )
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

    fixed_state = unpack_fixed_state(
        np.asarray(packed, dtype=np.float64), layout=layout
    )

    assert all(isinstance(value, np.ndarray) for value in fixed_state.field_values)
    assert isinstance(fixed_state.feedback_values, np.ndarray)


def test_fixed_array_rhs_only_allocates_zero_defaults_for_missing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    layout = type(
        "Layout",
        (),
        {
            "field_names": ("A", "B"),
            "feedback_names": (),
            "active_shape": (1,),
        },
    )()
    state = RecyclingFixedState(
        field_values=(
            jnp.asarray([1.0], dtype=jnp.float64),
            jnp.asarray([2.0], dtype=jnp.float64),
        ),
        feedback_values=jnp.asarray([], dtype=jnp.float64),
    )
    original_zeros_like = fixed_residual_mod.jnp.zeros_like
    zero_shapes: list[tuple[int, ...]] = []

    def tracking_zeros_like(value, *args, **kwargs):
        zero_shapes.append(tuple(value.shape))
        return original_zeros_like(value, *args, **kwargs)

    monkeypatch.setattr(fixed_residual_mod.jnp, "zeros_like", tracking_zeros_like)
    rhs = build_fixed_array_rhs(
        lambda fields, _feedback: {"A": fields["A"] + 1.0},
        layout=layout,
        feedback_rhs_function=lambda _fields, feedback: feedback,
    )

    result = rhs(state)

    np.testing.assert_allclose(np.asarray(result.field_values[0]), np.asarray([2.0]))
    np.testing.assert_allclose(np.asarray(result.field_values[1]), np.asarray([0.0]))
    assert zero_shapes == [(1,)]


def test_fixed_array_state_rhs_evaluates_shared_kernel_once() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    layout = type(
        "Layout",
        (),
        {
            "field_names": ("A", "B"),
            "feedback_names": ("controller",),
            "active_shape": (2,),
        },
    )()
    state = RecyclingFixedState(
        field_values=(
            jnp.asarray([1.0, 2.0], dtype=jnp.float64),
            jnp.asarray([3.0, 4.0], dtype=jnp.float64),
        ),
        feedback_values=jnp.asarray([0.5], dtype=jnp.float64),
    )
    call_count = 0

    def coupled_rhs(
        active_fields: dict[str, object], feedback_values: object
    ) -> RecyclingFixedState:
        nonlocal call_count
        call_count += 1
        controller = jnp.asarray(feedback_values)[0]
        density_rhs = jnp.asarray(active_fields["A"]) + controller
        pressure_rhs = 2.0 * jnp.asarray(active_fields["B"])
        feedback_rhs = jnp.asarray([jnp.mean(density_rhs + pressure_rhs)])
        return RecyclingFixedState(
            field_values=(density_rhs, pressure_rhs),
            feedback_values=feedback_rhs,
        )

    rhs = build_fixed_array_state_rhs(coupled_rhs, layout=layout)

    result = rhs(state)

    assert call_count == 1
    np.testing.assert_allclose(np.asarray(result.field_values[0]), [1.5, 2.5])
    np.testing.assert_allclose(np.asarray(result.field_values[1]), [6.0, 8.0])
    np.testing.assert_allclose(np.asarray(result.feedback_values), [9.0])


def test_unpack_fixed_state_rejects_static_layout_contract_mismatches() -> None:
    layout = type(
        "Layout",
        (),
        {
            "field_names": ("A",),
            "feedback_names": ("controller",),
            "active_shape": (2,),
            "field_size": 2,
            "active_slices": (slice(None),),
            "field_templates": (np.zeros(2, dtype=np.float64),),
        },
    )()

    with pytest.raises(ValueError, match="one-dimensional"):
        unpack_fixed_state(np.ones((3, 1), dtype=np.float64), layout=layout)
    with pytest.raises(ValueError, match="has size 2, expected 3"):
        unpack_fixed_state(np.ones(2, dtype=np.float64), layout=layout)

    bad_layout = type(
        "Layout",
        (),
        {
            "field_names": ("A",),
            "feedback_names": (),
            "active_shape": (2,),
            "field_size": 3,
            "active_slices": (slice(None),),
            "field_templates": (np.zeros(2, dtype=np.float64),),
        },
    )()
    with pytest.raises(ValueError, match="field_size does not match"):
        unpack_fixed_state(np.ones(2, dtype=np.float64), layout=bad_layout)


def test_fixed_array_rhs_rejects_static_shape_and_key_mismatches() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    layout = type(
        "Layout",
        (),
        {
            "field_names": ("A", "B"),
            "feedback_names": ("controller",),
            "active_shape": (2,),
        },
    )()
    state = RecyclingFixedState(
        field_values=(
            jnp.asarray([1.0, 2.0], dtype=jnp.float64),
            jnp.asarray([3.0, 4.0], dtype=jnp.float64),
        ),
        feedback_values=jnp.asarray([0.5], dtype=jnp.float64),
    )

    unknown_key_rhs = build_fixed_array_rhs(
        lambda fields, _feedback: {"typo": fields["A"]},
        layout=layout,
        feedback_rhs_function=lambda _fields, feedback: feedback,
    )
    with pytest.raises(ValueError, match="unknown layout entries: 'typo'"):
        unknown_key_rhs(state)

    bad_field_shape_rhs = build_fixed_array_rhs(
        lambda _fields, _feedback: {"A": jnp.ones((2, 1), dtype=jnp.float64)},
        layout=layout,
        feedback_rhs_function=lambda _fields, feedback: feedback,
    )
    with pytest.raises(ValueError, match=r"Field RHS for 'A' has shape \(2, 1\)"):
        bad_field_shape_rhs(state)

    bad_feedback_shape_rhs = build_fixed_array_rhs(
        lambda fields, _feedback: {"A": fields["A"]},
        layout=layout,
        feedback_rhs_function=lambda _fields, _feedback: jnp.ones(
            (1, 1), dtype=jnp.float64
        ),
    )
    with pytest.raises(ValueError, match=r"Feedback RHS has shape \(1, 1\)"):
        bad_feedback_shape_rhs(state)

    bad_state = RecyclingFixedState(
        field_values=(state.field_values[0],),
        feedback_values=state.feedback_values,
    )
    valid_rhs = build_fixed_array_rhs(
        lambda fields, _feedback: {"A": fields["A"]},
        layout=layout,
        feedback_rhs_function=lambda _fields, feedback: feedback,
    )
    with pytest.raises(ValueError, match="field count does not match"):
        valid_rhs(bad_state)


def test_fixed_array_state_rhs_rejects_shape_and_type_contract_mismatches() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    layout = type(
        "Layout",
        (),
        {
            "field_names": ("density",),
            "feedback_names": ("controller",),
            "active_shape": (2,),
        },
    )()
    good_state = RecyclingFixedState(
        field_values=(jnp.asarray([1.0, 2.0], dtype=jnp.float64),),
        feedback_values=jnp.asarray([0.5], dtype=jnp.float64),
    )
    passthrough_rhs = build_fixed_array_state_rhs(
        lambda fields, feedback: RecyclingFixedState(
            field_values=(fields["density"],),
            feedback_values=feedback,
        ),
        layout=layout,
    )

    bad_field_shape_state = RecyclingFixedState(
        field_values=([1.0],),
        feedback_values=jnp.asarray([0.5], dtype=jnp.float64),
    )
    with pytest.raises(
        ValueError, match="Fixed state field 'density' has shape"
    ):
        passthrough_rhs(bad_field_shape_state)

    bad_feedback_shape_state = RecyclingFixedState(
        field_values=good_state.field_values,
        feedback_values=[0.5, 0.6],
    )
    with pytest.raises(ValueError, match="feedback_values has shape"):
        passthrough_rhs(bad_feedback_shape_state)

    wrong_type_rhs = build_fixed_array_state_rhs(
        lambda fields, _feedback: {"density": fields["density"]},
        layout=layout,
    )
    with pytest.raises(TypeError, match="State RHS must return"):
        wrong_type_rhs(good_state)


def test_fixed_residual_batched_jvp_rejects_bad_tangent_shapes() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    residual = lambda state: 2.0 * jnp.asarray(state, dtype=jnp.float64)
    packed_state = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)

    with pytest.raises(ValueError, match="exactly one leading batch axis"):
        fixed_residual_jvp_batch_action(
            residual,
            packed_state,
            jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64),
        )
    with pytest.raises(ValueError, match="Batched residual tangent entries"):
        fixed_residual_jvp_batch_action(
            residual,
            packed_state,
            jnp.ones((2, 2), dtype=jnp.float64),
        )


def test_instrumented_fixed_residual_linearized_action_tracks_dispatches() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    residual_call_count = 0

    def residual(state):
        nonlocal residual_call_count
        residual_call_count += 1
        state_array = jnp.asarray(state, dtype=jnp.float64)
        return jnp.asarray(
            (
                state_array[0] + 2.0 * state_array[1],
                state_array[1] ** 2 + state_array[2],
                jnp.sin(state_array[0]) + 0.5 * state_array[2],
            ),
            dtype=jnp.float64,
        )

    state = jnp.asarray([0.25, 0.5, -0.75], dtype=jnp.float64)
    direction = jnp.asarray([1.0, -0.25, 0.125], dtype=jnp.float64)
    tangent_batch = jnp.stack((direction, -2.0 * direction))

    action = build_fixed_residual_linearized_action(residual, state)
    dense_jacobian = jax.jacfwd(residual)(state)

    np.testing.assert_allclose(
        np.asarray(action.residual_value),
        np.asarray(residual(state)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(action.apply(direction)),
        np.asarray(dense_jacobian @ direction),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(action.apply_batch(tangent_batch)),
        np.asarray(jax.vmap(lambda tangent: dense_jacobian @ tangent)(tangent_batch)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    diagnostics = action.diagnostics()
    assert diagnostics["state_shape"] == (3,)
    assert diagnostics["call_count"] == 1
    assert diagnostics["batched_call_count"] == 1
    assert diagnostics["dispatch_seconds"] >= 0.0
    assert diagnostics["batched_dispatch_seconds"] >= 0.0
    assert residual_call_count >= 1


def test_instrumented_fixed_residual_linearized_action_rejects_bad_shapes() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    residual = lambda state: 3.0 * jnp.asarray(state, dtype=jnp.float64)
    action = build_fixed_residual_linearized_action(
        residual,
        jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64),
    )

    with pytest.raises(ValueError, match="Residual tangent has shape"):
        action.apply(jnp.ones((2,), dtype=jnp.float64))
    with pytest.raises(ValueError, match="exactly one leading batch axis"):
        action.apply_batch(jnp.ones((3,), dtype=jnp.float64))
    with pytest.raises(ValueError, match="Batched residual tangent entries"):
        action.apply_batch(jnp.ones((2, 2), dtype=jnp.float64))


def test_fixed_residual_linearized_update_solves_exact_linear_newton_step() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    matrix = jnp.asarray(
        (
            (4.0, 0.5, -0.25),
            (0.0, 3.0, 0.5),
            (0.25, -0.75, 2.5),
        ),
        dtype=jnp.float64,
    )
    exact_root = jnp.asarray([0.2, -0.4, 0.6], dtype=jnp.float64)

    def residual(state):
        return matrix @ (jnp.asarray(state, dtype=jnp.float64) - exact_root)

    candidate = jnp.asarray([1.0, 0.5, -0.25], dtype=jnp.float64)
    result = solve_fixed_residual_linearized_update(
        residual,
        candidate,
        linear_tolerance=1.0e-12,
        linear_restart=3,
        linear_maxiter=4,
        solve_method="batched",
    )

    np.testing.assert_allclose(
        np.asarray(candidate + result.update),
        np.asarray(exact_root),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert result.linear_update_residual_inf_norm < 1.0e-10
    assert result.linear_update_relative_residual < 1.0e-10
    assert result.diagnostics["call_count"] >= 1
    assert result.diagnostics["solve_call_count"] >= 1
    assert result.diagnostics["linear_operator_jitted"] is False
    assert result.diagnostics["linearization_reused"] is False
    assert result.diagnostics["solve_method"] == "batched"


def test_fixed_residual_linearized_update_supports_jitted_operator() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    diagonal = jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float64)
    residual = lambda state: diagonal * (jnp.asarray(state, dtype=jnp.float64) - 1.0)
    candidate = jnp.asarray([2.0, -1.0, 0.0], dtype=jnp.float64)

    result = solve_fixed_residual_linearized_update(
        residual,
        candidate,
        linear_tolerance=1.0e-12,
        linear_restart=3,
        linear_maxiter=4,
        jit_linear_operator=True,
    )

    np.testing.assert_allclose(
        np.asarray(candidate + result.update),
        np.ones(3, dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert result.linear_update_relative_residual < 1.0e-10
    assert result.diagnostics["linear_operator_jitted"] is True
    assert result.diagnostics["linearization_reused"] is False
    assert result.diagnostics["call_count"] == 0
    assert result.diagnostics["solve_call_count"] == 0


def test_fixed_residual_linearized_action_update_reuses_existing_action() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    diagonal = jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float64)
    residual = lambda state: diagonal * (jnp.asarray(state, dtype=jnp.float64) - 1.0)
    candidate = jnp.asarray([2.0, -1.0, 0.0], dtype=jnp.float64)
    action = build_fixed_residual_linearized_action(residual, candidate)
    action.apply(jnp.ones(3, dtype=jnp.float64)).block_until_ready()

    result = solve_fixed_residual_linearized_action_update(
        action,
        linear_tolerance=1.0e-12,
        linear_restart=3,
        linear_maxiter=4,
        jit_linear_operator=True,
    )

    np.testing.assert_allclose(
        np.asarray(candidate + result.update),
        np.ones(3, dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert result.linear_update_relative_residual < 1.0e-10
    assert result.diagnostics["linearization_reused"] is True
    assert result.diagnostics["linear_operator_jitted"] is True
    assert result.diagnostics["call_count"] == 1
    assert result.diagnostics["solve_call_count"] == 0
    assert result.diagnostics["solve_batched_call_count"] == 0


def test_fixed_residual_linearized_update_can_skip_update_residual_diagnostic() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    diagonal = jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float64)
    residual = lambda state: diagonal * (jnp.asarray(state, dtype=jnp.float64) - 1.0)
    candidate = jnp.asarray([2.0, -1.0, 0.0], dtype=jnp.float64)

    result = solve_fixed_residual_linearized_update(
        residual,
        candidate,
        linear_tolerance=1.0e-12,
        linear_restart=3,
        linear_maxiter=4,
        diagnose_update_residual=False,
    )

    np.testing.assert_allclose(
        np.asarray(candidate + result.update),
        np.ones(3, dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert result.linear_update_residual_inf_norm is None
    assert result.linear_update_relative_residual is None
    assert result.diagnostics["linear_update_residual_checked"] is False


def test_fixed_residual_linearized_update_rejects_rhs_shape_mismatch() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    residual = lambda state: 2.0 * jnp.asarray(state, dtype=jnp.float64)

    with pytest.raises(ValueError, match="Linearized residual RHS has shape"):
        solve_fixed_residual_linearized_update(
            residual,
            jnp.ones(3, dtype=jnp.float64),
            rhs=jnp.ones(2, dtype=jnp.float64),
        )


def test_fixed_host_rhs_bridge_matches_dthe_packed_rhs_oracle() -> None:
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        fields,
        feedback_integrals,
        layout,
    ) = _dthe_context()
    fixed_state = fixed_state_from_fields(
        fields, feedback_integrals=feedback_integrals, layout=layout
    )

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


def test_active_array_backward_euler_residual_matches_full_field_oracle_on_dthe_deck() -> (
    None
):
    pytest.importorskip("jax")
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        fields,
        feedback_integrals,
        _,
    ) = _dthe_context()
    kwargs = {
        "runtime_model": runtime_model,
        "feedback_integrals": feedback_integrals,
        "mesh": mesh,
        "metrics": metrics,
        "dataset_scalars": scalars,
        "timestep": 1.0e-6,
    }
    oracle = build_recycling_1d_backward_euler_residual_context(
        config,
        fields,
        rhs_backend="fixed_full_field_array",
        **kwargs,
    )
    active = build_recycling_1d_backward_euler_residual_context(
        config,
        fields,
        rhs_backend="active_array",
        **kwargs,
    )
    state = np.asarray(oracle.packed_initial_guess, dtype=np.float64)

    assert tuple(active.field_names) == tuple(oracle.field_names)
    np.testing.assert_allclose(
        np.asarray(active.residual(state), dtype=np.float64),
        np.asarray(oracle.residual(state), dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_active_array_bdf2_residual_matches_full_field_oracle_on_dthe_deck() -> None:
    pytest.importorskip("jax")
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        fields,
        feedback_integrals,
        _,
    ) = _dthe_context()
    previous_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True) * (1.0 - 1.0e-6)
        for name, value in fields.items()
    }
    previous_feedback_integrals = {
        name: value - 1.0e-8 for name, value in feedback_integrals.items()
    }
    kwargs = {
        "runtime_model": runtime_model,
        "feedback_integrals": feedback_integrals,
        "previous_feedback_integrals": previous_feedback_integrals,
        "mesh": mesh,
        "metrics": metrics,
        "dataset_scalars": scalars,
        "timestep": 1.0e-6,
        "previous_timestep": 1.25e-6,
    }
    oracle = build_recycling_1d_bdf2_residual_context(
        config,
        fields,
        previous_fields,
        rhs_backend="fixed_full_field_array",
        **kwargs,
    )
    active = build_recycling_1d_bdf2_residual_context(
        config,
        fields,
        previous_fields,
        rhs_backend="active_array",
        **kwargs,
    )
    state = np.asarray(oracle.packed_initial_guess, dtype=np.float64)

    assert tuple(active.field_names) == tuple(oracle.field_names)
    np.testing.assert_allclose(
        np.asarray(active.residual(state), dtype=np.float64),
        np.asarray(oracle.residual(state), dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_bdf2_residual_context_accepts_supplied_initial_guess_fields() -> None:
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        fields,
        feedback_integrals,
        _,
    ) = _dthe_fixture_context()
    previous_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True) * (1.0 - 1.0e-6)
        for name, value in fields.items()
    }
    previous_feedback_integrals = {
        name: value - 1.0e-8 for name, value in feedback_integrals.items()
    }
    initial_guess_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True) * (1.0 + 2.0e-5)
        for name, value in fields.items()
    }

    context = build_recycling_1d_bdf2_residual_context(
        config,
        fields,
        previous_fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        previous_feedback_integrals=previous_feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=1.0e-6,
        previous_timestep=1.25e-6,
        initial_guess_fields=initial_guess_fields,
    )
    expected = pack_recycling_active_state(
        initial_guess_fields,
        feedback_integrals=feedback_integrals,
        field_names=runtime_model.field_names,
        feedback_names=(),
        mesh=mesh,
        layout=context.layout,
    )

    np.testing.assert_allclose(
        np.asarray(context.packed_initial_guess, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_active_array_fixture_residual_supports_jit_jvp_and_vmap() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        fields,
        feedback_integrals,
        _,
    ) = _dthe_fixture_context()
    kwargs = {
        "runtime_model": runtime_model,
        "feedback_integrals": feedback_integrals,
        "mesh": mesh,
        "metrics": metrics,
        "dataset_scalars": scalars,
        "timestep": 1.0e-6,
    }
    oracle = build_recycling_1d_backward_euler_residual_context(
        config,
        fields,
        rhs_backend="fixed_full_field_array",
        **kwargs,
    )
    active = build_recycling_1d_backward_euler_residual_context(
        config,
        fields,
        rhs_backend="active_array",
        **kwargs,
    )
    state = jnp.asarray(oracle.packed_initial_guess, dtype=jnp.float64)
    direction = jnp.sin(jnp.arange(state.size, dtype=jnp.float64) * 0.017)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1.0e-30)
    second_direction = jnp.cos(jnp.arange(state.size, dtype=jnp.float64) * 0.011)
    second_direction = second_direction / jnp.maximum(
        jnp.linalg.norm(second_direction),
        1.0e-30,
    )

    active_residual = jax.jit(active.residual)
    oracle_residual = jax.jit(oracle.residual)
    np.testing.assert_allclose(
        np.asarray(active_residual(state), dtype=np.float64),
        np.asarray(oracle_residual(state), dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    _, jvp_value = jax.jvp(active_residual, (state,), (direction,))
    epsilon = 1.0e-6
    finite_difference = (
        active_residual(state + epsilon * direction)
        - active_residual(state - epsilon * direction)
    ) / (2.0 * epsilon)
    relative_error = jnp.linalg.norm(jvp_value - finite_difference) / jnp.maximum(
        jnp.linalg.norm(finite_difference),
        1.0e-30,
    )
    assert float(relative_error) < 1.0e-6

    tangent_batch = jnp.stack(
        (
            direction,
            second_direction,
            0.25 * direction - 0.5 * second_direction,
        )
    )
    serial_jvps = jnp.stack(
        tuple(
            jax.jvp(active_residual, (state,), (tangent_batch[index],))[1]
            for index in range(tangent_batch.shape[0])
        )
    )
    batched_jvp = jax.jit(
        lambda base_state, tangents: fixed_residual_jvp_batch_action(
            active_residual,
            base_state,
            tangents,
        )
    )(state, tangent_batch)
    np.testing.assert_allclose(
        np.asarray(batched_jvp, dtype=np.float64),
        np.asarray(serial_jvps, dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    batch = jnp.stack((state, state + 1.0e-9 * direction))
    np.testing.assert_allclose(
        np.asarray(jax.vmap(active_residual)(batch), dtype=np.float64),
        np.asarray(jax.vmap(oracle_residual)(batch), dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_jax_linearized_recycling_step_reaches_full_fixed_residual_without_host_barrier() -> (
    None
):
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
    assert info.nonlinear_iterations == 0
    assert info.diagnostics["residual_evaluation_count"] == 1
    assert info.diagnostics["jacobian_refresh_count"] == 0
    assert info.diagnostics["jacobian_assembly_seconds"] >= 0.0
    assert info.diagnostics["jacobian_mode"] == "jax_linearized:jax_gmres"
    assert info.diagnostics["linear_solver_backend"] == "jax_gmres"
    assert info.diagnostics["linear_solver_status"] is None
    assert info.diagnostics["linear_solver_success"] is None
    assert info.diagnostics["linear_solver_reported_iterations"] is None


def test_jax_linearized_recycling_step_supports_dthe_fixed_residual() -> None:
    pytest.importorskip("jax")
    config, mesh, metrics, scalars, runtime_model, fields, feedback_integrals, _ = (
        _dthe_context()
    )

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
    assert info.nonlinear_iterations == 0
    assert info.diagnostics["residual_evaluation_count"] == 1
    assert info.diagnostics["jacobian_refresh_count"] == 0
    assert info.diagnostics["jacobian_mode"] == "jax_linearized:jax_gmres"
    assert info.diagnostics["linear_solver_backend"] == "jax_gmres"


def test_active_array_jax_linearized_recycling_step_supports_dthe_residual() -> None:
    pytest.importorskip("jax")
    config, mesh, metrics, scalars, runtime_model, fields, feedback_integrals, _ = (
        _dthe_context()
    )

    _, _, info = advance_recycling_1d_backward_euler_step(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=1.0e-6,
        solver_mode="active_array_jax_linearized",
        residual_tolerance=1.0e-6,
        max_nonlinear_iterations=1,
    )

    assert info.residual_inf_norm < 1.0e-8
    assert info.nonlinear_iterations == 0
    assert info.diagnostics["rhs_backend"] == "active_array"
    assert info.diagnostics["residual_evaluation_count"] == 1
    assert info.diagnostics["jacobian_refresh_count"] == 0
    assert info.diagnostics["jacobian_mode"] == "jax_linearized:jax_gmres"
    assert info.diagnostics["linear_solver_backend"] == "jax_gmres"


def test_backward_euler_residual_context_exposes_jvp_gate_on_dthe_deck() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    config, mesh, metrics, scalars, runtime_model, fields, feedback_integrals, _ = (
        _dthe_context()
    )
    context = build_recycling_1d_backward_euler_residual_context(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=1.0e-6,
    )
    state = jnp.asarray(context.packed_previous_state, dtype=jnp.float64)
    direction = jnp.sin(jnp.arange(state.size, dtype=jnp.float64) * 0.01)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1.0e-30)
    residual = jax.jit(context.residual)
    _, jvp_value = jax.jvp(residual, (state,), (direction,))
    epsilon = 1.0e-6
    finite_difference = (
        residual(state + epsilon * direction) - residual(state - epsilon * direction)
    ) / (2.0 * epsilon)
    relative_error = jnp.linalg.norm(jvp_value - finite_difference) / jnp.maximum(
        jnp.linalg.norm(finite_difference),
        1.0e-30,
    )

    assert tuple(context.field_names) == tuple(runtime_model.field_names)
    assert float(relative_error) < 1.0e-6


def test_bdf2_residual_context_exposes_jvp_gate_on_dthe_deck() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    config, mesh, metrics, scalars, runtime_model, fields, feedback_integrals, _ = (
        _dthe_context()
    )
    previous_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True)
        for name, value in fields.items()
    }
    for name, value in previous_fields.items():
        previous_fields[name] = value * (1.0 - 1.0e-6)
    previous_feedback_integrals = {
        name: value - 1.0e-8 for name, value in feedback_integrals.items()
    }
    context = build_recycling_1d_bdf2_residual_context(
        config,
        fields,
        previous_fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        previous_feedback_integrals=previous_feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=1.0e-6,
        previous_timestep=1.25e-6,
    )
    state = jnp.asarray(context.packed_previous_state, dtype=jnp.float64)
    direction = jnp.cos(jnp.arange(state.size, dtype=jnp.float64) * 0.013)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1.0e-30)
    residual = jax.jit(context.residual)
    _, jvp_value = jax.jvp(residual, (state,), (direction,))
    epsilon = 1.0e-6
    finite_difference = (
        residual(state + epsilon * direction) - residual(state - epsilon * direction)
    ) / (2.0 * epsilon)
    relative_error = jnp.linalg.norm(jvp_value - finite_difference) / jnp.maximum(
        jnp.linalg.norm(finite_difference),
        1.0e-30,
    )

    assert tuple(context.field_names) == tuple(runtime_model.field_names)
    assert float(relative_error) < 1.0e-6
