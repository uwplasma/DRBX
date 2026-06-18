from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import jax_drb.native.recycling_1d as recycling


class _FakeConfig:
    def __init__(self, **solver_options: float) -> None:
        self._solver_options = solver_options

    def has_option(self, section: str, key: str) -> bool:
        return section == "solver" and key in self._solver_options

    def parsed(self, section: str, key: str) -> float:
        if not self.has_option(section, key):
            raise KeyError((section, key))
        return self._solver_options[key]


class _FakeRuntimeConfig:
    def __init__(self, **runtime_options: float) -> None:
        self._runtime_options = runtime_options

    def has_option(self, section: str, key: str) -> bool:
        return section == "runtime" and key in self._runtime_options

    def parsed(self, section: str, key: str) -> float:
        if not self.has_option(section, key):
            raise KeyError((section, key))
        return self._runtime_options[key]


def _runtime_model(
    field_names: tuple[str, ...] = ("N",), feedback_names: tuple[str, ...] = ("ctrl",)
):
    return SimpleNamespace(
        field_names=field_names, feedback_names=feedback_names, controllers={}
    )


def _fields(value: float = 1.0) -> dict[str, np.ndarray]:
    return {"N": np.array([value], dtype=np.float64)}


class _FakeSparsity:
    shape = (2, 2)
    nnz = 0
    indptr = np.zeros(3, dtype=np.int32)
    indices = np.array([], dtype=np.int32)

    def tocsc(self) -> "_FakeSparsity":
        return self


def test_implicit_history_rejects_negative_steps() -> None:
    with pytest.raises(ValueError, match="steps must be non-negative"):
        recycling.advance_recycling_1d_implicit_history(
            _FakeConfig(),
            mesh=object(),
            metrics=object(),
            dataset_scalars={},
            timestep=1.0,
            steps=-1,
        )


@pytest.mark.parametrize(
    ("solver_mode", "target_name"),
    (
        ("continuation", "_advance_recycling_1d_continuation_history"),
        ("adaptive_be", "_advance_recycling_1d_adaptive_be_history"),
        ("adaptive_bdf", "_advance_recycling_1d_adaptive_bdf_history"),
        (
            "adaptive_bdf_active_array_jax_linearized",
            "_advance_recycling_1d_adaptive_bdf_history",
        ),
        (
            "adaptive_bdf_active_array_jax_linearized_lineax",
            "_advance_recycling_1d_adaptive_bdf_history",
        ),
        ("bdf", "_advance_recycling_1d_bdf_history"),
        ("bdf_fixed_full_field_jvp", "_advance_recycling_1d_bdf_history"),
        ("bdf_active_array_jvp", "_advance_recycling_1d_bdf_history"),
        ("fixed_bdf2_jax_linearized", "_advance_recycling_1d_fixed_bdf2_history"),
        (
            "fixed_bdf2_jax_linearized_lineax",
            "_advance_recycling_1d_fixed_bdf2_history",
        ),
        (
            "fixed_bdf2_active_array_jax_linearized",
            "_advance_recycling_1d_fixed_bdf2_history",
        ),
        (
            "fixed_bdf2_active_array_jax_linearized_lineax",
            "_advance_recycling_1d_fixed_bdf2_history",
        ),
    ),
)
def test_implicit_history_dispatches_solver_modes(
    solver_mode: str,
    target_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _runtime_model()
    sentinel = object()
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        recycling, "_build_recycling_runtime_model", lambda *args, **kwargs: model
    )
    monkeypatch.setattr(
        recycling, "_build_recycling_state_fields", lambda *args, **kwargs: _fields()
    )

    def fake_history(*args: object, **kwargs: object) -> object:
        calls["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(recycling, target_name, fake_history)

    result = recycling.advance_recycling_1d_implicit_history(
        _FakeConfig(),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        timestep=2.0,
        steps=3,
        solver_mode=solver_mode,
        residual_tolerance=1.0e-7,
        max_nonlinear_iterations=9,
    )

    assert result is sentinel
    assert calls["kwargs"]["runtime_model"] is model
    assert calls["kwargs"]["field_names"] == ("N",)
    assert calls["kwargs"]["feedback_names"] == ("ctrl",)
    if solver_mode == "bdf_fixed_full_field_jvp":
        assert calls["kwargs"]["jacobian_mode"] == "jvp"
        assert calls["kwargs"]["rhs_backend"] == "fixed_full_field_array"
        assert calls["kwargs"]["solver_mode_label"] == "bdf_fixed_full_field_jvp"
    if solver_mode == "bdf_active_array_jvp":
        assert calls["kwargs"]["jacobian_mode"] == "jvp"
        assert calls["kwargs"]["rhs_backend"] == "active_array"
        assert calls["kwargs"]["solver_mode_label"] == "bdf_active_array_jvp"
    if solver_mode.startswith("adaptive_bdf_active_array_jax_linearized"):
        expected_step_solver = (
            "active_array_jax_linearized_lineax"
            if solver_mode.endswith("_lineax")
            else "active_array_jax_linearized"
        )
        assert calls["kwargs"]["step_solver_mode"] == expected_step_solver
    if solver_mode.startswith("fixed_bdf2_jax_linearized"):
        expected_step_solver = (
            "jax_linearized_lineax"
            if solver_mode.endswith("_lineax")
            else "jax_linearized"
        )
        assert calls["kwargs"]["step_solver_mode"] == expected_step_solver
        assert calls["kwargs"]["solver_mode_label"] == solver_mode
    if solver_mode.startswith("fixed_bdf2_active_array_jax_linearized"):
        expected_step_solver = (
            "active_array_jax_linearized_lineax"
            if solver_mode.endswith("_lineax")
            else "active_array_jax_linearized"
        )
        assert calls["kwargs"]["step_solver_mode"] == expected_step_solver
        assert calls["kwargs"]["solver_mode_label"] == solver_mode
    if solver_mode not in {"bdf", "bdf_fixed_full_field_jvp", "bdf_active_array_jvp"}:
        assert calls["kwargs"]["residual_tolerance"] == 1.0e-7
        assert calls["kwargs"]["max_nonlinear_iterations"] == 9


def test_generic_implicit_history_accumulates_fields_integrals_and_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _runtime_model()
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        recycling, "_build_recycling_runtime_model", lambda *args, **kwargs: model
    )
    monkeypatch.setattr(
        recycling, "_build_recycling_state_fields", lambda *args, **kwargs: _fields()
    )

    def fake_step(config, fields, *, feedback_integrals, timestep, **kwargs):
        next_fields = {"N": fields["N"] + timestep}
        next_integrals = {"ctrl": feedback_integrals["ctrl"] + timestep}
        return next_fields, next_integrals, object()

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_step
    )
    monkeypatch.setattr(
        recycling,
        "_build_recycling_progress_details",
        lambda **kwargs: (
            {"stage": "progress", **kwargs},
            kwargs["interval_started_at"],
        ),
    )

    result = recycling.advance_recycling_1d_implicit_history(
        _FakeConfig(),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        timestep=0.5,
        steps=2,
        solver_mode="sparse",
        progress_callback=events.append,
    )

    np.testing.assert_allclose(
        result.variable_history["N"][:, 0], np.array([1.0, 1.5, 2.0])
    )
    np.testing.assert_allclose(
        result.feedback_integral_history["ctrl"], np.array([0.0, 0.5, 1.0])
    )
    assert [event["interval_index"] for event in events] == [1, 2]
    assert all(event["solver_mode"] == "sparse" for event in events)


def test_fixed_bdf2_jax_linearized_history_uses_startup_bdf2_and_packed_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, bool]] = []
    events: list[dict[str, object]] = []

    def make_info(method: str) -> recycling.Recycling1DImplicitStepInfo:
        return recycling.Recycling1DImplicitStepInfo(
            residual_inf_norm=0.25 if method == "backward_euler" else 0.125,
            active_size=1,
            nonlinear_iterations=2,
            linear_iterations=3,
            diagnostics={
                "solver_mode": "jax_linearized_lineax",
                "rhs_backend": "fixed_full_field_array",
                "residual_evaluation_count": 4,
                "jacobian_refresh_count": 1,
                "linear_solve_seconds": 0.5,
                "residual_evaluation_seconds": 0.25,
                "residual_jitted": True,
            },
        )

    def fake_backward_euler_step(
        config,
        fields,
        *,
        feedback_integrals,
        solver_mode,
        evolve_feedback_integrals,
        **kwargs,
    ):
        calls.append(("be", solver_mode, evolve_feedback_integrals))
        assert feedback_integrals == {"ctrl": 0.0}
        return (
            {"N": fields["N"] + 1.0},
            {"ctrl": feedback_integrals["ctrl"] + 1.0},
            make_info("backward_euler"),
        )

    def fake_bdf2_step(
        config,
        fields,
        previous_fields,
        *,
        feedback_integrals,
        previous_feedback_integrals,
        solver_mode,
        evolve_feedback_integrals,
        previous_timestep,
        **kwargs,
    ):
        calls.append(("bdf2", solver_mode, evolve_feedback_integrals))
        assert previous_timestep == 0.5
        assert previous_feedback_integrals["ctrl"] < feedback_integrals["ctrl"]
        assert np.asarray(previous_fields["N"]).shape == np.asarray(fields["N"]).shape
        return (
            {"N": fields["N"] + 10.0},
            {"ctrl": feedback_integrals["ctrl"] + 10.0},
            make_info("bdf2"),
        )

    monkeypatch.setattr(
        recycling,
        "advance_recycling_1d_backward_euler_step",
        fake_backward_euler_step,
    )
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf2_step)
    monkeypatch.setattr(
        recycling,
        "_build_recycling_progress_details",
        lambda **kwargs: (
            {
                "interval_index": kwargs["interval_index"],
                "solver_mode": kwargs["solver_mode"],
            },
            kwargs["interval_started_at"],
        ),
    )

    result = recycling._advance_recycling_1d_fixed_bdf2_history(
        _FakeConfig(),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        timestep=0.5,
        steps=3,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=4,
        progress_callback=events.append,
        step_solver_mode="jax_linearized_lineax",
        solver_mode_label="fixed_bdf2_jax_linearized_lineax",
    )

    assert calls == [
        ("be", "jax_linearized_lineax", True),
        ("bdf2", "jax_linearized_lineax", True),
        ("bdf2", "jax_linearized_lineax", True),
    ]
    np.testing.assert_allclose(
        result.variable_history["N"][:, 0], [1.0, 2.0, 12.0, 22.0]
    )
    np.testing.assert_allclose(
        result.feedback_integral_history["ctrl"], [0.0, 1.0, 11.0, 21.0]
    )
    assert events == [
        {"interval_index": 1, "solver_mode": "fixed_bdf2_jax_linearized_lineax"},
        {"interval_index": 2, "solver_mode": "fixed_bdf2_jax_linearized_lineax"},
        {"interval_index": 3, "solver_mode": "fixed_bdf2_jax_linearized_lineax"},
    ]
    diagnostics = result.diagnostics
    assert diagnostics["fixed_bdf2_startup_steps"] == 1
    assert diagnostics["fixed_bdf2_bdf2_steps"] == 2
    assert diagnostics["fixed_bdf2_fixed_full_field_rhs_steps"] == 3
    assert diagnostics["fixed_bdf2_jax_linearized_action_steps"] == 3
    assert diagnostics["fixed_bdf2_lineax_action_steps"] == 3
    assert diagnostics["fixed_bdf2_residual_jitted_steps"] == 3
    assert diagnostics["fixed_bdf2_evolve_feedback_integrals"] is True


def test_fixed_bdf2_active_array_history_aggregates_solver_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step_modes: list[str] = []

    def make_info() -> recycling.Recycling1DImplicitStepInfo:
        return recycling.Recycling1DImplicitStepInfo(
            residual_inf_norm=1.0e-12,
            active_size=1,
            nonlinear_iterations=1,
            linear_iterations=2,
            diagnostics={
                "rhs_backend": "active_array",
                "solver_mode": "active_array_jax_linearized",
                "residual_evaluation_count": 3,
                "jacobian_refresh_count": 0,
                "linear_solve_seconds": 0.25,
                "residual_evaluation_seconds": 0.125,
                "residual_jitted": True,
                "converged": True,
                "linear_solver_success": True,
                "linear_preconditioner": "local_block_diag",
                "linear_preconditioner_build_count": 2,
                "linear_preconditioner_build_seconds": 0.125,
            },
        )

    def fake_backward_euler_step(
        config,
        fields,
        *,
        feedback_integrals,
        solver_mode,
        **kwargs,
    ):
        step_modes.append(str(solver_mode))
        return (
            {"N": fields["N"] + 1.0},
            {"ctrl": feedback_integrals["ctrl"] + 1.0},
            make_info(),
        )

    def fake_bdf2_step(
        config,
        fields,
        previous_fields,
        *,
        feedback_integrals,
        solver_mode,
        **kwargs,
    ):
        step_modes.append(str(solver_mode))
        return (
            {"N": fields["N"] + 1.0},
            {"ctrl": feedback_integrals["ctrl"] + 1.0},
            make_info(),
        )

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_backward_euler_step
    )
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf2_step)

    result = recycling._advance_recycling_1d_fixed_bdf2_history(
        _FakeConfig(),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        timestep=0.5,
        steps=2,
        residual_tolerance=1.0e-7,
        max_nonlinear_iterations=3,
        step_solver_mode="active_array_jax_linearized",
        solver_mode_label="fixed_bdf2_active_array_jax_linearized",
    )

    assert step_modes == ["active_array_jax_linearized", "active_array_jax_linearized"]
    diagnostics = result.diagnostics
    assert diagnostics["fixed_bdf2_solver_mode"] == (
        "fixed_bdf2_active_array_jax_linearized"
    )
    assert diagnostics["fixed_bdf2_step_solver_mode"] == "active_array_jax_linearized"
    assert diagnostics["fixed_bdf2_startup_steps"] == 1
    assert diagnostics["fixed_bdf2_bdf2_steps"] == 1
    assert diagnostics["fixed_bdf2_fixed_full_field_rhs_steps"] == 0
    assert diagnostics["fixed_bdf2_active_array_rhs_steps"] == 2
    assert diagnostics["fixed_bdf2_jax_linearized_action_steps"] == 2
    assert diagnostics["fixed_bdf2_residual_jitted_steps"] == 2
    assert diagnostics["fixed_bdf2_linear_preconditioner"] == "local_block_diag"
    assert diagnostics["fixed_bdf2_total_linear_preconditioner_build_count"] == 4
    assert diagnostics[
        "fixed_bdf2_total_linear_preconditioner_build_seconds"
    ] == pytest.approx(0.25)
    assert diagnostics["fixed_bdf2_unconverged_solver_steps"] == 0
    assert diagnostics["fixed_bdf2_unknown_convergence_solver_steps"] == 0
    assert diagnostics["fixed_bdf2_linear_solver_failed_steps"] == 0
    np.testing.assert_allclose(result.variable_history["N"][:, 0], [1.0, 2.0, 3.0])


def test_fixed_bdf2_history_uses_internal_substeps_without_extra_output_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float, float | None]] = []
    events: list[dict[str, object]] = []

    def make_info() -> recycling.Recycling1DImplicitStepInfo:
        return recycling.Recycling1DImplicitStepInfo(
            residual_inf_norm=1.0e-12,
            active_size=1,
            nonlinear_iterations=1,
            linear_iterations=1,
            diagnostics={
                "rhs_backend": "active_array",
                "solver_mode": "active_array_jax_linearized",
                "residual_jitted": True,
                "converged": True,
                "linear_solver_success": True,
            },
        )

    def fake_backward_euler_step(
        config,
        fields,
        *,
        feedback_integrals,
        timestep,
        **kwargs,
    ):
        calls.append(("be", float(timestep), None))
        return (
            {"N": fields["N"] + 1.0},
            {"ctrl": feedback_integrals["ctrl"] + 1.0},
            make_info(),
        )

    def fake_bdf2_step(
        config,
        fields,
        previous_fields,
        *,
        feedback_integrals,
        previous_feedback_integrals,
        timestep,
        previous_timestep,
        **kwargs,
    ):
        calls.append(("bdf2", float(timestep), float(previous_timestep)))
        assert previous_feedback_integrals["ctrl"] < feedback_integrals["ctrl"]
        assert np.asarray(previous_fields["N"]).shape == np.asarray(fields["N"]).shape
        return (
            {"N": fields["N"] + 1.0},
            {"ctrl": feedback_integrals["ctrl"] + 1.0},
            make_info(),
        )

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_backward_euler_step
    )
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf2_step)
    monkeypatch.setattr(
        recycling,
        "_build_recycling_progress_details",
        lambda **kwargs: (
            {
                "interval_index": kwargs["interval_index"],
                "accepted_dt": kwargs["accepted_dt"],
                "stored_states": kwargs["stored_states"],
            },
            kwargs["interval_started_at"],
        ),
    )

    result = recycling._advance_recycling_1d_fixed_bdf2_history(
        _FakeRuntimeConfig(recycling_fixed_bdf2_max_internal_timestep=0.5),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        timestep=1.0,
        steps=2,
        residual_tolerance=1.0e-7,
        max_nonlinear_iterations=3,
        progress_callback=events.append,
        step_solver_mode="active_array_jax_linearized",
        solver_mode_label="fixed_bdf2_active_array_jax_linearized",
    )

    assert calls == [
        ("be", 0.5, None),
        ("bdf2", 0.5, 0.5),
        ("bdf2", 0.5, 0.5),
        ("bdf2", 0.5, 0.5),
    ]
    np.testing.assert_allclose(result.variable_history["N"][:, 0], [1.0, 3.0, 5.0])
    np.testing.assert_allclose(
        result.feedback_integral_history["ctrl"], [0.0, 2.0, 4.0]
    )
    assert events == [
        {"interval_index": 1, "accepted_dt": 0.5, "stored_states": 2},
        {"interval_index": 2, "accepted_dt": 0.5, "stored_states": 3},
    ]
    diagnostics = result.diagnostics
    assert diagnostics["fixed_bdf2_startup_steps"] == 1
    assert diagnostics["fixed_bdf2_bdf2_steps"] == 3
    assert diagnostics["fixed_bdf2_internal_substeps"] == 4
    assert diagnostics["fixed_bdf2_max_output_substeps"] == 2
    assert diagnostics["fixed_bdf2_max_internal_timestep"] == 0.5
    assert diagnostics["fixed_bdf2_active_array_rhs_steps"] == 4
    assert diagnostics["fixed_bdf2_jax_linearized_action_steps"] == 4


def test_adaptive_be_history_accumulates_interval_results_and_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []

    def fake_interval(config, fields, *, feedback_integrals, suggested_dt, **kwargs):
        return (
            {"N": fields["N"] + 1.0},
            {"ctrl": feedback_integrals["ctrl"] + 2.0},
            suggested_dt + 0.25,
        )

    monkeypatch.setattr(
        recycling, "_advance_recycling_1d_adaptive_be_interval", fake_interval
    )
    monkeypatch.setattr(
        recycling,
        "_build_recycling_progress_details",
        lambda **kwargs: (
            {"stage": "progress", **kwargs},
            kwargs["interval_started_at"],
        ),
    )

    result = recycling._advance_recycling_1d_adaptive_be_history(
        _FakeConfig(),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        timestep=1.0,
        steps=2,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
        progress_callback=events.append,
    )

    np.testing.assert_allclose(
        result.variable_history["N"][:, 0], np.array([1.0, 2.0, 3.0])
    )
    np.testing.assert_allclose(
        result.feedback_integral_history["ctrl"], np.array([0.0, 2.0, 4.0])
    )
    assert [event["solver_mode"] for event in events] == ["adaptive_be", "adaptive_be"]


def test_adaptive_bdf_history_threads_previous_state_and_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []
    previous_seen: list[object] = []
    monkeypatch.setattr(
        recycling, "_initial_recycling_continuation_dt", lambda *args, **kwargs: 0.5
    )

    def fake_interval(
        config,
        fields,
        *,
        feedback_integrals,
        previous_fields,
        previous_integrals,
        previous_dt,
        **kwargs,
    ):
        previous_seen.append(previous_fields)
        next_fields = {"N": fields["N"] + 1.0}
        next_integrals = {"ctrl": feedback_integrals["ctrl"] + 3.0}
        stats = recycling._new_adaptive_bdf_interval_stats(kwargs["step_solver_mode"])
        stats["adaptive_bdf_accepted_steps"] = 2
        stats["adaptive_bdf_rejected_steps"] = 1 if previous_fields is None else 0
        stats["adaptive_bdf_startup_trials"] = 1
        stats["adaptive_bdf_bdf2_trials"] = 1
        stats["adaptive_bdf_bdf2_accepted_steps"] = 1
        stats["adaptive_bdf_min_accepted_dt"] = 0.25
        stats["adaptive_bdf_max_accepted_dt"] = 0.5
        stats["adaptive_bdf_last_error_ratio"] = 0.1 if previous_fields is None else 0.2
        stats["adaptive_bdf_max_error_ratio"] = 0.8 if previous_fields is None else 0.6
        stats["adaptive_bdf_last_accepted_error_ratio"] = (
            0.1 if previous_fields is None else 0.2
        )
        stats["adaptive_bdf_max_accepted_error_ratio"] = (
            0.7 if previous_fields is None else 0.6
        )
        return next_fields, next_integrals, fields, feedback_integrals, 0.5, 0.5, stats

    monkeypatch.setattr(
        recycling, "_advance_recycling_1d_adaptive_bdf_interval", fake_interval
    )
    monkeypatch.setattr(
        recycling,
        "_build_recycling_progress_details",
        lambda **kwargs: (
            {"stage": "progress", **kwargs},
            kwargs["interval_started_at"],
        ),
    )

    result = recycling._advance_recycling_1d_adaptive_bdf_history(
        _FakeConfig(),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        timestep=1.0,
        steps=2,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
        progress_callback=events.append,
    )

    assert previous_seen[0] is None
    assert previous_seen[1] is not None
    np.testing.assert_allclose(
        result.variable_history["N"][:, 0], np.array([1.0, 2.0, 3.0])
    )
    np.testing.assert_allclose(
        result.feedback_integral_history["ctrl"], np.array([0.0, 3.0, 6.0])
    )
    assert [event["solver_mode"] for event in events] == [
        "adaptive_bdf",
        "adaptive_bdf",
    ]
    assert result.diagnostics["adaptive_bdf_interval_count"] == 2
    assert result.diagnostics["adaptive_bdf_accepted_steps"] == 4
    assert result.diagnostics["adaptive_bdf_rejected_steps"] == 1
    assert result.diagnostics["adaptive_bdf_startup_trials"] == 2
    assert result.diagnostics["adaptive_bdf_bdf2_trials"] == 2
    assert result.diagnostics["adaptive_bdf_bdf2_accepted_steps"] == 2
    assert result.diagnostics["adaptive_bdf_min_accepted_dt"] == 0.25
    assert result.diagnostics["adaptive_bdf_max_accepted_dt"] == 0.5
    assert result.diagnostics["adaptive_bdf_last_error_ratio"] == 0.2
    assert result.diagnostics["adaptive_bdf_max_error_ratio"] == 0.8
    assert result.diagnostics["adaptive_bdf_last_accepted_error_ratio"] == 0.2
    assert result.diagnostics["adaptive_bdf_max_accepted_error_ratio"] == 0.7
    assert result.diagnostics["adaptive_bdf_step_solver_mode"] == "sparse"


def test_choose_recycling_next_dt_handles_finished_zero_and_scaled_errors() -> None:
    assert (
        recycling._choose_recycling_next_dt(
            0.5, error_ratio=1.0, order=1, remaining=0.0, minimum_dt=0.1
        )
        == 0.5
    )
    assert (
        recycling._choose_recycling_next_dt(
            0.5, error_ratio=0.0, order=1, remaining=2.0, minimum_dt=0.1
        )
        == 1.0
    )
    assert (
        recycling._choose_recycling_next_dt(
            0.5, error_ratio=float("nan"), order=1, remaining=2.0, minimum_dt=0.1
        )
        == 1.0
    )
    assert (
        recycling._choose_recycling_next_dt(
            0.5, error_ratio=8.0, order=1, remaining=2.0, minimum_dt=0.1
        )
        == 0.25
    )


def test_adaptive_bdf_minimum_dt_preserves_full_window_policy() -> None:
    assert recycling._adaptive_bdf_minimum_dt(5000.0) == pytest.approx(5000.0 / 8192.0)


def test_adaptive_bdf_minimum_dt_scales_below_short_diagnostic_window() -> None:
    assert recycling._adaptive_bdf_minimum_dt(0.05) == pytest.approx(0.05 / 64.0)


def test_adaptive_bdf_minimum_dt_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError, match="output_timestep must be positive"):
        recycling._adaptive_bdf_minimum_dt(0.0)


def test_record_adaptive_bdf_step_solver_info_counts_convergence_states() -> None:
    stats = recycling._new_adaptive_bdf_interval_stats("jax_linearized")

    recycling._record_adaptive_bdf_step_solver_info(
        stats,
        SimpleNamespace(
            linear_iterations=7,
            diagnostics={
                "converged": True,
                "rhs_backend": "fixed_full_field_array",
                "jacobian_mode": "jax_linearized:jax_gmres",
                "residual_evaluation_count": 3,
                "residual_evaluation_seconds": 0.5,
                "jacobian_refresh_count": 2,
                "jacobian_assembly_seconds": 0.25,
                "linear_solve_seconds": 0.75,
                "line_search_seconds": 0.125,
                "linear_solver_tolerance": 1.0e-5,
                "linear_solver_success": False,
                "linear_preconditioner": "local_block_diag",
                "linear_preconditioner_build_count": 2,
                "linear_preconditioner_build_seconds": 0.075,
                "jvp_direction_workspace_reuses": 1,
                "jvp_jacobian_batch_count": 2,
                "jvp_jacobian_prebuilt_direction_batch_uses": 1,
                "jvp_jacobian_total_seconds": 0.2,
                "jvp_jacobian_linearize_seconds": 0.03,
                "jvp_jacobian_tangent_build_seconds": 0.04,
                "jvp_jacobian_push_seconds": 0.05,
                "jvp_jacobian_device_execute_seconds": 0.02,
                "jvp_jacobian_host_transfer_seconds": 0.03,
                "jvp_jacobian_sparse_assembly_seconds": 0.06,
            },
        ),
    )
    recycling._record_adaptive_bdf_step_solver_info(
        stats,
        SimpleNamespace(
            diagnostics={
                "converged": False,
                "rhs_backend": "fixed_full_field_array",
                "jacobian_mode": "jvp",
            }
        ),
    )
    recycling._record_adaptive_bdf_step_solver_info(
        stats,
        SimpleNamespace(
            diagnostics={
                "converged": True,
                "rhs_backend": "active_array",
                "jacobian_mode": "jax_linearized:jax_gmres",
            }
        ),
    )
    recycling._record_adaptive_bdf_step_solver_info(
        stats,
        SimpleNamespace(
            diagnostics={
                "converged": True,
                "rhs_backend": "active_array",
                "jacobian_mode": "jax_linearized:lineax_gmres",
            }
        ),
    )
    recycling._record_adaptive_bdf_step_solver_info(
        stats,
        SimpleNamespace(
            diagnostics={
                "converged": True,
                "rhs_backend": "fixed_full_field_array",
                "jacobian_mode": "jax_linearized:jax_bicgstab",
                "linear_solver_backend": "jax_bicgstab",
                "linear_solver_success": None,
            }
        ),
    )
    recycling._record_adaptive_bdf_step_solver_info(
        stats,
        SimpleNamespace(
            diagnostics={
                "converged": True,
                "rhs_backend": "host_bridge",
                "jacobian_mode": "fd",
            }
        ),
    )
    recycling._record_adaptive_bdf_step_solver_info(stats, SimpleNamespace())

    assert stats["adaptive_bdf_trial_solver_steps"] == 7
    assert stats["adaptive_bdf_unconverged_solver_steps"] == 1
    assert stats["adaptive_bdf_unknown_convergence_solver_steps"] == 1
    assert stats["adaptive_bdf_fixed_full_field_rhs_solver_steps"] == 3
    assert stats["adaptive_bdf_active_array_rhs_solver_steps"] == 2
    assert stats["adaptive_bdf_host_bridge_rhs_solver_steps"] == 1
    assert stats["adaptive_bdf_jax_linearized_action_solver_steps"] == 4
    assert stats["adaptive_bdf_lineax_action_solver_steps"] == 1
    assert stats["adaptive_bdf_bicgstab_action_solver_steps"] == 1
    assert stats["adaptive_bdf_sparse_jvp_jacobian_solver_steps"] == 1
    assert stats["adaptive_bdf_fd_jacobian_solver_steps"] == 1
    assert stats["adaptive_bdf_residual_evaluation_count"] == 3
    assert stats["adaptive_bdf_jacobian_refresh_count"] == 2
    assert stats["adaptive_bdf_linear_iterations"] == 7
    assert stats["adaptive_bdf_linear_solver_tolerance"] == pytest.approx(1.0e-5)
    assert stats["adaptive_bdf_linear_preconditioner"] == "local_block_diag"
    assert stats["adaptive_bdf_linear_preconditioner_build_count"] == 2
    assert stats["adaptive_bdf_linear_preconditioner_build_seconds"] == pytest.approx(
        0.075
    )
    assert stats["adaptive_bdf_linear_solver_failed_steps"] == 1
    assert stats["adaptive_bdf_unknown_linear_solver_steps"] == 1
    assert stats["adaptive_bdf_sparse_jvp_workspace_reuses"] == 1
    assert stats["adaptive_bdf_jvp_jacobian_batch_count"] == 2
    assert stats["adaptive_bdf_jvp_jacobian_prebuilt_direction_batch_uses"] == 1
    assert stats["adaptive_bdf_residual_evaluation_seconds"] == pytest.approx(0.5)
    assert stats["adaptive_bdf_jacobian_assembly_seconds"] == pytest.approx(0.25)
    assert stats["adaptive_bdf_linear_solve_seconds"] == pytest.approx(0.75)
    assert stats["adaptive_bdf_line_search_seconds"] == pytest.approx(0.125)
    assert stats["adaptive_bdf_jvp_jacobian_total_seconds"] == pytest.approx(0.2)
    assert stats["adaptive_bdf_jvp_jacobian_linearize_seconds"] == pytest.approx(0.03)
    assert stats["adaptive_bdf_jvp_jacobian_tangent_build_seconds"] == pytest.approx(
        0.04
    )
    assert stats["adaptive_bdf_jvp_jacobian_push_seconds"] == pytest.approx(0.05)
    assert stats["adaptive_bdf_jvp_jacobian_device_execute_seconds"] == pytest.approx(
        0.02
    )
    assert stats["adaptive_bdf_jvp_jacobian_host_transfer_seconds"] == pytest.approx(
        0.03
    )
    assert stats["adaptive_bdf_jvp_jacobian_sparse_assembly_seconds"] == pytest.approx(
        0.06
    )


def test_record_adaptive_bdf_step_solver_info_does_not_count_initial_convergence_as_unknown_linear_solver() -> (
    None
):
    stats = recycling._new_adaptive_bdf_interval_stats("sparse_jvp")

    recycling._record_adaptive_bdf_step_solver_info(
        stats,
        SimpleNamespace(
            linear_iterations=0,
            diagnostics={
                "converged": True,
                "rhs_backend": "fixed_full_field_array",
                "jacobian_mode": "jvp",
                "linear_solver_backend": None,
                "linear_solver_success": None,
            },
        ),
    )

    assert stats["adaptive_bdf_trial_solver_steps"] == 1
    assert stats["adaptive_bdf_unknown_linear_solver_steps"] == 0
    assert stats["adaptive_bdf_linear_solver_failed_steps"] == 0


def test_adaptive_bdf_interval_stats_accumulates_timing_fields() -> None:
    total = recycling._new_adaptive_bdf_interval_stats("jax_linearized")
    step = recycling._new_adaptive_bdf_interval_stats("jax_linearized")
    step["adaptive_bdf_interval_wall_seconds"] = 10.0
    step["adaptive_bdf_startup_trial_seconds"] = 1.0
    step["adaptive_bdf_backward_euler_trial_seconds"] = 2.0
    step["adaptive_bdf_bdf2_trial_seconds"] = 3.0
    step["adaptive_bdf_error_estimator_seconds"] = 0.1
    step["adaptive_bdf_residual_evaluation_seconds"] = 4.0
    step["adaptive_bdf_jacobian_assembly_seconds"] = 5.0
    step["adaptive_bdf_linear_solve_seconds"] = 6.0
    step["adaptive_bdf_line_search_seconds"] = 7.0
    step["adaptive_bdf_linear_preconditioner"] = "local_block_diag"
    step["adaptive_bdf_linear_preconditioner_build_count"] = 17
    step["adaptive_bdf_linear_preconditioner_build_seconds"] = 7.5
    step["adaptive_bdf_jvp_jacobian_total_seconds"] = 8.0
    step["adaptive_bdf_jvp_jacobian_linearize_seconds"] = 9.0
    step["adaptive_bdf_jvp_jacobian_tangent_build_seconds"] = 10.0
    step["adaptive_bdf_jvp_jacobian_push_seconds"] = 11.0
    step["adaptive_bdf_jvp_jacobian_device_execute_seconds"] = 12.0
    step["adaptive_bdf_jvp_jacobian_host_transfer_seconds"] = 13.0
    step["adaptive_bdf_jvp_jacobian_sparse_assembly_seconds"] = 14.0
    step["adaptive_bdf_residual_evaluation_count"] = 8
    step["adaptive_bdf_jacobian_refresh_count"] = 9
    step["adaptive_bdf_linear_iterations"] = 10
    step["adaptive_bdf_linear_solver_tolerance"] = 1.0e-4
    step["adaptive_bdf_linear_solver_failed_steps"] = 11
    step["adaptive_bdf_jvp_jacobian_batch_count"] = 12
    step["adaptive_bdf_jvp_jacobian_prebuilt_direction_batch_uses"] = 13
    step["adaptive_bdf_reused_history_after_rejection"] = 14
    step["adaptive_bdf_bicgstab_action_solver_steps"] = 15
    step["adaptive_bdf_unknown_linear_solver_steps"] = 16

    recycling._accumulate_adaptive_bdf_interval_stats(total, step)

    assert total["adaptive_bdf_interval_wall_seconds"] == pytest.approx(10.0)
    assert total["adaptive_bdf_startup_trial_seconds"] == pytest.approx(1.0)
    assert total["adaptive_bdf_backward_euler_trial_seconds"] == pytest.approx(2.0)
    assert total["adaptive_bdf_bdf2_trial_seconds"] == pytest.approx(3.0)
    assert total["adaptive_bdf_error_estimator_seconds"] == pytest.approx(0.1)
    assert total["adaptive_bdf_residual_evaluation_seconds"] == pytest.approx(4.0)
    assert total["adaptive_bdf_jacobian_assembly_seconds"] == pytest.approx(5.0)
    assert total["adaptive_bdf_linear_solve_seconds"] == pytest.approx(6.0)
    assert total["adaptive_bdf_line_search_seconds"] == pytest.approx(7.0)
    assert total["adaptive_bdf_linear_preconditioner"] == "local_block_diag"
    assert total["adaptive_bdf_linear_preconditioner_build_count"] == 17
    assert total["adaptive_bdf_linear_preconditioner_build_seconds"] == pytest.approx(
        7.5
    )
    assert total["adaptive_bdf_jvp_jacobian_total_seconds"] == pytest.approx(8.0)
    assert total["adaptive_bdf_jvp_jacobian_linearize_seconds"] == pytest.approx(9.0)
    assert total["adaptive_bdf_jvp_jacobian_tangent_build_seconds"] == pytest.approx(
        10.0
    )
    assert total["adaptive_bdf_jvp_jacobian_push_seconds"] == pytest.approx(11.0)
    assert total["adaptive_bdf_jvp_jacobian_device_execute_seconds"] == pytest.approx(
        12.0
    )
    assert total["adaptive_bdf_jvp_jacobian_host_transfer_seconds"] == pytest.approx(
        13.0
    )
    assert total["adaptive_bdf_jvp_jacobian_sparse_assembly_seconds"] == pytest.approx(
        14.0
    )
    assert total["adaptive_bdf_residual_evaluation_count"] == 8
    assert total["adaptive_bdf_jacobian_refresh_count"] == 9
    assert total["adaptive_bdf_linear_iterations"] == 10
    assert total["adaptive_bdf_linear_solver_tolerance"] == pytest.approx(1.0e-4)
    assert total["adaptive_bdf_linear_solver_failed_steps"] == 11
    assert total["adaptive_bdf_jvp_jacobian_batch_count"] == 12
    assert total["adaptive_bdf_jvp_jacobian_prebuilt_direction_batch_uses"] == 13
    assert total["adaptive_bdf_reused_history_after_rejection"] == 14
    assert total["adaptive_bdf_bicgstab_action_solver_steps"] == 15
    assert total["adaptive_bdf_unknown_linear_solver_steps"] == 16


def test_adaptive_bdf_rejected_history_reuse_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not recycling._resolve_recycling_adaptive_bdf_reuse_rejected_history(
        _FakeConfig(), step_solver_mode="sparse"
    )
    assert recycling._resolve_recycling_adaptive_bdf_reuse_rejected_history(
        _FakeConfig(), step_solver_mode="jax_linearized"
    )
    assert not recycling._resolve_recycling_adaptive_bdf_reuse_rejected_history(
        _FakeRuntimeConfig(recycling_adaptive_bdf_reuse_rejected_history="false"),
        step_solver_mode="jax_linearized",
    )

    monkeypatch.setenv(
        "JAX_DRB_RECYCLING_ADAPTIVE_BDF_REUSE_REJECTED_HISTORY", "true"
    )
    assert recycling._resolve_recycling_adaptive_bdf_reuse_rejected_history(
        _FakeConfig(), step_solver_mode="sparse"
    )


def test_adaptive_bdf_trace_flushes_start_record_before_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    trace_path = tmp_path / "adaptive_bdf_trace.jsonl"

    def fake_step(*args, **kwargs):
        raise RuntimeError("synthetic implicit failure")

    monkeypatch.setenv("JAX_DRB_RECYCLING_ADAPTIVE_BDF_TRACE_JSONL", str(trace_path))
    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_step
    )

    with pytest.raises(RuntimeError, match="synthetic implicit failure"):
        recycling._advance_recycling_1d_startup_step(
            _FakeConfig(),
            _fields(),
            runtime_model=_runtime_model(),
            feedback_integrals={"ctrl": 0.0},
            field_names=("N",),
            feedback_names=("ctrl",),
            mesh=object(),
            metrics=object(),
            dataset_scalars={},
            timestep=1.0,
            residual_tolerance=1.0e-8,
            max_nonlinear_iterations=20,
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-9,
            step_solver_mode="jax_linearized",
        )

    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "dt": 1.0,
            "event": "start",
            "step_solver_mode": "jax_linearized",
            "time": records[0]["time"],
            "trial_kind": "startup_full_backward_euler",
            "use_bdf2": False,
        }
    ]


def test_adaptive_be_interval_retries_then_accepts_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []
    error_ratios = iter((2.0, 0.05, 1.0))

    def fake_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        calls.append(float(timestep))
        return (
            {"N": fields["N"] + timestep},
            {"ctrl": feedback_integrals["ctrl"] + timestep},
            object(),
        )

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_step
    )
    monkeypatch.setattr(
        recycling,
        "_recycling_state_error_ratio",
        lambda *args, **kwargs: next(error_ratios),
    )

    fields, integrals, next_dt = recycling._advance_recycling_1d_adaptive_be_interval(
        _FakeConfig(rtol=1.0e-5, atol=1.0e-8),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        output_timestep=1.0,
        suggested_dt=1.0,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
    )

    assert calls[:3] == [1.0, 0.5, 0.5]
    assert fields["N"][0] > 1.0
    assert integrals["ctrl"] > 0.0
    assert next_dt >= 0.25


def test_adaptive_be_interval_accepts_minimum_dt_after_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def fake_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        calls.append(float(timestep))
        return (
            {"N": fields["N"] + timestep},
            {"ctrl": feedback_integrals["ctrl"] + timestep},
            object(),
        )

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_step
    )
    monkeypatch.setattr(
        recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: 5.0
    )

    fields, integrals, next_dt = recycling._advance_recycling_1d_adaptive_be_interval(
        _FakeConfig(rtol=1.0e-5, atol=1.0e-8),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        output_timestep=0.25,
        suggested_dt=0.25,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
    )

    assert calls == [0.25, 0.125, 0.125]
    np.testing.assert_allclose(fields["N"], np.array([1.25]))
    assert integrals["ctrl"] == 0.25
    assert next_dt == 0.25


def test_adaptive_bdf_interval_uses_bdf2_when_previous_step_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    be_calls: list[float] = []
    bdf_calls: list[float] = []

    def fake_be_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        be_calls.append(float(timestep))
        return (
            {"N": fields["N"] + 0.5},
            {"ctrl": feedback_integrals["ctrl"] + 0.5},
            object(),
        )

    def fake_bdf_step(
        config, fields, previous_fields, *, timestep, feedback_integrals, **kwargs
    ):
        bdf_calls.append(float(timestep))
        return (
            {"N": fields["N"] + 1.0},
            {"ctrl": feedback_integrals["ctrl"] + 1.0},
            object(),
        )

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_be_step
    )
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf_step)
    monkeypatch.setattr(
        recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: 0.6
    )

    (
        fields,
        integrals,
        previous_fields,
        previous_integrals,
        previous_dt,
        next_dt,
        stats,
    ) = recycling._advance_recycling_1d_adaptive_bdf_interval(
        _FakeConfig(rtol=1.0e-5, atol=1.0e-8),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        previous_fields=_fields(0.5),
        previous_integrals={"ctrl": -0.5},
        previous_dt=0.5,
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        output_timestep=0.5,
        suggested_dt=0.5,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
    )

    assert be_calls == [0.5]
    assert bdf_calls == [0.5]
    np.testing.assert_allclose(fields["N"], np.array([2.0]))
    assert integrals["ctrl"] == 1.0
    assert previous_fields is not None
    assert previous_integrals == {"ctrl": 0.0}
    assert previous_dt == 0.5
    assert next_dt == 0.5
    assert stats["adaptive_bdf_accepted_steps"] == 1
    assert stats["adaptive_bdf_rejected_steps"] == 0
    assert stats["adaptive_bdf_bdf2_trials"] == 1
    assert stats["adaptive_bdf_bdf2_accepted_steps"] == 1
    assert stats["adaptive_bdf_startup_trials"] == 0
    assert stats["adaptive_bdf_min_accepted_dt"] == 0.5
    assert stats["adaptive_bdf_max_accepted_dt"] == 0.5
    assert stats["adaptive_bdf_last_error_ratio"] == pytest.approx(0.2)
    assert stats["adaptive_bdf_last_accepted_error_ratio"] == pytest.approx(0.2)
    assert stats["adaptive_bdf_max_accepted_error_ratio"] == pytest.approx(0.2)


def test_adaptive_bdf_interval_reuses_sparse_jvp_workspace_across_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = object()
    received_workspaces: list[object | None] = []

    monkeypatch.setattr(
        recycling, "_build_recycling_sparse_jvp_workspace", lambda **kwargs: workspace
    )

    def fake_be_step(
        config,
        fields,
        *,
        timestep,
        feedback_integrals,
        sparse_jvp_workspace=None,
        **kwargs,
    ):
        received_workspaces.append(sparse_jvp_workspace)
        return (
            {"N": fields["N"] + 0.5},
            {"ctrl": feedback_integrals["ctrl"] + 0.5},
            SimpleNamespace(
                diagnostics={
                    "converged": True,
                    "rhs_backend": "fixed_full_field_array",
                    "jacobian_mode": "jvp",
                    "jvp_direction_workspace_reuses": 1,
                },
                linear_iterations=0,
            ),
        )

    def fake_bdf_step(
        config,
        fields,
        previous_fields,
        *,
        timestep,
        feedback_integrals,
        sparse_jvp_workspace=None,
        **kwargs,
    ):
        received_workspaces.append(sparse_jvp_workspace)
        return (
            {"N": fields["N"] + 1.0},
            {"ctrl": feedback_integrals["ctrl"] + 1.0},
            SimpleNamespace(
                diagnostics={
                    "converged": True,
                    "rhs_backend": "fixed_full_field_array",
                    "jacobian_mode": "jvp",
                    "jvp_direction_workspace_reuses": 1,
                },
                linear_iterations=0,
            ),
        )

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_be_step
    )
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf_step)
    monkeypatch.setattr(
        recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: 0.6
    )

    _, _, _, _, _, _, stats = recycling._advance_recycling_1d_adaptive_bdf_interval(
        _FakeConfig(rtol=1.0e-5, atol=1.0e-8),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        previous_fields=_fields(0.5),
        previous_integrals={"ctrl": -0.5},
        previous_dt=0.5,
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        output_timestep=0.5,
        suggested_dt=0.5,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
        step_solver_mode="sparse_jvp",
    )

    assert received_workspaces == [workspace, workspace]
    assert stats["adaptive_bdf_sparse_jvp_workspace_reuses"] == 2


def test_adaptive_bdf_interval_rejects_bdf2_above_promotion_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    be_calls: list[float] = []
    bdf_calls: list[float] = []
    startup_calls: list[float] = []

    def fake_be_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        be_calls.append(float(timestep))
        return (
            {"N": fields["N"] + 0.5},
            {"ctrl": feedback_integrals["ctrl"] + 0.5},
            object(),
        )

    def fake_bdf_step(
        config, fields, previous_fields, *, timestep, feedback_integrals, **kwargs
    ):
        bdf_calls.append(float(timestep))
        return (
            {"N": fields["N"] + 1.0},
            {"ctrl": feedback_integrals["ctrl"] + 1.0},
            object(),
        )

    def fake_startup_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        startup_calls.append(float(timestep))
        return (
            {"N": fields["N"] + timestep},
            {"ctrl": feedback_integrals["ctrl"] + timestep},
            0.5,
        )

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_be_step
    )
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf_step)
    monkeypatch.setattr(
        recycling, "_advance_recycling_1d_startup_step", fake_startup_step
    )
    bdf_error_ratios = iter((2.88, 0.3))
    monkeypatch.setattr(
        recycling,
        "_recycling_state_error_ratio",
        lambda *args, **kwargs: next(bdf_error_ratios),
    )

    (
        fields,
        integrals,
        previous_fields,
        previous_integrals,
        previous_dt,
        next_dt,
        stats,
    ) = recycling._advance_recycling_1d_adaptive_bdf_interval(
        _FakeConfig(rtol=1.0e-5, atol=1.0e-8),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        previous_fields=_fields(0.5),
        previous_integrals={"ctrl": -0.5},
        previous_dt=0.5,
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        output_timestep=0.5,
        suggested_dt=0.5,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
    )

    assert be_calls == [0.5, 0.25]
    assert bdf_calls == [0.5, 0.25]
    assert startup_calls == [0.25]
    np.testing.assert_allclose(fields["N"], np.array([2.25]))
    assert integrals["ctrl"] == 1.25
    assert previous_fields is not None
    assert previous_integrals == {"ctrl": 0.25}
    assert previous_dt == 0.25
    assert next_dt == 0.25
    assert stats["adaptive_bdf_rejected_steps"] == 1
    assert stats["adaptive_bdf_reused_history_after_rejection"] == 0
    assert stats["adaptive_bdf_accepted_steps"] == 2
    assert stats["adaptive_bdf_bdf2_accepted_steps"] == 1
    assert stats["adaptive_bdf_max_error_ratio"] == pytest.approx(0.96)
    assert stats["adaptive_bdf_max_accepted_error_ratio"] == pytest.approx(0.5)


def test_adaptive_bdf_interval_uses_startup_when_previous_state_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_calls: list[float] = []

    def fake_startup_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        startup_calls.append(float(timestep))
        return {"N": fields["N"] + 2.0}, {"ctrl": feedback_integrals["ctrl"] + 2.0}, 0.5

    monkeypatch.setattr(
        recycling, "_advance_recycling_1d_startup_step", fake_startup_step
    )

    (
        fields,
        integrals,
        previous_fields,
        previous_integrals,
        previous_dt,
        next_dt,
        stats,
    ) = recycling._advance_recycling_1d_adaptive_bdf_interval(
        _FakeConfig(rtol=1.0e-5, atol=1.0e-8),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        previous_fields=None,
        previous_integrals=None,
        previous_dt=None,
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        output_timestep=0.5,
        suggested_dt=0.5,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
    )

    assert startup_calls == [0.5]
    np.testing.assert_allclose(fields["N"], np.array([3.0]))
    assert integrals["ctrl"] == 2.0
    assert previous_fields is not None
    assert previous_integrals == {"ctrl": 0.0}
    assert previous_dt == 0.5
    assert next_dt == 0.5
    assert stats["adaptive_bdf_accepted_steps"] == 1
    assert stats["adaptive_bdf_rejected_steps"] == 0
    assert stats["adaptive_bdf_startup_trials"] == 1
    assert stats["adaptive_bdf_bdf2_trials"] == 0
    assert stats["adaptive_bdf_last_error_ratio"] == pytest.approx(0.5)
    assert stats["adaptive_bdf_last_accepted_error_ratio"] == pytest.approx(0.5)
    assert stats["adaptive_bdf_max_accepted_error_ratio"] == pytest.approx(0.5)


def test_adaptive_bdf_interval_falls_back_to_be_at_minimum_dt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    be_calls: list[float] = []
    bdf_calls: list[float] = []

    def fake_be_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        be_calls.append(float(timestep))
        return (
            {"N": fields["N"] + 0.5},
            {"ctrl": feedback_integrals["ctrl"] + 0.5},
            object(),
        )

    def fake_bdf_step(
        config, fields, previous_fields, *, timestep, feedback_integrals, **kwargs
    ):
        bdf_calls.append(float(timestep))
        return (
            {"N": fields["N"] + 10.0},
            {"ctrl": feedback_integrals["ctrl"] + 10.0},
            object(),
        )

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_be_step
    )
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf_step)
    monkeypatch.setattr(
        recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: 6.0
    )
    monkeypatch.setattr(
        recycling,
        "_adaptive_bdf_minimum_dt",
        lambda output_timestep: float(output_timestep),
    )

    (
        fields,
        integrals,
        previous_fields,
        previous_integrals,
        previous_dt,
        next_dt,
        stats,
    ) = recycling._advance_recycling_1d_adaptive_bdf_interval(
        _FakeConfig(rtol=1.0e-5, atol=1.0e-8),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        previous_fields=_fields(0.5),
        previous_integrals={"ctrl": -0.5},
        previous_dt=0.25,
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        output_timestep=0.25,
        suggested_dt=0.25,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
    )

    assert be_calls == [0.25]
    assert bdf_calls == [0.25]
    np.testing.assert_allclose(fields["N"], np.array([1.5]))
    assert integrals["ctrl"] == 0.5
    assert previous_fields is not None
    assert previous_integrals == {"ctrl": 0.0}
    assert previous_dt == 0.25
    assert next_dt == 0.25
    assert stats["adaptive_bdf_accepted_steps"] == 1
    assert stats["adaptive_bdf_minimum_dt_fallbacks"] == 1
    assert stats["adaptive_bdf_rejected_steps"] == 0
    assert stats["adaptive_bdf_bdf2_trials"] == 1
    assert stats["adaptive_bdf_bdf2_accepted_steps"] == 0
    assert stats["adaptive_bdf_last_error_ratio"] == pytest.approx(2.0)
    assert stats["adaptive_bdf_last_accepted_error_ratio"] == pytest.approx(2.0)
    assert stats["adaptive_bdf_max_accepted_error_ratio"] == pytest.approx(2.0)


def test_adaptive_bdf_interval_resets_sparse_history_after_rejected_nonminimum_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    be_calls: list[float] = []
    bdf_calls: list[float] = []
    startup_calls: list[float] = []

    def fake_be_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        be_calls.append(float(timestep))
        return (
            {"N": fields["N"] + 0.5},
            {"ctrl": feedback_integrals["ctrl"] + 0.5},
            object(),
        )

    def fake_bdf_step(
        config, fields, previous_fields, *, timestep, feedback_integrals, **kwargs
    ):
        bdf_calls.append(float(timestep))
        return (
            {"N": fields["N"] + 10.0},
            {"ctrl": feedback_integrals["ctrl"] + 10.0},
            object(),
        )

    def fake_startup_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        startup_calls.append(float(timestep))
        return (
            {"N": fields["N"] + timestep},
            {"ctrl": feedback_integrals["ctrl"] + timestep},
            0.5,
        )

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_be_step
    )
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf_step)
    monkeypatch.setattr(
        recycling, "_advance_recycling_1d_startup_step", fake_startup_step
    )
    bdf_error_ratios = iter((6.0, 0.3))
    monkeypatch.setattr(
        recycling,
        "_recycling_state_error_ratio",
        lambda *args, **kwargs: next(bdf_error_ratios),
    )
    monkeypatch.setattr(
        recycling, "_choose_recycling_next_dt", lambda *args, **kwargs: 1.0
    )

    (
        fields,
        integrals,
        previous_fields,
        previous_integrals,
        previous_dt,
        next_dt,
        stats,
    ) = recycling._advance_recycling_1d_adaptive_bdf_interval(
        _FakeConfig(rtol=1.0e-5, atol=1.0e-8),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        previous_fields=_fields(0.5),
        previous_integrals={"ctrl": -0.5},
        previous_dt=0.5,
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        output_timestep=0.5,
        suggested_dt=0.5,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
    )

    assert be_calls == [0.5, 0.25]
    assert bdf_calls == [0.5, 0.25]
    assert startup_calls == [0.25]
    np.testing.assert_allclose(fields["N"], np.array([11.25]))
    assert integrals["ctrl"] == 10.25
    assert previous_fields is not None
    assert previous_integrals == {"ctrl": 0.25}
    assert previous_dt == 0.25
    assert next_dt == 0.5
    assert stats["adaptive_bdf_accepted_steps"] == 2
    assert stats["adaptive_bdf_rejected_steps"] == 1
    assert stats["adaptive_bdf_reused_history_after_rejection"] == 0
    assert stats["adaptive_bdf_startup_trials"] == 1
    assert stats["adaptive_bdf_bdf2_trials"] == 2
    assert stats["adaptive_bdf_bdf2_accepted_steps"] == 1
    assert stats["adaptive_bdf_max_error_ratio"] == pytest.approx(2.0)
    assert stats["adaptive_bdf_last_error_ratio"] == pytest.approx(0.1)
    assert stats["adaptive_bdf_max_accepted_error_ratio"] == pytest.approx(0.5)
    assert stats["adaptive_bdf_last_accepted_error_ratio"] == pytest.approx(0.1)
    assert stats["adaptive_bdf_min_accepted_dt"] == 0.25
    assert stats["adaptive_bdf_max_accepted_dt"] == 0.25


def test_adaptive_bdf_interval_reuses_jax_history_after_rejected_nonminimum_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    be_calls: list[float] = []
    bdf_calls: list[float] = []
    startup_calls: list[float] = []

    def fake_be_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        be_calls.append(float(timestep))
        return (
            {"N": fields["N"] + 0.5},
            {"ctrl": feedback_integrals["ctrl"] + 0.5},
            object(),
        )

    def fake_bdf_step(
        config, fields, previous_fields, *, timestep, feedback_integrals, **kwargs
    ):
        bdf_calls.append(float(timestep))
        return (
            {"N": fields["N"] + 10.0},
            {"ctrl": feedback_integrals["ctrl"] + 10.0},
            object(),
        )

    def fake_startup_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        startup_calls.append(float(timestep))
        return (
            {"N": fields["N"] + timestep},
            {"ctrl": feedback_integrals["ctrl"] + timestep},
            0.5,
        )

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_be_step
    )
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf_step)
    monkeypatch.setattr(
        recycling, "_advance_recycling_1d_startup_step", fake_startup_step
    )
    bdf_error_ratios = iter((6.0, 0.3, 0.3))
    monkeypatch.setattr(
        recycling,
        "_recycling_state_error_ratio",
        lambda *args, **kwargs: next(bdf_error_ratios),
    )
    monkeypatch.setattr(
        recycling, "_choose_recycling_next_dt", lambda *args, **kwargs: 1.0
    )

    (
        fields,
        integrals,
        previous_fields,
        previous_integrals,
        previous_dt,
        next_dt,
        stats,
    ) = recycling._advance_recycling_1d_adaptive_bdf_interval(
        _FakeConfig(rtol=1.0e-5, atol=1.0e-8),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        previous_fields=_fields(0.5),
        previous_integrals={"ctrl": -0.5},
        previous_dt=0.5,
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        output_timestep=0.5,
        suggested_dt=0.5,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
        step_solver_mode="jax_linearized",
    )

    assert be_calls == [0.5, 0.25, 0.25]
    assert bdf_calls == [0.5, 0.25, 0.25]
    assert startup_calls == []
    np.testing.assert_allclose(fields["N"], np.array([21.0]))
    assert integrals["ctrl"] == 20.0
    assert previous_fields is not None
    np.testing.assert_allclose(previous_fields["N"], np.array([11.0]))
    assert previous_integrals == {"ctrl": 10.0}
    assert previous_dt == 0.25
    assert next_dt == 0.5
    assert stats["adaptive_bdf_accepted_steps"] == 2
    assert stats["adaptive_bdf_rejected_steps"] == 1
    assert stats["adaptive_bdf_reused_history_after_rejection"] == 1
    assert stats["adaptive_bdf_startup_trials"] == 0
    assert stats["adaptive_bdf_bdf2_trials"] == 3
    assert stats["adaptive_bdf_bdf2_accepted_steps"] == 2
    assert stats["adaptive_bdf_max_error_ratio"] == pytest.approx(2.0)
    assert stats["adaptive_bdf_last_error_ratio"] == pytest.approx(0.1)
    assert stats["adaptive_bdf_max_accepted_error_ratio"] == pytest.approx(0.1)
    assert stats["adaptive_bdf_last_accepted_error_ratio"] == pytest.approx(0.1)


def test_adaptive_bdf_interval_preserves_previous_state_when_next_dt_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_timesteps: list[float | None] = []

    def fake_be_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        return (
            {"N": fields["N"] + 0.5},
            {"ctrl": feedback_integrals["ctrl"] + 0.5},
            object(),
        )

    def fake_bdf_step(
        config, fields, previous_fields, *, timestep, feedback_integrals, **kwargs
    ):
        previous_timesteps.append(kwargs.get("previous_timestep"))
        return (
            {"N": fields["N"] + 1.0},
            {"ctrl": feedback_integrals["ctrl"] + 1.0},
            object(),
        )

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_be_step
    )
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf_step)
    monkeypatch.setattr(
        recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: 0.3
    )
    monkeypatch.setattr(
        recycling, "_choose_recycling_next_dt", lambda *args, **kwargs: 0.25
    )

    (
        fields,
        integrals,
        previous_fields,
        previous_integrals,
        previous_dt,
        next_dt,
        stats,
    ) = recycling._advance_recycling_1d_adaptive_bdf_interval(
        _FakeConfig(rtol=1.0e-5, atol=1.0e-8),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        previous_fields=_fields(0.5),
        previous_integrals={"ctrl": -0.5},
        previous_dt=0.5,
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        output_timestep=0.5,
        suggested_dt=0.5,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
    )

    np.testing.assert_allclose(fields["N"], np.array([2.0]))
    assert integrals["ctrl"] == 1.0
    assert previous_fields is not None
    assert previous_integrals == {"ctrl": 0.0}
    assert previous_dt == 0.5
    assert next_dt == 0.25
    assert stats["adaptive_bdf_accepted_steps"] == 1
    assert stats["adaptive_bdf_rejected_steps"] == 0
    assert stats["adaptive_bdf_bdf2_trials"] == 1
    assert stats["adaptive_bdf_bdf2_accepted_steps"] == 1
    assert stats["adaptive_bdf_last_error_ratio"] == pytest.approx(0.1)
    assert previous_timesteps == [0.5]


def test_startup_step_uses_full_and_two_half_be_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def fake_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        calls.append(float(timestep))
        return (
            {"N": fields["N"] + timestep},
            {"ctrl": feedback_integrals["ctrl"] + timestep},
            object(),
        )

    monkeypatch.setattr(
        recycling, "advance_recycling_1d_backward_euler_step", fake_step
    )
    monkeypatch.setattr(
        recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: 0.125
    )

    fields, integrals, error_ratio = recycling._advance_recycling_1d_startup_step(
        _FakeConfig(),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        timestep=1.0,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=20,
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-9,
    )

    assert calls == [1.0, 0.5, 0.5]
    np.testing.assert_allclose(fields["N"], np.array([2.0]))
    assert integrals["ctrl"] == 1.0
    assert error_ratio == 0.125


def test_recycling_state_error_ratio_includes_fields_and_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recycling, "_recycling_active_domain_slices", lambda mesh: slice(None)
    )

    ratio = recycling._recycling_state_error_ratio(
        {"N": np.array([1.0, 3.0])},
        {"ctrl": 1.0},
        {"N": np.array([2.0, 3.0])},
        {"ctrl": 2.0},
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        relative_tolerance=1.0,
        absolute_tolerance=0.0,
    )

    assert ratio == pytest.approx(np.sqrt((0.5**2 + 0.0**2 + 0.5**2) / 3.0))
    assert (
        recycling._recycling_state_error_ratio(
            {},
            {},
            {},
            {},
            field_names=(),
            feedback_names=(),
            mesh=object(),
            relative_tolerance=1.0,
            absolute_tolerance=0.0,
        )
        == 0.0
    )


def test_recycling_state_error_ratio_supports_field_absolute_tolerance_floors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recycling, "_recycling_active_domain_slices", lambda mesh: slice(None)
    )

    unfloored = recycling._recycling_state_error_ratio(
        {"NVd+": np.array([0.0, 0.0]), "Pe": np.array([1.0, 1.0])},
        {},
        {"NVd+": np.array([1.0e-3, 0.0]), "Pe": np.array([1.0, 1.0])},
        {},
        field_names=("NVd+", "Pe"),
        feedback_names=(),
        mesh=object(),
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-12,
    )
    floored = recycling._recycling_state_error_ratio(
        {"NVd+": np.array([0.0, 0.0]), "Pe": np.array([1.0, 1.0])},
        {},
        {"NVd+": np.array([1.0e-3, 0.0]), "Pe": np.array([1.0, 1.0])},
        {},
        field_names=("NVd+", "Pe"),
        feedback_names=(),
        mesh=object(),
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-12,
        field_absolute_tolerance_floors={"NVd+": 1.0e-2},
    )

    assert unfloored > 1.0e4
    assert floored < unfloored / 100.0


def test_recycling_state_error_contributors_rank_fields_and_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recycling, "_recycling_active_domain_slices", lambda mesh: slice(None)
    )

    contributors = recycling._recycling_state_error_contributors(
        {"N": np.array([1.0, 3.0]), "Pe": np.array([4.0, 4.0])},
        {"ctrl": 1.0},
        {"N": np.array([2.0, 3.0]), "Pe": np.array([5.0, 4.0])},
        {"ctrl": 4.0},
        field_names=("N", "Pe"),
        feedback_names=("ctrl",),
        mesh=object(),
        relative_tolerance=1.0,
        absolute_tolerance=0.0,
    )

    assert contributors["component_count"] == 5
    assert contributors["overall_ratio"] == pytest.approx(
        np.sqrt((0.5**2 + 0.0**2 + 0.2**2 + 0.0**2 + 0.75**2) / 5.0)
    )
    assert contributors["dominant"]["name"] == "ctrl"
    assert [entry["name"] for entry in contributors["fields"]] == ["N", "Pe"]
    assert contributors["fields"][0]["rms_ratio"] == pytest.approx(
        np.sqrt((0.5**2 + 0.0**2) / 2.0)
    )
    assert contributors["fields"][0]["rms_difference"] == pytest.approx(
        np.sqrt((1.0**2 + 0.0**2) / 2.0)
    )
    assert contributors["fields"][0]["min_scale"] == pytest.approx(2.0)
    assert contributors["fields"][0]["max_scale"] == pytest.approx(3.0)
    assert contributors["fields"][1]["max_abs_ratio"] == pytest.approx(0.2)
    assert contributors["fields"][1]["max_abs_difference"] == pytest.approx(1.0)
    assert contributors["fields"][1]["mean_scale"] == pytest.approx(4.5)
    assert contributors["feedback"][0]["rms_ratio"] == pytest.approx(0.75)
    assert contributors["feedback"][0]["max_abs_difference"] == pytest.approx(3.0)
    assert contributors["feedback"][0]["min_scale"] == pytest.approx(4.0)


def test_recycling_state_error_contributors_apply_field_atol_floors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recycling, "_recycling_active_domain_slices", lambda mesh: slice(None)
    )

    contributors = recycling._recycling_state_error_contributors(
        {"NVd+": np.array([0.0, 0.0])},
        {},
        {"NVd+": np.array([1.0e-3, 0.0])},
        {},
        field_names=("NVd+",),
        feedback_names=(),
        mesh=object(),
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-12,
        field_absolute_tolerance_floors={"NVd+": 1.0e-2},
    )

    assert contributors["fields"][0]["min_scale"] == pytest.approx(1.0e-2)
    assert contributors["fields"][0]["rms_ratio"] == pytest.approx(
        np.sqrt((0.1**2 + 0.0**2) / 2.0)
    )


def test_scale_adaptive_bdf_error_contributors_matches_embedded_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recycling, "_recycling_active_domain_slices", lambda mesh: slice(None)
    )
    contributors = recycling._recycling_state_error_contributors(
        {"N": np.array([1.0, 3.0])},
        {"ctrl": 1.0},
        {"N": np.array([2.0, 3.0])},
        {"ctrl": 2.0},
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        relative_tolerance=1.0,
        absolute_tolerance=0.0,
    )

    scaled = recycling._scale_adaptive_bdf_error_contributors(contributors, 1.0 / 3.0)

    assert scaled["overall_ratio"] == pytest.approx(contributors["overall_ratio"] / 3.0)
    assert scaled["dominant"]["name"] == "ctrl"
    assert scaled["fields"][0]["rms_ratio"] == pytest.approx(
        contributors["fields"][0]["rms_ratio"] / 3.0
    )
    assert scaled["fields"][0]["rms_difference"] == pytest.approx(
        contributors["fields"][0]["rms_difference"] / 3.0
    )
    assert scaled["fields"][0]["min_scale"] == pytest.approx(
        contributors["fields"][0]["min_scale"]
    )
    assert scaled["feedback"][0]["squared_error_sum"] == pytest.approx(
        contributors["feedback"][0]["squared_error_sum"] / 9.0
    )
    assert scaled["feedback"][0]["max_abs_difference"] == pytest.approx(
        contributors["feedback"][0]["max_abs_difference"] / 3.0
    )


def test_json_ready_adaptive_bdf_trace_value_normalizes_nested_diagnostics() -> None:
    class CustomDiagnostic:
        def __str__(self) -> str:
            return "custom-diagnostic"

    value = recycling._json_ready_adaptive_bdf_trace_value(
        {
            "scalar": np.float64(1.25),
            "array": np.array([1, 2], dtype=np.int64),
            "nonfinite": float("inf"),
            "tuple": (np.int64(3), float("nan")),
            "fallback": CustomDiagnostic(),
        }
    )

    assert value == {
        "scalar": 1.25,
        "array": [1, 2],
        "nonfinite": None,
        "tuple": [3, None],
        "fallback": "custom-diagnostic",
    }


def test_write_adaptive_bdf_trace_record_records_solver_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    trace_path = tmp_path / "nested" / "trace.jsonl"
    monkeypatch.setenv("JAX_DRB_RECYCLING_ADAPTIVE_BDF_TRACE_JSONL", str(trace_path))

    recycling._write_adaptive_bdf_trace_record(
        event="end",
        trial_kind="bdf2_corrector",
        dt=0.125,
        use_bdf2=True,
        step_solver_mode="sparse_jvp",
        elapsed_seconds=1.5,
        error_ratio=float("nan"),
        info=SimpleNamespace(
            residual_inf_norm=float("inf"),
            nonlinear_iterations=4,
            linear_iterations=9,
            diagnostics={
                "rhs_backend": "fixed_full_field_array",
                "jacobian_mode": "jvp",
                "converged": True,
                "residual_evaluation_count": 3,
                "residual_evaluation_seconds": 0.25,
                "jacobian_refresh_count": 2,
                "jacobian_assembly_seconds": 0.5,
                "linear_solve_seconds": 0.75,
                "line_search_seconds": 0.125,
                "linear_solver_backend": "scipy_spsolve",
                "linear_solver_status": "ok",
                "linear_solver_success": False,
                "linear_solver_reported_iterations": 7,
                "jvp_direction_batch_count": 5,
                "jvp_direction_build_seconds": 0.01,
                "jvp_jacobian_total_seconds": 0.2,
                "jvp_jacobian_linearize_seconds": 0.03,
                "jvp_jacobian_tangent_build_seconds": 0.04,
                "jvp_jacobian_push_seconds": 0.05,
                "jvp_jacobian_device_execute_seconds": 0.02,
                "jvp_jacobian_host_transfer_seconds": 0.03,
                "jvp_jacobian_sparse_assembly_seconds": 0.06,
                "jvp_jacobian_batch_count": 5,
                "jvp_jacobian_prebuilt_direction_batch_uses": 1,
                "jvp_direction_workspace_reuses": 1,
            },
        ),
    )

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["error_ratio"] is None
    assert record["residual_inf_norm"] is None
    assert record["nonlinear_iterations"] == 4
    assert record["linear_iterations"] == 9
    assert record["rhs_backend"] == "fixed_full_field_array"
    assert record["jacobian_mode"] == "jvp"
    assert record["linear_solver_success"] is False
    assert record["jvp_jacobian_device_execute_seconds"] == 0.02
    assert record["jvp_jacobian_host_transfer_seconds"] == 0.03
    assert record["jvp_jacobian_prebuilt_direction_batch_uses"] == 1
    assert record["jvp_direction_workspace_reuses"] == 1


def test_record_adaptive_bdf_error_ratios_ignore_nonfinite_maxima() -> None:
    stats = recycling._new_adaptive_bdf_interval_stats("sparse_jvp")

    recycling._record_adaptive_bdf_error_ratio(stats, float("nan"))
    recycling._record_adaptive_bdf_accepted_error_ratio(stats, float("inf"))
    assert stats["adaptive_bdf_last_error_ratio"] is None
    assert stats["adaptive_bdf_max_error_ratio"] is None
    assert stats["adaptive_bdf_last_accepted_error_ratio"] is None
    assert stats["adaptive_bdf_max_accepted_error_ratio"] is None

    recycling._record_adaptive_bdf_error_ratio(stats, 0.4)
    recycling._record_adaptive_bdf_error_ratio(stats, 0.2)
    recycling._record_adaptive_bdf_error_ratio(stats, 0.8)
    recycling._record_adaptive_bdf_accepted_error_ratio(stats, 0.3)
    recycling._record_adaptive_bdf_accepted_error_ratio(stats, 0.1)
    recycling._record_adaptive_bdf_accepted_error_ratio(stats, 0.6)
    assert stats["adaptive_bdf_last_error_ratio"] == 0.8
    assert stats["adaptive_bdf_max_error_ratio"] == 0.8
    assert stats["adaptive_bdf_last_accepted_error_ratio"] == 0.6
    assert stats["adaptive_bdf_max_accepted_error_ratio"] == 0.6


def test_adaptive_bdf_error_contributors_if_tracing_respects_trace_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        recycling, "_recycling_active_domain_slices", lambda mesh: slice(None)
    )

    without_trace = recycling._adaptive_bdf_error_contributors_if_tracing(
        {"N": np.array([0.0])},
        {},
        {"N": np.array([1.0])},
        {},
        field_names=("N",),
        feedback_names=(),
        mesh=object(),
        relative_tolerance=1.0,
        absolute_tolerance=1.0e-12,
    )
    assert without_trace is None

    monkeypatch.setenv(
        "JAX_DRB_RECYCLING_ADAPTIVE_BDF_TRACE_JSONL", str(tmp_path / "trace.jsonl")
    )
    contributors = recycling._adaptive_bdf_error_contributors_if_tracing(
        {"NVd+": np.array([0.0]), "Pe": np.array([0.0]), "Nd": np.array([0.0])},
        {},
        {
            "NVd+": np.array([1.0e-3]),
            "Pe": np.array([1.0e-4]),
            "Nd": np.array([1.0e-7]),
        },
        {},
        field_names=("NVd+", "Pe", "Nd"),
        feedback_names=(),
        mesh=object(),
        relative_tolerance=0.0,
        absolute_tolerance=1.0e-12,
        field_absolute_tolerance_floors={"NVd+": 1.0e-2, "Pe": 1.0e-3, "Nd": 1.0e-6},
    )

    assert contributors is not None
    by_name = {entry["name"]: entry for entry in contributors["fields"]}
    assert by_name["NVd+"]["min_scale"] == pytest.approx(1.0e-2)
    assert by_name["Pe"]["min_scale"] == pytest.approx(1.0e-3)
    assert by_name["Nd"]["min_scale"] == pytest.approx(1.0e-6)
    assert contributors["overall_ratio"] == pytest.approx(0.1)


def test_scale_adaptive_bdf_error_contributors_handles_none_nonfinite_and_missing_differences() -> (
    None
):
    assert recycling._scale_adaptive_bdf_error_contributors(None, 0.5) is None

    contributors = {
        "overall_ratio": float("inf"),
        "component_count": 1,
        "fields": [
            {
                "name": "N",
                "rms_ratio": float("inf"),
                "max_abs_ratio": 2.0,
                "mean_abs_ratio": 1.0,
                "squared_error_sum": float("inf"),
            }
        ],
        "feedback": [],
    }

    scaled = recycling._scale_adaptive_bdf_error_contributors(contributors, 0.25)
    assert scaled["overall_ratio"] == float("inf")
    assert scaled["dominant"]["name"] == "N"
    assert scaled["fields"][0]["rms_ratio"] == float("inf")
    assert scaled["fields"][0]["max_abs_ratio"] == pytest.approx(0.5)
    assert "rms_difference" not in scaled["fields"][0]


def test_recycling_state_error_contributors_reports_nonfinite_and_empty_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recycling, "_recycling_active_domain_slices", lambda mesh: slice(None)
    )

    with pytest.warns(RuntimeWarning, match="invalid value encountered"):
        contributors = recycling._recycling_state_error_contributors(
            {"N": np.array([1.0, 0.0]), "Empty": np.array([], dtype=np.float64)},
            {"ctrl": 1.0},
            {"N": np.array([np.inf, 1.0]), "Empty": np.array([], dtype=np.float64)},
            {"ctrl": np.inf},
            field_names=("N", "Empty"),
            feedback_names=("ctrl",),
            mesh=object(),
            relative_tolerance=1.0,
            absolute_tolerance=0.0,
        )

    by_name = {entry["name"]: entry for entry in contributors["fields"]}
    assert contributors["overall_ratio"] == float("inf")
    assert contributors["dominant"]["name"] == "N"
    assert by_name["N"]["nonfinite_count"] == 1
    assert by_name["Empty"]["component_count"] == 0
    assert by_name["Empty"]["min_scale"] == float("inf")
    assert contributors["feedback"][0]["nonfinite_count"] == 1
    assert contributors["feedback"][0]["rms_difference"] == float("inf")


def test_adaptive_bdf_trace_records_error_contributors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("JAX_DRB_RECYCLING_ADAPTIVE_BDF_TRACE_JSONL", str(trace_path))

    recycling._write_adaptive_bdf_trace_record(
        event="error_estimate",
        trial_kind="startup_embedded_difference",
        dt=0.25,
        use_bdf2=False,
        step_solver_mode="sparse_jvp",
        error_ratio=0.5,
        error_contributors={
            "overall_ratio": 0.5,
            "component_count": 1,
            "dominant": {"name": "Pe", "rms_ratio": 0.5},
            "fields": [{"name": "Pe", "rms_ratio": 0.5}],
            "feedback": [],
        },
    )

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["error_contributors"]["overall_ratio"] == 0.5
    assert record["error_contributors"]["dominant"]["name"] == "Pe"


def test_resolve_recycling_adaptive_bdf_momentum_atol_floor_prefers_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX_DRB_RECYCLING_ADAPTIVE_BDF_MOMENTUM_ATOL_FLOOR", "0.5")
    monkeypatch.setenv("JAX_DRB_RECYCLING_ADAPTIVE_BDF_DENSITY_ATOL_FLOOR", "0.05")
    monkeypatch.setenv("JAX_DRB_RECYCLING_ADAPTIVE_BDF_PRESSURE_ATOL_FLOOR", "0.005")
    config = _FakeRuntimeConfig(
        recycling_adaptive_bdf_momentum_atol_floor=0.25,
        recycling_adaptive_bdf_density_atol_floor=0.025,
        recycling_adaptive_bdf_pressure_atol_floor=0.0025,
    )

    assert recycling._resolve_recycling_adaptive_bdf_momentum_atol_floor(config) == 0.25
    assert recycling._resolve_recycling_adaptive_bdf_density_atol_floor(config) == 0.025
    assert (
        recycling._resolve_recycling_adaptive_bdf_pressure_atol_floor(config) == 0.0025
    )
    assert recycling._resolve_recycling_adaptive_bdf_field_atol_floors(
        config, ("NVd+", "Pe", "NVt", "Nd")
    ) == {
        "NVd+": 0.25,
        "NVt": 0.25,
        "Pe": 0.0025,
        "Nd": 0.025,
    }


@pytest.mark.parametrize(
    ("env_name", "resolver"),
    [
        (
            "JAX_DRB_RECYCLING_ADAPTIVE_BDF_MOMENTUM_ATOL_FLOOR",
            recycling._resolve_recycling_adaptive_bdf_momentum_atol_floor,
        ),
        (
            "JAX_DRB_RECYCLING_ADAPTIVE_BDF_DENSITY_ATOL_FLOOR",
            recycling._resolve_recycling_adaptive_bdf_density_atol_floor,
        ),
        (
            "JAX_DRB_RECYCLING_ADAPTIVE_BDF_PRESSURE_ATOL_FLOOR",
            recycling._resolve_recycling_adaptive_bdf_pressure_atol_floor,
        ),
    ],
)
def test_resolve_recycling_adaptive_bdf_component_atol_floor_env_fallback(
    env_name: str,
    resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env_name, "0.125")
    assert resolver(None) == 0.125
    monkeypatch.setenv(env_name, "bad")
    assert resolver(None) is None


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("bad", None),
        (None, None),
        (-1.0, None),
        (float("nan"), None),
        (0.25, 0.25),
    ],
)
def test_resolve_recycling_adaptive_bdf_component_atol_floor_config_validation(
    configured: object,
    expected: float | None,
) -> None:
    config = _FakeRuntimeConfig(recycling_adaptive_bdf_density_atol_floor=configured)
    assert (
        recycling._resolve_recycling_adaptive_bdf_density_atol_floor(config) == expected
    )


@pytest.mark.parametrize("env_value", ["", "   ", "0.0", "-1.0", "nan"])
def test_resolve_recycling_adaptive_bdf_component_atol_floor_rejects_blank_or_nonpositive_env(
    env_value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX_DRB_RECYCLING_ADAPTIVE_BDF_PRESSURE_ATOL_FLOOR", env_value)
    assert recycling._resolve_recycling_adaptive_bdf_pressure_atol_floor(None) is None


def test_resolve_recycling_fixed_bdf2_max_internal_timestep_prefers_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX_DRB_RECYCLING_FIXED_BDF2_MAX_INTERNAL_TIMESTEP", "0.25")
    config = _FakeRuntimeConfig(recycling_fixed_bdf2_max_internal_timestep=0.5)

    assert recycling._resolve_recycling_fixed_bdf2_max_internal_timestep(config) == 0.5


def test_resolve_recycling_fixed_bdf2_max_internal_timestep_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX_DRB_RECYCLING_FIXED_BDF2_MAX_INTERNAL_TIMESTEP", "0.125")
    assert recycling._resolve_recycling_fixed_bdf2_max_internal_timestep(None) == 0.125

    monkeypatch.setenv("JAX_DRB_RECYCLING_FIXED_BDF2_MAX_INTERNAL_TIMESTEP", "bad")
    assert recycling._resolve_recycling_fixed_bdf2_max_internal_timestep(None) is None


@pytest.mark.parametrize("value", ["", "   ", "0.0", "-1.0", "nan"])
def test_resolve_recycling_fixed_bdf2_max_internal_timestep_rejects_invalid_env(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX_DRB_RECYCLING_FIXED_BDF2_MAX_INTERNAL_TIMESTEP", value)
    assert recycling._resolve_recycling_fixed_bdf2_max_internal_timestep(None) is None


def test_initial_recycling_continuation_dt_uses_field_count_cutover() -> None:
    assert (
        recycling._initial_recycling_continuation_dt(
            _runtime_model(field_names=("N",)), timestep=250.0
        )
        == 100.0
    )
    assert (
        recycling._initial_recycling_continuation_dt(
            _runtime_model(field_names=tuple(f"f{i}" for i in range(11))),
            timestep=250.0,
        )
        == 25.0
    )
    assert (
        recycling._initial_recycling_continuation_dt(
            _runtime_model(field_names=tuple(f"f{i}" for i in range(11))), timestep=10.0
        )
        == 10.0
    )


def test_bdf_history_raises_on_failed_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_solve_ivp(*args, **kwargs):
        return SimpleNamespace(
            success=False, message="linear solve failed", y=np.zeros((2, 1))
        )

    monkeypatch.setattr("scipy.integrate.solve_ivp", fake_solve_ivp)
    monkeypatch.setattr(recycling, "_recycling_active_shape", lambda mesh: (1,))
    monkeypatch.setattr(
        recycling,
        "_build_recycling_residual_sparsity",
        lambda **kwargs: _FakeSparsity(),
    )
    monkeypatch.setattr(recycling, "_build_recycling_color_groups", lambda **kwargs: ())
    layout = SimpleNamespace(
        field_names=("N",),
        feedback_names=("ctrl",),
        active_slices=(slice(None),),
        active_shape=(1,),
        field_size=1,
        field_templates=(np.array([1.0], dtype=np.float64),),
    )
    monkeypatch.setattr(
        recycling, "_build_recycling_packed_state_layout", lambda **kwargs: layout
    )
    monkeypatch.setattr(
        recycling,
        "_pack_recycling_active_state",
        lambda *args, **kwargs: np.array([1.0, 0.0]),
    )

    with pytest.raises(RuntimeError, match="linear solve failed"):
        recycling._advance_recycling_1d_bdf_history(
            _FakeConfig(rtol=1.0e-5, atol=1.0e-8),
            _fields(),
            runtime_model=_runtime_model(),
            feedback_integrals={"ctrl": 0.0},
            field_names=("N",),
            feedback_names=("ctrl",),
            mesh=object(),
            metrics=object(),
            dataset_scalars={},
            timestep=0.5,
            steps=1,
        )


def test_bdf_history_unpacks_sanitizes_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []
    rhs_call_count = 0
    captured_parallel_workers: list[int] = []

    def fake_solve_ivp(rhs, time_span, y0, **kwargs):
        np.testing.assert_allclose(kwargs["t_eval"], np.array([0.0, 0.5, 1.0]))
        assert kwargs["method"] == "BDF"
        np.testing.assert_allclose(rhs(0.0, y0), np.zeros(2))
        assert kwargs["jac"](0.0, y0) == "jacobian"
        return SimpleNamespace(
            success=True,
            message="ok",
            y=np.array(
                [
                    [1.0, 2.0, 3.0],
                    [0.0, 0.5, 1.0],
                ],
                dtype=np.float64,
            ),
        )

    def fake_unpack(packed_state, **kwargs):
        return {"N": np.array([packed_state[0]])}, {"ctrl": float(packed_state[1])}

    monkeypatch.setattr("scipy.integrate.solve_ivp", fake_solve_ivp)
    monkeypatch.setattr(recycling, "_recycling_active_shape", lambda mesh: (1,))
    monkeypatch.setattr(
        recycling,
        "_build_recycling_residual_sparsity",
        lambda **kwargs: _FakeSparsity(),
    )
    monkeypatch.setattr(recycling, "_build_recycling_color_groups", lambda **kwargs: ())
    layout = SimpleNamespace(
        field_names=("N",),
        feedback_names=("ctrl",),
        active_slices=(slice(None),),
        active_shape=(1,),
        field_size=1,
        field_templates=(np.array([1.0], dtype=np.float64),),
    )
    monkeypatch.setattr(
        recycling, "_build_recycling_packed_state_layout", lambda **kwargs: layout
    )
    monkeypatch.setattr(
        recycling,
        "_pack_recycling_active_state",
        lambda *args, **kwargs: np.array([1.0, 0.0]),
    )
    monkeypatch.setattr(recycling, "_unpack_recycling_active_state", fake_unpack)
    monkeypatch.setenv("JAX_DRB_FD_JACOBIAN_THREADS", "2")

    def fake_packed_rhs(*args, **kwargs):
        nonlocal rhs_call_count
        rhs_call_count += 1
        return np.zeros(2)

    monkeypatch.setattr(recycling, "_compute_recycling_1d_packed_rhs", fake_packed_rhs)

    def fake_build_jacobian(*args, parallel_workers: int, **kwargs):
        captured_parallel_workers.append(parallel_workers)
        return "jacobian"

    monkeypatch.setattr(
        recycling, "build_sparse_difference_quotient_jacobian", fake_build_jacobian
    )
    monkeypatch.setattr(
        recycling, "_sanitize_recycling_fields", lambda config, fields: fields
    )
    monkeypatch.setattr(
        recycling, "_sanitize_feedback_integrals", lambda integrals, **kwargs: integrals
    )
    monkeypatch.setattr(
        recycling,
        "_build_recycling_progress_details",
        lambda **kwargs: (
            {
                "interval_index": kwargs["interval_index"],
                "solver_mode": kwargs["solver_mode"],
            },
            0.0,
        ),
    )

    result = recycling._advance_recycling_1d_bdf_history(
        _FakeConfig(rtol=1.0e-5, atol=1.0e-8),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        timestep=0.5,
        steps=2,
        progress_callback=events.append,
    )

    np.testing.assert_allclose(
        result.variable_history["N"][:, 0], np.array([1.0, 2.0, 3.0])
    )
    np.testing.assert_allclose(
        result.feedback_integral_history["ctrl"], np.array([0.0, 0.5, 1.0])
    )
    assert events == [
        {"interval_index": 1, "solver_mode": "bdf"},
        {"interval_index": 2, "solver_mode": "bdf"},
    ]
    assert rhs_call_count == 1
    assert captured_parallel_workers == [2]
    assert result.diagnostics["bdf_rhs_evaluation_count"] == 1
    assert result.diagnostics["bdf_rhs_cache_hit_count"] == 1
    assert result.diagnostics["bdf_rhs_callback_seconds"] >= 0.0
    assert result.diagnostics["bdf_rhs_evaluation_seconds"] >= 0.0
    assert result.diagnostics["bdf_rhs_object_evaluation_seconds"] >= 0.0
    assert result.diagnostics["bdf_rhs_numpy_conversion_seconds"] >= 0.0
    assert result.diagnostics["bdf_jacobian_callback_count"] == 1
    assert result.diagnostics["bdf_jacobian_callback_seconds"] >= 0.0
    assert result.diagnostics["bdf_jacobian_base_rhs_evaluation_count"] == 1
    assert result.diagnostics["bdf_jvp_rhs_evaluation_count"] == 0
    assert result.diagnostics["bdf_jacobian_mode"] == "fd"
    assert result.diagnostics["bdf_rhs_backend"] == "host_bridge"
    assert result.diagnostics["bdf_jvp_batch_size"] is None
    assert result.diagnostics["bdf_jacobian_parallel_workers"] == 2
    assert result.diagnostics["bdf_solve_seconds"] >= 0.0
    assert result.diagnostics["bdf_active_size"] == 2
    assert result.diagnostics["bdf_sparse_nnz"] == 0
    assert result.diagnostics["bdf_color_group_count"] == 0


@pytest.mark.parametrize(
    ("rhs_backend", "builder_attr", "solver_mode_label"),
    (
        (
            "fixed_full_field_array",
            "_build_fixed_full_field_recycling_rhs",
            "bdf_fixed_full_field_jvp",
        ),
        ("active_array", "_build_active_array_recycling_rhs", "bdf_active_array_jvp"),
    ),
)
def test_bdf_history_opt_in_uses_selected_fixed_layout_rhs_and_jvp(
    rhs_backend: str,
    builder_attr: str,
    solver_mode_label: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_solve_ivp(rhs, time_span, y0, **kwargs):
        np.testing.assert_allclose(rhs(0.0, y0), np.zeros(2))
        assert kwargs["jac"](0.0, y0) == "jvp-jacobian"
        return SimpleNamespace(
            success=True,
            message="ok",
            y=np.stack(
                [np.asarray(y0, dtype=np.float64), np.asarray(y0, dtype=np.float64)],
                axis=1,
            ),
        )

    def fake_unpack(packed_state, **kwargs):
        return {"N": np.array([packed_state[0]])}, {"ctrl": float(packed_state[1])}

    layout = SimpleNamespace(
        field_names=("N",),
        feedback_names=("ctrl",),
        active_slices=(slice(None),),
        active_shape=(1,),
        field_size=1,
        field_templates=(np.array([1.0], dtype=np.float64),),
    )

    def fake_fixed_layout_rhs(*args, **kwargs):
        captured["builder_kwargs"] = kwargs

        def rhs(state):
            return recycling._RecyclingFixedState(
                field_values=tuple(
                    np.zeros_like(value) for value in state.field_values
                ),
                feedback_values=np.zeros_like(state.feedback_values),
            )

        return rhs

    def fake_jvp_jacobian(residual, state, **kwargs):
        captured["jvp_kwargs"] = kwargs
        value = np.asarray(residual(state), dtype=np.float64)
        assert value.shape == np.asarray(state, dtype=np.float64).shape
        kwargs["timing_callback"](
            {
                "total_seconds": 0.125,
                "linearize_seconds": 0.05,
                "tangent_build_seconds": 0.01,
                "push_seconds": 0.06,
                "device_execute_seconds": 0.02,
                "host_transfer_seconds": 0.04,
                "sparse_assembly_seconds": 0.005,
                "batch_count": 2,
                "prebuilt_direction_batches": 1,
                "group_count": 4,
                "state_size": int(np.asarray(state, dtype=np.float64).size),
                "nnz": 3,
            }
        )
        return "jvp-jacobian"

    monkeypatch.setattr("scipy.integrate.solve_ivp", fake_solve_ivp)
    monkeypatch.setattr(recycling, "_recycling_active_shape", lambda mesh: (1,))
    monkeypatch.setattr(
        recycling,
        "_build_recycling_residual_sparsity",
        lambda **kwargs: _FakeSparsity(),
    )
    monkeypatch.setattr(recycling, "_build_recycling_color_groups", lambda **kwargs: ())
    monkeypatch.setattr(
        recycling, "_build_recycling_packed_state_layout", lambda **kwargs: layout
    )
    monkeypatch.setattr(
        recycling,
        "_pack_recycling_active_state",
        lambda *args, **kwargs: np.array([1.0, 0.0]),
    )
    monkeypatch.setattr(recycling, "_unpack_recycling_active_state", fake_unpack)
    monkeypatch.setattr(
        recycling, builder_attr, fake_fixed_layout_rhs
    )
    monkeypatch.setattr(recycling, "build_sparse_jvp_jacobian", fake_jvp_jacobian)
    monkeypatch.setattr(
        recycling, "_sanitize_recycling_fields", lambda config, fields: fields
    )
    monkeypatch.setattr(
        recycling, "_sanitize_feedback_integrals", lambda integrals, **kwargs: integrals
    )

    result = recycling._advance_recycling_1d_bdf_history(
        _FakeConfig(),
        _fields(),
        runtime_model=_runtime_model(),
        feedback_integrals={"ctrl": 0.0},
        field_names=("N",),
        feedback_names=("ctrl",),
        mesh=object(),
        metrics=object(),
        dataset_scalars={},
        timestep=0.5,
        steps=1,
        jacobian_mode="jvp",
        rhs_backend=rhs_backend,
        solver_mode_label=solver_mode_label,
    )

    assert captured["builder_kwargs"]["layout"] is layout
    assert captured["builder_kwargs"]["feedback_timestep"] is None
    assert "difference_plan" in captured["jvp_kwargs"]
    assert captured["jvp_kwargs"]["direction_batches"] == ()
    assert result.diagnostics["bdf_rhs_evaluation_count"] == 1
    assert result.diagnostics["bdf_rhs_callback_seconds"] >= 0.0
    assert result.diagnostics["bdf_rhs_evaluation_seconds"] >= 0.0
    assert result.diagnostics["bdf_rhs_object_evaluation_seconds"] >= 0.0
    assert result.diagnostics["bdf_rhs_numpy_conversion_seconds"] >= 0.0
    assert result.diagnostics["bdf_jacobian_base_rhs_evaluation_count"] == 0
    assert result.diagnostics["bdf_jvp_rhs_evaluation_count"] == 1
    assert result.diagnostics["bdf_jvp_jacobian_batch_count"] == 2
    assert result.diagnostics["bdf_jvp_jacobian_prebuilt_direction_batch_uses"] == 1
    assert result.diagnostics["bdf_jvp_jacobian_linearize_seconds"] == 0.05
    assert result.diagnostics["bdf_jvp_jacobian_push_seconds"] == 0.06
    assert result.diagnostics["bdf_jvp_jacobian_device_execute_seconds"] == 0.02
    assert result.diagnostics["bdf_jvp_jacobian_host_transfer_seconds"] == 0.04
    assert result.diagnostics["bdf_jvp_jacobian_sparse_assembly_seconds"] == 0.005
    assert result.diagnostics["bdf_jvp_jacobian_tangent_build_seconds"] == 0.01
    assert result.diagnostics["bdf_jvp_jacobian_total_seconds"] == 0.125
    assert result.diagnostics["bdf_jacobian_mode"] == "jvp"
    assert result.diagnostics["bdf_rhs_backend"] == rhs_backend
    assert result.diagnostics["bdf_jvp_direction_batch_count"] == 0
    assert result.variable_history["N"].shape == (2, 1)


def test_bdf_jacobian_parallel_worker_env_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JAX_DRB_FD_JACOBIAN_THREADS", raising=False)
    assert recycling._resolve_recycling_bdf_jacobian_parallel_workers() == 1

    monkeypatch.setenv("JAX_DRB_FD_JACOBIAN_THREADS", "4")
    assert recycling._resolve_recycling_bdf_jacobian_parallel_workers() == 4

    monkeypatch.setenv("JAX_DRB_FD_JACOBIAN_THREADS", "0")
    assert recycling._resolve_recycling_bdf_jacobian_parallel_workers() == 1

    monkeypatch.setenv("JAX_DRB_FD_JACOBIAN_THREADS", "not-an-int")
    assert recycling._resolve_recycling_bdf_jacobian_parallel_workers() == 1
