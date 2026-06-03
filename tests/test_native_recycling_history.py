from __future__ import annotations

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


def _runtime_model(field_names: tuple[str, ...] = ("N",), feedback_names: tuple[str, ...] = ("ctrl",)):
    return SimpleNamespace(field_names=field_names, feedback_names=feedback_names, controllers={})


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
        ("bdf", "_advance_recycling_1d_bdf_history"),
        ("bdf_fixed_full_field_jvp", "_advance_recycling_1d_bdf_history"),
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
    monkeypatch.setattr(recycling, "_build_recycling_runtime_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(recycling, "_build_recycling_state_fields", lambda *args, **kwargs: _fields())

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
    if solver_mode not in {"bdf", "bdf_fixed_full_field_jvp"}:
        assert calls["kwargs"]["residual_tolerance"] == 1.0e-7
        assert calls["kwargs"]["max_nonlinear_iterations"] == 9


def test_generic_implicit_history_accumulates_fields_integrals_and_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _runtime_model()
    events: list[dict[str, object]] = []
    monkeypatch.setattr(recycling, "_build_recycling_runtime_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(recycling, "_build_recycling_state_fields", lambda *args, **kwargs: _fields())

    def fake_step(config, fields, *, feedback_integrals, timestep, **kwargs):
        next_fields = {"N": fields["N"] + timestep}
        next_integrals = {"ctrl": feedback_integrals["ctrl"] + timestep}
        return next_fields, next_integrals, object()

    monkeypatch.setattr(recycling, "advance_recycling_1d_backward_euler_step", fake_step)
    monkeypatch.setattr(
        recycling,
        "_build_recycling_progress_details",
        lambda **kwargs: ({"stage": "progress", **kwargs}, kwargs["interval_started_at"]),
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

    np.testing.assert_allclose(result.variable_history["N"][:, 0], np.array([1.0, 1.5, 2.0]))
    np.testing.assert_allclose(result.feedback_integral_history["ctrl"], np.array([0.0, 0.5, 1.0]))
    assert [event["interval_index"] for event in events] == [1, 2]
    assert all(event["solver_mode"] == "sparse" for event in events)


def test_adaptive_be_history_accumulates_interval_results_and_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []

    def fake_interval(config, fields, *, feedback_integrals, suggested_dt, **kwargs):
        return {"N": fields["N"] + 1.0}, {"ctrl": feedback_integrals["ctrl"] + 2.0}, suggested_dt + 0.25

    monkeypatch.setattr(recycling, "_advance_recycling_1d_adaptive_be_interval", fake_interval)
    monkeypatch.setattr(
        recycling,
        "_build_recycling_progress_details",
        lambda **kwargs: ({"stage": "progress", **kwargs}, kwargs["interval_started_at"]),
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

    np.testing.assert_allclose(result.variable_history["N"][:, 0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(result.feedback_integral_history["ctrl"], np.array([0.0, 2.0, 4.0]))
    assert [event["solver_mode"] for event in events] == ["adaptive_be", "adaptive_be"]


def test_adaptive_bdf_history_threads_previous_state_and_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []
    previous_seen: list[object] = []
    monkeypatch.setattr(recycling, "_initial_recycling_continuation_dt", lambda *args, **kwargs: 0.5)

    def fake_interval(config, fields, *, feedback_integrals, previous_fields, previous_integrals, previous_dt, **kwargs):
        previous_seen.append(previous_fields)
        next_fields = {"N": fields["N"] + 1.0}
        next_integrals = {"ctrl": feedback_integrals["ctrl"] + 3.0}
        return next_fields, next_integrals, fields, feedback_integrals, 0.5, 0.5

    monkeypatch.setattr(recycling, "_advance_recycling_1d_adaptive_bdf_interval", fake_interval)
    monkeypatch.setattr(
        recycling,
        "_build_recycling_progress_details",
        lambda **kwargs: ({"stage": "progress", **kwargs}, kwargs["interval_started_at"]),
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
    np.testing.assert_allclose(result.variable_history["N"][:, 0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(result.feedback_integral_history["ctrl"], np.array([0.0, 3.0, 6.0]))
    assert [event["solver_mode"] for event in events] == ["adaptive_bdf", "adaptive_bdf"]


def test_choose_recycling_next_dt_handles_finished_zero_and_scaled_errors() -> None:
    assert recycling._choose_recycling_next_dt(0.5, error_ratio=1.0, order=1, remaining=0.0, minimum_dt=0.1) == 0.5
    assert recycling._choose_recycling_next_dt(0.5, error_ratio=0.0, order=1, remaining=2.0, minimum_dt=0.1) == 1.0
    assert recycling._choose_recycling_next_dt(0.5, error_ratio=float("nan"), order=1, remaining=2.0, minimum_dt=0.1) == 1.0
    assert recycling._choose_recycling_next_dt(0.5, error_ratio=8.0, order=1, remaining=2.0, minimum_dt=0.1) == 0.25


def test_adaptive_be_interval_retries_then_accepts_step(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []
    error_ratios = iter((2.0, 0.05, 1.0))

    def fake_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        calls.append(float(timestep))
        return {"N": fields["N"] + timestep}, {"ctrl": feedback_integrals["ctrl"] + timestep}, object()

    monkeypatch.setattr(recycling, "advance_recycling_1d_backward_euler_step", fake_step)
    monkeypatch.setattr(recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: next(error_ratios))

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


def test_adaptive_be_interval_accepts_minimum_dt_after_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    def fake_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        calls.append(float(timestep))
        return {"N": fields["N"] + timestep}, {"ctrl": feedback_integrals["ctrl"] + timestep}, object()

    monkeypatch.setattr(recycling, "advance_recycling_1d_backward_euler_step", fake_step)
    monkeypatch.setattr(recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: 5.0)

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
        return {"N": fields["N"] + 0.5}, {"ctrl": feedback_integrals["ctrl"] + 0.5}, object()

    def fake_bdf_step(config, fields, previous_fields, *, timestep, feedback_integrals, **kwargs):
        bdf_calls.append(float(timestep))
        return {"N": fields["N"] + 1.0}, {"ctrl": feedback_integrals["ctrl"] + 1.0}, object()

    monkeypatch.setattr(recycling, "advance_recycling_1d_backward_euler_step", fake_be_step)
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf_step)
    monkeypatch.setattr(recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: 0.6)

    fields, integrals, previous_fields, previous_integrals, previous_dt, next_dt = recycling._advance_recycling_1d_adaptive_bdf_interval(
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


def test_adaptive_bdf_interval_uses_startup_when_previous_state_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_calls: list[float] = []

    def fake_startup_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        startup_calls.append(float(timestep))
        return {"N": fields["N"] + 2.0}, {"ctrl": feedback_integrals["ctrl"] + 2.0}, 0.5

    monkeypatch.setattr(recycling, "_advance_recycling_1d_startup_step", fake_startup_step)

    fields, integrals, previous_fields, previous_integrals, previous_dt, next_dt = recycling._advance_recycling_1d_adaptive_bdf_interval(
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


def test_adaptive_bdf_interval_falls_back_to_be_at_minimum_dt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    be_calls: list[float] = []
    bdf_calls: list[float] = []

    def fake_be_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        be_calls.append(float(timestep))
        return {"N": fields["N"] + 0.5}, {"ctrl": feedback_integrals["ctrl"] + 0.5}, object()

    def fake_bdf_step(config, fields, previous_fields, *, timestep, feedback_integrals, **kwargs):
        bdf_calls.append(float(timestep))
        return {"N": fields["N"] + 10.0}, {"ctrl": feedback_integrals["ctrl"] + 10.0}, object()

    monkeypatch.setattr(recycling, "advance_recycling_1d_backward_euler_step", fake_be_step)
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf_step)
    monkeypatch.setattr(recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: 6.0)

    fields, integrals, previous_fields, previous_integrals, previous_dt, next_dt = recycling._advance_recycling_1d_adaptive_bdf_interval(
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


def test_adaptive_bdf_interval_halves_rejected_nonminimum_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    be_calls: list[float] = []
    bdf_calls: list[float] = []
    startup_calls: list[float] = []

    def fake_be_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        be_calls.append(float(timestep))
        return {"N": fields["N"] + 0.5}, {"ctrl": feedback_integrals["ctrl"] + 0.5}, object()

    def fake_bdf_step(config, fields, previous_fields, *, timestep, feedback_integrals, **kwargs):
        bdf_calls.append(float(timestep))
        return {"N": fields["N"] + 10.0}, {"ctrl": feedback_integrals["ctrl"] + 10.0}, object()

    def fake_startup_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        startup_calls.append(float(timestep))
        return {"N": fields["N"] + timestep}, {"ctrl": feedback_integrals["ctrl"] + timestep}, 0.5

    monkeypatch.setattr(recycling, "advance_recycling_1d_backward_euler_step", fake_be_step)
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf_step)
    monkeypatch.setattr(recycling, "_advance_recycling_1d_startup_step", fake_startup_step)
    monkeypatch.setattr(recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: 6.0)
    monkeypatch.setattr(recycling, "_choose_recycling_next_dt", lambda *args, **kwargs: 1.0)

    fields, integrals, previous_fields, previous_integrals, previous_dt, next_dt = recycling._advance_recycling_1d_adaptive_bdf_interval(
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
    assert startup_calls == [0.25, 0.25]
    np.testing.assert_allclose(fields["N"], np.array([1.5]))
    assert integrals["ctrl"] == 0.5
    assert previous_fields is None
    assert previous_integrals is None
    assert previous_dt is None
    assert next_dt == 0.5


def test_adaptive_bdf_interval_resets_previous_state_when_next_dt_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_be_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        return {"N": fields["N"] + 0.5}, {"ctrl": feedback_integrals["ctrl"] + 0.5}, object()

    def fake_bdf_step(config, fields, previous_fields, *, timestep, feedback_integrals, **kwargs):
        return {"N": fields["N"] + 1.0}, {"ctrl": feedback_integrals["ctrl"] + 1.0}, object()

    monkeypatch.setattr(recycling, "advance_recycling_1d_backward_euler_step", fake_be_step)
    monkeypatch.setattr(recycling, "advance_recycling_1d_bdf2_step", fake_bdf_step)
    monkeypatch.setattr(recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: 0.3)
    monkeypatch.setattr(recycling, "_choose_recycling_next_dt", lambda *args, **kwargs: 0.25)

    fields, integrals, previous_fields, previous_integrals, previous_dt, next_dt = recycling._advance_recycling_1d_adaptive_bdf_interval(
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
    assert previous_fields is None
    assert previous_integrals is None
    assert previous_dt is None
    assert next_dt == 0.25


def test_startup_step_uses_full_and_two_half_be_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    def fake_step(config, fields, *, timestep, feedback_integrals, **kwargs):
        calls.append(float(timestep))
        return {"N": fields["N"] + timestep}, {"ctrl": feedback_integrals["ctrl"] + timestep}, object()

    monkeypatch.setattr(recycling, "advance_recycling_1d_backward_euler_step", fake_step)
    monkeypatch.setattr(recycling, "_recycling_state_error_ratio", lambda *args, **kwargs: 0.125)

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
    monkeypatch.setattr(recycling, "_recycling_active_domain_slices", lambda mesh: slice(None))

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


def test_initial_recycling_continuation_dt_uses_field_count_cutover() -> None:
    assert recycling._initial_recycling_continuation_dt(_runtime_model(field_names=("N",)), timestep=250.0) == 100.0
    assert recycling._initial_recycling_continuation_dt(_runtime_model(field_names=tuple(f"f{i}" for i in range(11))), timestep=250.0) == 25.0
    assert recycling._initial_recycling_continuation_dt(_runtime_model(field_names=tuple(f"f{i}" for i in range(11))), timestep=10.0) == 10.0


def test_bdf_history_raises_on_failed_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_solve_ivp(*args, **kwargs):
        return SimpleNamespace(success=False, message="linear solve failed", y=np.zeros((2, 1)))

    monkeypatch.setattr("scipy.integrate.solve_ivp", fake_solve_ivp)
    monkeypatch.setattr(recycling, "_recycling_active_shape", lambda mesh: (1,))
    monkeypatch.setattr(recycling, "_build_recycling_residual_sparsity", lambda **kwargs: _FakeSparsity())
    monkeypatch.setattr(recycling, "_build_recycling_color_groups", lambda **kwargs: ())
    layout = SimpleNamespace(
        field_names=("N",),
        feedback_names=("ctrl",),
        active_slices=(slice(None),),
        active_shape=(1,),
        field_size=1,
        field_templates=(np.array([1.0], dtype=np.float64),),
    )
    monkeypatch.setattr(recycling, "_build_recycling_packed_state_layout", lambda **kwargs: layout)
    monkeypatch.setattr(recycling, "_pack_recycling_active_state", lambda *args, **kwargs: np.array([1.0, 0.0]))

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


def test_bdf_history_unpacks_sanitizes_and_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(recycling, "_build_recycling_residual_sparsity", lambda **kwargs: _FakeSparsity())
    monkeypatch.setattr(recycling, "_build_recycling_color_groups", lambda **kwargs: ())
    layout = SimpleNamespace(
        field_names=("N",),
        feedback_names=("ctrl",),
        active_slices=(slice(None),),
        active_shape=(1,),
        field_size=1,
        field_templates=(np.array([1.0], dtype=np.float64),),
    )
    monkeypatch.setattr(recycling, "_build_recycling_packed_state_layout", lambda **kwargs: layout)
    monkeypatch.setattr(recycling, "_pack_recycling_active_state", lambda *args, **kwargs: np.array([1.0, 0.0]))
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

    monkeypatch.setattr(recycling, "build_sparse_difference_quotient_jacobian", fake_build_jacobian)
    monkeypatch.setattr(recycling, "_sanitize_recycling_fields", lambda config, fields: fields)
    monkeypatch.setattr(recycling, "_sanitize_feedback_integrals", lambda integrals, **kwargs: integrals)
    monkeypatch.setattr(
        recycling,
        "_build_recycling_progress_details",
        lambda **kwargs: ({"interval_index": kwargs["interval_index"], "solver_mode": kwargs["solver_mode"]}, 0.0),
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

    np.testing.assert_allclose(result.variable_history["N"][:, 0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(result.feedback_integral_history["ctrl"], np.array([0.0, 0.5, 1.0]))
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


def test_bdf_history_opt_in_uses_fixed_full_field_rhs_and_jvp(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_solve_ivp(rhs, time_span, y0, **kwargs):
        np.testing.assert_allclose(rhs(0.0, y0), np.zeros(2))
        assert kwargs["jac"](0.0, y0) == "jvp-jacobian"
        return SimpleNamespace(
            success=True,
            message="ok",
            y=np.stack([np.asarray(y0, dtype=np.float64), np.asarray(y0, dtype=np.float64)], axis=1),
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

    def fake_fixed_full_field_rhs(*args, **kwargs):
        captured["builder_kwargs"] = kwargs

        def rhs(state):
            return recycling._RecyclingFixedState(
                field_values=tuple(np.zeros_like(value) for value in state.field_values),
                feedback_values=np.zeros_like(state.feedback_values),
            )

        return rhs

    def fake_jvp_jacobian(residual, state, **kwargs):
        captured["jvp_kwargs"] = kwargs
        value = np.asarray(residual(state), dtype=np.float64)
        assert value.shape == np.asarray(state, dtype=np.float64).shape
        return "jvp-jacobian"

    monkeypatch.setattr("scipy.integrate.solve_ivp", fake_solve_ivp)
    monkeypatch.setattr(recycling, "_recycling_active_shape", lambda mesh: (1,))
    monkeypatch.setattr(recycling, "_build_recycling_residual_sparsity", lambda **kwargs: _FakeSparsity())
    monkeypatch.setattr(recycling, "_build_recycling_color_groups", lambda **kwargs: ())
    monkeypatch.setattr(recycling, "_build_recycling_packed_state_layout", lambda **kwargs: layout)
    monkeypatch.setattr(recycling, "_pack_recycling_active_state", lambda *args, **kwargs: np.array([1.0, 0.0]))
    monkeypatch.setattr(recycling, "_unpack_recycling_active_state", fake_unpack)
    monkeypatch.setattr(recycling, "_build_fixed_full_field_recycling_rhs", fake_fixed_full_field_rhs)
    monkeypatch.setattr(recycling, "build_sparse_jvp_jacobian", fake_jvp_jacobian)
    monkeypatch.setattr(recycling, "_sanitize_recycling_fields", lambda config, fields: fields)
    monkeypatch.setattr(recycling, "_sanitize_feedback_integrals", lambda integrals, **kwargs: integrals)

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
        rhs_backend="fixed_full_field_array",
        solver_mode_label="bdf_fixed_full_field_jvp",
    )

    assert captured["builder_kwargs"]["layout"] is layout
    assert captured["builder_kwargs"]["feedback_timestep"] is None
    assert "difference_plan" in captured["jvp_kwargs"]
    assert result.diagnostics["bdf_rhs_evaluation_count"] == 1
    assert result.diagnostics["bdf_rhs_callback_seconds"] >= 0.0
    assert result.diagnostics["bdf_rhs_evaluation_seconds"] >= 0.0
    assert result.diagnostics["bdf_rhs_object_evaluation_seconds"] >= 0.0
    assert result.diagnostics["bdf_rhs_numpy_conversion_seconds"] >= 0.0
    assert result.diagnostics["bdf_jacobian_base_rhs_evaluation_count"] == 0
    assert result.diagnostics["bdf_jvp_rhs_evaluation_count"] == 1
    assert result.diagnostics["bdf_jacobian_mode"] == "jvp"
    assert result.diagnostics["bdf_rhs_backend"] == "fixed_full_field_array"
    assert result.variable_history["N"].shape == (2, 1)


def test_bdf_jacobian_parallel_worker_env_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JAX_DRB_FD_JACOBIAN_THREADS", raising=False)
    assert recycling._resolve_recycling_bdf_jacobian_parallel_workers() == 1

    monkeypatch.setenv("JAX_DRB_FD_JACOBIAN_THREADS", "4")
    assert recycling._resolve_recycling_bdf_jacobian_parallel_workers() == 4

    monkeypatch.setenv("JAX_DRB_FD_JACOBIAN_THREADS", "0")
    assert recycling._resolve_recycling_bdf_jacobian_parallel_workers() == 1

    monkeypatch.setenv("JAX_DRB_FD_JACOBIAN_THREADS", "not-an-int")
    assert recycling._resolve_recycling_bdf_jacobian_parallel_workers() == 1
