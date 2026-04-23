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
    return SimpleNamespace(field_names=field_names, feedback_names=feedback_names)


def _fields(value: float = 1.0) -> dict[str, np.ndarray]:
    return {"N": np.array([value], dtype=np.float64)}


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
    if solver_mode != "bdf":
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
