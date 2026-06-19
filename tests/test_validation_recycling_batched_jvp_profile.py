from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import jax_drb.validation.recycling_batched_jvp_profile as profile_module
from jax_drb.validation.recycling_batched_jvp_profile import (
    _check_pmap_identity,
    RecyclingBatchedJvpProblem,
    build_recycling_batched_jvp_problem,
    create_recycling_batched_jvp_profile_package,
    profile_recycling_batched_jvp_problem,
    summarize_recycling_batched_jvp_scaling,
)


class _ReadyArray:
    def __init__(self, value):
        self._value = np.asarray(value, dtype=np.float64)

    def block_until_ready(self):
        return self._value


class _FakeJax:
    def __init__(self, *, scale: float = 1.0):
        self.scale = float(scale)

    def pmap(self, function, *, devices):
        def mapped(block):
            return _ReadyArray(self.scale * function(block))

        return mapped


def test_recycling_batched_jvp_pmap_identity_helper_requires_multiple_devices() -> None:
    passed, max_abs_error, skip_reason = _check_pmap_identity(_FakeJax(), np, ("cpu0",))

    assert passed is False
    assert max_abs_error is None
    assert skip_reason == "fewer than two visible JAX devices"


def test_recycling_batched_jvp_pmap_identity_helper_accepts_identity_map() -> None:
    passed, max_abs_error, skip_reason = _check_pmap_identity(
        _FakeJax(), np, ("gpu0", "gpu1")
    )

    assert passed is True
    assert max_abs_error == 0.0
    assert skip_reason is None


def test_recycling_batched_jvp_pmap_identity_helper_rejects_corrupt_map() -> None:
    passed, max_abs_error, skip_reason = _check_pmap_identity(
        _FakeJax(scale=0.0), np, ("gpu0", "gpu1")
    )

    assert passed is False
    assert max_abs_error > 0.0
    assert "pmap identity check failed" in str(skip_reason)


def test_recycling_batched_jvp_scaling_summary_selects_best_metrics() -> None:
    summary = summarize_recycling_batched_jvp_scaling(
        [
            {
                "batch_size": 1,
                "residual_speedup_vs_serial": 1.0,
                "jvp_speedup_vs_serial": 1.0,
                "batched_residual_states_per_second": 10.0,
                "batched_jvp_states_per_second": 8.0,
                "pmap_jvp_states_per_second": None,
            },
            {
                "batch_size": 4,
                "residual_speedup_vs_serial": 2.5,
                "jvp_speedup_vs_serial": 3.0,
                "batched_residual_states_per_second": 25.0,
                "batched_jvp_states_per_second": 20.0,
                "pmap_jvp_states_per_second": 18.0,
                "pmap_device_count": 2,
                "pmap_batch_size": 4,
                "pmap_jvp_speedup_vs_batched": 1.5,
                "pmap_jvp_speedup_vs_serial": 4.0,
            },
        ]
    )

    assert summary["batch_count"] == 2
    assert summary["batch_sizes"] == [1, 4]
    assert summary["max_batch_size"] == 4
    assert summary["throughput_units"] == "states_per_second"
    assert summary["best_residual_speedup_vs_serial"] == {
        "batch_size": 4,
        "speedup": 2.5,
    }
    assert summary["best_jvp_speedup_vs_serial"] == {
        "batch_size": 4,
        "speedup": 3.0,
    }
    assert summary["best_residual_batch_efficiency"] == {
        "batch_size": 1,
        "speedup": 1.0,
        "efficiency": 1.0,
    }
    assert summary["best_jvp_batch_efficiency"] == {
        "batch_size": 1,
        "speedup": 1.0,
        "efficiency": 1.0,
    }
    assert summary["best_batched_residual_throughput"] == {
        "batch_size": 4,
        "states_per_second": 25.0,
    }
    assert summary["best_batched_jvp_throughput"] == {
        "batch_size": 4,
        "states_per_second": 20.0,
    }
    assert summary["best_pmap_jvp_throughput"] == {
        "batch_size": 4,
        "states_per_second": 18.0,
    }
    assert summary["best_pmap_jvp_speedup_vs_batched"] == {
        "batch_size": 4,
        "speedup": 1.5,
    }
    assert summary["best_pmap_jvp_speedup_vs_serial"] == {
        "batch_size": 4,
        "speedup": 4.0,
    }
    assert summary["best_pmap_jvp_device_efficiency_vs_serial"] == {
        "batch_size": 4,
        "pmap_batch_size": 4,
        "device_count": 2,
        "speedup": 4.0,
        "device_efficiency": 2.0,
    }


def test_recycling_batched_jvp_scaling_summary_handles_no_pmap_results() -> None:
    summary = summarize_recycling_batched_jvp_scaling(
        [
            {
                "batch_size": 1,
                "residual_speedup_vs_serial": 1.0,
                "jvp_speedup_vs_serial": 1.0,
                "batched_residual_states_per_second": 10.0,
                "batched_jvp_states_per_second": 8.0,
                "pmap_jvp_states_per_second": None,
            }
        ]
    )

    assert summary["best_pmap_jvp_throughput"] is None
    assert summary["best_pmap_jvp_speedup_vs_batched"] is None
    assert summary["best_pmap_jvp_speedup_vs_serial"] is None
    assert summary["best_pmap_jvp_device_efficiency_vs_serial"] is None


def test_recycling_batched_jvp_problem_uses_fixed_full_field_backend_by_default(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("jax")
    captured: dict[str, object] = {}
    input_path = tmp_path / "BOUT.inp"
    input_path.write_text("nout = 1\n", encoding="utf-8")

    mesh = SimpleNamespace(xstart=0, xend=1, ystart=0, yend=1, nz=1)
    runtime_model = SimpleNamespace(feedback_names=("flux",))
    context = SimpleNamespace(
        residual=lambda state: state,
        packed_previous_state=np.array([1.0, 2.0], dtype=np.float64),
        field_names=("Ne",),
        feedback_names=("flux",),
    )

    monkeypatch.setattr(profile_module, "load_bout_input", lambda path: {"path": path})
    monkeypatch.setattr(
        profile_module.RunConfiguration,
        "from_config",
        staticmethod(lambda config: SimpleNamespace()),
    )
    monkeypatch.setattr(profile_module, "build_structured_mesh", lambda *args: mesh)
    monkeypatch.setattr(
        profile_module, "build_structured_metrics", lambda *args: object()
    )
    monkeypatch.setattr(
        profile_module,
        "resolved_dataset_scalars",
        lambda run_config: {"rho_s0": 1.0, "Tnorm": 1.0},
    )
    monkeypatch.setattr(
        profile_module,
        "_build_recycling_runtime_model",
        lambda *args, **kwargs: runtime_model,
    )
    monkeypatch.setattr(
        profile_module,
        "_build_recycling_state_fields",
        lambda runtime_model: {"Ne": np.ones((2, 2, 1), dtype=np.float64)},
    )

    def fake_residual_context(*args, **kwargs):
        captured.update(kwargs)
        return context

    monkeypatch.setattr(
        profile_module,
        "build_recycling_1d_backward_euler_residual_context",
        fake_residual_context,
    )

    problem = build_recycling_batched_jvp_problem(input_path)

    assert captured["rhs_backend"] == "fixed_full_field_array"
    assert problem.rhs_backend == "fixed_full_field_array"
    assert problem.state_size == 2


def test_recycling_batched_jvp_problem_accepts_active_array_backend(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("jax")
    captured: dict[str, object] = {}
    input_path = tmp_path / "BOUT.inp"
    input_path.write_text("nout = 1\n", encoding="utf-8")

    mesh = SimpleNamespace(xstart=0, xend=1, ystart=0, yend=1, nz=1)
    runtime_model = SimpleNamespace(feedback_names=("flux",))
    context = SimpleNamespace(
        residual=lambda state: state,
        packed_previous_state=np.array([1.0, 2.0], dtype=np.float64),
        field_names=("Ne",),
        feedback_names=("flux",),
    )

    monkeypatch.setattr(profile_module, "load_bout_input", lambda path: {"path": path})
    monkeypatch.setattr(
        profile_module.RunConfiguration,
        "from_config",
        staticmethod(lambda config: SimpleNamespace()),
    )
    monkeypatch.setattr(profile_module, "build_structured_mesh", lambda *args: mesh)
    monkeypatch.setattr(
        profile_module, "build_structured_metrics", lambda *args: object()
    )
    monkeypatch.setattr(
        profile_module,
        "resolved_dataset_scalars",
        lambda run_config: {"rho_s0": 1.0, "Tnorm": 1.0},
    )
    monkeypatch.setattr(
        profile_module,
        "_build_recycling_runtime_model",
        lambda *args, **kwargs: runtime_model,
    )
    monkeypatch.setattr(
        profile_module,
        "_build_recycling_state_fields",
        lambda runtime_model: {"Ne": np.ones((2, 2, 1), dtype=np.float64)},
    )

    def fake_residual_context(*args, **kwargs):
        captured.update(kwargs)
        return context

    monkeypatch.setattr(
        profile_module,
        "build_recycling_1d_backward_euler_residual_context",
        fake_residual_context,
    )

    problem = build_recycling_batched_jvp_problem(
        input_path, rhs_backend="active_array"
    )

    assert captured["rhs_backend"] == "active_array"
    assert problem.rhs_backend == "active_array"
    assert problem.state_size == 2


def test_recycling_batched_jvp_profile_reports_progress_events() -> None:
    pytest.importorskip("jax")

    problem = RecyclingBatchedJvpProblem(
        residual=lambda state: 2.0 * state,
        base_state=np.array([1.0, 2.0], dtype=np.float64),
        field_names=("Ne",),
        feedback_names=(),
        mesh_active_shape=(1, 2, 1),
        state_size=2,
        rhs_backend="fixed_full_field_array",
    )
    events: list[dict[str, object]] = []

    report = profile_recycling_batched_jvp_problem(
        problem,
        batch_sizes=(1,),
        timed_runs=1,
        enable_pmap=False,
        check_objective_grad=False,
        progress_callback=events.append,
    )

    event_names = [str(event["event"]) for event in events]
    assert event_names[:4] == [
        "profile_start",
        "base_residual_warmup_complete",
        "base_jvp_warmup_complete",
        "jvp_fd_check_complete",
    ]
    assert "batch_start" in event_names
    assert "batch_warmup_complete" in event_names
    assert "batch_complete" in event_names
    assert report["warmup_timing"]["base_residual_warmup_seconds"] >= 0.0
    assert report["batch_results"][0]["batch_warmup_seconds"] >= 0.0


def test_create_recycling_batched_jvp_profile_package_writes_progress_jsonl(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "BOUT.inp"
    input_path.write_text("nout = 1\n", encoding="utf-8")
    output_dir = tmp_path / "profile"
    fake_problem = SimpleNamespace(
        rhs_backend="active_array",
        state_size=3,
        mesh_active_shape=(1, 3, 1),
    )

    monkeypatch.setattr(
        profile_module,
        "build_recycling_batched_jvp_problem",
        lambda *args, **kwargs: fake_problem,
    )

    def fake_profile(problem, *, progress_callback=None, **kwargs):
        assert problem is fake_problem
        assert progress_callback is not None
        progress_callback({"event": "fake_profile_complete", "state_size": 3})
        return {
            "case": "fake",
            "batch_results": [],
            "throughput_summary": {},
            "warmup_timing": {},
        }

    monkeypatch.setattr(
        profile_module,
        "profile_recycling_batched_jvp_problem",
        fake_profile,
    )

    report = create_recycling_batched_jvp_profile_package(
        input_path,
        output_dir,
        rhs_backend="active_array",
    )

    progress_path = output_dir / "profile_progress.jsonl"
    records = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event"] for record in records] == [
        "problem_build_start",
        "problem_build_complete",
        "fake_profile_complete",
    ]
    assert report["profile_progress_jsonl"] == "profile_progress.jsonl"
    assert (output_dir / "profile_summary.json").is_file()
