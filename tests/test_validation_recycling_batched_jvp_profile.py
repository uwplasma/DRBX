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
        batch_sizes=(3,),
        timed_runs=1,
        enable_pmap=False,
        check_objective_grad=False,
        residual_partition_size=2,
        jvp_partition_size=2,
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
    assert "batch_direction_build_complete" in event_names
    assert "batch_residual_warmup_complete" in event_names
    assert "batch_jvp_warmup_complete" in event_names
    assert "batch_serial_warmup_complete" in event_names
    assert "batch_warmup_complete" in event_names
    assert "linearized_action_check_complete" in event_names
    assert "batch_complete" in event_names
    assert event_names[-1] == "profile_complete"
    assert report["warmup_timing"]["base_residual_warmup_seconds"] >= 0.0
    action_diagnostics = report["linearized_action_diagnostics"]
    assert action_diagnostics["state_shape"] == (2,)
    assert action_diagnostics["call_count"] == 1
    assert action_diagnostics["batched_call_count"] == 1
    assert action_diagnostics["jvp_max_abs_error_vs_direct_jvp"] == 0.0
    assert action_diagnostics["batched_jvp_max_abs_error_vs_direct_jvp"] == 0.0
    assert action_diagnostics["residual_max_abs_error_vs_jit"] == 0.0
    assert report["batch_results"][0]["batch_warmup_seconds"] >= 0.0
    assert report["batch_results"][0]["direction_build_seconds"] >= 0.0
    assert report["batch_results"][0]["batched_residual_warmup_seconds"] >= 0.0
    assert report["batch_results"][0]["batched_jvp_warmup_seconds"] >= 0.0
    assert report["batch_results"][0]["serial_warmup_seconds"] >= 0.0
    assert report["batch_results"][0]["residual_partition_size"] == 2
    assert report["batch_results"][0]["residual_partition_count"] == 2
    assert report["batch_results"][0]["jvp_partition_size"] == 2
    assert report["batch_results"][0]["jvp_partition_count"] == 2
    assert report["batch_results"][0]["residual_batched_serial_max_abs_error"] == 0.0
    assert report["batch_results"][0]["jvp_batched_serial_max_abs_error"] == 0.0


def test_recycling_batched_jvp_profile_records_linearized_update_health() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    diagonal = jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float64)
    exact_root = jnp.asarray([0.25, -0.5, 1.5], dtype=jnp.float64)
    problem = RecyclingBatchedJvpProblem(
        residual=lambda state: diagonal
        * (jnp.asarray(state, dtype=jnp.float64) - exact_root),
        base_state=np.array([1.0, 2.0, -3.0], dtype=np.float64),
        field_names=("Ne",),
        feedback_names=(),
        mesh_active_shape=(1, 3, 1),
        state_size=3,
        rhs_backend="active_array",
    )
    events: list[dict[str, object]] = []

    report = profile_recycling_batched_jvp_problem(
        problem,
        batch_sizes=(2,),
        timed_runs=1,
        enable_pmap=False,
        check_objective_grad=False,
        check_linearized_update=True,
        linearized_update_tolerance=1.0e-12,
        linearized_update_restart=3,
        linearized_update_maxiter=4,
        linearized_update_solve_method="batched",
        linearized_update_jit_operator=True,
        linearized_update_preconditioner="none",
        progress_callback=events.append,
    )

    event_names = [str(event["event"]) for event in events]
    assert "linearized_update_check_complete" in event_names
    diagnostics = report["linearized_update_diagnostics"]
    assert diagnostics["check_enabled"] is True
    assert diagnostics["solver_success"] in (True, None)
    assert diagnostics["linear_update_residual_checked"] is True
    assert diagnostics["linear_update_relative_residual"] < 1.0e-10
    assert diagnostics["candidate_residual_inf_norm"] < 1.0e-10
    assert diagnostics["update_inf_norm"] > 0.0
    assert diagnostics["preconditioner"] == "none"
    assert diagnostics["preconditioner_diagnostics"] == {
        "name": "none",
        "build_seconds": 0.0,
        "jvp_diagonal_size": 0,
    }
    assert diagnostics["jit_linear_operator"] is True
    assert diagnostics["solve_method"] == "batched"
    assert diagnostics["action_diagnostics"]["linear_operator_jitted"] is True
    assert diagnostics["action_diagnostics"]["linearization_reused"] is True
    assert diagnostics["action_diagnostics"]["solve_call_count"] == 0


def test_recycling_batched_jvp_profile_can_skip_linearized_update_residual_check() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    diagonal = jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float64)
    exact_root = jnp.asarray([0.25, -0.5, 1.5], dtype=jnp.float64)
    problem = RecyclingBatchedJvpProblem(
        residual=lambda state: diagonal
        * (jnp.asarray(state, dtype=jnp.float64) - exact_root),
        base_state=np.array([1.0, 2.0, -3.0], dtype=np.float64),
        field_names=("Ne",),
        feedback_names=(),
        mesh_active_shape=(1, 3, 1),
        state_size=3,
        rhs_backend="active_array",
    )

    report = profile_recycling_batched_jvp_problem(
        problem,
        batch_sizes=(1,),
        timed_runs=1,
        enable_pmap=False,
        check_objective_grad=False,
        check_linearized_update=True,
        linearized_update_tolerance=1.0e-12,
        linearized_update_restart=3,
        linearized_update_maxiter=4,
        linearized_update_jit_operator=True,
        linearized_update_diagnose_residual=False,
    )

    diagnostics = report["linearized_update_diagnostics"]
    assert diagnostics["linear_update_residual_checked"] is False
    assert diagnostics["linear_update_relative_residual"] is None
    assert diagnostics["linear_update_residual_inf_norm"] is None
    assert diagnostics["candidate_residual_inf_norm"] < 1.0e-10
    assert diagnostics["action_diagnostics"]["linear_update_residual_checked"] is False


def test_recycling_batched_jvp_profile_records_jvp_diag_preconditioner() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    diagonal = jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float64)
    exact_root = jnp.asarray([0.25, -0.5, 1.5], dtype=jnp.float64)
    problem = RecyclingBatchedJvpProblem(
        residual=lambda state: diagonal
        * (jnp.asarray(state, dtype=jnp.float64) - exact_root),
        base_state=np.array([1.0, 2.0, -3.0], dtype=np.float64),
        field_names=("Ne",),
        feedback_names=(),
        mesh_active_shape=(1, 3, 1),
        state_size=3,
        rhs_backend="active_array",
    )

    report = profile_recycling_batched_jvp_problem(
        problem,
        batch_sizes=(1,),
        timed_runs=1,
        enable_pmap=False,
        check_objective_grad=False,
        check_linearized_update=True,
        linearized_update_tolerance=1.0e-12,
        linearized_update_restart=3,
        linearized_update_maxiter=4,
        linearized_update_jit_operator=True,
        linearized_update_preconditioner="jvp_diag",
        linearized_update_preconditioner_floor=1.0e-12,
        linearized_update_preconditioner_max_unknowns=3,
    )

    diagnostics = report["linearized_update_diagnostics"]
    preconditioner_diagnostics = diagnostics["preconditioner_diagnostics"]
    assert diagnostics["preconditioner"] == "jvp_diag"
    assert diagnostics["linear_update_relative_residual"] < 1.0e-10
    assert diagnostics["candidate_residual_inf_norm"] < 1.0e-10
    assert diagnostics["action_diagnostics"]["preconditioner_used"] is True
    assert diagnostics["action_diagnostics"]["linearization_reused"] is True
    assert diagnostics["action_diagnostics"]["solve_call_count"] == 0
    assert preconditioner_diagnostics["name"] == "jvp_diag"
    assert preconditioner_diagnostics["jvp_diagonal_size"] == 3
    assert preconditioner_diagnostics["build_seconds"] >= 0.0
    assert preconditioner_diagnostics["floor"] == 1.0e-12
    assert preconditioner_diagnostics["max_unknowns"] == 3
    assert preconditioner_diagnostics["raw_diagonal_min_abs"] == 2.0
    assert preconditioner_diagnostics["raw_diagonal_max_abs"] == 8.0


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
