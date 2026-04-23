from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from jax_drb.cli import main
import jax_drb.native as native_module
from jax_drb.native import run_input_case
from jax_drb.native.runner import NativeRestartState
from jax_drb.parity.arrays import load_portable_array_payload
from jax_drb.runtime import load_restart_bundle


_DIFFUSION_INPUT = """
nout = 3
timestep = 5

[mesh]
nx = 10
ny = 10
nz = 1

dx = 0.0075 + 0.005*x
dy = 0.01
dz = 0.01

J = 1

[solver]
mxstep = 1000

[model]
components = h

[h]
type = evolve_density, evolve_pressure, anomalous_diffusion
AA = 1
charge = 1
anomalous_D = 2
thermal_conduction = false

[Nh]
function = 1 + H(x - 0.25) * H(0.75-x) * exp(-(y-π)^2)
bndry_all = neumann

[Ph]
function = Nh:function
bndry_all = neumann
"""


_DIFFUSION_TOML_INPUT = """
[time]
nout = 2
timestep = 5.0

[runtime]
precision = "float32"

[mesh]
nx = 10
ny = 10
nz = 1
dx = { expr = "0.0075 + 0.005*x" }
dy = 0.01
dz = 0.01
J = 1

[solver]
mxstep = 1000

[model]
components = ["h"]

[species.h]
type = ["evolve_density", "evolve_pressure", "anomalous_diffusion"]
AA = 1
charge = 1
anomalous_D = 2
thermal_conduction = false

[fields.Nh]
function = { expr = "1 + H(x - 0.25) * H(0.75-x) * exp(-(y-π)^2)" }
bndry_all = "neumann"

[fields.Ph]
function = { ref = "Nh:function" }
bndry_all = "neumann"
"""

_DIFFUSION_TOML_CONFIGURED_RUN = """
[time]
nout = 2
timestep = 5.0

[runtime]
precision = "float32"

[runtime.logging]
verbosity = "detailed"
verbose = true
quiet = false

[mesh]
nx = 10
ny = 10
nz = 1
dx = { expr = "0.0075 + 0.005*x" }
dy = 0.01
dz = 0.01
J = 1

[solver]
mxstep = 1000

[model]
components = ["h"]

[species.h]
type = ["evolve_density", "evolve_pressure", "anomalous_diffusion"]
AA = 1
charge = 1
anomalous_D = 2
thermal_conduction = false

[fields.Nh]
function = { expr = "1 + H(x - 0.25) * H(0.75-x) * exp(-(y-π)^2)" }
bndry_all = "neumann"

[fields.Ph]
function = { ref = "Nh:function" }
bndry_all = "neumann"

[output]
directory = "__OUTPUT_DIR__"
write_summary = true
write_arrays = true
write_restart = true
write_log = true
"""


def test_run_command_writes_artifacts_and_restart_bundle(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "diffusion_restartable.inp"
    input_path.write_text(_DIFFUSION_INPUT, encoding="utf-8")
    output_dir = tmp_path / "run"

    exit_code = main(["run", str(input_path), "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "diffusion_restartable_summary.json").exists()
    assert (output_dir / "diffusion_restartable_arrays.npz").exists()
    restart_path = output_dir / "diffusion_restartable_restart.npz"
    assert restart_path.exists()
    assert (output_dir / "diffusion_restartable_run_log.json").exists()

    captured = capsys.readouterr().out
    assert "Run Summary" in captured
    assert "restart" in captured.lower()

    restart = load_restart_bundle(restart_path)
    assert restart.case_name == "diffusion_restartable"
    assert restart.completed_steps == 3
    assert restart.current_time == 15.0
    assert tuple(sorted(restart.state_variables)) == ("Nh", "Ph")

    run_log = json.loads((output_dir / "diffusion_restartable_run_log.json").read_text(encoding="utf-8"))
    assert run_log["capability_tier"] == "native_exact"
    assert run_log["restart_supported"] is True
    assert run_log["event_count"] == len(run_log["events"])
    assert "artifacts" in run_log["event_stages"]
    assert run_log["run_configuration"]["time"] == {"nout": 3, "timestep": 5.0}
    assert run_log["run_configuration"]["mesh"]["nx"] == 10
    assert run_log["run_configuration"]["solver"]["mxstep"] == 1000
    assert run_log["restart_info"]["saved_completed_steps"] == 3
    assert run_log["restart_info"]["saved_current_time"] == 15.0

    arrays = load_portable_array_payload(output_dir / "diffusion_restartable_arrays.npz")
    assert tuple(sorted(arrays["variables"])) == ("Nh", "Ph")


def test_run_command_can_resume_from_restart_bundle(tmp_path: Path) -> None:
    input_path = tmp_path / "diffusion_restartable.inp"
    input_path.write_text(_DIFFUSION_INPUT, encoding="utf-8")

    first_output_dir = tmp_path / "run_first"
    resume_output_dir = tmp_path / "run_resume"

    assert main(["run", str(input_path), "--output-dir", str(first_output_dir), "--quiet"]) == 0
    restart_path = first_output_dir / "diffusion_restartable_restart.npz"
    assert restart_path.exists()

    assert main(
        [
            "run",
            str(input_path),
            "--output-dir",
            str(resume_output_dir),
            "--restart-in",
            str(restart_path),
            "--resume-steps",
            "2",
            "--case-name",
            "diffusion_resume",
            "--quiet",
        ]
    ) == 0

    resume_log = json.loads((resume_output_dir / "diffusion_resume_run_log.json").read_text(encoding="utf-8"))
    assert resume_log["restart_info"]["input_completed_steps"] == 3
    assert resume_log["restart_info"]["requested_additional_steps"] == 2
    assert resume_log["restart_info"]["saved_completed_steps"] == 2

    resumed = run_input_case(
        input_path,
        case_name="diffusion_resume",
        parity_mode="run",
        restart_state=None,
        output_steps=5,
    )
    restart = load_restart_bundle(restart_path)
    resumed_from_restart = run_input_case(
        input_path,
        case_name="diffusion_resume",
        parity_mode="run",
        restart_state=NativeRestartState(
            time_offset=restart.current_time,
            completed_steps=restart.completed_steps,
            configured_timestep=restart.configured_timestep,
            variables=restart.state_variables,
        ),
        output_steps=2,
    )

    np.testing.assert_allclose(resumed_from_restart.variables["Nh"][-1], resumed.variables["Nh"][-1], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(resumed_from_restart.variables["Ph"][-1], resumed.variables["Ph"][-1], rtol=1e-8, atol=1e-10)
    assert resumed_from_restart.time_points == (15.0, 20.0, 25.0)


def test_run_command_supports_bare_toml_invocation_and_configured_float32(tmp_path: Path) -> None:
    input_path = tmp_path / "diffusion_restartable.toml"
    input_path.write_text(_DIFFUSION_TOML_INPUT, encoding="utf-8")
    output_dir = tmp_path / "run"

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jax_drb",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--quiet",
        ],
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0
    assert (output_dir / "diffusion_restartable_summary.json").exists()
    assert (output_dir / "diffusion_restartable_arrays.npz").exists()

    run_log = json.loads((output_dir / "diffusion_verbose_run_log.json").read_text(encoding="utf-8"))
    assert run_log["capability_tier"] == "native_exact"
    assert run_log["run_configuration"]["runtime"]["precision"] == "float32"
    assert run_log["run_configuration"]["runtime"]["backend"]
    assert run_log["run_configuration"]["runtime"]["jax_version"]
    assert run_log["run_configuration"]["runtime"]["python_version"]
    assert run_log["run_configuration"]["runtime"]["platform"]
    assert run_log["run_configuration"]["runtime"]["process_id"] > 0
    assert run_log["run_configuration"]["runtime"]["elapsed_seconds"] is not None


def test_run_command_accepts_toml_input_and_records_precision(tmp_path: Path) -> None:
    input_path = tmp_path / "diffusion_restartable.toml"
    input_path.write_text(_DIFFUSION_TOML_INPUT, encoding="utf-8")
    output_dir = tmp_path / "run_toml"

    exit_code = main(["run", str(input_path), "--output-dir", str(output_dir), "--quiet"])

    assert exit_code == 0
    run_log = json.loads((output_dir / "diffusion_verbose_run_log.json").read_text(encoding="utf-8"))
    assert run_log["run_configuration"]["runtime"]["precision"] == "float32"
    arrays = load_portable_array_payload(output_dir / "diffusion_restartable_arrays.npz")
    assert tuple(sorted(arrays["variables"])) == ("Nh", "Ph")


def test_main_accepts_bare_input_file_without_explicit_run_subcommand(tmp_path: Path) -> None:
    input_path = tmp_path / "diffusion_restartable.toml"
    input_path.write_text(_DIFFUSION_TOML_INPUT, encoding="utf-8")
    output_dir = tmp_path / "run_bare"

    exit_code = main([str(input_path), "--output-dir", str(output_dir), "--quiet"])

    assert exit_code == 0
    assert (output_dir / "diffusion_restartable_summary.json").exists()
    run_log = json.loads((output_dir / "diffusion_verbose_run_log.json").read_text(encoding="utf-8"))
    assert run_log["run_configuration"]["runtime"]["precision"] == "float32"


def test_run_command_precision_flag_overrides_input_precision(tmp_path: Path) -> None:
    input_path = tmp_path / "diffusion_restartable.toml"
    input_path.write_text(_DIFFUSION_TOML_INPUT.replace('precision = "float32"', 'precision = "float64"'), encoding="utf-8")
    output_dir = tmp_path / "run_override"

    exit_code = main([str(input_path), "--precision", "float32", "--output-dir", str(output_dir), "--quiet"])

    assert exit_code == 0
    run_log = json.loads((output_dir / "diffusion_restartable_run_log.json").read_text(encoding="utf-8"))
    assert run_log["run_configuration"]["runtime"]["precision"] == "float32"


def test_main_routes_bare_input_path_to_run_command(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "diffusion_restartable.toml"
    input_path.write_text(_DIFFUSION_TOML_INPUT, encoding="utf-8")

    exit_code = main([str(input_path), "--dry-run"])

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "input:" in captured
    assert "scheduled components:" in captured


def test_run_command_accepts_bare_toml_input_and_records_precision(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "diffusion_restartable.toml"
    input_path.write_text(_DIFFUSION_TOML_INPUT, encoding="utf-8")
    output_dir = tmp_path / "run_toml"

    exit_code = main([str(input_path), "--output-dir", str(output_dir)])

    assert exit_code == 0
    summary_path = output_dir / "diffusion_restartable_summary.json"
    arrays_path = output_dir / "diffusion_restartable_arrays.npz"
    restart_path = output_dir / "diffusion_restartable_restart.npz"
    run_log_path = output_dir / "diffusion_restartable_run_log.json"
    assert summary_path.exists()
    assert arrays_path.exists()
    assert restart_path.exists()
    assert run_log_path.exists()

    captured = capsys.readouterr().out
    assert "Run Summary" in captured
    assert "float32" in captured

    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    assert run_log["capability_tier"] == "native_exact"
    assert run_log["run_configuration"]["runtime"]["precision"] == "float32"

    restart = load_restart_bundle(restart_path)
    assert restart.completed_steps == 2

    arrays = load_portable_array_payload(arrays_path)
    assert tuple(sorted(arrays["variables"])) == ("Nh", "Ph")


def test_run_command_reads_output_and_logging_from_toml(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "configured_output"
    input_path = tmp_path / "diffusion_configured.toml"
    input_path.write_text(_DIFFUSION_TOML_CONFIGURED_RUN.replace("__OUTPUT_DIR__", output_dir.as_posix()), encoding="utf-8")

    exit_code = main([str(input_path)])

    assert exit_code == 0
    summary_path = output_dir / "diffusion_configured_summary.json"
    arrays_path = output_dir / "diffusion_configured_arrays.npz"
    restart_path = output_dir / "diffusion_configured_restart.npz"
    run_log_path = output_dir / "diffusion_configured_run_log.json"
    assert summary_path.exists()
    assert arrays_path.exists()
    assert restart_path.exists()
    assert run_log_path.exists()

    captured = capsys.readouterr().out
    assert "Loaded input configuration" in captured
    assert "Launching native run" in captured
    assert "Resolved artifact destinations" in captured
    assert "Wrote summary JSON" in captured
    assert "Run Summary" in captured

    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    assert run_log["run_configuration"]["runtime"]["precision"] == "float32"
    assert run_log["run_configuration"]["runtime"]["logging"]["verbosity"] == "detailed"
    assert run_log["run_configuration"]["runtime"]["logging"]["verbose"] is True
    assert run_log["run_configuration"]["runtime"]["logging"]["quiet"] is False
    assert run_log["run_configuration"]["output"]["directory"] == str(output_dir)
    assert run_log["run_configuration"]["output"]["working_directory"]
    assert "/Users/" not in json.dumps(run_log, sort_keys=True)
    assert len(run_log["events"]) >= 3
    assert run_log["event_count"] == len(run_log["events"])
    assert "artifacts" in run_log["event_stages"]
    assert run_log["events"][0]["stage"] == "configuration"


def test_run_command_verbose_flag_enables_detailed_terminal_events(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "diffusion_verbose.toml"
    input_path.write_text(_DIFFUSION_TOML_INPUT, encoding="utf-8")
    output_dir = tmp_path / "run_verbose"

    exit_code = main([str(input_path), "--output-dir", str(output_dir), "--verbose"])

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Loaded input configuration" in captured
    assert "Launching native run" in captured
    assert "Resolved artifact destinations" in captured
    assert "Wrote arrays NPZ" in captured

    run_log = json.loads((output_dir / "diffusion_verbose_run_log.json").read_text(encoding="utf-8"))
    assert run_log["run_configuration"]["runtime"]["logging"]["verbosity"] == "detailed"
    assert run_log["run_configuration"]["runtime"]["logging"]["verbose"] is True
    assert run_log["event_count"] == len(run_log["events"])


def test_run_command_verbose_relay_prints_progress_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    input_path = tmp_path / "diffusion_verbose.toml"
    input_path.write_text(_DIFFUSION_TOML_INPUT, encoding="utf-8")
    output_dir = tmp_path / "run_progress"
    baseline_run_input_case = native_module.run_input_case

    def fake_run_input_case(*args, **kwargs):
        event_logger = kwargs.get("event_logger")
        if event_logger is not None:
            event_logger(
                {
                    "stage": "progress",
                    "message": "Completed recycling transient interval",
                    "details": {
                        "interval_index": 1,
                        "steps": 2,
                        "solver_mode": "continuation",
                        "accepted_dt": 6.25,
                        "stored_states": 2,
                        "fraction_complete": 0.5,
                        "estimated_remaining_seconds": 12.5,
                        "live_progress": True,
                    },
                }
            )
        return baseline_run_input_case(*args, **kwargs)

    monkeypatch.setattr(native_module, "run_input_case", fake_run_input_case)

    exit_code = main([str(input_path), "--output-dir", str(output_dir), "--verbose"])

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Completed recycling transient interval" in captured
    assert "solver_mode" in captured
    run_log = json.loads((output_dir / "diffusion_verbose_run_log.json").read_text(encoding="utf-8"))
    assert "progress" in run_log["event_stages"]


def test_run_command_progress_summary_includes_eta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    input_path = tmp_path / "diffusion_summary.toml"
    input_path.write_text(_DIFFUSION_TOML_INPUT, encoding="utf-8")
    output_dir = tmp_path / "run_progress_summary"
    baseline_run_input_case = native_module.run_input_case

    def fake_run_input_case(*args, **kwargs):
        event_logger = kwargs.get("event_logger")
        if event_logger is not None:
            event_logger(
                {
                    "stage": "progress",
                    "message": "Completed recycling transient interval",
                    "details": {
                        "interval_index": 1,
                        "steps": 4,
                        "solver_mode": "adaptive_bdf",
                        "accepted_dt": 5.0,
                        "stored_states": 2,
                        "fraction_complete": 0.25,
                        "estimated_remaining_seconds": 42.0,
                        "live_progress": True,
                    },
                }
            )
        return baseline_run_input_case(*args, **kwargs)

    monkeypatch.setattr(native_module, "run_input_case", fake_run_input_case)

    exit_code = main([str(input_path), "--output-dir", str(output_dir)])

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "interval 1/4" in captured
    assert "25.0%" in captured
    assert "eta 42.0s" in captured


def test_run_input_case_verbose_emits_python_driver_events(tmp_path: Path) -> None:
    input_path = tmp_path / "diffusion_restartable.inp"
    input_path.write_text(_DIFFUSION_INPUT, encoding="utf-8")
    events: list[dict[str, object]] = []

    result = run_input_case(
        input_path,
        case_name="diffusion_driver_verbose",
        parity_mode="run",
        output_steps=1,
        verbose=True,
        event_logger=events.append,
    )

    assert result.time_points == (0.0, 5.0)
    assert [str(event["stage"]) for event in events] == ["configuration", "mesh", "run", "summary"]


def test_run_command_reads_restart_request_from_toml(tmp_path: Path) -> None:
    input_path = tmp_path / "diffusion_restartable.toml"
    input_path.write_text(_DIFFUSION_TOML_INPUT, encoding="utf-8")
    first_output_dir = tmp_path / "run_first"
    assert main(["run", str(input_path), "--output-dir", str(first_output_dir), "--quiet"]) == 0

    resume_output_dir = tmp_path / "run_resume_from_toml"
    resume_restart = first_output_dir / "diffusion_restartable_restart.npz"
    resume_input = tmp_path / "diffusion_resume.toml"
    resume_input.write_text(
        _DIFFUSION_TOML_CONFIGURED_RUN.replace("__OUTPUT_DIR__", resume_output_dir.as_posix())
        + f'\n[restart]\ninput = "{resume_restart.as_posix()}"\nresume_steps = 2\n',
        encoding="utf-8",
    )

    assert main([str(resume_input), "--case-name", "diffusion_resume_from_toml", "--quiet"]) == 0

    resume_log = json.loads((resume_output_dir / "diffusion_resume_from_toml_run_log.json").read_text(encoding="utf-8"))
    assert resume_log["restart_info"]["loaded_from"] == str(resume_restart)
    assert resume_log["restart_info"]["requested_additional_steps"] == 2
    assert resume_log["run_configuration"]["restart_request"]["restart_in"] == str(resume_restart)
    assert resume_log["run_configuration"]["restart_request"]["resume_steps"] == 2
