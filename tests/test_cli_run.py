from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jax_drb.cli import main
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
    assert "restart supported: yes" in captured

    restart = load_restart_bundle(restart_path)
    assert restart.case_name == "diffusion_restartable"
    assert restart.completed_steps == 3
    assert restart.current_time == 15.0
    assert tuple(sorted(restart.state_variables)) == ("Nh", "Ph")

    run_log = json.loads((output_dir / "diffusion_restartable_run_log.json").read_text(encoding="utf-8"))
    assert run_log["restart_supported"] is True
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
