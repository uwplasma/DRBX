from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


_REPO = Path(__file__).resolve().parents[1]


def _load_script_module(relative_path: str, module_name: str):
    path = _REPO / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_fluid_1d_mms_convergence_report_shows_refinement_improvement() -> None:
    module = _load_script_module("scripts/run_fluid_1d_mms_convergence.py", "fluid_1d_mms_convergence")

    report = module.build_convergence_report(
        resolutions=(32, 64, 128),
        timestep=0.05,
        steps=2,
        substeps=20,
    )

    runs = report["runs"]
    assert runs[1]["errors"]["density_l2"] < runs[0]["errors"]["density_l2"]
    assert runs[2]["errors"]["density_l2"] < runs[1]["errors"]["density_l2"]
    assert runs[1]["errors"]["pressure_l2"] < runs[0]["errors"]["pressure_l2"]
    assert runs[2]["errors"]["pressure_l2"] < runs[1]["errors"]["pressure_l2"]
    assert runs[1]["errors"]["momentum_l2"] < runs[0]["errors"]["momentum_l2"]
    assert runs[2]["errors"]["momentum_l2"] < runs[1]["errors"]["momentum_l2"]

    first_order = report["observed_orders"][0]
    assert first_order["density_order"] > 1.5
    assert first_order["pressure_order"] > 1.5
    assert first_order["momentum_order"] > 1.5


def test_fluid_1d_mms_convergence_script_writes_json_report(tmp_path: Path) -> None:
    output_path = tmp_path / "mms_convergence.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "run_fluid_1d_mms_convergence.py"),
            "--resolution",
            "32",
            "--resolution",
            "64",
            "--output",
            str(output_path),
        ],
        cwd=_REPO,
        env={
            **os.environ,
            **{
                "PYTHONPATH": str(_REPO / "src"),
                "JAX_ENABLE_X64": "true",
                "DKX_PRECISION": "float64",
            },
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["case"] == "fluid_1d_mms_convergence"
    assert payload["resolutions"] == [32, 64]
