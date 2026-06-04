from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_recycling_jvp_promotion_gate.py"
    spec = importlib.util.spec_from_file_location("run_recycling_jvp_promotion_gate", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recycling_jvp_promotion_gate_builds_single_ion_command() -> None:
    module = _load_module()
    gate_case = module.GATE_CASES["recycling_1d_one_step"]
    command = module._build_case_command(
        gate_case,
        reference_root=Path("/tmp/reference-root"),
        python_executable="python",
    )

    assert command[:2] == ["python", str(module.REPO_ROOT / "scripts" / "compare_recycling_transient_modes.py")]
    assert command[command.index("--case") + 1] == "recycling_1d_one_step"
    assert command[command.index("--reference-root") + 1] == "/tmp/reference-root"
    assert command.count("--mode") == 2
    assert "bdf" in command
    assert "bdf_fixed_full_field_jvp" in command
    assert "--diagnostics-only" in command
    assert "--require-fixed-jvp-diagnostics" in command
    assert command[command.index("--require-bdf-pairwise-max") + 1] == "1.00000000e-05"
    assert command[command.index("--mode-timeout-seconds") + 1] == "150"
    assert command.count("--field") == 3
    assert "Pe" in command
    assert "Nd+" in command
    assert "Pd+" in command


def test_recycling_jvp_promotion_gate_defaults_to_all_cases() -> None:
    module = _load_module()

    assert tuple(gate.case for gate in module._selected_cases(())) == (
        "recycling_1d_one_step",
        "recycling_dthe_one_step",
    )
    assert tuple(gate.case for gate in module._selected_cases(("recycling_dthe_one_step",))) == (
        "recycling_dthe_one_step",
    )
