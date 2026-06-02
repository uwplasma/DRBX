from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_profile_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "profile_curated_case.py"
    spec = importlib.util.spec_from_file_location("profile_curated_case", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile_script = _load_profile_script()


def test_json_ready_diagnostics_preserves_native_run_counters() -> None:
    result = SimpleNamespace(
        diagnostics={
            "recycling_transient_solver_mode": "bdf_fixed_full_field_jvp",
            "bdf_jacobian_mode": "jvp",
            "bdf_jacobian_base_rhs_evaluation_count": np.int64(0),
            "bdf_jacobian_callback_seconds": np.float64(0.25),
            "bdf_jvp_batch_size": None,
        }
    )

    diagnostics = profile_script._json_ready_diagnostics(result)

    assert diagnostics == {
        "recycling_transient_solver_mode": "bdf_fixed_full_field_jvp",
        "bdf_jacobian_mode": "jvp",
        "bdf_jacobian_base_rhs_evaluation_count": 0,
        "bdf_jacobian_callback_seconds": 0.25,
        "bdf_jvp_batch_size": None,
    }
