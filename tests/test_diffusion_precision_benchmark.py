from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "diffusion_precision_benchmark.py"
    spec = importlib.util.spec_from_file_location("diffusion_precision_benchmark", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_precision_input_rewrites_runtime_precision_only() -> None:
    module = _load_module()
    template = """
    [time]
    nout = 3
    timestep = 5.0

    [runtime]
    precision = "float64"

    [mesh]
    nx = 16
    """.strip()

    updated = module.build_precision_input(template, "float32")

    assert 'precision = "float32"' in updated
    assert 'precision = "float64"' not in updated
    assert "nx = 16" in updated
