import json
import subprocess
import sys
from pathlib import Path

import pytest


TOLERANCES = {
    "jaxdrb": {
        "B": 1e-12,
        "dpar_factor": 1e-12,
        "curv_x": 1e-12,
        "curv_y": 1e-12,
    },
    "hermes": {
        "B": 0.05,
        "dpar_factor": 0.05,
        "curv_x": 0.25,
        "curv_y": 0.25,
    },
    "gbs": {
        "curv_x": 0.2,
        "curv_y": 0.2,
    },
}


def test_alignment_metrics_regression(tmp_path):
    root = Path("/Users/rogerio/local/jax_drb")
    script = root / "benchmarks/run_alignment.py"
    baseline_file = root / "benchmarks/reference/alignment_metrics.json"

    hermes_grid = root / "external/hermes-3/examples_min/salpha_grid/salpha.nc"
    gbs_results = root / "external/gbs/bin/results_min_00.h5"

    if not script.exists() or not baseline_file.exists():
        pytest.skip("Alignment scripts or baseline metrics missing")

    if not hermes_grid.exists() and not gbs_results.exists():
        pytest.skip("External benchmark outputs missing")

    output_dir = tmp_path / "alignment"
    cmd = [
        sys.executable,
        str(script),
        "--compare-only",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.check_call(cmd)

    metrics_file = output_dir / "alignment_metrics.json"
    assert metrics_file.exists()

    baseline = json.loads(baseline_file.read_text())
    current = json.loads(metrics_file.read_text())

    for code, base_metrics in baseline.items():
        if code not in current:
            continue
        for key, base_stats in base_metrics.items():
            if key not in current[code]:
                continue
            if "rel_error" not in base_stats or "rel_error" not in current[code][key]:
                continue
            diff = abs(current[code][key]["rel_error"] - base_stats["rel_error"])
            tol = TOLERANCES.get(code, {}).get(key, 0.2)
            assert diff <= tol
