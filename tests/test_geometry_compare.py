import re
import subprocess
import sys
from pathlib import Path

import pytest


def _parse_rel_errors(output: str) -> dict[str, float]:
    rels = {}
    for line in output.strip().splitlines():
        m = re.match(r"(\w+): .*rel_error': ([0-9.eE+-]+)", line)
        if m:
            rels[m.group(1)] = float(m.group(2))
    return rels


def test_geometry_compare_salpha_metrics():
    script = Path("/Users/rogerio/local/jax_drb/tools/compare_geometry_metrics.py")
    config = Path("/Users/rogerio/local/jax_drb/configs/benchmarks/salpha_hermes_match.toml")
    grid = Path(
        "/Users/rogerio/local/jax_drb/external/hermes-3/examples_min/salpha_grid/salpha.nc"
    )

    if not script.exists() or not config.exists() or not grid.exists():
        pytest.skip("Required geometry comparison inputs missing")

    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(config),
        "--bout-grid",
        str(grid),
        "--mapping",
        "canonical",
        "--x-index",
        "0",
    ]

    out = subprocess.check_output(cmd, text=True)
    rels = _parse_rel_errors(out)

    assert rels.get("dpar_factor", 1.0) < 0.01
    assert rels.get("B", 1.0) < 0.1
    assert rels.get("curv_x", 10.0) < 1.4
    assert rels.get("curv_y", 10.0) < 1.2
