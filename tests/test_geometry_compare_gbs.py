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


def test_geometry_compare_gbs_shapes():
    script = Path("/Users/rogerio/local/jax_drb/tools/compare_geometry_gbs.py")
    config = Path("/Users/rogerio/local/jax_drb/configs/benchmarks/salpha_gbs_match.toml")
    gbs = Path("/Users/rogerio/local/jax_drb/external/gbs/bin/results_min_00.h5")

    if not script.exists() or not config.exists() or not gbs.exists():
        pytest.skip("Required GBS geometry comparison inputs missing")

    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(config),
        "--gbs-file",
        str(gbs),
        "--mapping",
        "canonical",
    ]

    out = subprocess.check_output(cmd, text=True)
    rels = _parse_rel_errors(out)

    assert rels.get("curv_x", 10.0) < 0.8
    assert rels.get("curv_y", 10.0) < 0.8
