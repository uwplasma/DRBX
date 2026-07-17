"""Correctness tests for the sharded two-field RK4 step."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tests import fci_sharded_2field_case as case


def test_single_device_sharded_step_matches_direct() -> None:
    """A (1, 1, 1) shard_map step must reproduce the direct unsharded step."""

    result = case.run_equivalence_case(
        shape=(12, 8, 8),
        shard_counts=(1, 1, 1),
        steps=2,
        dt=1.0e-3,
    )
    assert result["direct_density_max"] > 0.5
    assert result["max_abs_diff"] < 1.0e-13


def test_multi_device_sharded_step_matches_direct_in_subprocess() -> None:
    """A (2, 2, 1) four-device trajectory must match the direct trajectory.

    The XLA host device count must be configured before JAX is imported, so
    the case runs in a subprocess with ``--xla_force_host_platform_device_count``.
    """

    env = dict(os.environ)
    existing_flags = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = f"{existing_flags} --xla_force_host_platform_device_count=4".strip()
    env.pop("DRBX_HOST_DEVICE_COUNT", None)

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(case.__file__).resolve()),
            "--shape", "16", "16", "8",
            "--shard-counts", "2", "2", "1",
            "--steps", "5",
            "--dt", "1e-3",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"subprocess failed with code {completed.returncode}:\n{completed.stderr[-4000:]}"
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    print(f"multi-device max-abs difference: {payload['max_abs_diff']:.3e}")
    assert payload["max_abs_diff"] < 1.0e-12
