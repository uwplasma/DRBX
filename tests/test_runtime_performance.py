from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from dkx.runtime import resolve_host_device_count


def test_resolve_host_device_count_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DKX_HOST_DEVICE_COUNT", raising=False)

    assert resolve_host_device_count() is None


def test_resolve_host_device_count_validates_positive_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DKX_HOST_DEVICE_COUNT", "4")
    assert resolve_host_device_count() == 4

    monkeypatch.setenv("DKX_HOST_DEVICE_COUNT", "0")
    assert resolve_host_device_count() is None

    monkeypatch.setenv("DKX_HOST_DEVICE_COUNT", "-2")
    with pytest.raises(ValueError):
        resolve_host_device_count()

    monkeypatch.setenv("DKX_HOST_DEVICE_COUNT", "abc")
    with pytest.raises(ValueError):
        resolve_host_device_count()


def test_runtime_parallel_summary_respects_host_device_count_in_fresh_process() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import json
        from dkx.runtime import runtime_parallel_summary
        print(json.dumps(runtime_parallel_summary(), sort_keys=True))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    env["DKX_HOST_DEVICE_COUNT"] = "3"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["requested_host_device_count"] == 3
    assert payload["configured_host_device_count"] == 3
    assert payload["local_device_count"] == 3
    assert payload["explicit_host_device_parallelism_enabled"] is True
