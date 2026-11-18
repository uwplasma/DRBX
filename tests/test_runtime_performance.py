from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from drbx.runtime import resolve_host_device_count


def test_resolve_host_device_count_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DRBX_HOST_DEVICE_COUNT", raising=False)

    assert resolve_host_device_count() is None


def test_resolve_host_device_count_validates_positive_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRBX_HOST_DEVICE_COUNT", "4")
    assert resolve_host_device_count() == 4

    monkeypatch.setenv("DRBX_HOST_DEVICE_COUNT", "0")
    assert resolve_host_device_count() is None

    monkeypatch.setenv("DRBX_HOST_DEVICE_COUNT", "-2")
    with pytest.raises(ValueError):
        resolve_host_device_count()

    monkeypatch.setenv("DRBX_HOST_DEVICE_COUNT", "abc")
    with pytest.raises(ValueError):
        resolve_host_device_count()


def test_runtime_parallel_summary_respects_host_device_count_in_fresh_process() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import json
        from drbx.runtime import runtime_parallel_summary
        print(json.dumps(runtime_parallel_summary(), sort_keys=True))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    env["DRBX_HOST_DEVICE_COUNT"] = "3"
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


def test_precision_is_resolved_from_argument_then_config_then_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from drbx.config.boutinp import parse_bout_input
    from drbx.runtime.performance import resolve_runtime_precision, runtime_numpy_dtype

    monkeypatch.delenv("DRBX_PRECISION", raising=False)
    config = parse_bout_input("[runtime]\nprecision = float32\n")

    assert resolve_runtime_precision() == "float64"
    assert resolve_runtime_precision(config=config) == "float32"
    assert resolve_runtime_precision(requested=" FLOAT64 ", config=config) == "float64"
    monkeypatch.setenv("DRBX_PRECISION", "float32")
    assert resolve_runtime_precision() == "float32"
    assert runtime_numpy_dtype() is np.float32
    with pytest.raises(ValueError, match="Unsupported precision"):
        resolve_runtime_precision(requested="float16")


def test_host_device_flag_replaces_only_its_own_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from drbx.runtime import performance

    flag = performance._HOST_DEVICE_FLAG_PREFIX
    monkeypatch.setenv("XLA_FLAGS", f"--xla_a=1 {flag}2")
    performance._configure_host_device_count_xla_flags(2)  # already configured: no-op even after jax import
    assert os.environ["XLA_FLAGS"] == f"--xla_a=1 {flag}2"
    with pytest.raises(RuntimeError, match="before importing jax"):
        performance._configure_host_device_count_xla_flags(4)

    monkeypatch.delitem(sys.modules, "jax")  # restored by monkeypatch at teardown
    performance._configure_host_device_count_xla_flags(4)
    assert os.environ["XLA_FLAGS"] == f"--xla_a=1 {flag}4"
    assert performance._extract_host_device_count_from_flags(f"{flag}x") is None
