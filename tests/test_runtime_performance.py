from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from drbx.runtime import resolve_host_device_count


def _runtime_cache_payload(
    repo_root: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    script = textwrap.dedent(
        """
        import json
        import jax
        from drbx.runtime import configure_jax_runtime

        cache_dir = configure_jax_runtime(precision="float64")
        print(json.dumps({
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "configured_cache_dir": jax.config.jax_compilation_cache_dir,
            "cache_enabled": jax.config.jax_enable_compilation_cache,
            "min_compile_time": jax.config.jax_persistent_cache_min_compile_time_secs,
            "min_entry_size": jax.config.jax_persistent_cache_min_entry_size_bytes,
        }, sort_keys=True))
        """
    )
    env = dict(os.environ)
    for name in (
        "DRBX_CACHE_DIR",
        "DRBX_DISABLE_COMPILATION_CACHE",
        "DRBX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "DRBX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
        "XDG_CACHE_HOME",
    ):
        env.pop(name, None)
    env.update(environment)
    env["PYTHONPATH"] = str(repo_root / "src")
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
    return json.loads(completed.stdout.strip().splitlines()[-1])


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


def test_runtime_cache_dir_precedence_and_default(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    drbx_cache = tmp_path / "drbx-cache"
    jax_cache = tmp_path / "jax-cache"
    payload = _runtime_cache_payload(
        repo_root,
        {
            "DRBX_CACHE_DIR": str(drbx_cache),
            "JAX_COMPILATION_CACHE_DIR": str(jax_cache),
        },
    )
    assert payload["cache_dir"] == str(drbx_cache)
    assert payload["configured_cache_dir"] == str(drbx_cache)

    payload = _runtime_cache_payload(
        repo_root,
        {"JAX_COMPILATION_CACHE_DIR": str(jax_cache)},
    )
    assert payload["cache_dir"] == str(jax_cache)
    assert payload["configured_cache_dir"] == str(jax_cache)

    xdg_cache = tmp_path / "xdg-cache"
    payload = _runtime_cache_payload(
        repo_root,
        {"XDG_CACHE_HOME": str(xdg_cache)},
    )
    expected_default = xdg_cache / "drbx" / "jax_compilation_cache"
    assert payload["cache_dir"] == str(expected_default)
    assert payload["configured_cache_dir"] == str(expected_default)


def test_runtime_cache_threshold_precedence_and_defaults(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = tmp_path / "cache"
    payload = _runtime_cache_payload(
        repo_root,
        {
            "DRBX_CACHE_DIR": str(cache_dir),
            "DRBX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "2.5",
            "DRBX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "4096",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "7.5",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "8192",
        },
    )
    assert payload["min_compile_time"] == 2.5
    assert payload["min_entry_size"] == 4096

    payload = _runtime_cache_payload(
        repo_root,
        {
            "DRBX_CACHE_DIR": str(cache_dir),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "7.5",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "8192",
        },
    )
    assert payload["min_compile_time"] == 7.5
    assert payload["min_entry_size"] == 8192

    payload = _runtime_cache_payload(
        repo_root,
        {"DRBX_CACHE_DIR": str(cache_dir)},
    )
    assert payload["min_compile_time"] == 0.0
    assert payload["min_entry_size"] == 0


def test_runtime_cache_disable_still_wins_over_jax_cache_dir(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = _runtime_cache_payload(
        repo_root,
        {
            "DRBX_DISABLE_COMPILATION_CACHE": "true",
            "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "jax-cache"),
        },
    )
    assert payload["cache_dir"] is None
    assert payload["cache_enabled"] is False
