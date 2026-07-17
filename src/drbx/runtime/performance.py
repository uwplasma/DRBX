from __future__ import annotations

import os
import platform
import shlex
import sys
from pathlib import Path
from typing import Any

from ..config.boutinp import BoutConfig

_VALID_PRECISIONS = {"float32", "float64"}
_HOST_DEVICE_FLAG_PREFIX = "--xla_force_host_platform_device_count="


def resolve_runtime_precision(
    *,
    requested: str | None = None,
    config: BoutConfig | None = None,
) -> str:
    candidate = requested
    if candidate is None and config is not None and config.has_option("runtime", "precision"):
        parsed = config.parsed("runtime", "precision")
        candidate = str(parsed)
    if candidate is None:
        candidate = os.environ.get("DRBX_PRECISION", "float64")
    normalized = str(candidate).strip().lower()
    if normalized not in _VALID_PRECISIONS:
        raise ValueError(f"Unsupported precision {candidate!r}; expected one of {sorted(_VALID_PRECISIONS)}")
    return normalized


def resolve_host_device_count(*, requested: int | str | None = None) -> int | None:
    candidate = requested
    if candidate is None:
        candidate = os.environ.get("DRBX_HOST_DEVICE_COUNT")
    if candidate is None:
        return None
    normalized = str(candidate).strip()
    if not normalized or normalized == "0":
        return None
    try:
        value = int(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported host device count {candidate!r}; expected a positive integer") from exc
    if value <= 0:
        raise ValueError(f"Unsupported host device count {candidate!r}; expected a positive integer")
    return value


def configure_jax_runtime(
    *,
    precision: str | None = None,
    host_device_count: int | str | None = None,
) -> Path | None:
    resolved_precision = resolve_runtime_precision(requested=precision)
    os.environ["DRBX_PRECISION"] = resolved_precision
    resolved_host_device_count = resolve_host_device_count(requested=host_device_count)
    if resolved_host_device_count is not None:
        os.environ["DRBX_HOST_DEVICE_COUNT"] = str(resolved_host_device_count)
        _configure_host_device_count_xla_flags(resolved_host_device_count)
    if os.environ.get("DRBX_DISABLE_COMPILATION_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}:
        cache_dir = None
    else:
        cache_dir = _compilation_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

    import jax
    jax.config.update("jax_enable_x64", resolved_precision == "float64")
    if cache_dir is not None:
        from jax.experimental.compilation_cache import compilation_cache as compilation_cache

        jax.config.update("jax_enable_compilation_cache", True)
        jax.config.update("jax_compilation_cache_dir", str(cache_dir))
        if "DRBX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" in os.environ:
            min_compile_time = float(os.environ["DRBX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"])
        else:
            min_compile_time = 0.0
        if "DRBX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES" in os.environ:
            min_entry_size = int(os.environ["DRBX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"])
        else:
            min_entry_size = 0
        jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", min_entry_size)
        compilation_cache.set_cache_dir(str(cache_dir))
    return cache_dir


def runtime_numpy_dtype(*, precision: str | None = None) -> Any:
    import numpy as np

    resolved = resolve_runtime_precision(requested=precision)
    return np.float32 if resolved == "float32" else np.float64


def runtime_jax_dtype(*, precision: str | None = None) -> Any:
    import jax.numpy as jnp

    resolved = resolve_runtime_precision(requested=precision)
    return jnp.float32 if resolved == "float32" else jnp.float64


def runtime_parallel_summary() -> dict[str, Any]:
    import jax

    requested_host_devices = resolve_host_device_count()
    xla_flags = os.environ.get("XLA_FLAGS", "")
    configured_host_devices = _extract_host_device_count_from_flags(xla_flags)
    return {
        "backend": jax.default_backend(),
        "cpu_count": os.cpu_count(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "devices": [str(device) for device in jax.devices()],
        "requested_host_device_count": requested_host_devices,
        "configured_host_device_count": configured_host_devices,
        "explicit_host_device_parallelism_enabled": (
            jax.default_backend() == "cpu" and jax.local_device_count() > 1
        ),
        "xla_flags": xla_flags,
    }


def _compilation_cache_dir() -> Path:
    override = os.environ.get("DRBX_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return _default_user_cache_root() / "drbx" / "jax_compilation_cache"


def _default_user_cache_root() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser()
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches"
    return Path.home() / ".cache"


def _configure_host_device_count_xla_flags(host_device_count: int) -> None:
    existing_flags = os.environ.get("XLA_FLAGS", "")
    configured_count = _extract_host_device_count_from_flags(existing_flags)
    if configured_count == host_device_count:
        return
    if "jax" in sys.modules:
        raise RuntimeError(
            "DRBX_HOST_DEVICE_COUNT must be set before importing jax/drbx so CPU devices can be configured."
        )
    tokens = [token for token in shlex.split(existing_flags) if not token.startswith(_HOST_DEVICE_FLAG_PREFIX)]
    tokens.append(f"{_HOST_DEVICE_FLAG_PREFIX}{host_device_count}")
    os.environ["XLA_FLAGS"] = " ".join(tokens)


def _extract_host_device_count_from_flags(flags: str) -> int | None:
    for token in shlex.split(flags):
        if token.startswith(_HOST_DEVICE_FLAG_PREFIX):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return None
    return None
