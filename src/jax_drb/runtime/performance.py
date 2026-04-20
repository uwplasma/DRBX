from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from ..config.boutinp import BoutConfig

_VALID_PRECISIONS = {"float32", "float64"}


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
        candidate = os.environ.get("JAX_DRB_PRECISION", "float64")
    normalized = str(candidate).strip().lower()
    if normalized not in _VALID_PRECISIONS:
        raise ValueError(f"Unsupported precision {candidate!r}; expected one of {sorted(_VALID_PRECISIONS)}")
    return normalized


def configure_jax_runtime(*, precision: str | None = None) -> Path | None:
    resolved_precision = resolve_runtime_precision(requested=precision)
    os.environ["JAX_DRB_PRECISION"] = resolved_precision
    if os.environ.get("JAX_DRB_DISABLE_COMPILATION_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}:
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
        if "JAX_DRB_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" in os.environ:
            min_compile_time = float(os.environ["JAX_DRB_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"])
        else:
            min_compile_time = 0.0
        if "JAX_DRB_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES" in os.environ:
            min_entry_size = int(os.environ["JAX_DRB_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"])
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


def _compilation_cache_dir() -> Path:
    override = os.environ.get("JAX_DRB_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return _default_user_cache_root() / "jax_drb" / "jax_compilation_cache"


def _default_user_cache_root() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser()
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches"
    return Path.home() / ".cache"
