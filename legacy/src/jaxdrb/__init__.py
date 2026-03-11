from __future__ import annotations

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

__all__ = ["__version__"]

__version__ = "0.1.0"
