from __future__ import annotations

from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.reference.paths import repo_root, require_reference_root


# Keep the default test runtime aligned with the documented package default so
# float64-capable paths do not silently truncate when a test imports jax.numpy
# before touching the native runtime helpers.
configure_jax_runtime(precision="float64")

REPO_ROOT = repo_root()
REFERENCE_ROOT = require_reference_root()
REFERENCE_BINARY_ROOT = REFERENCE_ROOT
BASELINE_REFERENCE_DIR = REPO_ROOT / "references" / "baselines" / "reference"
BASELINE_ARRAY_DIR = REPO_ROOT / "references" / "baselines" / "reference_arrays"


def reference_input(relative_path: str) -> Path:
    return REFERENCE_ROOT / relative_path
