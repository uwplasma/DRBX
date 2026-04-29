from __future__ import annotations

from pathlib import Path

import pytest

from jax_drb.runtime import configure_jax_runtime
from jax_drb.runtime.artifacts import ensure_reference_baselines
from jax_drb.reference.paths import default_reference_root, repo_root


# Keep the default test runtime aligned with the documented package default so
# float64-capable paths do not silently truncate when a test imports jax.numpy
# before touching the native runtime helpers.
configure_jax_runtime(precision="float64")

REPO_ROOT = repo_root()
REFERENCE_ROOT = default_reference_root()
REFERENCE_BINARY_ROOT = REFERENCE_ROOT
BASELINE_REFERENCE_DIR = REPO_ROOT / "references" / "baselines" / "reference"
BASELINE_ARRAY_DIR = REPO_ROOT / "references" / "baselines" / "reference_arrays"


ensure_reference_baselines(root=REPO_ROOT)


@pytest.fixture(autouse=True)
def _reset_default_jax_runtime_precision() -> None:
    """Keep tests isolated from in-process CLI precision switches."""

    configure_jax_runtime(precision="float64")
    yield
    configure_jax_runtime(precision="float64")


def reference_input(relative_path: str) -> Path:
    if REFERENCE_ROOT is None:
        raise FileNotFoundError(
            "Set JAX_DRB_REFERENCE_ROOT to a checkout containing the external benchmark decks."
        )
    return REFERENCE_ROOT / relative_path
