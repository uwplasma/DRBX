from __future__ import annotations

import shutil
import subprocess

import pytest

from drbx.runtime import configure_jax_runtime
from drbx.runtime.paths import repo_root


# Keep the default test runtime aligned with the documented package default so
# float64-capable paths do not silently truncate when a test imports jax.numpy
# before touching the native runtime helpers.
configure_jax_runtime(precision="float64")

REPO_ROOT = repo_root()


@pytest.fixture(autouse=True)
def _reset_default_jax_runtime_precision() -> None:
    """Keep tests isolated from in-process CLI precision switches."""

    configure_jax_runtime(precision="float64")
    yield
    configure_jax_runtime(precision="float64")


@pytest.fixture
def require_working_ffmpeg() -> None:
    """Skip movie tests unless ffmpeg is present and can actually start."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is unavailable")
    result = subprocess.run(
        [ffmpeg, "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg is installed but cannot start: {result.stderr.strip()}")
