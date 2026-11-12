from __future__ import annotations

import gc
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


@pytest.fixture(autouse=True, scope="module")
def _release_compiled_programs():
    """Drop the compiled programs of a finished test module.

    The fast suite compiles hundreds of programs in one process; on Python 3.10 (jax 0.6.2) it reached 11 GB and was
    killed on a hosted runner.  Clearing between modules keeps the process near the memory of its largest module.
    """

    yield
    import jax

    jax.clear_caches()
    gc.collect()


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
