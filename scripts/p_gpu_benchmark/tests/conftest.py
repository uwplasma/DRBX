"""Test setup: CPU JAX with x64 (set before jax is imported), the benchmark directory on ``sys.path``, the P10 inputs fixture."""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ.setdefault("NPROC", "2")
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import pytest                                                                            # noqa: E402

DEFAULT_INPUTS = HERE.parents[3] / "work" / "p10_campaign_inputs_20261007"


@pytest.fixture(scope="session")
def inputs_dir():
    p = Path(os.environ.get("P_GPU_BENCH_INPUTS", DEFAULT_INPUTS))
    if not (p / "laplacian" / "filtered" / "N32" / "laplacian_metric.npz").is_file():
        pytest.skip(f"N32 filtered laplacian metric not found under {p} (set P_GPU_BENCH_INPUTS)")
    return p
