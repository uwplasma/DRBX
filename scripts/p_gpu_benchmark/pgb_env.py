"""Process environment of the P GPU benchmark: set before ``jax`` is imported (jax-free module).

``gpu``: JAX on the node's CUDA devices; ``JAX_PLATFORMS`` is never set (and a ``JAX_PLATFORMS=cpu`` inherited from the shell is an
error, not silently overridden). ``cpu``: the same node's CPUs (``JAX_PLATFORMS=cpu``); the thread count comes from
``--cpu-threads`` or the environment variable :data:`THREADS_ENV` chosen by the remote setup (unset: no cap, the process affinity).
"""
from __future__ import annotations

import os

THREADS_ENV = "P_GPU_BENCH_CPU_THREADS"
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NPROC")


def resolve_threads(cpu_threads: int | None, environ=None) -> int | None:
    environ = os.environ if environ is None else environ
    if cpu_threads is not None:
        return int(cpu_threads)
    v = environ.get(THREADS_ENV)
    return int(v) if v else None


def build_env(backend: str, cpu_threads: int | None = None, environ: dict | None = None) -> dict:
    """The environment variables to set (``{name: value}``; ``None`` = unchanged) for ``backend``; pure function of ``environ``."""
    environ = dict(os.environ if environ is None else environ)
    if backend not in ("gpu", "cpu"):
        raise ValueError(f"backend must be 'gpu' or 'cpu', got {backend!r}")
    out = {"JAX_ENABLE_X64": "true"}
    if backend == "cpu":
        out["JAX_PLATFORMS"] = "cpu"
        n = resolve_threads(cpu_threads, environ)
        if n is not None:
            for k in _THREAD_VARS:
                out[k] = str(n)
            flags = environ.get("XLA_FLAGS", "")
            if "intra_op_parallelism_threads" not in flags:
                flags = (flags + f" --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={n}").strip()
            out["XLA_FLAGS"] = flags
    else:
        if environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu":
            raise SystemExit("--backend gpu but JAX_PLATFORMS=cpu is set in the environment: unset it")
        if environ.get("CUDA_VISIBLE_DEVICES") == "":
            raise SystemExit("--backend gpu but CUDA_VISIBLE_DEVICES is empty (no device visible): unset it")
        out.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "false"))
    return out


def configure(backend: str, cpu_threads: int | None = None) -> dict:
    """Apply :func:`build_env` to ``os.environ`` (call before importing jax); returns what was set."""
    env = build_env(backend, cpu_threads)
    os.environ.update(env)
    return env


def affinity_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1
