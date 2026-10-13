"""Honest timing and memory probes (JAX): AOT compile time separate, warm-up, ``block_until_ready`` per repetition, medians."""
from __future__ import annotations

import os
import resource
import sys
import time

import jax
import numpy as np


def block(x):
    return jax.block_until_ready(x)


def time_jit(fn, *args, reps: int = 20, warmup: int = 2) -> tuple[dict, object]:
    """AOT-compile ``fn(*args)`` (``compile_s``), call it ``warmup`` times, then ``reps`` timed calls, each blocked.

    Returns ``(stats, last_output)``; ``stats`` has ``compile_s, first_call_s, median_s, min_s, max_s, mean_s, std_s, reps``.
    The arguments are traced inputs (pytrees are fine), so factors are not compile-time constants.
    """
    jf = jax.jit(fn)
    t0 = time.perf_counter()
    compiled = jf.lower(*args).compile()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = block(compiled(*args))
    first = time.perf_counter() - t0
    for _ in range(max(warmup - 1, 0)):
        out = block(compiled(*args))
    ts = []
    for _ in range(int(reps)):
        t0 = time.perf_counter()
        out = block(compiled(*args))
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts)
    return dict(compile_s=compile_s, first_call_s=first, median_s=float(np.median(a)), min_s=float(a.min()), max_s=float(a.max()),
                mean_s=float(a.mean()), std_s=float(a.std()), reps=int(reps)), out


def peak_rss_gib() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**30 if sys.platform == "darwin" else r / 2**20


def current_rss_gib() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except Exception:                                                              # noqa: BLE001
        return peak_rss_gib()


def host_mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:                                                              # noqa: BLE001
        return None


def device_memory(dev=None) -> dict:
    """``{bytes_in_use, peak_bytes_in_use, bytes_limit}`` of the (first) device, or ``{}`` where the runtime has no stats (CPU)."""
    dev = dev or jax.devices()[0]
    try:
        s = dev.memory_stats()
    except Exception:                                                              # noqa: BLE001
        s = None
    if not s:
        return {}
    return {k: int(s[k]) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if k in s}


def free_budget_bytes(backend: str, override_gb: float | None = None) -> int | None:
    """Memory (bytes) a case may use: free device memory on GPU, available host memory on CPU; ``None``: unknown (no skipping)."""
    if override_gb is not None:
        return int(override_gb * 2**30)
    if backend == "gpu":
        m = device_memory()
        if "bytes_limit" in m:
            return m["bytes_limit"] - m.get("bytes_in_use", 0)
        return None
    return host_mem_available_bytes()


def tree_nbytes(tree) -> int:
    return int(sum(getattr(x, "nbytes", 0) for x in jax.tree_util.tree_leaves(tree)))


def is_oom(exc: BaseException) -> bool:
    s = f"{type(exc).__name__}: {exc}".lower()
    return isinstance(exc, MemoryError) or any(k in s for k in ("resource_exhausted", "out of memory", "oom", "failed to allocate",
                                                               "cannot allocate"))


def environment_record() -> dict:
    """``jax.devices()``, platform, device kind and memory of the process (call after the backend is chosen)."""
    devs = jax.devices()
    rec = {"default_backend": jax.default_backend(), "device_count": len(devs),
           "devices": [{"id": d.id, "platform": d.platform, "kind": getattr(d, "device_kind", None), "str": str(d)} for d in devs],
           "memory": device_memory(devs[0]), "jax": jax.__version__, "x64": bool(jax.config.jax_enable_x64),
           "matmul_precision": str(jax.config.jax_default_matmul_precision)}
    try:
        import jaxlib
        rec["jaxlib"] = jaxlib.__version__
    except Exception:                                                              # noqa: BLE001
        pass
    return rec
