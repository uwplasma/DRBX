from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import jax


@contextmanager
def jax_trace(outdir: str | Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        jax.profiler.start_trace(str(outdir))
        yield
    finally:
        jax.profiler.stop_trace()


def save_device_memory_profile(outdir: str | Path, name: str = "memory_profile.pb") -> Path | None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    try:
        jax.profiler.save_device_memory_profile(str(path))
    except Exception:
        return None
    return path


def save_hlo(lowered: Any, outdir: str | Path, name: str = "compiled") -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    try:
        hlo = lowered.compiler_ir(dialect="hlo").as_text()
        path = outdir / f"{name}.hlo.txt"
        path.write_text(hlo)
        written["hlo"] = path
    except Exception:
        pass
    try:
        stable = lowered.compiler_ir(dialect="stablehlo").as_text()
        path = outdir / f"{name}.stablehlo.txt"
        path.write_text(stable)
        written["stablehlo"] = path
    except Exception:
        pass
    if not written:
        try:
            mlir = lowered.as_text()
            path = outdir / f"{name}.mlir.txt"
            path.write_text(mlir)
            written["mlir"] = path
        except Exception:
            pass
    return written


def save_compile_stats(lowered: Any, outdir: str | Path, name: str = "compile_stats.json") -> Path | None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        compiled = lowered.compile()
        stats = compiled.runtime_executable().size_in_bytes()
        payload = {
            "executable_bytes": int(stats),
            "backend": jax.default_backend(),
            "devices": [d.device_kind for d in jax.devices()],
        }
    except Exception:
        return None
    path = outdir / name
    path.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True))
    return path
