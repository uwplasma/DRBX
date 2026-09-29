"""One chunked, resumable process-pool runner (P08 step-1 task 4, "Runner"
half -- see ``work/p08_step1_consolidation_design_20260928/design.md``
sections 3 and 6).

Lineage: ``scripts/p05n_field_derived_global/campaign.py`` /
``scripts/p06n_field_derived_global/campaign.py`` (AST-identical
``sha``/``digest``/``write``/``save``/``rss``/``lock``/``valid_unit``/
``run_stage`` helpers -- see design section 1's "Runner / lock" row). This
module generalizes that lineage into small, callable pieces
(``build_artifact.py`` supplies the actual row-building ``compute``
function); it carries no P08-specific row logic of its own.

Guarantees kept from the campaign lineage:

* **plan** -- a fixed, deterministic list of ``{stage, n, start, stop}``
  units, written once and checked byte-identical on every resume.
* **valid_unit** -- a unit's chunk file and JSON receipt are both required;
  the receipt's ``identity``/``unit`` must match exactly, and the chunk
  file's sha256 must match the receipt's recorded ``sha256`` (a corrupted or
  half-written chunk is never silently reused).
* **one writer** -- every chunk/receipt file is written to a per-process
  temporary name and atomically renamed into place (``os.replace``), so a
  concurrent reader never observes a partial file.
* **lock** -- ``lock(output)`` takes an exclusive, non-blocking advisory
  lock on ``<output>/.runner.lock`` for the duration of a build, refusing a
  second concurrent invocation against the same output root.
* **spawn** -- the process pool always uses the ``spawn`` multiprocessing
  context (never ``fork``), the same JAX-safety requirement the campaign
  lineage enforces.
* **peak RSS per worker** -- every unit's receipt records this worker's own
  ``peak_rss_gib`` and the ``rss_method`` that produced it. ``run_stage``
  resets the worker's high-water mark at the start of each unit (Linux:
  writing ``5`` to ``/proc/self/clear_refs``) and the receipt reads
  ``VmHWM`` from ``/proc/self/status`` (``"vmhwm_reset"``); a worker's
  ``ru_maxrss`` can otherwise report the controller's high-water mark (every
  N48 unit reported the controller's 29.3 GiB). Where ``/proc`` is unavailable (macOS) it
  falls back to ``resource.getrusage(RUSAGE_SELF).ru_maxrss``
  (``"ru_maxrss"``, scaled for the platform). ``run_stage``'s summary reports
  the maximum across every unit executed in this invocation
  (``peak_worker_rss_gib``) and, separately, the controller process's own peak
  (``controller_peak_rss_gib``).
* **CPU backend check** -- ``require_cpu_backend`` mirrors the campaign
  lineage's ``if jax.default_backend() != 'cpu': raise ...``, called once
  per worker process at initialization.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import resource
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path

import numpy as np

__all__ = [
    "sha256_file", "sha256_bytes", "digest", "write_json", "save_npz", "peak_rss_gib", "reset_peak_rss",
    "lock", "require_cpu_backend", "chunk_units", "unit_path", "valid_unit", "run_stage",
]


# ---------------------------------------------------------------------------
# Small persistence helpers (mirrors p05n_field_derived_global/campaign.py's
# sha/digest/write/save/rss, generalized to a plain module -- no campaign
# module-global state).
# ---------------------------------------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(obj) -> str:
    """A stable sha256 of a JSON-safe object (dict/list/str/int/float/bool/None)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def write_json(path, obj) -> None:
    """Atomic write: temp file (per-process name) then ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save_npz(path, **arrays) -> None:
    """Atomic ``np.savez`` (uncompressed -- callers needing compression pass
    already-encoded bytes through ``write_bytes``/``os.replace`` themselves)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


# ``/proc/self`` (a module global so tests can point it at a temp dir).
_PROC_SELF = Path("/proc/self")
# How this process's peak RSS is measured: set to "vmhwm_reset" by a
# successful :func:`reset_peak_rss`, otherwise the ``ru_maxrss`` fallback.
_RSS_METHOD = "ru_maxrss"


def _vmhwm_gib() -> float:
    """``VmHWM`` (the process's peak RSS) from ``/proc/self/status``, in GiB."""
    for line in (_PROC_SELF / "status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 2 ** 20  # kB -> GiB
    raise ValueError("no VmHWM line in /proc/self/status")


def _ru_maxrss_gib() -> float:
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)


def reset_peak_rss() -> str:
    """Start a fresh peak-RSS window for this process and return the method
    that will measure it: ``"vmhwm_reset"`` after ``clear_refs`` reset ``VmHWM``
    (Linux), else ``"ru_maxrss"`` (the never-reset, process-lifetime high-water
    mark, which a spawned child inherits from its parent)."""
    global _RSS_METHOD
    try:
        (_PROC_SELF / "clear_refs").write_text("5")
        _vmhwm_gib()
        _RSS_METHOD = "vmhwm_reset"
    except (OSError, ValueError):
        _RSS_METHOD = "ru_maxrss"
    return _RSS_METHOD


def peak_rss_gib() -> float:
    """This process's peak RSS in GiB since the last :func:`reset_peak_rss`
    (``VmHWM``), or its lifetime ``ru_maxrss`` if no reset succeeded."""
    if _RSS_METHOD == "vmhwm_reset":
        try:
            return _vmhwm_gib()
        except (OSError, ValueError):
            pass
    return _ru_maxrss_gib()


def _controller_peak_rss_gib() -> float:
    """The controller's own lifetime peak (never reset): ``VmHWM`` if readable, else ``ru_maxrss``."""
    try:
        return _vmhwm_gib()
    except (OSError, ValueError):
        return _ru_maxrss_gib()


def _run_unit(compute, unit):
    """Worker-side wrapper: reset the peak-RSS window, then run one unit."""
    reset_peak_rss()
    return compute(unit)


@contextmanager
def lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def require_cpu_backend() -> str:
    """Mirrors the campaign lineage's worker-side JAX CPU-backend check."""
    import jax
    backend = jax.default_backend()
    if backend != "cpu":
        raise ValueError(f"JAX CPU backend required, got {backend!r}")
    return backend


# ---------------------------------------------------------------------------
# Chunk planning / unit checkpoints.
#
# A unit may need more than one on-disk array file (e.g. a "cells" unit
# writes both its R1 PointRowChunk and its (possibly empty) Neumann
# companion chunk); ``parts`` names each file, and one JSON receipt per unit
# covers every part's sha256 together, so a unit is only ever "valid" (and
# only ever skipped on resume) when every one of its parts is present and
# intact.
# ---------------------------------------------------------------------------
def chunk_units(stage: str, n: int, count: int, chunk_size: int) -> list[dict]:
    """A deterministic ``{stage, n, start, stop}`` unit list covering ``[0, count)``."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return [{"stage": stage, "n": int(n), "start": j, "stop": min(j + chunk_size, count)}
            for j in range(0, count, chunk_size)]


def unit_path(output, unit, part: str = "chunk") -> Path:
    suffix = "" if part == "chunk" else f".{part}"
    return (Path(output) / "_chunks" / f"N{unit['n']}" / unit["stage"]
            / f"{unit['start']:08d}-{unit['stop'] - 1:08d}{suffix}.npz")


def receipt_path(output, unit) -> Path:
    return (Path(output) / "_chunks" / f"N{unit['n']}" / unit["stage"]
            / f"{unit['start']:08d}-{unit['stop'] - 1:08d}.json")


def valid_unit(output, unit, identity, parts=("chunk",)) -> bool:
    """A unit is valid iff its receipt and every named part's file exist,
    the receipt's identity/unit/part-set match exactly, and every part
    file's sha256 matches the receipt's recorded sha256 for that part.
    Raises (never silently discards) on any other mismatch -- a
    stale/corrupt checkpoint is a build error, not something to quietly
    recompute past."""
    receipt = receipt_path(output, unit)
    if not receipt.exists():
        return False
    item = json.loads(receipt.read_text())
    if item["identity"] != identity or item["unit"] != unit:
        raise ValueError(f"stale checkpoint {receipt}")
    if set(item.get("parts", {})) != set(parts):
        raise ValueError(f"checkpoint part set mismatch at {receipt}")
    for part in parts:
        path = unit_path(output, unit, part)
        if not path.exists():
            return False
        if sha256_file(path) != item["parts"][part]:
            raise ValueError(f"corrupt checkpoint {path}")
    return True


def write_unit(output, unit, identity, *, chunks: dict, started: float, extra: dict | None = None) -> dict:
    """Write every part's npz (``chunks`` = ``{part_name: {array_name: array}}``)
    plus one receipt covering all parts, atomically. Returns the receipt."""
    shas = {}
    for part, arrays in chunks.items():
        path = unit_path(output, unit, part)
        save_npz(path, **arrays)
        shas[part] = sha256_file(path)
    receipt = {"identity": identity, "unit": unit, "parts": shas,
               "seconds": time.time() - started, "peak_rss_gib": peak_rss_gib(),
               "rss_method": _RSS_METHOD, "pid": os.getpid()}
    if extra:
        receipt.update(extra)
    write_json(receipt_path(output, unit), receipt)
    return receipt


# ---------------------------------------------------------------------------
# Process-pool execution.
# ---------------------------------------------------------------------------
def run_stage(output, stage, units, identity, *, compute, initializer, initargs, workers,
              parts=("chunk",), max_tasks_per_worker=None, max_units=None) -> dict:
    """Execute every not-yet-valid unit of ``units`` in a spawn process pool.

    ``compute(unit)`` runs inside a worker (after ``initializer(*initargs)``
    has populated that worker's state) and must itself write every one of
    the unit's ``parts`` (via :func:`write_unit`, so :func:`valid_unit`
    accepts it before ``compute`` returns), then return a JSON-safe dict
    with at least ``seconds`` and ``peak_rss_gib`` (every other key is
    passed through into this stage's execution summary verbatim, for the
    caller's own diagnostics).

    Resumable: already-valid units are skipped and counted as ``resumed``;
    a unit whose ``compute`` raises fails the whole stage (after recording
    ``<output>/failure.json``) rather than silently dropping it.
    """
    todo, resumed = [], 0
    for u in units:
        if valid_unit(output, u, identity, parts=parts):
            resumed += 1
        else:
            todo.append(u)
    if max_units is not None:
        todo = todo[:max_units]

    records = []
    started = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                  initializer=initializer, initargs=initargs,
                                  max_tasks_per_child=max_tasks_per_worker) as pool:
            pending = {}
            iterator = iter(todo)

            def fill():
                for _ in range(max(0, 2 * workers - len(pending))):
                    u = next(iterator, None)
                    if u is None:
                        break
                    pending[pool.submit(_run_unit, compute, u)] = u

            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    u = pending.pop(future)
                    try:
                        records.append(future.result())
                    except Exception as exc:
                        write_json(Path(output) / "failure.json",
                                   {"stage": stage, "unit": u, "error": repr(exc), "time": time.time()})
                        raise
                write_json(Path(output) / "progress.json",
                           {"stage": stage, "completed_this_invocation": len(records),
                            "total_units": len(units), "time": time.time()})
                fill()

    summary = {
        "stage": stage,
        "requested_workers": workers,
        "executed_units": len(records),
        "resumed_units": resumed,
        "total_units": len(units),
        "peak_worker_rss_gib": max((r.get("peak_rss_gib", 0.0) for r in records), default=0.0),
        "controller_peak_rss_gib": _controller_peak_rss_gib(),
        "worker_seconds": sum(r.get("seconds", 0.0) for r in records),
        "wall_seconds": time.time() - started,
    }
    write_json(Path(output) / "executions" / f"{time.time_ns()}_{stage}.json", summary)
    return summary
