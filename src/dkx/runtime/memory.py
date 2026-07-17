from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class PeakRssMeasurement:
    start_rss_bytes: int | None
    end_rss_bytes: int | None
    peak_rss_bytes: int | None
    peak_rss_delta_bytes: int | None
    sample_count: int
    sampling_interval_seconds: float
    status: str


def measure_peak_rss(
    function: Callable[[], T],
    *,
    sampling_interval_seconds: float = 0.10,
) -> tuple[T, PeakRssMeasurement]:
    samples: list[int] = []
    sample_failures = 0
    stop_event = threading.Event()
    root_pid = os.getpid()

    def append_sample() -> None:
        nonlocal sample_failures
        value = process_tree_rss_bytes(root_pid)
        if value is None:
            sample_failures += 1
            return
        samples.append(value)

    append_sample()

    def sample_loop() -> None:
        while not stop_event.wait(sampling_interval_seconds):
            append_sample()

    sampler = threading.Thread(target=sample_loop, name="dkx-rss-sampler", daemon=True)
    sampler.start()
    try:
        result = function()
    finally:
        stop_event.set()
        sampler.join(timeout=max(1.0, 4.0 * sampling_interval_seconds))
    append_sample()
    start_rss_bytes = samples[0] if samples else None
    end_rss_bytes = samples[-1] if samples else None
    peak_rss_bytes = max(samples) if samples else None
    peak_rss_delta_bytes = (
        None
        if peak_rss_bytes is None or start_rss_bytes is None
        else max(int(peak_rss_bytes - start_rss_bytes), 0)
    )
    status = "sampled_process_tree_rss"
    if not samples:
        status = "unavailable"
    elif sample_failures:
        status = "sampled_process_tree_rss_with_partial_failures"
    return result, PeakRssMeasurement(
        start_rss_bytes=start_rss_bytes,
        end_rss_bytes=end_rss_bytes,
        peak_rss_bytes=peak_rss_bytes,
        peak_rss_delta_bytes=peak_rss_delta_bytes,
        sample_count=len(samples),
        sampling_interval_seconds=float(sampling_interval_seconds),
        status=status,
    )


def process_tree_rss_bytes(root_pid: int) -> int | None:
    pids = process_tree_pids(root_pid)
    rss_kib = 0
    observed = False
    for pid in pids:
        pid_rss = process_rss_kib(pid)
        if pid_rss is None:
            continue
        rss_kib += pid_rss
        observed = True
    return int(rss_kib * 1024) if observed else None


def process_tree_pids(root_pid: int) -> list[int]:
    pids: list[int] = [int(root_pid)]
    frontier: list[int] = [int(root_pid)]
    while frontier:
        parent = frontier.pop()
        try:
            completed = subprocess.run(
                ["pgrep", "-P", str(parent)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            continue
        if completed.returncode not in (0, 1):
            continue
        children = [int(value) for value in completed.stdout.split() if value.isdigit()]
        for child in children:
            if child not in pids:
                pids.append(child)
                frontier.append(child)
    return pids


def process_rss_kib(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    if not text:
        return None
    try:
        return int(float(text.splitlines()[-1].strip()))
    except ValueError:
        return None


def bytes_to_mebibytes(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value / (1024.0 * 1024.0))
