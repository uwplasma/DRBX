"""Focused tests for the standalone Poisson-bracket comparison helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# The repository runs pytest from DRBX/, while this standalone utility lives
# beside the test and is intentionally not part of the drbx package.
sys.path.insert(0, str(Path(__file__).parent))

from compare_poisson_bracket_runs import (
    SnapshotRecord,
    find_failure_reason,
    near_axis_fft_proxy,
    pair_snapshots,
    parse_log_text,
    parse_snapshot_time_from_name,
    relative_integral_drift,
    relative_l2,
    weighted_relative_l2,
)


def test_parse_log_summary_and_failure_reason() -> None:
    text = """\
[simulation] compiled sharded rk4 advance in 12.5 s
[simulation] time integration: final_time=1.500000e-01, num_steps=225, dt=6.666667e-04
[------------------------]     1/225 t=6.666667e-04 step=2.0s gmres-iters(avg4)=4
[------------------------]     2/225 t=1.333333e-03 step=3.0s gmres-iters(avg4)=5
[diagnostics] step=3 positivity failure: density[-1, 2]
Traceback (most recent call last):
  ...
FloatingPointError: density became non-finite
"""
    summary = parse_log_text(text, "run.log")
    assert summary.status == "failed"
    assert summary.latest_step == 2
    assert summary.total_steps == 225
    assert summary.latest_time == 1.333333e-3
    assert summary.failure_step == 3
    assert summary.failure_time == 3 * 6.666667e-4
    assert summary.compile_seconds == 12.5
    assert summary.runtime_step_seconds["mean_step_seconds"] == 2.5
    assert summary.runtime_step_seconds["median_step_seconds"] == 2.5
    assert summary.step_seconds_by_step == {"1": 2.0, "2": 3.0}
    assert summary.failure_reason == "FloatingPointError: density became non-finite"


def test_parse_completed_log_without_failure() -> None:
    summary = parse_log_text("[x] 3/3 t=0.15 step=4.0s\n")
    assert summary.status == "completed"
    assert summary.latest_step == 3
    assert summary.runtime_step_seconds["estimated_advance_seconds"] == 4.0


def test_snapshot_filename_and_requested_time_pairing() -> None:
    name = "hsx_pb_direct_32.snapshot_t5d000000000000em03.npz"
    assert parse_snapshot_time_from_name(name) == 5.0e-3
    direct = [SnapshotRecord("d", 0.005, 0.005333, 8)]
    compatible = [SnapshotRecord("c", 0.00500000000001, 0.005333, 8)]
    assert pair_snapshots(direct, compatible) == [(direct[0], compatible[0])]


def test_numerical_metrics_and_integral_drift() -> None:
    reference = np.ones((2, 2, 2))
    candidate = 2.0 * reference
    weights = np.arange(1.0, 9.0).reshape(reference.shape)
    assert relative_l2(reference, candidate) == 1.0
    assert weighted_relative_l2(reference, candidate, weights) == 1.0
    assert relative_integral_drift(reference, candidate, weights) == 1.0


def test_near_axis_fft_reports_high_mode_energy() -> None:
    field = np.zeros((4, 8, 8))
    eta = np.arange(8)[None, None, :]
    field[:] = np.cos(2.0 * np.pi * 4.0 * eta / 8.0)
    proxy = near_axis_fft_proxy(field, first_rings=2, high_mode_cutoff=4)
    assert proxy["rings"] == 2
    assert proxy["high_mode_energy"] > 0.0
    assert proxy["high_mode_fraction"] > 0.99
