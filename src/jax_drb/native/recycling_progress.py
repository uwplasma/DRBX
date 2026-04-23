from __future__ import annotations

import time
from typing import Any


def build_recycling_progress_details(
    *,
    interval_index: int,
    steps: int,
    solver_mode: str,
    accepted_dt: float,
    stored_states: int,
    output_timestep: float,
    run_started_at: float,
    interval_started_at: float,
    now: float | None = None,
    live_progress: bool = True,
) -> tuple[dict[str, Any], float]:
    current_time = time.perf_counter() if now is None else float(now)
    completed_intervals = max(int(interval_index), 0)
    total_intervals = max(int(steps), 0)
    elapsed_seconds = max(current_time - float(run_started_at), 0.0)
    interval_elapsed_seconds = max(current_time - float(interval_started_at), 0.0)
    remaining_intervals = max(total_intervals - completed_intervals, 0)
    fraction_complete = 1.0 if total_intervals <= 0 else min(completed_intervals / float(total_intervals), 1.0)
    mean_interval_seconds = (
        elapsed_seconds / float(completed_intervals)
        if completed_intervals > 0
        else interval_elapsed_seconds
    )
    estimated_remaining_seconds = max(mean_interval_seconds * float(remaining_intervals), 0.0)
    total_simulated_time = max(float(output_timestep) * float(total_intervals), 0.0)
    simulated_time = min(float(output_timestep) * float(completed_intervals), total_simulated_time)
    return (
        {
            "interval_index": completed_intervals,
            "steps": total_intervals,
            "solver_mode": solver_mode,
            "accepted_dt": float(accepted_dt),
            "stored_states": int(stored_states),
            "completed_intervals": completed_intervals,
            "remaining_intervals": remaining_intervals,
            "fraction_complete": float(fraction_complete),
            "elapsed_seconds": float(elapsed_seconds),
            "interval_elapsed_seconds": float(interval_elapsed_seconds),
            "estimated_remaining_seconds": float(estimated_remaining_seconds),
            "simulated_time": float(simulated_time),
            "total_simulated_time": float(total_simulated_time),
            "live_progress": bool(live_progress),
        },
        current_time,
    )
