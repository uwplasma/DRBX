from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import (
    build_recycling_1d_backward_euler_residual_context,
    _build_recycling_runtime_model,
    _build_recycling_state_fields,
)
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.runtime.run_config import RunConfiguration


@dataclass(frozen=True)
class RecyclingBatchedJvpProblem:
    residual: Callable[[object], object]
    base_state: object
    field_names: tuple[str, ...]
    feedback_names: tuple[str, ...]
    mesh_active_shape: tuple[int, int, int]
    state_size: int
    rhs_backend: str


def _block_until_ready(value):
    import jax

    return jax.block_until_ready(value)


def _median_seconds(samples: list[float]) -> float:
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _states_per_second(batch_size: int, seconds: float | None) -> float | None:
    if seconds is None:
        return None
    seconds = float(seconds)
    if seconds <= 0.0:
        return None
    return float(batch_size) / seconds


def _emit_progress(
    progress_callback: Callable[[dict[str, object]], None] | None,
    **record: object,
) -> None:
    if progress_callback is not None:
        progress_callback(dict(record))


def _write_progress_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def _best_metric(
    batch_results: list[dict[str, object]], metric_name: str, value_name: str
) -> dict[str, object] | None:
    candidates: list[tuple[float, int]] = []
    for index, result in enumerate(batch_results):
        value = result.get(metric_name)
        if value is None or isinstance(value, bool):
            continue
        metric_value = float(value)
        if not np.isfinite(metric_value):
            continue
        candidates.append((metric_value, index))
    if not candidates:
        return None
    _, best_index = max(candidates, key=lambda item: item[0])
    best = batch_results[best_index]
    return {
        "batch_size": int(best["batch_size"]),
        value_name: float(best[metric_name]),
    }


def _best_speedup_efficiency(
    batch_results: list[dict[str, object]], metric_name: str
) -> dict[str, object] | None:
    candidates: list[tuple[float, int]] = []
    for index, result in enumerate(batch_results):
        speedup = result.get(metric_name)
        if speedup is None or isinstance(speedup, bool):
            continue
        batch_size = int(result.get("batch_size", 0))
        if batch_size <= 0:
            continue
        efficiency = float(speedup) / float(batch_size)
        if not np.isfinite(efficiency):
            continue
        candidates.append((efficiency, index))
    if not candidates:
        return None
    _, best_index = max(candidates, key=lambda item: item[0])
    best = batch_results[best_index]
    batch_size = int(best["batch_size"])
    speedup = float(best[metric_name])
    return {
        "batch_size": batch_size,
        "speedup": speedup,
        "efficiency": speedup / float(batch_size),
    }


def _best_pmap_device_efficiency(
    batch_results: list[dict[str, object]], metric_name: str
) -> dict[str, object] | None:
    candidates: list[tuple[float, int]] = []
    for index, result in enumerate(batch_results):
        speedup = result.get(metric_name)
        if speedup is None or isinstance(speedup, bool):
            continue
        device_count = int(result.get("pmap_device_count", 0))
        if device_count <= 0:
            continue
        efficiency = float(speedup) / float(device_count)
        if not np.isfinite(efficiency):
            continue
        candidates.append((efficiency, index))
    if not candidates:
        return None
    _, best_index = max(candidates, key=lambda item: item[0])
    best = batch_results[best_index]
    device_count = int(best["pmap_device_count"])
    speedup = float(best[metric_name])
    return {
        "batch_size": int(best["batch_size"]),
        "pmap_batch_size": int(best.get("pmap_batch_size", 0)),
        "device_count": device_count,
        "speedup": speedup,
        "device_efficiency": speedup / float(device_count),
    }


def summarize_recycling_batched_jvp_scaling(
    batch_results: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize the scaling envelope from per-batch profiler measurements."""

    batch_sizes = [int(result["batch_size"]) for result in batch_results]
    return {
        "batch_count": len(batch_results),
        "batch_sizes": batch_sizes,
        "max_batch_size": None if not batch_sizes else max(batch_sizes),
        "throughput_units": "states_per_second",
        "best_residual_speedup_vs_serial": _best_metric(
            batch_results, "residual_speedup_vs_serial", "speedup"
        ),
        "best_jvp_speedup_vs_serial": _best_metric(
            batch_results, "jvp_speedup_vs_serial", "speedup"
        ),
        "best_residual_batch_efficiency": _best_speedup_efficiency(
            batch_results, "residual_speedup_vs_serial"
        ),
        "best_jvp_batch_efficiency": _best_speedup_efficiency(
            batch_results, "jvp_speedup_vs_serial"
        ),
        "best_batched_residual_throughput": _best_metric(
            batch_results, "batched_residual_states_per_second", "states_per_second"
        ),
        "best_batched_jvp_throughput": _best_metric(
            batch_results, "batched_jvp_states_per_second", "states_per_second"
        ),
        "best_pmap_jvp_throughput": _best_metric(
            batch_results, "pmap_jvp_states_per_second", "states_per_second"
        ),
        "best_pmap_jvp_speedup_vs_batched": _best_metric(
            batch_results, "pmap_jvp_speedup_vs_batched", "speedup"
        ),
        "best_pmap_jvp_speedup_vs_serial": _best_metric(
            batch_results, "pmap_jvp_speedup_vs_serial", "speedup"
        ),
        "best_pmap_jvp_device_efficiency_vs_serial": _best_pmap_device_efficiency(
            batch_results, "pmap_jvp_speedup_vs_serial"
        ),
    }


def _norm(value) -> float:
    import jax.numpy as jnp

    return float(jnp.linalg.norm(jnp.ravel(value)))


def _check_pmap_identity(jax, jnp, devices) -> tuple[bool, float | None, str | None]:
    """Verify that the visible multi-device runtime preserves identity data."""

    device_count = len(devices)
    if device_count <= 1:
        return False, None, "fewer than two visible JAX devices"
    probe = jnp.arange(device_count * 8, dtype=jnp.float64).reshape((device_count, 8))
    try:
        identity = jax.pmap(lambda block: block, devices=devices)
        mapped = identity(probe).block_until_ready()
        max_abs_error = float(jnp.max(jnp.abs(mapped - probe)))
    except (
        Exception
    ) as exc:  # pragma: no cover - depends on optional multi-device runtimes
        return False, None, f"pmap identity check raised {type(exc).__name__}: {exc}"
    if max_abs_error > 1.0e-12:
        return (
            False,
            max_abs_error,
            f"pmap identity check failed with max_abs_error={max_abs_error:.3e}",
        )
    return True, max_abs_error, None


def _deterministic_directions(batch_size: int, state_size: int, *, phase: float = 0.2):
    import jax.numpy as jnp

    coordinates = jnp.arange(
        int(batch_size) * int(state_size), dtype=jnp.float64
    ).reshape((int(batch_size), int(state_size)))
    directions = jnp.sin(3.1e-4 * coordinates + float(phase)) + 0.5 * jnp.cos(
        1.7e-4 * coordinates + 0.3
    )
    return directions / jnp.maximum(
        jnp.linalg.norm(directions, axis=1, keepdims=True), 1.0e-30
    )


def build_recycling_batched_jvp_problem(
    input_path: str | Path,
    *,
    overrides: tuple[str, ...] = (),
    timestep: float = 1.0e-4,
    evolve_feedback_integrals: bool = False,
    rhs_backend: str = "fixed_full_field_array",
) -> RecyclingBatchedJvpProblem:
    """Build the real D/T/He recycling residual used for batched JVP gates."""

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    config = load_bout_input(Path(input_path).expanduser().resolve())
    if overrides:
        config = apply_bout_overrides(config, tuple(overrides))
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    fields = _build_recycling_state_fields(runtime_model)
    feedback_integrals = {name: 0.0 for name in runtime_model.feedback_names}
    context = build_recycling_1d_backward_euler_residual_context(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=float(timestep),
        evolve_feedback_integrals=bool(evolve_feedback_integrals),
        rhs_backend=str(rhs_backend),
    )
    base_state = jnp.asarray(context.packed_previous_state, dtype=jnp.float64)
    return RecyclingBatchedJvpProblem(
        residual=context.residual,
        base_state=base_state,
        field_names=context.field_names,
        feedback_names=context.feedback_names,
        mesh_active_shape=(
            int(mesh.xend - mesh.xstart + 1),
            int(mesh.yend - mesh.ystart + 1),
            int(mesh.nz),
        ),
        state_size=int(base_state.size),
        rhs_backend=str(rhs_backend),
    )


def _time_repeated(
    callable_: Callable[[], object], *, timed_runs: int
) -> tuple[float, list[float]]:
    samples: list[float] = []
    for _ in range(max(1, int(timed_runs))):
        started_at = perf_counter()
        _block_until_ready(callable_())
        samples.append(perf_counter() - started_at)
    return _median_seconds(samples), samples


def profile_recycling_batched_jvp_problem(
    problem: RecyclingBatchedJvpProblem,
    *,
    batch_sizes: tuple[int, ...] = (1, 4, 16, 64),
    perturbation_scale: float = 1.0e-6,
    fd_epsilon: float = 1.0e-6,
    timed_runs: int = 5,
    enable_pmap: bool = True,
    check_objective_grad: bool = True,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Measure fixed-work vectorized residual/JVP throughput on a real residual."""

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    residual = problem.residual
    base_state = jnp.asarray(problem.base_state, dtype=jnp.float64)
    _emit_progress(
        progress_callback,
        event="profile_start",
        rhs_backend=str(problem.rhs_backend),
        state_size=int(problem.state_size),
        mesh_active_shape=list(problem.mesh_active_shape),
        batch_sizes=[int(size) for size in batch_sizes],
        timed_runs=int(timed_runs),
        check_objective_grad=bool(check_objective_grad),
    )
    residual_jit = jax.jit(residual)
    started_at = perf_counter()
    residual_jit(base_state).block_until_ready()
    base_residual_warmup_seconds = perf_counter() - started_at
    _emit_progress(
        progress_callback,
        event="base_residual_warmup_complete",
        seconds=float(base_residual_warmup_seconds),
    )

    def single_jvp(state, tangent):
        return jax.jvp(residual, (state,), (tangent,))[1]

    jvp_jit = jax.jit(single_jvp)

    def objective(state):
        value = residual(state)
        return 0.5 * jnp.mean(jnp.square(value))

    objective_jit = jax.jit(objective)
    base_direction = _deterministic_directions(1, problem.state_size)[0]
    started_at = perf_counter()
    base_jvp = jvp_jit(base_state, base_direction).block_until_ready()
    base_jvp_warmup_seconds = perf_counter() - started_at
    _emit_progress(
        progress_callback,
        event="base_jvp_warmup_complete",
        seconds=float(base_jvp_warmup_seconds),
    )
    started_at = perf_counter()
    fd_jvp = (
        residual_jit(base_state + float(fd_epsilon) * base_direction)
        - residual_jit(base_state - float(fd_epsilon) * base_direction)
    ) / (2.0 * float(fd_epsilon))
    fd_jvp = fd_jvp.block_until_ready()
    jvp_fd_relative_error = _norm(base_jvp - fd_jvp) / max(1.0e-30, _norm(fd_jvp))
    fd_jvp_check_seconds = perf_counter() - started_at
    _emit_progress(
        progress_callback,
        event="jvp_fd_check_complete",
        seconds=float(fd_jvp_check_seconds),
        jvp_fd_relative_error=float(jvp_fd_relative_error),
    )

    grad_directional = None
    fd_directional = None
    objective_directional_relative_error = None
    objective_grad_check_seconds = None
    if check_objective_grad:
        started_at = perf_counter()
        objective_grad_jit = jax.jit(jax.grad(objective))
        objective_grad = objective_grad_jit(base_state).block_until_ready()
        grad_directional = float(jnp.vdot(objective_grad, base_direction))
        fd_directional = float(
            (
                objective_jit(base_state + float(fd_epsilon) * base_direction)
                - objective_jit(base_state - float(fd_epsilon) * base_direction)
            )
            / (2.0 * float(fd_epsilon))
        )
        objective_directional_relative_error = abs(
            grad_directional - fd_directional
        ) / max(1.0e-30, abs(fd_directional))
        objective_grad_check_seconds = perf_counter() - started_at
        _emit_progress(
            progress_callback,
            event="objective_grad_check_complete",
            seconds=float(objective_grad_check_seconds),
            objective_directional_relative_error=float(
                objective_directional_relative_error
            ),
        )

    batch_results: list[dict[str, object]] = []
    devices = tuple(jax.local_devices())
    pmap_sanity_passed = None
    pmap_sanity_max_abs_error = None
    pmap_skip_reason = None
    if enable_pmap:
        pmap_sanity_passed, pmap_sanity_max_abs_error, pmap_skip_reason = (
            _check_pmap_identity(jax, jnp, devices)
        )
    pmap_enabled = bool(enable_pmap and pmap_sanity_passed)
    batched_residual = jax.jit(jax.vmap(residual))
    batched_jvp = jax.jit(jax.vmap(single_jvp))
    for batch_size in tuple(int(size) for size in batch_sizes):
        _emit_progress(
            progress_callback,
            event="batch_start",
            batch_size=int(batch_size),
        )
        started_at = perf_counter()
        directions = _deterministic_directions(batch_size, problem.state_size)
        states = base_state[None, :] + float(perturbation_scale) * directions
        direction_build_seconds = perf_counter() - started_at
        _emit_progress(
            progress_callback,
            event="batch_direction_build_complete",
            batch_size=int(batch_size),
            seconds=float(direction_build_seconds),
        )
        started_at = perf_counter()
        batched_residual(states).block_until_ready()
        batched_residual_warmup_seconds = perf_counter() - started_at
        _emit_progress(
            progress_callback,
            event="batch_residual_warmup_complete",
            batch_size=int(batch_size),
            seconds=float(batched_residual_warmup_seconds),
        )
        started_at = perf_counter()
        batched_jvp(states, directions).block_until_ready()
        batched_jvp_warmup_seconds = perf_counter() - started_at
        _emit_progress(
            progress_callback,
            event="batch_jvp_warmup_complete",
            batch_size=int(batch_size),
            seconds=float(batched_jvp_warmup_seconds),
        )
        started_at = perf_counter()
        for state, direction in zip(states, directions, strict=True):
            residual_jit(state).block_until_ready()
            jvp_jit(state, direction).block_until_ready()
        serial_warmup_seconds = perf_counter() - started_at
        _emit_progress(
            progress_callback,
            event="batch_serial_warmup_complete",
            batch_size=int(batch_size),
            seconds=float(serial_warmup_seconds),
        )
        batch_warmup_seconds = (
            batched_residual_warmup_seconds
            + batched_jvp_warmup_seconds
            + serial_warmup_seconds
        )
        _emit_progress(
            progress_callback,
            event="batch_warmup_complete",
            batch_size=int(batch_size),
            seconds=float(batch_warmup_seconds),
        )

        serial_residual_seconds, serial_residual_samples = _time_repeated(
            lambda: tuple(residual_jit(state) for state in states),
            timed_runs=timed_runs,
        )
        batched_residual_seconds, batched_residual_samples = _time_repeated(
            lambda: batched_residual(states),
            timed_runs=timed_runs,
        )
        serial_jvp_seconds, serial_jvp_samples = _time_repeated(
            lambda: tuple(
                jvp_jit(state, direction)
                for state, direction in zip(states, directions, strict=True)
            ),
            timed_runs=timed_runs,
        )
        batched_jvp_seconds, batched_jvp_samples = _time_repeated(
            lambda: batched_jvp(states, directions),
            timed_runs=timed_runs,
        )
        serial_residual_values = jnp.stack([residual_jit(state) for state in states])
        serial_jvp_values = jnp.stack(
            [
                jvp_jit(state, direction)
                for state, direction in zip(states, directions, strict=True)
            ]
        )
        batched_residual_values = batched_residual(states)
        batched_jvp_values = batched_jvp(states, directions)
        residual_mismatch = float(
            jnp.max(jnp.abs(batched_residual_values - serial_residual_values))
        )
        jvp_mismatch = float(jnp.max(jnp.abs(batched_jvp_values - serial_jvp_values)))

        pmap_jvp_seconds = None
        pmap_jvp_samples = None
        pmap_jvp_speedup_vs_batched = None
        pmap_jvp_speedup_vs_serial = None
        pmap_device_count = 0
        pmap_batch_size = 0
        pmap_jvp_batched_max_abs_error = None
        if pmap_enabled and len(devices) > 1 and batch_size >= len(devices):
            pmap_device_count = min(len(devices), batch_size)
            pmap_batch_size = (batch_size // pmap_device_count) * pmap_device_count
            if pmap_batch_size >= pmap_device_count:
                per_device = pmap_batch_size // pmap_device_count
                sharded_states = states[:pmap_batch_size].reshape(
                    (pmap_device_count, per_device, problem.state_size)
                )
                sharded_directions = directions[:pmap_batch_size].reshape(
                    (pmap_device_count, per_device, problem.state_size)
                )
                pmap_jvp = jax.pmap(
                    lambda shard_states, shard_directions: jax.vmap(single_jvp)(
                        shard_states, shard_directions
                    ),
                    devices=devices[:pmap_device_count],
                )
                pmap_jvp(sharded_states, sharded_directions).block_until_ready()
                pmap_jvp_values = pmap_jvp(
                    sharded_states, sharded_directions
                ).block_until_ready()
                reference_jvp_values = batched_jvp_values[:pmap_batch_size].reshape(
                    pmap_jvp_values.shape
                )
                pmap_jvp_batched_max_abs_error = float(
                    jnp.max(jnp.abs(pmap_jvp_values - reference_jvp_values))
                )
                pmap_jvp_seconds, pmap_jvp_samples = _time_repeated(
                    lambda: pmap_jvp(sharded_states, sharded_directions),
                    timed_runs=timed_runs,
                )
                pmap_jvp_speedup_vs_batched = batched_jvp_seconds / max(
                    pmap_jvp_seconds, 1.0e-30
                )
                pmap_jvp_speedup_vs_serial = serial_jvp_seconds / max(
                    pmap_jvp_seconds, 1.0e-30
                )

        batch_results.append(
            {
                "batch_size": int(batch_size),
                "direction_build_seconds": float(direction_build_seconds),
                "batch_warmup_seconds": float(batch_warmup_seconds),
                "batched_residual_warmup_seconds": float(
                    batched_residual_warmup_seconds
                ),
                "batched_jvp_warmup_seconds": float(batched_jvp_warmup_seconds),
                "serial_warmup_seconds": float(serial_warmup_seconds),
                "serial_residual_seconds_median": float(serial_residual_seconds),
                "batched_residual_seconds_median": float(batched_residual_seconds),
                "serial_residual_states_per_second": _states_per_second(
                    batch_size, serial_residual_seconds
                ),
                "batched_residual_states_per_second": _states_per_second(
                    batch_size, batched_residual_seconds
                ),
                "residual_speedup_vs_serial": float(
                    serial_residual_seconds / max(batched_residual_seconds, 1.0e-30)
                ),
                "serial_jvp_seconds_median": float(serial_jvp_seconds),
                "batched_jvp_seconds_median": float(batched_jvp_seconds),
                "serial_jvp_states_per_second": _states_per_second(
                    batch_size, serial_jvp_seconds
                ),
                "batched_jvp_states_per_second": _states_per_second(
                    batch_size, batched_jvp_seconds
                ),
                "jvp_speedup_vs_serial": float(
                    serial_jvp_seconds / max(batched_jvp_seconds, 1.0e-30)
                ),
                "serial_residual_samples": [
                    float(value) for value in serial_residual_samples
                ],
                "batched_residual_samples": [
                    float(value) for value in batched_residual_samples
                ],
                "serial_jvp_samples": [float(value) for value in serial_jvp_samples],
                "batched_jvp_samples": [float(value) for value in batched_jvp_samples],
                "residual_batched_serial_max_abs_error": residual_mismatch,
                "jvp_batched_serial_max_abs_error": jvp_mismatch,
                "pmap_device_count": int(pmap_device_count),
                "pmap_batch_size": int(pmap_batch_size),
                "pmap_jvp_seconds_median": pmap_jvp_seconds,
                "pmap_jvp_states_per_second": _states_per_second(
                    pmap_batch_size, pmap_jvp_seconds
                ),
                "pmap_jvp_samples": None
                if pmap_jvp_samples is None
                else [float(value) for value in pmap_jvp_samples],
                "pmap_jvp_speedup_vs_batched": pmap_jvp_speedup_vs_batched,
                "pmap_jvp_speedup_vs_serial": pmap_jvp_speedup_vs_serial,
                "pmap_jvp_batched_max_abs_error": pmap_jvp_batched_max_abs_error,
            }
        )
        _emit_progress(
            progress_callback,
            event="batch_complete",
            batch_size=int(batch_size),
            direction_build_seconds=float(direction_build_seconds),
            batch_warmup_seconds=float(batch_warmup_seconds),
            batched_residual_warmup_seconds=float(batched_residual_warmup_seconds),
            batched_jvp_warmup_seconds=float(batched_jvp_warmup_seconds),
            serial_warmup_seconds=float(serial_warmup_seconds),
            residual_speedup_vs_serial=float(
                serial_residual_seconds / max(batched_residual_seconds, 1.0e-30)
            ),
            jvp_speedup_vs_serial=float(
                serial_jvp_seconds / max(batched_jvp_seconds, 1.0e-30)
            ),
            residual_batched_serial_max_abs_error=float(residual_mismatch),
            jvp_batched_serial_max_abs_error=float(jvp_mismatch),
            pmap_device_count=int(pmap_device_count),
        )

    return {
        "case": "recycling_batched_jvp_profile",
        "backend": jax.default_backend(),
        "devices": [
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "kind": str(device.device_kind),
            }
            for device in devices
        ],
        "pmap_requested": bool(enable_pmap),
        "pmap_enabled": bool(pmap_enabled),
        "pmap_sanity_passed": pmap_sanity_passed,
        "pmap_sanity_max_abs_error": pmap_sanity_max_abs_error,
        "pmap_skip_reason": pmap_skip_reason,
        "mesh_active_shape": list(problem.mesh_active_shape),
        "state_size": int(problem.state_size),
        "rhs_backend": str(problem.rhs_backend),
        "field_names": list(problem.field_names),
        "feedback_names": list(problem.feedback_names),
        "timed_runs": int(timed_runs),
        "perturbation_scale": float(perturbation_scale),
        "fd_epsilon": float(fd_epsilon),
        "warmup_timing": {
            "base_residual_warmup_seconds": float(base_residual_warmup_seconds),
            "base_jvp_warmup_seconds": float(base_jvp_warmup_seconds),
            "fd_jvp_check_seconds": float(fd_jvp_check_seconds),
            "objective_grad_check_seconds": None
            if objective_grad_check_seconds is None
            else float(objective_grad_check_seconds),
        },
        "differentiability": {
            "jvp_fd_relative_error": float(jvp_fd_relative_error),
            "objective_grad_checked": bool(check_objective_grad),
            "objective_grad_directional_derivative": grad_directional,
            "objective_fd_directional_derivative": fd_directional,
            "objective_directional_relative_error": objective_directional_relative_error,
        },
        "batch_results": batch_results,
        "throughput_summary": summarize_recycling_batched_jvp_scaling(batch_results),
        "interpretation": (
            "This gate measures the real fixed-layout D/T/He recycling residual under jit, vmap, jvp, grad, "
            "and optional pmap after a multi-device identity sanity check. It is the current differentiable "
            "residual-throughput lane; it does not promote the full production BDF output-window solve as the "
            "default implicit backend."
        ),
    }


def create_recycling_batched_jvp_profile_package(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    overrides: tuple[str, ...] = (),
    batch_sizes: tuple[int, ...] = (1, 4, 16, 64),
    timestep: float = 1.0e-4,
    perturbation_scale: float = 1.0e-6,
    fd_epsilon: float = 1.0e-6,
    timed_runs: int = 5,
    enable_pmap: bool = True,
    check_objective_grad: bool = True,
    rhs_backend: str = "fixed_full_field_array",
) -> dict[str, object]:
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    progress_path = output_path / "profile_progress.jsonl"
    progress_path.unlink(missing_ok=True)

    def progress_callback(record: dict[str, object]) -> None:
        _write_progress_record(progress_path, record)

    progress_callback(
        {
            "event": "problem_build_start",
            "rhs_backend": str(rhs_backend),
            "input_path": "<input-path>/BOUT.inp",
            "overrides": list(overrides),
        }
    )
    problem = build_recycling_batched_jvp_problem(
        input_path,
        overrides=overrides,
        timestep=timestep,
        rhs_backend=rhs_backend,
    )
    progress_callback(
        {
            "event": "problem_build_complete",
            "rhs_backend": str(problem.rhs_backend),
            "state_size": int(problem.state_size),
            "mesh_active_shape": list(problem.mesh_active_shape),
        }
    )
    report = profile_recycling_batched_jvp_problem(
        problem,
        batch_sizes=batch_sizes,
        perturbation_scale=perturbation_scale,
        fd_epsilon=fd_epsilon,
        timed_runs=timed_runs,
        enable_pmap=enable_pmap,
        check_objective_grad=check_objective_grad,
        progress_callback=progress_callback,
    )
    report = {
        **report,
        "input_path": "<input-path>/BOUT.inp",
        "overrides": list(overrides),
        "timestep": float(timestep),
        "rhs_backend": str(rhs_backend),
        "profile_progress_jsonl": "profile_progress.jsonl",
    }
    (output_path / "profile_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
