#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter


def _parse_point_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not counts or any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError("point counts must be positive integers")
    return counts


def _median(values: list[float]) -> float:
    import numpy as np

    return float(np.median(np.asarray(values, dtype=np.float64)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile batched atomic-rate and autodiff-throughput kernels on the active JAX backend."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs") / "data" / "runtime_profile_artifacts" / "atomic_rate_throughput_gate",
    )
    parser.add_argument("--point-counts", type=_parse_point_counts, default=(4096, 65536, 262144, 1048576, 4194304))
    parser.add_argument("--timed-runs", type=int, default=5)
    parser.add_argument(
        "--host-device-count",
        type=int,
        default=None,
        help=(
            "Expose this many CPU devices before importing JAX. Use this for "
            "local CPU pmap throughput checks; it does not change visible "
            "accelerator-device counts."
        ),
    )
    parser.add_argument(
        "--enable-pmap",
        action="store_true",
        help=(
            "Also time fixed-work pmap across visible devices. This is opt-in "
            "because some developer GPU environments expose devices that fail "
            "pmap parity checks even when single-device JAX is healthy."
        ),
    )
    return parser.parse_args()


def _configure_host_devices(args: argparse.Namespace) -> None:
    if args.host_device_count is None:
        return
    host_device_count = max(1, int(args.host_device_count))
    os.environ["JAX_DRB_HOST_DEVICE_COUNT"] = str(host_device_count)
    existing_flags = os.environ.get("XLA_FLAGS", "").strip()
    flag = f"--xla_force_host_platform_device_count={host_device_count}"
    if "xla_force_host_platform_device_count" not in existing_flags:
        os.environ["XLA_FLAGS"] = f"{existing_flags} {flag}".strip()


def _check_pmap_identity(jax, jnp, devices) -> tuple[bool, float | None, str | None]:
    """Verify that the visible multi-device runtime preserves an identity map."""

    device_count = len(devices)
    if device_count <= 1:
        return False, None, "fewer than two visible JAX devices"
    probe = jnp.arange(device_count * 8, dtype=jnp.float64).reshape((device_count, 8))
    try:
        identity = jax.pmap(lambda block: block, devices=devices)
        mapped = identity(probe).block_until_ready()
        max_abs_error = float(jnp.max(jnp.abs(mapped - probe)))
    except Exception as exc:  # pragma: no cover - depends on optional multi-device runtimes
        return False, None, f"pmap identity check raised {type(exc).__name__}: {exc}"
    if max_abs_error > 1.0e-12:
        return False, max_abs_error, f"pmap identity check failed with max_abs_error={max_abs_error:.3e}"
    return True, max_abs_error, None


def main() -> None:
    args = _parse_args()
    _configure_host_devices(args)

    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    from jax_drb.native.recycling_atomic import eval_amjuel_fit, hydrogen_cx_sigmav, load_amjuel_rate

    jax.config.update("jax_enable_x64", True)
    sigma_v_coeffs, _, _ = load_amjuel_rate("d", "iz")
    density = jnp.asarray(2.0e18, dtype=jnp.float64)
    dataset_scalars = {"Nnorm": 1.0e19, "Omega_ci": 2.0e6}

    def log_combined_rate(log_te):
        te = jnp.exp(log_te)
        return jnp.log(eval_amjuel_fit(te, density, sigma_v_coeffs) + hydrogen_cx_sigmav(te, dataset_scalars))

    rate_kernel = jit(vmap(log_combined_rate))
    grad_kernel = jit(vmap(grad(log_combined_rate)))
    devices = tuple(jax.local_devices())
    pmap_device_count = len(devices)

    def rate_block(log_temperature_block):
        return vmap(log_combined_rate)(log_temperature_block)

    def grad_block(log_temperature_block):
        return vmap(grad(log_combined_rate))(log_temperature_block)

    pmap_rate_kernel = None
    pmap_grad_kernel = None
    pmap_sanity_passed = None
    pmap_sanity_max_abs_error = None
    pmap_skip_reason = None
    if args.enable_pmap:
        pmap_sanity_passed, pmap_sanity_max_abs_error, pmap_skip_reason = _check_pmap_identity(jax, jnp, devices)
    if args.enable_pmap and pmap_sanity_passed:
        pmap_rate_kernel = jax.pmap(rate_block, devices=devices)
        pmap_grad_kernel = jax.pmap(grad_block, devices=devices)

    results = []
    for point_count in tuple(int(count) for count in args.point_counts):
        log_temperature = jnp.linspace(jnp.log(0.2), jnp.log(300.0), point_count, dtype=jnp.float64)
        rate_kernel(log_temperature).block_until_ready()
        grad_kernel(log_temperature).block_until_ready()
        rate_samples: list[float] = []
        grad_samples: list[float] = []
        for _ in range(max(1, int(args.timed_runs))):
            started_at = perf_counter()
            rate_kernel(log_temperature).block_until_ready()
            rate_samples.append(perf_counter() - started_at)
            started_at = perf_counter()
            grad_kernel(log_temperature).block_until_ready()
            grad_samples.append(perf_counter() - started_at)
        pmap_rate_samples: list[float] = []
        pmap_grad_samples: list[float] = []
        pmap_rate_seconds_median = None
        pmap_grad_seconds_median = None
        pmap_rate_speedup_vs_single = None
        pmap_grad_speedup_vs_single = None
        pmap_max_abs_rate_mismatch = None
        pmap_max_abs_grad_mismatch = None
        pmap_parity_passed = None
        if pmap_rate_kernel is not None and point_count >= pmap_device_count:
            sharded_count = (point_count // pmap_device_count) * pmap_device_count
            sharded_temperature = log_temperature[:sharded_count].reshape(
                (pmap_device_count, sharded_count // pmap_device_count)
            )
            pmap_rate_kernel(sharded_temperature).block_until_ready()
            pmap_grad_kernel(sharded_temperature).block_until_ready()
            for _ in range(max(1, int(args.timed_runs))):
                started_at = perf_counter()
                pmap_rate_kernel(sharded_temperature).block_until_ready()
                pmap_rate_samples.append(perf_counter() - started_at)
                started_at = perf_counter()
                pmap_grad_kernel(sharded_temperature).block_until_ready()
                pmap_grad_samples.append(perf_counter() - started_at)
            pmap_rate_seconds_median = _median(pmap_rate_samples)
            pmap_grad_seconds_median = _median(pmap_grad_samples)
            pmap_rate_speedup_vs_single = _median(rate_samples) / max(pmap_rate_seconds_median, 1.0e-30)
            pmap_grad_speedup_vs_single = _median(grad_samples) / max(pmap_grad_seconds_median, 1.0e-30)
            pmap_rate = pmap_rate_kernel(sharded_temperature).reshape((sharded_count,))
            pmap_grad = pmap_grad_kernel(sharded_temperature).reshape((sharded_count,))
            single_rate = rate_kernel(log_temperature[:sharded_count])
            single_grad = grad_kernel(log_temperature[:sharded_count])
            pmap_max_abs_rate_mismatch = float(jnp.max(jnp.abs(pmap_rate - single_rate)))
            pmap_max_abs_grad_mismatch = float(jnp.max(jnp.abs(pmap_grad - single_grad)))
            pmap_parity_passed = bool(pmap_max_abs_rate_mismatch <= 1.0e-10 and pmap_max_abs_grad_mismatch <= 1.0e-10)
            if not pmap_parity_passed:
                pmap_rate_speedup_vs_single = None
                pmap_grad_speedup_vs_single = None
        results.append(
            {
                "point_count": int(point_count),
                "rate_seconds_median": _median(rate_samples),
                "grad_seconds_median": _median(grad_samples),
                "rate_samples": [float(value) for value in rate_samples],
                "grad_samples": [float(value) for value in grad_samples],
                "pmap_device_count": int(pmap_device_count if pmap_rate_kernel is not None else 0),
                "pmap_rate_seconds_median": pmap_rate_seconds_median,
                "pmap_grad_seconds_median": pmap_grad_seconds_median,
                "pmap_rate_speedup_vs_single": pmap_rate_speedup_vs_single,
                "pmap_grad_speedup_vs_single": pmap_grad_speedup_vs_single,
                "pmap_rate_samples": [float(value) for value in pmap_rate_samples],
                "pmap_grad_samples": [float(value) for value in pmap_grad_samples],
                "pmap_rate_single_max_abs_error": pmap_max_abs_rate_mismatch,
                "pmap_grad_single_max_abs_error": pmap_max_abs_grad_mismatch,
                "pmap_parity_passed": pmap_parity_passed,
            }
        )

    showcase_point_count = min(65536, max(int(count) for count in args.point_counts))
    showcase_log_temperature = jnp.linspace(jnp.log(0.2), jnp.log(300.0), showcase_point_count, dtype=jnp.float64)

    def shifted_mean_rate(log_temperature_shift):
        shifted = showcase_log_temperature + log_temperature_shift
        return jnp.mean(jnp.exp(rate_kernel(shifted)))

    shifted_mean_rate_jit = jit(shifted_mean_rate)
    shifted_mean_rate_grad = jit(grad(shifted_mean_rate))
    shift0 = jnp.asarray(0.0, dtype=jnp.float64)
    epsilon = jnp.asarray(1.0e-4, dtype=jnp.float64)
    objective_value = float(shifted_mean_rate_jit(shift0))
    autodiff_sensitivity = float(shifted_mean_rate_grad(shift0))
    finite_difference_sensitivity = float(
        (shifted_mean_rate_jit(shift0 + epsilon) - shifted_mean_rate_jit(shift0 - epsilon)) / (2.0 * epsilon)
    )
    sensitivity_relative_error = abs(autodiff_sensitivity - finite_difference_sensitivity) / max(
        1.0e-30,
        abs(finite_difference_sensitivity),
    )

    report = {
        "case": "atomic_rate_throughput_gate",
        "backend": jax.default_backend(),
        "devices": [
            {"id": int(device.id), "platform": str(device.platform), "kind": str(device.device_kind)}
            for device in devices
        ],
        "host_device_count_requested": None if args.host_device_count is None else int(args.host_device_count),
        "pmap_requested": bool(args.enable_pmap),
        "pmap_enabled": bool(pmap_rate_kernel is not None),
        "pmap_sanity_passed": pmap_sanity_passed,
        "pmap_sanity_max_abs_error": pmap_sanity_max_abs_error,
        "pmap_skip_reason": pmap_skip_reason,
        "timed_runs": int(args.timed_runs),
        "results": results,
        "differentiability_showcase": {
            "objective": "mean combined AMJUEL ionisation plus charge-exchange rate under a log-temperature shift",
            "point_count": int(showcase_point_count),
            "objective_value": objective_value,
            "autodiff_sensitivity": autodiff_sensitivity,
            "finite_difference_sensitivity": finite_difference_sensitivity,
            "sensitivity_relative_error": float(sensitivity_relative_error),
        },
        "interpretation": (
            "This gate measures a batched reaction-source kernel and its reverse-mode derivative. "
            "When --enable-pmap is set and more than one local device is visible, it first checks a "
            "pmap identity map and only then measures fixed-work pmap throughput against the single-device kernel. "
            "It is the current GPU-throughput evidence for source "
            "terms; full recycling implicit output-window GPU speedup is intentionally not claimed by this gate."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "profile_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
