#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    return parser.parse_args()


def main() -> None:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    from jax_drb.native.recycling_atomic import eval_amjuel_fit, hydrogen_cx_sigmav, load_amjuel_rate

    args = _parse_args()
    jax.config.update("jax_enable_x64", True)
    sigma_v_coeffs, _, _ = load_amjuel_rate("d", "iz")
    density = jnp.asarray(2.0e18, dtype=jnp.float64)
    dataset_scalars = {"Nnorm": 1.0e19, "Omega_ci": 2.0e6}

    def log_combined_rate(log_te):
        te = jnp.exp(log_te)
        return jnp.log(eval_amjuel_fit(te, density, sigma_v_coeffs) + hydrogen_cx_sigmav(te, dataset_scalars))

    rate_kernel = jit(vmap(log_combined_rate))
    grad_kernel = jit(vmap(grad(log_combined_rate)))
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
        results.append(
            {
                "point_count": int(point_count),
                "rate_seconds_median": _median(rate_samples),
                "grad_seconds_median": _median(grad_samples),
                "rate_samples": [float(value) for value in rate_samples],
                "grad_samples": [float(value) for value in grad_samples],
            }
        )

    report = {
        "case": "atomic_rate_throughput_gate",
        "backend": jax.default_backend(),
        "devices": [
            {"id": int(device.id), "platform": str(device.platform), "kind": str(device.device_kind)}
            for device in jax.local_devices()
        ],
        "timed_runs": int(args.timed_runs),
        "results": results,
        "interpretation": (
            "This gate measures a batched reaction-source kernel and its reverse-mode derivative. "
            "It is the current GPU-throughput evidence for source terms; full recycling implicit "
            "output-window GPU speedup is intentionally not claimed by this gate."
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
