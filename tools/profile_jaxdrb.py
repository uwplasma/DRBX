#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxdrb.driver import build_system_from_config
from jaxdrb.io import load_config
from jaxdrb.integrators import build_rk4_scan
from jaxdrb.profiling import jax_trace, save_device_memory_profile, save_hlo, save_compile_stats


def _block_until_ready(tree):
    def wait(x):
        return x.block_until_ready() if hasattr(x, "block_until_ready") else x

    return jax.tree_util.tree_map(wait, tree)


def main() -> None:
    p = argparse.ArgumentParser(description="Profile jax_drb kernels and memory.")
    p.add_argument("--config", required=True, help="Path to jax_drb TOML config")
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--outdir", default="benchmarks/profiles/jaxdrb")
    p.add_argument("--trace", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hlo", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    built = build_system_from_config(cfg.data)
    system = built.system
    state = built.state

    def diag_fn(t, y):
        return jnp.asarray(t)

    runner, nsave, rem = build_rk4_scan(
        system.rhs, args.dt, args.steps, max(args.steps, 1), diag_fn, rhs_remat=False
    )
    lowered = runner.lower(state)
    if args.hlo:
        save_hlo(lowered, outdir, name="jaxdrb_scan")
    save_compile_stats(lowered, outdir, name="compile_stats.json")

    for _ in range(max(args.warmup, 0)):
        _block_until_ready(runner(state))

    start = time.perf_counter()
    if args.trace:
        with jax_trace(outdir):
            _block_until_ready(runner(state))
    else:
        _block_until_ready(runner(state))
    elapsed = time.perf_counter() - start

    if args.memory:
        save_device_memory_profile(outdir, name="memory_profile.pb")

    summary = outdir / "timing.txt"
    summary.write_text(
        "\n".join(
            [
                f"backend: {jax.default_backend()}",
                f"devices: {[d.device_kind for d in jax.devices()]}",
                f"steps: {args.steps}",
                f"dt: {args.dt}",
                f"elapsed_s: {elapsed:.6f}",
                f"time_per_step_s: {elapsed / max(args.steps, 1):.6e}",
            ]
        )
    )

    print(f"Profile outputs written to: {outdir}")


if __name__ == "__main__":
    main()
