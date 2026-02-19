#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxdrb.driver import build_system_from_config
from jaxdrb.io import load_config
from jaxdrb.profiling import jax_trace, save_device_memory_profile, save_hlo, save_compile_stats


def _tree_add(y, dy, scale: float = 1.0):
    def add(a, b):
        if a is None or b is None:
            return None
        return a + scale * b

    return jax.tree_util.tree_map(add, y, dy, is_leaf=lambda x: x is None)


def _rk4_step(rhs, t, y, dt):
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, _tree_add(y, k1, 0.5 * dt))
    k3 = rhs(t + 0.5 * dt, _tree_add(y, k2, 0.5 * dt))
    k4 = rhs(t + dt, _tree_add(y, k3, dt))
    acc = _tree_add(k1, k2, 2.0)
    acc = _tree_add(acc, k3, 2.0)
    acc = _tree_add(acc, k4, 1.0)
    return _tree_add(y, acc, dt / 6.0)


def build_runner(rhs, dt: float, steps: int):
    dt = float(dt)

    def body(carry, _):
        t, y = carry
        y = _rk4_step(rhs, t, y, dt)
        return (t + dt, y), None

    def run(state):
        (t, y), _ = jax.lax.scan(body, (0.0, state), None, length=int(steps))
        return t, y

    return jax.jit(run)


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

    runner = build_runner(system.rhs, args.dt, args.steps)

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
