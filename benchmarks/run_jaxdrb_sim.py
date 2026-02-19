#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.driver import build_system_from_config
from jaxdrb.io import load_config


@dataclass
class SimOutput:
    times: np.ndarray
    rms_n: np.ndarray
    rms_Te: np.ndarray
    rms_omega: np.ndarray
    rms_phi: np.ndarray
    snapshot_n: np.ndarray
    snapshot_Te: np.ndarray
    snapshot_omega: np.ndarray
    snapshot_phi: np.ndarray
    point_n: np.ndarray
    point_Te: np.ndarray
    point_phi: np.ndarray
    point_idx: tuple[int, int, int]


def _rms(arr: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.mean(jnp.asarray(arr) ** 2))


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


def run_sim(
    config_path: Path,
    dt: float,
    nsteps: int,
    save_every: int,
    point_idx: tuple[int, int, int] | None = None,
) -> SimOutput:
    cfg = load_config(str(config_path))
    built = build_system_from_config(cfg.data)
    system = built.system
    state = built.state

    rhs = system.rhs

    steps = list(range(0, nsteps + 1, save_every))
    times = np.zeros(len(steps))
    rms_n = np.zeros(len(steps))
    rms_Te = np.zeros(len(steps))
    rms_omega = np.zeros(len(steps))
    rms_phi = np.zeros(len(steps))
    point_n = np.zeros(len(steps))
    point_Te = np.zeros(len(steps))
    point_phi = np.zeros(len(steps))

    if point_idx is None:
        nz, nx, ny = state.n.shape
        point_idx = (nz // 2, nx // 2, ny // 2)

    t = 0.0
    idx = 0
    for step in range(nsteps + 1):
        if step % save_every == 0:
            times[idx] = t
            n_phys = system._phys_n(state.n)
            Te_phys = system._phys_Te(state.Te)
            phi = system._phi_from_omega(state.omega, n=n_phys)
            rms_n[idx] = float(_rms(n_phys))
            rms_Te[idx] = float(_rms(Te_phys))
            rms_omega[idx] = float(_rms(state.omega))
            rms_phi[idx] = float(_rms(phi))
            z0, x0, y0 = point_idx
            point_n[idx] = float(n_phys[z0, x0, y0])
            point_Te[idx] = float(Te_phys[z0, x0, y0])
            point_phi[idx] = float(phi[z0, x0, y0])
            idx += 1
        if step == nsteps:
            break
        state = _rk4_step(rhs, t, state, dt)
        t += dt

    snapshot_n = np.asarray(system._phys_n(state.n))
    snapshot_Te = np.asarray(system._phys_Te(state.Te))
    snapshot_omega = np.asarray(state.omega)
    snapshot_phi = np.asarray(system._phi_from_omega(state.omega, n=system._phys_n(state.n)))

    return SimOutput(
        times=times,
        rms_n=rms_n,
        rms_Te=rms_Te,
        rms_omega=rms_omega,
        rms_phi=rms_phi,
        snapshot_n=snapshot_n,
        snapshot_Te=snapshot_Te,
        snapshot_omega=snapshot_omega,
        snapshot_phi=snapshot_phi,
        point_n=point_n,
        point_Te=point_Te,
        point_phi=point_phi,
        point_idx=point_idx,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Run a short jax_drb simulation and dump diagnostics.")
    p.add_argument("--config", required=True)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--nsteps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--output", required=True)
    p.add_argument("--point-idx", default=None, help="Point index as z,x,y (comma-separated)")
    args = p.parse_args()

    point_idx = None
    if args.point_idx:
        parts = [int(p) for p in args.point_idx.split(",")]
        if len(parts) != 3:
            raise ValueError("point-idx must be z,x,y")
        point_idx = (parts[0], parts[1], parts[2])

    output = run_sim(Path(args.config), args.dt, args.nsteps, args.save_every, point_idx=point_idx)

    np.savez(
        args.output,
        times=output.times,
        rms_n=output.rms_n,
        rms_Te=output.rms_Te,
        rms_omega=output.rms_omega,
        rms_phi=output.rms_phi,
        snapshot_n=output.snapshot_n,
        snapshot_Te=output.snapshot_Te,
        snapshot_omega=output.snapshot_omega,
        snapshot_phi=output.snapshot_phi,
        point_n=output.point_n,
        point_Te=output.point_Te,
        point_phi=output.point_phi,
        point_idx=np.asarray(output.point_idx),
    )


if __name__ == "__main__":
    main()
