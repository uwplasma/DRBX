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
from jaxdrb.integrators import build_rk4_scan


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

    if point_idx is None:
        nz, nx, ny = state.n.shape
        point_idx = (nz // 2, nx // 2, ny // 2)

    def diag_fn(t, y):
        n_phys = system._phys_n(y.n)
        Te_phys = system._phys_Te(y.Te)
        phi = system._phi_from_omega(y.omega, n=n_phys)
        z0, x0, y0 = point_idx
        return (
            _rms(n_phys),
            _rms(Te_phys),
            _rms(y.omega),
            _rms(phi),
            n_phys[z0, x0, y0],
            Te_phys[z0, x0, y0],
            phi[z0, x0, y0],
        )

    runner, nsave, rem = build_rk4_scan(system.rhs, dt, nsteps, save_every, diag_fn)
    final_state, diag_series = runner(state)
    rms_n, rms_Te, rms_omega, rms_phi, point_n, point_Te, point_phi = diag_series

    base_times = np.arange(nsave, dtype=np.float64) * (save_every * dt)
    if rem > 0:
        base_times[-1] = nsteps * dt
    times = base_times

    n_phys = system._phys_n(final_state.n)
    Te_phys = system._phys_Te(final_state.Te)
    phi = system._phi_from_omega(final_state.omega, n=n_phys)
    snapshot_n = np.asarray(jax.device_get(n_phys))
    snapshot_Te = np.asarray(jax.device_get(Te_phys))
    snapshot_omega = np.asarray(jax.device_get(final_state.omega))
    snapshot_phi = np.asarray(jax.device_get(phi))

    return SimOutput(
        times=np.asarray(times),
        rms_n=np.asarray(jax.device_get(rms_n)),
        rms_Te=np.asarray(jax.device_get(rms_Te)),
        rms_omega=np.asarray(jax.device_get(rms_omega)),
        rms_phi=np.asarray(jax.device_get(rms_phi)),
        snapshot_n=snapshot_n,
        snapshot_Te=snapshot_Te,
        snapshot_omega=snapshot_omega,
        snapshot_phi=snapshot_phi,
        point_n=np.asarray(jax.device_get(point_n)),
        point_Te=np.asarray(jax.device_get(point_Te)),
        point_phi=np.asarray(jax.device_get(point_phi)),
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
