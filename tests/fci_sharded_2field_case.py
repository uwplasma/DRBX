"""Shared shifted-torus two-field case for sharded-step equivalence tests.

This module is imported by ``tests/test_fci_sharded_2field.py`` for the
single-device sanity check and executed as a script inside a subprocess for
the multi-device check, where ``XLA_FLAGS`` must force the host device count
before JAX is imported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax.numpy as jnp

from drbx.geometry import FciGeometry3D, build_shifted_torus_geometry
from drbx.native import (
    Fci2FieldRhsParameters,
    Fci2FieldState,
    make_sharded_2field_step,
)
RHO_STAR = 1.0
ONE_SHARD_COUNTS = (1, 1, 1)
X_MIN = 0.15
X_MAX = 1.0


def build_case_geometry(shape: tuple[int, int, int]) -> FciGeometry3D:
    """Build the host-side shifted-torus geometry for local staging."""

    return build_shifted_torus_geometry(shape)


def build_initial_state(geometry) -> Fci2FieldState:
    """Smooth positive-density free-decay initial condition."""

    x = geometry.grid.x.centers[:, None, None]
    theta = geometry.grid.y.centers[None, :, None]
    zeta = geometry.grid.z.centers[None, None, :]
    envelope = jnp.sin(jnp.pi * (x - X_MIN) / (X_MAX - X_MIN))
    density = 1.0 + 0.05 * envelope * jnp.cos(2.0 * theta) * jnp.sin(zeta)
    v_parallel = 0.02 * envelope * jnp.sin(theta) * jnp.cos(2.0 * zeta)
    shape = geometry.shape
    return Fci2FieldState(
        density=jnp.broadcast_to(density, shape).astype(jnp.float64),
        v_parallel=jnp.broadcast_to(v_parallel, shape).astype(jnp.float64),
        density_background=jnp.ones(shape, dtype=jnp.float64),
    )


def run_sharded_steps(
    geometry,
    state: Fci2FieldState,
    *,
    dt: float,
    steps: int,
    shard_counts: tuple[int, int, int],
) -> Fci2FieldState:
    """Advance the two-field model through the local sharded RK4 step."""

    parameters = Fci2FieldRhsParameters(rho_star=RHO_STAR)
    step_fn, _info = make_sharded_2field_step(
        geometry,
        shard_counts,
        parameters,
        None,
        dt=dt,
    )
    for _ in range(steps):
        state = step_fn(state)
    return state


def max_state_difference(lhs: Fci2FieldState, rhs: Fci2FieldState) -> float:
    lhs_density = np.asarray(lhs.density)
    rhs_density = np.asarray(rhs.density)
    lhs_v_parallel = np.asarray(lhs.v_parallel)
    rhs_v_parallel = np.asarray(rhs.v_parallel)
    return max(
        float(np.max(np.abs(lhs_density - rhs_density))),
        float(np.max(np.abs(lhs_v_parallel - rhs_v_parallel))),
    )


def run_equivalence_case(
    *,
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
    steps: int,
    dt: float,
) -> dict[str, object]:
    geometry = build_case_geometry(shape)
    initial_state = build_initial_state(geometry)
    one_shard_state = run_sharded_steps(
        geometry,
        initial_state,
        dt=dt,
        steps=steps,
        shard_counts=ONE_SHARD_COUNTS,
    )
    multi_shard_state = run_sharded_steps(
        geometry,
        initial_state,
        dt=dt,
        steps=steps,
        shard_counts=shard_counts,
    )
    return {
        "max_abs_diff": max_state_difference(one_shard_state, multi_shard_state),
        "one_shard_density_max": float(jnp.max(jnp.abs(one_shard_state.density))),
        "multi_shard_density_max": float(jnp.max(jnp.abs(multi_shard_state.density))),
        "one_shard_counts": list(ONE_SHARD_COUNTS),
        "multi_shard_counts": list(shard_counts),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sharded two-field equivalence case.")
    parser.add_argument("--shape", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--shard-counts", type=int, nargs=3, default=(2, 2, 1))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    args = parser.parse_args(argv)

    result = run_equivalence_case(
        shape=tuple(args.shape),
        shard_counts=tuple(args.shard_counts),
        steps=args.steps,
        dt=args.dt,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
