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

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax.numpy as jnp

from jax_drb.geometry import build_curvature_coefficients
from jax_drb.native import (
    Fci2FieldRhsParameters,
    Fci2FieldState,
    compute_2field_rhs,
    make_sharded_2field_step,
    rk4_step,
)
from tests.test_mms_shifted_torus_2_field import (
    build_shifted_torus_2field_geometry,
    x_max,
    x_min,
)


RHO_STAR = 1.0
PERIODIC_AXES = (False, True, True)


def build_case_geometry(shape: tuple[int, int, int]):
    """Reuse the shifted-torus geometry builder from the two-field MMS tests."""

    return build_shifted_torus_2field_geometry(shape)


def build_initial_state(geometry) -> Fci2FieldState:
    """Smooth positive-density free-decay initial condition."""

    x = geometry.grid.x.centers[:, None, None]
    theta = geometry.grid.y.centers[None, :, None]
    zeta = geometry.grid.z.centers[None, None, :]
    envelope = jnp.sin(jnp.pi * (x - float(x_min)) / (float(x_max) - float(x_min)))
    density = 1.0 + 0.05 * envelope * jnp.cos(2.0 * theta) * jnp.sin(zeta)
    v_parallel = 0.02 * envelope * jnp.sin(theta) * jnp.cos(2.0 * zeta)
    shape = geometry.shape
    return Fci2FieldState(
        density=jnp.broadcast_to(density, shape).astype(jnp.float64),
        v_parallel=jnp.broadcast_to(v_parallel, shape).astype(jnp.float64),
        density_background=jnp.ones(shape, dtype=jnp.float64),
    )


def run_direct_steps(
    geometry,
    state: Fci2FieldState,
    *,
    dt: float,
    steps: int,
) -> Fci2FieldState:
    """Advance the unsharded global-geometry two-field model by RK4 steps."""

    parameters = Fci2FieldRhsParameters(rho_star=RHO_STAR)
    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=PERIODIC_AXES)

    def _rhs_fn(stage_state, stage_time, carry):
        del stage_time
        result, _timings = compute_2field_rhs(
            stage_state,
            geometry=geometry,
            parameters=parameters,
            curvature_coefficients=curvature_coefficients,
            density_face_bc=None,
            phi_face_bc=None,
            v_parallel_face_bc=None,
        )
        return result.rhs, carry, None

    for _ in range(steps):
        state = rk4_step(state, time=0.0, timestep=dt, rhs_fn=_rhs_fn, carry=None).state
    return state


def run_sharded_steps(
    geometry,
    state: Fci2FieldState,
    *,
    dt: float,
    steps: int,
    shard_counts: tuple[int, int, int],
) -> Fci2FieldState:
    """Advance the same model through the sharded shard_map RK4 step."""

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
    return max(
        float(jnp.max(jnp.abs(jnp.asarray(lhs.density) - jnp.asarray(rhs.density)))),
        float(jnp.max(jnp.abs(jnp.asarray(lhs.v_parallel) - jnp.asarray(rhs.v_parallel)))),
    )


def run_equivalence_case(
    *,
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
    steps: int,
    dt: float,
) -> dict[str, float]:
    geometry = build_case_geometry(shape)
    initial_state = build_initial_state(geometry)
    direct_state = run_direct_steps(geometry, initial_state, dt=dt, steps=steps)
    sharded_state = run_sharded_steps(
        geometry,
        initial_state,
        dt=dt,
        steps=steps,
        shard_counts=shard_counts,
    )
    return {
        "max_abs_diff": max_state_difference(direct_state, sharded_state),
        "direct_density_max": float(jnp.max(jnp.abs(direct_state.density))),
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
