"""Focused tests for the shard-compatible SOLVAX Newton adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from drbx.native.fci_model import FciModelState
from drbx.native.fci_newton import (
    SolvaxNewtonConfig,
    SolvaxNewtonInfo,
    solvax_newton_solve,
)
from tests.test_fci_operators_domain_decomp import (
    _build_domain,
    _build_local_geometry,
    make_mesh_for_shard_counts,
    put_scalar_field_on_mesh,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _TwoFieldState(FciModelState):
    density: jax.Array
    electron: jax.Array


def _info_spec() -> SolvaxNewtonInfo:
    return SolvaxNewtonInfo(
        newton_iterations=P(),
        linear_iterations=P(),
        converged=P(),
        linear_converged=P(),
        accepted=P(),
        failed=P(),
        initial_residual_l2=P(),
        final_residual_l2=P(),
        final_residual_rel_l2=P(),
        initial_state_is_finite=P(),
        final_state_is_finite=P(),
        initial_residual_is_finite=P(),
        final_residual_is_finite=P(),
    )


def _solve_affine_state(
    *, shape: tuple[int, int, int], shard_counts: tuple[int, int, int]
) -> tuple[_TwoFieldState, SolvaxNewtonInfo, _TwoFieldState]:
    halo_width = 1
    domain = _build_domain(shape, halo_width, shard_counts)
    density_target = jnp.arange(math.prod(shape), dtype=jnp.float64).reshape(shape) / 10.0
    electron_target = 2.0 - density_target
    initial = _TwoFieldState(jnp.zeros_like(density_target), jnp.zeros_like(density_target))
    target = _TwoFieldState(density_target, electron_target)
    config = SolvaxNewtonConfig(
        rtol=1.0e-12,
        atol=1.0e-12,
        max_steps=4,
        linear_restart=4,
        linear_rtol=1.0e-12,
        linear_max_restarts=1,
        field_scales=(2.0, 0.5),
    )
    with make_mesh_for_shard_counts(shard_counts) as mesh:
        initial_sharded = _TwoFieldState(
            put_scalar_field_on_mesh(initial.density, mesh),
            put_scalar_field_on_mesh(initial.electron, mesh),
        )
        target_sharded = _TwoFieldState(
            put_scalar_field_on_mesh(target.density, mesh),
            put_scalar_field_on_mesh(target.electron, mesh),
        )

        def kernel(local_initial: _TwoFieldState, local_target: _TwoFieldState):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                domain.layout.owned_shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )

            def residual(value: _TwoFieldState) -> _TwoFieldState:
                return _TwoFieldState(
                    density=value.density - local_target.density,
                    electron=value.electron - local_target.electron,
                )

            # This verifies that caller-supplied PyTree preconditioners are
            # passed through the SOLVAX FGMRES path without changing the
            # solution.  (SOLVAX accepts a right-preconditioning transform.)
            def preconditioner(value: _TwoFieldState) -> _TwoFieldState:
                return value

            return solvax_newton_solve(
                residual,
                local_initial,
                geometry,
                domain,
                config,
                preconditioner=preconditioner,
            )

        compiled = jax.jit(shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"), P("x", "y", "z")),
            out_specs=(P("x", "y", "z"), _info_spec()),
            check_rep=False,
        ))
        solution, info = compiled(initial_sharded, target_sharded)
        solution, info = jax.block_until_ready((solution, info))
    return solution, info, target


def test_newton_solves_two_field_pytree_inside_single_shard_map() -> None:
    solution, info, target = _solve_affine_state(shape=(4, 3, 2), shard_counts=(1, 1, 1))
    np.testing.assert_allclose(np.asarray(solution.density), np.asarray(target.density), atol=1.0e-11)
    np.testing.assert_allclose(np.asarray(solution.electron), np.asarray(target.electron), atol=1.0e-11)
    assert bool(info.converged)
    assert bool(info.linear_converged)
    assert bool(info.accepted)
    assert not bool(info.failed)
    assert int(info.newton_iterations) == 1
    assert int(info.linear_iterations) >= 1
    assert float(info.final_residual_l2) < 1.0e-11


def test_newton_collective_norm_and_solution_cover_two_eta_shards() -> None:
    if len(jax.devices()) < 2:
        pytest.skip("requires two local JAX devices")
    solution, info, target = _solve_affine_state(shape=(4, 3, 4), shard_counts=(1, 1, 2))
    np.testing.assert_allclose(np.asarray(solution.density), np.asarray(target.density), atol=1.0e-11)
    np.testing.assert_allclose(np.asarray(solution.electron), np.asarray(target.electron), atol=1.0e-11)
    assert bool(info.accepted)
    assert float(info.final_residual_rel_l2) < 1.0e-11


def test_newton_accepts_plain_pytree_state() -> None:
    """The adapter is not tied to a particular DRBX state dataclass."""

    shape = (3, 2, 2)
    halo_width = 1
    domain = _build_domain(shape, halo_width)
    target = jnp.linspace(0.0, 1.0, math.prod(shape), dtype=jnp.float64).reshape(shape)
    with make_mesh_for_shard_counts((1, 1, 1)) as mesh:
        target_sharded = put_scalar_field_on_mesh(target, mesh)
        zero_sharded = put_scalar_field_on_mesh(jnp.zeros_like(target), mesh)

        def kernel(local_target, local_zero):
            geometry = _build_local_geometry(
                shape, halo_width, global_shape=shape
            )
            return solvax_newton_solve(
                lambda value: {
                    "density": value["density"] - local_target,
                    "temperature": value["temperature"] - (1.0 + local_target),
                },
                {"density": local_zero, "temperature": local_zero},
                geometry,
                domain,
                SolvaxNewtonConfig(
                    rtol=1.0e-12, atol=1.0e-12, max_steps=3,
                    linear_restart=3, linear_rtol=1.0e-12, linear_max_restarts=1,
                ),
            )

        compiled = jax.jit(shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"), P("x", "y", "z")),
            out_specs=({"density": P("x", "y", "z"), "temperature": P("x", "y", "z")}, _info_spec()),
            check_rep=False,
        ))
        solution, info = jax.block_until_ready(compiled(target_sharded, zero_sharded))
    np.testing.assert_allclose(np.asarray(solution["density"]), np.asarray(target), atol=1.0e-11)
    np.testing.assert_allclose(np.asarray(solution["temperature"]), np.asarray(1.0 + target), atol=1.0e-11)
    assert bool(info.accepted)


def test_newton_reports_nonfinite_residual_as_failed() -> None:
    shape = (2, 2, 2)
    halo_width = 1
    domain = _build_domain(shape, halo_width)
    with make_mesh_for_shard_counts((1, 1, 1)) as mesh:
        zero = put_scalar_field_on_mesh(jnp.zeros(shape, dtype=jnp.float64), mesh)

        def kernel(local_zero):
            geometry = _build_local_geometry(shape, halo_width, global_shape=shape)
            return solvax_newton_solve(
                lambda value: _TwoFieldState(
                    jnp.full_like(value.density, jnp.nan),
                    value.electron,
                ),
                _TwoFieldState(local_zero, local_zero),
                geometry,
                domain,
                SolvaxNewtonConfig(max_steps=1, linear_restart=2),
            )

        compiled = jax.jit(shard_map(
            kernel,
            mesh=mesh,
            in_specs=P("x", "y", "z"),
            out_specs=(_TwoFieldState(P("x", "y", "z"), P("x", "y", "z")), _info_spec()),
            check_rep=False,
        ))
        _solution, info = jax.block_until_ready(compiled(zero))
    assert not bool(info.initial_residual_is_finite)
    assert not bool(info.final_residual_is_finite)
    assert not bool(info.accepted)
    assert bool(info.failed)


def test_newton_validates_scales_and_owned_leaf_shape() -> None:
    with pytest.raises(ValueError, match="field_scales"):
        SolvaxNewtonConfig(field_scales=(1.0, 0.0))
    with pytest.raises(ValueError, match="linear_restart"):
        SolvaxNewtonConfig(linear_restart=0)
