from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax_drb.geometry import HaloLayout3D, LocalDomain3D, ShardSpec3D
from jax_drb.native import Rk4Stepper, sum_stage_outputs
from jax_drb.native.fci_boundaries import LocalBoundaryData3D
from jax_drb.native.fci_halo import PreparedLocalState3D
from jax_drb.native.fci_model import FciModelState, inject_owned_state_to_halo


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ScalarState(FciModelState):
    field: jax.Array


@dataclass(frozen=True)
class _GeometryWithLayout:
    layout: HaloLayout3D


def _domain(
    owned_shape: tuple[int, int, int] = (2, 2, 2),
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    mesh_axis_names: tuple[str | None, str | None, str | None] = (
        None,
        None,
        None,
    ),
) -> LocalDomain3D:
    layout = HaloLayout3D(owned_shape, halo_width=1)
    global_shape = tuple(
        int(local) * int(count)
        for local, count in zip(owned_shape, shard_counts)
    )
    return LocalDomain3D(
        layout=layout,
        shard_spec=ShardSpec3D(
            global_shape=global_shape,
            owned_start=(0, 0, 0),
            owned_stop=owned_shape,
            shard_index=(0, 0, 0),
            shard_counts=shard_counts,
            periodic_axes=(False, False, False),
            halo_width=layout.halo_width,
        ),
        mesh_axis_names=mesh_axis_names,
    )


@dataclass
class _RecordingPreparer:
    calls: list[tuple[tuple[int, ...], object]] | None = None

    def __call__(
        self,
        state_owned: _ScalarState,
        geometry: _GeometryWithLayout,
        domain: LocalDomain3D,
        cut_wall_geometry: object | None = None,
    ) -> PreparedLocalState3D:
        if self.calls is not None:
            self.calls.append((state_owned.field.shape, cut_wall_geometry))
        return PreparedLocalState3D(
            state_halo=inject_owned_state_to_halo(state_owned, domain.layout),
            boundary_data=LocalBoundaryData3D(),
        )


@dataclass(frozen=True)
class _ScalarPreparedRhs:
    """Concrete scalar model RHS: prepare stage, then compute owned RHS."""

    rate: jax.Array | float
    offset: jax.Array | float
    geometry: _GeometryWithLayout
    domain: LocalDomain3D
    state_preparer: _RecordingPreparer

    def __call__(
        self,
        state_owned: _ScalarState,
        stage_time: float | jax.Array,
        carry: jax.Array | None,
    ) -> tuple[_ScalarState, jax.Array | None, jax.Array]:
        prepared_stage = self.state_preparer(
            state_owned,
            self.geometry,
            self.domain,
            None,
        )
        owned = prepared_stage.state_halo.field[
            self.domain.layout.owned_slices_cell
        ]
        rhs = _ScalarState(field=self.rate * owned + self.offset)
        next_carry = None if carry is None else carry + 1
        return rhs, next_carry, jnp.asarray([stage_time, jnp.mean(owned)])


@dataclass(frozen=True)
class _BadShapeRhs:
    def __call__(
        self,
        state_owned: _ScalarState,
        stage_time: float | jax.Array,
        carry: None,
    ) -> tuple[_ScalarState, None, jax.Array]:
        del state_owned, stage_time
        return _ScalarState(field=jnp.zeros((1, 1, 1))), carry, jnp.array(0.0)


def test_rk4_stepper_scalar_rhs_prepares_every_stage() -> None:
    domain = _domain()
    geometry = _GeometryWithLayout(domain.layout)
    state = _ScalarState(
        field=jnp.arange(math.prod(domain.layout.owned_shape), dtype=jnp.float64)
        .reshape(domain.layout.owned_shape)
    )
    preparer = _RecordingPreparer(calls=[])
    rhs = _ScalarPreparedRhs(
        rate=-0.75,
        offset=0.25,
        geometry=geometry,
        domain=domain,
        state_preparer=preparer,
    )
    stepper = Rk4Stepper(rhs)

    object_result = stepper(
        state,
        time=0.0,
        timestep=0.2,
        carry=jnp.array(0, dtype=jnp.int32),
    )

    assert len(preparer.calls) == 4
    assert len(object_result.stage_aux) == 4
    assert all(shape == domain.layout.owned_shape for shape, _ in preparer.calls)
    assert int(object_result.carry) == 4
    z = -0.75 * 0.2
    homogeneous_factor = 1.0 + z + 0.5 * z**2 + (z**3) / 6.0 + (z**4) / 24.0
    steady_state = 0.25 / 0.75
    expected = steady_state + homogeneous_factor * (state.field - steady_state)
    assert jnp.allclose(object_result.state.field, expected)
    assert jnp.allclose(
        sum_stage_outputs(object_result.stage_aux),
        sum(object_result.stage_aux),
    )


def test_rk4_stepper_rejects_rhs_with_wrong_state_shape() -> None:
    domain = _domain()
    state = _ScalarState(field=jnp.ones(domain.layout.owned_shape))

    with pytest.raises(ValueError, match="must have shape"):
        Rk4Stepper(_BadShapeRhs())(
            state,
            time=0.0,
            timestep=0.1,
            carry=None,
        )


def test_rk4_stepper_shard_map_scalar_ode_matches_global_result() -> None:
    if len(jax.devices()) < 2:
        pytest.skip("requires at least two local JAX devices")

    devices = np.asarray(jax.devices()[:2], dtype=object)
    mesh = Mesh(devices, ("z",))
    owned_shape = (2, 2, 2)
    domain = _domain(
        owned_shape,
        shard_counts=(1, 1, 2),
        mesh_axis_names=(None, None, "z"),
    )
    geometry = _GeometryWithLayout(domain.layout)
    rate = jnp.asarray(-0.5, dtype=jnp.float64)
    timestep = jnp.asarray(0.125, dtype=jnp.float64)
    global_field = jnp.arange(16, dtype=jnp.float64).reshape((2, 2, 4))
    rhs = _ScalarPreparedRhs(
        rate=rate,
        offset=0.0,
        geometry=geometry,
        domain=domain,
        state_preparer=_RecordingPreparer(),
    )
    stepper = Rk4Stepper(rhs)

    def kernel(field_local: jax.Array) -> jax.Array:
        result = stepper(
            _ScalarState(field=field_local),
            time=0.0,
            timestep=timestep,
            carry=None,
        )
        return result.state.field

    sharding = NamedSharding(mesh, P(None, None, "z"))
    sharded_field = jax.device_put(global_field, sharding)
    mapped = shard_map(
        kernel,
        mesh=mesh,
        in_specs=P(None, None, "z"),
        out_specs=P(None, None, "z"),
        check_rep=False,
    )

    actual = mapped(sharded_field)
    z = rate * timestep
    rk4_factor = 1.0 + z + 0.5 * z**2 + (z**3) / 6.0 + (z**4) / 24.0
    expected = rk4_factor * global_field
    assert jnp.allclose(actual, expected)
