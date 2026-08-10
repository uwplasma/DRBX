"""Focused tests for production global-to-local FCI map lowering."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "src"), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drbx.geometry import (  # noqa: E402
    FCI_DEP_FIELD_INTERIOR,
    FCI_DEP_PHYSICAL_BOUNDARY,
    FciMaps3D,
    build_shifted_torus_geometry,
)
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_local_fci_geometry,
    build_local_fci_geometries,
    make_shard_mesh,
)


def _with_forward_boundary(geometry, *, x_value: float, z_value: float):
    maps = geometry.maps
    shape = geometry.shape
    forward_boundary = jnp.zeros(shape, dtype=bool).at[0, 0, 0].set(True)
    forward_endpoint_x = maps.forward_endpoint_x.at[0, 0, 0].set(x_value)
    forward_endpoint_y = maps.forward_endpoint_y.at[0, 0, 0].set(
        geometry.grid.y.centers[1]
    )
    forward_endpoint_z = maps.forward_endpoint_z.at[0, 0, 0].set(z_value)
    return dataclasses.replace(
        geometry,
        maps=FciMaps3D(
            forward_x=maps.forward_x,
            forward_y=maps.forward_y,
            backward_x=maps.backward_x,
            backward_y=maps.backward_y,
            forward_endpoint_x=forward_endpoint_x,
            forward_endpoint_y=forward_endpoint_y,
            forward_endpoint_z=forward_endpoint_z,
            backward_endpoint_x=maps.backward_endpoint_x,
            backward_endpoint_y=maps.backward_endpoint_y,
            backward_endpoint_z=maps.backward_endpoint_z,
            forward_length=maps.forward_length,
            backward_length=maps.backward_length,
            forward_boundary=forward_boundary,
            backward_boundary=maps.backward_boundary,
        ),
    )


def _single_shard_map_metadata(sharded, *, include_maps: bool):
    mesh = make_shard_mesh((1, 1, 1))
    spec = P("x", "y", "z")
    sharding = NamedSharding(mesh, spec)
    cell_fields = jax.device_put(sharded.cell_fields, sharding)
    if include_maps:
        map_fields = jax.device_put(sharded.map_fields, sharding)

        def kernel(cell_owned, maps_owned):
            local = assemble_local_fci_geometry(sharded, cell_owned, maps_owned)
            direction = local.maps.forward
            return (
                direction.target_valid,
                direction.endpoint_kind,
                direction.local.active,
                direction.local.dependency_kind,
                direction.local.value_slot,
                direction.local.source_i,
                direction.local.weight,
            )

        return jax.jit(
            jax.shard_map(
                kernel,
                mesh=mesh,
                in_specs=(spec, spec),
                out_specs=(spec, spec, P(), P(), P(), P(), P()),
                check_vma=False,
            )
        )(cell_fields, map_fields)

    def kernel(cell_owned):
        local = assemble_local_fci_geometry(sharded, cell_owned)
        return local.maps.forward.local.active

    return jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=spec,
            out_specs=P(),
            check_vma=False,
        )
    )(cell_fields)


def test_single_shard_retains_outer_wall_rows_and_axis_regular_rows() -> None:
    base = build_shifted_torus_geometry((4, 8, 4), construct_fci_maps=True)
    dz = float(base.grid.z.centers[1] - base.grid.z.centers[0])
    wall = _with_forward_boundary(
        base,
        x_value=float(base.grid.x.faces[-1]),
        z_value=float(base.grid.z.centers[0] + 0.25 * dz),
    )
    sharded = build_local_fci_geometries(base, (1, 1, 1))
    assert sharded.maps_valid
    assert sharded.map_fields is not None

    wall_sharded = build_local_fci_geometries(wall, (1, 1, 1))
    (
        target_valid,
        endpoint_kind,
        active_flat,
        kinds_flat,
        slots_flat,
        source_i_flat,
        weights_flat,
    ) = (
        _single_shard_map_metadata(wall_sharded, include_maps=True)
    )
    assert bool(np.asarray(target_valid[0, 0, 0]))
    assert int(np.asarray(endpoint_kind[0, 0, 0])) == FCI_DEP_PHYSICAL_BOUNDARY

    kinds = np.asarray(kinds_flat).reshape(-1, 8)[0]
    active = np.asarray(active_flat).reshape(-1, 8)[0]
    slots = np.asarray(slots_flat).reshape(-1, 8)[0]
    assert np.all(active)
    # The existing sampler reads PHYSICAL_BOUNDARY endpoints from the prepared
    # ghost/leg halo, so the dependency rows remain FIELD_INTERIOR.  The wall
    # classification is carried separately by endpoint_kind and value_slot.
    assert np.all(kinds == FCI_DEP_FIELD_INTERIOR)
    assert np.all(slots == 0)
    source_i = np.asarray(source_i_flat).reshape(-1, 8)[0]
    h = wall_sharded.domain.layout.halo_width
    nx = wall_sharded.domain.layout.owned_shape[0]
    assert np.all(source_i[::2] == h + nx - 1)
    assert np.all(source_i[1::2] == h + nx)
    weights = np.asarray(weights_flat).reshape(-1, 8)[0]
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights[::2].sum(), 0.5)
    assert np.isclose(weights[1::2].sum(), 0.5)

    # Lower-axis endpoint is represented by the signed-radial ghost/first-cell
    # pair and a half-turn in theta, not by an invalid target or wall closure.
    axis = _with_forward_boundary(
        base,
        x_value=float(base.grid.x.faces[0]),
        z_value=float(base.grid.z.centers[0]),
    )
    axis_sharded = build_local_fci_geometries(
        axis,
        (1, 1, 1),
        axis_regular_axes=(True, False, False),
    )
    axis_metadata = _single_shard_map_metadata(axis_sharded, include_maps=True)
    axis_sources = np.asarray(axis_metadata[5]).reshape(-1, 8)[0]
    axis_kinds = np.asarray(axis_metadata[3]).reshape(-1, 8)[0]
    assert bool(np.asarray(axis_metadata[0][0, 0, 0]))
    assert int(np.asarray(axis_metadata[1][0, 0, 0])) == FCI_DEP_FIELD_INTERIOR
    assert axis_sources[0] == axis_sharded.domain.layout.halo_width - 1
    assert axis_sources[1] == axis_sharded.domain.layout.halo_width
    assert np.all(axis_kinds[:4] == FCI_DEP_FIELD_INTERIOR)


def test_invalid_nan_maps_preserve_inactive_coordinate_path() -> None:
    geometry = build_shifted_torus_geometry((4, 8, 4), construct_fci_maps=True)
    maps = geometry.maps
    invalid = dataclasses.replace(
        geometry,
        maps=FciMaps3D(
            forward_x=maps.forward_x.at[0, 0, 0].set(jnp.nan),
            forward_y=maps.forward_y,
            backward_x=maps.backward_x,
            backward_y=maps.backward_y,
            forward_endpoint_x=maps.forward_endpoint_x,
            forward_endpoint_y=maps.forward_endpoint_y,
            forward_endpoint_z=maps.forward_endpoint_z,
            backward_endpoint_x=maps.backward_endpoint_x,
            backward_endpoint_y=maps.backward_endpoint_y,
            backward_endpoint_z=maps.backward_endpoint_z,
            forward_length=maps.forward_length,
            backward_length=maps.backward_length,
            forward_boundary=maps.forward_boundary,
            backward_boundary=maps.backward_boundary,
        ),
    )
    sharded = build_local_fci_geometries(invalid, (1, 1, 1))
    assert not sharded.maps_valid
    assert sharded.map_fields is None
    active = _single_shard_map_metadata(sharded, include_maps=False)
    assert not bool(np.asarray(active).any())


def test_multi_shard_lowering_preserves_remote_periodic_requests() -> None:
    """Exercise lowering under shard_map with periodic theta/eta decomposition."""

    env = dict(os.environ)
    flags = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count=4".strip()
    env["JAX_ENABLE_X64"] = "true"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--multi-shard-child"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-5000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["all_targets_valid"]
    assert result["all_four_bilinear_rows"]
    assert result["has_remote_requests"]


def _run_multi_shard_child() -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from drbx.native.fci_sharding import make_shard_mesh

    shape = (8, 8, 4)
    shard_counts = (2, 2, 1)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=True)
    sharded = build_local_fci_geometries(geometry, shard_counts)
    mesh = make_shard_mesh(shard_counts)
    spec = P("x", "y", "z")
    sharding = NamedSharding(mesh, spec)
    cell_fields = jax.device_put(sharded.cell_fields, sharding)
    map_fields = jax.device_put(sharded.map_fields, sharding)
    owned = sharded.domain.layout.owned_shape

    def kernel(cell_owned, maps_owned):
        local = assemble_local_fci_geometry(sharded, cell_owned, maps_owned)
        direction = local.maps.forward
        local_active = direction.local.active.reshape(owned + (8,))
        remote_active = direction.remote.active.reshape(owned + (8,))
        return direction.target_valid, local_active, remote_active

    result = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(spec, spec),
            out_specs=(spec, spec, spec),
            check_vma=False,
        )
    )(cell_fields, map_fields)
    target_valid, local_active, remote_active = (np.asarray(value) for value in result)
    print(
        json.dumps(
            {
                "all_targets_valid": bool(target_valid.all()),
                "all_four_bilinear_rows": bool(
                    np.all((local_active.astype(np.int32) + remote_active.astype(np.int32)).sum(axis=-1) == 4)
                ),
                "has_remote_requests": bool(remote_active.any()),
            }
        )
    )


if __name__ == "__main__" and "--multi-shard-child" in sys.argv:
    _run_multi_shard_child()
