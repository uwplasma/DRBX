"""Two-eta-shard equivalence checks for outgoing-FCI-face RLP transfers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_two_eta_shards_match_global_outgoing_face_transfers() -> None:
    """P_e/R_e retain theta aggregates and FCI seam provenance under sharding."""

    env = dict(os.environ)
    flags = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count=4".strip()
    env["JAX_ENABLE_X64"] = "true"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT / "src"), env.get("PYTHONPATH", "")))
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--two-eta-shard-child"],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["device_count"] >= 4
    assert result["owner_eta_local"]
    assert result["has_cross_eta_seam_endpoints"]
    assert result["max_prolong_error"] == 0.0
    assert result["max_restrict_error"] == 0.0
    assert result["max_center_identity_error"] == 0.0
    assert result["remote_seam_active"]
    assert result["remote_transpose_error"] < 2.0e-12


def _topology(layout, *, global_eta_offset: int, global_neta: int):
    """Theta pair aggregate with source-edge endpoints crossing eta seams."""

    import jax.numpy as jnp
    from drbx.native.fci_operators import LocalOutgoingFciFaceTopology3D

    nx, ny, nz = layout.owned_shape
    ii, jj, kk = jnp.indices((nx, ny, nz), dtype=jnp.int32)
    owner_j = jnp.where(jj == 1, 0, jj)
    active_owner = jj != 1
    edge_measure = 1.0 + 2.0 * jj.astype(jnp.float64) + 0.25 * (
        kk + global_eta_offset
    ).astype(jnp.float64)
    aggregate_measure = jnp.zeros((nx, ny, nz), dtype=jnp.float64).at[
        ii, owner_j, kk
    ].add(edge_measure)
    global_k = kk + global_eta_offset
    endpoint_k = (global_k + 1) % global_neta
    return LocalOutgoingFciFaceTopology3D(
        layout=layout,
        edge_owner_i=ii,
        edge_owner_j=owner_j,
        edge_owner_k=kk,
        is_active_owner=active_owner,
        edge_active=jnp.ones((nx, ny, nz), dtype=bool),
        edge_measure=edge_measure,
        aggregate_measure=aggregate_measure,
        edge_destination_i=ii,
        edge_destination_j=jj,
        edge_destination_k=endpoint_k,
        edge_interpolation_provenance=jnp.stack(
            (global_k.astype(jnp.float64), endpoint_k.astype(jnp.float64)), axis=-1
        ),
    )


def _run_two_eta_shard_child() -> None:
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P

    from drbx.geometry import HaloLayout3D
    from drbx.native.fci_operators import (
        prolong_local_outgoing_fci_face_owner_field,
        restrict_local_outgoing_fci_face_field,
    )
    from drbx.native.fci_sharding import make_shard_mesh
    from drbx.native.fci_halo import RemoteFciDependencyExchange
    from drbx.geometry import (
        FCI_DEP_FIELD_INTERIOR,
        LocalFciDirectionMap,
        LocalFciLocalDependencyTable,
        LocalFciRemoteDependencyTable,
        LocalDomain3D,
        ShardSpec3D,
        StencilBuilderContext,
    )

    if len(jax.devices()) < 4:
        raise RuntimeError("forced host backend did not expose four devices")
    shape = (2, 4, 4)
    local_shape = (2, 4, 1)
    mesh = make_shard_mesh((1, 1, 4))
    spec = P("x", "y", "z")
    sharding = NamedSharding(mesh, spec)
    ii, jj, kk = np.indices(shape, dtype=np.int32)
    # Stale alias values deliberately differ: P_e must select only active
    # face owners, while R_e must return zero in those alias slots.
    owner = 100.0 * ii + 10.0 * jj + kk
    fine = 7.0 * ii + 3.0 * jj + kk

    global_topology = _topology(
        HaloLayout3D(shape, halo_width=1), global_eta_offset=0, global_neta=shape[2]
    )
    expected_prolong = prolong_local_outgoing_fci_face_owner_field(
        jnp.asarray(owner), global_topology
    )
    expected_restrict = restrict_local_outgoing_fci_face_field(
        jnp.asarray(fine), global_topology
    )

    def kernel(owner_local, fine_local):
        shard_k = lax.axis_index("z")
        topology = _topology(
            HaloLayout3D(local_shape, halo_width=1),
            global_eta_offset=shard_k,
            global_neta=shape[2],
        )
        prolonged = prolong_local_outgoing_fci_face_owner_field(owner_local, topology)
        restricted = restrict_local_outgoing_fci_face_field(fine_local, topology)
        # Center fields remain a separate, cell-shaped owner space.  With no
        # cell RLP in this fixture their final restriction is identity.
        center_restricted = fine_local
        owner_k_local = topology.edge_owner_k
        endpoints = topology.edge_destination_k
        return prolonged, restricted, center_restricted, owner_k_local, endpoints

    outputs = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(spec, spec),
            out_specs=(spec, spec, spec, spec, spec),
            check_vma=False,
        )
    )(
        jax.device_put(jnp.asarray(owner), sharding),
        jax.device_put(jnp.asarray(fine), sharding),
    )
    prolonged, restricted, center, owner_k, endpoints = map(np.asarray, outputs)
    owner_eta_local = bool(np.all(owner_k == 0))
    global_k = np.broadcast_to(np.arange(shape[2]), shape)
    seam = ((global_k == 1) & (endpoints == 2)) | ((global_k == 3) & (endpoints == 0))

    # This is deliberately a decomposed primitive proof: an actual remote FCI
    # dependency request crosses every eta seam, and the JAX transpose is
    # checked under the same shard_map collectives used by the Galerkin core.
    remote_layout = HaloLayout3D((1, 1, 1), halo_width=1)

    def remote_kernel(local_halo, cotangent):
        shard = lax.axis_index("z")
        domain = LocalDomain3D(
            layout=remote_layout,
            shard_spec=ShardSpec3D(
                global_shape=(1, 1, 4), owned_start=(0, 0, 0),
                owned_stop=(1, 1, 1), shard_index=(0, 0, 0),
                shard_counts=(1, 1, 4), periodic_axes=(False, False, True),
                halo_width=1,
            ),
            mesh_axis_names=(None, None, "z"),
        )
        remote = LocalFciRemoteDependencyTable(
            target_flat=jnp.array([0], dtype=jnp.int32),
            weight=jnp.ones((1,), dtype=jnp.float64),
            receive_slot=jnp.array([0], dtype=jnp.int32), active=jnp.array([True]),
            request_active=jnp.array([True]),
            request_dependency_kind=jnp.array([FCI_DEP_FIELD_INTERIOR], dtype=jnp.int32),
            request_source_global_i=jnp.array([0], dtype=jnp.int32),
            request_source_global_j=jnp.array([0], dtype=jnp.int32),
            request_source_global_k=jnp.array([(shard + 1) % 4], dtype=jnp.int32),
            request_source_shard_index=jnp.array([[0, 0, (shard + 1) % 4]], dtype=jnp.int32),
            request_source_shard_linear=jnp.array([(shard + 1) % 4], dtype=jnp.int32),
            request_source_owner_local_i=jnp.array([1], dtype=jnp.int32),
            request_source_owner_local_j=jnp.array([1], dtype=jnp.int32),
            request_source_owner_local_k=jnp.array([1], dtype=jnp.int32),
            request_value_slot=jnp.array([0], dtype=jnp.int32),
        )
        direction = LocalFciDirectionMap(
            layout=remote_layout,
            local=LocalFciLocalDependencyTable(
                target_flat=jnp.array([0], dtype=jnp.int32),
                source_i=jnp.array([0], dtype=jnp.int32),
                source_j=jnp.array([0], dtype=jnp.int32),
                source_k=jnp.array([0], dtype=jnp.int32),
                weight=jnp.zeros((1,), dtype=jnp.float64),
                active=jnp.array([False]),
            ),
            remote=remote,
            connection_length=jnp.ones((1, 1, 1), dtype=jnp.float64),
        )
        exchange = lambda value: RemoteFciDependencyExchange()(
            field_halo=value, direction=direction,
            context=StencilBuilderContext(layout=remote_layout, domain=domain),
            cut_wall_bc=None,
        )
        remote_value = exchange(local_halo)
        adjoint = jax.linear_transpose(exchange, jnp.zeros_like(local_halo))(cotangent)[0]
        error = lax.psum(jnp.sum(remote_value * cotangent) - jnp.sum(local_halo * adjoint), "z")
        return error, jnp.asarray([shard], dtype=jnp.int32)

    remote_input = jnp.arange(3 * 3 * 12, dtype=jnp.float64).reshape((3, 3, 12))
    remote_cotangent = 0.3 + jnp.arange(4, dtype=jnp.float64)
    remote_outputs = jax.jit(jax.shard_map(
        remote_kernel, mesh=mesh, in_specs=(P(None, None, "z"), P("z")),
        out_specs=(P(), P("z",)), check_vma=False,
    ))(jax.device_put(remote_input, NamedSharding(mesh, P(None, None, "z"))),
       jax.device_put(remote_cotangent, NamedSharding(mesh, P("z"))))
    remote_error, remote_shards = map(np.asarray, remote_outputs)
    print(json.dumps({
        "device_count": len(jax.devices()),
        "owner_eta_local": bool(owner_eta_local),
        "has_cross_eta_seam_endpoints": bool(np.any(seam)),
        "max_prolong_error": float(np.max(np.abs(prolonged - np.asarray(expected_prolong)))),
        "max_restrict_error": float(np.max(np.abs(restricted - np.asarray(expected_restrict)))),
        "max_center_identity_error": float(np.max(np.abs(center - fine))),
        "remote_seam_active": bool(np.array_equal(remote_shards, np.arange(4))),
        "remote_transpose_error": float(np.abs(remote_error)),
    }))


if __name__ == "__main__":
    if "--two-eta-shard-child" not in sys.argv:
        raise SystemExit("expected --two-eta-shard-child")
    _run_two_eta_shard_child()
