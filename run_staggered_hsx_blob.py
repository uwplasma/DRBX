#!/usr/bin/env python3
"""Versioned launcher for the experimental source-edge FCI velocity layout.

This leaves the shared HSX driver unmodified.  It loads that driver with this
worktree's ``src`` tree, consumes the one experimental switch, and records the
layout choice in the process environment consumed by ``LocalFciDrbEBRhs``.
Vi/Ve use outgoing FCI source edges under ``fci-staggered``; their
perpendicular terms reconstruct face values at centers and project the result
back to outgoing faces.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


WORKTREE = Path(__file__).resolve().parent
SHARED_DRIVER = WORKTREE.parent / "simulate_hsx_blob.py"


def _transform_shared_driver_source(source: str) -> str:
    """Install the bounded staggered-face wiring into the shared driver text."""

    replacements = (
        (
            'DRBX_SRC = SCRIPT_DIR / "DRBX" / "src"',
            f"DRBX_SRC = Path({str(WORKTREE / 'src')!r})",
        ),
        (
            "from drbx.native.fci_operators import (  # noqa: E402\n"
            "    build_local_perp_laplacian_face_projectors,\n"
            "    expand_local_control_volume_owner_field,\n"
            ")",
            "from drbx.native.fci_operators import (  # noqa: E402\n"
            "    OUTGOING_FCI_FACE_OWNERSHIP_POLICY,\n"
            "    build_local_outgoing_fci_face_topology_from_geometry,\n"
            "    build_local_perp_laplacian_face_projectors,\n"
            "    expand_local_control_volume_owner_field,\n"
            "    prolong_local_outgoing_fci_face_owner_field,\n"
            ")",
        ),
        (
            "    control_volume_field_count: int = RLP_PACKED_FIELD_COUNT,\n"
            "    owner_host_geometry=None,\n"
            ") -> FciDrbEBState:",
            "    control_volume_field_count: int = RLP_PACKED_FIELD_COUNT,\n"
            "    owner_host_geometry=None,\n"
            "    outgoing_face_topology_host=None,\n"
            ") -> FciDrbEBState:",
        ),
        (
            "        result[name] = np.where(owner_mask, averaged, 0.0)\n"
            "    return FciDrbEBState(**result)\n",
            "        # Vi/Ve begin as centered physical data.  In the staggered\n"
            "        # layout they must remain fine until the local FCI c2f/Re\n"
            "        # initialization pass below; applying PcRc here would erase\n"
            "        # their source-edge support before that projection.\n"
            "        if (\n"
            "            os.environ.get(\"DRBX_PARALLEL_VELOCITY_LAYOUT\") == \"fci-staggered\"\n"
            "            and name in (\"Vi\", \"Ve\")\n"
            "        ):\n"
            "            result[name] = raw\n"
            "        else:\n"
            "            result[name] = np.where(owner_mask, averaged, 0.0)\n"
            "    return FciDrbEBState(**result)\n",
        ),
        (
            "    if owner_host_geometry is not None:\n"
            "        initial_state = _aggregate_initial_owner_state(initial_state, owner_host_geometry)\n"
            "        _assert_owner_sparse(initial_state, owner_host_geometry)\n"
            "        print(\n"
            "            \"[simulation] initial state volume-aggregated into canonical owners; \"\n"
            "            \"all merged aliases are zero\",\n"
            "            flush=True,\n"
            "        )\n",
            "    if owner_host_geometry is not None:\n"
            "        initial_state = _aggregate_initial_owner_state(initial_state, owner_host_geometry)\n"
            "        if os.environ.get(\"DRBX_PARALLEL_VELOCITY_LAYOUT\") != \"fci-staggered\":\n"
            "            _assert_owner_sparse(initial_state, owner_host_geometry)\n"
            "        print(\n"
            "            \"[simulation] initial scalar state volume-aggregated into canonical owners; \"\n"
            "            \"staggered Vi/Ve are projected through FCI faces before evolution\",\n"
            "            flush=True,\n"
            "        )\n",
        ),
        (
            "    def materialized_state(current_state: FciDrbEBState) -> FciDrbEBState:\n"
            "        if owner_host_geometry is None:\n"
            "            return current_state\n"
            "        return _materialize_owner_state(current_state, owner_host_geometry)\n",
            "    def materialized_state(current_state: FciDrbEBState) -> FciDrbEBState:\n"
            "        materialized = (\n"
            "            current_state if owner_host_geometry is None\n"
            "            else _materialize_owner_state(current_state, owner_host_geometry)\n"
            "        )\n"
            "        if outgoing_face_topology_host is None:\n"
            "            return materialized\n"
            "        face_active = jnp.asarray(outgoing_face_topology_host[\"edge_active\"], dtype=bool)\n"
            "        owner_i = jnp.asarray(outgoing_face_topology_host[\"edge_owner_i\"], dtype=jnp.int32)\n"
            "        owner_j = jnp.asarray(outgoing_face_topology_host[\"edge_owner_j\"], dtype=jnp.int32)\n"
            "        owner_k = jnp.asarray(outgoing_face_topology_host[\"edge_owner_k\"], dtype=jnp.int32)\n"
            "        def materialize_face(values):\n"
            "            return jnp.where(face_active, values[owner_i, owner_j, owner_k], 0.0)\n"
            "        return materialized.replace(\n"
            "            Vi=materialize_face(current_state.Vi),\n"
            "            Ve=materialize_face(current_state.Ve),\n"
            "        )\n",
        ),
        (
            "def _assert_owner_sparse(state: FciDrbEBState, host_geometry) -> None:\n"
            "    mask = ~np.asarray(host_geometry.topology.is_active_owner, dtype=bool)\n"
            "    maximum = max(\n"
            "        (float(np.max(np.abs(np.asarray(value)[mask]))) for _, value in state.field_items()),\n"
            "        default=0.0,\n"
            "    )\n",
            "def _assert_owner_sparse(state: FciDrbEBState, host_geometry, outgoing_face_topology_host=None) -> None:\n"
            "    cell_mask = ~np.asarray(host_geometry.topology.is_active_owner, dtype=bool)\n"
            "    face_mask = (cell_mask if outgoing_face_topology_host is None else\n"
            "                 ~np.asarray(outgoing_face_topology_host[\"is_active_owner\"], dtype=bool))\n"
            "    maximum = max(\n"
            "        float(np.max(np.abs(np.asarray(value)[mask])) if np.any(mask) else 0.0)\n"
            "        for name, value in state.field_items()\n"
            "        for mask in (face_mask if name in (\"Vi\", \"Ve\") else cell_mask,)\n"
            "    )\n",
        ),
        (
            "def _materialize_owner_array(array: np.ndarray, host_geometry) -> np.ndarray:\n"
            "    \"\"\"Expand a leading-term-axis full-grid diagnostic array.\"\"\"\n"
            "\n"
            "    value = np.asarray(array, dtype=np.float64)\n"
            "    if host_geometry is None or value.ndim != 4:\n"
            "        return value\n"
            "    owner_index = np.asarray(host_geometry.topology.owner_index, dtype=np.int32)\n"
            "    return value[(slice(None),) + tuple(np.moveaxis(owner_index, -1, 0))]\n\n",
            "def _materialize_owner_array(array: np.ndarray, host_geometry) -> np.ndarray:\n"
            "    \"\"\"Expand a leading-term-axis cell-owner diagnostic array.\"\"\"\n"
            "\n"
            "    value = np.asarray(array, dtype=np.float64)\n"
            "    if host_geometry is None or value.ndim != 4:\n"
            "        return value\n"
            "    owner_index = np.asarray(host_geometry.topology.owner_index, dtype=np.int32)\n"
            "    return value[(slice(None),) + tuple(np.moveaxis(owner_index, -1, 0))]\n\n"
            "def _materialize_face_owner_array(array: np.ndarray, face_topology_host) -> np.ndarray:\n"
            "    \"\"\"Expand a leading-term-axis outgoing-face owner diagnostic array.\"\"\"\n"
            "\n"
            "    value = np.asarray(array, dtype=np.float64)\n"
            "    if face_topology_host is None or value.ndim != 4:\n"
            "        return value\n"
            "    result = value[(slice(None),) + (\n"
            "        np.asarray(face_topology_host[\"edge_owner_i\"], dtype=np.int32),\n"
            "        np.asarray(face_topology_host[\"edge_owner_j\"], dtype=np.int32),\n"
            "        np.asarray(face_topology_host[\"edge_owner_k\"], dtype=np.int32),\n"
            "    )]\n"
            "    return np.where(np.asarray(face_topology_host[\"edge_active\"], dtype=bool)[None], result, 0.0)\n\n",
        ),
        (
            "    control_volume_geometry=None,\n"
            "    control_volume_boundary_bc=None,\n"
            ") -> LocalFciDrbEBRhs:",
            "    control_volume_geometry=None,\n"
            "    control_volume_boundary_bc=None,\n"
            "    outgoing_face_topology=None,\n"
            ") -> LocalFciDrbEBRhs:",
        ),
        (
            "        control_volume_boundary_bc=control_volume_boundary_bc,\n"
            "    )\n"
            "    model = LocalFciDrbEBRhs(",
            "        control_volume_boundary_bc=control_volume_boundary_bc,\n"
            "        outgoing_face_topology=outgoing_face_topology,\n"
            "    )\n"
            "    model = LocalFciDrbEBRhs(",
        ),
        (
            "        return build_local_eb_model(\n"
            "            local_geometry,",
            "        local_outgoing_face_topology = None\n"
            "        if os.environ.get(\"DRBX_PARALLEL_VELOCITY_LAYOUT\") == \"fci-staggered\":\n"
            "            if local_geometry.maps is None:\n"
            "                raise ValueError(\"fci-staggered requires local FCI maps\")\n"
            "            if (\n"
            "                local_control_volume_geometry is None\n"
            "                or not local_control_volume_geometry.has_angular_agglomeration\n"
            "            ):\n"
            "                raise ValueError(\"fci-staggered requires angular-RLP control-volume geometry\")\n"
            "            local_outgoing_face_topology = (\n"
            "                build_local_outgoing_fci_face_topology_from_geometry(\n"
            "                    local_control_volume_geometry.cells, local_geometry.maps\n"
            "                )\n"
            "            )\n"
            "        return build_local_eb_model(\n"
            "            local_geometry,",
        ),
        (
            "            control_volume_boundary_bc=control_volume_boundary_bc,\n"
            "        )\n\n"
            "    wall_projectors = None",
            "            control_volume_boundary_bc=control_volume_boundary_bc,\n"
            "            outgoing_face_topology=local_outgoing_face_topology,\n"
            "        )\n\n"
            "    wall_projectors = None",
        ),
        (
            "    shard_count = int(np.prod(sharded_geometry.shard_counts))\n",
            "    def diagnostic_state(current_state: FciDrbEBState, local_control_volume_geometry, outgoing_face_topology=None) -> FciDrbEBState:\n"
            "        if local_control_volume_geometry is None:\n"
            "            return current_state\n"
            "        cells = local_control_volume_geometry.cells\n"
            "        scalar = lambda value: expand_local_control_volume_owner_field(value, cells)\n"
            "        face = outgoing_face_topology\n"
            "        if face is None:\n"
            "            return current_state.replace(\n"
            "                density=scalar(current_state.density), phi=scalar(current_state.phi),\n"
            "                Te=scalar(current_state.Te), Ti=scalar(current_state.Ti),\n"
            "                Vi=scalar(current_state.Vi), Ve=scalar(current_state.Ve),\n"
            "                vorticity=scalar(current_state.vorticity),\n"
            "            )\n"
            "        return current_state.replace(\n"
            "            density=scalar(current_state.density), phi=scalar(current_state.phi),\n"
            "            Te=scalar(current_state.Te), Ti=scalar(current_state.Ti),\n"
            "            Vi=prolong_local_outgoing_fci_face_owner_field(current_state.Vi, face),\n"
            "            Ve=prolong_local_outgoing_fci_face_owner_field(current_state.Ve, face),\n"
            "            vorticity=scalar(current_state.vorticity),\n"
            "        )\n"
            "\n"
            "    shard_count = int(np.prod(sharded_geometry.shard_counts))\n",
        ),
        (
            "    domain = sharded_geometry.domain\n"
            "    print(\n"
            "        f\"sharded geometry inputs ready in ",
            "    staggered_face_provenance = None\n"
            "    staggered_face_topology_host = None\n"
            "    if os.environ.get(\"DRBX_PARALLEL_VELOCITY_LAYOUT\") == \"fci-staggered\":\n"
            "        if (\n"
            "            control_volume_descriptor is None\n"
            "            or control_volume_assembler is None\n"
            "            or control_volume_fields is None\n"
            "            or sharded_geometry.map_fields is None\n"
            "        ):\n"
            "            raise ValueError(\"fci-staggered provenance requires angular RLP and FCI maps\")\n"
            "\n"
            "        def staggered_face_preflight(cell_fields_owned, map_fields_owned, cv_fields_owned):\n"
            "            local_geometry = assemble_local_fci_geometry(\n"
            "                sharded_geometry, cell_fields_owned, map_fields_owned\n"
            "            )\n"
            "            local_cv = control_volume_assembler(\n"
            "                control_volume_descriptor, cv_fields_owned, local_geometry\n"
            "            )\n"
            "            topology = build_local_outgoing_fci_face_topology_from_geometry(\n"
            "                local_cv.cells, local_geometry.maps\n"
            "            )\n"
            "            return (\n"
            "                topology.edge_owner_i + jax.lax.axis_index(\"x\") * topology.shape[0],\n"
            "                topology.edge_owner_j + jax.lax.axis_index(\"y\") * topology.shape[1],\n"
            "                topology.edge_owner_k + jax.lax.axis_index(\"z\") * topology.shape[2],\n"
            "                topology.edge_active, topology.is_active_owner,\n"
            "                topology.edge_measure, topology.aggregate_measure,\n"
            "                topology.edge_destination_i + jax.lax.axis_index(\"x\") * topology.shape[0],\n"
            "                topology.edge_destination_j + jax.lax.axis_index(\"y\") * topology.shape[1],\n"
            "                topology.edge_destination_k + jax.lax.axis_index(\"z\") * topology.shape[2],\n"
            "                topology.edge_interpolation_provenance, topology.edge_destination_support,\n"
            "            )\n"
            "\n"
            "        staggered_face_preflight_compiled = jax.jit(jax.shard_map(\n"
            "            staggered_face_preflight, mesh=mesh,\n"
            "            in_specs=(P(\"x\", \"y\", \"z\", None),) * 3,\n"
            "            out_specs=(P(\"x\", \"y\", \"z\"),) * 10 + (P(\"x\", \"y\", \"z\", None),) * 2,\n"
            "            check_vma=True,\n"
            "        ))\n"
            "        staggered_face_arrays = tuple(np.asarray(value) for value in (\n"
            "            staggered_face_preflight_compiled(\n"
            "                jax.device_put(jnp.asarray(sharded_geometry.cell_fields, dtype=jnp.float64),\n"
            "                               NamedSharding(mesh, P(\"x\", \"y\", \"z\", None))),\n"
            "                jax.device_put(jnp.asarray(sharded_geometry.map_fields, dtype=jnp.float64),\n"
            "                               NamedSharding(mesh, P(\"x\", \"y\", \"z\", None))),\n"
            "                jax.device_put(jnp.asarray(control_volume_fields, dtype=jnp.float64),\n"
            "                               NamedSharding(mesh, P(\"x\", \"y\", \"z\", None))),\n"
            "            )\n"
            "        ))\n"
            "        (face_owner_i, face_owner_j, face_owner_k, face_active, face_owner_active,\n"
            "         face_measure, face_aggregate_measure, face_destination_i, face_destination_j,\n"
            "         face_destination_k, face_provenance, face_destination_support) = staggered_face_arrays\n"
            "        fine_indices = np.indices(face_active.shape, dtype=np.int32)\n"
            "        face_alias = face_active & (\n"
            "            (face_owner_i != fine_indices[0]) | (face_owner_j != fine_indices[1])\n"
            "            | (face_owner_k != fine_indices[2])\n"
            "        )\n"
            "        face_member_count = np.zeros(face_active.shape, dtype=np.int64)\n"
            "        np.add.at(face_member_count,\n"
            "                  (face_owner_i[face_active], face_owner_j[face_active], face_owner_k[face_active]), 1)\n"
            "\n"
            "        def face_sha256(*arrays):\n"
            "            digest = hashlib.sha256()\n"
            "            for array in arrays:\n"
            "                canonical = np.ascontiguousarray(array)\n"
            "                digest.update(str(canonical.dtype).encode())\n"
            "                digest.update(np.asarray(canonical.shape, dtype=np.int64).tobytes())\n"
            "                digest.update(canonical.tobytes())\n"
            "            return digest.hexdigest()\n"
            "\n"
            "        staggered_face_provenance = {\n"
            "            \"face_basis_policy\": OUTGOING_FCI_FACE_OWNERSHIP_POLICY,\n"
            "            \"face_basis_version\": OUTGOING_FCI_FACE_OWNERSHIP_POLICY.rsplit(\"-v\", 1)[-1],\n"
            "            \"fine_face_count\": int(np.count_nonzero(face_active)),\n"
            "            \"face_owner_count\": int(np.count_nonzero(face_owner_active)),\n"
            "            \"face_alias_count\": int(np.count_nonzero(face_alias)),\n"
            "            \"face_max_fine_edges_per_owner\": int(np.max(face_member_count, initial=0)),\n"
            "            \"face_owner_map_sha256\": face_sha256(\n"
            "                face_owner_i, face_owner_j, face_owner_k, face_active, face_owner_active\n"
            "            ),\n"
            "            \"face_measure_sha256\": face_sha256(face_measure, face_aggregate_measure),\n"
            "            \"face_provenance_sha256\": face_sha256(\n"
            "                face_destination_i, face_destination_j, face_destination_k, face_provenance,\n"
            "                face_destination_support\n"
            "            ),\n"
            "        }\n"
            "        staggered_face_topology_host = {\n"
            "            \"edge_owner_i\": face_owner_i, \"edge_owner_j\": face_owner_j,\n"
            "            \"edge_owner_k\": face_owner_k, \"edge_active\": face_active,\n"
            "            \"is_active_owner\": face_owner_active, \"edge_measure\": face_measure,\n"
            "            \"aggregate_measure\": face_aggregate_measure,\n"
            "            \"edge_destination_i\": face_destination_i,\n"
            "            \"edge_destination_j\": face_destination_j,\n"
            "            \"edge_destination_k\": face_destination_k,\n"
            "            \"edge_interpolation_provenance\": face_provenance,\n"
            "            \"edge_destination_support\": face_destination_support,\n"
            "        }\n"
            "        print(\n"
            "            \"[staggered-face-preflight] \"\n"
            "            f\"fine={staggered_face_provenance['fine_face_count']}, \"\n"
            "            f\"owners={staggered_face_provenance['face_owner_count']}, \"\n"
            "            f\"aliases={staggered_face_provenance['face_alias_count']}, \"\n"
            "            f\"max_edges_per_owner={staggered_face_provenance['face_max_fine_edges_per_owner']}\",\n"
            "            flush=True,\n"
            "        )\n"
            "    domain = sharded_geometry.domain\n"
            "    print(\n"
            "        f\"sharded geometry inputs ready in ",
        ),
        (
            "    phi_start = time.perf_counter()\n",
            "    if outgoing_face_topology_host is not None:\n"
            "        # The host initializer supplies centered physical Vi/Ve.  Do\n"
            "        # this conversion after local FCI geometry/halos exist so the\n"
            "        # initial staggered values are c2f followed by R_e, not the\n"
            "        # generic cell P_cR_c aggregation used by scalar leaves.\n"
            "        def project_initial_staggered_velocities(\n"
            "            local_state: FciDrbEBState,\n"
            "            cell_fields_owned: jax.Array,\n"
            "            map_fields_owned: jax.Array,\n"
            "            control_volume_fields_owned: jax.Array,\n"
            "            local_wall_projectors: UpwindEquilibriumWallProjectors | None,\n"
            "        ) -> FciDrbEBState:\n"
            "            model = build_local_model(\n"
            "                cell_fields_owned, map_fields_owned,\n"
            "                control_volume_fields_owned, local_wall_projectors,\n"
            "            )\n"
            "            face_bc = model._face_bcs(local_state)\n"
            "            context = model._stencil_builder_context()\n"
            "            def project(values, bc):\n"
            "                return model._owner_face_field(\n"
            "                    model._restrict_fine_face_field(\n"
            "                        model._center_owned_to_outgoing_face(values, bc, context)\n"
            "                    )\n"
            "                )\n"
            "            return local_state.replace(\n"
            "                Vi=project(local_state.Vi, face_bc.Vi),\n"
            "                Ve=project(local_state.Ve, face_bc.Ve),\n"
            "            )\n"
            "        project_initial_staggered = jax.jit(jax.shard_map(\n"
            "            project_initial_staggered_velocities, mesh=mesh,\n"
            "            in_specs=(state_spec, geometry_spec, geometry_spec, geometry_spec, wall_projector_specs),\n"
            "            out_specs=state_spec, check_vma=False,\n"
            "        ))\n"
            "        state = project_initial_staggered(\n"
            "            state, cell_fields, map_fields, control_volume_fields, wall_projectors\n"
            "        )\n"
            "        jax.block_until_ready(state)\n"
            "        if owner_host_geometry is not None:\n"
            "            _assert_owner_sparse(\n"
            "                state, owner_host_geometry, outgoing_face_topology_host\n"
            "            )\n"
            "    phi_start = time.perf_counter()\n",
        ),
        (
            '            "parallel_operator_scheme": str(args.parallel_operator_scheme),',
            '            "parallel_operator_scheme": str(args.parallel_operator_scheme),\n'
            '            "parallel_velocity_layout": os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT", "cell-centered"),\n'
            '            "field_locations": {"Vi": "fci-outgoing-face/source-edge" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else "cell-center", "Ve": "fci-outgoing-face/source-edge" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else "cell-center"},\n'
            '            "face_owner_layout": (None if staggered_face_provenance is None else staggered_face_provenance["face_basis_policy"]),\n'
            '            "outgoing_edge_mass_convention": "raw-fluid-cell-volume" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else None,\n'
            '            "cell_velocity_projection": "PcRc-after-face-to-center" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else None,\n'
            '            "stiff_momentum_force_projection": "source-cell-PcRc-before-face-Re" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else None,\n'
            '            "initial_velocity_projection": "center-to-outgoing-face-Re" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else None,\n'
            '            **({} if staggered_face_provenance is None else staggered_face_provenance),\n'
            '            "perpendicular_velocity_geometry": "face-to-center-perpendicular-center-to-face",',
        ),
        (
            "    base_output_payload.update(_snapshot_metric_payload(global_geometry))\n",
            "    if outgoing_face_topology_host is not None:\n"
            "        base_output_payload.update({\n"
            "            f\"face_topology_{name}\": np.asarray(value)\n"
            "            for name, value in outgoing_face_topology_host.items()\n"
            "        })\n"
            "    base_output_payload.update(_snapshot_metric_payload(global_geometry))\n",
        ),
        (
            "                payload[\"Ve_rhs_terms\"] = _materialize_owner_array(\n"
            "                    term_fields, owner_host_geometry\n"
            "                ).astype(np.float64)",
            "                payload[\"Ve_rhs_terms\"] = _materialize_face_owner_array(\n"
            "                    term_fields, outgoing_face_topology_host\n"
            "                ).astype(np.float64)",
        ),
        (
            "        run_metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),\n",
            "        run_metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),\n"
            "        **({\n"
            "            f\"face_topology_{name}\": np.asarray(value)\n"
            "            for name, value in outgoing_face_topology_host.items()\n"
            "        } if outgoing_face_topology_host is not None else {}),\n",
        ),
        (
            "            _assert_owner_sparse(state, owner_host_geometry)\n",
            "            _assert_owner_sparse(state, owner_host_geometry, outgoing_face_topology_host)\n",
        ),
        (
            "        _assert_owner_sparse(initial_state, owner_host_geometry)\n",
            "        _assert_owner_sparse(initial_state, owner_host_geometry, staggered_face_topology_host)\n",
        ),
        (
            "        owner_host_geometry=owner_host_geometry,\n"
            "    )\n",
            "        owner_host_geometry=owner_host_geometry,\n"
            "        outgoing_face_topology_host=staggered_face_topology_host,\n"
            "    )\n",
        ),
    )
    # Every stage/output diagnostic receives its locally built topology.  Keep
    # these exact call-site rewrites separate from the one-occurrence anchors:
    # the shared driver intentionally invokes this helper in several RK/IMEX
    # branches.
    for field_name in ("stage", "next_state", "result.state", "local_state"):
        source = source.replace(
            f"diagnostic_state({field_name}, model.control_volume_geometry)",
            f"diagnostic_state({field_name}, model.control_volume_geometry, model.outgoing_face_topology)",
        )
    source = source.replace(
        "diagnostic_state(\n                next_state, model.control_volume_geometry\n            )",
        "diagnostic_state(next_state, model.control_volume_geometry, model.outgoing_face_topology)",
    )
    source = source.replace(
        "diagnostic_state(\n                result.state, model.control_volume_geometry\n            )",
        "diagnostic_state(result.state, model.control_volume_geometry, model.outgoing_face_topology)",
    )
    source = source.replace(
        "diagnostic_state(\n                local_state, model.control_volume_geometry\n            )",
        "diagnostic_state(local_state, model.control_volume_geometry, model.outgoing_face_topology)",
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(
                "shared driver no longer matches the staggered launcher "
                f"transformation anchor: {old[:72]!r}"
            )
        source = source.replace(old, new, 1)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--parallel-velocity-layout",
        choices=("cell-centered", "fci-staggered"),
        default="cell-centered",
    )
    args, remaining = parser.parse_known_args(sys.argv[1:])

    def option_value(option: str) -> str | None:
        if option in remaining:
            index = remaining.index(option)
            return remaining[index + 1] if index + 1 < len(remaining) else None
        prefix = option + "="
        return next((value[len(prefix):] for value in remaining if value.startswith(prefix)), None)

    if args.parallel_velocity_layout == "fci-staggered":
        if option_value("--topology") != "toroidal":
            raise SystemExit(
                "fci-staggered requires --topology toroidal; the production "
                "Galerkin boundary-flux residual is not implemented for open "
                "or cut-wall parallel endpoints"
            )
        if option_value("--parallel-operator-scheme") != "fci":
            raise SystemExit("fci-staggered requires --parallel-operator-scheme fci")
        integrator = option_value("--time-integrator")
        if integrator is not None and integrator != "rk4":
            raise SystemExit("fci-staggered currently supports --time-integrator rk4 only")
        leg_scheme = option_value("--fci-parallel-leg-scheme")
        if leg_scheme is not None and leg_scheme != "centered":
            raise SystemExit("fci-staggered currently requires --fci-parallel-leg-scheme centered")
        if option_value("--restart-from") is not None:
            raise SystemExit(
                "fci-staggered restart is disabled until face-basis-aware restart "
                "fingerprints and Vi/Ve face reaggregation are implemented; run fresh"
            )
    os.environ["DRBX_PARALLEL_VELOCITY_LAYOUT"] = args.parallel_velocity_layout
    print(
        "[staggered launcher] parallel_velocity_layout="
        f"{args.parallel_velocity_layout}; Vi/Ve field locations="
        + ("fci-outgoing-face/source-edge" if args.parallel_velocity_layout == "fci-staggered" else "cell-center")
        + "; perpendicular Vi/Ve operators=face-to-center-perpendicular-center-to-face",
        flush=True,
    )
    source = _transform_shared_driver_source(SHARED_DRIVER.read_text())
    if str(WORKTREE / "src") not in sys.path:
        sys.path.insert(0, str(WORKTREE / "src"))
    sys.argv = [str(SHARED_DRIVER), *remaining]
    namespace = {"__name__": "__main__", "__file__": str(SHARED_DRIVER)}
    exec(compile(source, str(SHARED_DRIVER), "exec"), namespace)


if __name__ == "__main__":
    main()
