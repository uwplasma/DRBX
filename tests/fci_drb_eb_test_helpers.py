"""Shared local geometry fixtures for EB operator tests."""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp

from drbx.geometry import build_local_curvature_coefficients
from drbx.native import (
    HaloExchange3D,
    LocalFciDrbEBRhs,
    LocalPeriodicTopologyRule3D,
    SolvaxGmresConfig,
    TopologyHaloFiller3D,
    assemble_local_fci_geometry,
    build_local_fci_geometries,
    build_local_perp_laplacian_face_projectors,
    make_shard_mesh,
)
from drbx.native.fci_drb_EB_rhs import FciDrbEBState
from jax.sharding import NamedSharding, PartitionSpec as P

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from shifted_torus_eb_mms_data import (  # noqa: E402
    _mms_exact_state,
    build_shifted_torus_eb_mms_context,
)
from test_mms_shifted_torus_EB_sharded import (  # noqa: E402
    AXIS_REGULAR_AXES,
    HALO_WIDTH,
    PERIODIC_AXES,
    _build_face_bcs,
    _ghost_filler,
)


def _context_and_sharded_inputs():
    """Return a small shifted-torus local geometry and smooth state."""

    shape = (4, 6, 4)
    context = build_shifted_torus_eb_mms_context(shape)
    state = _mms_exact_state(context, 0.013)
    mesh = make_shard_mesh((1, 1, 1))
    local = build_local_fci_geometries(
        context.geometry,
        (1, 1, 1),
        halo_width=HALO_WIDTH,
        periodic_axes=PERIODIC_AXES,
    )
    partition = P("x", "y", "z")
    sharding = NamedSharding(mesh, partition)
    fields = tuple(
        jax.device_put(jnp.asarray(getattr(state, name), dtype=jnp.float64), sharding)
        for name in ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")
    )
    return context, mesh, local, partition, fields, jax.device_put(
        local.cell_fields, sharding
    )


def _build_rhs(context, local, geometry):
    domain = local.domain
    return LocalFciDrbEBRhs(
        geometry=geometry,
        domain=domain,
        halo_exchange=HaloExchange3D(),
        topology_filler=TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),)),
        physical_ghost_filler=_ghost_filler(HALO_WIDTH),
        parameters=context.parameters,
        curvature_coefficients_owned=build_local_curvature_coefficients(
            geometry,
            domain,
            periodic_axes=PERIODIC_AXES,
            axis_regular_axes=AXIS_REGULAR_AXES,
        ),
        face_projectors=build_local_perp_laplacian_face_projectors(
            geometry, domain, axis_regular_axes=AXIS_REGULAR_AXES
        ),
        gmres_config=SolvaxGmresConfig(
            tol=1.0e-10,
            atol=1.0e-10,
            maxiter=80,
            restart=20,
            acceptance_tol=1.0e-8,
            acceptance_atol=1.0e-8,
            project_mean_zero=False,
            regularization_epsilon=context.parameters.phi_inversion_regularization,
        ),
        face_bc_builder=_build_face_bcs,
        curvature_scheme="direct",
        curvature_inflow_closure="central",
    )


__all__ = ["_build_rhs", "_context_and_sharded_inputs"]
