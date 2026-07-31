"""End-to-end sharded regression for the EB ARK additive split."""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

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

from shifted_torus_eb_mms_data import _mms_exact_state, build_shifted_torus_eb_mms_context  # noqa: E402
from test_mms_shifted_torus_EB_sharded import (  # noqa: E402
    AXIS_REGULAR_AXES,
    HALO_WIDTH,
    PERIODIC_AXES,
    _build_face_bcs,
    _ghost_filler,
)


def test_sharded_imex_split_sums_to_the_production_rhs() -> None:
    """The actual local/sharded split is exact for a supplied stage phi."""

    shape = (4, 6, 4)
    context = build_shifted_torus_eb_mms_context(shape)
    state = _mms_exact_state(context, 0.013)
    mesh = make_shard_mesh((1, 1, 1))
    local = build_local_fci_geometries(
        context.geometry, (1, 1, 1), halo_width=HALO_WIDTH, periodic_axes=PERIODIC_AXES
    )
    partition = P("x", "y", "z")
    sharding = NamedSharding(mesh, partition)
    fields = tuple(
        jax.device_put(jnp.asarray(getattr(state, name), dtype=jnp.float64), sharding)
        for name in ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")
    )
    cell_fields = jax.device_put(local.cell_fields, sharding)
    ghost = _ghost_filler(HALO_WIDTH)

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        domain = local.domain
        rhs = LocalFciDrbEBRhs(
            geometry=geometry,
            domain=domain,
            halo_exchange=HaloExchange3D(),
            topology_filler=TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),)),
            physical_ghost_filler=ghost,
            parameters=context.parameters,
            curvature_coefficients_owned=build_local_curvature_coefficients(
                geometry, domain, periodic_axes=PERIODIC_AXES, axis_regular_axes=AXIS_REGULAR_AXES
            ),
            face_projectors=build_local_perp_laplacian_face_projectors(
                geometry, domain, axis_regular_axes=AXIS_REGULAR_AXES
            ),
            gmres_config=SolvaxGmresConfig(
                tol=1.0e-8, atol=1.0e-8, maxiter=20, restart=20,
                acceptance_tol=1.0e-6, acceptance_atol=1.0e-6,
                project_mean_zero=False,
                regularization_epsilon=context.parameters.phi_inversion_regularization,
            ),
            face_bc_builder=_build_face_bcs,
            curvature_scheme="direct",
            curvature_inflow_closure="central",
        )
        stage = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        full = rhs.evaluate_stage(stage, phi_owned=phi)
        implicit = rhs.evaluate_implicit_rhs(stage, phi_owned=phi)
        explicit = rhs.evaluate_explicit_rhs(stage, phi_owned=phi)
        errors = jnp.stack((
            jnp.max(jnp.abs(full.density - explicit.density - implicit.density)),
            jnp.max(jnp.abs(full.Te - explicit.Te - implicit.Te)),
            jnp.max(jnp.abs(full.Ti - explicit.Ti)),
            jnp.max(jnp.abs(full.Vi - explicit.Vi)),
            jnp.max(jnp.abs(full.Ve - explicit.Ve - implicit.Ve)),
            jnp.max(jnp.abs(full.vorticity - explicit.vorticity - implicit.vorticity)),
        ))
        return errors

    compiled = jax.jit(jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(partition,) * 8,
        out_specs=P(),
        check_vma=False,
    ))
    errors = np.asarray(jax.block_until_ready(compiled(*fields, cell_fields)))
    np.testing.assert_allclose(errors, 0.0, atol=2.0e-11, rtol=0.0)
