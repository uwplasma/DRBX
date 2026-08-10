"""Integration-grade checks for the local EB IMEX stage interface."""

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
from drbx.native.fci_drb_EB_rhs import (
    FciDrbEBImplicitState,
    FciDrbEBState,
    eb_state_with_implicit_state,
    implicit_state_from_eb_state,
)
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


def _context_and_sharded_inputs():
    """Return a small, real shifted-torus local geometry and smooth state."""

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
    return context, mesh, local, partition, fields, jax.device_put(local.cell_fields, sharding)


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
            geometry, domain, periodic_axes=PERIODIC_AXES, axis_regular_axes=AXIS_REGULAR_AXES
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


def test_production_phi_reconstruction_satisfies_imex_polarization_residual() -> None:
    """The algebraic IMEX row has exactly the production phi-solver sign."""

    context, mesh, local, partition, fields, cell_fields = _context_and_sharded_inputs()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        rhs = _build_rhs(context, local, geometry)
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        solved_phi, info = rhs.reconstruct_phi(state, return_diagnostics=True)
        algebraic = rhs.polarization_residual(state, phi_owned=solved_phi)
        return jnp.stack((
            jnp.max(jnp.abs(algebraic)),
            info.final_residual_l2,
            info.final_residual_rel_l2,
            jnp.asarray(info.failed, dtype=jnp.float64),
        ))

    compiled = jax.jit(jax.shard_map(
        kernel, mesh=mesh, in_specs=(partition,) * 8, out_specs=P(), check_vma=False
    ))
    residual_max, final_l2, final_rel_l2, failed = np.asarray(
        jax.block_until_ready(compiled(*fields, cell_fields))
    )
    assert failed == 0.0
    assert final_rel_l2 <= 1.0e-8
    # The max norm is intentionally looser than the configured global L2
    # acceptance criterion, but still verifies the residual's sign and scale.
    assert residual_max <= 2.0e-7, (residual_max, final_l2, final_rel_l2)


def test_implicit_stage_residual_jvp_matches_centered_difference() -> None:
    """Matrix-free Newton JVPs are correct on the real local operator stack."""

    context, mesh, local, partition, fields, cell_fields = _context_and_sharded_inputs()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        rhs = _build_rhs(context, local, geometry)
        known = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        z = implicit_state_from_eb_state(known)
        predictor = z.replace(
            density=z.density * 0.999,
            Te=z.Te * 1.001,
            Ti=z.Ti * 0.999,
            Ve=z.Ve + 0.003,
            vorticity=z.vorticity * 0.998,
            phi=z.phi * 1.002,
        )
        direction = FciDrbEBImplicitState(
            density=0.03 * jnp.sin(density),
            phi=0.01 * jnp.cos(phi),
            Te=0.02 * jnp.sin(Te),
            Ti=0.017 * jnp.cos(Ti + 0.2),
            Ve=0.02 * jnp.cos(Ve),
            vorticity=0.01 * jnp.sin(vorticity + 0.4),
        )
        residual = lambda value: rhs.implicit_stage_residual(
            value, predictor, known, dt_gamma=1.0e-4
        )
        _, jvp = jax.jvp(residual, (z,), (direction,))
        epsilon = jnp.asarray(2.0e-6, dtype=jnp.float64)
        plus = residual(z.axpy(direction, scale=epsilon))
        minus = residual(z.axpy(direction, scale=-epsilon))
        finite_difference = plus.axpy(minus, scale=-1.0).map_fields(
            lambda value: value / (2.0 * epsilon)
        )
        error = jnp.concatenate(tuple(
            jnp.ravel(jnp.abs(getattr(jvp, name) - getattr(finite_difference, name)))
            for name in jvp.field_names()
        ))
        reference = jnp.concatenate(tuple(
            jnp.ravel(jnp.abs(getattr(jvp, name)))
            for name in jvp.field_names()
        ))
        return jnp.stack((jnp.max(error), jnp.maximum(jnp.max(reference), 1.0)))

    compiled = jax.jit(jax.shard_map(
        kernel, mesh=mesh, in_specs=(partition,) * 8, out_specs=P(), check_vma=False
    ))
    error, reference = np.asarray(jax.block_until_ready(compiled(*fields, cell_fields)))
    assert error / reference < 2.0e-5, (error, reference)


def test_implicit_state_contains_ti_but_keeps_vi_out_of_newton_unknown() -> None:
    """Ti is implicit while Vi remains the sole stage-known ion field."""

    full = FciDrbEBState(*(
        jnp.full((2, 2, 2), float(index + 1), dtype=jnp.float64)
        for index in range(7)
    ))
    z = implicit_state_from_eb_state(full)
    changed = z.replace(density=z.density + 2.0, Te=z.Te - 0.25, Ti=z.Ti + 0.5, Ve=z.Ve + 3.0)
    merged = eb_state_with_implicit_state(full, changed)
    assert z.field_names() == ("density", "phi", "Te", "Ti", "Ve", "vorticity")
    assert len(jax.tree_util.tree_leaves(z)) == 6
    np.testing.assert_array_equal(merged.Ti, changed.Ti)
    np.testing.assert_array_equal(merged.Vi, full.Vi)
    np.testing.assert_array_equal(merged.density, changed.density)
    np.testing.assert_array_equal(merged.Te, changed.Te)
    np.testing.assert_array_equal(merged.Ve, changed.Ve)
