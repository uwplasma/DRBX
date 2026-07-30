"""Sharded, regular-grid shifted-torus full-EB MMS regression.

This test deliberately keeps ``FciGeometry3D`` as host-side staging, then
executes the discrete RHS exclusively through ``shard_map`` with
``LocalFciDrbEBRhs``.  Analytic MMS data comes from the independent helper;
the legacy global EB test is not imported.

The test evaluates the spatial residual at one fixed MMS time.  Supplying the
analytic potential avoids making the convergence gate depend on the separate
distributed elliptic-solver tolerance; the six evolved equations still pass
through the complete local EB RHS and physical halo closure.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry import (
    build_local_curvature_coefficients,
)
from drbx.native import (
    FciDrbEBRhsParameters,
    FciDrbEBState,
    GhostFillWeights1D,
    HaloExchange3D,
    LocalBoundaryFaceBC3D,
    LocalFciDrbEBFaceBCBundle,
    LocalFciDrbEBRhs,
    PhysicalGhostCellFiller3D,
    SolvaxGmresConfig,
    TopologyHaloFiller3D,
    LocalPeriodicTopologyRule3D,
    assemble_local_fci_geometry,
    build_local_fci_geometries,
    build_local_perp_laplacian_face_projectors,
    make_shard_mesh,
)
from drbx.native.fci_boundaries import BC_DIRICHLET
from jax.sharding import NamedSharding, PartitionSpec as P

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from shifted_torus_eb_mms_data import (  # noqa: E402
    _analytic_eb_rhs_from_data,
    _data_at,
    _mms_exact_state,
    build_shifted_torus_eb_mms_context,
)


PERIODIC_AXES = (False, True, True)
AXIS_REGULAR_AXES = (False, False, False)
HALO_WIDTH = 2
MMS_TIME = 0.013


def _ghost_filler(halo_width: int) -> PhysicalGhostCellFiller3D:
    """Use the regular physical Dirichlet ghost reconstruction."""

    dirichlet = GhostFillWeights1D(
        owned_weights=-jnp.ones((halo_width, 1), dtype=jnp.float64),
        bc_weights=2.0 * jnp.ones((halo_width,), dtype=jnp.float64),
    )
    neutral = GhostFillWeights1D(
        owned_weights=jnp.ones((halo_width, 1), dtype=jnp.float64),
        bc_weights=jnp.zeros((halo_width,), dtype=jnp.float64),
    )
    return PhysicalGhostCellFiller3D(
        dirichlet=(dirichlet, dirichlet, dirichlet),
        neumann_lower=(neutral, neutral, neutral),
        neumann_upper=(neutral, neutral, neutral),
    )


def _radial_dirichlet_bc(
    geometry,
    domain,
    lower_value: float,
    upper_value: float,
) -> LocalBoundaryFaceBC3D:
    """Construct fixed-time MMS data on the physical radial faces.

    The imported MMS envelope is ``sin(pi*s)^6``.  Thus all fields have
    constant value and zero first derivative on the two radial walls: n, Te,
    and Ti are one, while phi, velocities, and vorticity are zero.
    """

    bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    lower_active = domain.runtime_has_physical_lower(0)
    upper_active = domain.runtime_has_physical_upper(0)
    kind_x = bc.kind_x
    value_x = bc.value_x
    mask_x = bc.mask_x
    kind_x = kind_x.at[0].set(
        jnp.where(lower_active, BC_DIRICHLET, kind_x[0])
    )
    kind_x = kind_x.at[-1].set(
        jnp.where(upper_active, BC_DIRICHLET, kind_x[-1])
    )
    value_x = value_x.at[0].set(jnp.where(lower_active, lower_value, value_x[0]))
    value_x = value_x.at[-1].set(jnp.where(upper_active, upper_value, value_x[-1]))
    mask_x = mask_x.at[0].set(jnp.where(lower_active, True, mask_x[0]))
    mask_x = mask_x.at[-1].set(jnp.where(upper_active, True, mask_x[-1]))
    return replace(bc, kind_x=kind_x, value_x=value_x, mask_x=mask_x)


def _build_face_bcs(state, geometry, domain, parameters) -> LocalFciDrbEBFaceBCBundle:
    del state, parameters
    one = _radial_dirichlet_bc(geometry, domain, 1.0, 1.0)
    zero = _radial_dirichlet_bc(geometry, domain, 0.0, 0.0)
    return LocalFciDrbEBFaceBCBundle(
        density=one,
        phi=zero,
        Te=one,
        Ti=one,
        Vi=zero,
        Ve=zero,
        vorticity=zero,
    )


def _sharded_rhs(
    geometry,
    state: FciDrbEBState,
    parameters: FciDrbEBRhsParameters,
    *,
    shard_counts: tuple[int, int, int],
    halo_width: int = HALO_WIDTH,
) -> FciDrbEBState:
    """Evaluate one full local EB RHS through a compiled ``shard_map``."""

    mesh = make_shard_mesh(shard_counts)
    sharded_geometry = build_local_fci_geometries(
        geometry,
        shard_counts,
        halo_width=halo_width,
        periodic_axes=PERIODIC_AXES,
    )
    partition = P("x", "y", "z")
    state_sharding = NamedSharding(mesh, partition)
    cell_fields_sharded = jax.device_put(sharded_geometry.cell_fields, state_sharding)

    physical_ghost_filler = _ghost_filler(halo_width)
    state_fields = tuple(
        jax.device_put(jnp.asarray(getattr(state, name), dtype=jnp.float64), state_sharding)
        for name in ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cell_fields):
        local_geometry = assemble_local_fci_geometry(sharded_geometry, cell_fields)
        domain = sharded_geometry.domain
        local_curvature = build_local_curvature_coefficients(
            local_geometry,
            domain,
            periodic_axes=PERIODIC_AXES,
            axis_regular_axes=AXIS_REGULAR_AXES,
        )
        local_face_projectors = build_local_perp_laplacian_face_projectors(
            local_geometry,
            domain,
            axis_regular_axes=AXIS_REGULAR_AXES,
        )
        rhs = LocalFciDrbEBRhs(
            geometry=local_geometry,
            domain=domain,
            halo_exchange=HaloExchange3D(),
            topology_filler=TopologyHaloFiller3D(
                rules=(LocalPeriodicTopologyRule3D(),)
            ),
            physical_ghost_filler=physical_ghost_filler,
            parameters=parameters,
            curvature_coefficients_owned=local_curvature,
            face_projectors=local_face_projectors,
            gmres_config=SolvaxGmresConfig(
                tol=1.0e-8,
                atol=1.0e-8,
                maxiter=20,
                restart=20,
                acceptance_tol=1.0e-6,
                acceptance_atol=1.0e-6,
                project_mean_zero=False,
                regularization_epsilon=parameters.phi_inversion_regularization,
            ),
            face_bc_builder=_build_face_bcs,
            curvature_scheme="direct",
            curvature_inflow_closure="central",
        )
        current = FciDrbEBState(
            density=density,
            phi=phi,
            Te=Te,
            Ti=Ti,
            Vi=Vi,
            Ve=Ve,
            vorticity=vorticity,
        )
        result = rhs.evaluate_stage(current, phi_owned=phi)
        return (
            result.density,
            result.phi,
            result.Te,
            result.Ti,
            result.Vi,
            result.Ve,
            result.vorticity,
        )

    sharded_kernel = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 8,
            out_specs=(partition,) * 7,
            check_vma=False,
        )
    )
    outputs = sharded_kernel(*state_fields, cell_fields_sharded)
    return FciDrbEBState(*[jax.block_until_ready(value) for value in outputs])


def _sharded_rk4_step(
    geometry,
    state: FciDrbEBState,
    parameters: FciDrbEBRhsParameters,
    *,
    timestep: float,
    shard_counts: tuple[int, int, int],
    halo_width: int = HALO_WIDTH,
) -> tuple[FciDrbEBState, bool]:
    """Advance all seven EB fields through four distributed phi solves."""

    mesh = make_shard_mesh(shard_counts)
    local = build_local_fci_geometries(geometry, shard_counts, halo_width=halo_width, periodic_axes=PERIODIC_AXES)
    partition = P("x", "y", "z")
    sharding = NamedSharding(mesh, partition)
    fields = lambda s: tuple(jax.device_put(jnp.asarray(getattr(s, n), dtype=jnp.float64), sharding) for n in ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity"))
    cell_fields = jax.device_put(local.cell_fields, sharding)
    ghost = _ghost_filler(halo_width)

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cell_fields):
        geom = assemble_local_fci_geometry(local, cell_fields)
        domain = local.domain
        curvature = build_local_curvature_coefficients(geom, domain, periodic_axes=PERIODIC_AXES, axis_regular_axes=AXIS_REGULAR_AXES)
        projectors = build_local_perp_laplacian_face_projectors(geom, domain, axis_regular_axes=AXIS_REGULAR_AXES)
        rhs = LocalFciDrbEBRhs(
            geometry=geom, domain=domain, halo_exchange=HaloExchange3D(),
            topology_filler=TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),)),
            physical_ghost_filler=ghost, parameters=parameters,
            curvature_coefficients_owned=curvature, face_projectors=projectors,
            gmres_config=SolvaxGmresConfig(tol=1.0e-8, atol=1.0e-8, maxiter=20, restart=20, acceptance_tol=1.0e-6, acceptance_atol=1.0e-6, project_mean_zero=False, regularization_epsilon=parameters.phi_inversion_regularization),
            face_bc_builder=_build_face_bcs, curvature_scheme="direct", curvature_inflow_closure="central",
        )
        current = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        accepted = True
        ks = []
        for stage_index in range(4):
            solved, info = rhs.reconstruct_phi(current, return_diagnostics=True)
            accepted = accepted & jnp.all(info.converged) & jnp.all(~info.failed) & jnp.all(info.phi_is_finite)
            stage = current.replace(phi=solved)
            k = rhs.evaluate_stage(stage, phi_owned=solved)
            ks.append(k)
            if stage_index < 3:
                scale = 0.5 if stage_index < 2 else 1.0
                current = FciDrbEBState(
                    density=density + scale * timestep * k.density,
                    phi=solved,
                    Te=Te + scale * timestep * k.Te,
                    Ti=Ti + scale * timestep * k.Ti,
                    Vi=Vi + scale * timestep * k.Vi,
                    Ve=Ve + scale * timestep * k.Ve,
                    vorticity=vorticity + scale * timestep * k.vorticity,
                )
        next_state = FciDrbEBState(
            density=density + timestep * (ks[0].density + 2 * ks[1].density + 2 * ks[2].density + ks[3].density) / 6,
            phi=ks[3].phi,
            Te=Te + timestep * (ks[0].Te + 2 * ks[1].Te + 2 * ks[2].Te + ks[3].Te) / 6,
            Ti=Ti + timestep * (ks[0].Ti + 2 * ks[1].Ti + 2 * ks[2].Ti + ks[3].Ti) / 6,
            Vi=Vi + timestep * (ks[0].Vi + 2 * ks[1].Vi + 2 * ks[2].Vi + ks[3].Vi) / 6,
            Ve=Ve + timestep * (ks[0].Ve + 2 * ks[1].Ve + 2 * ks[2].Ve + ks[3].Ve) / 6,
            vorticity=vorticity + timestep * (ks[0].vorticity + 2 * ks[1].vorticity + 2 * ks[2].vorticity + ks[3].vorticity) / 6,
        )
        return (
            next_state.density, next_state.phi, next_state.Te, next_state.Ti,
            next_state.Vi, next_state.Ve, next_state.vorticity, accepted,
        )

    mapped = jax.jit(jax.shard_map(kernel, mesh=mesh, in_specs=(partition,) * 8, out_specs=(partition,) * 7 + (P(),), check_vma=False))
    outputs = mapped(*fields(state), cell_fields)
    return FciDrbEBState(*[jax.block_until_ready(v) for v in outputs[:7]]), bool(np.asarray(jax.block_until_ready(outputs[7])))


def _state_rms_error(actual: FciDrbEBState, expected: FciDrbEBState) -> float:
    """Measure the six evolved equations away from physical radial faces."""

    interior = (slice(2, -2), slice(None), slice(None))
    fields = ("density", "Te", "Ti", "Vi", "Ve", "vorticity")
    values = []
    for name in fields:
        error = np.asarray(getattr(actual, name)[interior] - getattr(expected, name)[interior])
        values.append(np.mean(error * error))
    return float(np.sqrt(np.mean(values)))


def _run_residual(shape: tuple[int, int, int], shard_counts=(1, 1, 1)) -> float:
    # Build only the host staging geometry and analytic coefficient data.
    context = build_shifted_torus_eb_mms_context(shape)
    exact_state = _mms_exact_state(context, MMS_TIME)
    analytic_data = _data_at(context, MMS_TIME)
    expected_rhs = _analytic_eb_rhs_from_data(analytic_data, context)
    computed_rhs = _sharded_rhs(
        context.geometry,
        exact_state,
        context.parameters,
        shard_counts=tuple(shard_counts),
    )
    return _state_rms_error(computed_rhs, expected_rhs)


def test_sharded_shifted_torus_eb_mms_spatial_convergence() -> None:
    """The complete local EB RHS error decreases under spatial refinement."""

    coarse = _run_residual((8, 12, 8))
    fine = _run_residual((12, 18, 12))
    order = float(np.log(coarse / fine) / np.log(12.0 / 8.0))
    print(
        f"sharded shifted-torus EB MMS: coarse={coarse:.6e}, "
        f"fine={fine:.6e}, order={order:.3f}"
    )
    assert np.isfinite(coarse) and np.isfinite(fine)
    assert order > 1.5, (coarse, fine, order)


@pytest.mark.skipif(len(jax.devices()) < 2, reason="requires at least two JAX devices")
def test_sharded_shifted_torus_eb_mms_matches_single_shard() -> None:
    """Toroidal decomposition preserves the local full-EB residual."""

    shape = (8, 12, 8)
    single = _run_residual(shape, shard_counts=(1, 1, 1))
    sharded = _run_residual(shape, shard_counts=(1, 1, 2))
    print(f"sharded shifted-torus EB consistency: single={single:.6e}, sharded={sharded:.6e}")
    assert np.isfinite(single) and np.isfinite(sharded)
    np.testing.assert_allclose(sharded, single, rtol=2.0e-5, atol=2.0e-9)


def test_sharded_shifted_torus_eb_full_rk4_equilibrium() -> None:
    """A full sharded RK4 step keeps the exact equilibrium finite and accepted."""

    shape = (4, 6, 4)
    context = build_shifted_torus_eb_mms_context(shape)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    state = FciDrbEBState(jnp.ones(shape), zeros, jnp.ones(shape), jnp.ones(shape), zeros, zeros, zeros)
    advanced, accepted = _sharded_rk4_step(
        context.geometry,
        state,
        context.parameters,
        timestep=1.0e-4,
        shard_counts=(1, 1, 1),
    )
    assert accepted
    for name in ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity"):
        values = np.asarray(getattr(advanced, name))
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(values, np.asarray(getattr(state, name)), atol=1.0e-10)
