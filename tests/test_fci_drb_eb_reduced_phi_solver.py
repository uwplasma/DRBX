"""EB-RHS integration tests for the axis-core reduced phi solver."""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from drbx.geometry import build_local_curvature_coefficients  # noqa: E402
from drbx.native import (  # noqa: E402
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
from drbx.native.fci_drb_EB_rhs import FciDrbEBState  # noqa: E402
from drbx.native.fci_operators import AxisCoreReducedSpace3D  # noqa: E402

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

AXIS_CORE_AXES = (True, False, False)


def _inputs(shape=(8, 8, 4)):
    context = build_shifted_torus_eb_mms_context(shape)
    state = _mms_exact_state(context, 0.013)
    mesh = make_shard_mesh((1, 1, 1))
    local = build_local_fci_geometries(
        context.geometry,
        (1, 1, 1),
        halo_width=HALO_WIDTH,
        periodic_axes=PERIODIC_AXES,
        axis_regular_axes=AXIS_CORE_AXES,
    )
    sharding = NamedSharding(mesh, P("x", "y", "z"))
    fields = tuple(
        jax.device_put(jnp.asarray(getattr(state, name), dtype=jnp.float64), sharding)
        for name in ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")
    )
    state_owned = FciDrbEBState(*fields)
    cell_fields = jax.device_put(jnp.asarray(local.cell_fields), sharding)
    return context, mesh, local, P("x", "y", "z"), state_owned, cell_fields


def _rhs(context, local, geometry, **kwargs):
    domain = local.domain
    kwargs.setdefault("axis_regular_axes", AXIS_CORE_AXES)
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
            axis_regular_axes=AXIS_CORE_AXES,
        ),
        face_projectors=build_local_perp_laplacian_face_projectors(
            geometry,
            domain,
            axis_regular_axes=AXIS_CORE_AXES,
        ),
        gmres_config=SolvaxGmresConfig(
            tol=1.0e-8,
            atol=1.0e-8,
            maxiter=20,
            restart=10,
            acceptance_tol=1.0e-7,
            acceptance_atol=1.0e-7,
            project_mean_zero=False,
            regularization_epsilon=context.parameters.phi_inversion_regularization,
            preconditioner="line-u",
        ),
        face_bc_builder=_build_face_bcs,
        curvature_scheme="direct",
        curvature_inflow_closure="central",
        **kwargs,
    )


def test_default_phi_solver_space_is_full_grid():
    context, mesh, local, partition, _, cell_fields = _inputs()

    def kernel(local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        rhs = _rhs(context, local, geometry)
        return jnp.asarray(rhs.phi_solver_space == "full-grid")

    value = jax.jit(jax.shard_map(
        kernel, mesh=mesh, in_specs=(partition,), out_specs=P(), check_vma=False
    ))(cell_fields)
    assert bool(value)


def test_default_full_grid_dispatch_matches_explicit_full_grid():
    context, mesh, local, partition, state, cell_fields = _inputs()

    def kernel(
        density, phi, Te, Ti, Vi, Ve, vorticity, local_cell_fields
    ):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        default_rhs = _rhs(context, local, geometry)
        explicit_rhs = _rhs(context, local, geometry, phi_solver_space="full-grid")
        state_local = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        default_phi = default_rhs.reconstruct_phi(state_local)
        explicit_phi = explicit_rhs.reconstruct_phi(state_local)
        return jnp.max(jnp.abs(default_phi - explicit_phi))

    error = jax.jit(jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(partition,) * 8,
        out_specs=P(),
        check_vma=False,
    ))(
        state.density,
        state.phi,
        state.Te,
        state.Ti,
        state.Vi,
        state.Ve,
        state.vorticity,
        cell_fields,
    )
    assert float(error) == 0.0


def test_reduced_space_builds_from_the_rhs_face_gradient_context():
    context, mesh, local, partition, _, cell_fields = _inputs()

    def kernel(local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        rhs = _rhs(context, local, geometry, phi_solver_space="axis-core-reduced")
        space = rhs.build_axis_core_reduced_phi_space()
        return jnp.asarray((
            space.reconstruction.polynomial_degree,
            space.reconstruction.observation_ring_count,
            space.reconstruction.target_ring_count,
        ))

    values = jax.jit(jax.shard_map(
        kernel, mesh=mesh, in_specs=(partition,), out_specs=P(), check_vma=False
    ))(cell_fields)
    assert tuple(map(int, values)) == (3, 6, 3)


def test_reduced_phi_reconstruction_returns_compatible_diagnostics_and_is_jittable():
    context, mesh, local, partition, state, cell_fields = _inputs()

    def kernel(
        density, phi, Te, Ti, Vi, Ve, vorticity, local_cell_fields
    ):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        rhs = _rhs(context, local, geometry, phi_solver_space="axis-core-reduced")
        state_local = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        phi_out, info = rhs.reconstruct_phi(state_local, return_diagnostics=True)
        return phi_out, info.converged, info.final_residual_rel_l2

    compiled = jax.jit(jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(partition,) * 8,
        out_specs=(partition, P(), P()),
        check_vma=False,
    ))
    phi, converged, relative = compiled(
        state.density,
        state.phi,
        state.Te,
        state.Ti,
        state.Vi,
        state.Ve,
        state.vorticity,
        cell_fields,
    )
    phi, converged, relative = jax.block_until_ready((phi, converged, relative))
    assert jnp.all(jnp.isfinite(phi))
    assert bool(converged)
    assert jnp.isfinite(relative)


def test_reduced_space_rejects_non_axis_topology_and_unsupported_preconditioner():
    context, mesh, local, partition, _, cell_fields = _inputs()

    def invalid_topology(local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        _rhs(
            context,
            local,
            geometry,
            phi_solver_space="axis-core-reduced",
            axis_regular_axes=(False, False, False),
        )
        return jnp.asarray(0)

    with pytest.raises(ValueError, match="lower radial axis regularity"):
        jax.jit(jax.shard_map(
            invalid_topology,
            mesh=mesh,
            in_specs=(partition,),
            out_specs=P(),
            check_vma=False,
        ))(cell_fields)


def test_reduced_payload_rejects_context_degree_mismatch():
    context, mesh, local, partition, _, cell_fields = _inputs()

    def mismatched_payload(local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        matching_rhs = _rhs(
            context, local, geometry, phi_solver_space="axis-core-reduced"
        )
        payload = matching_rhs.build_axis_core_reduced_phi_space()
        mismatched_rhs = _rhs(
            context,
            local,
            geometry,
            phi_solver_space="axis-core-reduced",
            axis_core_gradient_polynomial_degree=2,
            axis_core_reduced_space=payload,
        )
        mismatched_rhs.build_axis_core_reduced_phi_space()
        return jnp.asarray(0)

    with pytest.raises(ValueError, match="degree/rings do not match"):
        jax.jit(jax.shard_map(
            mismatched_payload,
            mesh=mesh,
            in_specs=(partition,),
            out_specs=P(),
            check_vma=False,
        ))(cell_fields)



def test_reduced_space_rejects_unsupported_preconditioner():
    context, mesh, local, partition, _, cell_fields = _inputs()

    def invalid_preconditioner(local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        bad_config = SolvaxGmresConfig(preconditioner="jacobi")
        LocalFciDrbEBRhs(
            geometry=geometry,
            domain=local.domain,
            halo_exchange=HaloExchange3D(),
            topology_filler=TopologyHaloFiller3D(
                rules=(LocalPeriodicTopologyRule3D(),)
            ),
            physical_ghost_filler=_ghost_filler(HALO_WIDTH),
            parameters=context.parameters,
            curvature_coefficients_owned=None,
            face_projectors=(None, None, None),
            gmres_config=bad_config,
            face_bc_builder=_build_face_bcs,
            axis_regular_axes=AXIS_CORE_AXES,
            curvature_scheme="disabled",
            phi_solver_space="axis-core-reduced",
        )
        return jnp.asarray(0)

    with pytest.raises(ValueError, match="only GMRES preconditioner='none' or 'line-u'"):
        jax.jit(jax.shard_map(
            invalid_preconditioner,
            mesh=mesh,
            in_specs=(partition,),
            out_specs=P(),
            check_vma=False,
        ))(cell_fields)
