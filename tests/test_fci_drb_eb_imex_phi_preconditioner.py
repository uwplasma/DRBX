"""Focused coverage for the EB IMEX phi-line preconditioner.

The preconditioner is intentionally tested as a *right* preconditioner for
the driver-scaled algebraic row.  Consequently its phi input is
``dt_gamma * R_phi`` and its phi output approximates the corresponding
potential increment.  The other five IMEX leaves, including the ordinary
differential ``Ti`` leaf, are identity mapped.
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P
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
    build_eb_imex_phi_line_u_preconditioner,
)

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
    """Return a small mapped geometry, smooth fields, and sharding metadata."""

    shape = (6, 6, 4)
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
            geometry,
            domain,
            periodic_axes=PERIODIC_AXES,
            axis_regular_axes=AXIS_REGULAR_AXES,
        ),
        face_projectors=build_local_perp_laplacian_face_projectors(
            geometry,
            domain,
            axis_regular_axes=AXIS_REGULAR_AXES,
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
            preconditioner="line-u",
        ),
        face_bc_builder=_build_face_bcs,
        curvature_scheme="direct",
        curvature_inflow_closure="central",
    )


def test_phi_line_u_preconditioner_reduces_scaled_polarization_residual() -> None:
    """The phi block lowers the residual of the driver-scaled algebraic row."""

    context, mesh, local, partition, fields, cell_fields = _context_and_sharded_inputs()
    dt_gamma = 0.017

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        rhs = _build_rhs(context, local, geometry)
        # Constant Ti has zero polarization contribution with the MMS wall
        # condition, so this produces a pure homogeneous phi operator sample.
        base = FciDrbEBState(
            density,
            jnp.zeros_like(phi),
            Te,
            jnp.ones_like(Ti),
            Vi,
            Ve,
            jnp.zeros_like(vorticity),
        )
        # The MMS phi satisfies its radial Dirichlet trace and contains u
        # variation, making it a meaningful line-u preconditioner target.
        target = base.replace(phi=phi)
        scaled_rhs_phi = dt_gamma * rhs.polarization_residual(
            target,
            phi_owned=phi,
        )
        residual = FciDrbEBImplicitState(
            density=0.25 + density,
            phi=scaled_rhs_phi,
            Te=0.5 + Te,
            Ti=0.75 + Ti,
            Ve=-0.75 + Ve,
            vorticity=1.25 + vorticity,
        )
        preconditioner = build_eb_imex_phi_line_u_preconditioner(rhs, dt_gamma)
        corrected = preconditioner(residual)
        corrected_state = base.replace(phi=corrected.phi)
        scaled_defect = (
            dt_gamma
            * rhs.polarization_residual(
                corrected_state,
                phi_owned=corrected.phi,
            )
            - scaled_rhs_phi
        )
        active = geometry.active_cell_mask_owned
        initial_l2 = jnp.sum(jnp.where(active, scaled_rhs_phi * scaled_rhs_phi, 0.0))
        defect_l2 = jnp.sum(jnp.where(active, scaled_defect * scaled_defect, 0.0))
        unchanged = jnp.stack((
            jnp.max(jnp.abs(corrected.density - residual.density)),
            jnp.max(jnp.abs(corrected.Te - residual.Te)),
            jnp.max(jnp.abs(corrected.Ti - residual.Ti)),
            jnp.max(jnp.abs(corrected.Ve - residual.Ve)),
            jnp.max(jnp.abs(corrected.vorticity - residual.vorticity)),
        ))
        return lax.psum(jnp.stack((initial_l2, defect_l2)), ("x", "y", "z")), unchanged

    compiled = jax.jit(jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(partition,) * 8,
        out_specs=(P(), P()),
        check_vma=False,
    ))
    norms, unchanged = jax.block_until_ready(compiled(*fields, cell_fields))
    initial_l2, defect_l2 = np.asarray(norms)
    unchanged = np.asarray(unchanged)
    assert initial_l2 > 0.0
    # A line-u solve is deliberately approximate because mixed and v bands
    # are dropped, but it should still provide a meaningful residual decrease.
    assert defect_l2 / initial_l2 < 0.8, (initial_l2, defect_l2)
    np.testing.assert_array_equal(unchanged, np.zeros(5))


def test_phi_line_u_preconditioner_has_inverse_dt_gamma_scaling() -> None:
    """Changing the DIRK scale rescales only the phi correction by 1/dt."""

    context, mesh, local, partition, fields, cell_fields = _context_and_sharded_inputs()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        rhs = _build_rhs(context, local, geometry)
        # A deterministic nonzero residual with no relation to a particular
        # stage is enough because the preconditioner is a linear right map.
        value = FciDrbEBImplicitState(
            density=density,
            phi=0.2 + jnp.sin(1.7 * phi),
            Te=Te,
            Ti=Ti,
            Ve=Ve,
            vorticity=vorticity,
        )
        coarse = build_eb_imex_phi_line_u_preconditioner(rhs, 0.04)(value)
        fine = build_eb_imex_phi_line_u_preconditioner(rhs, 0.02)(value)
        active = geometry.active_cell_mask_owned
        scale_error = jnp.max(jnp.where(
            active,
            jnp.abs(fine.phi - 2.0 * coarse.phi),
            0.0,
        ))
        reference = jnp.maximum(
            jnp.max(jnp.where(active, jnp.abs(fine.phi), 0.0)),
            1.0,
        )
        unchanged = jnp.stack((
            jnp.max(jnp.abs(fine.density - value.density)),
            jnp.max(jnp.abs(fine.Te - value.Te)),
            jnp.max(jnp.abs(fine.Ti - value.Ti)),
            jnp.max(jnp.abs(fine.Ve - value.Ve)),
            jnp.max(jnp.abs(fine.vorticity - value.vorticity)),
        ))
        return jnp.stack((scale_error, reference)), unchanged

    compiled = jax.jit(jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(partition,) * 8,
        out_specs=(P(), P()),
        check_vma=False,
    ))
    scales, unchanged = jax.block_until_ready(compiled(*fields, cell_fields))
    scale_error, reference = np.asarray(scales)
    unchanged = np.asarray(unchanged)
    assert scale_error / reference < 2.0e-13, (scale_error, reference)
    np.testing.assert_array_equal(unchanged, np.zeros(5))
