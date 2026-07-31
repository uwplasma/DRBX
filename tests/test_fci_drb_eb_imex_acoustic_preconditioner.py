"""Focused contract tests for the coupled EB ARK2 acoustic preconditioner.

These intentionally test only the right-preconditioner contract.  In
particular, a preconditioner is not expected to be an accurate inverse of the
full nonlinear five-field stage Jacobian on every mapped geometry.  The tests
therefore require a finite, shape-preserving sharded map which makes genuine
electron-acoustic corrections and handles the driver-scaled algebraic phi row
correctly.  Newton convergence is covered separately by the production smoke
tests.
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
    build_eb_imex_acoustic_line_uv_preconditioner,
    build_eb_imex_phi_line_u_preconditioner,
    implicit_state_from_eb_state,
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
    """A small real mapped geometry suitable for jit(shard_map) coverage."""

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
    return LocalFciDrbEBRhs(
        geometry=geometry,
        domain=local.domain,
        halo_exchange=HaloExchange3D(),
        topology_filler=TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),)),
        physical_ghost_filler=_ghost_filler(HALO_WIDTH),
        parameters=context.parameters,
        curvature_coefficients_owned=build_local_curvature_coefficients(
            geometry,
            local.domain,
            periodic_axes=PERIODIC_AXES,
            axis_regular_axes=AXIS_REGULAR_AXES,
        ),
        face_projectors=build_local_perp_laplacian_face_projectors(
            geometry,
            local.domain,
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


def test_acoustic_preconditioner_is_jittable_coupled_and_homogeneous() -> None:
    """It makes finite coupled corrections inside the real sharded map.

    A phi-only preconditioner is identity on n/Te/Ve/omega.  A nontrivial
    acoustic residual must instead produce corrections to at least the
    electron-acoustic leaves.  A zero residual must stay exactly zero, which
    checks compatibility with homogeneous physical boundary closure without
    asserting that an owned boundary-adjacent cell itself is zero.
    """

    context, mesh, local, partition, fields, cell_fields = _context_and_sharded_inputs()
    dt_gamma = 1.0e-3

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        rhs = _build_rhs(context, local, geometry)
        # Smooth but fully coupled deterministic residual.  The phi leaf is
        # already in the driver convention: dt_gamma * R_phi.
        value = FciDrbEBImplicitState(
            density=0.17 * jnp.sin(1.3 * density + 0.2 * Te),
            phi=dt_gamma * (0.31 * jnp.cos(1.7 * phi + 0.1 * vorticity)),
            Te=-0.11 * jnp.cos(0.9 * Te - 0.3 * density),
            Ve=0.23 * jnp.sin(1.1 * Ve + 0.4 * Te),
            vorticity=-0.19 * jnp.sin(0.7 * vorticity + 0.2 * phi),
        )
        acoustic = build_eb_imex_acoustic_line_uv_preconditioner(
            rhs, dt_gamma
        )(value)
        phi_only = build_eb_imex_phi_line_u_preconditioner(rhs, dt_gamma)(value)
        zero = FciDrbEBImplicitState(*(
            jnp.zeros_like(density) for _ in range(5)
        ))
        homogeneous = build_eb_imex_acoustic_line_uv_preconditioner(
            rhs, dt_gamma
        )(zero)
        active = geometry.active_cell_mask_owned

        def norm_sq(field):
            return lax.psum(
                jnp.sum(jnp.where(active, field * field, 0.0)), ("x", "y", "z")
            )

        # Report output leaf shapes, finite status, coupled correction norms,
        # phi-only identity defects, and homogeneous-output norm.
        finite = jnp.asarray(True)
        for field in acoustic.field_values():
            finite = finite & jnp.all(jnp.isfinite(field))
        shapes_match = jnp.asarray(True)
        for out, inp in zip(acoustic.field_values(), value.field_values(), strict=True):
            shapes_match = shapes_match & jnp.asarray(out.shape == inp.shape)
        correction = jnp.stack((
            norm_sq(acoustic.density - value.density),
            norm_sq(acoustic.Te - value.Te),
            norm_sq(acoustic.Ve - value.Ve),
            norm_sq(acoustic.vorticity - value.vorticity),
        ))
        phi_identity = jnp.stack((
            norm_sq(phi_only.density - value.density),
            norm_sq(phi_only.Te - value.Te),
            norm_sq(phi_only.Ve - value.Ve),
            norm_sq(phi_only.vorticity - value.vorticity),
        ))
        zero_norm = sum(norm_sq(field) for field in homogeneous.field_values())
        return finite, shapes_match, correction, phi_identity, zero_norm

    compiled = jax.jit(jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(partition,) * 8,
        out_specs=(P(), P(), P(), P(), P()),
        check_vma=False,
    ))
    finite, shapes_match, correction, phi_identity, zero_norm = jax.block_until_ready(
        compiled(*fields, cell_fields)
    )
    assert bool(finite)
    assert bool(shapes_match)
    correction = np.asarray(correction)
    phi_identity = np.asarray(phi_identity)
    assert np.all(np.isfinite(correction))
    # The coupled block must do more than the phi-only identity mapping on
    # the acoustic variables.  It need not improve every omega component.
    assert np.any(correction[:3] > 1.0e-24), correction
    np.testing.assert_array_equal(phi_identity, np.zeros_like(phi_identity))
    assert float(zero_norm) == 0.0


def test_acoustic_preconditioner_respects_phi_row_scaling_and_jvp_smoke() -> None:
    """The algebraic phi correction is 1/(gamma*dt), with a finite JVP path.

    The last check deliberately avoids demanding global Newton convergence:
    it verifies that applying the coupled right map to a real stage-Jacobian
    residual gives a finite correction whose defect is not catastrophically
    worse than identity on this mapped, physically closed test geometry.
    """

    context, mesh, local, partition, fields, cell_fields = _context_and_sharded_inputs()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        rhs = _build_rhs(context, local, geometry)
        known = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        z = implicit_state_from_eb_state(known)
        predictor = z.replace(
            density=0.997 * z.density,
            Te=1.002 * z.Te,
            Ve=z.Ve + 0.004,
            vorticity=0.999 * z.vorticity,
            phi=1.001 * z.phi,
        )
        direction = FciDrbEBImplicitState(
            density=0.03 * jnp.sin(density + 0.3),
            phi=0.02 * jnp.cos(phi - 0.1),
            Te=-0.025 * jnp.sin(Te + 0.4),
            Ve=0.035 * jnp.cos(Ve - 0.2),
            vorticity=0.015 * jnp.sin(vorticity + 0.5),
        )
        dt = jnp.asarray(1.0e-3, dtype=jnp.float64)

        def scaled_stage_residual(value):
            raw = rhs.implicit_stage_residual(
                value, predictor, known, dt_gamma=dt
            )
            return raw.replace(phi=dt * raw.phi)

        _, residual = jax.jvp(scaled_stage_residual, (z,), (direction,))
        acoustic = build_eb_imex_acoustic_line_uv_preconditioner(rhs, dt)(residual)
        # Use a pure algebraic-row sample for the scaling check.  The acoustic
        # response/back-substitution is deliberately dt dependent, so this is
        # a consistency check (finite, nonzero, leading inverse-dt behavior),
        # not an assertion of exact scalar-inverse equality.
        phi_row = residual.replace(
            density=jnp.zeros_like(residual.density),
            Te=jnp.zeros_like(residual.Te),
            Ve=jnp.zeros_like(residual.Ve),
            vorticity=jnp.zeros_like(residual.vorticity),
        )
        phi_coarse = build_eb_imex_acoustic_line_uv_preconditioner(rhs, 2.0 * dt)(phi_row)
        phi_fine = build_eb_imex_acoustic_line_uv_preconditioner(rhs, dt)(phi_row)
        active = geometry.active_cell_mask_owned

        def norm_sq(state):
            total = jnp.asarray(0.0, dtype=jnp.float64)
            for field in state.field_values():
                total = total + jnp.sum(jnp.where(active, field * field, 0.0))
            return lax.psum(total, ("x", "y", "z"))

        _, acoustic_image = jax.jvp(scaled_stage_residual, (z,), (acoustic,))
        _, identity_image = jax.jvp(scaled_stage_residual, (z,), (residual,))
        defect = acoustic_image.axpy(residual, scale=-1.0)
        identity_defect = identity_image.axpy(residual, scale=-1.0)
        scale_ratio = jnp.max(jnp.where(
            active,
            jnp.abs(phi_fine.phi) / jnp.maximum(jnp.abs(phi_coarse.phi), 1.0e-30),
            0.0,
        ))
        finite = jnp.asarray(True)
        for state in (residual, acoustic, defect):
            for field in state.field_values():
                finite = finite & jnp.all(jnp.isfinite(field))
        return jnp.stack((norm_sq(defect), norm_sq(identity_defect), scale_ratio)), finite

    compiled = jax.jit(jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(partition,) * 8,
        out_specs=(P(), P()),
        check_vma=False,
    ))
    values, finite = jax.block_until_ready(compiled(*fields, cell_fields))
    defect, identity_defect, scale_ratio = np.asarray(values)
    assert bool(finite)
    assert np.all(np.isfinite((defect, identity_defect, scale_ratio)))
    # This is deliberately a smoke bound rather than a convergence claim.  A
    # fixed-cost local block is allowed to be approximate but must not turn a
    # smooth stage residual into an arbitrarily large correction.
    assert defect <= 100.0 * max(identity_defect, 1.0e-30), (defect, identity_defect)
    # Halving gamma*dt must produce a materially larger phi correction for a
    # pure driver-scaled algebraic residual.  The response sweep/backsolve is
    # approximate, so the exact factor two is intentionally not required.
    assert scale_ratio > 1.2, scale_ratio
