"""Focused checks for the opt-in characteristic wall current/phi pairing."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from drbx.native import FciDrbEBState
from drbx.native.fci_drb_EB_rhs import (
    parallel_characteristic_wall_data,
    build_local_fci_drb_eb_operator_boundary_bundle,
)
from drbx.native.fci_sharding import assemble_local_fci_geometry
from drbx.native.fci_operators import local_grad_parallel_op_fci
from drbx.geometry import build_local_fci_stencil_from_field

from fci_drb_eb_test_helpers import _build_rhs, _context_and_sharded_inputs
from test_fci_drb_eb_parallel_flux_pairing import _mapped_fixture


def test_wall_data_current_is_the_first_order_incoming_characteristic_current():
    center = jnp.asarray([[1.2, 1.1, 0.9, 0.25, -0.15]])
    minus = jnp.asarray([[1.0, 1.0, 1.0, 0.0, 0.0]])
    plus = jnp.asarray([[1.1, 0.95, 1.05, -0.1, 0.2]])
    info = parallel_characteristic_wall_data(
        center, minus, plus, jnp.asarray([0.2]), jnp.asarray([0.3]),
        4.0, 10.0,
        backward_wall=jnp.asarray([True]), forward_wall=jnp.asarray([True]),
        backward_wall_state=minus, forward_wall_state=plus,
    )
    center_current = center[..., 0] * (center[..., 3] - center[..., 4])
    for direction in ("backward", "forward"):
        projected = info[f"{direction}_wall_projected_state"]
        delta = projected - center
        expected = (
            center_current
            + (center[..., 3] - center[..., 4]) * delta[..., 0]
            + center[..., 0] * (delta[..., 3] - delta[..., 4])
        )
        nonlinear = projected[..., 0] * (
            projected[..., 3] - projected[..., 4]
        )
        quadratic = delta[..., 0] * (delta[..., 3] - delta[..., 4])
        np.testing.assert_allclose(
            info[f"{direction}_wall_characteristic_current"], expected,
        )
        np.testing.assert_allclose(
            info[f"{direction}_wall_projected_current"], expected,
        )
        np.testing.assert_allclose(
            info[f"{direction}_wall_projected_nonlinear_current"], nonlinear,
        )
        np.testing.assert_allclose(
            info[f"{direction}_wall_current_quadratic_remainder"], quadratic,
        )
    # The endpoint state is the projected wall state, not the unprojected
    # candidate supplied by the interpolation stencil.
    assert not np.allclose(
        np.asarray(info["backward_endpoint_current"]), np.asarray(minus[..., 0] * (minus[..., 3] - minus[..., 4]))
    )


def test_characteristic_sat_decomposition_is_exact_on_mapped_fixture():
    context, mesh, local, partition, fields, cell_fields, map_fields, sharded = (
        _mapped_fixture()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        # The shifted-torus mapped fixture has ordinary closed legs.  Mark
        # its two radial edge planes as physical FCI endpoints so this small
        # real mapped operator exercises the wall branch without changing
        # the interpolation tables.
        backward_kind = jnp.zeros_like(geometry.maps.backward.endpoint_kind).at[0].set(2)
        forward_kind = jnp.zeros_like(geometry.maps.forward.endpoint_kind).at[-1].set(2)
        geometry = replace(
            geometry,
            maps=replace(
                geometry.maps,
                backward=replace(geometry.maps.backward, endpoint_kind=backward_kind),
                forward=replace(geometry.maps.forward, endpoint_kind=forward_kind),
            ),
        )
        rhs = replace(
            _build_rhs(context, local, geometry),
            parallel_operator_scheme="fci",
            parallel_material_scheme="production-path",
            parallel_flux_pairing="support-core",
            parallel_boundary_pairing="characteristic-sat",
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        face_bc = rhs._face_bcs(state)
        state_halo = rhs._prepare_state_halo(state, face_bc)
        operator_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
            state_halo, geometry, rhs.domain, face_bc, tau=rhs.parameters.tau
        )
        parallel_boundary = rhs._parallel_operator_boundary(
            state_halo=state_halo, operator_boundary=operator_boundary
        )
        terms = rhs._fci_parallel_terms(
            state_halo=state_halo,
            face_bc=face_bc,
            operator_boundary=operator_boundary,
            parallel_boundary=parallel_boundary,
            context=rhs._stencil_builder_context(),
            return_electron_force_diagnostics=True,
        )
        homogeneous = terms["characteristic_sat_homogeneous_current_divergence"]
        affine = terms["characteristic_sat_affine_current_divergence"]
        sat = terms["characteristic_sat_current_divergence"]
        target = rhs._fci_pair_target_mask(include_physical_wall=True)
        wall = (
            (geometry.maps.forward.endpoint_kind == 2)
            | (geometry.maps.backward.endpoint_kind == 2)
        )
        _, standard_divergence, _ = rhs._fci_current_phi_boundary_pair(
            face_bc=face_bc, context=rhs._stencil_builder_context()
        )
        current = density * (Vi - Ve)
        mass = rhs._fci_pair_cell_mass()
        stencil_context = rhs._stencil_builder_context()
        current_q_halo, current_q_forward, current_q_backward = (
            rhs._fci_prepare_flux_q(
                current, parallel_boundary.current, stencil_context
            )
        )
        current_q_stencil = build_local_fci_stencil_from_field(
            current_q_halo,
            geometry,
            stencil_context,
            forward_remote_values=current_q_forward,
            backward_remote_values=current_q_backward,
        )
        inverse_b_halo, inverse_b_forward, inverse_b_backward = (
            rhs._fci_prepare_inverse_b(face_bc, stencil_context)
        )
        inverse_b_stencil = build_local_fci_stencil_from_field(
            inverse_b_halo,
            geometry,
            stencil_context,
            forward_remote_values=inverse_b_forward,
            backward_remote_values=inverse_b_backward,
        )
        endpoint_values = terms["material_characteristic_endpoint_values"]
        center_values = jnp.stack((density, Te, Ti, Vi, Ve), axis=-1)
        endpoint_delta = endpoint_values - center_values[..., None, :]
        center_current = density * (Vi - Ve)
        endpoint_currents = (
            center_current[..., None]
            + (Vi - Ve)[..., None] * endpoint_delta[..., 0]
            + density[..., None]
            * (endpoint_delta[..., 3] - endpoint_delta[..., 4])
        )
        manual_q_stencil = replace(
            current_q_stencil,
            minus=jnp.where(
                geometry.maps.backward.endpoint_kind == 2,
                endpoint_currents[..., 0] * inverse_b_stencil.minus,
                current_q_stencil.minus,
            ),
            plus=jnp.where(
                geometry.maps.forward.endpoint_kind == 2,
                endpoint_currents[..., 1] * inverse_b_stencil.plus,
                current_q_stencil.plus,
            ),
        )
        manual_sat = geometry.cell_bfield.Bmag_owned * local_grad_parallel_op_fci(
            manual_q_stencil, geometry
        )
        # The homogeneous current divergence is the negative adjoint of the
        # corresponding phi gradient; obtain the latter from the same pair.
        homogeneous_gradient, _, _ = rhs._fci_current_phi_boundary_pair(
            face_bc=face_bc,
            context=rhs._stencil_builder_context(),
            wall_endpoint_current_values=(
                jnp.zeros(geometry.owned_shape),
                jnp.zeros(geometry.owned_shape),
            ),
        )
        adjoint_error = jnp.abs(
            jnp.sum(mass * phi * homogeneous)
            + jnp.sum(mass * homogeneous_gradient(phi) * current)
        )
        green_remainder = jnp.sum(mass * phi * (sat - homogeneous))
        green_error = jnp.abs(
            jnp.sum(mass * phi * sat)
            + jnp.sum(mass * homogeneous_gradient(phi) * current)
            - green_remainder
        )
        # An independently prescribed, nonzero wall-current lift exercises
        # the affine part even when the MMS current happens to match the
        # characteristic projection at some synthetic wall rows.
        _, custom_divergence, _ = rhs._fci_current_phi_boundary_pair(
            face_bc=face_bc,
            context=stencil_context,
            wall_endpoint_current_values=(
                jnp.full(geometry.owned_shape, 0.123),
                jnp.full(geometry.owned_shape, -0.234),
            ),
            build_adjoint=False,
        )
        custom_sat = custom_divergence(current)
        custom_remainder = jnp.sum(mass * phi * (custom_sat - homogeneous))
        custom_green_error = jnp.abs(
            jnp.sum(mass * phi * custom_sat)
            + jnp.sum(mass * homogeneous_gradient(phi) * current)
            - custom_remainder
        )
        ordinary_error = jnp.max(jnp.where(
            ~wall,
            jnp.abs(homogeneous - standard_divergence(current)),
            0.0,
        ))
        return jnp.asarray((
            jnp.max(jnp.abs(sat - homogeneous - affine)),
            jnp.max(jnp.where(
                target,
                jnp.abs(terms["vorticity_current_flux_div"] - sat),
                0.0,
            )),
            jnp.sum(wall),
            jnp.max(jnp.abs(terms["material_characteristic_endpoint_values"])),
            adjoint_error,
            ordinary_error,
            jnp.max(jnp.abs(affine)),
            jnp.max(jnp.abs(
                jnp.sum(
                    terms["material_upwind_correction_components"], axis=-2
                ) - terms["parallel_material_residual"]
            )),
            jnp.max(jnp.where(target, jnp.abs(manual_sat - sat), 0.0)),
            green_error,
            custom_green_error,
            custom_remainder,
        ))

    compiled = jax.jit(jax.shard_map(
        kernel, mesh=mesh, in_specs=(partition,) * 9,
        out_specs=P(), check_vma=False,
    ))
    error, vorticity_error, wall_count, endpoint_norm, adjoint_error, ordinary_error, affine_norm, material_sum_error, manual_sat_error, green_error, custom_green_error, custom_remainder = np.asarray(
        compiled(*fields, cell_fields, map_fields)
    )
    assert wall_count > 0
    assert error < 2.0e-12
    assert vorticity_error < 2.0e-12
    assert endpoint_norm > 0.0
    assert adjoint_error < 2.0e-10
    assert ordinary_error < 2.0e-12
    # The MMS state can have a current which is already characteristic at
    # the synthetic wall rows, so its affine lift may vanish; the independent
    # wall-data test above covers the non-equilibrium characteristic-current
    # case.
    assert np.isfinite(affine_norm)
    assert material_sum_error < 2.0e-12
    assert manual_sat_error < 2.0e-12
    assert green_error < 2.0e-10
    assert custom_green_error < 2.0e-10
    assert np.isfinite(custom_remainder)


def test_characteristic_sat_validation_rejects_non_fci_or_non_support_path():
    context, mesh, local, partition, fields, cell_fields = (
        _context_and_sharded_inputs()
    )

    def invalid(density, phi, Te, Ti, Vi, Ve, vorticity, cells):
        geometry = assemble_local_fci_geometry(local, cells)
        replace(
            _build_rhs(context, local, geometry),
            parallel_boundary_pairing="characteristic-sat",
        )
        return jnp.asarray(0.0)

    compiled = jax.jit(jax.shard_map(
        invalid, mesh=mesh, in_specs=(partition,) * 8,
        out_specs=P(), check_vma=False,
    ))
    with pytest.raises(ValueError, match="characteristic-sat"):
        compiled(*fields, cell_fields)
