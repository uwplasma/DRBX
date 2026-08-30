"""Focused checks for the opt-in characteristic wall current/phi pairing."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from drbx.native import FciDrbEBRhsParameters, FciDrbEBState
from drbx.native.fci_drb_EB_rhs import (
    parallel_characteristic_wall_data,
    build_local_fci_drb_eb_operator_boundary_bundle,
)
from drbx.native.characteristic_wall_residual import (
    apply_maximally_dissipative_characteristic_wall,
)
from drbx.native.fci_parallel_production_flux import (
    parallel_characteristic_decomposition,
    parallel_characteristic_matrix,
    parallel_short_wall_backward_euler,
    parallel_short_wall_material_data,
    parallel_target_row_material_residual,
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


@pytest.mark.parametrize(
    "wall_law", ("primitive-least-residual", "energy-absorbing")
)
def test_characteristic_sat_decomposition_is_exact_on_mapped_fixture(wall_law):
    context, mesh, local, partition, fields, cell_fields, map_fields, sharded = (
        _mapped_fixture()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        # The MMS profile vanishes at its radial edges.  Give the energy law a
        # small, admissible wall mismatch so both the projected endpoint and
        # its first-order current lift are exercised on the real mapped path.
        if wall_law == "energy-absorbing":
            density = density.at[0].add(2.0e-2)
            density = density.at[-1].add(-1.5e-2)
            Vi = Vi.at[0].add(3.0e-2)
            Vi = Vi.at[-1].add(-2.0e-2)
            Ve = Ve.at[0].add(-2.5e-2)
            Ve = Ve.at[-1].add(1.0e-2)
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        base_context = replace(
            context,
            parameters=replace(
                context.parameters,
                parallel_characteristic_wall_law="primitive-least-residual",
            ),
        )
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
            _build_rhs(base_context, local, geometry),
            parameters=replace(
                base_context.parameters,
                parallel_characteristic_wall_law=wall_law,
            ),
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
        # Independently rebuild the energy wall map from the public spectral
        # decomposition.  This avoids making endpoint consistency a comparison
        # against the same wall-data helper used by the production assembly.
        matrix = parallel_characteristic_matrix(
            density, Te, Ti, Vi, Ve,
            rhs.parameters.tau, rhs.parameters.mi_over_me,
        )
        eigenvalues, right, left, spectral_valid = (
            parallel_characteristic_decomposition(matrix)
        )
        equilibrium = jnp.broadcast_to(
            jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0), dtype=jnp.float64),
            center_values.shape,
        )
        expected_backward_endpoint, _ = (
            apply_maximally_dissipative_characteristic_wall(
                center_values, equilibrium, -eigenvalues, right, left,
                thermodynamic_components=3,
                spectral_valid=spectral_valid,
            )
        )
        expected_forward_endpoint, _ = (
            apply_maximally_dissipative_characteristic_wall(
                center_values, equilibrium, eigenvalues, right, left,
                thermodynamic_components=3,
                spectral_valid=spectral_valid,
            )
        )
        # The production residual and its diagnostic endpoint report must use
        # the same energy wall projection as this independent reconstruction.
        backward_endpoint_consistency = jnp.max(jnp.where(
            geometry.maps.backward.endpoint_kind[..., None] == 2,
            jnp.abs(endpoint_values[..., 0, :] - expected_backward_endpoint),
            0.0,
        ))
        forward_endpoint_consistency = jnp.max(jnp.where(
            geometry.maps.forward.endpoint_kind[..., None] == 2,
            jnp.abs(endpoint_values[..., 1, :] - expected_forward_endpoint),
            0.0,
        ))
        endpoint_consistency = (
            jnp.maximum(backward_endpoint_consistency, forward_endpoint_consistency)
            if wall_law == "energy-absorbing" else jnp.asarray(0.0)
        )
        endpoint_delta = endpoint_values - center_values[..., None, :]
        wall_projection_delta = jnp.maximum(
            jnp.max(jnp.where(
                geometry.maps.backward.endpoint_kind[..., None] == 2,
                jnp.abs(endpoint_delta[..., 0, :]),
                0.0,
            )),
            jnp.max(jnp.where(
                geometry.maps.forward.endpoint_kind[..., None] == 2,
                jnp.abs(endpoint_delta[..., 1, :]),
                0.0,
            )),
        )
        center_current = density * (Vi - Ve)
        endpoint_currents = (
            center_current[..., None]
            + (Vi - Ve)[..., None] * endpoint_delta[..., 0]
            + density[..., None]
            * (endpoint_delta[..., 3] - endpoint_delta[..., 4])
        )
        backward_current_consistency = jnp.max(jnp.where(
            geometry.maps.backward.endpoint_kind == 2,
            jnp.abs(
                endpoint_currents[..., 0]
                - (
                    center_current
                    + (Vi - Ve) * (expected_backward_endpoint[..., 0] - density)
                    + density
                    * (
                        expected_backward_endpoint[..., 3]
                        - expected_backward_endpoint[..., 4]
                        - Vi + Ve
                    )
                )
            ),
            0.0,
        ))
        forward_current_consistency = jnp.max(jnp.where(
            geometry.maps.forward.endpoint_kind == 2,
            jnp.abs(
                endpoint_currents[..., 1]
                - (
                    center_current
                    + (Vi - Ve) * (expected_forward_endpoint[..., 0] - density)
                    + density
                    * (
                        expected_forward_endpoint[..., 3]
                        - expected_forward_endpoint[..., 4]
                        - Vi + Ve
                    )
                )
            ),
            0.0,
        ))
        current_consistency = (
            jnp.maximum(backward_current_consistency, forward_current_consistency)
            if wall_law == "energy-absorbing" else jnp.asarray(0.0)
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
            endpoint_consistency,
            current_consistency,
            wall_projection_delta,
            jnp.asarray(jnp.all(jnp.where(
                wall,
                terms["parallel_material_diagnostics"]["admissible"],
                True,
            )), dtype=jnp.float64),
            green_error,
            custom_green_error,
            custom_remainder,
        ))

    compiled = jax.jit(jax.shard_map(
        kernel, mesh=mesh, in_specs=(partition,) * 9,
        out_specs=P(), check_vma=False,
    ))
    error, vorticity_error, wall_count, endpoint_norm, adjoint_error, ordinary_error, affine_norm, material_sum_error, manual_sat_error, endpoint_consistency, current_consistency, wall_projection_delta, wall_admissible, green_error, custom_green_error, custom_remainder = np.asarray(
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
    if wall_law == "energy-absorbing":
        assert wall_projection_delta > 0.0
    assert material_sum_error < 2.0e-12
    assert manual_sat_error < 2.0e-12
    assert endpoint_consistency < 2.0e-12
    assert current_consistency < 2.0e-12
    assert wall_admissible == 1.0
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


def test_energy_absorbing_wall_law_validation_rejects_unknown_selector():
    with pytest.raises(ValueError, match="parallel_characteristic_wall_law"):
        FciDrbEBRhsParameters(parallel_characteristic_wall_law="not-a-law")


@pytest.mark.parametrize(
    ("rhs_overrides", "message"),
    (
        (
            {"parallel_material_scheme": "legacy"},
            "parallel_material_scheme",
        ),
        (
            {
                "parallel_operator_scheme": "fci",
                "parallel_material_scheme": "production-path",
                "parallel_flux_pairing": "support-core",
                "parallel_boundary_pairing": "current-phi",
            },
            "parallel_boundary_pairing",
        ),
        (
            {
                "parallel_operator_scheme": "fci",
                "parallel_material_scheme": "production-path",
                "parallel_flux_pairing": "support-core",
                "parallel_boundary_pairing": "characteristic-sat",
                "parallel_inflow_closure": "local-characteristic",
            },
            "parallel_inflow_closure",
        ),
    ),
)
def test_energy_absorbing_wall_law_validation_guards_rhs_path(
    rhs_overrides, message
):
    context, mesh, local, partition, fields, cell_fields = (
        _context_and_sharded_inputs()
    )

    def invalid(density, phi, Te, Ti, Vi, Ve, vorticity, cells):
        geometry = assemble_local_fci_geometry(local, cells)
        base_context = replace(
            context,
            parameters=replace(
                context.parameters,
                parallel_characteristic_wall_law="primitive-least-residual",
            ),
        )
        params = replace(
            base_context.parameters,
            parallel_characteristic_wall_law="energy-absorbing",
        )
        replace(
            _build_rhs(base_context, local, geometry),
            parameters=params,
            **rhs_overrides,
        )
        return jnp.asarray(0.0)

    compiled = jax.jit(jax.shard_map(
        invalid, mesh=mesh, in_specs=(partition,) * 8,
        out_specs=P(), check_vma=False,
    ))
    with pytest.raises(ValueError, match=message):
        compiled(*fields, cell_fields)


def test_rhs_default_wall_law_is_the_primitive_path(monkeypatch):
    monkeypatch.delenv("DRBX_PARALLEL_CHARACTERISTIC_WALL_LAW", raising=False)
    parameters = FciDrbEBRhsParameters()
    assert parameters.parallel_characteristic_wall_law == (
        "primitive-least-residual"
    )

    center = jnp.asarray([1.2, 1.1, 0.9, 0.25, -0.15])
    candidate = jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0])
    default = parallel_characteristic_wall_data(
        center, candidate, candidate, 0.2, 0.3, 4.0, 10.0,
        backward_wall=True, forward_wall=True,
        backward_wall_state=candidate, forward_wall_state=candidate,
    )
    explicit = parallel_characteristic_wall_data(
        center, candidate, candidate, 0.2, 0.3, 4.0, 10.0,
        backward_wall=True, forward_wall=True,
        backward_wall_state=candidate, forward_wall_state=candidate,
        parallel_characteristic_wall_law="primitive-least-residual",
    )
    for name in (
        "backward_endpoint_state", "forward_endpoint_state",
        "backward_endpoint_current", "forward_endpoint_current",
    ):
        np.testing.assert_allclose(default[name], explicit[name])


def test_energy_absorbing_rhs_path_keeps_invalid_wall_candidates_finite():
    center = jnp.broadcast_to(
        jnp.asarray((1.2, 1.1, 0.9, 0.25, -0.15), dtype=jnp.float64),
        (2, 5),
    )
    nan_candidate = jnp.full_like(center, jnp.nan)
    residual, diagnostics = jax.jit(
        parallel_target_row_material_residual,
        static_argnames=("parallel_characteristic_wall_law",),
    )(
        center, center, center, jnp.asarray((0.2, 0.3)),
        jnp.asarray((0.3, 0.2)), 4.0, 10.0,
        backward_wall=jnp.asarray((True, True)),
        forward_wall=jnp.asarray((True, True)),
        backward_wall_state=nan_candidate,
        forward_wall_state=nan_candidate,
        parallel_characteristic_wall_law="energy-absorbing",
    )
    assert bool(jnp.all(jnp.isfinite(residual)))
    assert bool(jnp.all(diagnostics["admissible"]))
    assert bool(jnp.all(~diagnostics["fallback"]))


def test_all_physical_walls_selection_masks_only_physical_directions():
    center = jnp.broadcast_to(
        jnp.asarray((1.2, 1.1, 0.9, 0.25, -0.15), dtype=jnp.float64),
        (3, 5),
    )
    walls_backward = jnp.asarray((True, False, True))
    walls_forward = jnp.asarray((False, True, False))
    residual, jacobian, info = parallel_short_wall_material_data(
        center, center, center, jnp.asarray((100.0, 100.0, 100.0)),
        jnp.asarray((100.0, 100.0, 100.0)), 4.0, 10.0,
        selection_dt=0.0,
        backward_wall=walls_backward,
        forward_wall=walls_forward,
        parallel_characteristic_wall_law="energy-absorbing",
        parallel_short_leg_selection="all-physical-walls",
    )
    np.testing.assert_array_equal(info["selected_backward_wall"], walls_backward)
    np.testing.assert_array_equal(info["selected_forward_wall"], walls_forward)
    np.testing.assert_array_equal(
        info["selected_wall"], walls_backward | walls_forward
    )
    assert bool(jnp.all(jnp.isfinite(residual)))
    assert bool(jnp.all(jnp.isfinite(jacobian)))


def test_all_physical_walls_omits_explicit_material_and_be_includes_one_wall():
    center = jnp.broadcast_to(
        jnp.asarray((1.2, 1.1, 0.9, 0.25, -0.15), dtype=jnp.float64),
        (2, 5),
    )
    walls_backward = jnp.asarray((True, False))
    walls_forward = jnp.asarray((False, False))
    residual, diagnostics = parallel_target_row_material_residual(
        center, center, center, jnp.asarray((100.0, 100.0)),
        jnp.asarray((100.0, 100.0)), 4.0, 10.0,
        backward_wall=walls_backward,
        forward_wall=walls_forward,
        parallel_characteristic_wall_law="energy-absorbing",
        parallel_short_leg_selection="all-physical-walls",
    )
    updated, increment, info = parallel_short_wall_backward_euler(
        center, center, center, jnp.asarray((100.0, 100.0)),
        jnp.asarray((100.0, 100.0)), 4.0, 10.0,
        selection_dt=0.0,
        solve_dt=1.0e-3,
        backward_wall=walls_backward,
        forward_wall=walls_forward,
        parallel_characteristic_wall_law="energy-absorbing",
        parallel_short_leg_selection="all-physical-walls",
    )
    assert bool(diagnostics["omitted_backward_wall"][0])
    assert not bool(diagnostics["omitted_backward_wall"][1])
    assert bool(jnp.all(jnp.isfinite(updated)))
    assert bool(jnp.all(jnp.isfinite(increment)))
    assert bool(jnp.any(jnp.abs(increment[0]) > 0.0))
    np.testing.assert_array_equal(increment[1], jnp.zeros(5))
    np.testing.assert_array_equal(info["selected_backward_wall"], walls_backward)
    np.testing.assert_array_equal(info["selected_forward_wall"], walls_forward)
    # The explicit and BE paths partition the same wall material action.
    np.testing.assert_allclose(
        residual[0], jnp.zeros(5), atol=1.0e-14, rtol=0.0
    )


def test_short_wall_backward_euler_propagates_nonfinite_local_solve():
    center = jnp.broadcast_to(
        jnp.asarray((1.2, 1.1, 0.9, 0.25, -0.15), dtype=jnp.float64),
        (2, 5),
    )
    updated, increment, info = parallel_short_wall_backward_euler(
        center,
        center,
        center,
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((1.0, 1.0)),
        4.0,
        10.0,
        selection_dt=jnp.asarray((0.02, 0.02)),
        solve_dt=jnp.asarray((jnp.nan, 1.0e-3)),
        backward_wall=jnp.asarray((True, False)),
        forward_wall=jnp.asarray((False, False)),
    )
    assert bool(jnp.any(~jnp.isfinite(increment[0])))
    assert bool(jnp.any(~jnp.isfinite(updated[0])))
    assert bool(info["implicit_solve_fallback"][0])
    assert not bool(info["implicit_finite"][0])
    assert bool(jnp.all(jnp.isfinite(increment[1])))
    np.testing.assert_array_equal(increment[1], jnp.zeros(5))
    assert not bool(info["implicit_solve_fallback"][1])
    assert bool(info["implicit_finite"][1])


@pytest.mark.parametrize(
    ("law", "treatment", "message"),
    (
        ("primitive-least-residual", "local-backward-euler", "wall_law"),
        ("energy-absorbing", "explicit", "short_leg_treatment"),
    ),
)
def test_all_physical_walls_requires_energy_law_and_implicit_treatment(
    law, treatment, message
):
    context, mesh, local, partition, fields, cell_fields = (
        _context_and_sharded_inputs()
    )

    def invalid(density, phi, Te, Ti, Vi, Ve, vorticity, cells):
        geometry = assemble_local_fci_geometry(local, cells)
        base_context = replace(
            context,
            parameters=replace(
                context.parameters,
                parallel_characteristic_wall_law="primitive-least-residual",
            ),
        )
        replace(
            _build_rhs(base_context, local, geometry),
            parameters=replace(
                base_context.parameters,
                parallel_characteristic_wall_law=law,
            ),
            parallel_operator_scheme="fci",
            parallel_material_scheme="production-path",
            parallel_flux_pairing="support-core",
            parallel_boundary_pairing="characteristic-sat",
            parallel_short_leg_treatment=treatment,
            parallel_short_leg_selection="all-physical-walls",
        )
        return jnp.asarray(0.0)

    compiled = jax.jit(jax.shard_map(
        invalid, mesh=mesh, in_specs=(partition,) * 8,
        out_specs=P(), check_vma=False,
    ))
    with pytest.raises(ValueError, match=message):
        compiled(*fields, cell_fields)


def test_all_physical_walls_be_updates_complete_local_block_on_real_mapped_rhs():
    context, mesh, local, partition, fields, cell_fields, map_fields, sharded = (
        _mapped_fixture()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
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
        base_context = replace(
            context,
            parameters=replace(
                context.parameters,
                parallel_characteristic_wall_law="primitive-least-residual",
            ),
        )
        rhs = replace(
            _build_rhs(base_context, local, geometry),
            parameters=replace(
                base_context.parameters,
                parallel_characteristic_wall_law="energy-absorbing",
            ),
            parallel_operator_scheme="fci",
            parallel_material_scheme="production-path",
            parallel_flux_pairing="support-core",
            parallel_boundary_pairing="characteristic-sat",
            parallel_short_leg_treatment="local-backward-euler",
            parallel_short_leg_selection="all-physical-walls",
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        updated = rhs.apply_short_leg_implicit_material_step(
            state, solve_dt=1.0e-3, selection_dt=0.0, phi_owned=phi
        )
        material_delta = jnp.max(jnp.stack((
            jnp.abs(updated.density - density),
            jnp.abs(updated.Te - Te),
            jnp.abs(updated.Ti - Ti),
            jnp.abs(updated.Vi - Vi),
            jnp.abs(updated.Ve - Ve),
        )))
        return jnp.asarray((
            material_delta,
            jnp.max(jnp.abs(updated.phi - phi)),
            jnp.max(jnp.abs(updated.vorticity - vorticity)),
            jnp.max(jnp.stack(tuple(
                jnp.max(jnp.abs(value))
                for value in (updated.density, updated.Te, updated.Ti,
                              updated.Vi, updated.Ve)
            ))),
        ))

    compiled = jax.jit(jax.shard_map(
        kernel, mesh=mesh, in_specs=(partition,) * 9,
        out_specs=P(), check_vma=False,
    ))
    material_delta, phi_delta, vorticity_delta, material_norm = np.asarray(
        compiled(*fields, cell_fields, map_fields)
    )
    assert np.isfinite(material_delta)
    assert material_norm < np.inf
    assert phi_delta == 0.0
    assert vorticity_delta == 0.0


@pytest.mark.slow
def test_mapped_selected_row_partition_reconstructs_the_unsplit_rhs():
    """The explicit and implicit handoff must lose or duplicate no RHS term."""

    context, mesh, local, partition, fields, cell_fields, map_fields, sharded = (
        _mapped_fixture()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        # Exercise a nontrivial wall residual instead of the fixture's nearly
        # homogeneous radial edge values.
        density = density.at[0].add(2.0e-2).at[-1].add(-1.5e-2)
        Ti = Ti.at[0].add(3.0e-2).at[-1].add(-2.0e-2)
        Vi = Vi.at[0].add(2.5e-2).at[-1].add(-1.0e-2)
        Ve = Ve.at[0].add(-2.0e-2).at[-1].add(1.5e-2)
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        backward_kind = jnp.zeros_like(
            geometry.maps.backward.endpoint_kind
        ).at[0].set(2)
        forward_kind = jnp.zeros_like(
            geometry.maps.forward.endpoint_kind
        ).at[-1].set(2)
        geometry = replace(
            geometry,
            maps=replace(
                geometry.maps,
                backward=replace(
                    geometry.maps.backward, endpoint_kind=backward_kind
                ),
                forward=replace(
                    geometry.maps.forward, endpoint_kind=forward_kind
                ),
            ),
        )
        base_context = replace(
            context,
            parameters=replace(
                context.parameters,
                parallel_characteristic_wall_law="primitive-least-residual",
            ),
        )
        common = dict(
            parallel_operator_scheme="fci",
            parallel_material_scheme="production-path",
            parallel_flux_pairing="support-core",
            parallel_boundary_pairing="characteristic-sat",
            parallel_short_leg_selection="cfl",
            parallel_short_leg_cfl_limit=1.0e-12,
        )
        base = replace(
            _build_rhs(base_context, local, geometry),
            parameters=replace(
                base_context.parameters,
                parallel_characteristic_wall_law="energy-absorbing",
            ),
            **common,
        )
        unsplit = replace(base, parallel_short_leg_treatment="explicit")
        split = replace(
            base, parallel_short_leg_treatment="local-backward-euler"
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        full_rhs = unsplit.evaluate_stage(state, phi_owned=phi)
        explicit_rhs = split.evaluate_stage(
            state, phi_owned=phi, short_leg_selection_dt=1.0
        )
        _updated, _increment, info = split.apply_short_leg_implicit_material_step(
            state,
            solve_dt=1.0e-3,
            selection_dt=1.0,
            phi_owned=phi,
            return_increment=True,
        )
        selected = info["selected_complete_residual_owner"]
        recovered = explicit_rhs.replace(
            density=explicit_rhs.density + selected[..., 0],
            Te=explicit_rhs.Te + selected[..., 1],
            Ti=explicit_rhs.Ti + selected[..., 2],
            Vi=explicit_rhs.Vi + selected[..., 3],
            Ve=explicit_rhs.Ve + selected[..., 4],
        )
        errors = jnp.stack(tuple(
            jnp.max(jnp.abs(actual - expected))
            for (_, actual), (_, expected) in zip(
                recovered.field_items(), full_rhs.field_items(), strict=True
            )
        ))
        selected_count = jnp.count_nonzero(info["selected_wall"])
        selected_force = jnp.max(jnp.abs(info["selected_coupled_force"]))
        return errors, selected_count, selected_force

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 9,
            out_specs=(P(), P(), P()),
            check_vma=False,
        )
    )
    errors, selected_count, selected_force = compiled(
        *fields, cell_fields, map_fields
    )
    np.testing.assert_allclose(np.asarray(errors), 0.0, atol=2.0e-10)
    assert int(np.asarray(selected_count)) > 0
    assert float(np.asarray(selected_force)) > 0.0
