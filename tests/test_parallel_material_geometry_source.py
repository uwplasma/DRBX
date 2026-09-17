"""RLP regression for the raw-metric material ``div(b)`` source."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.geometry.fci_control_volumes import (  # noqa: E402
    build_polar_angular_agglomeration_geometry,
)
from drbx.geometry import FCI_DEP_PHYSICAL_BOUNDARY  # noqa: E402
from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_drb_EB_rhs import (  # noqa: E402
    RHS_TERM_NAMES,
    build_local_fci_drb_eb_operator_boundary_bundle,
)
from drbx.native.fci_boundaries import (  # noqa: E402
    LocalControlVolumeBoundaryBC3D,
)
from drbx.native.fci_operators import (  # noqa: E402
    aggregate_local_control_volume_average,
)
from drbx.native.fci_parallel_production_flux import (  # noqa: E402
    parallel_vorticity_second_order_upwind_residual,
)
from drbx.native.fci_angular_agglomeration import (  # noqa: E402
    assemble_local_polar_angular_agglomeration_geometry,
    build_sharded_polar_angular_agglomeration_payload,
)
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_single_device_local_fci_geometry,
    build_local_fci_geometries,
)
from fci_drb_eb_test_helpers import _build_rhs  # noqa: E402
from shifted_torus_eb_mms_data import (  # noqa: E402
    build_shifted_torus_eb_mms_context,
)


SHEAR = 0.37


def _mark_forward_eta_wall(maps):
    return replace(
        maps,
        forward=replace(
            maps.forward,
            endpoint_kind=maps.forward.endpoint_kind.at[..., -1].set(
                FCI_DEP_PHYSICAL_BOUNDARY
            ),
            target_valid=maps.forward.target_valid.at[..., -1].set(True),
        ),
    )


def _invalidate_material_map_targets(maps):
    return replace(
        maps,
        forward=replace(
            maps.forward,
            target_valid=jnp.zeros_like(maps.forward.target_valid),
        ),
        backward=replace(
            maps.backward,
            target_valid=jnp.zeros_like(maps.backward.target_valid),
        ),
    )


def _material_source_case(
    n: int = 16,
    *,
    radial_count: int = 3,
    rlp_jacobian=None,
    angular_group_size=None,
):
    shape = (int(radial_count), n, n)
    context = build_shifted_torus_eb_mms_context(shape)
    context = replace(
        context,
        parameters=replace(
            context.parameters,
            parallel_characteristic_wall_law="energy-absorbing",
        ),
    )
    base = context.geometry
    ii, jj, kk = np.meshgrid(
        np.arange(shape[0], dtype=float),
        np.arange(n, dtype=float),
        np.arange(n, dtype=float),
        indexing="ij",
    )
    y = np.asarray(base.grid.y_centers)
    z = np.asarray(base.grid.z_centers)
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    ds = float(np.hypot(SHEAR * dy, dz))
    phase = jnp.asarray(y)[None, :, None] + jnp.asarray(z)[None, None, :]
    B = jnp.broadcast_to(jnp.exp(0.2 * jnp.cos(phase)), shape)
    phase_step = SHEAR * dy + dz
    B_forward = jnp.broadcast_to(
        jnp.exp(0.2 * jnp.cos(phase + phase_step)), shape
    )
    B_backward = jnp.broadcast_to(
        jnp.exp(0.2 * jnp.cos(phase - phase_step)), shape
    )
    zeros = jnp.zeros(shape, dtype=bool)
    maps = replace(
        base.maps,
        forward_x=jnp.asarray(ii),
        forward_y=jnp.asarray(jj + SHEAR),
        backward_x=jnp.asarray(ii),
        backward_y=jnp.asarray(jj - SHEAR),
        forward_endpoint_x=jnp.asarray(base.grid.x_centers[ii.astype(int)]),
        forward_endpoint_y=jnp.asarray(y[jj.astype(int)] + SHEAR * dy),
        forward_endpoint_z=jnp.asarray(z[kk.astype(int)] + dz),
        backward_endpoint_x=jnp.asarray(base.grid.x_centers[ii.astype(int)]),
        backward_endpoint_y=jnp.asarray(y[jj.astype(int)] - SHEAR * dy),
        backward_endpoint_z=jnp.asarray(z[kk.astype(int)] - dz),
        forward_endpoint_bmag=B_forward,
        backward_endpoint_bmag=B_backward,
        forward_length=jnp.full(shape, ds),
        backward_length=jnp.full(shape, ds),
        forward_boundary=zeros,
        backward_boundary=zeros,
    )
    global_geometry = replace(
        base,
        maps=maps,
        cell_bfield=replace(base.cell_bfield, Bmag=B),
    )
    sharded = build_local_fci_geometries(
        global_geometry,
        (1, 1, 1),
        halo_width=2,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )
    local = assemble_single_device_local_fci_geometry(sharded)
    host_sharded = replace(
        sharded,
        domain=replace(sharded.domain, mesh_axis_names=(None, None, None)),
    )
    rhs = _build_rhs(context, host_sharded, local)

    if rlp_jacobian is None:
        rlp_jacobian = lambda points: np.maximum(
            np.asarray(points)[..., 0], 1.0e-14
        )
    host_rlp = build_polar_angular_agglomeration_geometry(
        np.asarray(base.grid.x_faces),
        np.asarray(base.grid.y_faces),
        np.asarray(base.grid.z_faces),
        rlp_jacobian,
        quadrature_order=3,
        angular_group_size=(
            angular_group_size
            if angular_group_size is not None
            else ((n, 2, 1) if radial_count == 3 else None)
        ),
    )
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host_rlp, host_sharded.domain
    )
    local_rlp = assemble_local_polar_angular_agglomeration_geometry(
        descriptor, packed, local
    )
    rhs_rlp = replace(
        rhs,
        control_volume_geometry=local_rlp,
        control_volume_boundary_bc=LocalControlVolumeBoundaryBC3D.empty(),
        poisson_bracket_scheme="compatible-flux",
        axis_regular_axes=(True, False, False),
    )

    one = jnp.ones(shape, dtype=jnp.float64)
    zero = jnp.zeros(shape, dtype=jnp.float64)
    state = FciDrbEBState(one, zero, one, one, zero, zero, zero)
    face_bc = rhs._face_bcs(state)
    stencil_context = rhs._stencil_builder_context()
    exact = np.broadcast_to(
        0.2 * np.sin(np.asarray(phase)) * phase_step / ds,
        shape,
    )
    return rhs, rhs_rlp, face_bc, stencil_context, exact


def test_material_geometry_source_ignores_rlp_owner_reconstruction():
    rhs, rhs_rlp, face_bc, context, exact = _material_source_case()
    fallback = jnp.full(rhs.geometry.owned_shape, -17.0, dtype=jnp.float64)
    plain = rhs._fci_second_order_material_div_b(
        face_bc, context, fallback_div_b=fallback
    )
    with_rlp = rhs_rlp._fci_second_order_material_div_b(
        face_bc, context, fallback_div_b=fallback
    )

    np.testing.assert_array_equal(
        np.asarray(with_rlp["material_div_b"]),
        np.asarray(plain["material_div_b"]),
    )
    assert bool(np.all(np.asarray(with_rlp["material_div_b_valid"])))
    assert not bool(np.any(np.asarray(with_rlp["material_div_b_fallback"])))
    relative_error = np.linalg.norm(np.asarray(with_rlp["material_div_b"]) - exact)
    relative_error /= np.linalg.norm(exact)
    assert relative_error < 0.06


def test_material_geometry_source_invalid_raw_donor_uses_fallback():
    _rhs, rhs_rlp, face_bc, context, _exact = _material_source_case(8)
    local = rhs_rlp.geometry
    owned = local.cell_bfield.owned_slices_in_halo
    bad_B = np.asarray(local.cell_bfield.Bmag_owned).copy()
    bad_B[0, 0, 0] = np.nan
    bad_bfield = replace(
        local.cell_bfield,
        Bmag_halo=local.cell_bfield.Bmag_halo.at[owned].set(jnp.asarray(bad_B)),
    )
    rhs_bad = replace(rhs_rlp, geometry=replace(local, cell_bfield=bad_bfield))
    fallback = jnp.full(local.owned_shape, 23.0, dtype=jnp.float64)
    result = rhs_bad._fci_second_order_material_div_b(
        face_bc, context, fallback_div_b=fallback
    )
    mask = np.asarray(result["material_div_b_fallback"])
    assert int(np.count_nonzero(mask)) > 1
    np.testing.assert_array_equal(
        np.asarray(result["material_div_b"])[mask],
        np.full(np.count_nonzero(mask), 23.0),
    )

    raw_result = replace(
        rhs_bad,
        parallel_material_div_b_fallback_scheme="raw-metric",
    )._fci_second_order_material_div_b(
        face_bc, context, fallback_div_b=fallback
    )
    raw_used = np.asarray(raw_result["material_div_b_raw_fallback_used"])
    legacy_used = np.asarray(raw_result["material_div_b_legacy_fallback_used"])
    assert bool(np.any(raw_used))
    assert bool(np.any(legacy_used))
    assert not bool(np.any(raw_used & legacy_used))
    np.testing.assert_array_equal(
        np.asarray(raw_result["material_div_b"])[legacy_used],
        np.full(np.count_nonzero(legacy_used), 23.0),
    )


def test_raw_metric_div_b_fallback_converges_without_owner_projection():
    errors = []
    for n in (16, 32):
        _plain, rhs, face_bc, context, exact = _material_source_case(n)
        rhs = replace(
            rhs,
            geometry=replace(
                rhs.geometry,
                material_maps=_invalidate_material_map_targets(
                    rhs.geometry.material_maps
                ),
            ),
            parallel_material_div_b_fallback_scheme="raw-metric",
        )
        legacy = jnp.full(rhs.geometry.owned_shape, -17.0, dtype=jnp.float64)
        result = rhs._fci_second_order_material_div_b(
            face_bc,
            context,
            fallback_div_b=legacy,
        )
        assert not bool(np.any(np.asarray(result["material_div_b_valid"])))
        assert bool(
            np.all(np.asarray(result["material_div_b_raw_fallback_used"]))
        )
        assert not bool(
            np.any(np.asarray(result["material_div_b_legacy_fallback_used"]))
        )
        value = np.asarray(result["material_div_b"])
        assert bool(np.all(np.isfinite(value)))
        assert not bool(np.any(value == -17.0))
        errors.append(np.linalg.norm(value - exact) / np.linalg.norm(exact))
    observed = np.log2(errors[0] / errors[1])
    assert observed >= 1.8, (errors, observed)


def test_raw_metric_div_b_fallback_uses_physical_endpoint_bmag():
    _plain, rhs, face_bc, context, _exact = _material_source_case(8)
    material_maps = _invalidate_material_map_targets(rhs.geometry.material_maps)
    canonical_maps = _mark_forward_eta_wall(rhs.geometry.maps)

    def evaluate(endpoint_bmag: float):
        forward = canonical_maps.forward
        wall_bmag = forward.endpoint_bmag.at[..., -1].set(endpoint_bmag)
        geometry = replace(
            rhs.geometry,
            maps=replace(
                canonical_maps,
                forward=replace(forward, endpoint_bmag=wall_bmag),
            ),
            material_maps=material_maps,
        )
        result = replace(
            rhs,
            geometry=geometry,
            parallel_material_div_b_fallback_scheme="raw-metric",
        )._fci_second_order_material_div_b(
            face_bc,
            context,
            fallback_div_b=jnp.full(geometry.owned_shape, -17.0),
        )
        assert bool(
            np.all(np.asarray(result["material_div_b_raw_fallback_used"]))
        )
        assert not bool(
            np.any(np.asarray(result["material_div_b_legacy_fallback_used"]))
        )
        return np.asarray(result["material_div_b"]), geometry

    value_2, geometry = evaluate(2.0)
    value_4, _ = evaluate(4.0)
    wall = np.zeros(geometry.owned_shape, dtype=bool)
    wall[..., -1] = True
    np.testing.assert_array_equal(value_4[~wall], value_2[~wall])

    B0 = np.asarray(geometry.cell_bfield.Bmag_owned)
    hm = np.asarray(geometry.maps.backward.connection_length)
    hp = np.asarray(geometry.maps.forward.connection_length)
    expected_delta = B0 * hm / (hp * (hm + hp)) * (1.0 / 4.0 - 1.0 / 2.0)
    np.testing.assert_allclose(
        value_4[wall] - value_2[wall],
        expected_delta[wall],
        rtol=2.0e-13,
        atol=2.0e-13,
    )


@pytest.mark.parametrize("ion_speed", (0.17, -0.17))
def test_h_mf_vorticity_wall_uses_directional_second_hop_or_h_first_order(
    ion_speed,
):
    """A downwind wall must not force the selected H/MF lane through P."""

    _plain, rhs, _face_bc, _context, _exact = _material_source_case(8)
    geometry = replace(
        rhs.geometry,
        maps=_mark_forward_eta_wall(rhs.geometry.maps),
        material_maps=_mark_forward_eta_wall(rhs.geometry.material_maps),
    )
    rhs = replace(
        rhs,
        geometry=geometry,
        parallel_operator_scheme="fci",
        parallel_flux_pairing="support-core",
        parallel_material_scheme="production-path",
        parallel_boundary_pairing="characteristic-sat",
        parallel_vorticity_advection_scheme="h-mf-second-order",
        parallel_material_fallback_representation="h-mf-consistent",
        parallel_material_div_b_fallback_scheme="raw-metric",
    )
    shape = geometry.owned_shape
    theta = jnp.asarray(geometry.grid.y_centers_owned)[None, :, None]
    eta = jnp.asarray(geometry.grid.z_centers_owned)[None, None, :]
    phase = theta + 0.7 * eta

    def fine(value):
        return jnp.broadcast_to(value, shape)

    raw = FciDrbEBState(
        density=fine(2.0 + 0.04 * jnp.sin(phase)),
        phi=fine(0.1 + 0.01 * jnp.cos(phase)),
        Te=fine(3.0 + 0.03 * jnp.cos(phase + 0.2)),
        Ti=fine(5.0 + 0.02 * jnp.sin(2.0 * phase)),
        Vi=fine(ion_speed),
        Ve=fine(-0.03),
        vorticity=fine(0.2 + 0.07 * jnp.sin(phase - 0.1)),
    )
    cells = rhs.control_volume_geometry.cells
    state = raw.map_fields(
        lambda value: aggregate_local_control_volume_average(
            value, cells, rhs.domain
        )
    )
    face_bc = rhs._face_bcs(state)
    state_halo = rhs._prepare_state_halo(state, face_bc)
    operator_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
        state_halo,
        geometry,
        rhs.domain,
        face_bc,
        tau=rhs.parameters.tau,
    )
    parallel_boundary = rhs._parallel_operator_boundary(
        state_halo=state_halo,
        operator_boundary=operator_boundary,
    )
    context = rhs._stencil_builder_context()
    terms = rhs._fci_parallel_terms(
        state_halo=state_halo,
        face_bc=face_bc,
        operator_boundary=operator_boundary,
        parallel_boundary=parallel_boundary,
        context=context,
    )
    diagnostics = terms["parallel_material_diagnostics"]

    backward_valid = np.asarray(
        diagnostics["vorticity_backward_second_valid"]
    )
    forward_valid = np.asarray(
        diagnostics["vorticity_forward_second_valid"]
    )
    second_used = np.asarray(diagnostics["vorticity_h_second_order_used"])
    h_first_used = np.asarray(
        diagnostics["vorticity_h_first_order_fallback_used"]
    )
    p_used = np.asarray(diagnostics["vorticity_legacy_p_fallback_used"])
    selected_valid = backward_valid if ion_speed > 0.0 else forward_valid
    np.testing.assert_array_equal(second_used, selected_valid)
    np.testing.assert_array_equal(h_first_used, ~selected_valid)
    assert not bool(np.any(p_used))
    wall = np.zeros(shape, dtype=bool)
    wall[..., -1] = True
    assert not bool(np.any(forward_valid[wall]))
    if ion_speed > 0.0:
        assert bool(np.all(second_used[wall]))
        assert not bool(np.any(h_first_used[wall]))
    else:
        assert not bool(np.any(second_used[wall]))
        assert bool(np.all(h_first_used[wall]))

    characteristic = rhs._fci_parallel_characteristic_wall_data(
        state_halo=state_halo,
        face_bc=face_bc,
        parallel_boundary=parallel_boundary,
        context=context,
        evaluate_wall_data=True,
    )
    material = rhs._fci_second_order_material_data(
        characteristic,
        face_bc,
        context,
    )
    vorticity = rhs._fci_second_order_vorticity_data(
        state.vorticity,
        face_bc,
        context,
        material,
    )
    expected = parallel_vorticity_second_order_upwind_residual(
        vorticity["center"],
        vorticity["minus"],
        vorticity["plus"],
        vorticity["minus2"],
        vorticity["plus2"],
        material["center"][..., 3],
        vorticity["dx_minus"],
        vorticity["dx_plus"],
        vorticity["dx_minus2"],
        vorticity["dx_plus2"],
        backward_second_valid=vorticity["backward_second_valid"],
        forward_second_valid=vorticity["forward_second_valid"],
    )
    np.testing.assert_allclose(
        np.asarray(terms["vorticity_parallel_advection"]),
        np.asarray(expected),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


@pytest.mark.slow
def test_h_mf_rlp_wall_local_be_restricts_only_selected_active_owners():
    """Exercise the repaired wall/implicit representation on angular owners."""

    _plain, rhs, _face_bc, _context, _exact = _material_source_case(8)
    geometry = replace(
        rhs.geometry,
        maps=_mark_forward_eta_wall(rhs.geometry.maps),
        material_maps=_mark_forward_eta_wall(rhs.geometry.material_maps),
    )
    rhs = replace(
        rhs,
        geometry=geometry,
        parameters=replace(
            rhs.parameters,
            parallel_characteristic_wall_law="physical-boundary-state",
        ),
        physical_wall_model_name="no-flow",
        parallel_operator_scheme="fci",
        parallel_flux_pairing="support-core",
        parallel_material_scheme="production-path",
        parallel_boundary_pairing="characteristic-sat",
        parallel_material_fallback_representation="h-mf-consistent",
        parallel_short_leg_treatment="local-backward-euler",
        parallel_short_leg_selection="all-physical-walls",
    )
    shape = geometry.owned_shape
    theta = jnp.asarray(geometry.grid.y_centers_owned)[None, :, None]
    eta = jnp.asarray(geometry.grid.z_centers_owned)[None, None, :]
    phase = theta + eta

    def fine(value):
        return jnp.broadcast_to(value, shape)

    raw = FciDrbEBState(
        density=fine(2.0 + 0.08 * jnp.sin(phase)),
        phi=fine(0.1 + 0.02 * jnp.cos(phase)),
        Te=fine(3.0 + 0.06 * jnp.cos(phase + 0.2)),
        Ti=fine(5.0 + 0.07 * jnp.sin(2.0 * phase)),
        Vi=fine(0.04 * jnp.sin(eta)),
        Ve=fine(-0.03 * jnp.sin(eta)),
        vorticity=jnp.zeros(shape, dtype=jnp.float64),
    )
    cells = rhs.control_volume_geometry.cells
    state = raw.map_fields(
        lambda value: aggregate_local_control_volume_average(
            value, cells, rhs.domain
        )
    )
    dy = float(
        geometry.grid.y_centers_owned[1] - geometry.grid.y_centers_owned[0]
    )
    dz = float(
        geometry.grid.z_centers_owned[1] - geometry.grid.z_centers_owned[0]
    )
    solve_dt = float(np.hypot(SHEAR * dy, dz)) * 1.0e-8
    updated, increment, info = rhs.apply_short_leg_implicit_material_step(
        state,
        solve_dt=solve_dt,
        selection_dt=solve_dt,
        phi_owned=state.phi,
        return_increment=True,
    )

    increment_values = jnp.stack(
        tuple(
            getattr(increment, name)
            for name in ("density", "Te", "Ti", "Vi", "Ve")
        ),
        axis=-1,
    )
    complete = info["selected_complete_residual_owner"]
    active = np.asarray(cells.is_active_owner)
    assert int(np.count_nonzero(np.asarray(info["selected_wall"]))) == 3 * 8
    assert bool(np.all(np.asarray(info["implicit_finite"])))
    np.testing.assert_array_equal(np.asarray(increment_values)[~active], 0.0)
    np.testing.assert_array_equal(np.asarray(complete)[~active], 0.0)
    relative_tangent_defect = np.linalg.norm(
        np.asarray(increment_values / solve_dt - complete)[active]
    ) / np.linalg.norm(np.asarray(complete)[active])
    assert relative_tangent_defect < 1.0e-5
    np.testing.assert_array_equal(np.asarray(updated.phi), np.asarray(state.phi))
    np.testing.assert_array_equal(
        np.asarray(updated.vorticity), np.asarray(state.vorticity)
    )


def test_mms_counterfactual_payload_preserves_live_identities():
    _rhs, rhs, _face_bc, _context, _exact = _material_source_case(8)
    geometry = rhs.geometry
    cells = rhs.control_volume_geometry.cells
    domain = rhs.domain
    y = jnp.asarray(geometry.grid.y_centers_owned)[None, :, None]
    z = jnp.asarray(geometry.grid.z_centers_owned)[None, None, :]
    phase = y + 0.7 * z

    def fine(base, amplitude, shift=0.0):
        return jnp.broadcast_to(
            base + amplitude * jnp.sin(phase + shift),
            geometry.owned_shape,
        )

    raw = FciDrbEBState(
        density=fine(2.0, 0.08),
        phi=fine(0.1, 0.03, 0.2),
        Te=fine(3.0, 0.06, 0.4),
        Ti=fine(5.0, 0.07, 0.7),
        Vi=fine(0.15, 0.04, 0.1),
        Ve=fine(-0.08, 0.03, 0.5),
        vorticity=fine(0.02, 0.01, 0.9),
    )
    state = raw.map_fields(
        lambda value: aggregate_local_control_volume_average(
            value, cells, domain
        )
    )
    result = rhs.evaluate_stage(
        state,
        source_owned=state.zeros_like(),
        phi_owned=state.phi,
        return_rhs_term_fields=True,
        return_mms_counterfactual_fields=True,
        diagnostic_raw_state=raw,
    )
    _rhs_state, ledger, material, forces, poisson, generalized = result

    assert material.shape == (4, 5) + geometry.owned_shape
    assert forces.shape == (4,) + geometry.owned_shape
    assert poisson.shape == (4, 6) + geometry.owned_shape
    assert generalized.shape == (5,) + geometry.owned_shape
    assert bool(np.all(np.isfinite(np.asarray(material))))
    assert bool(np.all(np.isfinite(np.asarray(forces))))
    assert bool(np.all(np.isfinite(np.asarray(poisson))))
    assert bool(np.all(np.isfinite(np.asarray(generalized))))
    np.testing.assert_allclose(
        np.asarray(material[2]),
        np.asarray(material[0] - material[1]),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(forces[2]),
        np.asarray(forces[0] + forces[1]),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(poisson[3]),
        np.asarray(ledger[:, 0]),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(generalized[1]),
        np.asarray(generalized[3] + generalized[4]),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    electrostatic_slot = RHS_TERM_NAMES[4].index("electrostatic")
    np.testing.assert_allclose(
        np.asarray(generalized[2]),
        np.asarray(
            ledger[4, electrostatic_slot] / rhs.parameters.mi_over_me
        ),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
