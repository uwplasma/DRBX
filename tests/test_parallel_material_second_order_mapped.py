"""Small real-map acceptance tests for second-order material transport.

The geometry is synthetic but uses the production fractional-map payload and
the production map lowering/stencil builder.  The canonical ``maps`` member
is retained for the ordinary FCI operators; only ``material_maps`` is replaced
by the prescribed fractional shear used by these tests.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_drb_EB_rhs import (  # noqa: E402
    build_local_fci_drb_eb_operator_boundary_bundle,
)
from drbx.native.fci_parallel_production_flux import (  # noqa: E402
    parallel_characteristic_matrix,
    parallel_target_row_material_residual,
)
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_local_fci_geometry,
    build_local_fci_geometries,
    make_shard_mesh,
)
from fci_drb_eb_test_helpers import (  # noqa: E402
    _build_rhs,
)
from shifted_torus_eb_mms_data import (  # noqa: E402
    build_shifted_torus_eb_mms_context,
)


SHEAR = 0.37
TAU = 1.0
MU = 1836.0
PHASE_ETA = 1.0


def _fractional_shear_geometry(
    context, *, shear: float, variable_b: bool = False, constant_b: bool = False
):
    """Return a geometry whose material map is eta+/-1 with theta shear."""

    base = context.geometry
    nx, ny, nz = base.shape
    ii, jj, kk = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),
        np.arange(nz, dtype=float),
        indexing="ij",
    )
    y = np.asarray(base.grid.y_centers)
    z = np.asarray(base.grid.z_centers)
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    ds = float(np.sqrt((SHEAR * dy) ** 2 + dz**2))
    phase = jnp.asarray(y)[None, :, None] + jnp.asarray(z)[None, None, :]
    b_center = jnp.exp(0.2 * jnp.cos(phase))
    phase_shift = SHEAR * dy + dz
    b_forward = jnp.exp(0.2 * jnp.cos(phase + phase_shift))
    b_backward = jnp.exp(0.2 * jnp.cos(phase - phase_shift))
    zeros = jnp.zeros(base.shape, dtype=bool)
    ones = jnp.ones(base.shape, dtype=jnp.float64)
    custom_maps = replace(
        base.maps,
        forward_x=jnp.asarray(ii),
        forward_y=jnp.asarray(jj + shear),
        backward_x=jnp.asarray(ii),
        backward_y=jnp.asarray(jj - shear),
        forward_endpoint_x=jnp.asarray(y[0] * 0.0 + base.grid.x_centers[ii.astype(int)]),
        forward_endpoint_y=jnp.asarray(y[jj.astype(int)] + shear * dy),
        forward_endpoint_z=jnp.asarray(z[kk.astype(int)] + dz),
        backward_endpoint_x=jnp.asarray(y[0] * 0.0 + base.grid.x_centers[ii.astype(int)]),
        backward_endpoint_y=jnp.asarray(y[jj.astype(int)] - shear * dy),
        backward_endpoint_z=jnp.asarray(z[kk.astype(int)] - dz),
        forward_endpoint_bmag=jnp.broadcast_to(b_forward, base.shape),
        backward_endpoint_bmag=jnp.broadcast_to(b_backward, base.shape),
        forward_length=jnp.full(base.shape, ds),
        backward_length=jnp.full(base.shape, ds),
        forward_boundary=zeros,
        backward_boundary=zeros,
    )
    geometry = replace(base, maps=custom_maps)
    if variable_b or constant_b:
        b_value = (
            jnp.broadcast_to(b_center, base.shape)
            if variable_b
            else jnp.ones(base.shape, dtype=jnp.float64)
        )
        endpoint_value = (
            (jnp.broadcast_to(b_forward, base.shape), jnp.broadcast_to(b_backward, base.shape))
            if variable_b
            else (jnp.ones(base.shape, dtype=jnp.float64), jnp.ones(base.shape, dtype=jnp.float64))
        )
        geometry = replace(
            geometry,
            maps=replace(
                custom_maps,
                forward_endpoint_bmag=endpoint_value[0],
                backward_endpoint_bmag=endpoint_value[1],
            ),
            cell_bfield=replace(
                base.cell_bfield,
                Bmag=b_value,
            ),
        )
    return geometry


def _smooth_fields(geometry, *, constant: bool = False):
    """Five-field wave with an analytic derivative along the sheared map."""

    y = np.asarray(geometry.grid.y_centers)
    z = np.asarray(geometry.grid.z_centers)
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    theta = jnp.asarray(y)[None, :, None]
    eta = jnp.asarray(z)[None, None, :]
    phase = theta + PHASE_ETA * eta
    if constant:
        constant_state = jnp.asarray((2.0, 3.0, 5.0, 0.2, -0.1))
        return (
            jnp.broadcast_to(constant_state, geometry.shape + (5,)),
            jnp.zeros(geometry.shape + (5,), dtype=jnp.float64),
        )
    fields = (
        2.0 + 0.08 * jnp.sin(phase),
        3.0 + 0.06 * jnp.cos(phase + 0.2),
        5.0 + 0.07 * jnp.sin(2.0 * phase),
        0.15 + 0.04 * jnp.cos(phase - 0.1),
        -0.08 + 0.03 * jnp.sin(phase + 0.4),
    )
    state = jnp.stack(
        tuple(jnp.broadcast_to(value, geometry.shape) for value in fields),
        axis=-1,
    )
    directional_phase = (SHEAR * dy + PHASE_ETA * dz) / np.sqrt(
        (SHEAR * dy) ** 2 + dz**2
    )
    derivative = jnp.broadcast_to(
        jnp.stack(
            (
                0.08 * jnp.cos(phase) * directional_phase,
                -0.06 * jnp.sin(phase + 0.2) * directional_phase,
                0.14 * jnp.cos(2.0 * phase) * directional_phase,
                -0.04 * jnp.sin(phase - 0.1) * directional_phase,
                0.03 * jnp.cos(phase + 0.4) * directional_phase,
            ),
            axis=-1,
        ),
        geometry.shape + (5,),
    )
    return state, derivative


def _synthetic_runtime_case(
    n: int,
    *,
    eta_shards: int = 1,
    variable_b: bool = False,
    constant_state: bool = False,
    constant_b: bool = False,
):
    shape = (3, n, n)
    context = build_shifted_torus_eb_mms_context(shape)
    prod_context = replace(
        context,
        parameters=replace(
            context.parameters,
            tau=TAU,
            mi_over_me=MU,
            parallel_characteristic_wall_law="physical-boundary-state",
        ),
    )
    custom_global = _fractional_shear_geometry(
        prod_context, shear=SHEAR, variable_b=variable_b, constant_b=constant_b
    )
    mesh = make_shard_mesh((1, 1, eta_shards))
    original = build_local_fci_geometries(
        custom_global,
        (1, 1, eta_shards),
        halo_width=2,
        periodic_axes=(False, True, True),
    )
    custom = build_local_fci_geometries(
        custom_global,
        (1, 1, eta_shards),
        halo_width=2,
        periodic_axes=(False, True, True),
    )
    partition = P("x", "y", "z")
    state_fields, derivative = _smooth_fields(
        custom_global, constant=constant_state
    )
    return prod_context, original, custom, state_fields, derivative, mesh, partition


def _material_residual(case, *, spatial_order: int, use_material_div_b: bool = False):
    context, original, custom, state_fields, derivative, mesh, partition = case

    def kernel(density, te, ti, vi, ve, phi, vorticity, dstate, cells, maps, material_maps):
        canonical = assemble_local_fci_geometry(original, cells, maps)
        material = assemble_local_fci_geometry(custom, cells, material_maps)
        # Keep canonical maps for current/phi and install only the dedicated
        # material map lowered from the fractional shear payload.
        geometry = replace(canonical, material_maps=material.material_maps)
        builder_context = replace(
            context,
            parameters=replace(
                context.parameters,
                parallel_characteristic_wall_law="energy-absorbing",
            ),
        )
        rhs = replace(
            _build_rhs(builder_context, original, geometry),
            parallel_operator_scheme="fci",
            parallel_flux_pairing="support-core",
            parallel_material_scheme="production-path",
            parallel_boundary_pairing="characteristic-sat",
            parameters=context.parameters,
        )
        state = FciDrbEBState(density, phi, te, ti, vi, ve, vorticity)
        face_bc = rhs._face_bcs(state)
        state_halo = rhs._prepare_state_halo(state, face_bc)
        operator_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
            state_halo, geometry, rhs.domain, face_bc, tau=rhs.parameters.tau
        )
        parallel_boundary = rhs._parallel_operator_boundary(
            state_halo=state_halo, operator_boundary=operator_boundary
        )
        stencil_context = rhs._stencil_builder_context()
        characteristic_data = rhs._fci_parallel_characteristic_wall_data(
            state_halo=state_halo,
            face_bc=face_bc,
            parallel_boundary=parallel_boundary,
            context=stencil_context,
            evaluate_wall_data=False,
        )
        second = rhs._fci_second_order_material_data(
            characteristic_data, face_bc, stencil_context
        )
        dx_minus = characteristic_data["primitive_stencils"][0].dx_min
        dx_plus = characteristic_data["primitive_stencils"][0].dx_plus
        residual, diagnostics = parallel_target_row_material_residual(
            second["center"], second["minus"], second["plus"],
            dx_minus, dx_plus, TAU, MU,
            spatial_order=spatial_order,
            minus2=second["minus2"], plus2=second["plus2"],
            dx_minus2=second["dx_minus2"], dx_plus2=second["dx_plus2"],
            backward_second_valid=second["backward_second_valid"],
            forward_second_valid=second["forward_second_valid"],
            backward_centered_closure=second["backward_centered_closure"],
            forward_centered_closure=second["forward_centered_closure"],
            div_b=(
                second["material_div_b"]
                if use_material_div_b
                else jnp.zeros_like(dx_minus)
            ),
        )
        matrix = jax.vmap(
            lambda value: parallel_characteristic_matrix(*value, tau=TAU, mu=MU)
        )(second["center"].reshape((-1, 5)))
        exact = -jnp.einsum(
            "nij,nj->ni", matrix, dstate.reshape((-1, 5))
        ).reshape(second["center"].shape)
        if use_material_div_b:
            density, te, ti, vi, ve = [second["center"][..., i] for i in range(5)]
            current = density * (vi - ve)
            div_b = second["material_div_b"]
            exact = exact + jnp.stack(
                (
                    -density * ve * div_b,
                    2.0 * te / (3.0 * density)
                    * (0.71 * current - density * ve)
                    * div_b,
                    2.0 * ti / (3.0 * density)
                    * (current - density * vi)
                    * div_b,
                    jnp.zeros_like(div_b),
                    jnp.zeros_like(div_b),
                ),
                axis=-1,
            )
        return (
            residual,
            exact,
            diagnostics["ordinary_row"],
            second["backward_second_valid"],
            second["forward_second_valid"],
            second["material_div_b"],
            second["material_div_b_valid"],
            second["material_div_b_fallback"],
        )

    compiled = jax.jit(jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(partition,) * 11,
        out_specs=(partition, partition, partition, partition, partition, partition, partition, partition),
        check_vma=False,
    ))
    fields = tuple(jnp.asarray(state_fields[..., index]) for index in range(5))
    zeros = jnp.zeros(state_fields.shape[:-1], dtype=jnp.float64)
    return tuple(
        np.asarray(value)
        for value in compiled(
            *fields,
            zeros,
            zeros,
            jnp.asarray(derivative),
            original.cell_fields,
            original.map_fields,
            custom.map_fields,
        )
    )


def test_real_material_map_helper_and_second_order_kernel():
    errors = []
    field_errors = []
    for n in (16, 32, 64):
        case = _synthetic_runtime_case(n)
        residual, exact, ordinary_row, backward_valid, forward_valid, _div_b, _div_valid, _div_fallback = _material_residual(
            case, spatial_order=2
        )
        assert bool(np.all(np.asarray(backward_valid)))
        assert bool(np.all(np.asarray(forward_valid)))
        assert bool(np.all(np.asarray(ordinary_row)))
        assert bool(np.all(np.isfinite(residual)))
        errors.append(np.linalg.norm(residual - exact) / np.linalg.norm(exact))
        error_flat = (residual - exact).reshape((-1, 5))
        exact_flat = exact.reshape((-1, 5))
        field_errors.append(
            np.linalg.norm(error_flat, axis=0) / np.linalg.norm(exact_flat, axis=0)
        )
    observed = np.log2(errors[-2] / errors[-1])
    field_observed = np.log2(field_errors[-2] / field_errors[-1])
    assert observed >= 1.8, (errors, observed, field_errors, field_observed)
    assert np.all(field_observed >= 1.8), (field_errors, field_observed)


def test_material_map_keeps_canonical_fci_maps_unchanged():
    case = _synthetic_runtime_case(8)
    _context, canonical, _material, *_ = case
    before = np.asarray(canonical.map_fields).copy()
    _material_residual(case, spatial_order=2)
    np.testing.assert_array_equal(np.asarray(canonical.map_fields), before)


def test_variable_b_material_div_b_uses_raw_metric_and_converges():
    errors = []
    source_errors = []
    div_errors = []
    for n in (16, 32, 64):
        case = _synthetic_runtime_case(n, variable_b=True, constant_state=True)
        residual, exact, ordinary_row, backward_valid, forward_valid, material_div_b, div_valid, div_fallback = _material_residual(
            case, spatial_order=2, use_material_div_b=True
        )
        assert bool(np.all(np.asarray(ordinary_row)))
        assert bool(np.all(np.asarray(backward_valid)))
        assert bool(np.all(np.asarray(forward_valid)))
        assert bool(np.all(np.asarray(div_valid)))
        assert not bool(np.any(np.asarray(div_fallback)))
        assert bool(np.all(np.isfinite(residual)))
        context, _geometry, _material, _state, _derivative, *_ = case
        y = np.asarray(context.geometry.grid.y_centers)
        z = np.asarray(context.geometry.grid.z_centers)
        phase = y[None, :, None] + z[None, None, :]
        dy = float(y[1] - y[0])
        dz = float(z[1] - z[0])
        directional_phase = (SHEAR * dy + dz) / np.sqrt((SHEAR * dy) ** 2 + dz**2)
        expected_div_b = 0.2 * np.sin(phase) * directional_phase
        div_errors.append(
            np.linalg.norm(np.asarray(material_div_b) - expected_div_b)
            / np.linalg.norm(expected_div_b)
        )
        div = np.broadcast_to(expected_div_b, np.asarray(material_div_b).shape)
        expected_source = np.stack(
            (0.2 * div, 0.626 * div, (1.0 / 3.0) * div,
             np.zeros_like(div), np.zeros_like(div)),
            axis=-1,
        )
        source_delta = np.asarray(residual) - expected_source
        source_errors.append(
            np.linalg.norm(source_delta.reshape((-1, 5)), axis=0)
            / np.maximum(np.linalg.norm(expected_source.reshape((-1, 5)), axis=0), 1.0)
        )
    div_order = np.log2(div_errors[-2] / div_errors[-1])
    source_orders = np.log2(source_errors[-2] / source_errors[-1])
    assert div_order >= 1.8, (div_errors, div_order)
    assert np.all(source_orders[:3] >= 1.8), (source_errors, source_orders)
    assert np.all(np.asarray(source_errors)[-1, 3:] <= 1.0e-8)


def test_constant_b_material_div_b_is_zero_and_valid():
    case = _synthetic_runtime_case(
        16, variable_b=False, constant_b=True, constant_state=True
    )
    residual, exact, ordinary_row, backward_valid, forward_valid, material_div_b, div_valid, div_fallback = _material_residual(
        case, spatial_order=2, use_material_div_b=True
    )
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1.0e-11)
    np.testing.assert_allclose(exact, 0.0, rtol=0.0, atol=1.0e-11)
    np.testing.assert_allclose(material_div_b, 0.0, rtol=0.0, atol=1.0e-11)
    assert bool(np.all(np.asarray(div_valid)))
    assert not bool(np.any(np.asarray(div_fallback)))
    assert bool(np.all(np.asarray(ordinary_row)))
    assert bool(np.all(np.asarray(backward_valid)))
    assert bool(np.all(np.asarray(forward_valid)))


@pytest.mark.skipif(jax.local_device_count() < 4, reason="requires four local CPU devices")
def test_material_map_four_eta_shards_matches_single_device():
    single = _material_residual(_synthetic_runtime_case(16), spatial_order=2)
    sharded = _material_residual(
        _synthetic_runtime_case(16, eta_shards=4), spatial_order=2
    )
    for expected, actual in zip(single, sharded):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2.0e-12, atol=2.0e-12)
