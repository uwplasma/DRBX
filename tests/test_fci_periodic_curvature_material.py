"""Focused tests for periodic material traces in production curvature."""

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drbx.geometry import (  # noqa: E402
    LocalCurvatureFaceCoefficients3D,
    StencilBuilderContext,
    build_shifted_torus_geometry,
    build_local_conservative_stencil_from_field,
)
from drbx.native.fci_halo import (  # noqa: E402
    HaloExchange3D,
    LocalPeriodicTopologyRule3D,
    TopologyHaloFiller3D,
)
from drbx.native.fci_model import inject_owned_field_to_halo  # noqa: E402
from drbx.native.fci_operators import (  # noqa: E402
    _axis_face_samples_from_halo,
    build_periodic_curvature_endpoint_face_states,
    local_curvature_production_path_op,
)
from drbx.native.fci_curvature_production_flux import (  # noqa: E402
    reconstruct_third_order_face_states,
)
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_local_fci_geometry,
    assemble_single_device_local_fci_geometry,
    build_local_fci_geometries,
    make_shard_mesh,
)


def _close_fields(fields, geometry, domain, topology):
    exchange = HaloExchange3D()
    return tuple(
        topology(
            exchange(
                inject_owned_field_to_halo(field, geometry.layout), domain
            ),
            domain,
        )
        for field in fields
    )


def _fourier_owner_fields(geometry):
    shape = geometry.owned_shape
    ii, jj, kk = np.indices(shape)
    theta = 2.0 * np.pi * (jj + 0.5) / shape[1]
    eta = 2.0 * np.pi * (kk + 0.5) / shape[2]
    theta_average = np.sinc(2.0 / shape[1])
    eta_average = np.sinc(1.0 / shape[2])
    active = np.ones(shape, dtype=bool)
    fields = []
    for phase in (0.0, 0.31, -0.22, 0.47):
        field = (
            2.0
            + 0.17 * theta_average * np.cos(2.0 * theta + phase)
            + 0.11 * eta_average * np.sin(eta - phase)
        )
        fields.append(jnp.asarray(np.where(active, field, 0.0)))
    return tuple(fields)


def _fourier_face_values(shape, theta_face, eta_face=None):
    """Exact cell-average Fourier trace at a theta or eta face."""

    theta_centers = 2.0 * np.pi * (np.arange(shape[1]) + 0.5) / shape[1]
    eta_centers = 2.0 * np.pi * (np.arange(shape[2]) + 0.5) / shape[2]
    theta_average = np.sinc(2.0 / shape[1])
    eta_average = np.sinc(1.0 / shape[2])
    theta = theta_centers if theta_face is None else theta_face
    eta = eta_centers if eta_face is None else eta_face
    values = []
    for phase in (0.0, 0.31, -0.22, 0.47):
        values.append(
            2.0
            + 0.17 * (
                np.cos(2.0 * theta + phase)
                if theta_face is not None
                else theta_average * np.cos(2.0 * theta + phase)
            )
            + 0.11 * (
                np.sin(eta - phase)
                if eta_face is not None
                else eta_average * np.sin(eta - phase)
            )
        )
    return np.stack(values, axis=-1)


def _fourier_trace_errors(n):
    geometry = build_shifted_torus_geometry((2, n, n), construct_fci_maps=False)
    local = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=2)
    local_geometry = assemble_single_device_local_fci_geometry(local)
    domain = replace(local.domain, mesh_axis_names=(None, None, None))
    topology = TopologyHaloFiller3D(
        (LocalPeriodicTopologyRule3D((False, True, True)),)
    )
    endpoints = build_periodic_curvature_endpoint_face_states(
        _close_fields(_fourier_owner_fields(local_geometry), local_geometry, domain, topology),
        local_geometry,
        periodic_axes=domain.periodic_axes,
    )
    theta_lower = np.asarray(endpoints[1][0][0])
    theta_lower_right = np.asarray(endpoints[1][0][1])
    theta_upper = np.asarray(endpoints[1][1][0])
    theta_upper_right = np.asarray(endpoints[1][1][1])
    eta_lower = np.asarray(endpoints[2][0][0])
    eta_lower_right = np.asarray(endpoints[2][0][1])
    eta_upper = np.asarray(endpoints[2][1][0])
    eta_upper_right = np.asarray(endpoints[2][1][1])
    expected_theta_lower = _fourier_face_values((2, n, n), 0.0)
    expected_theta_upper = _fourier_face_values((2, n, n), 2.0 * np.pi)
    expected_eta_lower = _fourier_face_values((2, n, n), None, 0.0)
    expected_eta_upper = _fourier_face_values((2, n, n), None, 2.0 * np.pi)
    return max(
        float(np.max(np.abs(theta_lower - expected_theta_lower[None, ...]))),
        float(np.max(np.abs(theta_lower_right - expected_theta_lower[None, ...]))),
        float(np.max(np.abs(theta_upper - expected_theta_upper[None, ...]))),
        float(np.max(np.abs(theta_upper_right - expected_theta_upper[None, ...]))),
        float(np.max(np.abs(eta_lower - expected_eta_lower[None, ...]))),
        float(np.max(np.abs(eta_lower_right - expected_eta_lower[None, ...]))),
        float(np.max(np.abs(eta_upper - expected_eta_upper[None, ...]))),
        float(np.max(np.abs(eta_upper_right - expected_eta_upper[None, ...]))),
    )


def _unsharded_face_reconstruction(halos, geometry, *, axis, face_index):
    all_samples = tuple(
        tuple(
            jnp.moveaxis(sample, axis, 0)
            for sample in _axis_face_samples_from_halo(
                halo, geometry, axis=axis
            )
        )
        for halo in halos
    )
    support = tuple(
        jnp.stack(
            tuple(
                field_samples[index][face_index : face_index + 1]
                for field_samples in all_samples
            ),
            axis=-1,
        )
        for index in range(4)
    )
    return reconstruct_third_order_face_states(*support)[:2]


def test_periodic_fourier_material_trace_refines_at_third_order():
    errors = tuple(_fourier_trace_errors(n) for n in (16, 32, 64))
    orders = tuple(np.log2(errors[index] / errors[index + 1]) for index in (0, 1))
    print({"errors_16_32_64": errors, "orders": orders})
    assert orders[0] > 2.5
    assert orders[1] > 2.5


def test_periodic_fourier_material_trace_and_constant_operator_control():
    shape = (4, 16, 16)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    local = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=2)
    local_geometry = assemble_single_device_local_fci_geometry(local)
    domain = replace(local.domain, mesh_axis_names=(None, None, None))
    topology = TopologyHaloFiller3D(
        (LocalPeriodicTopologyRule3D((False, True, True)),)
    )
    fields = _fourier_owner_fields(local_geometry)
    halos = _close_fields(fields, local_geometry, domain, topology)
    endpoints = build_periodic_curvature_endpoint_face_states(
        halos, local_geometry, periodic_axes=domain.periodic_axes
    )

    theta_trace = _fourier_face_values(shape, 0.0)
    eta_trace = _fourier_face_values(shape, None, 0.0)
    theta_values = np.asarray(endpoints[1][0][0])
    eta_values = np.asarray(endpoints[2][0][0])
    np.testing.assert_allclose(
        theta_values,
        np.broadcast_to(theta_trace[None, None, :, :], theta_values.shape),
        rtol=0.0, atol=5.0e-3,
    )
    np.testing.assert_allclose(
        eta_values,
        np.broadcast_to(eta_trace[None, None, :, :], eta_values.shape),
        rtol=0.0, atol=5.0e-3,
    )
    for pair in (endpoints[1], endpoints[2]):
        for side in pair:
            assert np.all(np.isfinite(np.asarray(side)))
            assert np.all(np.asarray(side) > 0.0)

    constant_fields = tuple(
        jnp.full(local_geometry.owned_shape, value, dtype=jnp.float64)
        for value in (2.0, 3.0, 4.0, 0.0)
    )
    constant_halos = _close_fields(
        constant_fields, local_geometry, domain, topology
    )
    constant_endpoints = build_periodic_curvature_endpoint_face_states(
        constant_halos, local_geometry, periodic_axes=domain.periodic_axes
    )
    for axis in (1, 2):
        for pair in constant_endpoints[axis]:
            np.testing.assert_allclose(np.asarray(pair[0]), np.asarray(pair[1]))
            np.testing.assert_allclose(
                np.asarray(pair[0]),
                np.broadcast_to(
                    np.asarray((2.0, 3.0, 4.0, 0.0)), np.asarray(pair[0]).shape
                ),
            )

    context = StencilBuilderContext(layout=domain.layout, domain=domain)
    stencils = tuple(
        build_local_conservative_stencil_from_field(halo, local_geometry, context)
        for halo in constant_halos
    )
    coefficients = LocalCurvatureFaceCoefficients3D(
        layout=domain.layout,
        x=jnp.ones(domain.layout.face_control_shape(0)),
        y=jnp.ones(domain.layout.face_control_shape(1)),
        z=jnp.ones(domain.layout.face_control_shape(2)),
    )
    residual = local_curvature_production_path_op(
        stencils,
        local_geometry,
        coefficients,
        tau=0.7,
        domain=domain,
        equilibrium=jnp.asarray((2.0, 3.0, 4.0, 0.0)),
        periodic_endpoint_face_states=constant_endpoints,
    )
    np.testing.assert_allclose(np.asarray(residual), 0.0, rtol=0.0, atol=2.0e-12)

    context = StencilBuilderContext(layout=domain.layout, domain=domain)
    fourier_stencils = tuple(
        build_local_conservative_stencil_from_field(halo, local_geometry, context)
        for halo in halos
    )
    adaptive = local_curvature_production_path_op(
        fourier_stencils,
        local_geometry,
        coefficients,
        tau=0.7,
        domain=domain,
        equilibrium=jnp.asarray((2.0, 2.0, 2.0, 2.0)),
        periodic_endpoint_face_states=endpoints,
    )
    legacy = local_curvature_production_path_op(
        fourier_stencils,
        local_geometry,
        coefficients,
        tau=0.7,
        domain=domain,
        equilibrium=jnp.asarray((2.0, 2.0, 2.0, 2.0)),
    )
    assert float(jnp.max(jnp.abs(adaptive - legacy))) > 1.0e-8


def test_periodic_material_endpoint_trace_matches_one_device_across_eta_shards():
    if len(jax.devices()) < 2:
        pytest.skip("requires two JAX devices for an eta-shard parity test")
    shape = (4, 8, 8)
    global_geometry = build_shifted_torus_geometry(
        shape, construct_fci_maps=False
    )
    one = build_local_fci_geometries(global_geometry, (1, 1, 1), halo_width=2)
    one_geometry = assemble_single_device_local_fci_geometry(one)
    one_domain = replace(one.domain, mesh_axis_names=(None, None, None))
    topology = TopologyHaloFiller3D(
        (LocalPeriodicTopologyRule3D((False, True, True)),)
    )
    owner = _fourier_owner_fields(one_geometry)
    global_halos = _close_fields(owner, one_geometry, one_domain, topology)
    split = build_local_fci_geometries(global_geometry, (1, 1, 2), halo_width=2)
    mesh = make_shard_mesh((1, 1, 2))
    input_spec = P("x", "y", "z", None)
    owner_global = jnp.stack(owner, axis=-1)

    def kernel(fci_owned, owner_owned):
        local_geometry = assemble_local_fci_geometry(split, fci_owned)
        local_fields = tuple(owner_owned[..., field] for field in range(4))
        states = build_periodic_curvature_endpoint_face_states(
            _close_fields(local_fields, local_geometry, split.domain, topology),
            local_geometry,
            periodic_axes=split.domain.periodic_axes,
        )
        return (
            states[1][0][0][0],
            states[1][0][1][0],
            states[1][1][0][0],
            states[1][1][1][0],
            jnp.expand_dims(states[2][0][0][0], axis=0),
            jnp.expand_dims(states[2][0][1][0], axis=0),
            jnp.expand_dims(states[2][1][0][0], axis=0),
            jnp.expand_dims(states[2][1][1][0], axis=0),
        )

    output_specs = (
        P("x", "z", None),
        P("x", "z", None),
        P("x", "z", None),
        P("x", "z", None),
        P("z", "x", None, None),
        P("z", "x", None, None),
        P("z", "x", None, None),
        P("z", "x", None, None),
    )
    got = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(input_spec, input_spec),
            out_specs=output_specs,
            check_vma=False,
        )
    )(
        jax.device_put(split.cell_fields, NamedSharding(mesh, input_spec)),
        jax.device_put(owner_global, NamedSharding(mesh, input_spec)),
    )

    theta_face_zero = _unsharded_face_reconstruction(
        global_halos, one_geometry, axis=1, face_index=0
    )
    theta_face_end = _unsharded_face_reconstruction(
        global_halos, one_geometry, axis=1, face_index=8
    )
    eta_face_zero = _unsharded_face_reconstruction(
        global_halos, one_geometry, axis=2, face_index=0
    )
    eta_face_middle = _unsharded_face_reconstruction(
        global_halos, one_geometry, axis=2, face_index=4
    )
    eta_face_end = _unsharded_face_reconstruction(
        global_halos, one_geometry, axis=2, face_index=8
    )
    expected_values = (
        np.asarray(theta_face_zero[0])[0],
        np.asarray(theta_face_zero[1])[0],
        np.asarray(theta_face_end[0])[0],
        np.asarray(theta_face_end[1])[0],
        np.stack((np.asarray(eta_face_zero[0])[0], np.asarray(eta_face_middle[0])[0]), axis=0),
        np.stack((np.asarray(eta_face_zero[1])[0], np.asarray(eta_face_middle[1])[0]), axis=0),
        np.stack((np.asarray(eta_face_middle[0])[0], np.asarray(eta_face_end[0])[0]), axis=0),
        np.stack((np.asarray(eta_face_middle[1])[0], np.asarray(eta_face_end[1])[0]), axis=0),
    )
    for actual, reference in zip(got, expected_values):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(reference),
            rtol=0.0, atol=2.0e-12,
        )
