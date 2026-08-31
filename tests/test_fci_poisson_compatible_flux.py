"""Focused tests for the experimental compatible-flux Poisson bracket."""

from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drbx.geometry import (  # noqa: E402
    HaloLayout3D,
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
    build_local_stencil_from_field,
)
from drbx.native.fci_boundaries import (  # noqa: E402
    CV_FACE_INTERIOR,
    CoordinateFaceValues3D,
    FaceFluxStencil3D,
    LocalBoundaryFaceTrace3D,
    LocalControlVolumeFieldClosure3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentReconstruction3D,
)
from drbx.native.fci_operators import (  # noqa: E402
    _compatible_characteristic_regular_flux,
    _compatible_flux_divergence,
    _compatible_flux_generator,
    _third_order_scalar_face_states_from_halo,
    local_poisson_bracket_compatible_flux_op,
    local_poisson_bracket_op,
)
from axis_regular_operator_support import owned, polar_fixture, scalar_field_halo  # noqa: E402
from test_fci_operators_domain_decomp import _build_domain, _build_local_geometry  # noqa: E402
from test_fci_gmres_control_volume_owner_space import _merged_control_volume  # noqa: E402
from test_fci_cutwall_slab_operators import (  # noqa: E402
    _cubic_face_functionals,
    _uniform_control_volume_cells,
    _unit_control_volume_face_rows,
)


def _conservative(field, geometry, context):
    return build_local_conservative_stencil_from_field(field, geometry, context)


def _trace_on_upper_x(
    geometry, face_values, *, value=None, lower=False, upper=True
):
    """Make a trace with an explicit physical upper-x value.

    The optional lower mask is useful for asserting that the toroidal axis
    face is ignored when ``axis_regular_axes[0]`` is enabled.
    """
    trace = LocalBoundaryFaceTrace3D.empty(geometry.layout)
    if value is None:
        value = face_values.x[-1]
    trace = LocalBoundaryFaceTrace3D(
        value_x=trace.value_x.at[-1].set(value),
        value_y=trace.value_y,
        value_z=trace.value_z,
        mask_x=trace.mask_x.at[-1].set(upper).at[0].set(lower),
        mask_y=trace.mask_y,
        mask_z=trace.mask_z,
        layout=geometry.layout,
    )
    return trace


def _corrupt_x_face_values(stencil, *, value):
    face_values = stencil.face_values
    corrupted_x = face_values.x.at[0].set(value).at[-1].set(value)
    return stencil.replace(
        face_values=type(face_values)(
            x=corrupted_x,
            y=face_values.y,
            z=face_values.z,
        )
    )


def _polar_pair(shape=(8, 32, 4)):
    geometry, domain, context, (r, theta, z), *_ = polar_fixture(shape=shape)
    f = scalar_field_halo(r, theta, z, lambda r, t, z: r * jnp.cos(t) + 0.1 * z)
    g = scalar_field_halo(r, theta, z, lambda r, t, z: r * jnp.sin(t) - 0.2 * z)
    return geometry, domain, context, f, g


def test_constants_return_zero_in_either_argument_on_axis_regular_grid():
    geometry, domain, context, f, _g = _polar_pair()
    constant = jnp.ones_like(f)
    f_stencil = _conservative(f, geometry, context)
    constant_stencil = _conservative(constant, geometry, context)

    fg = local_poisson_bracket_compatible_flux_op(
        f_stencil, constant_stencil, geometry,
        domain=domain, axis_regular_axes=(True, False, False),
    )
    gf = local_poisson_bracket_compatible_flux_op(
        constant_stencil, f_stencil, geometry,
        domain=domain, axis_regular_axes=(True, False, False),
    )
    assert jnp.allclose(fg, 0.0, atol=2.0e-13, rtol=0.0)
    assert jnp.allclose(gf, 0.0, atol=2.0e-13, rtol=0.0)

    fg_upwind = local_poisson_bracket_compatible_flux_op(
        f_stencil,
        constant_stencil,
        geometry,
        domain=domain,
        axis_regular_axes=(True, False, False),
        characteristic_scheme="third-order-upwind",
        g_field_halo=constant,
        g_positivity_floor=1.0e-12,
    )
    assert jnp.allclose(fg_upwind, 0.0, atol=2.0e-13, rtol=0.0)


def test_third_order_scalar_reconstruction_uses_canonical_four_cell_stencil():
    shape = (5, 4, 3)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    halo_shape = geometry.halo_shape
    values = jnp.arange(np.prod(halo_shape), dtype=jnp.float64).reshape(halo_shape)

    left, right, fallback = _third_order_scalar_face_states_from_halo(
        values,
        geometry,
        boundary_trace=None,
        axis_regular_axes=(False, False, False),
        positivity_floor=None,
    )

    h = geometry.layout.halo_width
    face = 2
    qm = values[h + face - 2, h : h + shape[1], h : h + shape[2]]
    q0 = values[h + face - 1, h : h + shape[1], h : h + shape[2]]
    q1 = values[h + face, h : h + shape[1], h : h + shape[2]]
    qp = values[h + face + 1, h : h + shape[1], h : h + shape[2]]
    np.testing.assert_allclose(left.x[face], (-qm + 5.0 * q0 + 2.0 * q1) / 6.0)
    np.testing.assert_allclose(right.x[face], (2.0 * q0 + 5.0 * q1 - qp) / 6.0)
    assert not bool(jnp.any(fallback.x))
    assert not bool(jnp.any(fallback.y))
    assert not bool(jnp.any(fallback.z))


def test_third_order_scalar_reconstruction_falls_back_sidewise_for_positivity():
    shape = (4, 3, 2)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    h = geometry.layout.halo_width
    values = jnp.ones(geometry.halo_shape, dtype=jnp.float64)
    # At owned x face 2 the outer samples make both reconstructed states
    # negative, while the two adjacent owner states remain admissible.
    values = values.at[h, :, :].set(10.0)
    values = values.at[h + 3, :, :].set(10.0)

    left, right, fallback = _third_order_scalar_face_states_from_halo(
        values,
        geometry,
        boundary_trace=None,
        axis_regular_axes=(False, False, False),
        positivity_floor=1.0e-12,
    )

    np.testing.assert_array_equal(left.x[2], jnp.ones_like(left.x[2]))
    np.testing.assert_array_equal(right.x[2], jnp.ones_like(right.x[2]))
    assert bool(jnp.all(fallback.x[2]))


def test_third_order_scalar_reconstruction_uses_first_order_wall_trace():
    shape = (4, 3, 2)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    h = geometry.layout.halo_width
    values = jnp.broadcast_to(
        jnp.arange(geometry.halo_shape[0], dtype=jnp.float64)[:, None, None],
        geometry.halo_shape,
    )
    trace = LocalBoundaryFaceTrace3D.empty(geometry.layout)
    upper_value = jnp.full(trace.value_x[-1].shape, 0.25, dtype=jnp.float64)
    trace = LocalBoundaryFaceTrace3D(
        value_x=trace.value_x.at[-1].set(upper_value),
        value_y=trace.value_y,
        value_z=trace.value_z,
        mask_x=trace.mask_x.at[-1].set(True),
        mask_y=trace.mask_y,
        mask_z=trace.mask_z,
        layout=geometry.layout,
    )

    left, right, fallback = _third_order_scalar_face_states_from_halo(
        values,
        geometry,
        boundary_trace=trace,
        axis_regular_axes=(False, False, False),
        positivity_floor=1.0e-12,
    )

    last_owner = values[h + shape[0] - 1, h : h + shape[1], h : h + shape[2]]
    np.testing.assert_array_equal(left.x[-1], last_owner)
    np.testing.assert_array_equal(right.x[-1], upper_value)
    assert bool(jnp.all(fallback.x[-1]))


def test_wall_characteristic_flux_selects_inflow_trace_and_outflow_owner():
    shape = (4, 3, 2)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    h = geometry.layout.halo_width
    values = jnp.broadcast_to(
        jnp.arange(geometry.halo_shape[0], dtype=jnp.float64)[:, None, None],
        geometry.halo_shape,
    )
    trace = LocalBoundaryFaceTrace3D.empty(geometry.layout)
    lower_value = jnp.full(trace.value_x[0].shape, 0.25, dtype=jnp.float64)
    upper_value = jnp.full(trace.value_x[-1].shape, 0.75, dtype=jnp.float64)
    trace = LocalBoundaryFaceTrace3D(
        value_x=trace.value_x.at[0].set(lower_value).at[-1].set(upper_value),
        value_y=trace.value_y,
        value_z=trace.value_z,
        mask_x=trace.mask_x.at[0].set(True).at[-1].set(True),
        mask_y=trace.mask_y,
        mask_z=trace.mask_z,
        layout=geometry.layout,
    )
    left, right, _ = _third_order_scalar_face_states_from_halo(
        values,
        geometry,
        boundary_trace=trace,
        axis_regular_axes=(False, False, False),
        positivity_floor=1.0e-12,
    )
    first_owner = values[h, h : h + shape[1], h : h + shape[2]]
    last_owner = values[
        h + shape[0] - 1, h : h + shape[1], h : h + shape[2]
    ]
    zero_y = jnp.zeros_like(left.y)
    zero_z = jnp.zeros_like(left.z)

    positive_x = jnp.ones_like(left.x).at[0].set(2.0).at[-1].set(2.0)
    positive_flux = _compatible_characteristic_regular_flux(
        FaceFluxStencil3D(positive_x, zero_y, zero_z), left, right
    )[0]
    np.testing.assert_array_equal(positive_flux[0] / 2.0, lower_value)
    np.testing.assert_array_equal(positive_flux[-1] / 2.0, last_owner)

    negative_x = -jnp.ones_like(left.x).at[0].set(2.0).at[-1].set(2.0)
    negative_flux = _compatible_characteristic_regular_flux(
        FaceFluxStencil3D(negative_x, zero_y, zero_z), left, right
    )[0]
    np.testing.assert_array_equal(negative_flux[0] / -2.0, first_owner)
    np.testing.assert_array_equal(negative_flux[-1] / -2.0, upper_value)


def test_third_order_scalar_reconstruction_is_shared_at_shard_face():
    global_shape = (4, 8, 4)
    local_shape = (4, 4, 4)
    reconstructed = []
    for shard_y in (0, 1):
        geometry = _build_local_geometry(
            local_shape,
            2,
            global_shape=global_shape,
            shard_index=(0, shard_y, 0),
        )
        x = geometry.grid.x.centers_halo[:, None, None]
        y = geometry.grid.y.centers_halo[None, :, None]
        z = geometry.grid.z.centers_halo[None, None, :]
        x, y, z = jnp.broadcast_arrays(x, y, z)
        values = 1.5 + 0.1 * x + 0.2 * jnp.sin(y) + 0.05 * jnp.cos(z)
        left, right, _ = _third_order_scalar_face_states_from_halo(
            values,
            geometry,
            boundary_trace=None,
            axis_regular_axes=(False, False, False),
            positivity_floor=1.0e-12,
        )
        reconstructed.append((left, right))

    left0, right0 = reconstructed[0]
    left1, right1 = reconstructed[1]
    np.testing.assert_allclose(left0.y[:, -1], left1.y[:, 0], atol=1.0e-14)
    np.testing.assert_allclose(right0.y[:, -1], right1.y[:, 0], atol=1.0e-14)


def test_third_order_scalar_reconstruction_is_shared_at_periodic_seam():
    shape = (4, 8, 3)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    h = geometry.layout.halo_width
    j = jnp.arange(-h, shape[1] + h, dtype=jnp.float64)
    periodic_line = jnp.sin(2.0 * jnp.pi * j / shape[1]) + 0.2 * jnp.cos(
        4.0 * jnp.pi * j / shape[1]
    )
    values = jnp.broadcast_to(
        periodic_line[None, :, None], geometry.halo_shape
    )

    left, right, fallback = _third_order_scalar_face_states_from_halo(
        values,
        geometry,
        boundary_trace=None,
        axis_regular_axes=(False, False, False),
        positivity_floor=None,
    )

    np.testing.assert_allclose(left.y[:, 0], left.y[:, -1], atol=1.0e-14)
    np.testing.assert_allclose(right.y[:, 0], right.y[:, -1], atol=1.0e-14)
    np.testing.assert_array_equal(fallback.y[:, 0], fallback.y[:, -1])


def test_third_order_characteristic_correction_damps_periodic_fourier_modes():
    """For RHS ``-bracket``, every resolved constant-speed mode is dissipative."""

    shape = (4, 8, 4)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    domain = _build_domain(shape, 2)
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    h = geometry.layout.halo_width
    j = jnp.arange(-h, shape[1] + h, dtype=jnp.float64)
    productions = []
    for mode in range(1, shape[1] // 2 + 1):
        line = jnp.cos(2.0 * jnp.pi * mode * j / shape[1])
        values = jnp.broadcast_to(line[None, :, None], geometry.halo_shape)
        stencil = _conservative(values, geometry, context)
        left, right, _ = _third_order_scalar_face_states_from_halo(
            values,
            geometry,
            boundary_trace=None,
            axis_regular_axes=(False, False, False),
            positivity_floor=None,
        )
        generator = FaceFluxStencil3D(
            x=jnp.zeros_like(stencil.face_values.x),
            y=jnp.ones_like(stencil.face_values.y),
            z=jnp.zeros_like(stencil.face_values.z),
        )
        upwind_flux = _compatible_characteristic_regular_flux(
            generator, left, right
        )
        correction_flux = tuple(
            full - velocity * centered
            for full, velocity, centered in zip(
                upwind_flux,
                (generator.x, generator.y, generator.z),
                (
                    stencil.face_values.x,
                    stencil.face_values.y,
                    stencil.face_values.z,
                ),
                strict=True,
            )
        )
        correction = _compatible_flux_divergence(
            FaceFluxStencil3D(*correction_flux),
            geometry,
            jacobian_floor=1.0e-30,
        ) / geometry.cell_metric.J_owned
        q_owned = values[
            h : h + shape[0], h : h + shape[1], h : h + shape[2]
        ]
        productions.append(
            float(
                jnp.sum(geometry.cell_metric.J_owned * q_owned * correction)
            )
        )
    assert min(productions) >= -1.0e-12
    assert productions[-1] > 0.0


def test_third_order_selector_is_unified_compatible_characteristic_bracket():
    """The selector replaces the physical action inside one compatible bracket."""

    shape = (6, 8, 4)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    domain = _build_domain(shape, 2)
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    x = geometry.grid.x.centers_halo[:, None, None]
    y = geometry.grid.y.centers_halo[None, :, None]
    z = geometry.grid.z.centers_halo[None, None, :]
    x, y, z = jnp.broadcast_arrays(x, y, z)
    f = 0.3 * x * x + jnp.sin(y) + 0.2 * jnp.cos(z)
    g = 1.2 + 0.1 * x + 0.15 * jnp.sin(2.0 * y - z)
    fs = _conservative(f, geometry, context)
    gs = _conservative(g, geometry, context)

    actual = jax.jit(
        lambda f_stencil, g_stencil, g_halo: (
            local_poisson_bracket_compatible_flux_op(
                f_stencil,
                g_stencil,
                geometry,
                domain=domain,
                characteristic_scheme="third-order-upwind",
                g_field_halo=g_halo,
                g_positivity_floor=1.0e-12,
            )
        )
    )(fs, gs, g)

    def manual_action(generator, argument, argument_halo, positivity_floor):
        velocity = _compatible_flux_generator(
            generator,
            geometry,
            domain=domain,
            axis_regular_axes=(False, False, False),
            b_floor=1.0e-30,
        )
        left, right, _ = _third_order_scalar_face_states_from_halo(
            argument_halo,
            geometry,
            boundary_trace=None,
            axis_regular_axes=(False, False, False),
            positivity_floor=positivity_floor,
        )
        upwind = CoordinateFaceValues3D(
            x=jnp.where(velocity.x >= 0.0, left.x, right.x),
            y=jnp.where(velocity.y >= 0.0, left.y, right.y),
            z=jnp.where(velocity.z >= 0.0, left.z, right.z),
        )
        weighted = FaceFluxStencil3D(
            x=velocity.x * upwind.x,
            y=velocity.y * upwind.y,
            z=velocity.z * upwind.z,
        )
        return (
            _compatible_flux_divergence(
                weighted, geometry, jacobian_floor=1.0e-30
            )
            - argument.x.center
            * _compatible_flux_divergence(
                velocity, geometry, jacobian_floor=1.0e-30
            )
        ) / geometry.cell_metric.J_owned

    centered = local_poisson_bracket_compatible_flux_op(
        fs, gs, geometry, domain=domain
    )
    velocity = _compatible_flux_generator(
        fs,
        geometry,
        domain=domain,
        axis_regular_axes=(False, False, False),
        b_floor=1.0e-30,
    )
    centered_weighted = FaceFluxStencil3D(
        x=velocity.x * gs.face_values.x,
        y=velocity.y * gs.face_values.y,
        z=velocity.z * gs.face_values.z,
    )
    centered_f_action = (
        _compatible_flux_divergence(
            centered_weighted, geometry, jacobian_floor=1.0e-30
        )
        - gs.x.center
        * _compatible_flux_divergence(
            velocity, geometry, jacobian_floor=1.0e-30
        )
    ) / geometry.cell_metric.J_owned
    manual = (
        centered
        + manual_action(fs, gs, g, 1.0e-12)
        - centered_f_action
    )
    np.testing.assert_allclose(actual, manual, atol=2.0e-13, rtol=2.0e-13)


def test_third_order_characteristic_scheme_validates_halo_and_selector():
    geometry, domain, context, f, g = _polar_pair(shape=(4, 8, 2))
    fs = _conservative(f, geometry, context)
    gs = _conservative(g, geometry, context)
    with pytest.raises(ValueError, match="characteristic_scheme"):
        local_poisson_bracket_compatible_flux_op(
            fs, gs, geometry, characteristic_scheme="tunable-rusanov"
        )
    with pytest.raises(ValueError, match="g_field_halo"):
        local_poisson_bracket_compatible_flux_op(
            fs,
            gs,
            geometry,
            domain=domain,
            characteristic_scheme="third-order-upwind",
        )


def test_third_order_characteristic_action_uses_control_volume_owner_space():
    shape = (4, 6, 3)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    domain = _build_domain(shape, 2)
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    x = geometry.grid.x.centers_halo[:, None, None]
    y = geometry.grid.y.centers_halo[None, :, None]
    z = geometry.grid.z.centers_halo[None, None, :]
    x, y, z = jnp.broadcast_arrays(x, y, z)
    f = x + 0.2 * jnp.sin(y) - 0.1 * jnp.cos(z)
    g = 1.0 + 0.05 * x + 0.02 * jnp.sin(y - z)
    control_volume = _merged_control_volume(
        geometry,
        source=(0, 1, 0),
        owner=(0, 0, 0),
    )
    empty = LocalControlVolumeFieldClosure3D.empty(max_rows=0)

    result = local_poisson_bracket_compatible_flux_op(
        _conservative(f, geometry, context),
        _conservative(g, geometry, context),
        geometry,
        domain=domain,
        control_volume_geometry=control_volume,
        f_field_closure=empty,
        g_field_closure=empty,
        characteristic_scheme="third-order-upwind",
        g_field_halo=g,
        g_positivity_floor=1.0e-12,
    )

    assert bool(jnp.all(jnp.isfinite(result)))
    assert float(result[0, 1, 0]) == 0.0


def test_unified_characteristic_bracket_uses_active_compact_face_fluxes():
    shape = (6, 6, 6)
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    domain = _build_domain(shape, 2)
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    x = geometry.grid.x.centers_halo[:, None, None]
    y = geometry.grid.y.centers_halo[None, :, None]
    z = geometry.grid.z.centers_halo[None, None, :]
    x, y, z = jnp.broadcast_arrays(x, y, z)
    f = x
    g = 1.0 + y
    faces = _unit_control_volume_face_rows(
        geometry,
        (
            (CV_FACE_INTERIOR, (1, 1, 1), (1, 2, 1), 1, (0.0, 0.0, 0.0), 1.0),
        ),
    )
    cells = _uniform_control_volume_cells(geometry)
    control_volume = LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=geometry.regular_face_geometry,
        irregular_faces=faces,
        reconstruction=LocalMomentReconstruction3D.empty(geometry.layout),
        face_functionals=_cubic_face_functionals(cells, faces),
    )
    quadrature_active = jnp.asarray(faces.quadrature_active, dtype=bool)

    def closure(*, face_value, gradient):
        rows = faces.max_rows
        patch_shape = quadrature_active.shape
        face_values = jnp.broadcast_to(
            jnp.asarray(face_value, dtype=jnp.float64)[:, None, None], patch_shape
        )
        face_gradients = jnp.broadcast_to(
            jnp.asarray(gradient, dtype=jnp.float64)[:, None, None, :],
            patch_shape + (3,),
        )
        return LocalControlVolumeFieldClosure3D(
            projected_flux=jnp.zeros((rows,), dtype=jnp.float64),
            parallel_flux=jnp.zeros((rows,), dtype=jnp.float64),
            parallel_gradient_flux=jnp.zeros((rows,), dtype=jnp.float64),
            valid=jnp.ones((rows,), dtype=bool),
            active=jnp.ones((rows,), dtype=bool),
            max_rows=rows,
            max_patches=faces.max_patches,
            face_value=face_values,
            face_gradient=face_gradients,
            face_value_valid=quadrature_active,
            face_gradient_valid=quadrature_active,
        )

    f_closure = closure(
        face_value=jnp.asarray((0.5,)),
        gradient=jnp.asarray(((1.0, 0.0, 0.0),)),
    )
    g_closure = closure(
        face_value=jnp.asarray((2.0,)),
        gradient=jnp.asarray(((0.0, 1.0, 0.0),)),
    )
    kwargs = dict(
        domain=domain,
        control_volume_geometry=control_volume,
        f_field_closure=f_closure,
        g_field_closure=g_closure,
    )
    fs = _conservative(f, geometry, context)
    gs = _conservative(g, geometry, context)
    centered = local_poisson_bracket_compatible_flux_op(
        fs, gs, geometry, **kwargs
    )
    characteristic = local_poisson_bracket_compatible_flux_op(
        fs,
        gs,
        geometry,
        characteristic_scheme="third-order-upwind",
        g_field_halo=g,
        g_positivity_floor=1.0e-12,
        **kwargs,
    )

    delta = characteristic - centered
    assert bool(jnp.all(jnp.isfinite(characteristic)))
    assert float(jnp.max(jnp.abs(delta))) > 0.0
    assert float(jnp.abs(delta[1, 1, 1])) > 0.0


def test_argument_antisymmetry_is_explicit_on_axis_regular_grid():
    geometry, domain, context, f, g = _polar_pair()
    f_stencil = _conservative(f, geometry, context)
    g_stencil = _conservative(g, geometry, context)
    fg = local_poisson_bracket_compatible_flux_op(
        f_stencil, g_stencil, geometry,
        domain=domain, axis_regular_axes=(True, False, False),
    )
    gf = local_poisson_bracket_compatible_flux_op(
        g_stencil, f_stencil, geometry,
        domain=domain, axis_regular_axes=(True, False, False),
    )
    assert jnp.max(jnp.abs(fg + gf)) < 2.0e-13

def test_axis_regular_result_is_finite_and_owned_axis_flux_is_safe():
    geometry, domain, context, f, g = _polar_pair()
    result = local_poisson_bracket_compatible_flux_op(
        _conservative(f, geometry, context),
        _conservative(g, geometry, context),
        geometry,
        domain=domain, axis_regular_axes=(True, False, False),
    )
    assert result.shape == geometry.owned_shape
    assert bool(jnp.all(jnp.isfinite(result)))
    assert float(jnp.max(jnp.abs(result[0]))) < 1.0e3


def test_axis_regular_requires_domain_and_rejects_other_regular_axes():
    geometry, _domain, context, f, g = _polar_pair(shape=(4, 8, 2))
    fs = _conservative(f, geometry, context)
    gs = _conservative(g, geometry, context)
    with pytest.raises(ValueError, match="domain"):
        local_poisson_bracket_compatible_flux_op(
            fs, gs, geometry, axis_regular_axes=(True, False, False)
        )
    with pytest.raises(NotImplementedError, match="lower x"):
        local_poisson_bracket_compatible_flux_op(
            fs, gs, geometry, axis_regular_axes=(False, True, False)
        )


def test_smooth_nonsingular_result_tracks_direct_bracket_over_interior():
    shape = (12, 24, 12)
    geometry = _build_local_geometry(shape, 1, global_shape=shape)

    def euclidean_metric(metric):
        ones = jnp.ones_like(metric.J_halo)
        zeros = jnp.zeros_like(metric.J_halo)
        return replace(
            metric,
            J_halo=ones,
            g11_halo=ones, g22_halo=ones, g33_halo=ones,
            g12_halo=zeros, g13_halo=zeros, g23_halo=zeros,
            g_11_halo=ones, g_22_halo=ones, g_33_halo=ones,
            g_12_halo=zeros, g_13_halo=zeros, g_23_halo=zeros,
        )

    def axial_bfield(field):
        b = jnp.zeros_like(field.B_contra_halo).at[..., 2].set(1.0)
        return replace(field, B_contra_halo=b, Bmag_halo=jnp.ones_like(field.Bmag_halo))

    geometry = replace(
        geometry,
        cell_metric=euclidean_metric(geometry.cell_metric),
        face_metric=replace(
            geometry.face_metric,
            x=euclidean_metric(geometry.face_metric.x),
            y=euclidean_metric(geometry.face_metric.y),
            z=euclidean_metric(geometry.face_metric.z),
        ),
        cell_bfield=axial_bfield(geometry.cell_bfield),
        face_bfield=replace(
            geometry.face_bfield,
            x=axial_bfield(geometry.face_bfield.x),
            y=axial_bfield(geometry.face_bfield.y),
            z=axial_bfield(geometry.face_bfield.z),
        ),
    )
    domain = _build_domain(shape, 1)
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    rho = geometry.grid.x.centers_halo[:, None, None]
    theta = geometry.grid.y.centers_halo[None, :, None]
    phi = geometry.grid.z.centers_halo[None, None, :]
    rho, theta, phi = jnp.broadcast_arrays(rho, theta, phi)
    f = jnp.sin(1.7 * rho) + 0.2 * jnp.cos(theta + 0.4 * phi)
    g = jnp.cos(0.9 * rho) + 0.3 * jnp.sin(2.0 * theta - phi)

    compatible = local_poisson_bracket_compatible_flux_op(
        _conservative(f, geometry, context),
        _conservative(g, geometry, context),
        geometry,
    )
    direct = local_poisson_bracket_op(
        build_local_stencil_from_field(f, geometry, context),
        build_local_stencil_from_field(g, geometry, context),
        geometry,
    )
    # Face-gradient and cell-gradient discretizations are distinct.  This is
    # a smooth-consistency check, not an exact weighted identity assertion.
    interior = (slice(2, -2), slice(2, -2), slice(2, -2))
    error = np.asarray(compatible[interior] - direct[interior])
    assert np.all(np.isfinite(error))
    assert float(np.max(np.abs(error))) < 0.25


def test_compatible_flux_bracket_uses_operand_specific_upper_x_traces():
    """Physical wall traces override corrupted operand face values.

    ``f_boundary_trace`` belongs to the f operand and ``g_boundary_trace`` to
    the g operand.  Both are tested independently so the API cannot silently
    use one trace for both bracket arguments.
    """
    geometry, domain, context, f, g = _polar_pair(shape=(6, 12, 4))
    f_stencil = _conservative(f, geometry, context)
    g_stencil = _conservative(g, geometry, context)
    baseline = local_poisson_bracket_compatible_flux_op(
        f_stencil,
        g_stencil,
        geometry,
        domain=domain,
        axis_regular_axes=(True, False, False),
    )

    corrupted_f = _corrupt_x_face_values(f_stencil, value=1.0e6)
    with_f_trace = local_poisson_bracket_compatible_flux_op(
        corrupted_f,
        g_stencil,
        geometry,
        domain=domain,
        axis_regular_axes=(True, False, False),
        f_boundary_trace=_trace_on_upper_x(geometry, f_stencil.face_values),
    )
    assert jnp.max(jnp.abs(with_f_trace - baseline)) < 1.0e-12

    corrupted_g = _corrupt_x_face_values(g_stencil, value=-1.0e6)
    with_g_trace = local_poisson_bracket_compatible_flux_op(
        f_stencil,
        corrupted_g,
        geometry,
        domain=domain,
        axis_regular_axes=(True, False, False),
        g_boundary_trace=_trace_on_upper_x(geometry, g_stencil.face_values),
    )
    assert jnp.max(jnp.abs(with_g_trace - baseline)) < 1.0e-12


def test_compatible_flux_bracket_ignores_unmasked_and_axis_lower_traces():
    """Only masked physical boundary faces are patched; the lower axis is not."""
    geometry, domain, context, f, g = _polar_pair(shape=(6, 12, 4))
    f_stencil = _corrupt_x_face_values(_conservative(f, geometry, context), value=1.0e6)
    g_stencil = _corrupt_x_face_values(_conservative(g, geometry, context), value=-1.0e6)
    baseline = local_poisson_bracket_compatible_flux_op(
        f_stencil,
        g_stencil,
        geometry,
        domain=domain,
        axis_regular_axes=(True, False, False),
    )

    # The upper value is deliberately left inactive.  The lower value is
    # deliberately active but belongs to the collapsed topology face.
    f_trace = _trace_on_upper_x(
        geometry, f_stencil.face_values, value=-7.0e5, lower=True, upper=False
    )
    g_trace = _trace_on_upper_x(
        geometry, g_stencil.face_values, value=8.0e5, lower=True, upper=False
    )
    unpatched = local_poisson_bracket_compatible_flux_op(
        f_stencil,
        g_stencil,
        geometry,
        domain=domain,
        axis_regular_axes=(True, False, False),
        f_boundary_trace=f_trace,
        g_boundary_trace=g_trace,
    )
    assert jnp.allclose(unpatched, baseline, atol=1.0e-12, rtol=0.0)


def test_compatible_flux_bracket_validates_boundary_trace_type_and_layout():
    geometry, domain, context, f, g = _polar_pair(shape=(4, 8, 2))
    f_stencil = _conservative(f, geometry, context)
    g_stencil = _conservative(g, geometry, context)
    with pytest.raises(TypeError, match="LocalBoundaryFaceTrace3D"):
        local_poisson_bracket_compatible_flux_op(
            f_stencil,
            g_stencil,
            geometry,
            domain=domain,
            f_boundary_trace=object(),
        )

    wrong_layout = HaloLayout3D((5, 8, 2), geometry.layout.halo_width)
    with pytest.raises(ValueError, match="HaloLayout3D"):
        local_poisson_bracket_compatible_flux_op(
            f_stencil,
            g_stencil,
            geometry,
            domain=domain,
            g_boundary_trace=LocalBoundaryFaceTrace3D.empty(wrong_layout),
        )
