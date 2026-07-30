from types import SimpleNamespace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drbx.geometry import (
    HaloLayout3D,
    LocalCurvatureFaceCoefficients3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    ShardSpec3D,
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
    build_local_curvature_face_coefficients,
)
from drbx.native.fci_operators import (
    _local_axis_face_values_from_stencil,
    local_curvature_conservative_op,
)
from test_fci_operators_domain_decomp import (
    _build_domain,
    _build_local_geometry,
)


def _zero_physical_halo_geometry():
    layout = HaloLayout3D((3, 4, 5), halo_width=1)
    shape = layout.cell_halo_shape
    domain = LocalDomain3D(
        ShardSpec3D(
            global_shape=(3, 4, 5),
            owned_start=(0, 0, 0),
            owned_stop=(3, 4, 5),
            shard_index=(0, 0, 0),
            shard_counts=(1, 1, 1),
            periodic_axes=(False, False, True),
            halo_width=1,
        ),
        layout,
    )

    # This deliberately reproduces assemble_local_fci_geometry's zero
    # physical cell-geometry halo convention.
    cell_metric = jnp.zeros(shape + (3, 3), dtype=jnp.float64)
    cell_metric = cell_metric.at[1:-1, 1:-1, 1:-1].set(
        jnp.broadcast_to(jnp.eye(3), (3, 4, 5, 3, 3))
    )
    cell_b = jnp.zeros(shape + (3,), dtype=jnp.float64)
    cell_b = cell_b.at[1:-1, 1:-1, 1:-1].set(
        jnp.broadcast_to(jnp.array([1.0, 0.2, 0.3]), (3, 4, 5, 3))
    )
    cell_bmag = jnp.zeros(shape, dtype=jnp.float64).at[1:-1, 1:-1, 1:-1].set(1.0)

    def face_metric(location):
        face_shape = layout.location_halo_shape(location)
        return SimpleNamespace(
            g_cov=jnp.broadcast_to(jnp.eye(3), face_shape + (3, 3))
        )

    def face_bfield(location):
        face_shape = layout.location_halo_shape(location)
        return SimpleNamespace(
            B_contra_halo=jnp.broadcast_to(
                jnp.array([1.0, 0.2, 0.3]), face_shape + (3,)
            ),
            Bmag_halo=jnp.ones(face_shape, dtype=jnp.float64),
        )

    axis = lambda n: SimpleNamespace(
        faces_owned=jnp.arange(n + 1, dtype=jnp.float64)
    )
    geometry = object.__new__(LocalFciGeometry3D)
    object.__setattr__(geometry, "layout", layout)
    object.__setattr__(geometry, "cell_metric", SimpleNamespace(g_cov=cell_metric))
    object.__setattr__(
        geometry,
        "cell_bfield",
        SimpleNamespace(B_contra_halo=cell_b, Bmag_halo=cell_bmag),
    )
    object.__setattr__(
        geometry,
        "face_metric",
        SimpleNamespace(
            x=face_metric("x_face"),
            y=face_metric("y_face"),
            z=face_metric("z_face"),
        ),
    )
    object.__setattr__(
        geometry,
        "face_bfield",
        SimpleNamespace(
            x=face_bfield("x_face"),
            y=face_bfield("y_face"),
            z=face_bfield("z_face"),
        ),
    )
    object.__setattr__(
        geometry,
        "grid",
        SimpleNamespace(x=axis(3), y=axis(4), z=axis(5)),
    )
    return geometry, domain


def test_curvature_face_coefficients_close_zero_physical_cell_halos():
    jax.config.update("jax_enable_x64", True)
    geometry, domain = _zero_physical_halo_geometry()
    coefficients = build_local_curvature_face_coefficients(geometry, domain)

    assert coefficients.x.shape == (4, 4, 5)
    assert coefficients.y.shape == (3, 5, 5)
    assert coefficients.z.shape == (3, 4, 6)
    assert max(float(jnp.max(jnp.abs(value))) for value in coefficients.axes) < 1.0

    div_q = (
        coefficients.x[1:] - coefficients.x[:-1]
        + coefficients.y[:, 1:] - coefficients.y[:, :-1]
        + coefficients.z[:, :, 1:] - coefficients.z[:, :, :-1]
    )
    assert float(jnp.max(jnp.abs(div_q))) < 1.0e-13

    constant_residual = div_q
    assert float(jnp.max(jnp.abs(constant_residual))) < 1.0e-13


def _operator_fixture(shape=(3, 4, 5), halo_width=1):
    geometry = _build_local_geometry(
        shape,
        halo_width,
        global_shape=shape,
    )
    layout = geometry.layout
    domain = _build_domain(shape, halo_width)
    field_halo = jnp.ones(layout.cell_halo_shape, dtype=jnp.float64)
    stencil = build_local_conservative_stencil_from_field(
        field_halo,
        geometry,
        StencilBuilderContext(layout=layout, domain=domain),
    )
    faces = tuple(layout.face_control_shape(axis) for axis in range(3))
    coefficients = LocalCurvatureFaceCoefficients3D(
        layout=layout,
        x=jnp.ones(faces[0], dtype=jnp.float64),
        y=jnp.ones(faces[1], dtype=jnp.float64),
        z=jnp.ones(faces[2], dtype=jnp.float64),
    )
    return geometry, domain, stencil, coefficients


def test_constant_field_has_zero_curvature_for_compatible_constant_face_flux():
    geometry, _domain, stencil, coefficients = _operator_fixture()
    result = local_curvature_conservative_op(stencil, geometry, coefficients)
    np.testing.assert_allclose(result, 0.0, atol=2.0e-12, rtol=0.0)


def test_weighted_sum_is_shared_face_flux_balance():
    geometry, domain, _stencil, coefficients = _operator_fixture()
    layout = geometry.layout
    field_halo = jnp.arange(
        np.prod(layout.cell_halo_shape),
        dtype=jnp.float64,
    ).reshape(layout.cell_halo_shape)
    stencil = build_local_conservative_stencil_from_field(
        field_halo,
        geometry,
        StencilBuilderContext(layout=layout, domain=domain),
    )
    result = local_curvature_conservative_op(stencil, geometry, coefficients)
    x_face = _local_axis_face_values_from_stencil(stencil.x, axis=0)
    y_face = _local_axis_face_values_from_stencil(stencil.y, axis=1)
    z_face = _local_axis_face_values_from_stencil(stencil.z, axis=2)
    dx = geometry.spacing.dx_owned
    dy = geometry.spacing.dy_owned
    dz = geometry.spacing.dz_owned
    expected = jnp.sum(coefficients.x[-1] * x_face[-1] * dy[-1] * dz[-1])
    expected -= jnp.sum(coefficients.x[0] * x_face[0] * dy[0] * dz[0])
    expected += jnp.sum(
        coefficients.y[:, -1]
        * y_face[:, -1]
        * dx[:, -1]
        * dz[:, -1]
    )
    expected -= jnp.sum(
        coefficients.y[:, 0]
        * y_face[:, 0]
        * dx[:, 0]
        * dz[:, 0]
    )
    expected += jnp.sum(
        coefficients.z[:, :, -1]
        * z_face[:, :, -1]
        * dx[:, :, -1]
        * dy[:, :, -1]
    )
    expected -= jnp.sum(
        coefficients.z[:, :, 0]
        * z_face[:, :, 0]
        * dx[:, :, 0]
        * dy[:, :, 0]
    )
    weighted = jnp.sum(
        result
        * jnp.asarray(geometry.cell_metric.J_owned)
        / jnp.asarray(geometry.cell_bfield.Bmag_owned)
        * dx
        * dy
        * dz
    )
    np.testing.assert_allclose(
        weighted,
        expected,
        atol=2.0e-11,
        rtol=2.0e-12,
    )


def test_curvature_operator_is_jit_compatible():
    geometry, _domain, stencil, coefficients = _operator_fixture((2, 2, 3))
    eager = local_curvature_conservative_op(stencil, geometry, coefficients)
    compiled = jax.jit(
        lambda value: local_curvature_conservative_op(
            value,
            geometry,
            coefficients,
        )
    )
    np.testing.assert_allclose(
        compiled(stencil),
        eager,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_coefficients_validate_face_shapes():
    layout = HaloLayout3D((2, 2, 2), 1)
    faces = tuple(layout.face_control_shape(axis) for axis in range(3))
    with np.testing.assert_raises(ValueError):
        LocalCurvatureFaceCoefficients3D(
            layout=layout,
            x=jnp.zeros((faces[0][0] - 1,) + faces[0][1:]),
            y=jnp.zeros(faces[1]),
            z=jnp.zeros(faces[2]),
        )
