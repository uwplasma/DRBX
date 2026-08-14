"""Focused tests for explicit conservative-operator boundary traces."""

from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drbx.geometry import (  # noqa: E402
    LocalCurvatureFaceCoefficients3D,
    StencilBuilderContext,
)
from drbx.native.fci_boundaries import (  # noqa: E402
    BC_NEUMANN,
    LocalBoundaryFaceBC3D,
    LocalBoundaryFaceTrace3D,
)
from drbx.native.fci_operators import (  # noqa: E402
    local_curvature_conservative_op,
    local_curvature_upwind_conservative_op,
    local_grad_parallel_op_conservative,
    local_parallel_div_b_op,
    local_parallel_flux_div_op,
)
from test_fci_geometry_axis_regular_curvature import _build_polar_geometry  # noqa: E402
from axis_regular_operator_support import scalar_field_halo  # noqa: E402


def _fixture():
    geometry, domain = _build_polar_geometry(shape=(6, 12, 5))
    r, theta, z = jnp.meshgrid(
        geometry.grid.x.centers,
        geometry.grid.y.centers,
        geometry.grid.z.centers,
        indexing="ij",
    )
    field = scalar_field_halo(
        r, theta, z, lambda r, t, z: 1.0 + 0.2 * r + 0.1 * jnp.sin(t) + 0.3 * z
    )
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    from drbx.geometry import build_local_conservative_stencil_from_field

    return geometry, domain, build_local_conservative_stencil_from_field(
        field, geometry, context
    )


def _constant_coefficients(geometry, *, value):
    return LocalCurvatureFaceCoefficients3D(
        layout=geometry.layout,
        x=jnp.full(geometry.layout.face_control_shape(0), value),
        y=jnp.full(geometry.layout.face_control_shape(1), value),
        z=jnp.full(geometry.layout.face_control_shape(2), value),
    )


def _trace(geometry, *, value=7.0, axis=2, side="upper"):
    faces = [geometry.layout.face_control_shape(a) for a in range(3)]
    values = [jnp.full(shape, value, dtype=jnp.float64) for shape in faces]
    masks = [jnp.zeros(shape, dtype=bool) for shape in faces]
    index = -1 if side == "upper" else 0
    if axis == 0:
        masks[axis] = masks[axis].at[index].set(True)
    elif axis == 1:
        masks[axis] = masks[axis].at[:, index, :].set(True)
    else:
        masks[axis] = masks[axis].at[:, :, index].set(True)
    return LocalBoundaryFaceTrace3D(*values, *masks, geometry.layout)


def test_parallel_wall_flux_uses_explicit_trace_over_ghost_midpoint():
    geometry, domain, local = _fixture()
    baseline = local_parallel_flux_div_op(local, geometry, domain)
    traced = local_parallel_flux_div_op(
        local, geometry, domain, boundary_trace=_trace(geometry, value=9.0)
    )
    assert jnp.max(jnp.abs(traced - baseline)) > 1.0e-8


def test_compatible_parallel_gradient_forwards_the_same_trace():
    geometry, domain, local = _fixture()
    div_b = local_parallel_div_b_op(
        local.replace(
            x=local.x.replace(
                center=jnp.ones_like(local.x.center),
                minus=jnp.ones_like(local.x.minus),
                plus=jnp.ones_like(local.x.plus),
            ),
            face_values=type(local.face_values)(
                x=jnp.ones_like(local.face_values.x),
                y=jnp.ones_like(local.face_values.y),
                z=jnp.ones_like(local.face_values.z),
            ),
        ),
        geometry,
        domain,
    )
    trace = _trace(geometry, value=5.0)
    direct = local_parallel_flux_div_op(local, geometry, domain, boundary_trace=trace)
    compatible = local_grad_parallel_op_conservative(
        local, geometry, domain, div_b=div_b, boundary_trace=trace
    )
    field = jnp.asarray(local.x.center)
    assert jnp.allclose(compatible, direct - field * div_b)


def test_centered_curvature_explicit_trace_overrides_neumann_legacy_bc():
    geometry, domain, local = _fixture()
    coefficients = LocalCurvatureFaceCoefficients3D(
        layout=geometry.layout,
        x=jnp.zeros(geometry.layout.face_control_shape(0)),
        y=jnp.zeros(geometry.layout.face_control_shape(1)),
        z=jnp.ones(geometry.layout.face_control_shape(2)),
    )
    legacy = LocalBoundaryFaceBC3D.empty(geometry.layout)
    legacy = type(legacy)(
        kind_x=legacy.kind_x, kind_y=legacy.kind_y,
        kind_z=legacy.kind_z.at[:, :, -1].set(BC_NEUMANN),
        value_x=legacy.value_x, value_y=legacy.value_y, value_z=legacy.value_z,
        mask_x=legacy.mask_x, mask_y=legacy.mask_y,
        mask_z=legacy.mask_z.at[:, :, -1].set(True), layout=geometry.layout,
    )
    baseline = local_curvature_conservative_op(
        local, geometry, coefficients, domain=domain, face_bc=legacy
    )
    traced = local_curvature_conservative_op(
        local, geometry, coefficients, domain=domain, face_bc=legacy,
        boundary_trace=_trace(geometry, value=11.0),
    )
    assert jnp.max(jnp.abs(traced - baseline)) > 1.0e-8


def test_axis_regular_lower_trace_cannot_inject_curvature_flux():
    geometry, domain, local = _fixture()
    faces = [geometry.layout.face_control_shape(a) for a in range(3)]
    coefficients = LocalCurvatureFaceCoefficients3D(
        layout=geometry.layout,
        x=jnp.zeros(faces[0]).at[0].set(1.0),
        y=jnp.zeros(faces[1]), z=jnp.zeros(faces[2]),
    )
    baseline = local_curvature_conservative_op(
        local, geometry, coefficients, domain=domain,
        axis_regular_axes=(True, False, False),
    )
    traced = local_curvature_conservative_op(
        local, geometry, coefficients, domain=domain,
        axis_regular_axes=(True, False, False),
        boundary_trace=_trace(geometry, value=100.0, axis=0, side="lower"),
    )
    assert jnp.allclose(traced, baseline)


def test_legacy_calls_and_upwind_curvature_remain_available():
    geometry, domain, local = _fixture()
    coefficients = _constant_coefficients(geometry, value=0.0)
    parallel_a = local_parallel_flux_div_op(local, geometry, domain)
    parallel_b = local_parallel_flux_div_op(
        local, geometry, domain, boundary_trace=None
    )
    centered_a = local_curvature_conservative_op(local, geometry, coefficients, domain=domain)
    centered_b = local_curvature_conservative_op(
        local, geometry, coefficients, domain=domain, boundary_trace=None
    )
    upwind = local_curvature_upwind_conservative_op(local, geometry, coefficients, domain=domain)
    assert jnp.allclose(parallel_a, parallel_b)
    assert jnp.allclose(centered_a, centered_b)
    assert jnp.all(jnp.isfinite(upwind))
