"""Focused tests for the experimental compatible-flux Poisson bracket."""

from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import replace

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
from drbx.native.fci_boundaries import LocalBoundaryFaceTrace3D  # noqa: E402
from drbx.native.fci_operators import (  # noqa: E402
    local_poisson_bracket_compatible_flux_op,
    local_poisson_bracket_op,
)
from axis_regular_operator_support import owned, polar_fixture, scalar_field_halo  # noqa: E402
from test_fci_operators_domain_decomp import _build_domain, _build_local_geometry  # noqa: E402


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
