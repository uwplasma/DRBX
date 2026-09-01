from types import SimpleNamespace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drbx.geometry import (
    HaloLayout3D,
    LocalControlVolumeCellGeometry3D,
    LocalCurvatureFaceCoefficients3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    ShardSpec3D,
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
    build_local_curvature_face_coefficients,
)
from drbx.native.fci_boundaries import (
    CoordinateFaceValues3D,
    LocalBoundaryFaceTrace3D,
    LocalControlVolumeFaceRows3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentReconstruction3D,
)
from drbx.native.fci_operators import (
    _curvature_bc_characteristic_wall_states,
    local_curvature_conservative_op,
    local_curvature_production_path_op,
)
from drbx.native.fci_curvature_production_flux import (
    curvature_face_linearized_fluctuations,
    curvature_strict_principal_matrix,
)
from test_fci_operators_domain_decomp import (
    _build_domain,
    _build_local_geometry,
)


def _legacy_axis_face_values_for_weighted_sum(stencil, *, axis: int) -> jnp.ndarray:
    """Return legacy arithmetic face values for the weighted-sum check."""

    center = jnp.asarray(stencil.center, dtype=jnp.float64)
    minus = jnp.asarray(stencil.minus, dtype=jnp.float64)
    plus = jnp.asarray(stencil.plus, dtype=jnp.float64)
    if center.ndim != 3:
        raise ValueError(f"stencil center must be 3D, got shape {center.shape}")
    center_slice = [slice(None)] * center.ndim
    center_slice[axis] = 0
    lower = 0.5 * (center[tuple(center_slice)] + minus[tuple(center_slice)])
    upper_faces = 0.5 * (center + plus)
    return jnp.concatenate(
        (jnp.expand_dims(lower, axis=axis), upper_faces),
        axis=axis,
    )


def _zero_physical_halo_geometry(
    *, periodic_axes=(False, False, True)
):
    layout = HaloLayout3D((3, 4, 5), halo_width=1)
    shape = layout.cell_halo_shape
    domain = LocalDomain3D(
        ShardSpec3D(
            global_shape=(3, 4, 5),
            owned_start=(0, 0, 0),
            owned_stop=(3, 4, 5),
            shard_index=(0, 0, 0),
            shard_counts=(1, 1, 1),
            periodic_axes=periodic_axes,
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


def test_curvature_wall_edges_are_shared_across_periodic_seams():
    """Physical-wall edge patches must retain tangential periodic topology."""

    jax.config.update("jax_enable_x64", True)
    geometry, domain = _zero_physical_halo_geometry(
        periodic_axes=(False, True, True)
    )
    h = geometry.layout.halo_width
    nx, ny, nz = geometry.layout.owned_shape
    wall_i = h + nx
    wall_b = geometry.face_bfield.x.B_contra_halo
    # Local face geometry carries assembled tangential halos.  Populate this
    # synthetic fixture with the same periodic contract before imposing the
    # nontrivial wall profile.
    theta_profile = jnp.pad(
        jnp.arange(ny, dtype=jnp.float64), (h, h), mode="wrap"
    )[:, None]
    eta_profile = jnp.pad(
        jnp.arange(nz, dtype=jnp.float64), (h, h), mode="wrap"
    )[None, :]
    wall_b = wall_b.at[
        wall_i, :, :, 2
    ].set(jnp.broadcast_to(theta_profile, (ny + 2 * h, nz + 2 * h)))
    wall_b = wall_b.at[
        wall_i, :, :, 1
    ].set(jnp.broadcast_to(eta_profile, (ny + 2 * h, nz + 2 * h)))
    geometry.face_bfield.x.B_contra_halo = wall_b

    coefficients = build_local_curvature_face_coefficients(geometry, domain)

    np.testing.assert_allclose(
        np.asarray(coefficients.y[:, 0, :]),
        np.asarray(coefficients.y[:, -1, :]),
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(coefficients.z[:, :, 0]),
        np.asarray(coefficients.z[:, :, -1]),
        rtol=0.0,
        atol=1.0e-13,
    )
    assert float(jnp.max(jnp.abs(coefficients.y[-1]))) > 0.1
    assert float(jnp.max(jnp.abs(coefficients.z[-1]))) > 0.1
    div_q = (
        coefficients.x[1:] - coefficients.x[:-1]
        + coefficients.y[:, 1:] - coefficients.y[:, :-1]
        + coefficients.z[:, :, 1:] - coefficients.z[:, :, :-1]
    )
    assert float(jnp.max(jnp.abs(div_q))) < 1.0e-13


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


def _constant_radial_wall_trace(layout, value):
    trace = LocalBoundaryFaceTrace3D.empty(layout)
    value_x = trace.value_x.at[0].set(value).at[-1].set(value)
    mask_x = trace.mask_x.at[0].set(True).at[-1].set(True)
    return LocalBoundaryFaceTrace3D(
        value_x=value_x,
        value_y=trace.value_y,
        value_z=trace.value_z,
        mask_x=mask_x,
        mask_y=trace.mask_y,
        mask_z=trace.mask_z,
        layout=layout,
    )


def _curvature_projector(state, *, tau, normal, incoming_positive):
    matrix = np.asarray(
        normal * curvature_strict_principal_matrix(
            jnp.asarray(state, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            tau,
        )
    )
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    inverse = np.linalg.inv(eigenvectors)
    incoming = (
        np.real(eigenvalues) > 1.0e-10
        if incoming_positive
        else np.real(eigenvalues) < -1.0e-10
    )
    return np.real(eigenvectors @ np.diag(incoming.astype(float)) @ inverse)


def _curvature_least_residual_state(
    interior, trace, *, tau, normal, incoming_positive
):
    matrix = np.asarray(
        normal * curvature_strict_principal_matrix(
            jnp.asarray(trace, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            tau,
        )
    )
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    incoming = (
        np.real(eigenvalues) > 1.0e-10
        if incoming_positive
        else np.real(eigenvalues) < -1.0e-10
    )
    basis = np.real(eigenvectors[:, incoming])
    coefficients = np.linalg.lstsq(
        basis[:3], np.asarray(trace)[:3] - np.asarray(interior)[:3],
        rcond=None,
    )[0]
    return np.asarray(interior) + basis @ coefficients


def test_bc_characteristic_wall_state_solves_full_thermodynamic_residual_on_electron_subspace():
    interior = jnp.asarray((1.2, 0.9, 1.1, 0.2), dtype=jnp.float64)
    trace = jnp.asarray((1.15, 0.95, 1.05, 0.0), dtype=jnp.float64)
    exterior, face, fallback = _curvature_bc_characteristic_wall_states(
        interior,
        trace,
        jnp.asarray(1.0),
        0.7,
        jnp.asarray(1.0),
        interior_on_right=True,
    )
    expected = _curvature_least_residual_state(
        interior, trace, tau=0.7, normal=1.0, incoming_positive=True
    )
    np.testing.assert_allclose(exterior, expected, atol=3.0e-12, rtol=0.0)
    assert np.linalg.norm(np.asarray(exterior[:3] - trace[:3])) < np.linalg.norm(
        np.asarray(interior[:3] - trace[:3])
    )
    incoming = _curvature_projector(
        trace, tau=0.7, normal=1.0, incoming_positive=True
    )
    outgoing = np.eye(4) - incoming
    np.testing.assert_allclose(
        outgoing @ np.asarray(exterior - interior),
        np.zeros(4),
        atol=3.0e-13,
        rtol=0.0,
    )
    assert not np.allclose(np.asarray(exterior), np.asarray(trace))
    np.testing.assert_allclose(face, trace, atol=0.0, rtol=0.0)
    assert not bool(fallback)


def test_bc_characteristic_wall_state_solves_full_thermodynamic_residual_on_ion_subspace():
    interior = jnp.asarray((1.2, 0.9, 1.1, 0.2), dtype=jnp.float64)
    trace = jnp.asarray((1.15, 0.95, 1.05, 0.0), dtype=jnp.float64)
    exterior, face, _fallback = _curvature_bc_characteristic_wall_states(
        interior,
        trace,
        jnp.asarray(1.0),
        0.7,
        jnp.asarray(1.0),
        interior_on_right=False,
    )
    expected = _curvature_least_residual_state(
        interior, trace, tau=0.7, normal=1.0, incoming_positive=False
    )
    np.testing.assert_allclose(exterior, expected, atol=3.0e-12, rtol=0.0)
    assert np.linalg.norm(np.asarray(exterior[:3] - trace[:3])) < np.linalg.norm(
        np.asarray(interior[:3] - trace[:3])
    )
    incoming = _curvature_projector(
        trace, tau=0.7, normal=1.0, incoming_positive=False
    )
    outgoing = np.eye(4) - incoming
    np.testing.assert_allclose(
        outgoing @ np.asarray(exterior - interior),
        np.zeros(4),
        atol=3.0e-13,
        rtol=0.0,
    )
    assert not np.allclose(np.asarray(exterior), np.asarray(trace))
    np.testing.assert_allclose(face, trace, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("interior_on_right", "normal", "incoming_positive"),
    (
        (True, -1.0, True),
        (False, -1.0, False),
    ),
)
def test_bc_characteristic_wall_state_reverses_mode_family_with_normal(
    interior_on_right, normal, incoming_positive
):
    interior = jnp.asarray((1.2, 0.9, 1.1, 0.2), dtype=jnp.float64)
    trace = jnp.asarray((1.15, 0.95, 1.05, 0.0), dtype=jnp.float64)
    exterior, _face, fallback = _curvature_bc_characteristic_wall_states(
        interior,
        trace,
        jnp.asarray(1.0),
        0.7,
        jnp.asarray(normal),
        interior_on_right=interior_on_right,
    )
    expected = _curvature_least_residual_state(
        interior,
        trace,
        tau=0.7,
        normal=normal,
        incoming_positive=incoming_positive,
    )
    np.testing.assert_allclose(exterior, expected, atol=3.0e-12, rtol=0.0)
    assert not bool(fallback)


def test_bc_characteristic_wall_state_is_jittable_and_tangent_is_owner():
    interior = jnp.asarray((1.2, 0.9, 1.1, 0.2), dtype=jnp.float64)
    trace = jnp.asarray((1.15, 0.95, 1.05, 0.0), dtype=jnp.float64)

    @jax.jit
    def close(owner, candidate, normal):
        return _curvature_bc_characteristic_wall_states(
            owner,
            candidate,
            jnp.asarray(1.0),
            0.7,
            normal,
            interior_on_right=True,
        )[0]

    np.testing.assert_allclose(
        close(interior, trace, jnp.asarray(0.0)), interior, atol=0.0, rtol=0.0
    )
    assert bool(jnp.all(jnp.isfinite(close(interior, trace, jnp.asarray(1.0)))))


def test_bc_characteristic_invalid_trace_propagates_without_owner_fallback():
    interior = jnp.asarray((1.2, 0.9, 1.1, 0.2), dtype=jnp.float64)
    trace = jnp.asarray((jnp.nan, 0.95, 1.05, 0.0), dtype=jnp.float64)
    exterior, face, invalid = _curvature_bc_characteristic_wall_states(
        interior,
        trace,
        jnp.asarray(1.0),
        0.7,
        jnp.asarray(1.0),
        interior_on_right=True,
    )
    assert bool(invalid)
    assert not bool(jnp.all(jnp.isfinite(exterior)))
    assert not bool(jnp.all(jnp.isfinite(face)))


def test_bc_characteristic_curvature_preserves_non_equilibrium_neumann_constant():
    geometry, domain, _stencil, coefficients = _operator_fixture()
    layout = geometry.layout
    context = StencilBuilderContext(layout=layout, domain=domain)
    state = (1.2, 0.9, 1.1, 0.0)
    fields = tuple(
        build_local_conservative_stencil_from_field(
            jnp.full(layout.cell_halo_shape, value, dtype=jnp.float64),
            geometry,
            context,
        )
        for value in state
    )
    traces = tuple(_constant_radial_wall_trace(layout, value) for value in state)
    result = local_curvature_production_path_op(
        fields,
        geometry,
        coefficients,
        tau=0.7,
        domain=domain,
        boundary_traces=traces,
        wall_flux_closure=(
            "bc-characteristic-operator-trace-canonical-face-state"
        ),
    )
    np.testing.assert_allclose(result, 0.0, atol=2.0e-12, rtol=0.0)


def test_bc_characteristic_curvature_applies_dirichlet_vorticity_only_at_wall():
    geometry, domain, _stencil, coefficients = _operator_fixture((5, 3, 2))
    layout = geometry.layout
    context = StencilBuilderContext(layout=layout, domain=domain)
    state = (1.2, 0.9, 1.1, 0.2)
    wall = (1.2, 0.9, 1.1, 0.0)
    fields = tuple(
        build_local_conservative_stencil_from_field(
            jnp.full(layout.cell_halo_shape, value, dtype=jnp.float64),
            geometry,
            context,
        )
        for value in state
    )
    traces = tuple(_constant_radial_wall_trace(layout, value) for value in wall)
    result = local_curvature_production_path_op(
        fields,
        geometry,
        coefficients,
        tau=0.7,
        domain=domain,
        boundary_traces=traces,
        wall_flux_closure=(
            "bc-characteristic-operator-trace-canonical-face-state"
        ),
    )
    assert float(jnp.linalg.norm(result[jnp.asarray((0, -1))])) > 0.0
    np.testing.assert_allclose(result[1:-1], 0.0, atol=2.0e-12, rtol=0.0)


def test_production_curvature_all_axes_has_directional_diagnostics_and_constant_null():
    """The owner-face path is active on x/y/z and preserves constants."""
    geometry, domain, _stencil, coefficients = _operator_fixture()
    layout = geometry.layout
    context = StencilBuilderContext(layout=layout, domain=domain)
    fields = tuple(
        build_local_conservative_stencil_from_field(
            jnp.full(layout.cell_halo_shape, value, dtype=jnp.float64),
            geometry,
            context,
        )
        for value in (1.0, 1.0, 1.0, 0.0)
    )
    result, diagnostics = local_curvature_production_path_op(
        fields,
        geometry,
        coefficients,
        tau=0.7,
        domain=domain,
        return_diagnostics=True,
    )
    assert result.shape == geometry.owned_shape + (4,)
    assert set(diagnostics) == {"directional_residual"}
    assert diagnostics["directional_residual"].shape == (3,) + geometry.owned_shape + (4,)
    np.testing.assert_allclose(
        jnp.sum(diagnostics["directional_residual"], axis=0),
        result,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(result, 0.0, atol=2.0e-12, rtol=0.0)


def test_production_curvature_physical_wall_is_one_sided_and_finite():
    geometry, domain, _stencil, coefficients = _operator_fixture()
    layout = geometry.layout
    context = StencilBuilderContext(layout=layout, domain=domain)
    fields = []
    for value in (1.1, 0.9, 1.2, 0.05):
        values = jnp.full(layout.cell_halo_shape, value, dtype=jnp.float64)
        # Make the first radial interior cell differ from the shared exterior
        # state; this exercises the lower physical wall face scatter.
        values = values.at[1, 1:-1, 1:-1].set(value * 1.02)
        fields.append(build_local_conservative_stencil_from_field(values, geometry, context))
    result = local_curvature_production_path_op(
        tuple(fields), geometry, coefficients, tau=0.7, domain=domain
    )
    assert result.shape == geometry.owned_shape + (4,)
    assert bool(jnp.all(jnp.isfinite(result)))


def test_production_curvature_face_state_uses_canonical_faces_and_interior_wall_trace():
    """The characteristic matrix sees ordinary face values, not wall exterior."""
    geometry, domain, _stencil, coefficients = _operator_fixture()
    layout = geometry.layout
    context = StencilBuilderContext(layout=layout, domain=domain)
    ii, jj, kk = jnp.meshgrid(
        jnp.arange(layout.cell_halo_shape[0]),
        jnp.arange(layout.cell_halo_shape[1]),
        jnp.arange(layout.cell_halo_shape[2]),
        indexing="ij",
    )
    fields = tuple(
        build_local_conservative_stencil_from_field(
            base + scale * (0.001 * ii**3 + 0.007 * ii**2 + 0.2 * ii + jnp.sin(0.6 * jj) - 0.3 * jnp.cos(0.4 * kk)),
            geometry,
            context,
        )
        for base, scale in ((1.0, 0.02), (1.0, 0.015), (1.0, 0.01), (0.0, 0.03))
    )
    baseline = local_curvature_production_path_op(
        fields, geometry, coefficients, tau=0.7, domain=domain
    )

    # Change only ordinary radial canonical face values.  The one-sided
    # reconstruction inputs stay untouched, so a response proves A is built
    # from ConservativeStencil3D.face_values.
    ordinary = []
    wall_changed = []
    for field_index, stencil in enumerate(fields):
        face_values = stencil.face_values
        x_changed = face_values.x
        x_changed = x_changed.at[1:-1].add(0.25 * (field_index + 1))
        ordinary.append(
            stencil.replace(
                face_values=CoordinateFaceValues3D(
                    x=x_changed, y=face_values.y, z=face_values.z
                )
            )
        )
        # Wall canonical values are deliberately absurd.  They must not be
        # used for A: physical wall A is linearized at the adjacent interior
        # reconstructed trace.
        x_wall = face_values.x.at[0].add(5.0 * (field_index + 1))
        x_wall = x_wall.at[-1].add(7.0 * (field_index + 1))
        wall_changed.append(
            stencil.replace(
                face_values=CoordinateFaceValues3D(
                    x=x_wall, y=face_values.y, z=face_values.z
                )
            )
        )
    changed = local_curvature_production_path_op(
        tuple(ordinary), geometry, coefficients, tau=0.7, domain=domain
    )
    assert float(jnp.linalg.norm(changed - baseline)) > 1.0e-10
    wall_result = local_curvature_production_path_op(
        tuple(wall_changed), geometry, coefficients, tau=0.7, domain=domain
    )
    np.testing.assert_allclose(wall_result, baseline, atol=2.0e-12, rtol=2.0e-12)


def test_production_curvature_identity_control_volume_matches_no_cv_measure():
    """An uncut owner topology must use the same transverse measure as no CV."""
    geometry, domain, _stencil, coefficients = _operator_fixture((4, 3, 2))
    layout = geometry.layout
    context = StencilBuilderContext(layout=layout, domain=domain)
    ii, jj, kk = jnp.meshgrid(
        jnp.arange(layout.cell_halo_shape[0]),
        jnp.arange(layout.cell_halo_shape[1]),
        jnp.arange(layout.cell_halo_shape[2]),
        indexing="ij",
    )
    fields = []
    for offset, scale in ((1.0, 0.03), (1.0, 0.02), (1.0, 0.01), (0.0, 0.04)):
        values = offset + scale * (0.7 * ii + 0.3 * jj - 0.2 * kk)
        fields.append(
            build_local_conservative_stencil_from_field(
                values.astype(jnp.float64), geometry, context
            )
        )

    raw_volume = (
        jnp.asarray(geometry.cell_metric.J_owned, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64)
    )
    cells = LocalControlVolumeCellGeometry3D.identity(
        layout,
        volume=raw_volume,
        centroid=jnp.zeros(layout.owned_shape + (3,), dtype=jnp.float64),
    )
    control_volume = LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=geometry.regular_face_geometry,
        irregular_faces=LocalControlVolumeFaceRows3D.empty(layout),
        reconstruction=LocalMomentReconstruction3D.empty(layout),
    )
    no_cv = local_curvature_production_path_op(
        tuple(fields), geometry, coefficients, tau=0.7, domain=domain
    )
    with_cv = local_curvature_production_path_op(
        tuple(fields), geometry, coefficients, tau=0.7, domain=domain,
        control_volume_geometry=control_volume,
    )
    np.testing.assert_allclose(with_cv, no_cv, atol=2.0e-10, rtol=2.0e-10)


def test_production_curvature_axis_activation_and_wall_orientation():
    """Nonconstant transverse data activates theta/eta; reversing Q reverses x."""
    geometry, domain, _stencil, coefficients = _operator_fixture()
    layout = geometry.layout
    context = StencilBuilderContext(layout=layout, domain=domain)
    ii, jj, kk = jnp.meshgrid(
        jnp.arange(layout.cell_halo_shape[0]),
        jnp.arange(layout.cell_halo_shape[1]),
        jnp.arange(layout.cell_halo_shape[2]),
        indexing="ij",
    )
    fields = tuple(
        build_local_conservative_stencil_from_field(
            base + 0.01 * jnp.sin(0.7 * jj) + 0.008 * jnp.cos(0.5 * kk),
            geometry,
            context,
        )
        for base in (1.0, 1.0, 1.0, 0.0)
    )
    full, full_info = local_curvature_production_path_op(
        fields, geometry, coefficients, tau=0.7, domain=domain,
        return_diagnostics=True,
    )
    assert float(jnp.linalg.norm(full_info["directional_residual"][1])) > 0.0
    assert float(jnp.linalg.norm(full_info["directional_residual"][2])) > 0.0
    plus_coefficients = LocalCurvatureFaceCoefficients3D(
        layout=layout, x=coefficients.x, y=jnp.zeros_like(coefficients.y),
        z=jnp.zeros_like(coefficients.z)
    )
    minus_coefficients = LocalCurvatureFaceCoefficients3D(
        layout=layout, x=-coefficients.x, y=jnp.zeros_like(coefficients.y),
        z=jnp.zeros_like(coefficients.z)
    )
    plus, plus_info = local_curvature_production_path_op(
        fields, geometry, plus_coefficients, tau=0.7, domain=domain,
        return_diagnostics=True,
    )
    minus, minus_info = local_curvature_production_path_op(
        fields, geometry, minus_coefficients, tau=0.7, domain=domain,
        return_diagnostics=True,
    )
    assert bool(jnp.all(jnp.isfinite(plus)))
    assert float(jnp.linalg.norm(plus_info["directional_residual"][0])) > 0.0
    # Reversing the transport orientation exchanges incoming/outgoing waves;
    # the p=1 split therefore is not an odd centered operator.  It must,
    # however, change the one-sided wall update rather than silently dropping
    # the boundary face.
    assert float(
        jnp.linalg.norm(
            minus_info["directional_residual"][0]
            - plus_info["directional_residual"][0]
        )
    ) > 0.0
    assert float(jnp.linalg.norm(plus_info["directional_residual"][0][0])) > 0.0


def test_production_curvature_directional_residual_closes_for_both_orientations():
    """Directional production residuals close for either face orientation."""
    geometry, domain, _stencil, coefficients = _operator_fixture()
    layout = geometry.layout
    context = StencilBuilderContext(layout=layout, domain=domain)
    ii, jj, kk = jnp.meshgrid(
        jnp.arange(layout.cell_halo_shape[0]),
        jnp.arange(layout.cell_halo_shape[1]),
        jnp.arange(layout.cell_halo_shape[2]),
        indexing="ij",
    )
    fields = tuple(
        build_local_conservative_stencil_from_field(
            base + scale * (0.2 * ii + jnp.sin(0.6 * jj) - 0.3 * jnp.cos(0.4 * kk)),
            geometry,
            context,
        )
        for base, scale in ((1.0, 0.02), (1.0, 0.015), (1.0, 0.01), (0.0, 0.03))
    )
    positive, positive_info = local_curvature_production_path_op(
        fields, geometry, coefficients, tau=0.7, domain=domain,
        return_diagnostics=True,
    )
    negative_coefficients = LocalCurvatureFaceCoefficients3D(
        layout=layout,
        x=-coefficients.x,
        y=-coefficients.y,
        z=-coefficients.z,
    )
    negative, negative_info = local_curvature_production_path_op(
        fields, geometry, negative_coefficients, tau=0.7, domain=domain,
        return_diagnostics=True,
    )
    for result, info in ((positive, positive_info), (negative, negative_info)):
        np.testing.assert_allclose(
            np.sum(np.asarray(info["directional_residual"]), axis=0),
            np.asarray(result),
            atol=2.0e-12,
            rtol=2.0e-12,
        )
    assert bool(jnp.all(jnp.isfinite(positive)))
    assert bool(jnp.all(jnp.isfinite(negative)))
    assert float(jnp.linalg.norm(negative - positive)) > 0.0


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
    x_face = _legacy_axis_face_values_for_weighted_sum(stencil.x, axis=0)
    y_face = _legacy_axis_face_values_for_weighted_sum(stencil.y, axis=1)
    z_face = _legacy_axis_face_values_for_weighted_sum(stencil.z, axis=2)
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
