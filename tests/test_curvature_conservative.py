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
    _radial_characteristic_fine_glue_integrated_residual,
    _radial_fine_glue_sat_integrated_residual,
    _bound_groupwise_transition_trace_correction,
    _constrain_groupwise_transition_flux_correction,
    _local_axis_face_values_from_stencil,
    local_curvature_conservative_components_op,
    local_curvature_conservative_op,
    local_curvature_production_path_op,
)
from drbx.native.fci_drb_EB_rhs import (
    LocalFciDrbEBRhs,
    background_curvature_characteristic_absolute_matrix,
    background_curvature_characteristic_metric,
)
from drbx.native.fci_curvature_production_flux import (
    curvature_face_linearized_fluctuations,
    curvature_strict_principal_matrix,
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


def test_bc_characteristic_wall_state_solves_electron_neumann_rows():
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
    # Positive-normal lower-wall inflow consists of the two electron-family
    # modes.  The copied/extrapolated density and Te residual rows are exact.
    np.testing.assert_allclose(exterior[:2], trace[:2], atol=2.0e-13, rtol=0.0)
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


def test_bc_characteristic_wall_state_solves_ion_neumann_row():
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
    # Positive-normal upper-wall inflow is the single ion-family mode, so the
    # copied/extrapolated Ti residual is the one imposed physical row.
    np.testing.assert_allclose(exterior[2], trace[2], atol=2.0e-13, rtol=0.0)
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
    ("interior_on_right", "normal", "constrained_components"),
    (
        (True, -1.0, (2,)),
        (False, -1.0, (0, 1)),
    ),
)
def test_bc_characteristic_wall_state_reverses_mode_family_with_normal(
    interior_on_right, normal, constrained_components
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
    np.testing.assert_allclose(
        np.asarray(exterior)[list(constrained_components)],
        np.asarray(trace)[list(constrained_components)],
        atol=2.0e-13,
        rtol=0.0,
    )
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


def test_equilibrium_wall_default_ignores_supplied_bc_traces_bitwise():
    geometry, domain, _stencil, coefficients = _operator_fixture()
    layout = geometry.layout
    context = StencilBuilderContext(layout=layout, domain=domain)
    fields = tuple(
        build_local_conservative_stencil_from_field(
            jnp.full(layout.cell_halo_shape, value, dtype=jnp.float64),
            geometry,
            context,
        )
        for value in (1.2, 0.9, 1.1, 0.2)
    )
    traces = tuple(
        _constant_radial_wall_trace(layout, value)
        for value in (3.0, 4.0, 5.0, 6.0)
    )
    baseline = local_curvature_production_path_op(
        fields, geometry, coefficients, tau=0.7, domain=domain
    )
    with_unused_traces = local_curvature_production_path_op(
        fields,
        geometry,
        coefficients,
        tau=0.7,
        domain=domain,
        boundary_traces=traces,
    )
    np.testing.assert_array_equal(with_unused_traces, baseline)


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
    assert diagnostics["directional_residual"].shape == (3,) + geometry.owned_shape + (4,)
    assert diagnostics["directional_centered_transfer"].shape == (
        (3,) + geometry.owned_shape + (4,)
    )
    assert diagnostics["directional_characteristic_dissipation"].shape == (
        (3,) + geometry.owned_shape + (4,)
    )
    assert diagnostics["dissipation_norm"].shape == (3,)
    np.testing.assert_allclose(
        diagnostics["directional_centered_transfer"]
        + diagnostics["directional_characteristic_dissipation"],
        diagnostics["directional_residual"],
        atol=0.0,
        rtol=0.0,
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


def test_production_curvature_upper_wall_ablation_removes_only_upper_face_lane():
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
            base + scale * (0.2 * ii + 0.03 * jj - 0.02 * kk),
            geometry,
            context,
        )
        for base, scale in (
            (1.0, 0.02), (1.0, 0.015), (1.0, 0.01), (0.0, 0.03)
        )
    )
    full, full_info = local_curvature_production_path_op(
        fields, geometry, coefficients, tau=0.7, domain=domain,
        return_diagnostics=True,
    )
    ablated, ablated_info = local_curvature_production_path_op(
        fields, geometry, coefficients, tau=0.7, domain=domain,
        radial_ablation="upper-physical-face", return_diagnostics=True,
    )
    full_radial = np.asarray(full_info["radial_provenance_residual"])
    ablated_radial = np.asarray(ablated_info["radial_provenance_residual"])
    np.testing.assert_allclose(ablated_radial[1], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        ablated_radial[[0, 2, 3, 4]],
        full_radial[[0, 2, 3, 4]],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(ablated),
        np.asarray(full) - full_radial[1],
        atol=2.0e-12,
        rtol=2.0e-12,
    )


def test_production_curvature_last_interior_ablation_preserves_wall_face():
    geometry, domain, _stencil, coefficients = _operator_fixture((4, 3, 2))
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
            base + scale * (ii**2 + 0.1 * jj - 0.05 * kk),
            geometry,
            context,
        )
        for base, scale in (
            (1.0, 0.01), (1.0, 0.008), (1.0, 0.006), (0.0, 0.012)
        )
    )
    full, full_info = local_curvature_production_path_op(
        fields, geometry, coefficients, tau=0.7, domain=domain,
        return_diagnostics=True,
    )
    ablated, ablated_info = local_curvature_production_path_op(
        fields, geometry, coefficients, tau=0.7, domain=domain,
        radial_ablation="last-interior-face", return_diagnostics=True,
    )
    np.testing.assert_allclose(
        np.asarray(ablated_info["radial_provenance_residual"])[1],
        np.asarray(full_info["radial_provenance_residual"])[1],
        atol=0.0,
        rtol=0.0,
    )
    assert float(jnp.linalg.norm(ablated - full)) > 0.0


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


def test_production_curvature_diagnostic_split_is_exact_and_orientation_even_odd():
    """The |A| lane is even in face orientation and the path lane is odd."""
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
        np.testing.assert_allclose(
            np.asarray(info["directional_centered_transfer"])
            + np.asarray(info["directional_characteristic_dissipation"]),
            np.asarray(info["directional_residual"]),
            atol=2.0e-12,
            rtol=2.0e-12,
        )
    np.testing.assert_allclose(
        np.asarray(negative_info["directional_centered_transfer"]),
        -np.asarray(positive_info["directional_centered_transfer"]),
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    np.testing.assert_allclose(
        np.asarray(negative_info["directional_characteristic_dissipation"]),
        np.asarray(positive_info["directional_characteristic_dissipation"]),
        atol=2.0e-11,
        rtol=2.0e-11,
    )


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


def test_radial_donor_cell_discriminator_replaces_only_interior_x_faces():
    geometry, domain, _stencil, _coefficients = _operator_fixture((4, 3, 2))
    layout = geometry.layout
    field_halo = jnp.arange(
        np.prod(layout.cell_halo_shape), dtype=jnp.float64
    ).reshape(layout.cell_halo_shape)
    stencil = build_local_conservative_stencil_from_field(
        field_halo,
        geometry,
        StencilBuilderContext(layout=layout, domain=domain),
    )
    qx = jnp.asarray(
        [
            [[0.0]],
            [[1.0]],
            [[-2.0]],
            [[0.5]],
            [[0.0]],
        ],
        dtype=jnp.float64,
    )
    qx = jnp.broadcast_to(qx, layout.face_control_shape(axis=0))
    coefficients = LocalCurvatureFaceCoefficients3D(
        layout=layout,
        x=qx,
        y=jnp.zeros(layout.face_control_shape(axis=1), dtype=jnp.float64),
        z=jnp.zeros(layout.face_control_shape(axis=2), dtype=jnp.float64),
    )

    centered_default = local_curvature_conservative_op(
        stencil, geometry, coefficients
    )
    centered_explicit = local_curvature_conservative_op(
        stencil,
        geometry,
        coefficients,
        radial_principal_face_scheme="centered",
    )
    np.testing.assert_array_equal(centered_default, centered_explicit)

    donor_face = jnp.asarray(stencil.face_values.x).at[1:-1].set(
        jnp.where(
            qx[1:-1] >= 0.0,
            jnp.asarray(stencil.x.center)[:-1],
            jnp.asarray(stencil.x.center)[1:],
        )
    )
    dx = jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)
    expected = (
        jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64)
        / jnp.asarray(geometry.cell_metric.J_owned, dtype=jnp.float64)
        * (qx[1:] * donor_face[1:] - qx[:-1] * donor_face[:-1])
        / dx
    )
    donor = local_curvature_conservative_op(
        stencil,
        geometry,
        coefficients,
        radial_principal_face_scheme="donor-cell",
    )
    np.testing.assert_allclose(donor, expected, atol=2.0e-12, rtol=2.0e-12)
    assert not np.array_equal(np.asarray(donor), np.asarray(centered_default))

    compiled = jax.jit(
        lambda value: local_curvature_conservative_op(
            value,
            geometry,
            coefficients,
            radial_principal_face_scheme="donor-cell",
        )
    )
    np.testing.assert_allclose(
        compiled(stencil), donor, atol=2.0e-12, rtol=2.0e-12
    )


def test_radial_donor_cell_discriminator_preserves_constants_and_flux_balance():
    geometry, domain, constant_stencil, coefficients = _operator_fixture((4, 3, 2))
    constant = local_curvature_conservative_op(
        constant_stencil,
        geometry,
        coefficients,
        radial_principal_face_scheme="donor-cell",
    )
    np.testing.assert_allclose(constant, 0.0, atol=2.0e-12, rtol=0.0)

    layout = geometry.layout
    field_halo = jnp.arange(
        np.prod(layout.cell_halo_shape), dtype=jnp.float64
    ).reshape(layout.cell_halo_shape)
    stencil = build_local_conservative_stencil_from_field(
        field_halo,
        geometry,
        StencilBuilderContext(layout=layout, domain=domain),
    )
    result = local_curvature_conservative_op(
        stencil,
        geometry,
        coefficients,
        radial_principal_face_scheme="donor-cell",
    )
    dx = jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)
    dy = jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64)
    dz = jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64)
    weighted = jnp.sum(
        result
        * jnp.asarray(geometry.cell_metric.J_owned)
        / jnp.asarray(geometry.cell_bfield.Bmag_owned)
        * dx
        * dy
        * dz
    )
    x_face = jnp.asarray(stencil.face_values.x)
    y_face = jnp.asarray(stencil.face_values.y)
    z_face = jnp.asarray(stencil.face_values.z)
    boundary_flux = jnp.sum(
        coefficients.x[-1] * x_face[-1] * dy[-1] * dz[-1]
    ) - jnp.sum(coefficients.x[0] * x_face[0] * dy[0] * dz[0])
    boundary_flux += jnp.sum(
        coefficients.y[:, -1] * y_face[:, -1] * dx[:, -1] * dz[:, -1]
    ) - jnp.sum(
        coefficients.y[:, 0] * y_face[:, 0] * dx[:, 0] * dz[:, 0]
    )
    boundary_flux += jnp.sum(
        coefficients.z[:, :, -1]
        * z_face[:, :, -1]
        * dx[:, :, -1]
        * dy[:, :, -1]
    ) - jnp.sum(
        coefficients.z[:, :, 0]
        * z_face[:, :, 0]
        * dx[:, :, 0]
        * dy[:, :, 0]
    )
    np.testing.assert_allclose(
        weighted, boundary_flux, atol=2.0e-11, rtol=2.0e-12
    )


def test_radial_donor_cell_discriminator_rejects_unknown_scheme():
    geometry, _domain, stencil, coefficients = _operator_fixture()
    with pytest.raises(ValueError, match="radial_principal_face_scheme"):
        local_curvature_conservative_op(
            stencil,
            geometry,
            coefficients,
            radial_principal_face_scheme="third-order",
        )


def test_radial_curvature_principal_face_environment_selector(monkeypatch):
    selector = LocalFciDrbEBRhs.__dataclass_fields__[
        "curvature_radial_principal_face_scheme"
    ].default_factory
    monkeypatch.delenv(
        "DRBX_CURVATURE_RADIAL_PRINCIPAL_FACE_SCHEME", raising=False
    )
    assert selector() == "centered"
    monkeypatch.setenv(
        "DRBX_CURVATURE_RADIAL_PRINCIPAL_FACE_SCHEME", "donor-cell"
    )
    assert selector() == "donor-cell"


def test_groupwise_transition_trace_limiter_preserves_mean_and_bounds():
    baseline = jnp.asarray((0.5, 0.5, 0.5, 2.0, 3.0, 7.0))
    correction = jnp.asarray((1.0, -1.0, 1.0, 0.4, -0.2, 9.0))
    fitted = baseline + correction
    minus = jnp.asarray((0.0, 0.0, 0.0, 1.0, 2.0, -100.0))
    plus = jnp.asarray((1.0, 1.0, 1.0, 4.0, 4.0, 100.0))
    groups = jnp.asarray((0, 0, 0, 1, 1, 2), dtype=jnp.int32)
    active = jnp.asarray((True, True, True, True, True, False))
    weights = np.asarray((1.0, 2.0, 1.0, 1.0, 2.0, 0.0))

    limited = jax.jit(
        lambda high: _bound_groupwise_transition_trace_correction(
            baseline,
            high,
            minus,
            plus,
            groups,
            active,
            num_groups=3,
        )
    )(fitted)
    limited = np.asarray(limited)

    np.testing.assert_allclose(limited[:3], (1.0, 0.0, 1.0))
    np.testing.assert_allclose(limited[3:5], np.asarray(fitted)[3:5])
    assert limited[5] == baseline[5]
    assert np.all(limited[active] >= np.minimum(minus, plus)[active])
    assert np.all(limited[active] <= np.maximum(minus, plus)[active])
    for group in (0, 1):
        select = np.asarray(groups) == group
        np.testing.assert_allclose(
            np.sum(weights[select] * (limited[select] - np.asarray(baseline)[select])),
            0.0,
            atol=2.0e-15,
            rtol=0.0,
        )


def test_groupwise_transition_flux_correction_is_conservative_and_noninjective():
    baseline = jnp.asarray((0.2, 0.5, -0.1, 1.0, 1.3, 9.0))
    fitted = jnp.asarray((0.9, -0.4, 0.7, 1.8, 0.6, -20.0))
    q_face = jnp.asarray((1.2, -0.7, 0.4, 0.5, 1.1, 3.0))
    minus = jnp.asarray((0.1, -0.2, 0.8, 0.2, 1.5, -4.0))
    plus = jnp.asarray((1.0, 0.6, -0.3, 1.4, 0.1, 7.0))
    weights = jnp.asarray((1.0, 2.0, 0.5, 1.5, 0.75, 4.0))
    groups = jnp.asarray((0, 0, 0, 1, 1, 2), dtype=jnp.int32)
    active = jnp.asarray((True, True, True, True, True, False))

    correction = jax.jit(
        lambda high: _constrain_groupwise_transition_flux_correction(
            baseline,
            high,
            q_face,
            minus,
            plus,
            weights,
            groups,
            active,
            num_groups=3,
        )
    )(fitted)
    correction = np.asarray(correction)

    assert np.all(np.isfinite(correction))
    assert correction[-1] == 0.0
    for group in (0, 1):
        select = np.asarray(active) & (np.asarray(groups) == group)
        group_weight = np.asarray(weights)[select]
        group_correction = correction[select]
        np.testing.assert_allclose(
            np.sum(group_weight * group_correction),
            0.0,
            atol=2.0e-14,
            rtol=0.0,
        )
        face_power = np.sum(
            group_weight
            * (np.asarray(minus)[select] - np.asarray(plus)[select])
            * group_correction
        )
        assert face_power <= 2.0e-14


def test_groupwise_transition_flux_correction_vanishes_without_trace_change():
    baseline = jnp.asarray((0.5, -0.2, 1.4))
    correction = _constrain_groupwise_transition_flux_correction(
        baseline,
        baseline,
        jnp.asarray((1.0, -0.4, 0.7)),
        jnp.asarray((0.1, 0.2, 0.3)),
        jnp.asarray((0.4, -0.1, 0.8)),
        jnp.asarray((1.0, 2.0, 0.5)),
        jnp.asarray((0, 0, 0), dtype=jnp.int32),
        jnp.asarray((True, True, True)),
        num_groups=1,
    )
    np.testing.assert_allclose(correction, 0.0, atol=0.0, rtol=0.0)


def test_directional_curvature_components_close_to_production_operator():
    jax.config.update("jax_enable_x64", True)
    geometry, domain, _stencil, _coefficients = _operator_fixture((3, 4, 5))
    layout = geometry.layout
    field_halo = jnp.sin(
        0.17 * jnp.arange(np.prod(layout.cell_halo_shape), dtype=jnp.float64)
    ).reshape(layout.cell_halo_shape)
    stencil = build_local_conservative_stencil_from_field(
        field_halo,
        geometry,
        StencilBuilderContext(layout=layout, domain=domain),
    )
    coefficients = LocalCurvatureFaceCoefficients3D(
        layout=layout,
        x=jnp.linspace(-0.8, 1.1, np.prod(layout.face_control_shape(0))).reshape(
            layout.face_control_shape(0)
        ),
        y=jnp.linspace(0.7, -0.4, np.prod(layout.face_control_shape(1))).reshape(
            layout.face_control_shape(1)
        ),
        z=jnp.linspace(-0.2, 0.9, np.prod(layout.face_control_shape(2))).reshape(
            layout.face_control_shape(2)
        ),
    )
    production = local_curvature_conservative_op(
        stencil, geometry, coefficients, domain=domain
    )
    components = local_curvature_conservative_components_op(
        stencil, geometry, coefficients, domain=domain
    )
    assert components.shape == (3,) + geometry.owned_shape
    np.testing.assert_allclose(
        np.sum(np.asarray(components), axis=0),
        np.asarray(production),
        atol=2.0e-12,
        rtol=2.0e-12,
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
def test_radial_fine_glue_sat_is_constant_preserving_and_noninjective():
    cell = jnp.asarray(
        [
            [[2.0], [2.0], [2.0], [2.0]],
            [[1.0], [3.0], [0.5], [4.0]],
            [[-2.0], [5.0], [0.25], [7.0]],
        ],
        dtype=jnp.float64,
    )
    q_face = jnp.linspace(-1.4, 1.6, 4 * 4).reshape((4, 4, 1))
    profile = (4, 2, 2)
    centers = jnp.asarray((0.5, 1.5, 2.5))
    faces = jnp.asarray((0.0, 1.0, 2.0, 3.0))
    area = jnp.ones_like(cell)
    residual = jax.jit(
        lambda value: _radial_fine_glue_sat_integrated_residual(
            q_face,
            value,
            centers,
            faces,
            area,
            profile,
            penalty=0.75,
        )
    )(cell)
    residual = np.asarray(residual)

    left_trace = np.asarray(cell[0])
    right_trace = 1.5 * np.asarray(cell[1]) - 0.5 * np.asarray(cell[2])
    jump = right_trace - left_trace
    expected_power = -np.sum(
        0.5 * 0.75 * np.abs(np.asarray(q_face[1])) * jump * jump
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(cell) * residual),
        expected_power,
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    assert expected_power <= 0.0

    constant = jnp.ones_like(cell) * 3.25
    unchanged = _radial_fine_glue_sat_integrated_residual(
        q_face, constant, centers, faces, area, profile, penalty=1.0
    )
    np.testing.assert_allclose(unchanged, 0.0, atol=0.0, rtol=0.0)

    # Away from the axis-special first transition, both one-sided trace maps
    # exactly reproduce a linear radial field.
    linear_centers = jnp.asarray((0.5, 1.5, 2.5, 3.5))
    linear_faces = jnp.asarray((0.0, 1.0, 2.0, 3.0, 4.0))
    linear = jnp.broadcast_to(linear_centers[:, None, None], (4, 4, 1))
    linear_q = jnp.ones((5, 4, 1), dtype=jnp.float64)
    profile_outer = (4, 4, 2, 2)
    linear_residual = _radial_fine_glue_sat_integrated_residual(
        linear_q,
        linear,
        linear_centers,
        linear_faces,
        jnp.ones_like(linear),
        profile_outer,
        penalty=1.0,
    )
    np.testing.assert_allclose(linear_residual, 0.0, atol=2.0e-15, rtol=0.0)


def test_radial_characteristic_fine_glue_is_exactly_dissipative_and_smooth_null():
    rng = np.random.default_rng(31)
    cell = jnp.asarray(rng.standard_normal((4, 3, 1, 4)), dtype=jnp.float64)
    q_face = jnp.asarray(rng.uniform(-1.5, 1.5, (5, 3, 1)), dtype=jnp.float64)
    tau = 1.25
    # Production |M| is not Euclidean symmetric.  The correct dissipative
    # statement is instead in its frozen characteristic H metric.
    metric = np.asarray(
        background_curvature_characteristic_metric(jnp.asarray(1.4), tau)
    )
    absolute = np.asarray(
        background_curvature_characteristic_absolute_matrix(jnp.asarray(1.4), tau)
    )
    blocks = jnp.broadcast_to(
        jnp.asarray(absolute, dtype=jnp.float64), (3, 3, 1, 4, 4)
    )
    centers = jnp.asarray((0.5, 1.5, 2.5, 3.5), dtype=jnp.float64)
    faces = jnp.asarray((0.0, 1.0, 2.0, 3.0, 4.0), dtype=jnp.float64)
    area = jnp.asarray(rng.uniform(0.7, 1.4, (4, 3, 1)), dtype=jnp.float64)
    profile = (8, 4, 4, 2)
    beta = 0.65

    residual = jax.jit(
        lambda value: _radial_characteristic_fine_glue_integrated_residual(
            q_face,
            value,
            blocks,
            centers,
            faces,
            area,
            profile,
            penalty=beta,
        )
    )(cell)
    legacy_explicit = _radial_characteristic_fine_glue_integrated_residual(
        q_face,
        cell,
        blocks,
        centers,
        faces,
        area,
        profile,
        penalty=beta,
        include_ordinary_faces=False,
    )
    np.testing.assert_allclose(residual, legacy_explicit, atol=3.0e-14, rtol=0.0)
    left_ratio = np.asarray((0.0, 0.5, 0.5))
    right_ratio = np.asarray((0.5, 0.5, 0.0))
    value = np.asarray(cell)
    left_far = np.concatenate((value[:1], value[:-2]), axis=0)
    right_far = np.concatenate((value[2:], value[-1:]), axis=0)
    left = (1.0 + left_ratio[:, None, None, None]) * value[:-1] - left_ratio[
        :, None, None, None
    ] * left_far
    right = (1.0 + right_ratio[:, None, None, None]) * value[1:] - right_ratio[
        :, None, None, None
    ] * right_far
    jump = right - left
    face_area = 0.5 * (np.asarray(area[:-1]) + np.asarray(area[1:]))
    transition = np.asarray((True, False, True))[:, None, None]
    expected_power = -np.sum(
        transition
        * 0.5
        * beta
        * np.abs(np.asarray(q_face[1:-1]))
        * face_area
        * np.einsum("...i,ij,...jk,...k->...", jump, metric, np.asarray(blocks), jump)
    )
    np.testing.assert_allclose(
        np.sum(
            np.einsum("...i,ij,...j->...", value, metric, np.asarray(residual))
        ),
        expected_power,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    assert expected_power < 0.0
    np.testing.assert_allclose(
        np.sum(np.asarray(residual), axis=(0, 1, 2)),
        0.0,
        atol=3.0e-13,
        rtol=0.0,
    )

    constant = jnp.broadcast_to(
        jnp.asarray((1.0, 2.0, 3.0, 4.0)), cell.shape
    )
    constant_residual = _radial_characteristic_fine_glue_integrated_residual(
        q_face,
        constant,
        blocks,
        centers,
        faces,
        area,
        profile,
        penalty=beta,
    )
    np.testing.assert_allclose(constant_residual, 0.0, atol=0.0, rtol=0.0)

    slopes = jnp.asarray((0.3, -0.2, 1.1, 0.7))
    linear = centers[:, None, None, None] * slopes[None, None, None, :]
    linear = jnp.broadcast_to(linear, cell.shape)
    outer_only = _radial_characteristic_fine_glue_integrated_residual(
        q_face,
        linear,
        blocks,
        centers,
        faces,
        area,
        (8, 8, 2, 2),
        penalty=beta,
    )
    np.testing.assert_allclose(outer_only, 0.0, atol=3.0e-15, rtol=0.0)

    inner = _radial_characteristic_fine_glue_integrated_residual(
        q_face,
        cell,
        blocks,
        centers,
        faces,
        area,
        profile,
        penalty=beta,
        transition_face=1,
    )
    outer = _radial_characteristic_fine_glue_integrated_residual(
        q_face,
        cell,
        blocks,
        centers,
        faces,
        area,
        profile,
        penalty=beta,
        transition_face=3,
    )
    np.testing.assert_allclose(residual, inner + outer, atol=3.0e-14, rtol=0.0)

    bulk = jax.jit(
        lambda value: _radial_characteristic_fine_glue_integrated_residual(
            q_face,
            value,
            blocks,
            centers,
            faces,
            area,
            profile,
            penalty=beta,
            include_ordinary_faces=True,
        )
    )(cell)
    # The profile has transition faces 1 and 3, so the difference is the
    # ordinary interior face 2.  It must be nonzero for this generic state.
    assert np.linalg.norm(np.asarray(bulk - residual)) > 1.0e-12
    bulk_expected_power = -np.sum(
        0.5
        * beta
        * np.abs(np.asarray(q_face[1:-1]))
        * face_area
        * np.einsum("...i,ij,...jk,...k->...", jump, metric, np.asarray(blocks), jump)
    )
    np.testing.assert_allclose(
        np.sum(np.einsum("...i,ij,...j->...", value, metric, np.asarray(bulk))),
        bulk_expected_power,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    assert bulk_expected_power < 0.0
    np.testing.assert_allclose(
        np.sum(np.asarray(bulk), axis=(0, 1, 2)),
        0.0,
        atol=3.0e-13,
        rtol=0.0,
    )
    # A one-transition profile can select each face independently.  Since the
    # trace maps do not depend on the profile, their sum is the bulk result.
    all_faces_by_parts = sum(
        (
            _radial_characteristic_fine_glue_integrated_residual(
                q_face,
                cell,
                blocks,
                centers,
                faces,
                area,
                tuple(8 if index < face else 4 for index in range(cell.shape[0])),
                penalty=beta,
            )
            for face in range(1, cell.shape[0])
        ),
        jnp.zeros_like(cell),
    )
    np.testing.assert_allclose(bulk, all_faces_by_parts, atol=3.0e-14, rtol=0.0)
    with pytest.raises(ValueError, match="bulk characteristic"):
        _radial_characteristic_fine_glue_integrated_residual(
            q_face,
            cell,
            blocks,
            centers,
            faces,
            area,
            profile,
            penalty=beta,
            transition_face=1,
            include_ordinary_faces=True,
        )

    quadratic = centers[:, None, None, None] ** 2 * slopes[None, None, None, :]
    quadratic = jnp.broadcast_to(quadratic, cell.shape)
    # The radial end-adjacent faces use the established zero-slope physical
    # closure; suppress only those two coefficients so this check isolates
    # the ordinary-face reconstruction used by the bulk scheme.
    quadratic_q_face = q_face.at[1].set(0.0).at[-2].set(0.0)
    quadratic_bulk = _radial_characteristic_fine_glue_integrated_residual(
        quadratic_q_face,
        quadratic,
        blocks,
        centers,
        faces,
        area,
        profile,
        penalty=beta,
        include_ordinary_faces=True,
    )
    # At ordinary faces on this uniform radial mesh, the two linear
    # extrapolations have a jump proportional to the third difference.  Thus
    # degree-two radial states are untouched by the bulk correction there.
    np.testing.assert_allclose(quadratic_bulk, 0.0, atol=4.0e-15, rtol=0.0)


def test_radial_fine_glue_sat_can_select_one_transition_face():
    cell = jnp.asarray(
        [[[1.0]], [[-2.0]], [[3.0]], [[-4.0]]], dtype=jnp.float64
    )
    q_face = jnp.ones((5, 1, 1), dtype=jnp.float64)
    centers = jnp.asarray((0.5, 1.5, 2.5, 3.5))
    faces = jnp.asarray((0.0, 1.0, 2.0, 3.0, 4.0))
    area = jnp.ones_like(cell)
    profile = (8, 4, 4, 2)

    all_faces = _radial_fine_glue_sat_integrated_residual(
        q_face, cell, centers, faces, area, profile, penalty=1.0
    )
    explicit_all_faces = _radial_fine_glue_sat_integrated_residual(
        q_face,
        cell,
        centers,
        faces,
        area,
        profile,
        penalty=1.0,
        transition_face=None,
    )
    inner = jax.jit(
        lambda value: _radial_fine_glue_sat_integrated_residual(
            q_face,
            value,
            centers,
            faces,
            area,
            profile,
            penalty=1.0,
            transition_face=1,
        )
    )(cell)
    outer = _radial_fine_glue_sat_integrated_residual(
        q_face,
        cell,
        centers,
        faces,
        area,
        profile,
        penalty=1.0,
        transition_face=3,
    )
    np.testing.assert_allclose(all_faces, explicit_all_faces, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(all_faces, inner + outer, atol=2.0e-15, rtol=0.0)
    assert not np.allclose(np.asarray(inner), np.asarray(outer))


@pytest.mark.parametrize("transition_face", (0, 4, 2))
def test_radial_fine_glue_sat_rejects_invalid_or_nontransition_face(
    transition_face,
):
    with pytest.raises(ValueError):
        _radial_fine_glue_sat_integrated_residual(
            jnp.ones((5, 1, 1), dtype=jnp.float64),
            jnp.ones((4, 1, 1), dtype=jnp.float64),
            jnp.asarray((0.5, 1.5, 2.5, 3.5)),
            jnp.asarray((0.0, 1.0, 2.0, 3.0, 4.0)),
            jnp.ones((4, 1, 1), dtype=jnp.float64),
            (8, 4, 4, 2),
            penalty=1.0,
            transition_face=transition_face,
        )
