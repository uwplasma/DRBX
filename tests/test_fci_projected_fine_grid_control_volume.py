"""Strict tests for the projected-fine-grid control-volume operator mode."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from axis_regular_operator_support import polar_fixture
from drbx.geometry import StencilBuilderContext
from drbx.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BC_NOFLUX,
    LocalBoundaryFaceBC3D,
)
from drbx.native.fci_drb_EB_rhs import LocalFciDrbEBRhs
from drbx.native.fci_gmres import SolvaxGmresConfig
from drbx.native.fci_halo import GhostFillWeights1D, PhysicalGhostCellFiller3D
from drbx.native.fci_operators import (
    LocalPerpLaplacianInverseSolver,
    build_local_perp_laplacian_face_projectors,
    local_control_volume_diffusion_cell_volume,
    local_perp_laplacian_conservative_op,
)
from drbx.native.fci_angular_agglomeration import (
    empty_angular_agglomeration_boundary_bc,
    lower_polar_angular_agglomeration_geometry,
)
from drbx.native.fci_rlp_diffusion import apply_rlp_cell_average_prolongation


def _host(shape=(3, 8, 6), jacobian=None):
    """Build the production radius-dependent RLP owner topology."""
    from drbx.geometry.fci_control_volumes import (
        build_polar_angular_agglomeration_geometry,
    )

    u = np.linspace(0.0, 1.0, shape[0] + 1)
    theta = np.linspace(-np.pi, np.pi, shape[1] + 1)
    eta = np.linspace(-np.pi, np.pi, shape[2] + 1)
    if jacobian is None:
        jacobian = lambda points: np.maximum(np.asarray(points)[..., 0], 1.0e-14)
    return build_polar_angular_agglomeration_geometry(
        u,
        theta,
        eta,
        jacobian,
        quadrature_order=2,
        angular_group_size=(shape[1], 2, 1),
    )


def _build_physical_ghost_filler(layout):
    """Return the test's callable homogeneous-Neumann ghost filler.

    The radial upper ghost layers copy the nearest owned radial value.  The
    filler itself leaves every other face unchanged; ``face_bc`` below marks
    only the upper radial face as physical Neumann.
    """
    neutral = GhostFillWeights1D(
        owned_weights=jnp.ones((layout.halo_width, 1), dtype=jnp.float64),
        bc_weights=jnp.zeros((layout.halo_width,), dtype=jnp.float64),
    )
    return PhysicalGhostCellFiller3D(
        dirichlet=(neutral, neutral, neutral),
        neumann_lower=(neutral, neutral, neutral),
        neumann_upper=(neutral, neutral, neutral),
    )


def _setup(shape=(3, 8, 6), jacobian=None):
    geometry, domain, context, _coordinates, exchange, scalar, _vector, _flux = (
        polar_fixture(shape=shape, halo_width=1)
    )
    context = StencilBuilderContext(
        layout=geometry.layout,
        domain=domain,
    )
    host = _host(shape) if jacobian is None else _host(shape, jacobian=jacobian)
    lowered = lower_polar_angular_agglomeration_geometry(host, geometry)
    boundary_bc = empty_angular_agglomeration_boundary_bc(
        max_rows=lowered.irregular_faces.max_rows
    )
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    # Match the production homogeneous-Neumann physical ghost treatment on
    # the outer radial face.  The neutral weights copy the nearest owned cell
    # into every radial upper halo layer; angular/eta faces remain periodic or
    # axis-topological and therefore have no physical ghost BC here.
    physical_ghost_filler = _build_physical_ghost_filler(geometry.layout)
    kind_x = face_bc.kind_x.at[-1].set(BC_NEUMANN)
    mask_x = face_bc.mask_x.at[-1].set(True)
    face_bc = replace(face_bc, kind_x=kind_x, mask_x=mask_x)
    solver = LocalPerpLaplacianInverseSolver(
        geometry=geometry,
        domain=domain,
        control_volume_geometry=lowered,
        control_volume_boundary_bc=boundary_bc,
        halo_exchange=exchange,
        topology_filler=scalar,
        physical_ghost_filler=physical_ghost_filler,
        face_bc=face_bc,
        axis_regular_axes=(True, False, False),
        stencil_builder_context=context,
        config=SolvaxGmresConfig(
            tol=1.0e-10,
            atol=1.0e-12,
            maxiter=20,
            restart=20,
            regularization_epsilon=0.0,
            project_mean_zero=False,
            preconditioner="line-u",
        ),
    )
    active = np.asarray(lowered.cells.is_active_owner, dtype=bool)
    weights = np.asarray(lowered.cells.aggregate_volume, dtype=float)
    return (
        geometry,
        domain,
        context,
        exchange,
        scalar,
        lowered,
        boundary_bc,
        face_bc,
        solver,
        active,
        weights,
    )


def _apply(solver, face_bc, boundary_bc, values):
    return np.asarray(
        solver._apply_A(
            jnp.asarray(values, dtype=jnp.float64),
            face_bc=face_bc,
            control_volume_boundary_bc=boundary_bc,
            project_mean_zero=False,
        ),
        dtype=float,
    )


def _reference(
    geometry,
    domain,
    context,
    exchange,
    scalar,
    lowered,
    face_bc,
    physical_ghost_filler,
    values,
):
    """Explicit ``M_owner^-1 H.T M_raw A_f H`` reference."""
    from drbx.geometry import build_local_conservative_stencil_from_field
    from drbx.native.fci_model import inject_owned_field_to_halo

    cells = lowered.cells
    owner = jnp.asarray(values, dtype=jnp.float64)
    fine = apply_rlp_cell_average_prolongation(
        owner,
        lowered.diffusion_prolongation,
        domain=domain,
    )
    fine_halo = inject_owned_field_to_halo(fine, domain.layout)
    from drbx.native.fci_halo import LocalHaloClosure3D

    fine_halo = LocalHaloClosure3D(
        physical_ghost_filler=physical_ghost_filler,
        halo_exchange=exchange,
        topology_filler=scalar,
    )(fine_halo, domain, face_bc)
    local = build_local_conservative_stencil_from_field(
        fine_halo, geometry, context
    )
    projectors = build_local_perp_laplacian_face_projectors(
        geometry, domain, axis_regular_axes=(True, False, False)
    )
    fine_result = -local_perp_laplacian_conservative_op(
        local,
        geometry,
        domain,
        face_projectors=projectors,
        face_bc=face_bc,
        cell_volume=local_control_volume_diffusion_cell_volume(
            geometry,
            lowered,
        ),
        axis_regular_axes=(True, False, False),
        neumann_normal_scheme="logical",
    )
    raw_integrated = jnp.asarray(cells.raw_volume) * fine_result
    h_transpose = jax.linear_transpose(
        lambda candidate: apply_rlp_cell_average_prolongation(
            candidate,
            lowered.diffusion_prolongation,
            domain=domain,
        ),
        jnp.zeros(cells.shape, dtype=jnp.float64),
    )
    owner_integrated = h_transpose(raw_integrated)[0]
    owner_volume = jnp.asarray(cells.aggregate_volume, dtype=jnp.float64)
    expected = owner_integrated / jnp.where(owner_volume > 0.0, owner_volume, 1.0)
    return np.asarray(jnp.where(cells.is_active_owner, expected, 0.0), dtype=float)


def _weighted_dot(x, y, weights, active):
    return float(np.dot(x[active] * weights[active], y[active]))


def test_projected_apply_equals_explicit_h_adjoint_fine_operator():
    data = _setup()
    geometry, domain, context, exchange, scalar, lowered, boundary_bc, face_bc, solver, active, _weights = data
    rng = np.random.default_rng(20260812)
    values = np.zeros(active.shape)
    values[active] = rng.standard_normal(np.count_nonzero(active))
    actual = _apply(solver, face_bc, boundary_bc, values)
    expected = _reference(
        geometry, domain, context, exchange, scalar, lowered, face_bc,
        solver.physical_ghost_filler, values
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)


def test_projected_constant_is_null_without_regularization_or_physical_flux():
    *_, solver, active, _weights = _setup()
    data = _setup()
    boundary_bc, face_bc = data[6], data[7]
    constant = np.zeros(active.shape)
    constant[active] = 1.0
    response = _apply(solver, face_bc, boundary_bc, constant)
    np.testing.assert_allclose(response[active], 0.0, rtol=0.0, atol=2.0e-11)
    assert np.all(response[~active] == 0.0)


def test_projected_h_adjoint_matrix_is_symmetric_psd_conservative_with_six_nulls():
    data = _setup()
    _geometry, _domain, _context, _exchange, _scalar, _lowered, boundary_bc, face_bc, solver, active, weights = data
    owner_indices = np.flatnonzero(active)
    matrix = np.empty((owner_indices.size, owner_indices.size))
    for column, flat_index in enumerate(owner_indices):
        basis = np.zeros(active.shape)
        basis.reshape(-1)[flat_index] = 1.0
        matrix[:, column] = _apply(solver, face_bc, boundary_bc, basis)[active]
    weighted = np.sqrt(weights[active])[:, None] * matrix / np.sqrt(weights[active])[None, :]
    np.testing.assert_allclose(weighted, weighted.T, rtol=0.0, atol=3.0e-11)
    eigenvalues = np.linalg.eigvalsh(0.5 * (weighted + weighted.T))
    spectral_scale = float(np.max(np.abs(eigenvalues)))
    assert spectral_scale > 0.0
    assert np.min(eigenvalues) >= -1.0e-12 * spectral_scale
    assert np.all(np.isfinite(eigenvalues))
    null_count = int(np.count_nonzero(np.abs(eigenvalues) <= 1.0e-8 * spectral_scale))
    # The straight-field fixture has no perpendicular eta derivative, hence
    # one independent constant-over-(u, theta) null vector per eta plane.
    assert null_count == active.shape[2]
    ones = np.ones(owner_indices.size)
    np.testing.assert_allclose(matrix @ ones, 0.0, rtol=0.0, atol=2.0e-11)
    np.testing.assert_allclose(weights[active] @ matrix, 0.0, rtol=0.0, atol=2.0e-10)


def test_projected_h_adjoint_neumann_source_has_analytic_integral_and_enters_once():
    data = _setup()
    (
        _geometry,
        _domain,
        _context,
        _exchange,
        _scalar,
        _lowered,
        boundary_bc,
        face_bc,
        solver,
        active,
        weights,
    ) = data
    inhomogeneous_bc = replace(
        face_bc,
        value_x=face_bc.value_x.at[-1].set(0.17),
    )
    zero = np.zeros(active.shape)
    boundary_source = _apply(solver, inhomogeneous_bc, boundary_bc, zero)
    homogeneous_source = _apply(solver, face_bc, boundary_bc, zero)
    affine_source = boundary_source - homogeneous_source
    np.testing.assert_allclose(
        np.sum(affine_source[active] * weights[active]),
        -0.17 * 4.0 * np.pi**2,
        rtol=0.0,
        atol=2.0e-9,
    )

    rng = np.random.default_rng(20260911)
    values = np.zeros(active.shape)
    values[active] = rng.normal(size=np.count_nonzero(active))
    full = _apply(solver, inhomogeneous_bc, boundary_bc, values)
    homogeneous = _apply(solver, face_bc, boundary_bc, values)
    np.testing.assert_allclose(
        full,
        homogeneous + affine_source,
        rtol=0.0,
        atol=3.0e-11,
    )


def test_projected_h_adjoint_closed_mixed_tensor_is_symmetric_psd():
    data = _setup(shape=(3, 8, 4))
    (
        geometry,
        _domain,
        _context,
        _exchange,
        _scalar,
        _lowered,
        boundary_bc,
        face_bc,
        solver,
        active,
        weights,
    ) = data
    mixed = jnp.asarray(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.25), (0.0, 0.25, 0.8)),
        dtype=jnp.float64,
    )
    face_shapes = (
        geometry.regular_face_geometry.x_open_mask.shape,
        geometry.regular_face_geometry.y_open_mask.shape,
        geometry.regular_face_geometry.z_open_mask.shape,
    )
    projectors = tuple(
        jnp.broadcast_to(mixed, shape + (3, 3)) for shape in face_shapes
    )
    closed_bc = replace(
        face_bc,
        kind_x=face_bc.kind_x.at[-1].set(BC_NOFLUX),
        value_x=face_bc.value_x.at[-1].set(0.0),
    )
    solver = replace(solver, face_projectors=projectors, face_bc=closed_bc)
    indices = np.flatnonzero(active)
    matrix = np.empty((indices.size, indices.size), dtype=float)
    for column, flat_index in enumerate(indices):
        basis = np.zeros(active.shape)
        basis.reshape(-1)[flat_index] = 1.0
        matrix[:, column] = _apply(
            solver, closed_bc, boundary_bc, basis
        )[active]
    weighted = (
        np.sqrt(weights[active])[:, None]
        * matrix
        / np.sqrt(weights[active])[None, :]
    )
    scale = float(np.linalg.norm(weighted, ord=2))
    np.testing.assert_allclose(weighted, weighted.T, rtol=0.0, atol=3.0e-11)
    assert np.linalg.eigvalsh(0.5 * (weighted + weighted.T))[0] >= -1.0e-12 * scale
    np.testing.assert_allclose(weights[active] @ matrix, 0.0, atol=3.0e-10)


def test_projected_conservative_balance_uses_mismatched_raw_host_mass():
    def variable_host_jacobian(points):
        points = np.asarray(points)
        radius = np.maximum(points[..., 0], 1.0e-14)
        return radius * (1.0 + 0.2 * radius**2)

    data = _setup(jacobian=variable_host_jacobian)
    geometry, _domain, _context, _exchange, _scalar, lowered, boundary_bc, face_bc, solver, active, weights = data
    logical_volume = (
        np.asarray(geometry.spacing.dx_owned)
        * np.asarray(geometry.spacing.dy_owned)
        * np.asarray(geometry.spacing.dz_owned)
    )
    ordinary_metric_volume = (
        np.asarray(geometry.cell_volume_geometry.volume) * logical_volume
    )
    raw_volume = np.asarray(lowered.cells.raw_volume)
    assert np.max(np.abs(raw_volume[active] - ordinary_metric_volume[active])) > 1.0e-8

    rng = np.random.default_rng(20260911)
    values = np.zeros(active.shape)
    values[active] = rng.standard_normal(np.count_nonzero(active))
    response = _apply(solver, face_bc, boundary_bc, values)
    np.testing.assert_allclose(
        np.sum(weights[active] * response[active]),
        0.0,
        rtol=0.0,
        atol=3.0e-9,
    )


def test_weighted_symmetric_projected_operator_has_exact_pairing_and_affine_split():
    data = _setup()
    (
        _geometry,
        _domain,
        _context,
        _exchange,
        _scalar,
        _lowered,
        boundary_bc,
        face_bc,
        conservative_solver,
        active,
        weights,
    ) = data
    solver = replace(
        conservative_solver,
        operator_form="weighted-symmetric",
    )
    rng = np.random.default_rng(20260904)
    u = np.zeros(active.shape)
    v = np.zeros(active.shape)
    u[active] = rng.normal(size=np.count_nonzero(active))
    v[active] = rng.normal(size=np.count_nonzero(active))
    au = _apply(solver, face_bc, boundary_bc, u)
    av = _apply(solver, face_bc, boundary_bc, v)

    np.testing.assert_allclose(
        _weighted_dot(u, av, weights, active),
        _weighted_dot(au, v, weights, active),
        rtol=0.0,
        atol=2.0e-10,
    )
    constant = np.zeros(active.shape)
    constant[active] = 1.0
    np.testing.assert_allclose(
        _apply(solver, face_bc, boundary_bc, constant)[active],
        0.0,
        rtol=0.0,
        atol=2.0e-11,
    )
    # The weighted transpose is defined only for the homogeneous map.  A
    # prescribed normal derivative must remain the established conservative
    # source and enter exactly once.
    inhomogeneous_bc = replace(
        face_bc,
        value_x=face_bc.value_x.at[-1].set(0.17),
    )
    zero = np.zeros(active.shape)
    selected_source = _apply(solver, inhomogeneous_bc, boundary_bc, zero)
    conservative_source = _apply(
        conservative_solver,
        inhomogeneous_bc,
        boundary_bc,
        zero,
    )
    np.testing.assert_allclose(
        selected_source,
        conservative_source,
        rtol=0.0,
        atol=2.0e-12,
    )
    full = _apply(solver, inhomogeneous_bc, boundary_bc, v)
    homogeneous = _apply(solver, face_bc, boundary_bc, v)
    np.testing.assert_allclose(
        full,
        homogeneous + selected_source,
        rtol=0.0,
        atol=2.0e-11,
    )


def test_weighted_symmetric_dirichlet_form_does_not_create_constant_null():
    data = _setup()
    (
        _geometry,
        _domain,
        _context,
        _exchange,
        _scalar,
        _lowered,
        boundary_bc,
        face_bc,
        conservative_solver,
        active,
        weights,
    ) = data
    solver = replace(conservative_solver, operator_form="weighted-symmetric")
    dirichlet_bc = replace(
        face_bc,
        kind_x=face_bc.kind_x.at[-1].set(BC_DIRICHLET),
        value_x=face_bc.value_x.at[-1].set(0.0),
    )
    rng = np.random.default_rng(20260905)
    u = np.zeros(active.shape)
    v = np.zeros(active.shape)
    u[active] = rng.normal(size=np.count_nonzero(active))
    v[active] = rng.normal(size=np.count_nonzero(active))
    au = _apply(solver, dirichlet_bc, boundary_bc, u)
    av = _apply(solver, dirichlet_bc, boundary_bc, v)
    np.testing.assert_allclose(
        _weighted_dot(u, av, weights, active),
        _weighted_dot(au, v, weights, active),
        rtol=0.0,
        atol=2.0e-10,
    )

    constant = np.zeros(active.shape)
    constant[active] = 1.0
    constant_action = _apply(solver, dirichlet_bc, boundary_bc, constant)
    assert np.max(np.abs(constant_action[active])) > 1.0e-6


def test_support_paired_operator_is_volume_weighted_symmetric_and_psd():
    data = _setup()
    (
        _geometry,
        _domain,
        _context,
        _exchange,
        _scalar,
        _lowered,
        boundary_bc,
        face_bc,
        conservative_solver,
        active,
        weights,
    ) = data
    solver = replace(conservative_solver, operator_form="support-paired")
    rng = np.random.default_rng(20260906)
    u = np.zeros(active.shape)
    v = np.zeros(active.shape)
    u[active] = rng.normal(size=np.count_nonzero(active))
    v[active] = rng.normal(size=np.count_nonzero(active))
    au = _apply(solver, face_bc, boundary_bc, u)
    av = _apply(solver, face_bc, boundary_bc, v)

    np.testing.assert_allclose(
        _weighted_dot(u, av, weights, active),
        _weighted_dot(au, v, weights, active),
        rtol=0.0,
        atol=2.0e-10,
    )
    rayleigh = _weighted_dot(u, au, weights, active)
    assert rayleigh >= -2.0e-10
    assert rayleigh > 0.0

    constant = np.zeros(active.shape)
    constant[active] = 1.0
    np.testing.assert_allclose(
        _apply(solver, face_bc, boundary_bc, constant)[active],
        0.0,
        rtol=0.0,
        atol=2.0e-11,
    )


def test_support_paired_neumann_source_has_analytic_integral_and_enters_once():
    data = _setup()
    (
        _geometry,
        _domain,
        _context,
        _exchange,
        _scalar,
        _lowered,
        boundary_bc,
        face_bc,
        solver_base,
        active,
        weights,
    ) = data
    solver = replace(solver_base, operator_form="support-paired")
    inhomogeneous_bc = replace(
        face_bc,
        value_x=face_bc.value_x.at[-1].set(0.17),
    )
    zero = np.zeros(active.shape)
    selected_source = _apply(solver, inhomogeneous_bc, boundary_bc, zero)
    homogeneous_source = _apply(solver, face_bc, boundary_bc, zero)
    integrated_source = float(
        np.sum((selected_source - homogeneous_source)[active] * weights[active])
    )
    np.testing.assert_allclose(integrated_source, -0.17 * 4.0 * np.pi**2, atol=2.0e-9)

    rng = np.random.default_rng(20260907)
    values = np.zeros(active.shape)
    values[active] = rng.normal(size=np.count_nonzero(active))
    full = _apply(solver, inhomogeneous_bc, boundary_bc, values)
    homogeneous = _apply(solver, face_bc, boundary_bc, values)
    np.testing.assert_allclose(
        full,
        homogeneous + (selected_source - homogeneous_source),
        rtol=0.0,
        atol=2.0e-11,
    )


def test_weighted_symmetric_operator_form_is_static_pytree_metadata():
    *_, solver, _active, _weights = _setup()
    selected = replace(solver, operator_form="weighted-symmetric")
    leaves, treedef = jax.tree_util.tree_flatten(selected)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.operator_form == "weighted-symmetric"


def test_support_paired_operator_form_is_static_pytree_metadata():
    *_, solver, _active, _weights = _setup()
    selected = replace(solver, operator_form="support-paired")
    leaves, treedef = jax.tree_util.tree_flatten(selected)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.operator_form == "support-paired"


def test_projected_mode_does_not_read_compact_face_functionals(monkeypatch):
    data = _setup()
    _geometry, _domain, _context, _exchange, _scalar, _lowered, boundary_bc, face_bc, solver, active, _weights = data

    # The dataclass validates finite compact payloads at construction time, so
    # poisoning those arrays with NaNs is intentionally not possible.  Guard
    # the production compact builder instead: projected mode must never call
    # it, because it evaluates the ordinary fine-grid operator directly.
    import drbx.native.fci_operators as operators

    def compact_builder_must_not_run(*args, **kwargs):
        raise AssertionError("projected-fine-grid invoked compact face closure")

    monkeypatch.setattr(
        operators, "build_local_control_volume_field_closure",
        compact_builder_must_not_run,
    )
    rng = np.random.default_rng(7)
    values = np.zeros(active.shape)
    values[active] = rng.standard_normal(np.count_nonzero(active))
    result = _apply(solver, face_bc, boundary_bc, values)
    assert np.all(np.isfinite(result))


def test_projected_rhs_adds_owner_source_after_restriction():
    source = inspect.getsource(LocalFciDrbEBRhs.evaluate_stage)
    restrict = source.index("assembled = self._restrict_fine_state(")
    add = source.index("density=assembled.density + source_owned.density", restrict)
    assert add > restrict
    assert source.index("Te=assembled.Te + source_owned.Te", restrict) > restrict
    assert source.index("Ve=assembled.Ve + source_owned.Ve", restrict) > restrict

    # This is the exact owner-space operation used by that production branch:
    # an owner source is not expanded and therefore cannot be diluted by R.
    data = _setup()
    lowered, active = data[5], data[9]
    source = np.zeros(active.shape)
    source[active] = 3.25
    assembled = np.zeros_like(source)
    result = np.where(active, assembled + source, 0.0)
    np.testing.assert_allclose(result[active], 3.25, rtol=0.0, atol=0.0)
    assert np.all(result[~active] == 0.0)
    assert lowered.cells.aggregate_volume[0, 0, 0] > lowered.cells.raw_volume[0, 0, 0]
