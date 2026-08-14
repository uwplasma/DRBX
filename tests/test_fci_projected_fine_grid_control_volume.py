"""Strict tests for the projected-fine-grid control-volume operator mode."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from axis_regular_operator_support import polar_fixture
from drbx.geometry import StencilBuilderContext
from drbx.native.fci_boundaries import BC_NEUMANN, LocalBoundaryFaceBC3D
from drbx.native.fci_drb_EB_rhs import LocalFciDrbEBRhs
from drbx.native.fci_gmres import SolvaxGmresConfig
from drbx.native.fci_halo import GhostFillWeights1D, PhysicalGhostCellFiller3D
from drbx.native.fci_operators import (
    LocalPerpLaplacianInverseSolver,
    aggregate_local_control_volume_average,
    build_local_perp_laplacian_face_projectors,
    expand_local_control_volume_owner_field,
    local_perp_laplacian_conservative_op,
)
from drbx.native.fci_angular_agglomeration import (
    empty_angular_agglomeration_boundary_bc,
    lower_polar_angular_agglomeration_geometry,
)


def _host(shape=(3, 8, 6)):
    """Build the production radius-dependent RLP owner topology."""
    from drbx.geometry.fci_control_volumes import (
        build_polar_angular_agglomeration_geometry,
    )

    u = np.linspace(0.0, 1.0, shape[0] + 1)
    theta = np.linspace(-np.pi, np.pi, shape[1] + 1)
    eta = np.linspace(-np.pi, np.pi, shape[2] + 1)
    return build_polar_angular_agglomeration_geometry(
        u,
        theta,
        eta,
        lambda points: np.maximum(np.asarray(points)[..., 0], 1.0e-14),
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


def _setup(shape=(3, 8, 6)):
    geometry, domain, context, _coordinates, exchange, scalar, _vector, _flux = (
        polar_fixture(shape=shape, halo_width=1)
    )
    context = StencilBuilderContext(
        layout=geometry.layout,
        domain=domain,
    )
    lowered = lower_polar_angular_agglomeration_geometry(_host(shape), geometry)
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
    """Explicit R A P reference for the projected mode."""
    from drbx.geometry import build_local_conservative_stencil_from_field
    from drbx.native.fci_model import inject_owned_field_to_halo

    cells = lowered.cells
    owner = jnp.asarray(values, dtype=jnp.float64)
    owner_halo = inject_owned_field_to_halo(owner, domain.layout)
    owner_halo = exchange(owner_halo, domain)
    fine = expand_local_control_volume_owner_field(
        owner, cells, owner_values_halo=owner_halo
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
        axis_regular_axes=(True, False, False),
        neumann_normal_scheme="logical",
    )
    return np.asarray(
        aggregate_local_control_volume_average(fine_result, cells, domain),
        dtype=float,
    )


def _weighted_dot(x, y, weights, active):
    return float(np.dot(x[active] * weights[active], y[active]))


def test_projected_apply_equals_explicit_expand_dense_operator_restrict():
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


def test_projected_matrix_is_volume_weighted_symmetric_and_nonnegative():
    data = _setup()
    _geometry, _domain, _context, _exchange, _scalar, _lowered, boundary_bc, face_bc, solver, active, weights = data
    owner_indices = np.flatnonzero(active)
    matrix = np.empty((owner_indices.size, owner_indices.size))
    for column, flat_index in enumerate(owner_indices):
        basis = np.zeros(active.shape)
        basis.reshape(-1)[flat_index] = 1.0
        matrix[:, column] = _apply(solver, face_bc, boundary_bc, basis)[active]
    weighted = np.sqrt(weights[active])[:, None] * matrix / np.sqrt(weights[active])[None, :]
    np.testing.assert_allclose(weighted, weighted.T, rtol=2.0e-10, atol=2.0e-10)
    eigenvalues = np.linalg.eigvalsh(0.5 * (weighted + weighted.T))
    assert eigenvalues.min() >= -2.0e-10
    assert eigenvalues.max() > 0.0


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
    restrict = source.index("assembled = self._restrict_fine_state")
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
