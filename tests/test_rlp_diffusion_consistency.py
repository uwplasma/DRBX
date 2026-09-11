"""Acceptance controls for the live angular-RLP diffusion operator.

These tests deliberately use the production owner-space action.  The smooth
polar field and its continuum Laplacian are evaluated independently, so a
``R A_f P`` implementation cannot make the reference pass by reproducing its
own stencil.  The convergence controls guard the owner-face
diffusion/polarization consistency fix against regression.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from axis_regular_operator_support import polar_fixture  # noqa: E402
from drbx.geometry import StencilBuilderContext  # noqa: E402
from drbx.geometry.fci_control_volumes import (  # noqa: E402
    build_polar_angular_agglomeration_geometry,
    build_radius_dependent_angular_group_profile,
)
from drbx.native.fci_angular_agglomeration import (  # noqa: E402
    empty_angular_agglomeration_boundary_bc,
    lower_polar_angular_agglomeration_geometry,
)
from drbx.native.fci_boundaries import (  # noqa: E402
    BC_NEUMANN,
    LocalBoundaryFaceBC3D,
)
from drbx.native.fci_drb_EB_rhs import LocalFciDrbEBRhs  # noqa: E402
from drbx.native.fci_gmres import SolvaxGmresConfig  # noqa: E402
from drbx.native.fci_halo import (  # noqa: E402
    GhostFillWeights1D,
    LocalHaloClosure3D,
    PhysicalGhostCellFiller3D,
)
from drbx.native.fci_model import inject_owned_field_to_halo  # noqa: E402
from drbx.native.fci_operators import (  # noqa: E402
    LocalPerpLaplacianInverseSolver,
    aggregate_local_control_volume_average,
    build_local_perp_laplacian_face_projectors,
    expand_local_control_volume_owner_field,
)


@dataclass
class _PolarOwnerCase:
    geometry: object
    domain: object
    context: StencilBuilderContext
    coordinates: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    exchange: object
    scalar_filler: object
    control_volume: object
    control_volume_bc: object
    face_bc: LocalBoundaryFaceBC3D
    solver: LocalPerpLaplacianInverseSolver
    active: np.ndarray
    weights: np.ndarray
    profile: np.ndarray
    action: object
    dense_matrix: tuple[np.ndarray, np.ndarray] | None = None


def _physical_ghost_filler(layout):
    neutral = GhostFillWeights1D(
        owned_weights=jnp.ones((layout.halo_width, 1), dtype=jnp.float64),
        bc_weights=jnp.zeros((layout.halo_width,), dtype=jnp.float64),
    )
    return PhysicalGhostCellFiller3D(
        dirichlet=(neutral, neutral, neutral),
        neumann_lower=(neutral, neutral, neutral),
        neumann_upper=(neutral, neutral, neutral),
    )


def _build_case(n: int, mode: str) -> _PolarOwnerCase:
    """Build one actual owner-space production solver on the polar fixture."""

    if mode not in {"automatic", "axis_only"}:
        raise ValueError(mode)
    # A straight-field polar fixture is block diagonal in eta.  One eta plane
    # gives the intended 2-D continuum control and makes the nullspace test
    # unambiguous (the physical constant is the only null vector).
    eta_count = 1
    fixture_data = polar_fixture(shape=(n, n, eta_count), halo_width=1)
    geometry, domain, _unused_context, coordinates, exchange, scalar, _vector, _flux = (
        fixture_data
    )
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    u_faces = np.linspace(0.0, 1.0, n + 1)
    theta_faces = np.linspace(0.0, 2.0 * np.pi, n + 1)
    eta_faces = np.linspace(0.0, 2.0 * np.pi, eta_count + 1)
    explicit_profile = (
        np.r_[n, np.ones(n - 1, dtype=np.int32)] if mode == "axis_only" else None
    )
    profile = build_radius_dependent_angular_group_profile(
        u_faces, theta_faces, explicit_profile=explicit_profile
    )
    host = build_polar_angular_agglomeration_geometry(
        u_faces,
        theta_faces,
        eta_faces,
        lambda points: np.maximum(np.asarray(points)[..., 0], 1.0e-14),
        quadrature_order=2,
        angular_group_size=profile,
    )
    lowered = lower_polar_angular_agglomeration_geometry(host, geometry)
    control_volume_bc = empty_angular_agglomeration_boundary_bc(
        max_rows=lowered.irregular_faces.max_rows
    )

    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    # Only the outer radial face is physical.  The neutral ghost filler gives
    # it homogeneous Neumann data; the lower radial face remains axis-regular.
    face_bc = replace(
        face_bc,
        kind_x=face_bc.kind_x.at[-1].set(BC_NEUMANN),
        mask_x=face_bc.mask_x.at[-1].set(True),
    )
    solver = LocalPerpLaplacianInverseSolver(
        geometry=geometry,
        domain=domain,
        control_volume_geometry=lowered,
        control_volume_boundary_bc=control_volume_bc,
        halo_exchange=exchange,
        topology_filler=scalar,
        physical_ghost_filler=_physical_ghost_filler(geometry.layout),
        face_bc=face_bc,
        axis_regular_axes=(True, False, False),
        stencil_builder_context=context,
        config=SolvaxGmresConfig(
            tol=1.0e-11,
            atol=1.0e-13,
            maxiter=40,
            restart=20,
            regularization_epsilon=0.0,
            project_mean_zero=False,
            preconditioner="line-u",
        ),
    )
    # Compile the live action once per cached case.  Dense independent
    # controls apply this action to many one-hot owner fields; constructing a
    # fresh jitted closure in that loop would trigger one compilation per
    # column and make the N=16/N=32 controls impractical.
    action = jax.jit(
        lambda field: solver._apply_A(
            field,
            face_bc=face_bc,
            control_volume_boundary_bc=control_volume_bc,
            project_mean_zero=False,
        )
    )
    active = np.asarray(lowered.cells.is_active_owner, dtype=bool)
    weights = np.asarray(lowered.cells.aggregate_volume, dtype=float)
    return _PolarOwnerCase(
        geometry=geometry,
        domain=domain,
        context=context,
        coordinates=coordinates,
        exchange=exchange,
        scalar_filler=scalar,
        control_volume=lowered,
        control_volume_bc=control_volume_bc,
        face_bc=face_bc,
        solver=solver,
        active=active,
        weights=weights,
        profile=np.asarray(profile, dtype=np.int32),
        action=action,
    )


@pytest.fixture(scope="module")
def polar_owner():
    """Factory for automatic and axis-only production polar owner cases."""

    cases: dict[tuple[int, str], _PolarOwnerCase] = {}

    def make(n: int, mode: str) -> _PolarOwnerCase:
        key = (int(n), mode)
        if key not in cases:
            cases[key] = _build_case(*key)
        return cases[key]

    return make


def _apply(case: _PolarOwnerCase, values: jnp.ndarray) -> np.ndarray:
    return np.asarray(case.action(values), dtype=float)


def _weighted_rms(values: np.ndarray, case: _PolarOwnerCase) -> float:
    mask = case.active
    weights = case.weights
    return float(
        np.sqrt(np.sum(weights[mask] * values[mask] ** 2) / np.sum(weights[mask]))
    )


def _manufactured_owner_data(case: _PolarOwnerCase):
    """Return genuine raw-cell averages of a smooth polar MMS and source."""

    owned = case.geometry.layout.owned_slices_cell
    r, _theta, _z = (np.asarray(value)[owned] for value in case.coordinates)
    n = int(r.shape[0])
    nodes, node_weights = np.polynomial.legendre.leggauss(6)
    u0 = np.arange(n, dtype=float) / n
    u1 = (np.arange(n, dtype=float) + 1.0) / n
    t0 = np.arange(n, dtype=float) * 2.0 * np.pi / n
    t1 = (np.arange(n, dtype=float) + 1.0) * 2.0 * np.pi / n
    ur = 0.5 * (u1 - u0)[:, None] * nodes[None, :] + 0.5 * (u1 + u0)[:, None]
    tr = 0.5 * (t1 - t0)[:, None] * nodes[None, :] + 0.5 * (t1 + t0)[:, None]
    quadrature = node_weights[:, None] * node_weights[None, :]
    field_fine = np.empty((n, n, 1), dtype=float)
    laplacian_fine = np.empty_like(field_fine)
    for i in range(n):
        rr = ur[i][:, None]
        for j in range(n):
            tt = tr[j][None, :]
            volume = np.sum(quadrature * rr)
            field_integrand = rr**2 * (1.0 - rr**2) ** 4 * np.cos(2.0 * tt) * rr
            lap_integrand = sum(
                (-1) ** k
                * math.comb(4, k)
                * ((2 + 2 * k) ** 2 - 4)
                * rr ** (2 * k)
                * np.cos(2.0 * tt)
                * rr
                for k in range(5)
            )
            field_fine[i, j, 0] = np.sum(quadrature * field_integrand) / volume
            laplacian_fine[i, j, 0] = np.sum(quadrature * lap_integrand) / volume
    field = aggregate_local_control_volume_average(
        jnp.asarray(field_fine, dtype=jnp.float64),
        case.control_volume.cells,
        case.domain,
    )
    positive_source = aggregate_local_control_volume_average(
        jnp.asarray(-laplacian_fine, dtype=jnp.float64),
        case.control_volume.cells,
        case.domain,
    )
    return np.asarray(field), np.asarray(positive_source)


def _harmonic_neumann_owner_data(case: _PolarOwnerCase) -> np.ndarray:
    """Return owner averages of ``r**2 cos(2 theta)``."""

    n = int(case.active.shape[0])
    nodes, node_weights = np.polynomial.legendre.leggauss(6)
    u0 = np.arange(n, dtype=float) / n
    u1 = (np.arange(n, dtype=float) + 1.0) / n
    t0 = np.arange(n, dtype=float) * 2.0 * np.pi / n
    t1 = (np.arange(n, dtype=float) + 1.0) * 2.0 * np.pi / n
    ur = 0.5 * (u1 - u0)[:, None] * nodes[None, :] + 0.5 * (u1 + u0)[:, None]
    tr = 0.5 * (t1 - t0)[:, None] * nodes[None, :] + 0.5 * (t1 + t0)[:, None]
    quadrature = node_weights[:, None] * node_weights[None, :]
    fine = np.empty((n, n, 1), dtype=float)
    for i in range(n):
        rr = ur[i][:, None]
        for j in range(n):
            tt = tr[j][None, :]
            volume = np.sum(quadrature * rr)
            fine[i, j, 0] = np.sum(
                quadrature * rr**2 * np.cos(2.0 * tt) * rr
            ) / volume
    return np.asarray(
        aggregate_local_control_volume_average(
            jnp.asarray(fine),
            case.control_volume.cells,
            case.domain,
        )
    )


def _dense_live_matrix(case: _PolarOwnerCase) -> tuple[np.ndarray, np.ndarray]:
    if case.dense_matrix is not None:
        return case.dense_matrix
    indices = np.flatnonzero(case.active)
    matrix = np.empty((indices.size, indices.size), dtype=float)
    for column, flat_index in enumerate(indices):
        basis = np.zeros(case.active.shape, dtype=float)
        basis.reshape(-1)[flat_index] = 1.0
        matrix[:, column] = _apply(case, jnp.asarray(basis))[case.active]
    case.dense_matrix = (matrix, indices)
    return case.dense_matrix


@pytest.mark.parametrize("mode", ("automatic", "axis_only"))
def test_continuum_action_converges_for_automatic_and_axis_only_rlp(polar_owner, mode):
    errors = []
    for n in (8, 16, 32):
        case = polar_owner(n, mode)
        field, exact = _manufactured_owner_data(case)
        errors.append(_weighted_rms(_apply(case, jnp.asarray(field)) - exact, case))
    # The q-dependent angular amplification in the old R A_f P action makes
    # the automatic sequence grow.  Both sequences must improve after the
    # owner-face consistency fix.  A modest factor avoids imposing a formal
    # order on the piecewise owner representation.
    assert errors[1] < 0.9 * errors[0], errors
    assert errors[2] < 0.9 * errors[1], errors


@pytest.mark.parametrize("mode", ("automatic", "axis_only"))
def test_independent_elliptic_solution_control_converges(polar_owner, mode):
    errors = []
    # N=8,16,32 are the supported smooth-control sequence.  The tiny N=4
    # fixture has a symmetry-degenerate manufactured mode and is not a
    # meaningful continuum solve control.
    for n in (8, 16, 32):
        case = polar_owner(n, mode)
        matrix, indices = _dense_live_matrix(case)
        _field, exact = _manufactured_owner_data(case)
        weights = case.weights[case.active]
        augmented = np.zeros((indices.size + 1, indices.size + 1), dtype=float)
        augmented[:-1, :-1] = matrix
        augmented[:-1, -1] = 1.0
        augmented[-1, :-1] = weights
        solution = np.linalg.solve(
            augmented,
            np.r_[exact[case.active], 0.0],
        )[:-1]
        expected = _field[case.active]
        expected -= np.dot(weights, expected) / np.sum(weights)
        errors.append(
            float(
                np.sqrt(
                    np.sum(weights * (solution - expected) ** 2) / np.sum(weights)
                )
            )
        )
    observed_orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(observed_orders >= 1.8), (errors, observed_orders)


def test_harmonic_nonzero_neumann_solution_converges_second_order(polar_owner):
    errors = []
    for n in (8, 16, 32):
        case = polar_owner(n, "automatic")
        matrix, indices = _dense_live_matrix(case)
        theta = (np.arange(n, dtype=float) + 0.5) * 2.0 * np.pi / n
        boundary_value = 2.0 * np.cos(2.0 * theta)[:, None]
        inhomogeneous = replace(
            case.face_bc,
            value_x=case.face_bc.value_x.at[-1].set(boundary_value),
        )
        zero = jnp.zeros(case.active.shape, dtype=jnp.float64)
        affine_source = np.asarray(
            case.solver.apply_positive_operator(
                zero,
                face_bc=inhomogeneous,
                control_volume_boundary_bc=case.control_volume_bc,
                project_mean_zero=False,
            )
        )[case.active]
        weights = case.weights[case.active]
        augmented = np.zeros((indices.size + 1, indices.size + 1), dtype=float)
        augmented[:-1, :-1] = matrix
        augmented[:-1, -1] = 1.0
        augmented[-1, :-1] = weights
        solution = np.linalg.solve(augmented, np.r_[-affine_source, 0.0])[:-1]
        expected = _harmonic_neumann_owner_data(case)[case.active]
        expected -= np.dot(weights, expected) / np.sum(weights)
        errors.append(
            float(
                np.sqrt(
                    np.sum(weights * (solution - expected) ** 2)
                    / np.sum(weights)
                )
            )
        )
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(orders >= 1.8), (errors, orders)


@pytest.mark.parametrize("mode", ("automatic", "axis_only"))
def test_live_owner_gmres_roundtrip_with_line_u_preconditioner(polar_owner, mode):
    case = polar_owner(8, mode)
    field, _exact = _manufactured_owner_data(case)
    field = np.where(case.active, field, 0.0)
    field -= np.where(
        case.active,
        np.sum(case.weights * field) / np.sum(case.weights[case.active]),
        0.0,
    )
    rhs = _apply(case, jnp.asarray(field))
    solution, info = case.solver.solve_rlp_owner(
        jnp.asarray(rhs),
        guess_owned=jnp.zeros_like(jnp.asarray(field)),
        return_diagnostics=True,
    )
    solution = np.asarray(solution)
    assert bool(info.converged)
    residual = _apply(case, jnp.asarray(solution)) - rhs
    assert _weighted_rms(residual, case) <= 2.0e-8
    error = solution - field
    error -= np.where(
        case.active,
        np.sum(case.weights * error) / np.sum(case.weights[case.active]),
        0.0,
    )
    assert _weighted_rms(error, case) <= 2.0e-7


@pytest.mark.parametrize("mode", ("automatic", "axis_only"))
def test_constant_is_exact_null_and_conservative_balance_holds(polar_owner, mode):
    case = polar_owner(8, mode)
    constant = np.zeros(case.active.shape, dtype=float)
    constant[case.active] = 1.0
    response = _apply(case, jnp.asarray(constant))
    np.testing.assert_allclose(response[case.active], 0.0, rtol=0.0, atol=3.0e-11)
    np.testing.assert_allclose(
        np.sum(case.weights[case.active] * response[case.active]),
        0.0,
        rtol=0.0,
        atol=3.0e-10,
    )
    rng = np.random.default_rng(20260911)
    random_field = np.zeros(case.active.shape, dtype=float)
    random_field[case.active] = rng.normal(size=np.count_nonzero(case.active))
    random_response = _apply(case, jnp.asarray(random_field))
    np.testing.assert_allclose(
        np.sum(case.weights[case.active] * random_response[case.active]),
        0.0,
        rtol=0.0,
        atol=3.0e-9,
    )


@pytest.mark.parametrize("mode", ("automatic", "axis_only"))
def test_live_operator_has_one_constant_null_and_positive_energy(polar_owner, mode):
    case = polar_owner(8, mode)
    matrix, _indices = _dense_live_matrix(case)
    weights = case.weights[case.active]
    weighted = np.sqrt(weights)[:, None] * matrix / np.sqrt(weights)[None, :]
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    scale = singular_values[0]
    assert singular_values[-1] <= 5.0e-10 * scale, singular_values
    assert np.count_nonzero(singular_values > 1.0e-8 * scale) == singular_values.size - 1
    symmetric_eigenvalues = np.linalg.eigvalsh(0.5 * (weighted + weighted.T))
    assert symmetric_eigenvalues[0] >= -2.0e-9
    rng = np.random.default_rng(20260911)
    vector = rng.normal(size=weights.size)
    energy = float(np.dot(vector * weights, matrix @ vector))
    assert energy >= -2.0e-9
    assert energy > 0.0


@pytest.mark.parametrize("operator_form", ("conservative", "weighted-symmetric", "support-paired"))
@pytest.mark.parametrize("mode", ("automatic", "axis_only"))
def test_evolved_diffusion_agrees_with_live_polarization_action(
    polar_owner,
    mode,
    operator_form,
):
    case = polar_owner(8, mode)
    rng = np.random.default_rng(17)
    owner = np.zeros(case.active.shape, dtype=float)
    owner[case.active] = rng.normal(size=np.count_nonzero(case.active))
    expanded = expand_local_control_volume_owner_field(
        jnp.asarray(owner), case.control_volume.cells
    )
    halo = LocalHaloClosure3D(
        physical_ghost_filler=case.solver.physical_ghost_filler,
        halo_exchange=case.exchange,
        topology_filler=case.scalar_filler,
    )(
        inject_owned_field_to_halo(expanded, case.domain.layout),
        case.domain,
        case.face_bc,
    )
    rhs_context = type(
        "_RhsDiffusionContext",
        (),
        {
            "geometry": case.geometry,
            "domain": case.domain,
            "_stencil_builder_context": lambda self: case.context,
            "stencil_builder_context": case.context,
            "control_volume_geometry": case.control_volume,
            "control_volume_boundary_bc": case.control_volume_bc,
            "halo_exchange": case.exchange,
            "topology_filler": case.scalar_filler,
            "physical_ghost_filler": case.solver.physical_ghost_filler,
            "face_projectors": build_local_perp_laplacian_face_projectors(
                case.geometry, case.domain, axis_regular_axes=(True, False, False)
            ),
            "axis_regular_axes": (True, False, False),
            "neumann_normal_scheme": "logical",
            "gmres_config": case.solver.config,
            "polarization_operator_form": operator_form,
            "polarization_coarse_data": None,
            "_polarization_solver": LocalFciDrbEBRhs._polarization_solver,
            "_positive_polarization_action": (
                LocalFciDrbEBRhs._positive_polarization_action
            ),
        },
    )()
    diffusion = -aggregate_local_control_volume_average(
        LocalFciDrbEBRhs._field_perp_diffusion(
            rhs_context, halo, case.face_bc, 1.0
        ),
        case.control_volume.cells,
        case.domain,
    )
    selected_solver = replace(case.solver, operator_form=operator_form)
    polarization = np.asarray(
        selected_solver.apply_positive_operator(
            jnp.asarray(owner),
            face_bc=case.face_bc,
            control_volume_boundary_bc=case.control_volume_bc,
            project_mean_zero=False,
        )
    )
    np.testing.assert_allclose(
        np.asarray(diffusion)[case.active],
        polarization[case.active],
        rtol=0.0,
        atol=3.0e-10,
    )
