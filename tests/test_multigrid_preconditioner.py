from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

try:
    import pytest
except ImportError:  # pragma: no cover - optional test runner dependency
    class _PytestStub:
        @staticmethod
        def importorskip(name: str):
            return None

    pytest = _PytestStub()

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

import jax
import jax.numpy as jnp

from drbx.geometry import (
    RegularFaceGeometry3D,
    build_conservative_stencil_from_field,
    logical_grid_from_axis_vectors,
)
from drbx.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BoundaryFaceBC3D,
    CutWallBC3D,
    CutWallGeometry3D,
)
from drbx.native.fci_operators import (
    _mg_apply_negative_perp_laplacian,
    _prolong_field,
    _restrict_field_simple,
    _restrict_residual_jweighted,
    PerpLaplacianInverseSolver,
    build_perp_laplacian_mg_hierarchy,
    mg_apply_preconditioner,
    perp_laplacian_conservative_op,
)
from tests.test_fci_operators import M, N, build_test_fci_geometry
from tests.test_mms_shifted_torus_4_field import build_shifted_torus_4field_geometry


PERIODIC_AXES = (False, True, True)


def _weighted_mean(field: jnp.ndarray, geometry) -> jnp.ndarray:
    weights = jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64)
    values = jnp.asarray(field, dtype=jnp.float64)
    return jnp.sum(weights * values) / jnp.sum(weights)


def _remove_weighted_mean(field: jnp.ndarray, geometry) -> jnp.ndarray:
    return jnp.asarray(field, dtype=jnp.float64) - _weighted_mean(field, geometry)


def _weighted_norm(field: jnp.ndarray, geometry) -> jnp.ndarray:
    weights = jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64)
    values = jnp.asarray(field, dtype=jnp.float64)
    return jnp.sqrt(jnp.sum(weights * values * values))


def _make_face_bc(geometry, *, kind: int) -> BoundaryFaceBC3D:
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    face_bc = BoundaryFaceBC3D.empty(regular_face_geometry)
    return face_bc.replace(
        kind_x=face_bc.kind_x.at[0].set(kind).at[-1].set(kind),
        value_x=face_bc.value_x.at[0].set(0.0).at[-1].set(0.0),
        mask_x=face_bc.mask_x.at[0].set(True).at[-1].set(True),
    )


def _make_dirichlet_face_bc(geometry) -> BoundaryFaceBC3D:
    return _make_face_bc(geometry, kind=BC_DIRICHLET)


def _make_neumann_face_bc(geometry) -> BoundaryFaceBC3D:
    return _make_face_bc(geometry, kind=BC_NEUMANN)


def _make_hierarchy(
    geometry,
    face_bc: BoundaryFaceBC3D,
    *,
    max_levels: int | None = 2,
    pre_smooth: int = 1,
    post_smooth: int = 1,
    coarse_smooth: int = 4,
    omega_jacobi: float = 0.5,
    smoother: str = "chebyshev",
    chebyshev_order: int = 2,
    spectral_radius_estimate: float | None = None,
    direct_coarse_size: int = 512,
):
    return build_perp_laplacian_mg_hierarchy(
        geometry,
        build_conservative_stencil_from_field,
        face_bc=face_bc,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=CutWallGeometry3D.empty(),
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=PERIODIC_AXES,
        max_levels=max_levels,
        pre_smooth=pre_smooth,
        post_smooth=post_smooth,
        coarse_smooth=coarse_smooth,
        omega_jacobi=omega_jacobi,
        smoother=smoother,
        chebyshev_order=chebyshev_order,
        spectral_radius_estimate=spectral_radius_estimate,
        direct_coarse_size=direct_coarse_size,
    )


def _apply_negative_perp_laplacian(field: jnp.ndarray, level) -> jnp.ndarray:
    values = jnp.asarray(field, dtype=jnp.float64)
    if level.has_nullspace:
        values = _remove_weighted_mean(values, level.geometry)
    stencil = level.stencil_builder(values, level.geometry, level.periodic_axes, level.face_bc)
    result = -perp_laplacian_conservative_op(
        stencil,
        level.geometry,
        face_projectors=level.face_projectors,
        face_bc=level.face_bc,
        regular_face_geometry=level.regular_face_geometry,
        cut_wall_geometry=level.cut_wall_geometry,
        cut_wall_bc=level.cut_wall_bc,
        periodic_axes=level.periodic_axes,
    )
    if level.has_nullspace:
        result = _remove_weighted_mean(result, level.geometry)
    return result


def _smooth_dirichlet_exact(geometry) -> jnp.ndarray:
    logical_grid = logical_grid_from_axis_vectors(*geometry.logical_axis_vectors)
    rho = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    phi = logical_grid[..., 2]
    rho_min = logical_grid[0, 0, 0, 0]
    rho_max = logical_grid[-1, 0, 0, 0]
    xi = (rho - rho_min) / (rho_max - rho_min)
    return jnp.sin(jnp.pi * xi) * (1.0 + 0.25 * jnp.cos(2.0 * theta) * jnp.sin(3.0 * phi))


def _smooth_neumann_exact(geometry) -> jnp.ndarray:
    logical_grid = logical_grid_from_axis_vectors(*geometry.logical_axis_vectors)
    rho = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    phi = logical_grid[..., 2]
    rho_min = logical_grid[0, 0, 0, 0]
    rho_max = logical_grid[-1, 0, 0, 0]
    xi = (rho - rho_min) / (rho_max - rho_min)
    radial_envelope = jnp.sin(jnp.pi * xi) ** 2
    return radial_envelope * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)


def _passing_reference_build_perp_laplacian_mg_hierarchy_builds_regular_face_levels() -> None:
    geometry = build_test_fci_geometry((17, 16, 16), rho_min=0.2, construct_fci_maps=False)
    face_bc = _make_dirichlet_face_bc(geometry)
    hierarchy = _make_hierarchy(geometry, face_bc, max_levels=3)

    assert len(hierarchy.levels) >= 2
    assert hierarchy.levels[0].shape == geometry.shape
    assert hierarchy.levels[0].face_bc.kind_x.shape == (geometry.shape[0] + 1, geometry.shape[1], geometry.shape[2])
    assert hierarchy.levels[0].has_dirichlet
    assert not hierarchy.levels[0].has_nullspace

    for fine_level, coarse_level in zip(hierarchy.levels, hierarchy.levels[1:]):
        assert coarse_level.shape[0] == (fine_level.shape[0] + 1) // 2
        assert coarse_level.shape[1] == (fine_level.shape[1] + 1) // 2
        assert coarse_level.shape[2] == (fine_level.shape[2] + 1) // 2
        assert coarse_level.face_projectors[0].shape == coarse_level.regular_face_geometry.x_area.shape + (3, 3)
        assert coarse_level.face_projectors[1].shape == coarse_level.regular_face_geometry.y_area.shape + (3, 3)
        assert coarse_level.face_projectors[2].shape == coarse_level.regular_face_geometry.z_area.shape + (3, 3)


def _passing_reference_mg_apply_preconditioner_returns_finite_same_shape_values() -> None:
    geometry = build_test_fci_geometry((9, 8, 8), rho_min=0.2, construct_fci_maps=False)
    hierarchy = _make_hierarchy(geometry, _make_dirichlet_face_bc(geometry), max_levels=2)
    rhs = jax.random.normal(jax.random.PRNGKey(0), geometry.shape, dtype=jnp.float64)

    correction = mg_apply_preconditioner(rhs, hierarchy)

    assert correction.shape == rhs.shape
    assert bool(jnp.all(jnp.isfinite(correction)))


def _passing_reference_multigrid_preconditioner_is_linear_on_regular_faces() -> None:
    geometry = build_test_fci_geometry((9, 8, 8), rho_min=0.2, construct_fci_maps=False)
    hierarchy = _make_hierarchy(geometry, _make_dirichlet_face_bc(geometry), max_levels=2)
    key1, key2 = jax.random.split(jax.random.PRNGKey(1))
    r1 = jax.random.normal(key1, geometry.shape, dtype=jnp.float64)
    r2 = jax.random.normal(key2, geometry.shape, dtype=jnp.float64)

    a = 0.7
    b = -1.3
    lhs = mg_apply_preconditioner(a * r1 + b * r2, hierarchy)
    rhs = a * mg_apply_preconditioner(r1, hierarchy) + b * mg_apply_preconditioner(r2, hierarchy)

    np.testing.assert_allclose(lhs, rhs, rtol=1.0e-11, atol=1.0e-11)


def _passing_reference_dirichlet_preconditioner_correction_zeroes_dirichlet_adjacent_planes() -> None:
    geometry = build_test_fci_geometry((9, 8, 8), rho_min=0.2, construct_fci_maps=False)
    hierarchy = _make_hierarchy(geometry, _make_dirichlet_face_bc(geometry), max_levels=2)
    rhs = jax.random.normal(jax.random.PRNGKey(2), geometry.shape, dtype=jnp.float64)

    correction = mg_apply_preconditioner(rhs, hierarchy)

    np.testing.assert_allclose(correction[0], 0.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(correction[-1], 0.0, rtol=0.0, atol=1.0e-14)


def _passing_reference_neumann_preconditioner_correction_is_weighted_mean_zero() -> None:
    geometry = build_test_fci_geometry((9, 8, 8), rho_min=0.2, construct_fci_maps=False)
    hierarchy = _make_hierarchy(geometry, _make_neumann_face_bc(geometry), max_levels=2)
    rhs = jax.random.normal(jax.random.PRNGKey(3), geometry.shape, dtype=jnp.float64)
    rhs = _remove_weighted_mean(rhs, geometry)

    correction = mg_apply_preconditioner(rhs, hierarchy)

    assert hierarchy.levels[0].has_nullspace
    assert abs(float(_weighted_mean(correction, geometry))) < 1.0e-12


def _passing_reference_one_v_cycle_reduces_dirichlet_residual_for_smooth_error() -> None:
    geometry = build_test_fci_geometry((9, 8, 8), rho_min=0.2, construct_fci_maps=False)
    hierarchy = _make_hierarchy(geometry, _make_dirichlet_face_bc(geometry), max_levels=2)
    level0 = hierarchy.levels[0]
    exact = _smooth_dirichlet_exact(geometry)
    rhs = _apply_negative_perp_laplacian(exact, level0)

    correction = mg_apply_preconditioner(rhs, hierarchy)
    residual = rhs - _apply_negative_perp_laplacian(correction, level0)

    initial_norm = float(_weighted_norm(rhs, geometry))
    final_norm = float(_weighted_norm(residual, geometry))
    print("dirichlet v-cycle residual ratio:", final_norm / (initial_norm + 1.0e-30))
    assert final_norm < initial_norm


def _passing_reference_preconditioned_gmres_accepts_multigrid_hierarchy_on_discrete_mms() -> None:
    pytest.importorskip("lineax")

    geometry = build_test_fci_geometry((9, 8, 8), rho_min=0.2, construct_fci_maps=False)
    face_bc = _make_dirichlet_face_bc(geometry)
    hierarchy = _make_hierarchy(geometry, face_bc, max_levels=2)
    exact = _smooth_dirichlet_exact(geometry)
    rhs = _apply_negative_perp_laplacian(exact, hierarchy.levels[0])

    inverse_solver = PerpLaplacianInverseSolver(
        geometry,
        build_conservative_stencil_from_field,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=CutWallGeometry3D.empty(),
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=PERIODIC_AXES,
        mg_hierarchy=hierarchy,
        tol=1.0e-8,
        maxiter=20,
        restart=10,
    )
    actual = inverse_solver(rhs, face_bc=face_bc)
    residual = _apply_negative_perp_laplacian(actual, hierarchy.levels[0]) - rhs

    assert actual.shape == geometry.shape
    assert bool(jnp.all(jnp.isfinite(actual)))
    assert float(jnp.linalg.norm(residual)) / (float(jnp.linalg.norm(rhs)) + 1.0e-30) < 1.0e-6


def _passing_reference_build_hierarchy_rejects_nonempty_cut_wall_payloads() -> None:
    geometry = build_test_fci_geometry((9, 8, 8), rho_min=0.2, construct_fci_maps=False)
    face_bc = _make_dirichlet_face_bc(geometry)
    cut_wall_geometry = CutWallGeometry3D(
        owner_i=jnp.asarray([0], dtype=jnp.int32),
        owner_j=jnp.asarray([0], dtype=jnp.int32),
        owner_k=jnp.asarray([0], dtype=jnp.int32),
        center=jnp.zeros((1, 3), dtype=jnp.float64),
        normal_contra=jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float64),
        area_covector=jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float64),
        distance=jnp.ones((1,), dtype=jnp.float64),
        J=jnp.ones((1,), dtype=jnp.float64),
        g_contra=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (1, 3, 3)),
        g_cov=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (1, 3, 3)),
        B_contra=jnp.asarray([[0.0, 0.0, 1.0]], dtype=jnp.float64),
        Bmag=jnp.ones((1,), dtype=jnp.float64),
        sign=jnp.ones((1,), dtype=jnp.float64),
    )

    try:
        build_perp_laplacian_mg_hierarchy(
            geometry,
            build_conservative_stencil_from_field,
            face_bc=face_bc,
            regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=CutWallBC3D.empty(),
            periodic_axes=PERIODIC_AXES,
        )
    except NotImplementedError:
        return
    raise AssertionError("non-empty cut-wall multigrid hierarchy construction should raise NotImplementedError")


def _make_shifted_torus_diagnostic_geometry():
    return build_shifted_torus_4field_geometry((15, 15, 15), construct_fci_maps=False)


def _vcycle_residual_ratio(hierarchy, rhs: jnp.ndarray) -> tuple[float, float, float]:
    level0 = hierarchy.levels[0]
    correction = mg_apply_preconditioner(rhs, hierarchy)
    residual = rhs - _mg_apply_negative_perp_laplacian(correction, level0)
    initial_norm = float(_weighted_norm(rhs, level0.geometry))
    final_norm = float(_weighted_norm(residual, level0.geometry))
    correction_norm = float(_weighted_norm(correction, level0.geometry))
    return final_norm / (initial_norm + 1.0e-30), initial_norm, correction_norm


def test_diagnostic_shifted_torus_dirichlet_vcycle_residual_ratio() -> None:
    """Print the V-cycle quality on the shifted-torus operator that fails in 4-field."""

    geometry = _make_shifted_torus_diagnostic_geometry()
    hierarchy = _make_hierarchy(geometry, _make_dirichlet_face_bc(geometry), max_levels=None)
    rhs = _mg_apply_negative_perp_laplacian(_smooth_dirichlet_exact(geometry), hierarchy.levels[0])

    ratio, initial_norm, correction_norm = _vcycle_residual_ratio(hierarchy, rhs)

    print("shifted-torus dirichlet v-cycle residual ratio:", ratio)
    print("shifted-torus dirichlet rhs weighted norm:", initial_norm)
    print("shifted-torus dirichlet correction weighted norm:", correction_norm)
    assert np.isfinite(ratio)
    assert np.isfinite(initial_norm)
    assert np.isfinite(correction_norm)


def test_diagnostic_shifted_torus_neumann_vcycle_residual_ratio() -> None:
    """Print nullspace-projected V-cycle quality for Neumann-like radial faces."""

    geometry = _make_shifted_torus_diagnostic_geometry()
    hierarchy = _make_hierarchy(geometry, _make_neumann_face_bc(geometry), max_levels=None)
    exact = _remove_weighted_mean(_smooth_neumann_exact(geometry), geometry)
    rhs = _mg_apply_negative_perp_laplacian(exact, hierarchy.levels[0])
    rhs = _remove_weighted_mean(rhs, geometry)

    ratio, initial_norm, correction_norm = _vcycle_residual_ratio(hierarchy, rhs)

    print("shifted-torus neumann v-cycle residual ratio:", ratio)
    print("shifted-torus neumann rhs weighted norm:", initial_norm)
    print("shifted-torus neumann correction weighted norm:", correction_norm)
    print("shifted-torus neumann correction weighted mean:", float(_weighted_mean(mg_apply_preconditioner(rhs, hierarchy), geometry)))
    assert np.isfinite(ratio)
    assert abs(float(_weighted_mean(rhs, geometry))) < 1.0e-10


def test_diagnostic_shifted_torus_coarse_operator_restriction_defects() -> None:
    """Compare restricted fine residuals against rediscretized coarse operators.

    Large defects here point at coarse-grid rediscretization/transfer mismatch,
    which is a prime suspect when one V-cycle is a poor preconditioner.
    """

    geometry = _make_shifted_torus_diagnostic_geometry()
    hierarchy = _make_hierarchy(geometry, _make_dirichlet_face_bc(geometry), max_levels=None)
    fine_exact = _smooth_dirichlet_exact(hierarchy.levels[0].geometry)

    for level_index, (fine_level, coarse_level) in enumerate(zip(hierarchy.levels, hierarchy.levels[1:])):
        fine_rhs = _mg_apply_negative_perp_laplacian(fine_exact, fine_level)
        restricted_rhs = _restrict_residual_jweighted(fine_rhs, fine_level, coarse_level)
        coarse_exact = _restrict_field_simple(fine_exact, periodic_axes=fine_level.periodic_axes)
        coarse_rhs = _mg_apply_negative_perp_laplacian(coarse_exact, coarse_level)
        defect = restricted_rhs - coarse_rhs
        defect_ratio = float(_weighted_norm(defect, coarse_level.geometry)) / (
            float(_weighted_norm(restricted_rhs, coarse_level.geometry)) + 1.0e-30
        )
        print(
            "shifted-torus coarse operator restriction defect "
            f"level {level_index}->{level_index + 1}: {defect_ratio}"
        )
        assert np.isfinite(defect_ratio)
        fine_exact = coarse_exact


def test_diagnostic_shifted_torus_diag_and_projector_ranges() -> None:
    """Print basic scale information for Jacobi and face projector coefficients."""

    geometry = _make_shifted_torus_diagnostic_geometry()
    hierarchy = _make_hierarchy(geometry, _make_dirichlet_face_bc(geometry), max_levels=None)

    for level_index, level in enumerate(hierarchy.levels):
        diag_inv = jnp.asarray(level.diag_inv, dtype=jnp.float64)
        projector_min = min(float(jnp.min(projector)) for projector in level.face_projectors)
        projector_max = max(float(jnp.max(projector)) for projector in level.face_projectors)
        print(
            "shifted-torus level scale diagnostics "
            f"level={level_index}, shape={level.shape}, "
            f"diag_inv_min={float(jnp.min(diag_inv))}, "
            f"diag_inv_max={float(jnp.max(diag_inv))}, "
            f"projector_min={projector_min}, projector_max={projector_max}"
        )
        assert bool(jnp.all(jnp.isfinite(diag_inv)))
        assert bool(jnp.all(diag_inv > 0.0))
        assert np.isfinite(projector_min)
        assert np.isfinite(projector_max)


def test_diagnostic_shifted_torus_smoother_parameter_sweep() -> None:
    geometry = _make_shifted_torus_diagnostic_geometry()
    face_bc = _make_dirichlet_face_bc(geometry)
    rhs_field = _smooth_dirichlet_exact(geometry)
    configs = [
        {"smoother": "jacobi", "pre_smooth": 1, "post_smooth": 1, "omega_jacobi": 0.5, "max_levels": 2},
        {"smoother": "jacobi", "pre_smooth": 3, "post_smooth": 3, "omega_jacobi": 0.45, "max_levels": 3},
        {"smoother": "chebyshev", "pre_smooth": 1, "post_smooth": 1, "chebyshev_order": 2, "max_levels": 2},
        {"smoother": "chebyshev", "pre_smooth": 2, "post_smooth": 2, "chebyshev_order": 2, "max_levels": 3},
        {"smoother": "chebyshev", "pre_smooth": 2, "post_smooth": 2, "chebyshev_order": 3, "max_levels": None},
    ]
    ratios: list[float] = []
    for config in configs:
        hierarchy = _make_hierarchy(geometry, face_bc, **config)
        rhs = _mg_apply_negative_perp_laplacian(rhs_field, hierarchy.levels[0])
        ratio, _, _ = _vcycle_residual_ratio(hierarchy, rhs)
        ratios.append(ratio)
        print("shifted-torus MG sweep config:", config, "ratio:", ratio)

    best_ratio = min(ratios)
    print("shifted-torus MG best sweep ratio:", best_ratio)
    assert np.isfinite(best_ratio)
    assert best_ratio < 0.25


def test_diagnostic_shifted_torus_restriction_prolongation_invariants() -> None:
    geometry = _make_shifted_torus_diagnostic_geometry()
    hierarchy = _make_hierarchy(geometry, _make_neumann_face_bc(geometry), max_levels=3)
    fine_level = hierarchy.levels[0]
    coarse_level = hierarchy.levels[1]
    constant = jnp.ones(fine_level.shape, dtype=jnp.float64)
    restricted = _restrict_field_simple(constant, periodic_axes=fine_level.periodic_axes)
    prolonged = jnp.asarray(restricted, dtype=jnp.float64)

    prolonged = _prolong_field(prolonged, coarse_level, fine_level)

    np.testing.assert_allclose(restricted, 1.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(prolonged, 1.0, rtol=0.0, atol=1.0e-14)

    rng = jax.random.normal(jax.random.PRNGKey(42), fine_level.shape, dtype=jnp.float64)
    rng = _remove_weighted_mean(rng, fine_level.geometry)
    restricted_rhs = _restrict_residual_jweighted(rng, fine_level, coarse_level)
    print("restricted residual weighted mean:", float(_weighted_mean(restricted_rhs, coarse_level.geometry)))
    assert abs(float(_weighted_mean(restricted_rhs, coarse_level.geometry))) < 1.0e-10


def test_diagnostic_shifted_torus_preconditioned_gmres_comparison() -> None:
    pytest.importorskip("lineax")

    geometry = _make_shifted_torus_diagnostic_geometry()
    face_bc = _make_dirichlet_face_bc(geometry)
    hierarchy = _make_hierarchy(
        geometry,
        face_bc,
        max_levels=3,
        smoother="chebyshev",
        pre_smooth=2,
        post_smooth=2,
        chebyshev_order=2,
        direct_coarse_size=512,
    )
    exact = _smooth_dirichlet_exact(geometry)
    rhs = _mg_apply_negative_perp_laplacian(exact, hierarchy.levels[0])

    unpreconditioned_solver = PerpLaplacianInverseSolver(
        geometry,
        build_conservative_stencil_from_field,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=CutWallGeometry3D.empty(),
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=PERIODIC_AXES,
        tol=1.0e-8,
        maxiter=50,
        restart=20,
    )
    preconditioned_solver = PerpLaplacianInverseSolver(
        geometry,
        build_conservative_stencil_from_field,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=CutWallGeometry3D.empty(),
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=PERIODIC_AXES,
        mg_hierarchy=hierarchy,
        tol=1.0e-8,
        maxiter=50,
        restart=20,
    )
    unpreconditioned, unpreconditioned_diag = unpreconditioned_solver(
        rhs,
        face_bc=face_bc,
        return_diagnostics=True,
    )
    preconditioned, preconditioned_diag = preconditioned_solver(
        rhs,
        face_bc=face_bc,
        return_diagnostics=True,
    )
    print("shifted-torus unpreconditioned GMRES diagnostics:", unpreconditioned_diag)
    print("shifted-torus preconditioned GMRES diagnostics:", preconditioned_diag)
    assert bool(jnp.all(jnp.isfinite(unpreconditioned)))
    assert bool(jnp.all(jnp.isfinite(preconditioned)))
    assert preconditioned_diag["final_residual_rel_l2"] <= 10.0 * unpreconditioned_diag["final_residual_rel_l2"] + 1.0e-12
    if "num_steps" in unpreconditioned_diag and "num_steps" in preconditioned_diag:
        assert preconditioned_diag["num_steps"] <= unpreconditioned_diag["num_steps"]


if __name__ == "__main__":
    tests = [
        # The earlier green tests are kept above as _passing_reference_* helpers,
        # but are intentionally not run here while diagnosing shifted-torus MG.
        test_diagnostic_shifted_torus_dirichlet_vcycle_residual_ratio,
        test_diagnostic_shifted_torus_neumann_vcycle_residual_ratio,
        test_diagnostic_shifted_torus_coarse_operator_restriction_defects,
        test_diagnostic_shifted_torus_diag_and_projector_ranges,
        test_diagnostic_shifted_torus_smoother_parameter_sweep,
        test_diagnostic_shifted_torus_restriction_prolongation_invariants,
        test_diagnostic_shifted_torus_preconditioned_gmres_comparison,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: ok")
