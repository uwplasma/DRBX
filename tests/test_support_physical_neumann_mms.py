"""Manufactured oblique physical-Neumann regression for support-paired FCI."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from axis_regular_operator_support import polar_fixture
from test_fci_projected_fine_grid_control_volume import _build_physical_ghost_filler
from drbx.geometry import SIDE_PHYSICAL, StencilBuilderContext
from drbx.native.fci_boundaries import BC_NEUMANN, LocalBoundaryFaceBC3D
from drbx.native.fci_halo import LocalPeriodicTopologyRule3D, TopologyHaloFiller3D
from drbx.native.fci_gmres import SolvaxGmresConfig
from drbx.native.fci_operators import LocalPerpLaplacianInverseSolver


def _identity_metric(metric):
    one = jnp.ones_like(metric.J_halo)
    zero = jnp.zeros_like(one)
    return replace(
        metric,
        J_halo=one,
        g11_halo=one,
        g22_halo=one,
        g33_halo=one,
        g12_halo=zero,
        g13_halo=zero,
        g23_halo=zero,
        g_11_halo=one,
        g_22_halo=one,
        g_33_halo=one,
        g_12_halo=zero,
        g_13_halo=zero,
        g_23_halo=zero,
    )


def _oblique_bfield(field):
    b = jnp.zeros_like(field.B_contra_halo)
    b = b.at[..., 0].set(0.5).at[..., 1].set(0.5).at[..., 2].set(np.sqrt(0.5))
    return replace(field, B_contra_halo=b, Bmag_halo=jnp.ones_like(field.Bmag_halo))


def _mms_dense_operator(nx: int, ny: int, *, nonzero_neumann: bool = False):
    geometry, domain, *_prefix, exchange, _scalar, _vector, _flux = polar_fixture(
        (nx, ny, 2), 1
    )
    domain = replace(
        domain,
        shard_spec=replace(
            domain.shard_spec,
            axis_regular_axes=(False, False, False),
            side_kind_lower=(SIDE_PHYSICAL,) + domain.shard_spec.side_kind_lower[1:],
        ),
    )
    face_metric = replace(
        geometry.face_metric,
        x=_identity_metric(geometry.face_metric.x),
        y=_identity_metric(geometry.face_metric.y),
        z=_identity_metric(geometry.face_metric.z),
    )
    face_bfield = replace(
        geometry.face_bfield,
        x=_oblique_bfield(geometry.face_bfield.x),
        y=_oblique_bfield(geometry.face_bfield.y),
        z=_oblique_bfield(geometry.face_bfield.z),
    )
    geometry = replace(
        geometry,
        cell_metric=_identity_metric(geometry.cell_metric),
        face_metric=face_metric,
        cell_bfield=_oblique_bfield(geometry.cell_bfield),
        face_bfield=face_bfield,
        cell_volume_geometry=replace(
            geometry.cell_volume_geometry,
            volume=jnp.ones((nx, ny, 2)),
        ),
    )
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    face_bc = replace(
        face_bc,
        kind_x=face_bc.kind_x.at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN),
        mask_x=face_bc.mask_x.at[0].set(True).at[-1].set(True),
    )
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) * 2.0 * np.pi / ny
    if nonzero_neumann:
        sin_y = jnp.sin(jnp.asarray(y))[:, None]
        face_bc = replace(
            face_bc,
            value_x=face_bc.value_x.at[0].set(jnp.broadcast_to(-sin_y, face_bc.value_x[0].shape)).at[-1].set(jnp.broadcast_to(3.0 * sin_y, face_bc.value_x[-1].shape)),
        )
    solver = LocalPerpLaplacianInverseSolver(
        geometry=geometry,
        domain=domain,
        face_bc=face_bc,
        halo_exchange=exchange,
        topology_filler=TopologyHaloFiller3D(
            rules=(LocalPeriodicTopologyRule3D(),)
        ),
        physical_ghost_filler=_build_physical_ghost_filler(geometry.layout),
        axis_regular_axes=(False, False, False),
        neumann_normal_scheme="physical",
        stencil_builder_context=StencilBuilderContext(
            layout=geometry.layout, domain=domain
        ),
        operator_form="support-paired",
        config=SolvaxGmresConfig(regularization_epsilon=0.0),
    )
    full = jax.jit(
        lambda values: solver._apply_A(
            values,
            face_bc=face_bc,
            control_volume_boundary_bc=None,
            project_mean_zero=False,
            boundary_is_homogeneous=True,
        )
    )
    energy = jax.jit(
        lambda values: solver._apply_A_support_paired(
            values,
            face_bc=face_bc,
            control_volume_boundary_bc=None,
            project_mean_zero=False,
            boundary_is_homogeneous=True,
            include_boundary_flux=False,
        )
    )
    affine = jax.jit(
        lambda values: solver._apply_A(values, face_bc=face_bc,
            control_volume_boundary_bc=None, project_mean_zero=False,
            boundary_is_homogeneous=False)
    )
    columns = []
    for index in range(nx * ny):
        basis = np.zeros((nx, ny))
        basis.ravel()[index] = 1.0
        field = np.repeat(basis[:, :, None], 2, axis=2)
        columns.append(
            np.stack(
                (
                    np.asarray(full(field)).mean(axis=2).ravel(),
                    np.asarray(energy(field)).mean(axis=2).ravel(),
                )
            )
        )
    matrices = np.stack(columns, axis=-1)
    exact = np.cos(np.pi * x[:, None]) * np.sin(y[None, :])
    rhs = (0.75 * np.pi**2 + 0.75) * exact - 0.5 * np.pi * np.sin(
        np.pi * x[:, None]
    ) * np.cos(y[None, :])
    source = np.asarray(affine(jnp.zeros((nx, ny, 2)))).mean(axis=2).ravel()
    return matrices, exact.ravel(), rhs.ravel(), source


def _solve_augmented(matrix, rhs):
    n = matrix.shape[0]
    augmented = np.block(
        [[matrix, np.ones((n, 1))], [np.ones((1, n)) / n, np.zeros((1, 1))]]
    )
    solution = np.linalg.solve(augmented, np.r_[rhs, 0.0])
    return solution[:-1], solution[-1], np.linalg.norm(
        augmented @ solution - np.r_[rhs, 0.0]
    )


def test_oblique_physical_neumann_mms_converges_only_with_full_boundary_flux():
    coarse, exact_coarse, rhs_coarse, _ = _mms_dense_operator(3, 8)
    fine, exact_fine, rhs_fine, _ = _mms_dense_operator(6, 16)
    coarse_full, coarse_energy = coarse
    fine_full, fine_energy = fine

    results = []
    for full, energy, exact, rhs in (
        (coarse_full, coarse_energy, exact_coarse, rhs_coarse),
        (fine_full, fine_energy, exact_fine, rhs_fine),
    ):
        solved, multiplier, residual = _solve_augmented(full, rhs)
        energy_solved, _energy_multiplier, _energy_residual = _solve_augmented(
            energy, rhs
        )
        results.append(
            {
                "full_error": np.linalg.norm(solved - exact) / np.linalg.norm(exact),
                "energy_error": np.linalg.norm(energy_solved - exact)
                / np.linalg.norm(exact),
                "lambda": abs(multiplier),
                "augmented_residual": residual,
                "skew": np.linalg.norm(full - full.T) / np.linalg.norm(full),
            }
        )

    # The refinement gate is the physically relevant fine-grid result.
    assert results[1]["full_error"] < 0.3 * results[0]["full_error"]
    assert results[1]["full_error"] < 0.12
    assert results[0]["energy_error"] > 0.8
    assert results[1]["energy_error"] > 0.8
    assert results[0]["skew"] > 1.0e-3
    assert results[1]["skew"] > 1.0e-3
    for result in results:
        assert result["augmented_residual"] < 1.0e-10
        assert abs(result["lambda"]) < 1.0e-10


def test_oblique_physical_neumann_mms_with_nonzero_normal_derivative():
    """Nonzero physical-normal data must enter the affine boundary load."""
    results = []
    for nx, ny in ((3, 8), (6, 16)):
        (full, _energy), _unused_exact, _unused_rhs, source = _mms_dense_operator(
            nx, ny, nonzero_neumann=True
        )
        x = (np.arange(nx) + 0.5) / nx
        y = (np.arange(ny) + 0.5) * 2.0 * np.pi / ny
        exact = ((1.0 + x[:, None] + x[:, None] ** 2) * np.sin(y[None, :])).ravel()
        rhs = (
            0.75 * (x[:, None] ** 2 + x[:, None] - 1.0) * np.sin(y[None, :])
            + 0.5 * (1.0 + 2.0 * x[:, None]) * np.cos(y[None, :])
        ).ravel()
        solved, multiplier, residual = _solve_augmented(full, rhs - source)
        result = {
            "shape": [nx, ny],
            "error": float(np.linalg.norm(solved - exact) / np.linalg.norm(exact)),
            "lambda": float(multiplier),
            "residual": float(residual),
            "source_norm": float(np.linalg.norm(source)),
        }
        print("nonzero physical-Neumann MMS", result)
        results.append(result)
    assert results[1]["error"] < 0.35 * results[0]["error"]
    # With the explicitly physical-normal closure, the observed fine error is
    # 7.84e-2; retain a tight gate around that measured result.
    assert results[1]["error"] < 0.09
    for result in results:
        assert result["residual"] < 1.0e-10
        assert result["lambda"] < 1.0e-10
        assert result["source_norm"] > 1.0
