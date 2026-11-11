from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from drbx.geometry import (
    build_synthetic_stellarator_geometry,
)
from drbx.native.fci_vorticity import (
    apply_fci_vorticity_operator,
    solve_fci_vorticity_potential_cg,
)
from drbx.validation import (
    create_stellarator_fci_geometry_campaign_package,
    create_stellarator_fci_operator_campaign_package,
    create_stellarator_drb_pytree_campaign_package,
    create_stellarator_sol_showcase_package,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fci_vorticity_jacobi_preconditioner_reduces_fixed_budget_residual() -> None:
    geometry = build_synthetic_stellarator_geometry(nx=10, ny=8, nz=18)
    phi_exact = (
        jnp.sin(2.0 * geometry.poloidal_angle - 3.0 * geometry.toroidal_angle)
        + 0.35 * geometry.radial * jnp.cos(
            3.0 * geometry.poloidal_angle + geometry.toroidal_angle
        )
    )
    density = (
        1.0
        + 0.20 * geometry.radial
        + 0.06 * jnp.cos(geometry.poloidal_angle - 2.0 * geometry.toroidal_angle)
    )
    vorticity = apply_fci_vorticity_operator(
        phi_exact,
        density,
        geometry.metric,
        regularization=0.5,
    )

    unpreconditioned = solve_fci_vorticity_potential_cg(
        vorticity,
        density,
        geometry.metric,
        iterations=10,
        regularization=0.5,
    )
    jacobi = solve_fci_vorticity_potential_cg(
        vorticity,
        density,
        geometry.metric,
        iterations=10,
        regularization=0.5,
        preconditioner="jacobi",
    )

    assert jacobi.preconditioner == "jacobi"
    assert float(jacobi.residual_l2) < 0.75 * float(unpreconditioned.residual_l2)


def test_fci_vorticity_boussinesq_non_boussinesq_limit_and_contrast() -> None:
    geometry = build_synthetic_stellarator_geometry(nx=8, ny=6, nz=14)
    phi = (
        jnp.sin(jnp.pi * geometry.radial)
        * jnp.cos(2.0 * geometry.poloidal_angle - 3.0 * geometry.toroidal_angle)
        + 0.15
        * geometry.radial
        * jnp.sin(geometry.poloidal_angle + 2.0 * geometry.toroidal_angle)
    )
    variable_density = (
        1.0
        + 0.40 * geometry.radial
        + 0.12 * jnp.cos(geometry.poloidal_angle - geometry.toroidal_angle)
    )
    boussinesq = apply_fci_vorticity_operator(
        phi,
        variable_density,
        geometry.metric,
        boussinesq=True,
    )
    non_boussinesq = apply_fci_vorticity_operator(
        phi,
        variable_density,
        geometry.metric,
        boussinesq=False,
    )
    relative_difference = np.linalg.norm(np.asarray(non_boussinesq - boussinesq)) / max(
        np.linalg.norm(np.asarray(boussinesq)),
        1.0e-30,
    )

    constant_coefficient_density = 2.3 * jnp.square(geometry.metric.Bxy)
    constant_boussinesq = apply_fci_vorticity_operator(
        phi,
        constant_coefficient_density,
        geometry.metric,
        boussinesq=True,
    )
    constant_non_boussinesq = apply_fci_vorticity_operator(
        phi,
        constant_coefficient_density,
        geometry.metric,
        boussinesq=False,
    )

    assert relative_difference > 5.0e-2
    assert np.max(np.abs(np.asarray(constant_boussinesq - constant_non_boussinesq))) < 1.0e-10


def test_stellarator_fci_geometry_campaign_generates_passing_artifacts(tmp_path: Path) -> None:
    artifacts = create_stellarator_fci_geometry_campaign_package(output_root=tmp_path / "geometry", nx=10, ny=8, nz=16)

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    assert report["map_diagnostics"]["radial_shift_linf_cells"] > 0.0


def test_stellarator_fci_operator_campaign_generates_passing_artifacts(tmp_path: Path) -> None:
    artifacts = create_stellarator_fci_operator_campaign_package(output_root=tmp_path / "operators")

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["interpolation_convergence_slope"] > 1.55
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


def test_stellarator_drb_pytree_campaign_generates_jvp_and_scaling_metrics(tmp_path: Path) -> None:
    artifacts = create_stellarator_drb_pytree_campaign_package(
        output_root=tmp_path / "pytree_drb",
        nx=8,
        ny=6,
        nz=12,
        steps=3,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["jvp_relative_error"] < 5.0e-3
    assert report["boussinesq_gate_potential_boussinesq"] is True
    assert report["non_boussinesq_gate_potential_boussinesq"] is False
    assert report["boussinesq_non_boussinesq_potential_relative_l2"] > 1.0e-4
    assert report["boussinesq_non_boussinesq_rhs_state_linf"] < 1.0e-12
    assert report["non_boussinesq_jvp_relative_error"] < 5.0e-3
    assert report["density_over_b_squared_contrast"] > 1.0
    assert report["potential_feedback_strength"] > 0.0
    assert report["potential_feedback_plasma_rhs_linf"] > 1.0e-8
    assert report["potential_feedback_neutral_rhs_linf"] < 1.0e-12
    assert report["potential_feedback_jvp_relative_error"] < 5.0e-3
    assert report["vmap_serial_linf"] < 1.0e-8
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    with np.load(artifacts.arrays_npz_path) as arrays:
        assert "potential_feedback_plasma_rhs_difference_slice" in arrays.files


def test_stellarator_sol_showcase_generates_dynamics_movie_and_metrics(tmp_path: Path) -> None:
    artifacts = create_stellarator_sol_showcase_package(output_root=tmp_path / "showcase", nx=10, ny=10, nz=18)

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["final_rms_fluctuation"] > 1.0e-3
    assert artifacts.snapshot_png_path.exists()
    assert artifacts.diagnostics_png_path.exists()
    assert artifacts.poster_png_path.exists()
    assert artifacts.movie_gif_path.exists()
