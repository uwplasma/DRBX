from __future__ import annotations

import os
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry import ConservativeStencilBuilder, RegularFaceGeometry3D, build_conservative_stencil_from_field
from drbx.native import Fci4FieldFreeDecayParameters, build_perp_laplacian_face_projectors
from drbx.native.fci_boundaries import CutWallBC3D, CutWallGeometry3D
from drbx.native.fci_operators import PerpLaplacianInverseSolver, perp_laplacian_conservative_op

from test_shifted_torus_4_field_free_decay import (  # noqa: E402
    _build_free_decay_boundary_conditions,
    build_shifted_torus_4field_geometry,
    resolution,
    rho_star,
    Te,
    mi_over_me,
    phi_inversion_regularization,
    phi_inversion_project_mean_zero,
    _phi_inversion_pin_point,
)


TARGET_TIME = 0.1435
TIME_OFFSET = 0.001
OUTPUT_PATH = Path("adiabatic_electron_free_decay_t015_omega_phi.png")
HISTORY_PATH = Path("adiabatic_electron_free_decay_histories.npz")


def _nearest_index(times: np.ndarray, target_time: float) -> int:
    if times.size == 0:
        raise ValueError("history is empty")
    return int(np.argmin(np.abs(times - float(target_time))))


def _load_history(history_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(history_path, allow_pickle=False) as history:
        times = np.asarray(history["times"], dtype=np.float64)
        omega = np.asarray(history["omega"], dtype=np.float64)
    return times, omega


def _rms(field: np.ndarray) -> float:
    field = np.asarray(field, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(field))))


def _max_abs(field: np.ndarray) -> float:
    field = np.asarray(field, dtype=np.float64)
    return float(np.max(np.abs(field)))


def _reconstruct_phi(
    geometry,
    omega_snapshot: np.ndarray,
    *,
    phi_face_bc,
    phi_inverse_solver: PerpLaplacianInverseSolver,
) -> np.ndarray:
    phi = phi_inverse_solver(
        jnp.asarray(omega_snapshot, dtype=jnp.float64),
        face_bc=phi_face_bc,
    )
    jax.block_until_ready(phi)
    return np.asarray(phi, dtype=np.float64)


def _print_phi_jump_diagnostics(
    geometry,
    phi_before_full: np.ndarray,
    phi_after_full: np.ndarray,
    omega_before_full: np.ndarray,
    omega_after_full: np.ndarray,
    *,
    phi_face_bc,
) -> None:
    delta_phi = np.asarray(phi_after_full - phi_before_full, dtype=np.float64)
    delta_omega = np.asarray(omega_after_full - omega_before_full, dtype=np.float64)
    a_delta_phi = _omega_from_phi_snapshot(geometry, delta_phi, phi_face_bc=phi_face_bc)
    l_delta_phi = np.asarray(-a_delta_phi, dtype=np.float64)

    weights = np.asarray(geometry.cell_metric.J, dtype=np.float64)
    numerator = float(np.sum(weights * delta_phi * a_delta_phi))
    denominator = float(np.sum(weights * np.square(delta_phi)))
    lambda_eff = numerator / denominator if denominator != 0.0 else float("nan")

    consistency_error = np.asarray(delta_omega - a_delta_phi, dtype=np.float64)
    tiny = 1.0e-300

    print("phi jump diagnostics:")
    print(f"  rms(phi_before)={_rms(phi_before_full):.6e}, rms(phi_after)={_rms(phi_after_full):.6e}")
    print(f"  max(abs(phi_before))={_max_abs(phi_before_full):.6e}, max(abs(phi_after))={_max_abs(phi_after_full):.6e}")
    print(f"  rms(omega_before)={_rms(omega_before_full):.6e}, rms(omega_after)={_rms(omega_after_full):.6e}")
    print(f"  max(abs(omega_before))={_max_abs(omega_before_full):.6e}, max(abs(omega_after))={_max_abs(omega_after_full):.6e}")
    print(f"  rms(delta_phi)={_rms(delta_phi):.6e}")
    print(f"  max(abs(delta_phi))={_max_abs(delta_phi):.6e}")
    print(f"  rms(delta_omega)={_rms(delta_omega):.6e}")
    print(f"  max(abs(delta_omega))={_max_abs(delta_omega):.6e}")
    print("  near-null check on delta_phi using the same perpendicular operator convention:")
    print(f"    rms(L_delta_phi)={_rms(l_delta_phi):.6e}")
    print(f"    max(abs(L_delta_phi))={_max_abs(l_delta_phi):.6e}")
    print(f"    rms(L_delta_phi) / rms(delta_phi)={_rms(l_delta_phi) / max(_rms(delta_phi), tiny):.6e}")
    print(f"    max(abs(L_delta_phi)) / max(abs(delta_phi))={_max_abs(l_delta_phi) / max(_max_abs(delta_phi), tiny):.6e}")
    print("  Rayleigh quotient for A = -grad_perp^2:")
    print(f"    numerator={numerator:.6e}")
    print(f"    denominator={denominator:.6e}")
    print(f"    lambda_eff={lambda_eff:.6e}")
    print("  delta_omega vs A delta_phi consistency:")
    print(f"    rms(A_delta_phi)={_rms(a_delta_phi):.6e}")
    print(f"    rms(consistency_error)={_rms(consistency_error):.6e}")
    print(
        f"    rms(consistency_error) / max(rms(delta_omega), tiny)="
        f"{_rms(consistency_error) / max(_rms(delta_omega), tiny):.6e}"
    )
    print(f"    max(abs(A_delta_phi))={_max_abs(a_delta_phi):.6e}")
    print(f"    max(abs(consistency_error))={_max_abs(consistency_error):.6e}")


def _save_comparison_figure(
    geometry,
    times: np.ndarray,
    omega_history: np.ndarray,
    before_index: int,
    after_index: int,
    *,
    phi_face_bc,
    phi_inverse_solver: PerpLaplacianInverseSolver,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    z_index = int(geometry.shape[2] // 2)
    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    theta_grid, radius_grid = np.meshgrid(theta_values, x_values)

    omega_before = omega_history[before_index, :, :, z_index]
    omega_after = omega_history[after_index, :, :, z_index]
    phi_before_full = _reconstruct_phi_snapshot(
        geometry,
        omega_history[before_index],
        phi_face_bc=phi_face_bc,
        phi_inverse_solver=phi_inverse_solver,
    )
    phi_after_full = _reconstruct_phi_snapshot(
        geometry,
        omega_history[after_index],
        phi_face_bc=phi_face_bc,
        phi_inverse_solver=phi_inverse_solver,
    )
    _print_phi_jump_diagnostics(
        geometry,
        phi_before_full,
        phi_after_full,
        omega_history[before_index],
        omega_history[after_index],
        phi_face_bc=phi_face_bc,
    )
    phi_before = phi_before_full[:, :, z_index]
    phi_after = phi_after_full[:, :, z_index]
    phi_before_norm = phi_before / max(float(np.max(np.abs(phi_before))), 1.0e-30)
    phi_after_norm = phi_after / max(float(np.max(np.abs(phi_after))), 1.0e-30)
    omega_from_phi_before = _omega_from_phi_snapshot(geometry, phi_before_full, phi_face_bc=phi_face_bc)[:, :, z_index]
    omega_from_phi_after = _omega_from_phi_snapshot(geometry, phi_after_full, phi_face_bc=phi_face_bc)[:, :, z_index]
    omega_diff_before = np.abs(omega_before - omega_from_phi_before)
    omega_diff_after = np.abs(omega_after - omega_from_phi_after)
    pin_index = tuple(int(axis // 2) for axis in geometry.shape)
    omega_diff_before[pin_index[0], pin_index[1]] = 0.0
    omega_diff_after[pin_index[0], pin_index[1]] = 0.0

    omega_vmax = float(np.max(np.abs(np.stack([omega_before, omega_after], axis=0))))
    phi_vmax = float(np.max(np.abs(np.stack([phi_before, phi_after], axis=0))))
    phi_norm_vmax = 1.0
    omega_from_phi_vmax = float(np.max(np.abs(np.stack([omega_from_phi_before, omega_from_phi_after], axis=0))))
    omega_diff_vmax = float(np.max(np.stack([omega_diff_before, omega_diff_after], axis=0)))
    omega_vmax = omega_vmax if omega_vmax > 0.0 else 1.0
    phi_vmax = phi_vmax if phi_vmax > 0.0 else 1.0
    omega_from_phi_vmax = omega_from_phi_vmax if omega_from_phi_vmax > 0.0 else 1.0
    omega_diff_vmax = omega_diff_vmax if omega_diff_vmax > 0.0 else 1.0
    shared_omega_vmax = omega_vmax

    fig, axes = plt.subplots(5, 2, figsize=(12.5, 19.0), subplot_kw={"projection": "polar"}, constrained_layout=True)
    omega_images = []
    phi_images = []
    phi_norm_images = []
    omega_from_phi_images = []
    omega_diff_images = []
    panels = (
        ("omega", omega_before, shared_omega_vmax, "coolwarm", float(times[before_index])),
        ("omega", omega_after, shared_omega_vmax, "coolwarm", float(times[after_index])),
        ("phi", phi_before, phi_vmax, "coolwarm", float(times[before_index])),
        ("phi", phi_after, phi_vmax, "coolwarm", float(times[after_index])),
        ("phi / max(|phi|)", phi_before_norm, phi_norm_vmax, "coolwarm", float(times[before_index])),
        ("phi / max(|phi|)", phi_after_norm, phi_norm_vmax, "coolwarm", float(times[after_index])),
        ("omega(from phi)", omega_from_phi_before, shared_omega_vmax, "coolwarm", float(times[before_index])),
        ("omega(from phi)", omega_from_phi_after, shared_omega_vmax, "coolwarm", float(times[after_index])),
        ("|omega - omega(from phi)|", omega_diff_before, omega_diff_vmax, "magma", float(times[before_index])),
        ("|omega - omega(from phi)|", omega_diff_after, omega_diff_vmax, "magma", float(times[after_index])),
    )

    for index, (ax, (label, field, vmax, cmap, time_value)) in enumerate(zip(axes.ravel(), panels, strict=True)):
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_ylim(0.0, float(x_values[-1]))
        ax.set_yticklabels([])
        ax.set_title(f"{label}, t={float(time_value):.3f}")
        image = ax.pcolormesh(
            theta_grid,
            radius_grid,
            field,
            shading="auto",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
        )
        if index < 2:
            omega_images.append(image)
        elif index < 4:
            phi_images.append(image)
        elif index < 6:
            phi_norm_images.append(image)
        elif index < 8:
            omega_from_phi_images.append(image)
        else:
            omega_diff_images.append(image)

    fig.colorbar(
        omega_images[0],
        ax=axes[0, :].tolist(),
        pad=0.08,
        shrink=0.85,
    )
    fig.colorbar(phi_images[0], ax=axes[1, :].tolist(), pad=0.08, shrink=0.85)
    fig.colorbar(phi_norm_images[0], ax=axes[2, :].tolist(), pad=0.08, shrink=0.85)
    fig.colorbar(omega_from_phi_images[0], ax=axes[3, :].tolist(), pad=0.08, shrink=0.85)
    fig.colorbar(omega_diff_images[0], ax=axes[4, :].tolist(), pad=0.08, shrink=0.85)

    fig.suptitle(f"Shifted-torus free decay, before/after t={TARGET_TIME:.2f}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _reconstruct_phi_snapshot(
    geometry,
    omega_snapshot: np.ndarray,
    *,
    phi_face_bc,
    phi_inverse_solver: PerpLaplacianInverseSolver,
) -> np.ndarray:
    phi = phi_inverse_solver(
        jnp.asarray(omega_snapshot, dtype=jnp.float64),
        face_bc=phi_face_bc,
    )
    jax.block_until_ready(phi)
    return np.asarray(phi, dtype=np.float64)


def _omega_from_phi_snapshot(
    geometry,
    phi_snapshot: np.ndarray,
    *,
    phi_face_bc,
) -> np.ndarray:
    conservative_stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    phi_stencil = conservative_stencil_builder(
        jnp.asarray(phi_snapshot, dtype=jnp.float64),
        geometry,
        (False, True, True),
        phi_face_bc,
    )
    omega = -perp_laplacian_conservative_op(
        phi_stencil,
        geometry,
        face_projectors=build_perp_laplacian_face_projectors(geometry),
        face_bc=phi_face_bc,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        periodic_axes=(False, True, True),
    )
    jax.block_until_ready(omega)
    return np.asarray(omega, dtype=np.float64)


def main() -> None:
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"missing history file: {HISTORY_PATH}")

    times, omega_history = _load_history(HISTORY_PATH)
    geometry = build_shifted_torus_4field_geometry((resolution, resolution, resolution))
    boundary_conditions = _build_free_decay_boundary_conditions(geometry, 0.0)
    parameters = Fci4FieldFreeDecayParameters(
        rho_star=rho_star,
        Te=Te,
        mi_over_me=mi_over_me,
        phi_inversion_tol=5e-5,
        phi_inversion_restart=300,
        phi_inversion_regularization=phi_inversion_regularization,
        density_perp_diffusion=1.0e-2,
        omega_perp_diffusion=1.0e-2,
        v_ion_parallel_perp_diffusion=1.0e-2,
        v_electron_parallel_perp_diffusion=1.0e-2,
    )
    print(
        "phi inversion operator/gauge: A = -L + epsilon I, "
        f"epsilon={float(parameters.phi_inversion_regularization):.6e}, "
        f"project_mean_zero={bool(phi_inversion_project_mean_zero)}"
    )

    face_projectors = build_perp_laplacian_face_projectors(geometry)
    conservative_stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    phi_inverse_solver = PerpLaplacianInverseSolver(
        geometry,
        conservative_stencil_builder,
        tol=float(parameters.phi_inversion_tol),
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        face_projectors=face_projectors,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=CutWallGeometry3D.empty(),
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=(False, True, True),
        pin_point=_phi_inversion_pin_point(geometry, parameters.phi_inversion_regularization),
        pin_value=0.0,
        project_mean_zero=phi_inversion_project_mean_zero,
        target_mean_phi=0.0 if phi_inversion_project_mean_zero else None,
        regularization_epsilon=float(parameters.phi_inversion_regularization),
        gmres_debug=True,
    )

    before_index = _nearest_index(times, TARGET_TIME - TIME_OFFSET)
    after_index = _nearest_index(times, TARGET_TIME + TIME_OFFSET)
    _save_comparison_figure(
        geometry,
        times,
        omega_history,
        before_index,
        after_index,
        phi_face_bc=boundary_conditions.phi_face_bc,
        phi_inverse_solver=phi_inverse_solver,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
