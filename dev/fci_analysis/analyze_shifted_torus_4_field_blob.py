from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from test_mms_shifted_torus_4_field import build_shifted_torus_4field_geometry
from test_shifted_torus_4_field_free_decay import (
    _build_free_decay_boundary_conditions,
    _save_shifted_torus_free_decay_time_traces,
)
from dkx.geometry import ConservativeStencilBuilder, RegularFaceGeometry3D, build_conservative_stencil_from_field
from dkx.native import Fci4FieldBlobParameters, build_perp_laplacian_face_projectors
from dkx.native.fci_boundaries import CutWallBC3D, CutWallGeometry3D
from dkx.native.fci_operators import PerpLaplacianInverseSolver


rho_star = 1.0
Te = 1.0
mi_over_me = 1836.0
phi_inversion_tol = 5.0e-5
phi_inversion_restart = 200
phi_inversion_maxiter = 100
zeta0 = np.pi


def _resolve_history_path(path: Path | None) -> Path:
    if path is None:
        return _THIS_DIR / "blob_4field_histories.npz"
    if path.is_dir():
        candidate = path / "blob_4field_histories.npz"
        if candidate.exists():
            return candidate
        candidates = sorted(path.glob("*_histories.npz"))
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(f"could not find a unique *_histories.npz file in {path}")
    return path


def _load_history(history_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(history_path, allow_pickle=False) as history:
        times = np.asarray(history["times"], dtype=np.float64)
        density = np.asarray(history["density"], dtype=np.float64)
        omega = np.asarray(history["omega"], dtype=np.float64)
        v_ion_parallel = np.asarray(history["v_ion_parallel"], dtype=np.float64)
        v_electron_parallel = np.asarray(history["v_electron_parallel"], dtype=np.float64)
        metadata: dict[str, object] = {}
        for key in history.files:
            if key in {"times", "density", "omega", "v_ion_parallel", "v_electron_parallel"}:
                continue
            value = history[key]
            metadata[key] = value.item() if getattr(value, "shape", ()) == () else value
    return times, density, omega, v_ion_parallel, v_electron_parallel, metadata


def _blob_z_indices(geometry, center: float, count: int = 4) -> tuple[int, ...]:
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    center_index = int(np.argmin(np.abs(z_values - float(center))))
    offsets = np.arange(-(count // 2), -(count // 2) + count, dtype=np.int64)
    return tuple(int((center_index + offset) % int(z_values.shape[0])) for offset in offsets)


def _reconstruct_phi_history(
    geometry,
    omega_history: np.ndarray,
    *,
    phi_face_bc,
    parameters: Fci4FieldBlobParameters,
) -> np.ndarray:
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
        pin_point=None,
        pin_value=0.0,
        project_mean_zero=False,
        target_mean_phi=None,
        regularization_epsilon=float(parameters.phi_inversion_regularization),
        gmres_debug=False,
        check_residual=False,
    )

    phi_history: list[np.ndarray] = []
    total_snapshots = int(len(omega_history))
    progress_stride = max(1, total_snapshots // 10)
    for index, omega in enumerate(omega_history):
        phi = phi_inverse_solver(
            -jnp.asarray(omega, dtype=jnp.float64),
            face_bc=phi_face_bc,
        )
        jax.block_until_ready(phi)
        phi_history.append(np.asarray(phi, dtype=np.float64))
        if (index + 1) % progress_stride == 0 or index + 1 == total_snapshots:
            print(f"reconstructing phi movie: {index + 1}/{total_snapshots}", flush=True)
    return np.asarray(phi_history, dtype=np.float64)


def _save_phi_movie(
    times: np.ndarray,
    phi_history: np.ndarray,
    geometry,
    *,
    output_path: str,
    frame_stride: int = 2,
    title: str = "Shifted-torus 4-field blob potential evolution",
    z_indices: tuple[int, int, int, int] | None = None,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    if z_indices is None:
        z_indices = tuple(int(idx) for idx in np.linspace(0, int(z_values.shape[0] - 1), 4))
    theta_grid, radius_grid = np.meshgrid(theta_values, x_values)

    phi_data = np.asarray(phi_history, dtype=np.float64)
    vmax = float(np.max(np.abs(phi_data)))
    vmax = vmax if vmax > 0.0 else 1.0
    phi_norm = colors.Normalize(vmin=-vmax, vmax=vmax)

    frame_indices = np.arange(0, int(times.shape[0]), max(1, int(frame_stride)), dtype=np.int64)
    if frame_indices[-1] != int(times.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times.shape[0]) - 1)

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.4), subplot_kw={"projection": "polar"}, constrained_layout=True)
    images = []
    for col, z_index in enumerate(z_indices):
        ax = axes[col]
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_ylim(0.0, float(x_values[-1]))
        ax.set_yticklabels([])
        ax.set_title(f"phi, zeta={z_values[z_index]:.3f}")
        image = ax.pcolormesh(
            theta_grid,
            radius_grid,
            phi_data[0, :, :, z_index],
            shading="auto",
            cmap="coolwarm",
            norm=phi_norm,
        )
        images.append(image)
    fig.colorbar(images[0], ax=list(axes), location="right", pad=0.02, shrink=0.88)
    suptitle = fig.suptitle(title)

    def update(frame_index: int):
        actual_index = int(frame_indices[frame_index])
        time_value = float(times[actual_index])
        for col, z_index in enumerate(z_indices):
            images[col].set_array(phi_data[actual_index, :, :, z_index].ravel())
            images[col].set_norm(phi_norm)
            axes[col].set_title(f"phi, zeta={z_values[z_index]:.3f}, t={time_value:.3f}")
        suptitle.set_text(f"{title}, t={time_value:.3f}")
        return images

    animator = animation.FuncAnimation(fig, update, frames=int(frame_indices.shape[0]), interval=100, blit=False)
    writer = animation.PillowWriter(fps=10)
    animator.save(output_path, writer=writer)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild plots from a saved shifted-torus blob run.")
    parser.add_argument(
        "history",
        nargs="?",
        type=Path,
        default=None,
        help="History .npz file or directory containing blob_4field_histories.npz",
    )
    args = parser.parse_args()

    history_path = _resolve_history_path(args.history)
    if not history_path.exists():
        raise FileNotFoundError(f"missing history file: {history_path}")

    times, density_history, omega_history, v_ion_history, v_electron_history, metadata = _load_history(history_path)
    resolution = int(metadata.get("resolution", density_history.shape[1]))
    geometry = build_shifted_torus_4field_geometry((resolution, resolution, resolution))
    boundary_conditions = _build_free_decay_boundary_conditions(geometry, 0.0)
    parameters = Fci4FieldBlobParameters(
        rho_star=rho_star,
        Te=Te,
        mi_over_me=mi_over_me,
        phi_inversion_tol=phi_inversion_tol,
        phi_inversion_maxiter=phi_inversion_maxiter,
        phi_inversion_restart=phi_inversion_restart,
        density_perp_diffusion=1.0e-3,
        omega_perp_diffusion=1.0e-3,
        v_ion_parallel_perp_diffusion=1.0e-4,
        v_electron_parallel_perp_diffusion=1.0e-3,
    )

    stem = history_path.stem.replace("_histories", "")
    output_dir = history_path.parent
    time_traces_path = output_dir / f"{stem}_time_traces.png"
    movie_path = output_dir / f"{stem}_phi.gif"

    print(f"rebuilding time traces from {history_path}", flush=True)
    phi_history = _reconstruct_phi_history(
        geometry,
        np.asarray(omega_history, dtype=np.float64),
        phi_face_bc=boundary_conditions.phi_face_bc,
        parameters=parameters,
    )
    _save_shifted_torus_free_decay_time_traces(
        jnp.asarray(times, dtype=jnp.float64),
        jnp.asarray(density_history, dtype=jnp.float64),
        jnp.asarray(omega_history, dtype=jnp.float64),
        jnp.asarray(v_ion_history, dtype=jnp.float64),
        jnp.asarray(v_electron_history, dtype=jnp.float64),
        geometry,
        boundary_conditions,
        parameters=parameters,
        output_path=str(time_traces_path),
        title="Shifted-torus 4-field blob time traces",
        phi_history=jnp.asarray(phi_history, dtype=jnp.float64),
    )
    print(f"rebuilding phi movie from {history_path}", flush=True)
    # _save_shifted_torus_free_decay_movie(...)
    _save_phi_movie(
        np.asarray(times, dtype=np.float64),
        phi_history,
        geometry,
        output_path=str(movie_path),
        frame_stride=2,
        title="Shifted-torus 4-field blob potential evolution",
        z_indices=_blob_z_indices(geometry, float(metadata.get("zeta0", zeta0))),
    )


if __name__ == "__main__":
    main()
