"""Seeded-filament dynamics on the rotating ellipse (B7, second half).

Seeds a localized density blob on the genuinely non-axisymmetric rotating-ellipse
geometry and evolves the four-field drift-reduced FCI model (density, vorticity,
ion/electron parallel velocity) as a short free run. The curvature drive spins up
vorticity from the pressure blob -- the interchange mechanism that moves a
filament -- and because the flux surfaces rotate with the toroidal angle, the
filament evolves differently in each toroidal plane.

The four-field blob driver (boundary conditions, phi inversion, RK4 stepping) is
the same one exercised by the validated harness in
``tests/test_shifted_torus_4_field_blob.py`` and gated on this geometry in
``tests/test_rotating_ellipse_filament.py``; only the geometry is swapped. The
figure is drawn in physical (R, Z) space so the rotating elliptical cross-section
is visible.

Run:

    PYTHONPATH=src python examples/stellarator/rotating_ellipse_filament_demo.py

writes ``output/rotating_ellipse_filament/`` with a PNG (density + vorticity in
the rotating cross-sections, initial vs evolved) and a JSON summary.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from jax_drb.geometry import build_rotating_ellipse_geometry, rotating_ellipse_position  # noqa: E402
from jax_drb.native.fci_4_field_rhs import Fci4FieldBlobParameters  # noqa: E402

# Reuse the validated four-field blob driver from the test harness.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from test_shifted_torus_4_field_blob import (  # noqa: E402
    _build_blob_initial_state,
    simulate_shifted_torus_4field_blob,
)
from test_shifted_torus_4_field_free_decay import _build_free_decay_boundary_conditions  # noqa: E402

R0 = 3.0
X_MIN, X_MAX = 0.2, 1.0
ELONGATION = 0.35
N_FIELD_PERIODS = 1
SHAPE = (20, 28, 12)
N_STEPS = 40
DT = 2.0e-3
OUTPUT_DIR = Path("output/rotating_ellipse_filament")


def _physical_plane(geometry, z_index):
    """(R, Z) coordinates of the cell centers in one toroidal plane."""
    x = np.asarray(geometry.grid.x.centers)
    theta = np.asarray(geometry.grid.y.centers)
    zeta = float(geometry.grid.z.centers[z_index])
    xx, tt = np.meshgrid(x, theta, indexing="ij")
    position = np.asarray(
        rotating_ellipse_position(
            jax.numpy.asarray(xx), jax.numpy.asarray(tt), jax.numpy.asarray(zeta),
            r0=R0, elongation=ELONGATION, n_field_periods=N_FIELD_PERIODS,
        )
    )
    major_radius = np.hypot(position[..., 0], position[..., 1])
    return major_radius, position[..., 2], zeta


def plot_filament(geometry, initial_state, final_state, z_indices, output_path):
    import matplotlib.pyplot as plt

    density0 = np.asarray(initial_state.density)
    density1 = np.asarray(final_state.density)
    omega1 = np.asarray(final_state.omega)
    d_vmax = float(np.max(np.abs(np.concatenate([density0 - 1.0, density1 - 1.0]).ravel())))
    w_vmax = float(np.max(np.abs(omega1))) or 1.0

    fig, axes = plt.subplots(3, len(z_indices), figsize=(4.2 * len(z_indices), 11.0), squeeze=False)
    for col, z_index in enumerate(z_indices):
        R, Z, zeta = _physical_plane(geometry, z_index)
        panels = [
            (density0[:, :, z_index] - 1.0, "RdBu_r", d_vmax, f"density seed, zeta={zeta:.2f}"),
            (density1[:, :, z_index] - 1.0, "RdBu_r", d_vmax, f"density evolved, zeta={zeta:.2f}"),
            (omega1[:, :, z_index], "PuOr", w_vmax, f"vorticity evolved, zeta={zeta:.2f}"),
        ]
        for row, (field, cmap, vmax, title) in enumerate(panels):
            ax = axes[row][col]
            mesh = ax.pcolormesh(R, Z, field, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("R")
            ax.set_ylabel("Z")
            fig.colorbar(mesh, ax=ax, shrink=0.85)

    fig.suptitle("Seeded filament on the rotating ellipse: vorticity generated in rotating cross-sections")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    geometry = build_rotating_ellipse_geometry(
        SHAPE, r0=R0, x_min=X_MIN, x_max=X_MAX, elongation=ELONGATION,
        n_field_periods=N_FIELD_PERIODS, iota=0.9, c_phi=3.0,
    )
    initial_state = _build_blob_initial_state(geometry)
    boundary_conditions = _build_free_decay_boundary_conditions(geometry, 0.0)
    parameters = Fci4FieldBlobParameters(rho_star=1.0, phi_inversion_tol=5.0e-5,
                                         phi_inversion_maxiter=100, phi_inversion_restart=200)

    final_state, *_history = simulate_shifted_torus_4field_blob(
        geometry, initial_state, boundary_conditions,
        parameters=parameters, final_time=N_STEPS * DT, timestep=DT,
    )

    density = np.asarray(final_state.density)
    omega = np.asarray(final_state.omega)
    summary = {
        "shape": list(SHAPE),
        "n_steps": N_STEPS,
        "dt": DT,
        "finite": bool(np.all(np.isfinite(density)) and np.all(np.isfinite(omega))),
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "omega_max_abs": float(np.max(np.abs(omega))),
        "v_ion_max_abs": float(np.max(np.abs(np.asarray(final_state.v_ion_parallel)))),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Bracket the blob's toroidal centre (zeta0 = pi) so each column shows the
    # filament in a differently-oriented cross-section.
    z_indices = [SHAPE[2] // 3, SHAPE[2] // 2, (2 * SHAPE[2]) // 3]
    plot_filament(geometry, initial_state, final_state, z_indices, OUTPUT_DIR / "rotating_ellipse_filament.png")
    print(f"wrote {OUTPUT_DIR / 'rotating_ellipse_filament.png'}")


if __name__ == "__main__":
    main()
