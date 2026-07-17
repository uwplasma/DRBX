"""Radial profile analysis of the saved nonlinear stellarator SOL history.

The script post-processes the compressed history NPZ written by
``nonlinear_turbulence.py`` (it does not rerun the simulation): it rebuilds the
matching synthetic stellarator geometry, bins the final/time-mean/RMS
fluctuation fields into radial profiles, forms curvature-weighted transport and
connection-length-weighted amplitude proxies, and traces the nonlinear energy
history. It prints the summary numbers and writes the four-panel profile
figure ``stellarator_nonlinear_turbulence_profiles.png`` next to the input NPZ
under ``docs/data/stellarator_fci_example_artifacts/nonlinear_turbulence``
(relative to the current working directory).

Requires the ``stellarator_nonlinear_turbulence.npz`` artifact: if it is
missing the script prints how to generate it and exits. The geometry
PARAMETERS below must match the ones used by ``nonlinear_turbulence.py``.

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/stellarator-fci/nonlinear_turbulence.py
    PYTHONPATH=src python examples/geometry-3D/stellarator-fci/turbulent_profile_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from drbx.geometry import build_synthetic_stellarator_geometry

# --- PARAMETERS ------------------------------------------------------------------
OUTPUT_ROOT = Path("docs/data/stellarator_fci_example_artifacts/nonlinear_turbulence")  # artifact root (cwd-relative)
CASE_LABEL = "stellarator_nonlinear_turbulence"
ARRAYS_PATH = OUTPUT_ROOT / f"{CASE_LABEL}.npz"
PROFILE_PATH = OUTPUT_ROOT / f"{CASE_LABEL}_profiles.png"

NX = 28
NY = 28
NZ = 56
FIELD_PERIODS = 5
ISLAND_MODE = 2
ISLAND_AMPLITUDE = 0.034
MIRROR_AMPLITUDE = 0.18
RADIAL_BINS = 24


def _radial_bin_average(values: np.ndarray, radial: np.ndarray, edges: np.ndarray) -> np.ndarray:
    profile = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            mask = (radial >= lower) & (radial <= upper)
        else:
            mask = (radial >= lower) & (radial < upper)
        if np.any(mask):
            profile.append(float(np.nanmean(values[mask])))
        else:
            profile.append(np.nan)
    return np.asarray(profile)


if not ARRAYS_PATH.exists():
    print(f"Missing required artifact: {ARRAYS_PATH}")
    print("This analysis post-processes the nonlinear SOL history; generate it first with")
    print("  PYTHONPATH=src python examples/geometry-3D/stellarator-fci/nonlinear_turbulence.py")
    raise SystemExit(1)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

with np.load(ARRAYS_PATH) as payload:
    history = np.asarray(payload["history"], dtype=np.float64)
    time = np.asarray(payload["time"], dtype=np.float64)
    curvature = np.asarray(payload["curvature"], dtype=np.float64)
    connection_length = np.asarray(payload["connection_length"], dtype=np.float64)

geometry = build_synthetic_stellarator_geometry(
    nx=NX,
    ny=NY,
    nz=NZ,
    field_periods=FIELD_PERIODS,
    island_mode=ISLAND_MODE,
    island_amplitude=ISLAND_AMPLITUDE,
    mirror_amplitude=MIRROR_AMPLITUDE,
)
radial = np.asarray(geometry.radial, dtype=np.float64)
edges = np.linspace(float(np.nanmin(radial)), float(np.nanmax(radial)), RADIAL_BINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])

final = history[-1]
time_mean = np.mean(history, axis=0)
time_rms = np.std(history, axis=0)
field_energy = np.mean(history * history, axis=(1, 2, 3))

dtheta = 2.0 * np.pi / history.shape[3]
potential_proxy = np.roll(history, 2, axis=3)
radial_velocity_proxy = -(
    np.roll(potential_proxy, -1, axis=3) - np.roll(potential_proxy, 1, axis=3)
) / (2.0 * dtheta)
radial_flux_proxy = np.mean(history * radial_velocity_proxy * curvature[None, ...], axis=0)

final_mean_profile = _radial_bin_average(final, radial, edges)
time_mean_profile = _radial_bin_average(time_mean, radial, edges)
rms_profile = _radial_bin_average(time_rms, radial, edges)
flux_profile = _radial_bin_average(radial_flux_proxy, radial, edges)
connection_weighted_profile = _radial_bin_average(
    np.abs(final) * connection_length / np.nanmean(connection_length),
    radial,
    edges,
)

fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), constrained_layout=True)

axes[0, 0].plot(centers, final_mean_profile, color="#0b3d91", lw=2.2, label="final mean")
axes[0, 0].plot(centers, time_mean_profile, color="#f28e2b", lw=1.8, ls="--", label="time mean")
axes[0, 0].set_xlabel("normalized radius")
axes[0, 0].set_ylabel("fluctuation proxy")
axes[0, 0].set_title("Radial fluctuation profile")
axes[0, 0].legend(frameon=False)
axes[0, 0].grid(alpha=0.25)

axes[0, 1].plot(centers, rms_profile, color="#2a9d8f", lw=2.2)
axes[0, 1].set_xlabel("normalized radius")
axes[0, 1].set_ylabel("time RMS")
axes[0, 1].set_title("Turbulence intensity")
axes[0, 1].grid(alpha=0.25)

axes[1, 0].plot(centers, flux_profile, color="#b23a48", lw=2.2, label="curvature-weighted flux")
axes[1, 0].plot(
    centers,
    connection_weighted_profile,
    color="#3d405b",
    lw=1.8,
    ls=":",
    label="connection-weighted amplitude",
)
axes[1, 0].set_xlabel("normalized radius")
axes[1, 0].set_ylabel("proxy value")
axes[1, 0].set_title("Transport and connection metrics")
axes[1, 0].legend(frameon=False)
axes[1, 0].grid(alpha=0.25)

axes[1, 1].plot(time, field_energy, color="#264653", lw=2.2)
axes[1, 1].set_xlabel("time")
axes[1, 1].set_ylabel(r"$\langle \tilde{n}^2 \rangle$")
axes[1, 1].set_title("Nonlinear energy trace")
axes[1, 1].grid(alpha=0.25)

fig.suptitle("Synthetic stellarator reduced SOL turbulence profiles", fontsize=14)
fig.savefig(PROFILE_PATH, dpi=220)
plt.close(fig)

print(f"wrote profile analysis: {PROFILE_PATH}")
print(f"final mean absolute fluctuation: {np.mean(np.abs(final)):.4e}")
print(f"peak radial RMS: {np.nanmax(rms_profile):.4e}")
print(f"integrated radial flux proxy: {np.trapezoid(flux_profile, centers):.4e}")
