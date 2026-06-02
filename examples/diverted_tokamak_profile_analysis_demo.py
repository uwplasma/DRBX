from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from jax_drb.runtime.artifacts import ensure_docs_media

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "docs" / "data" / "diverted_tokamak_turbulence_artifacts"
ARRAYS_PATH = OUTPUT_ROOT / "data" / "diverted_tokamak_turbulence_arrays.npz"
PROFILE_PATH = OUTPUT_ROOT / "images" / "diverted_tokamak_turbulence_profiles.png"


def _cell_corner_coordinates(center: np.ndarray) -> np.ndarray:
    padded = np.empty((center.shape[0] + 2, center.shape[1] + 2), dtype=np.float64)
    padded[1:-1, 1:-1] = center
    padded[0, 1:-1] = 2.0 * center[0, :] - center[1, :]
    padded[-1, 1:-1] = 2.0 * center[-1, :] - center[-2, :]
    padded[1:-1, 0] = 2.0 * center[:, 0] - center[:, 1]
    padded[1:-1, -1] = 2.0 * center[:, -1] - center[:, -2]
    padded[0, 0] = 2.0 * padded[0, 1] - padded[0, 2]
    padded[0, -1] = 2.0 * padded[0, -2] - padded[0, -3]
    padded[-1, 0] = 2.0 * padded[-1, 1] - padded[-1, 2]
    padded[-1, -1] = 2.0 * padded[-1, -2] - padded[-1, -3]
    return 0.25 * (
        padded[:-1, :-1]
        + padded[1:, :-1]
        + padded[:-1, 1:]
        + padded[1:, 1:]
    )


if not ARRAYS_PATH.exists():
    ensure_docs_media(root=REPO_ROOT)
if not ARRAYS_PATH.exists():
    raise FileNotFoundError(
        f"Missing {ARRAYS_PATH}. Run "
        "`python scripts/fetch_example_artifacts.py --skip-baselines` or "
        "`PYTHONPATH=src python examples/diverted_tokamak_movie_demo.py` first."
    )

PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

with np.load(ARRAYS_PATH) as payload:
    field_name = str(payload["field_name"].item())
    time = np.asarray(payload["time_points"], dtype=np.float64)
    field_history = np.asarray(payload["field_history_2d"], dtype=np.float64)
    rxy = np.asarray(payload["rxy"], dtype=np.float64)
    zxy = np.asarray(payload["zxy"], dtype=np.float64)
    psixy = np.asarray(payload["psixy"], dtype=np.float64)

final = field_history[-1]
radial_index = np.arange(final.shape[0])
final_poloidal_mean = np.mean(final, axis=1)
radial_rms = np.sqrt(np.mean(field_history * field_history, axis=(0, 2)))
global_rms = np.sqrt(np.mean(field_history * field_history, axis=(1, 2)))
lower_target = final[:, 0]
upper_target = final[:, -1]
lower_target_rms = np.sqrt(np.mean(field_history[:, :, 0] ** 2, axis=1))
upper_target_rms = np.sqrt(np.mean(field_history[:, :, -1] ** 2, axis=1))

fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)

axes[0, 0].plot(radial_index, final_poloidal_mean, color="#0b3d91", lw=2.2, label="final mean")
axes[0, 0].plot(radial_index, radial_rms, color="#f28e2b", lw=1.8, ls="--", label="history RMS")
axes[0, 0].set_xlabel("radial grid index")
axes[0, 0].set_ylabel(field_name)
axes[0, 0].set_title("Radial profile")
axes[0, 0].legend(frameon=False)
axes[0, 0].grid(alpha=0.25)

axes[0, 1].plot(radial_index, lower_target, color="#b23a48", lw=2.2, label="lower target")
axes[0, 1].plot(radial_index, upper_target, color="#2a9d8f", lw=2.2, label="upper target")
axes[0, 1].set_xlabel("radial grid index")
axes[0, 1].set_ylabel(field_name)
axes[0, 1].set_title("Target lineouts")
axes[0, 1].legend(frameon=False)
axes[0, 1].grid(alpha=0.25)

axes[1, 0].plot(time, global_rms, color="#264653", lw=2.2, label="domain RMS")
axes[1, 0].plot(time, lower_target_rms, color="#b23a48", lw=1.8, ls="--", label="lower target RMS")
axes[1, 0].plot(time, upper_target_rms, color="#2a9d8f", lw=1.8, ls=":", label="upper target RMS")
axes[1, 0].set_xlabel("time")
axes[1, 0].set_ylabel(f"RMS {field_name}")
axes[1, 0].set_title("Time traces")
axes[1, 0].legend(frameon=False)
axes[1, 0].grid(alpha=0.25)

r_corners = _cell_corner_coordinates(rxy)
z_corners = _cell_corner_coordinates(zxy)
mesh = axes[1, 1].pcolormesh(r_corners, z_corners, final, shading="flat", cmap="viridis")
try:
    axes[1, 1].contour(rxy, zxy, psixy, levels=[0.0], colors="white", linewidths=1.6)
except ValueError:
    pass
axes[1, 1].set_aspect("equal", adjustable="box")
axes[1, 1].set_xlabel("R")
axes[1, 1].set_ylabel("Z")
axes[1, 1].set_title("Final diverted-domain field")
fig.colorbar(mesh, ax=axes[1, 1], label=field_name)

fig.suptitle("Diverted tokamak turbulence profile analysis", fontsize=14)
fig.savefig(PROFILE_PATH, dpi=220)
plt.close(fig)

print(f"wrote profile analysis: {PROFILE_PATH}")
print(f"field: {field_name}")
print(f"domain RMS final/initial: {global_rms[-1]:.4e} / {global_rms[0]:.4e}")
print(f"lower target peak: {np.max(np.abs(lower_target)):.4e}")
print(f"upper target peak: {np.max(np.abs(upper_target)):.4e}")
