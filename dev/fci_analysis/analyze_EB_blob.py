from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from test_shifted_torus_EB_blob import (  # noqa: E402
    _build_eb_blob_geometry,
    _eb_blob_artifact_stem,
    _eb_blob_z_indices,
    _save_eb_blob_time_traces,
    z0,
)


TIME_CUT = 0.08


def _signed_norm(values: np.ndarray):
    import matplotlib.colors as colors

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("cannot build a movie from non-finite history values")
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    if vmin < 0.0 < vmax:
        bound = max(abs(vmin), abs(vmax))
        return colors.TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _sequential_norm(values: np.ndarray):
    import matplotlib.colors as colors

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("cannot build a movie from non-finite history values")
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _robust_signed_norm(values: np.ndarray, *, qlo: float = 2.0, qhi: float = 98.0):
    import matplotlib.colors as colors

    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot build a movie from non-finite history values")
    lo = float(np.percentile(finite, qlo))
    hi = float(np.percentile(finite, qhi))
    if np.isclose(lo, hi):
        spread = max(abs(lo), 1.0)
        lo, hi = -spread, spread
    if lo < 0.0 < hi:
        bound = max(abs(lo), abs(hi))
        return colors.TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
    return colors.Normalize(vmin=lo, vmax=hi)


def _signed_symlog_norm(values: np.ndarray):
    import matplotlib.colors as colors

    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot build a movie from non-finite history values")
    vmax = float(np.nanpercentile(np.abs(finite), 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(np.abs(finite)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    linthresh = max(float(np.nanpercentile(np.abs(finite), 1.0)), vmax * 1.0e-4, 1.0e-12)
    return colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)


def _save_eb_blob_recent_state_slices(
    times: np.ndarray,
    density_history: np.ndarray,
    phi_history: np.ndarray,
    te_history: np.ndarray,
    ti_history: np.ndarray,
    vi_history: np.ndarray,
    ve_history: np.ndarray,
    vorticity_history: np.ndarray,
    geometry,
    *,
    output_path: str,
    z_indices: tuple[int, int, int, int] | None = None,
    num_recent_timesteps: int = 5,
    title: str = "Shifted-torus EB blob recent state slices",
) -> None:
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    y_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    if z_indices is None:
        z_indices = tuple(int(idx) for idx in np.linspace(0, int(z_values.shape[0] - 1), 4))

    y_grid, radius_grid = np.meshgrid(y_values, x_values)

    density_fluctuation = np.asarray(density_history, dtype=np.float64) - 1.0
    te_fluctuation = np.asarray(te_history, dtype=np.float64) - 1.0
    ti_fluctuation = np.asarray(ti_history, dtype=np.float64) - 1.0
    field_specs = (
        ("density fluctuation", density_fluctuation, "coolwarm", _robust_signed_norm(density_fluctuation)),
        ("phi", np.asarray(phi_history, dtype=np.float64), "coolwarm", _robust_signed_norm(np.asarray(phi_history, dtype=np.float64))),
        ("Te fluctuation", te_fluctuation, "coolwarm", _robust_signed_norm(te_fluctuation)),
        ("Ti fluctuation", ti_fluctuation, "coolwarm", _robust_signed_norm(ti_fluctuation)),
        ("Vi", np.asarray(vi_history, dtype=np.float64), "coolwarm", _robust_signed_norm(np.asarray(vi_history, dtype=np.float64))),
        ("Ve", np.asarray(ve_history, dtype=np.float64), "coolwarm", _robust_signed_norm(np.asarray(ve_history, dtype=np.float64))),
        ("vorticity", np.asarray(vorticity_history, dtype=np.float64), "coolwarm", _signed_symlog_norm(np.asarray(vorticity_history, dtype=np.float64))),
    )

    num_snapshots = int(times.shape[0])
    if num_snapshots == 0:
        raise ValueError("cannot build a figure from an empty history")
    tail_count = min(5, num_snapshots)
    tail_start = max(0, num_snapshots - tail_count)
    early_pool_end = tail_start
    early_count = min(5, early_pool_end)
    if early_count > 0 and early_pool_end > 0:
        early_indices = np.linspace(0, early_pool_end - 1, num=early_count, dtype=np.int64)
    else:
        early_indices = np.asarray([], dtype=np.int64)
    tail_indices = np.arange(tail_start, num_snapshots, dtype=np.int64)
    recent_indices = np.unique(np.concatenate([early_indices, tail_indices]))

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(output_path).stem

    for snapshot_index in recent_indices:
        fig, axes = plt.subplots(
            nrows=len(field_specs),
            ncols=4,
            figsize=(17.0, 24.0),
            subplot_kw={"projection": "polar"},
            constrained_layout=True,
        )
        if len(field_specs) == 1:
            axes = np.asarray([axes])

        images = []
        for row, (field_name, field_data, cmap, norm) in enumerate(field_specs):
            for col, z_index in enumerate(z_indices):
                ax = axes[row, col]
                ax.set_theta_zero_location("E")
                ax.set_theta_direction(-1)
                ax.set_ylim(0.0, float(x_values[-1]))
                ax.set_yticklabels([])
                ax.set_title(f"z={z_values[z_index]:.3f}")
                image = ax.pcolormesh(
                    y_grid,
                    radius_grid,
                    field_data[snapshot_index, :, :, z_index],
                    shading="auto",
                    cmap=cmap,
                    norm=norm,
                )
                images.append(image)
            fig.colorbar(
                images[row * len(z_indices)],
                ax=list(axes[row, :]),
                location="right",
                pad=0.02,
                shrink=0.88,
            )
            axes[row, 0].set_ylabel(field_name, rotation=90, labelpad=28, fontsize=11)

        fig.suptitle(f"{title}, t={times[snapshot_index]:.3e}")
        fig.savefig(output_dir / f"{stem}_t{int(snapshot_index):05d}.png", dpi=160)
        plt.close(fig)


def _resolve_step_dump_dir(run_name: str) -> Path:
    return _THIS_DIR / run_name


def _load_eb_blob_step_history(step_dump_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step_files = sorted(step_dump_dir.glob("step_*.npz"))
    if not step_files:
        raise FileNotFoundError(f"no step_*.npz files found in {step_dump_dir}")

    snapshots: list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for fallback_index, step_file in enumerate(step_files):
        with np.load(step_file, allow_pickle=False) as step:
            step_index = int(step["step_index"]) if "step_index" in step.files else fallback_index
            time_value = float(step["time"]) if "time" in step.files else float(fallback_index)
            snapshots.append(
                (
                    step_index,
                    time_value,
                    np.asarray(step["density"], dtype=np.float64),
                    np.asarray(step["phi"], dtype=np.float64),
                    np.asarray(step["Te"], dtype=np.float64),
                    np.asarray(step["Ti"], dtype=np.float64),
                    np.asarray(step["Vi"], dtype=np.float64),
                    np.asarray(step["Ve"], dtype=np.float64),
                    np.asarray(step["vorticity"], dtype=np.float64),
                )
            )

    snapshots.sort(key=lambda item: item[0])
    times = np.asarray([item[1] for item in snapshots], dtype=np.float64)
    density_history = np.asarray([item[2] for item in snapshots], dtype=np.float64)
    phi_history = np.asarray([item[3] for item in snapshots], dtype=np.float64)
    te_history = np.asarray([item[4] for item in snapshots], dtype=np.float64)
    ti_history = np.asarray([item[5] for item in snapshots], dtype=np.float64)
    vi_history = np.asarray([item[6] for item in snapshots], dtype=np.float64)
    ve_history = np.asarray([item[7] for item in snapshots], dtype=np.float64)
    vorticity_history = np.asarray([item[8] for item in snapshots], dtype=np.float64)
    return times, density_history, phi_history, te_history, ti_history, vi_history, ve_history, vorticity_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved EB blob step dumps and rebuild time traces.")
    parser.add_argument(
        "--run_name",
        default="EB_perp_diffusion",
        help="Name of the step-dump directory under tests/, e.g. EB_perp_diffusion.",
    )
    args = parser.parse_args()

    step_dump_dir = _resolve_step_dump_dir(args.run_name)
    if not step_dump_dir.exists():
        raise FileNotFoundError(f"missing EB blob step dump directory: {step_dump_dir}")

    (
        times,
        density_history,
        phi_history,
        te_history,
        ti_history,
        vi_history,
        ve_history,
        vorticity_history,
    ) = _load_eb_blob_step_history(step_dump_dir)

    resolution = int(density_history.shape[1])
    geometry = _build_eb_blob_geometry((resolution, resolution, resolution), construct_fci_maps=False)
    z_indices = _eb_blob_z_indices(geometry, z0)

    artifact_stem = _eb_blob_artifact_stem(args.run_name)
    output_path = step_dump_dir / f"{artifact_stem}_time_traces.png"
    output_path_cut = step_dump_dir / f"{artifact_stem}_time_traces_tle_{float(TIME_CUT):.1f}.png"

    print(f"rebuilding EB blob time traces from {step_dump_dir}", flush=True)
    _save_eb_blob_time_traces(
        np.asarray(times, dtype=np.float64),
        np.asarray(density_history, dtype=np.float64),
        np.asarray(phi_history, dtype=np.float64),
        np.asarray(te_history, dtype=np.float64),
        np.asarray(ti_history, dtype=np.float64),
        np.asarray(vi_history, dtype=np.float64),
        np.asarray(ve_history, dtype=np.float64),
        np.asarray(vorticity_history, dtype=np.float64),
        geometry,
        output_path=str(output_path),
        title="Shifted-torus EB blob time traces",
    )
    print(f"saved time traces to {output_path}", flush=True)
    times_np = np.asarray(times, dtype=np.float64)
    cut_mask = times_np <= float(TIME_CUT)
    if not np.any(cut_mask):
        raise ValueError(f"no time samples at or below TIME_CUT={float(TIME_CUT):.3f}")
    _save_eb_blob_time_traces(
        times_np[cut_mask],
        np.asarray(density_history, dtype=np.float64)[cut_mask],
        np.asarray(phi_history, dtype=np.float64)[cut_mask],
        np.asarray(te_history, dtype=np.float64)[cut_mask],
        np.asarray(ti_history, dtype=np.float64)[cut_mask],
        np.asarray(vi_history, dtype=np.float64)[cut_mask],
        np.asarray(ve_history, dtype=np.float64)[cut_mask],
        np.asarray(vorticity_history, dtype=np.float64)[cut_mask],
        geometry,
        output_path=str(output_path_cut),
        title=f"Shifted-torus EB blob time traces, t <= {float(TIME_CUT):.1f}",
    )
    print(f"saved cut time traces to {output_path_cut}", flush=True)

    slices_output_path = step_dump_dir / f"{artifact_stem}_recent_state_slices.png"
    print(f"rebuilding recent state slice figure from {step_dump_dir}", flush=True)
    _save_eb_blob_recent_state_slices(
        np.asarray(times, dtype=np.float64),
        np.asarray(density_history, dtype=np.float64),
        np.asarray(phi_history, dtype=np.float64),
        np.asarray(te_history, dtype=np.float64),
        np.asarray(ti_history, dtype=np.float64),
        np.asarray(vi_history, dtype=np.float64),
        np.asarray(ve_history, dtype=np.float64),
        np.asarray(vorticity_history, dtype=np.float64),
        geometry,
        output_path=str(slices_output_path),
        z_indices=z_indices,
        num_recent_timesteps=10,
        title="Shifted-torus EB blob selected 10 timesteps, 4 toroidal slices",
    )
    print(f"saved recent state slices to {slices_output_path.parent} as per-timestep PNGs", flush=True)


if __name__ == "__main__":
    main()
