from __future__ import annotations

import argparse
from typing import Sequence
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from analyze_EB_density import _density_rhs_terms  # noqa: E402
from analyze_rhs_common import (  # noqa: E402
    FieldAnalyzerSpec,
    build_term_specs,
    eb_blob_artifact_stem,
    load_eb_blob_step_history,
    run_field_analyzer,
)
from test_shifted_torus_EB_blob import _build_eb_blob_geometry  # noqa: E402


DENSITY_TERM_SPECS = build_term_specs(
    (
        ("ExB bracket", "ExB bracket"),
        ("parallel Ve compression", "parallel Ve compression"),
        ("n * parallel Ve compression", "n * parallel Ve compression"),
        ("Ve * parallel density gradient", "Ve * parallel density gradient"),
        ("curvature", "curvature"),
        ("parallel diffusion", "parallel diffusion"),
        ("perp diffusion", "perp diffusion"),
    )
)

FIELD_SPEC = FieldAnalyzerSpec(
    display_name="density",
    output_suffix="density",
    term_specs=DENSITY_TERM_SPECS,
    sum_keys=(
        "ExB bracket",
        "parallel particle flux divergence",
        "curvature",
        "parallel diffusion",
        "perp diffusion",
    ),
    evaluate_terms=_density_rhs_terms,
    legend_ncol=4,
)


def _density_profile_history(density_history: np.ndarray, geometry) -> np.ndarray:
    density = jnp.asarray(density_history, dtype=jnp.float64)
    if density.ndim != 4:
        raise ValueError(f"density_history must have shape (time, nx, ny, nz), got {density.shape}")

    jacobian = jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64)
    if jacobian.shape != density.shape[1:]:
        raise ValueError(
            f"geometry jacobian shape {jacobian.shape} does not match density cell shape {density.shape[1:]}"
        )

    radial_weight = jnp.sum(jacobian, axis=(1, 2))
    if np.any(np.asarray(radial_weight == 0.0, dtype=bool)):
        raise ValueError("cannot build density profile with a zero radial jacobian weight")

    profile = jnp.sum(density * jacobian[None, :, :, :], axis=(2, 3)) / radial_weight[None, :]
    return np.asarray(profile, dtype=np.float64)


def _save_density_profile_movie(
    times: np.ndarray,
    density_history: np.ndarray,
    geometry,
    *,
    output_path: Path,
    title: str,
    frame_stride: int = 1,
    fps: int = 10,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    times_np = np.asarray(times, dtype=np.float64)
    profiles = _density_profile_history(density_history, geometry)
    if times_np.shape[0] != profiles.shape[0]:
        raise ValueError(f"times and profiles disagree: {times_np.shape[0]} times for {profiles.shape[0]} frames")

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    if x_values.shape[0] != profiles.shape[1]:
        raise ValueError(f"x grid has {x_values.shape[0]} cells but profiles have {profiles.shape[1]} radial values")

    finite_profiles = profiles[np.isfinite(profiles)]
    if finite_profiles.size == 0:
        raise ValueError("cannot build density profile movie from non-finite profile values")
    y_min = float(np.min(finite_profiles))
    y_max = float(np.max(finite_profiles))
    y_span = max(y_max - y_min, 1.0e-12)
    y_pad = 0.08 * y_span

    frame_stride = max(1, int(frame_stride))
    fps = max(1, int(fps))
    frame_indices = np.arange(0, int(times_np.shape[0]), frame_stride, dtype=np.int64)
    if frame_indices[-1] != int(times_np.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times_np.shape[0]) - 1)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    line, = ax.plot(x_values, profiles[int(frame_indices[0])], color="tab:blue", linewidth=2.4)
    ax.set_xlabel("radial coordinate x")
    ax.set_ylabel("J-weighted density profile")
    ax.set_xlim(float(np.min(x_values)), float(np.max(x_values)))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(True, alpha=0.28)
    title_artist = ax.set_title("")

    def update(movie_frame_index: int):
        actual_index = int(frame_indices[movie_frame_index])
        line.set_ydata(profiles[actual_index])
        title_artist.set_text(f"{title}\nstep={actual_index}, t={float(times_np[actual_index]):.6e}")
        return line, title_artist

    animator = animation.FuncAnimation(
        fig,
        update,
        frames=int(frame_indices.shape[0]),
        interval=max(1, int(round(1000.0 / float(fps)))),
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    animator.save(output_path, writer=writer)
    plt.close(fig)


def _density_profile_movie_path(rhs_plot_path: Path) -> Path:
    suffix = f"_{FIELD_SPEC.output_suffix}_rhs_terms.png"
    filename = rhs_plot_path.name
    artifact_stem = filename[: -len(suffix)] if filename.endswith(suffix) else eb_blob_artifact_stem(rhs_plot_path.stem)
    return rhs_plot_path.parent / f"{artifact_stem}_density_profile.gif"


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Frame stride to use when rendering the density profile GIF.",
    )
    density_args, remaining_argv = parser.parse_known_args(argv)

    rhs_plot_path = run_field_analyzer(FIELD_SPEC, remaining_argv)
    (
        times,
        density_history,
        _phi_history,
        _te_history,
        _ti_history,
        _vi_history,
        _ve_history,
        _vorticity_history,
    ) = load_eb_blob_step_history(rhs_plot_path.parent)
    density_shape = tuple(int(size) for size in np.asarray(density_history).shape[1:])
    geometry = _build_eb_blob_geometry(density_shape, construct_fci_maps=False)
    movie_path = _density_profile_movie_path(rhs_plot_path)
    _save_density_profile_movie(
        np.asarray(times, dtype=np.float64),
        np.asarray(density_history, dtype=np.float64),
        geometry,
        output_path=movie_path,
        title="EB blob J-weighted density profile",
        frame_stride=int(density_args.frame_stride),
    )
    print(f"saved density profile movie to {movie_path}", flush=True)
    return rhs_plot_path


_save_density_profile_gif = _save_density_profile_movie


if __name__ == "__main__":
    main()
