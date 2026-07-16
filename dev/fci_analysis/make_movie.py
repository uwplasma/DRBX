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

from analyze_EB_density import _load_eb_blob_step_history, _resolve_step_dump_dir  # noqa: E402
from test_shifted_torus_EB_blob import (  # noqa: E402
    _eb_blob_artifact_stem,
    _build_eb_blob_geometry,
    _load_eb_blob_history,
    _save_eb_blob_movie,
    radial_b_fraction,
)


DEFAULT_FRAME_STRIDE = 2


def _resolve_output_dir(run_name: str, output_path: Path | None) -> Path:
    if output_path is None:
        return Path(f"{_eb_blob_artifact_stem(run_name)}_outputs")
    candidate = Path(output_path)
    if candidate.is_file():
        return candidate.parent
    if candidate.name.endswith("_histories.npz"):
        return candidate.parent
    return candidate


def _load_movie_history(
    *,
    run_name: str,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    artifact_stem = _eb_blob_artifact_stem(run_name)
    history_path = output_dir / f"{artifact_stem}_histories.npz"
    if history_path.exists():
        print(f"loading EB blob movie history from {history_path}", flush=True)
        times, density_history, phi_history, te_history, ti_history, vi_history, ve_history, vorticity_history, _ = _load_eb_blob_history(
            history_path
        )
        return (
            np.asarray(times, dtype=np.float64),
            np.asarray(density_history, dtype=np.float64),
            np.asarray(phi_history, dtype=np.float64),
            np.asarray(te_history, dtype=np.float64),
            np.asarray(ti_history, dtype=np.float64),
            np.asarray(vi_history, dtype=np.float64),
            np.asarray(ve_history, dtype=np.float64),
            np.asarray(vorticity_history, dtype=np.float64),
            str(history_path),
        )

    step_dump_dir = _resolve_step_dump_dir(run_name, output_dir)
    if not step_dump_dir.exists():
        raise FileNotFoundError(
            f"missing EB blob history file {history_path} and missing step dump directory {step_dump_dir}"
        )
    print(f"loading EB blob movie step dumps from {step_dump_dir}", flush=True)
    times, density_history, phi_history, te_history, ti_history, vi_history, ve_history, vorticity_history = _load_eb_blob_step_history(
        step_dump_dir
    )
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(density_history, dtype=np.float64),
        np.asarray(phi_history, dtype=np.float64),
        np.asarray(te_history, dtype=np.float64),
        np.asarray(ti_history, dtype=np.float64),
        np.asarray(vi_history, dtype=np.float64),
        np.asarray(ve_history, dtype=np.float64),
        np.asarray(vorticity_history, dtype=np.float64),
        str(step_dump_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the shifted-torus EB blob movie from saved outputs.")
    parser.add_argument(
        "--run-name",
        default="eb_blob",
        help="Run name used to locate <run_name>_outputs and related artifacts.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output directory containing the saved EB blob history or step dumps.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        help="Use every Nth saved timestep when rendering the movie.",
    )
    args = parser.parse_args()

    run_name = str(args.run_name)
    artifact_stem = _eb_blob_artifact_stem(run_name)
    output_dir = _resolve_output_dir(run_name, args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    movie_path = output_dir / f"{artifact_stem}.gif"

    print(f"rebuilding EB blob movie for {run_name}", flush=True)
    print(f"movie output path: {movie_path}", flush=True)

    (
        times,
        density_history,
        phi_history,
        te_history,
        ti_history,
        vi_history,
        ve_history,
        vorticity_history,
        source_path,
    ) = _load_movie_history(run_name=run_name, output_dir=output_dir)
    geometry = _build_eb_blob_geometry(tuple(int(dim) for dim in density_history.shape[1:]), radial_fraction=radial_b_fraction)
    print(
        f"rebuilding movie from {source_path} with {int(times.shape[0])} snapshots and "
        f"frame_stride={int(args.frame_stride)}",
        flush=True,
    )

    _save_eb_blob_movie(
        times,
        density_history,
        phi_history,
        te_history,
        ti_history,
        vi_history,
        ve_history,
        vorticity_history,
        geometry,
        output_path=str(movie_path),
        frame_stride=int(args.frame_stride),
        title="Shifted-torus EB blob state evolution",
    )
    print(f"saved EB blob movie to {movie_path}", flush=True)


if __name__ == "__main__":
    main()
