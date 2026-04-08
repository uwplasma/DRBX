from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_restartable_diffusion_tutorial_writes_restart_and_plots(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root / "src")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "restartable_diffusion_tutorial.py"),
            "--output-root",
            str(tmp_path / "demo"),
            "--quiet",
        ],
        check=True,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    output_root = tmp_path / "demo"
    assert (output_root / "input" / "BOUT.inp").exists()
    assert (output_root / "run_first" / "restartable_diffusion_summary.json").exists()
    assert (output_root / "run_first" / "restartable_diffusion_arrays.npz").exists()
    assert (output_root / "run_first" / "restartable_diffusion_restart.npz").exists()
    assert (output_root / "run_first" / "restartable_diffusion_run_log.json").exists()
    assert (output_root / "run_resumed" / "restartable_diffusion_resumed_arrays.npz").exists()
    assert (output_root / "run_full" / "restartable_diffusion_full_arrays.npz").exists()
    assert (output_root / "data" / "restartable_diffusion_combined_history.npz").exists()
    assert (output_root / "images" / "restartable_diffusion_density_snapshots.png").stat().st_size > 0
    assert (output_root / "images" / "restartable_diffusion_restart_consistency.png").stat().st_size > 0
    assert (output_root / "images" / "restartable_diffusion_density_surface.png").stat().st_size > 0
    assert (output_root / "movies" / "restartable_diffusion_density.gif").stat().st_size > 0
    assert (output_root / "images" / "restartable_diffusion_density_surface.png").stat().st_size > 0
    assert (output_root / "movies" / "restartable_diffusion_density.gif").stat().st_size > 0

    analysis = json.loads((output_root / "data" / "restartable_diffusion_analysis.json").read_text(encoding="utf-8"))
    assert analysis["restart_current_time"] == 15.0
    assert analysis["first_segment_completed_steps"] == 3
    assert analysis["max_abs_density_diff_vs_uninterrupted"] < 1.0e-8
    assert analysis["max_abs_pressure_diff_vs_uninterrupted"] < 1.0e-8
