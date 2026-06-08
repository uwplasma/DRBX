from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPO_ROOT / "examples"


def _example_env() -> dict[str, str]:
    environment = dict(os.environ)
    pythonpath = str(REPO_ROOT / "src")
    if environment.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = pythonpath
    environment.setdefault("MPLBACKEND", "Agg")
    return environment


def _run_example(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 90,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=_example_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    assert completed.returncode == 0, (
        f"command failed: {' '.join(command)}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return completed


@pytest.mark.parametrize(
    "relative_script",
    [
        "restartable_diffusion_tutorial.py",
        "diffusion_precision_benchmark.py",
        "autodiff_diffusion_sensitivity_demo.py",
        "autodiff_diffusion_inverse_design_demo.py",
        "strong_scaling_diffusion_demo.py",
        "model_selection_guide.py",
    ],
)
def test_docs_argparse_examples_expose_subprocess_help(relative_script: str) -> None:
    script_path = EXAMPLES_ROOT / relative_script

    completed = _run_example(
        [sys.executable, str(script_path), "--help"],
        cwd=REPO_ROOT,
        timeout=30,
    )

    assert "usage:" in completed.stdout.lower()


def test_restartable_diffusion_tutorial_lightweight_subprocess_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "restartable"

    _run_example(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "restartable_diffusion_tutorial.py"),
            "--output-root",
            str(output_root),
            "--nx",
            "6",
            "--ny",
            "8",
            "--first-nout",
            "1",
            "--resume-nout",
            "1",
            "--timestep",
            "1.0",
            "--precision",
            "float32",
            "--skip-movie",
            "--quiet",
        ],
        cwd=REPO_ROOT,
        timeout=120,
    )

    assert (output_root / "input" / "input.toml").exists()
    assert (output_root / "run_first" / "restartable_diffusion_arrays.npz").exists()
    assert (output_root / "run_resumed" / "restartable_diffusion_resumed_arrays.npz").exists()
    assert (output_root / "run_full" / "restartable_diffusion_full_arrays.npz").exists()
    assert (output_root / "data" / "restartable_diffusion_analysis.json").exists()
    assert (output_root / "images" / "restartable_diffusion_density_snapshots.png").stat().st_size > 0
    assert (output_root / "images" / "restartable_diffusion_restart_consistency.png").stat().st_size > 0


def test_model_selection_guide_writes_parse_checked_starter_decks(tmp_path: Path) -> None:
    output_root = tmp_path / "model_selection"

    _run_example(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "model_selection_guide.py"),
            "--output-root",
            str(output_root),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        timeout=30,
    )

    summary_path = output_root / "model_selection_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert (output_root / "diffusion_start.toml").exists()
    assert (output_root / "open_field_concept.toml").exists()
    assert (output_root / "model_selection_guide.md").exists()
    assert "diffusion / scalar reduced transport" in {entry["name"] for entry in payload["model_families"]}
    assert payload["generated_decks"][0]["components"] == [
        "h:evolve_density",
        "h:evolve_pressure",
        "h:anomalous_diffusion",
    ]


def test_diverted_tokamak_release_array_examples_are_subprocess_self_contained(
    tmp_path: Path,
) -> None:
    mini_repo = tmp_path / "mini_repo"
    _copy_example_script(
        "diverted_tokamak_movie_demo.py",
        destination_root=mini_repo,
    )
    _copy_example_script(
        "diverted_tokamak_profile_analysis_demo.py",
        destination_root=mini_repo,
    )
    release_arrays_path = (
        mini_repo
        / "docs"
        / "data"
        / "diverted_tokamak_turbulence_artifacts"
        / "data"
        / "diverted_tokamak_turbulence_arrays.npz"
    )
    _write_tiny_diverted_tokamak_release_arrays(release_arrays_path)

    movie = _run_example(
        [sys.executable, str(mini_repo / "examples" / "diverted_tokamak_movie_demo.py")],
        cwd=mini_repo,
        timeout=120,
    )
    profile = _run_example(
        [sys.executable, str(mini_repo / "examples" / "diverted_tokamak_profile_analysis_demo.py")],
        cwd=mini_repo,
        timeout=60,
    )

    output_root = mini_repo / "docs" / "data" / "diverted_tokamak_turbulence_artifacts"
    assert "Creating diverted-tokamak figures and GIF from release arrays" in movie.stdout
    assert "Running curated benchmark case" not in movie.stdout
    assert "wrote profile analysis:" in profile.stdout
    assert (output_root / "data" / "diverted_tokamak_turbulence_analysis.json").exists()
    assert (output_root / "images" / "diverted_tokamak_turbulence_snapshots.png").stat().st_size > 0
    assert (output_root / "images" / "diverted_tokamak_turbulence_poster.png").stat().st_size > 0
    assert (output_root / "images" / "diverted_tokamak_turbulence_profiles.png").stat().st_size > 0
    assert (output_root / "movies" / "diverted_tokamak_turbulence.gif").stat().st_size > 0


def test_stellarator_fci_docs_analysis_commands_are_subprocess_self_contained(
    tmp_path: Path,
) -> None:
    _run_example(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "geometry-3D" / "stellarator-fci" / "geometry_plotting_demo.py"),
        ],
        cwd=tmp_path,
        timeout=60,
    )

    geometry_root = tmp_path / "docs" / "data" / "stellarator_fci_example_artifacts" / "geometry"
    assert (geometry_root / "stellarator_geometry_plotting_demo.npz").exists()
    assert (geometry_root / "stellarator_geometry_plotting_demo.png").stat().st_size > 0

    nonlinear_root = (
        tmp_path
        / "docs"
        / "data"
        / "stellarator_fci_example_artifacts"
        / "nonlinear_turbulence"
    )
    _write_tiny_stellarator_turbulence_release_arrays(
        nonlinear_root / "stellarator_nonlinear_turbulence_demo.npz"
    )

    completed = _run_example(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "geometry-3D" / "stellarator-fci" / "turbulent_profile_analysis_demo.py"),
        ],
        cwd=tmp_path,
        timeout=60,
    )

    assert "wrote profile analysis:" in completed.stdout
    assert (nonlinear_root / "stellarator_nonlinear_turbulence_demo_profiles.png").stat().st_size > 0


def test_vmec_extender_imported_field_demo_is_subprocess_self_contained(
    tmp_path: Path,
) -> None:
    completed = _run_example(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "geometry-3D" / "vmec-extender" / "imported_field_demo.py"),
        ],
        cwd=tmp_path,
        timeout=90,
    )

    artifact_root = tmp_path / "docs" / "data" / "vmec_extender_edge_field_artifacts"
    assert "edge summary:" in completed.stdout
    assert "sol summary:" in completed.stdout
    assert (artifact_root / "data" / "vmec_extender_edge_field_campaign.json").exists()
    assert (artifact_root / "data" / "vmec_extender_edge_field_campaign.npz").exists()
    assert (artifact_root / "images" / "vmec_extender_edge_field_campaign.png").stat().st_size > 0
    assert (artifact_root / "data" / "vmec_extender_sol_smoke.json").exists()
    assert (artifact_root / "data" / "vmec_extender_sol_smoke.npz").exists()
    assert (artifact_root / "images" / "vmec_extender_sol_smoke.png").stat().st_size > 0


def _copy_example_script(relative_script: str, *, destination_root: Path) -> Path:
    source = EXAMPLES_ROOT / relative_script
    destination = destination_root / "examples" / relative_script
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _write_tiny_diverted_tokamak_release_arrays(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-1.0, 1.0, 4, dtype=np.float64)[:, None]
    y = np.linspace(-0.7, 0.7, 5, dtype=np.float64)[None, :]
    rxy = 1.5 + 0.18 * x + 0.04 * y
    zxy = y + 0.05 * x
    psixy = x + 0.15 * y
    time_points = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    base = np.sin(np.pi * x) * np.cos(np.pi * y)
    field_history = np.stack(
        [
            np.zeros_like(base),
            0.1 * base,
            0.2 * base + 0.02 * x,
        ],
        axis=0,
    )
    np.savez_compressed(
        path,
        field_name=np.asarray("phi"),
        time_points=time_points,
        field_history_2d=field_history,
        rxy=rxy,
        zxy=zxy,
        psixy=psixy,
    )


def _write_tiny_stellarator_turbulence_release_arrays(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    radial = np.linspace(0.0, 1.0, 28, dtype=np.float64)[:, None, None]
    poloidal = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False, dtype=np.float64)[None, :, None]
    toroidal = np.linspace(0.0, 2.0 * np.pi, 56, endpoint=False, dtype=np.float64)[None, None, :]
    mode = np.exp(-((radial - 0.62) / 0.24) ** 2) * np.cos(3.0 * poloidal - 5.0 * toroidal)
    history = np.stack(
        [
            0.05 * mode,
            0.07 * mode + 0.01 * np.sin(toroidal),
            0.09 * mode + 0.02 * radial,
        ],
        axis=0,
    )
    curvature = 0.2 * np.cos(poloidal) + 0.1 * np.sin(toroidal)
    connection_length = 1.0 + radial + 0.05 * np.cos(5.0 * toroidal)
    np.savez_compressed(
        path,
        history=history.astype(np.float16),
        time=np.asarray([0.0, 0.08, 0.16], dtype=np.float32),
        curvature=curvature.astype(np.float32),
        connection_length=connection_length.astype(np.float32),
    )
