from __future__ import annotations

import shutil
from pathlib import Path

import analyze_parallel_operators


def _copy_step_subset(source_dir: Path, target_dir: Path, count: int = 3) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    step_files = sorted(source_dir.glob("step_*.npz"))
    if len(step_files) < count:
        raise RuntimeError(f"need at least {count} step dumps in {source_dir}")
    for source_file in step_files[:count]:
        shutil.copy2(source_file, target_dir / source_file.name)
    return target_dir


def test_parallel_operator_movie_smoke_writes_gif(tmp_path: Path) -> None:
    source_step_dir = Path(__file__).resolve().parent / "EB_long" / "step_dumps"
    step_dump_dir = _copy_step_subset(source_step_dir, tmp_path / "parallel_operator_movie")

    movie_path = analyze_parallel_operators.main(
        [
            "--run-name",
            "EB_perp_diffusion",
            "--output-path",
            str(step_dump_dir),
            "--perp-diffusion",
            "1.0e-5",
            "--frame-stride",
            "1",
            "--movie-fps",
            "5",
        ]
    )

    assert movie_path.exists()
    assert movie_path.stat().st_size > 0
    assert movie_path.name == "EB_perp_diffusion_parallel_operators.gif"
    assert movie_path.parent == step_dump_dir
