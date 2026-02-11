"""Regenerate DRB2D README assets (GIF + panel) reproducibly.

Writes:
  - docs/assets/images/drb2d_turbulence.gif
  - docs/assets/images/drb2d_turbulence_panel.png

Notes
-----
- The DRB2D movie is curvature-driven and uses the conservative advection core.
- We animate ``omega`` by default since it tends to show the most coherent 2D turbulence.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _maybe_optimize_gif_with_ffmpeg(path: Path) -> None:
    """Optionally optimize a GIF in-place using ffmpeg palettegen/paletteuse."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[make_drb2d_readme_assets] ffmpeg not found; skipping GIF optimization")
        return

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        palette = td_path / "palette.png"
        optimized = td_path / "optimized.gif"

        vf = "fps=12,scale=420:-1:flags=lanczos"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-vf",
                f"{vf},palettegen=max_colors=96:stats_mode=diff",
                str(palette),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-i",
                str(palette),
                "-lavfi",
                f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
                str(optimized),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        old_size = path.stat().st_size
        new_size = optimized.stat().st_size
        if new_size < old_size:
            shutil.copy2(optimized, path)
            print(
                f"[make_drb2d_readme_assets] optimized GIF: {old_size / 1e6:.2f} MB -> {new_size / 1e6:.2f} MB"
            )
        else:
            print("[make_drb2d_readme_assets] optimization not smaller; keeping original GIF")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "_out_make_drb2d_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo_root / "examples/08_nonlinear_drb2d/drb2d_movie.py"),
        "--out",
        str(out_dir),
        "--nx",
        "96",
        "--ny",
        "96",
        "--dt",
        "0.02",
        "--tmax",
        "20.0",
        "--save-stride",
        "10",
        "--solver",
        "tsit5",
        "--rtol",
        "1e-5",
        "--atol",
        "1e-8",
        "--field",
        "omega",
        "--seed",
        "0",
    ]
    print("[make_drb2d_readme_assets] running:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)

    src_gif = out_dir / "movie.gif"
    src_panel = out_dir / "panel.png"
    if not src_gif.exists() or not src_panel.exists():
        raise FileNotFoundError(f"Expected {src_gif} and {src_panel} to exist.")

    dst_dir = repo_root / "docs/assets/images"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_gif = dst_dir / "drb2d_turbulence.gif"
    dst_panel = dst_dir / "drb2d_turbulence_panel.png"

    shutil.copy2(src_gif, dst_gif)
    shutil.copy2(src_panel, dst_panel)
    _maybe_optimize_gif_with_ffmpeg(dst_gif)
    print(f"[make_drb2d_readme_assets] wrote {dst_gif}")
    print(f"[make_drb2d_readme_assets] wrote {dst_panel}")


if __name__ == "__main__":
    main()
