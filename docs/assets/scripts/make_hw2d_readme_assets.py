"""Regenerate HW2D README assets (GIF + panel) reproducibly.

Writes:
  - docs/assets/images/hw2d_turbulence.gif
  - docs/assets/images/hw2d_turbulence_panel.png
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _maybe_optimize_gif_with_ffmpeg(path: Path) -> None:
    """Optionally optimize a GIF in-place using ffmpeg palettegen/paletteuse."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[make_hw2d_readme_assets] ffmpeg not found; skipping GIF optimization")
        return

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        palette = td_path / "palette.png"
        optimized = td_path / "optimized.gif"

        # A compact, good-contrast palette with modest color count to keep the repo small.
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
                f"[make_hw2d_readme_assets] optimized GIF: {old_size / 1e6:.2f} MB -> {new_size / 1e6:.2f} MB"
            )
        else:
            print("[make_hw2d_readme_assets] optimization not smaller; keeping original GIF")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "_out_make_hw2d_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(**os.environ)
    env["PYTHONPATH"] = str(repo_root / "src") + (
        (":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else ""
    )

    cmd = [
        sys.executable,
        str(repo_root / "examples/08_nonlinear_hw2d/hw2d_movie.py"),
        "--out",
        str(out_dir),
        # Long-time but light movie: keep frames modest for repo size.
        "--nx",
        "48",
        "--ny",
        "48",
        "--dt",
        "0.01",
        "--tmax",
        "80.0",
        "--save-stride",
        "100",
        "--seed",
        "0",
    ]
    print("[make_hw2d_readme_assets] running:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True, env=env)

    src_gif = out_dir / "movie.gif"
    src_panel = out_dir / "panel.png"
    if not src_gif.exists() or not src_panel.exists():
        raise FileNotFoundError(f"Expected {src_gif} and {src_panel} to exist.")

    dst_dir = repo_root / "docs/assets/images"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_gif = dst_dir / "hw2d_turbulence.gif"
    dst_panel = dst_dir / "hw2d_turbulence_panel.png"

    shutil.copy2(src_gif, dst_gif)
    shutil.copy2(src_panel, dst_panel)
    _maybe_optimize_gif_with_ffmpeg(dst_gif)
    print(f"[make_hw2d_readme_assets] wrote {dst_gif}")
    print(f"[make_hw2d_readme_assets] wrote {dst_panel}")


if __name__ == "__main__":
    main()
