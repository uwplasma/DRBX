"""Regenerate FCI DRB3D README assets (two GIFs) reproducibly.

Writes:
  - docs/assets/images/fci_drb3d_periodic_turbulence.gif
  - docs/assets/images/fci_drb3d_sheath_turbulence.gif

Notes
-----
These are intentionally *small-grid* movies intended for the README/docs gallery and for
quick regression checks. They are not meant to represent production-resolution turbulence.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _maybe_optimize_gif_with_ffmpeg(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[make_fci_drb3d_readme_assets] ffmpeg not found; skipping GIF optimization")
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
                f"[make_fci_drb3d_readme_assets] optimized GIF: {old_size / 1e6:.2f} MB -> {new_size / 1e6:.2f} MB"
            )
        else:
            print("[make_fci_drb3d_readme_assets] optimization not smaller; keeping original GIF")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "_out_make_fci_drb3d_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(**os.environ)
    env["PYTHONPATH"] = str(repo_root / "src") + (
        (":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else ""
    )

    dst_dir = repo_root / "docs/assets/images"
    dst_dir.mkdir(parents=True, exist_ok=True)

    periodic_out = out_dir / "periodic"
    sheath_out = out_dir / "sheath"

    cmd_periodic = [
        sys.executable,
        str(repo_root / "examples/09_fci/fci_drb3d_full_movie_periodic.py"),
        "--out",
        str(periodic_out),
        "--nx",
        "24",
        "--ny",
        "24",
        "--nz",
        "10",
        "--dt",
        "0.01",
        "--tmax",
        "2.4",
        "--save-stride",
        "12",
        "--solver",
        "dopri5",
        "--seed",
        "0",
    ]
    print("[make_fci_drb3d_readme_assets] running:", " ".join(cmd_periodic))
    subprocess.run(cmd_periodic, cwd=repo_root, check=True, env=env)
    src = periodic_out / "movie.gif"
    if not src.exists():
        raise FileNotFoundError(f"Expected {src} to exist.")
    dst = dst_dir / "fci_drb3d_periodic_turbulence.gif"
    shutil.copy2(src, dst)
    _maybe_optimize_gif_with_ffmpeg(dst)
    print(f"[make_fci_drb3d_readme_assets] wrote {dst}")

    cmd_sheath = [
        sys.executable,
        str(repo_root / "examples/09_fci/fci_drb3d_full_movie_sheath.py"),
        "--out",
        str(sheath_out),
        "--nx",
        "22",
        "--ny",
        "22",
        "--nz",
        "12",
        "--dt",
        "0.006",
        "--tmax",
        "1.2",
        "--save-stride",
        "10",
        "--solver",
        "dopri5",
        "--seed",
        "0",
    ]
    print("[make_fci_drb3d_readme_assets] running:", " ".join(cmd_sheath))
    subprocess.run(cmd_sheath, cwd=repo_root, check=True, env=env)
    src = sheath_out / "movie.gif"
    if not src.exists():
        raise FileNotFoundError(f"Expected {src} to exist.")
    dst = dst_dir / "fci_drb3d_sheath_turbulence.gif"
    shutil.copy2(src, dst)
    _maybe_optimize_gif_with_ffmpeg(dst)
    print(f"[make_fci_drb3d_readme_assets] wrote {dst}")


if __name__ == "__main__":
    main()
