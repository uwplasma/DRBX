"""Regenerate DRB2D README assets reproducibly.

Writes:
  - docs/assets/images/drb2d_turbulence.gif
  - docs/assets/images/drb2d_turbulence_panel.png
  - docs/assets/images/drb2d_hot_ion_turbulence.gif
  - docs/assets/images/drb2d_hot_ion_turbulence_panel.png

Notes
-----
- These movies are chosen to be stable and fast on CPU-only laptops.
- Both cases use periodic BCs and the spectral Poisson solve to avoid CG/FD
  variability in README assets.
- For non-Boussinesq and curvature-proxy validation, see the dedicated tests
  and figures in `docs/nonlinear/` and `examples/08_nonlinear_drb2d/`.
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
    env = dict(**os.environ)
    env["PYTHONPATH"] = str(repo_root / "src") + (
        (":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else ""
    )

    cmd_cold = [
        sys.executable,
        str(repo_root / "examples/08_nonlinear_drb2d/drb2d_movie.py"),
        "--out",
        str(out_dir / "cold_ion"),
        "--nx",
        "64",
        "--ny",
        "64",
        "--dt",
        "0.02",
        "--tmax",
        "30.0",
        "--save-stride",
        "12",
        "--solver",
        "dopri5",
        "--fixed-step",
        "--seed",
        "0",
    ]
    print("[make_drb2d_readme_assets] running:", " ".join(cmd_cold))
    subprocess.run(cmd_cold, cwd=repo_root, check=True, env=env)

    cold_out = out_dir / "cold_ion"
    src_gif = cold_out / "movie.gif"
    src_panel = cold_out / "panel.png"
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

    cmd_hot = [
        sys.executable,
        str(repo_root / "examples/08_nonlinear_drb2d/drb2d_hot_ion_movie.py"),
        "--out",
        str(out_dir / "hot_ion"),
        "--nx",
        "64",
        "--ny",
        "64",
        "--dt",
        "0.015",
        "--tmax",
        "22.0",
        "--save-stride",
        "16",
        "--solver",
        "dopri5",
        "--fixed-step",
        "--seed",
        "0",
    ]
    print("[make_drb2d_readme_assets] running:", " ".join(cmd_hot))
    subprocess.run(cmd_hot, cwd=repo_root, check=True, env=env)

    hot_out = out_dir / "hot_ion"
    src_hot_gif = hot_out / "movie.gif"
    src_hot_panel = hot_out / "panel.png"
    if not src_hot_gif.exists() or not src_hot_panel.exists():
        raise FileNotFoundError(f"Expected {src_hot_gif} and {src_hot_panel} to exist.")
    dst_hot_gif = dst_dir / "drb2d_hot_ion_turbulence.gif"
    dst_hot_panel = dst_dir / "drb2d_hot_ion_turbulence_panel.png"
    shutil.copy2(src_hot_gif, dst_hot_gif)
    shutil.copy2(src_hot_panel, dst_hot_panel)
    _maybe_optimize_gif_with_ffmpeg(dst_hot_gif)
    print(f"[make_drb2d_readme_assets] wrote {dst_hot_gif}")
    print(f"[make_drb2d_readme_assets] wrote {dst_hot_panel}")


if __name__ == "__main__":
    main()
