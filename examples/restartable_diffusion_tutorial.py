from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from jax_drb.cli import main as cli_main
from jax_drb.native.deck_runner import build_portable_array_payload, load_portable_array_payload, write_portable_array_payload
from jax_drb.runtime import load_restart_bundle


@dataclass(frozen=True)
class RestartableDiffusionSettings:
    case_name: str
    output_root: Path
    nx: int
    ny: int
    nz: int
    timestep: float
    first_nout: int
    resume_nout: int
    precision: str
    cli_precision_override: str | None
    diffusion_coefficient: float
    dx_expression: str
    dy_expression: str
    dz_expression: str
    density_function: str
    pressure_function: str
    fps: int
    make_movie: bool
    quiet: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Tutorial-style restartable diffusion example. It writes a TOML input file, "
            "runs jax_drb, saves summary/arrays/restart/log outputs, resumes from the restart bundle, "
            "and renders Matplotlib QA plots from the saved .npz results."
        )
    )
    parser.add_argument("--case-name", default="restartable_diffusion")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "docs" / "data" / "restartable_diffusion_demo_artifacts",
    )
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--nz", type=int, default=1)
    parser.add_argument("--timestep", type=float, default=5.0)
    parser.add_argument("--first-nout", type=int, default=3)
    parser.add_argument("--resume-nout", type=int, default=2)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--cli-precision-override",
        choices=("float32", "float64"),
        default=None,
        help="Override the input-file precision with the explicit jax_drb --precision flag.",
    )
    parser.add_argument("--diffusion-coefficient", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--skip-movie", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> RestartableDiffusionSettings:
    return RestartableDiffusionSettings(
        case_name=args.case_name,
        output_root=args.output_root,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        timestep=args.timestep,
        first_nout=args.first_nout,
        resume_nout=args.resume_nout,
        precision=args.precision,
        cli_precision_override=args.cli_precision_override,
        diffusion_coefficient=args.diffusion_coefficient,
        dx_expression="0.0075 + 0.005*x",
        dy_expression="0.01",
        dz_expression="0.01",
        density_function="1 + H(x - 0.25) * H(0.75-x) * exp(-(y-π)^2)",
        pressure_function="Nh:function",
        fps=args.fps,
        make_movie=not args.skip_movie,
        quiet=args.quiet,
    )


def print_section(settings: RestartableDiffusionSettings, title: str) -> None:
    if settings.quiet:
        return
    print(f"\n{title}")
    print("-" * len(title))


def print_mapping(settings: RestartableDiffusionSettings, mapping: Mapping[str, Any]) -> None:
    if settings.quiet:
        return
    for key, value in mapping.items():
        print(f"  {key}: {value}")


def build_input_text(settings: RestartableDiffusionSettings) -> str:
    return f"""
[time]
nout = {settings.first_nout}
timestep = {settings.timestep:g}

[runtime]
precision = "{settings.precision}"

[mesh]
nx = {settings.nx}
ny = {settings.ny}
nz = {settings.nz}

dx = {{ expr = "{settings.dx_expression}" }}
dy = {settings.dy_expression}
dz = {settings.dz_expression}

J = 1

[solver]
mxstep = 1000

[model]
components = ["h"]

[species.h]
type = ["evolve_density", "evolve_pressure", "anomalous_diffusion"]
AA = 1
charge = 1
anomalous_D = {settings.diffusion_coefficient:g}
thermal_conduction = false

[fields.Nh]
function = {{ expr = "{settings.density_function}" }}
bndry_all = "neumann"

[fields.Ph]
function = {{ ref = "{settings.pressure_function}" }}
bndry_all = "neumann"
""".strip() + "\n"


def write_input_file(settings: RestartableDiffusionSettings) -> Path:
    input_dir = settings.output_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / "input.toml"
    input_path.write_text(build_input_text(settings), encoding="utf-8")
    return input_path


def build_run_argv(
    settings: RestartableDiffusionSettings,
    *,
    input_path: Path,
    case_name: str,
    output_dir: Path,
    restart_in: Path | None = None,
    resume_steps: int | None = None,
    quiet: bool | None = None,
) -> list[str]:
    argv = [str(input_path), "--case-name", case_name, "--output-dir", str(output_dir)]
    if settings.cli_precision_override is not None:
        argv.extend(["--precision", settings.cli_precision_override])
    if restart_in is not None:
        argv.extend(["--restart-in", str(restart_in)])
    if resume_steps is not None:
        argv.extend(["--resume-steps", str(resume_steps)])
    if quiet if quiet is not None else settings.quiet:
        argv.append("--quiet")
    return argv


def run_segment(
    settings: RestartableDiffusionSettings,
    *,
    input_path: Path,
    case_name: str,
    output_dir: Path,
    restart_in: Path | None = None,
    resume_steps: int | None = None,
    quiet: bool | None = None,
) -> None:
    argv = build_run_argv(
        settings,
        input_path=input_path,
        case_name=case_name,
        output_dir=output_dir,
        restart_in=restart_in,
        resume_steps=resume_steps,
        quiet=quiet,
    )
    print_mapping(settings, {"cli_argv": "jax_drb " + " ".join(argv)})
    exit_code = cli_main(argv)
    if exit_code != 0:
        raise RuntimeError(f"jax_drb run failed for {case_name} with exit code {exit_code}")


def output_paths(case_name: str, output_dir: Path) -> dict[str, Path]:
    return {
        "summary": output_dir / f"{case_name}_summary.json",
        "arrays": output_dir / f"{case_name}_arrays.npz",
        "restart": output_dir / f"{case_name}_restart.npz",
        "log": output_dir / f"{case_name}_run_log.json",
    }


def stitch_histories(first_payload: Mapping[str, Any], resumed_payload: Mapping[str, Any]) -> dict[str, Any]:
    first_times = np.asarray(first_payload["time_points"], dtype=np.float64)
    resumed_times = np.asarray(resumed_payload["time_points"], dtype=np.float64)
    stitched_times = np.concatenate([first_times, resumed_times[1:]], axis=0)
    stitched_variables: dict[str, np.ndarray] = {}
    for name, first_values in first_payload["variables"].items():
        resumed_values = resumed_payload["variables"][name]
        stitched_variables[name] = np.concatenate(
            [np.asarray(first_values, dtype=np.float64), np.asarray(resumed_values, dtype=np.float64)[1:]],
            axis=0,
        )
    stitched = dict(first_payload)
    stitched["time_points"] = stitched_times.tolist()
    stitched["variables"] = stitched_variables
    stitched["effective_output_points"] = int(stitched_times.size)
    return stitched


def save_combined_payload(settings: RestartableDiffusionSettings, payload: Mapping[str, Any]) -> Path:
    data_dir = settings.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    combined_path = data_dir / f"{settings.case_name}_combined_history.npz"
    array_payload = build_portable_array_payload(
        case_name=str(payload["case_name"]),
        parity_mode=str(payload["parity_mode"]),
        capability_tier=str(payload.get("capability_tier", "native_exact")),
        compare_variables=tuple(sorted(payload["variables"])),
        component_labels=tuple(payload.get("component_labels", [])),
        dimensions=payload.get("dimensions", {}),
        time_points=tuple(float(value) for value in payload["time_points"]),
        dataset_scalars=payload.get("dataset_scalars", {}),
        variables=payload["variables"],
        overrides=tuple(payload.get("overrides", [])),
        configured_nout=payload.get("configured_nout"),
        configured_timestep=payload.get("configured_timestep"),
        producer=str(payload.get("producer", "jax-drb")),
    )
    write_portable_array_payload(array_payload, combined_path)
    return combined_path


def write_analysis_json(
    settings: RestartableDiffusionSettings,
    *,
    first_restart_path: Path,
    stitched_payload: Mapping[str, Any],
    uninterrupted_payload: Mapping[str, Any],
) -> Path:
    data_dir = settings.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    restart_bundle = load_restart_bundle(first_restart_path)
    max_abs_density_diff = float(
        np.max(
            np.abs(
                np.asarray(stitched_payload["variables"]["Nh"], dtype=np.float64)
                - np.asarray(uninterrupted_payload["variables"]["Nh"], dtype=np.float64)
            )
        )
    )
    max_abs_pressure_diff = float(
        np.max(
            np.abs(
                np.asarray(stitched_payload["variables"]["Ph"], dtype=np.float64)
                - np.asarray(uninterrupted_payload["variables"]["Ph"], dtype=np.float64)
            )
        )
    )
    payload = {
        "case_name": settings.case_name,
        "configured_precision": settings.precision,
        "cli_precision_override": settings.cli_precision_override,
        "restart_case_name": restart_bundle.case_name,
        "first_segment_completed_steps": restart_bundle.completed_steps,
        "restart_current_time": restart_bundle.current_time,
        "stitched_time_points": stitched_payload["time_points"],
        "uninterrupted_time_points": uninterrupted_payload["time_points"],
        "max_abs_density_diff_vs_uninterrupted": max_abs_density_diff,
        "max_abs_pressure_diff_vs_uninterrupted": max_abs_pressure_diff,
    }
    analysis_path = data_dir / f"{settings.case_name}_analysis.json"
    analysis_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return analysis_path


def _extract_2d(array: np.ndarray, time_index: int) -> np.ndarray:
    field = np.asarray(array, dtype=np.float64)[time_index]
    if field.ndim == 3 and field.shape[-1] == 1:
        return field[:, :, 0]
    if field.ndim == 2:
        return field
    raise ValueError(f"expected a 2D field or singleton-z field, got shape {field.shape}")


def plot_density_snapshots(
    settings: RestartableDiffusionSettings,
    *,
    first_payload: Mapping[str, Any],
    resumed_payload: Mapping[str, Any],
    uninterrupted_payload: Mapping[str, Any],
) -> Path:
    images_dir = settings.output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    entries = (
        ("initial density", _extract_2d(np.asarray(first_payload["variables"]["Nh"]), 0)),
        ("first segment final", _extract_2d(np.asarray(first_payload["variables"]["Nh"]), -1)),
        ("resumed final", _extract_2d(np.asarray(resumed_payload["variables"]["Nh"]), -1)),
        ("uninterrupted final", _extract_2d(np.asarray(uninterrupted_payload["variables"]["Nh"]), -1)),
    )
    for axis, (title, field) in zip(axes.flat, entries, strict=True):
        image = axis.imshow(field.T, origin="lower", aspect="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, shrink=0.85)
    path = images_dir / f"{settings.case_name}_density_snapshots.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_restart_consistency(
    settings: RestartableDiffusionSettings,
    *,
    stitched_payload: Mapping[str, Any],
    uninterrupted_payload: Mapping[str, Any],
) -> Path:
    images_dir = settings.output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    stitched_density = np.asarray(stitched_payload["variables"]["Nh"], dtype=np.float64)
    full_density = np.asarray(uninterrupted_payload["variables"]["Nh"], dtype=np.float64)
    stitched_pressure = np.asarray(stitched_payload["variables"]["Ph"], dtype=np.float64)
    full_pressure = np.asarray(uninterrupted_payload["variables"]["Ph"], dtype=np.float64)
    time_points = np.asarray(stitched_payload["time_points"], dtype=np.float64)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(time_points, stitched_density.max(axis=(1, 2, 3)), marker="o", label="restarted peak Nh")
    axes[0].plot(time_points, full_density.max(axis=(1, 2, 3)), marker="s", linestyle="--", label="full peak Nh")
    axes[0].plot(time_points, stitched_pressure.max(axis=(1, 2, 3)), marker="^", label="restarted peak Ph")
    axes[0].plot(time_points, full_pressure.max(axis=(1, 2, 3)), marker="v", linestyle="--", label="full peak Ph")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("peak field value")
    axes[0].set_title("Restarted vs uninterrupted histories")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    density_error = np.max(np.abs(stitched_density - full_density), axis=(1, 2, 3))
    pressure_error = np.max(np.abs(stitched_pressure - full_pressure), axis=(1, 2, 3))
    axes[1].semilogy(time_points, density_error + 1e-30, marker="o", label="max |Nh restarted-full|")
    axes[1].semilogy(time_points, pressure_error + 1e-30, marker="s", label="max |Ph restarted-full|")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("max abs difference")
    axes[1].set_title("Restart consistency error")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    path = images_dir / f"{settings.case_name}_restart_consistency.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_density_surface(
    settings: RestartableDiffusionSettings,
    *,
    stitched_payload: Mapping[str, Any],
) -> Path:
    images_dir = settings.output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    density = _extract_2d(np.asarray(stitched_payload["variables"]["Nh"]), -1)
    x = np.arange(density.shape[0], dtype=np.float64)
    y = np.arange(density.shape[1], dtype=np.float64)
    grid_y, grid_x = np.meshgrid(y, x)

    figure = plt.figure(figsize=(8, 5.5), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(grid_y, grid_x, density, cmap="viridis", linewidth=0.0, antialiased=True)
    axis.set_xlabel("y")
    axis.set_ylabel("x")
    axis.set_zlabel("Nh")
    axis.set_title("Final restarted density surface")

    path = images_dir / f"{settings.case_name}_density_surface.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def render_density_movie(
    settings: RestartableDiffusionSettings,
    *,
    stitched_payload: Mapping[str, Any],
) -> Path | None:
    if not settings.make_movie:
        return None

    movies_dir = settings.output_root / "movies"
    movies_dir.mkdir(parents=True, exist_ok=True)
    density = np.asarray(stitched_payload["variables"]["Nh"], dtype=np.float64)
    time_points = np.asarray(stitched_payload["time_points"], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    image = axis.imshow(_extract_2d(density, 0).T, origin="lower", aspect="auto", cmap="viridis")
    title = axis.set_title(f"Nh, t = {time_points[0]:.2f}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.colorbar(image, ax=axis, shrink=0.85)

    def _update(frame_index: int):
        image.set_data(_extract_2d(density, frame_index).T)
        title.set_text(f"Nh, t = {time_points[frame_index]:.2f}")
        return image, title

    movie = animation.FuncAnimation(figure, _update, frames=len(time_points), interval=300, blit=False)
    path = movies_dir / f"{settings.case_name}_density.gif"
    try:
        movie.save(path, writer=animation.PillowWriter(fps=settings.fps))
    except Exception:
        plt.close(figure)
        return None
    plt.close(figure)
    return path


def print_artifacts(settings: RestartableDiffusionSettings, artifacts: Mapping[str, Path | None]) -> None:
    print_section(settings, "Generated Artifacts")
    print_mapping(settings, artifacts)


def main() -> int:
    settings = build_settings(parse_args())
    print_section(settings, "Requested Restartable Diffusion Demo")
    print_mapping(
        settings,
        {
            "case_name": settings.case_name,
            "output_root": settings.output_root,
            "mesh": f"{settings.nx} x {settings.ny} x {settings.nz}",
            "timestep": settings.timestep,
            "precision": settings.precision,
            "first_nout": settings.first_nout,
            "resume_nout": settings.resume_nout,
            "anomalous_D": settings.diffusion_coefficient,
            "fps": settings.fps,
            "make_movie": settings.make_movie,
        },
    )

    input_path = write_input_file(settings)
    print_section(settings, "Input Deck")
    if not settings.quiet:
        print(input_path.read_text(encoding="utf-8"))

    first_output_dir = settings.output_root / "run_first"
    resumed_output_dir = settings.output_root / "run_resumed"
    uninterrupted_output_dir = settings.output_root / "run_full"

    print_section(settings, "Run 1: fresh simulation")
    run_segment(
        settings,
        input_path=input_path,
        case_name=settings.case_name,
        output_dir=first_output_dir,
    )

    first_paths = output_paths(settings.case_name, first_output_dir)
    print_section(settings, "Run 2: continue from restart bundle")
    run_segment(
        settings,
        input_path=input_path,
        case_name=f"{settings.case_name}_resumed",
        output_dir=resumed_output_dir,
        restart_in=first_paths["restart"],
        resume_steps=settings.resume_nout,
    )

    print_section(settings, "Run 3: uninterrupted reference for restart QA")
    run_segment(
        settings,
        input_path=input_path,
        case_name=f"{settings.case_name}_full",
        output_dir=uninterrupted_output_dir,
        resume_steps=settings.first_nout + settings.resume_nout,
        quiet=True,
    )

    first_payload = load_portable_array_payload(first_paths["arrays"])
    resumed_paths = output_paths(f"{settings.case_name}_resumed", resumed_output_dir)
    resumed_payload = load_portable_array_payload(resumed_paths["arrays"])
    uninterrupted_paths = output_paths(f"{settings.case_name}_full", uninterrupted_output_dir)
    uninterrupted_payload = load_portable_array_payload(uninterrupted_paths["arrays"])

    stitched_payload = stitch_histories(first_payload, resumed_payload)
    stitched_payload["case_name"] = f"{settings.case_name}_stitched"

    combined_history_path = save_combined_payload(settings, stitched_payload)
    analysis_path = write_analysis_json(
        settings,
        first_restart_path=first_paths["restart"],
        stitched_payload=stitched_payload,
        uninterrupted_payload=uninterrupted_payload,
    )
    snapshots_path = plot_density_snapshots(
        settings,
        first_payload=first_payload,
        resumed_payload=resumed_payload,
        uninterrupted_payload=uninterrupted_payload,
    )
    consistency_path = plot_restart_consistency(
        settings,
        stitched_payload=stitched_payload,
        uninterrupted_payload=uninterrupted_payload,
    )
    surface_path = plot_density_surface(
        settings,
        stitched_payload=stitched_payload,
    )
    movie_path = render_density_movie(
        settings,
        stitched_payload=stitched_payload,
    )

    print_artifacts(
        settings,
        {
            "input_file": input_path,
            "first_summary": first_paths["summary"],
            "first_arrays": first_paths["arrays"],
            "first_restart": first_paths["restart"],
            "first_log": first_paths["log"],
            "resumed_summary": resumed_paths["summary"],
            "resumed_arrays": resumed_paths["arrays"],
            "resumed_restart": resumed_paths["restart"],
            "resumed_log": resumed_paths["log"],
            "full_arrays": uninterrupted_paths["arrays"],
            "combined_history": combined_history_path,
            "analysis_json": analysis_path,
            "density_snapshots": snapshots_path,
            "restart_consistency": consistency_path,
            "density_surface": surface_path,
            "density_movie": movie_path,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
