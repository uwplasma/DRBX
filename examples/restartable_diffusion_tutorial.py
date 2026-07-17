"""Tutorial: run, restart, and QA a native diffusion case end to end.

The script writes a TOML input deck from the constants below, then drives the
public ``drbx`` CLI entry point three times:

1. a fresh run of ``FIRST_NOUT`` output steps (summary/arrays/restart/log files);
2. a resumed run that continues ``RESUME_NOUT`` more steps from the saved
   restart bundle; and
3. an uninterrupted reference run of ``FIRST_NOUT + RESUME_NOUT`` steps.

It then stitches the first and resumed histories, verifies them against the
uninterrupted reference (max abs density/pressure differences), and renders
Matplotlib QA artifacts from the saved ``.npz`` results: density snapshots, a
restart-consistency plot, a 3D density surface, and (optionally) a GIF movie.

All artifacts land under ``OUTPUT_ROOT``
(``docs/data/restartable_diffusion_demo_artifacts`` relative to the current
working directory): the input deck under ``input/``, per-run outputs under
``run_first/``, ``run_resumed/``, and ``run_full/``, the combined history and
analysis JSON under ``data/``, images under ``images/``, and the movie under
``movies/``. Every generated path is printed.

Run from the repository root:

    PYTHONPATH=src python examples/restartable_diffusion_tutorial.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from drbx.cli import main as cli_main
from drbx.native.deck_runner import build_portable_array_payload, load_portable_array_payload, write_portable_array_payload
from drbx.runtime import load_restart_bundle

# --- PARAMETERS ------------------------------------------------------------------
CASE_NAME = "restartable_diffusion"  # prefix of every generated artifact
OUTPUT_ROOT = Path("docs/data/restartable_diffusion_demo_artifacts")  # artifact root (cwd-relative)
NX = 16                    # radial grid points
NY = 24                    # poloidal grid points
NZ = 1                     # toroidal planes (1 = thin 2D case)
TIMESTEP = 5.0             # time between output points
FIRST_NOUT = 3             # output steps in the first (interrupted) segment
RESUME_NOUT = 2            # output steps continued from the restart bundle
PRECISION = "float64"      # [runtime] precision written into the deck
CLI_PRECISION_OVERRIDE: str | None = None  # e.g. "float32" to exercise the --precision CLI flag
DIFFUSION_COEFFICIENT = 2.0  # anomalous_D of the hydrogen species
DX_EXPRESSION = "0.0075 + 0.005*x"  # spatially varying radial spacing
DY_EXPRESSION = "0.01"     # poloidal spacing
DZ_EXPRESSION = "0.01"     # toroidal spacing
DENSITY_FUNCTION = "1 + H(x - 0.25) * H(0.75-x) * exp(-(y-π)^2)"  # initial Nh
PRESSURE_FUNCTION = "Nh:function"  # initial Ph references the density function
MAKE_MOVIE = True          # set False to skip the GIF (fastest QA loop)
MOVIE_FPS = 4              # GIF frame rate
QUIET_RUNS = False         # set True to silence the per-run CLI progress output


# --- helpers ----------------------------------------------------------------------
def build_input_text() -> str:
    """Render the TOML deck for the diffusion case from the constants above."""

    return f"""
[time]
nout = {FIRST_NOUT}
timestep = {TIMESTEP:g}

[runtime]
precision = "{PRECISION}"

[mesh]
nx = {NX}
ny = {NY}
nz = {NZ}

dx = {{ expr = "{DX_EXPRESSION}" }}
dy = {DY_EXPRESSION}
dz = {DZ_EXPRESSION}

J = 1

[solver]
mxstep = 1000

[model]
components = ["h"]

[species.h]
type = ["evolve_density", "evolve_pressure", "anomalous_diffusion"]
AA = 1
charge = 1
anomalous_D = {DIFFUSION_COEFFICIENT:g}
thermal_conduction = false

[fields.Nh]
function = {{ expr = "{DENSITY_FUNCTION}" }}
bndry_all = "neumann"

[fields.Ph]
function = {{ ref = "{PRESSURE_FUNCTION}" }}
bndry_all = "neumann"
""".strip() + "\n"


def run_segment(
    *,
    input_path: Path,
    case_name: str,
    output_dir: Path,
    restart_in: Path | None = None,
    resume_steps: int | None = None,
    quiet: bool | None = None,
) -> None:
    """Invoke the public drbx CLI entry point for one run segment."""

    argv = [str(input_path), "--case-name", case_name, "--output-dir", str(output_dir)]
    if CLI_PRECISION_OVERRIDE is not None:
        argv.extend(["--precision", CLI_PRECISION_OVERRIDE])
    if restart_in is not None:
        argv.extend(["--restart-in", str(restart_in)])
    if resume_steps is not None:
        argv.extend(["--resume-steps", str(resume_steps)])
    if quiet if quiet is not None else QUIET_RUNS:
        argv.append("--quiet")
    print(f"  cli_argv: drbx {' '.join(argv)}")
    exit_code = cli_main(argv)
    if exit_code != 0:
        raise RuntimeError(f"drbx run failed for {case_name} with exit code {exit_code}")


def output_paths(case_name: str, output_dir: Path) -> dict[str, Path]:
    return {
        "summary": output_dir / f"{case_name}_summary.json",
        "arrays": output_dir / f"{case_name}_arrays.npz",
        "restart": output_dir / f"{case_name}_restart.npz",
        "log": output_dir / f"{case_name}_run_log.json",
    }


def stitch_histories(first_payload: Mapping[str, Any], resumed_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Concatenate the first and resumed histories, dropping the repeated point."""

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


def _extract_2d(array: np.ndarray, time_index: int) -> np.ndarray:
    field = np.asarray(array, dtype=np.float64)[time_index]
    if field.ndim == 3 and field.shape[-1] == 1:
        return field[:, :, 0]
    if field.ndim == 2:
        return field
    raise ValueError(f"expected a 2D field or singleton-z field, got shape {field.shape}")


# --- write the input deck ---------------------------------------------------------
print("Requested restartable diffusion demo")
print(f"  case_name: {CASE_NAME}")
print(f"  output_root: {OUTPUT_ROOT}")
print(f"  mesh: {NX} x {NY} x {NZ}, timestep {TIMESTEP}, precision {PRECISION}")
print(f"  first_nout: {FIRST_NOUT}, resume_nout: {RESUME_NOUT}, anomalous_D: {DIFFUSION_COEFFICIENT}")

input_dir = OUTPUT_ROOT / "input"
input_dir.mkdir(parents=True, exist_ok=True)
input_path = input_dir / "input.toml"
input_path.write_text(build_input_text(), encoding="utf-8")
print(f"\nInput deck written to {input_path}:")
print(input_path.read_text(encoding="utf-8"))

first_output_dir = OUTPUT_ROOT / "run_first"
resumed_output_dir = OUTPUT_ROOT / "run_resumed"
uninterrupted_output_dir = OUTPUT_ROOT / "run_full"

# --- run 1: fresh simulation ------------------------------------------------------
print("Run 1: fresh simulation")
run_segment(input_path=input_path, case_name=CASE_NAME, output_dir=first_output_dir)
first_paths = output_paths(CASE_NAME, first_output_dir)

# --- run 2: continue from the restart bundle --------------------------------------
print("Run 2: continue from restart bundle")
run_segment(
    input_path=input_path,
    case_name=f"{CASE_NAME}_resumed",
    output_dir=resumed_output_dir,
    restart_in=first_paths["restart"],
    resume_steps=RESUME_NOUT,
)
resumed_paths = output_paths(f"{CASE_NAME}_resumed", resumed_output_dir)

# --- run 3: uninterrupted reference for restart QA --------------------------------
print("Run 3: uninterrupted reference for restart QA")
run_segment(
    input_path=input_path,
    case_name=f"{CASE_NAME}_full",
    output_dir=uninterrupted_output_dir,
    resume_steps=FIRST_NOUT + RESUME_NOUT,
    quiet=True,
)
uninterrupted_paths = output_paths(f"{CASE_NAME}_full", uninterrupted_output_dir)

# --- stitch histories and save the combined payload -------------------------------
first_payload = load_portable_array_payload(first_paths["arrays"])
resumed_payload = load_portable_array_payload(resumed_paths["arrays"])
uninterrupted_payload = load_portable_array_payload(uninterrupted_paths["arrays"])

stitched_payload = stitch_histories(first_payload, resumed_payload)
stitched_payload["case_name"] = f"{CASE_NAME}_stitched"

data_dir = OUTPUT_ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)
combined_history_path = data_dir / f"{CASE_NAME}_combined_history.npz"
array_payload = build_portable_array_payload(
    case_name=str(stitched_payload["case_name"]),
    parity_mode=str(stitched_payload["parity_mode"]),
    capability_tier=str(stitched_payload.get("capability_tier", "native_exact")),
    compare_variables=tuple(sorted(stitched_payload["variables"])),
    component_labels=tuple(stitched_payload.get("component_labels", [])),
    dimensions=stitched_payload.get("dimensions", {}),
    time_points=tuple(float(value) for value in stitched_payload["time_points"]),
    dataset_scalars=stitched_payload.get("dataset_scalars", {}),
    variables=stitched_payload["variables"],
    overrides=tuple(stitched_payload.get("overrides", [])),
    configured_nout=stitched_payload.get("configured_nout"),
    configured_timestep=stitched_payload.get("configured_timestep"),
    producer=str(stitched_payload.get("producer", "drbx")),
)
write_portable_array_payload(array_payload, combined_history_path)

# --- restart-consistency analysis JSON --------------------------------------------
restart_bundle = load_restart_bundle(first_paths["restart"])
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
analysis_payload = {
    "case_name": CASE_NAME,
    "configured_precision": PRECISION,
    "cli_precision_override": CLI_PRECISION_OVERRIDE,
    "restart_case_name": restart_bundle.case_name,
    "first_segment_completed_steps": restart_bundle.completed_steps,
    "restart_current_time": restart_bundle.current_time,
    "stitched_time_points": stitched_payload["time_points"],
    "uninterrupted_time_points": uninterrupted_payload["time_points"],
    "max_abs_density_diff_vs_uninterrupted": max_abs_density_diff,
    "max_abs_pressure_diff_vs_uninterrupted": max_abs_pressure_diff,
}
analysis_path = data_dir / f"{CASE_NAME}_analysis.json"
analysis_path.write_text(json.dumps(analysis_payload, indent=2, sort_keys=True), encoding="utf-8")
print(f"restart QA: max |Nh restarted-full| = {max_abs_density_diff:.3e}, "
      f"max |Ph restarted-full| = {max_abs_pressure_diff:.3e}")

# --- QA plots ---------------------------------------------------------------------
images_dir = OUTPUT_ROOT / "images"
images_dir.mkdir(parents=True, exist_ok=True)

# 2x2 density snapshots: initial, first-segment final, resumed final, full final.
figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
snapshot_entries = (
    ("initial density", _extract_2d(np.asarray(first_payload["variables"]["Nh"]), 0)),
    ("first segment final", _extract_2d(np.asarray(first_payload["variables"]["Nh"]), -1)),
    ("resumed final", _extract_2d(np.asarray(resumed_payload["variables"]["Nh"]), -1)),
    ("uninterrupted final", _extract_2d(np.asarray(uninterrupted_payload["variables"]["Nh"]), -1)),
)
for axis, (title, field) in zip(axes.flat, snapshot_entries, strict=True):
    image = axis.imshow(field.T, origin="lower", aspect="auto", cmap="viridis")
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.colorbar(image, ax=axis, shrink=0.85)
snapshots_path = images_dir / f"{CASE_NAME}_density_snapshots.png"
figure.savefig(snapshots_path, dpi=180)
plt.close(figure)

# Restarted vs uninterrupted peak histories and the pointwise error trace.
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
consistency_path = images_dir / f"{CASE_NAME}_restart_consistency.png"
figure.savefig(consistency_path, dpi=180)
plt.close(figure)

# 3D surface of the final restarted density.
final_density = _extract_2d(stitched_density, -1)
x = np.arange(final_density.shape[0], dtype=np.float64)
y = np.arange(final_density.shape[1], dtype=np.float64)
grid_y, grid_x = np.meshgrid(y, x)
figure = plt.figure(figsize=(8, 5.5), constrained_layout=True)
axis = figure.add_subplot(111, projection="3d")
axis.plot_surface(grid_y, grid_x, final_density, cmap="viridis", linewidth=0.0, antialiased=True)
axis.set_xlabel("y")
axis.set_ylabel("x")
axis.set_zlabel("Nh")
axis.set_title("Final restarted density surface")
surface_path = images_dir / f"{CASE_NAME}_density_surface.png"
figure.savefig(surface_path, dpi=180)
plt.close(figure)

# Optional GIF movie of the stitched density history.
movie_path: Path | None = None
if MAKE_MOVIE:
    movies_dir = OUTPUT_ROOT / "movies"
    movies_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    image = axis.imshow(_extract_2d(stitched_density, 0).T, origin="lower", aspect="auto", cmap="viridis")
    title = axis.set_title(f"Nh, t = {time_points[0]:.2f}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.colorbar(image, ax=axis, shrink=0.85)

    def _update(frame_index: int):
        image.set_data(_extract_2d(stitched_density, frame_index).T)
        title.set_text(f"Nh, t = {time_points[frame_index]:.2f}")
        return image, title

    movie = animation.FuncAnimation(figure, _update, frames=len(time_points), interval=300, blit=False)
    movie_path = movies_dir / f"{CASE_NAME}_density.gif"
    try:
        movie.save(movie_path, writer=animation.PillowWriter(fps=MOVIE_FPS))
    except Exception:
        movie_path = None
    plt.close(figure)

# --- artifact listing -------------------------------------------------------------
print("\nGenerated artifacts")
for label, path in {
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
}.items():
    print(f"  {label}: {path}")
