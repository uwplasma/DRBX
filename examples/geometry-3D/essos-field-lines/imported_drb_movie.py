"""Reduced DRB movie on an imported ESSOS field, plus its promotion gates.

One script for the four imported-field movie workflows (pick with ``MODE``):

- ``"movie"``: trace the Landreman-Paul QA field with ESSOS, build the FCI map
  (``MAP_SOURCE`` = ``"coil"``, ``"vmec"`` or ``"hybrid"``), run the reduced
  DRB transient, and write report JSON, arrays NPZ, snapshot/diagnostic/poster
  PNGs and a GIF;
- ``"stationarity"``: the same transient, report-only, with a tail-window
  stationarity gate (``TAIL_FRACTION``, ``RELATIVE_TOLERANCE``, ``MIN_FRAMES``);
- ``"refinement"``: report-only grid (``GRID_SHAPES``) and timestep
  (``TIME_DT_VALUES`` on ``TIME_SHAPE``) sweeps plus the refinement summary;
- ``"refinement_summary"``: no simulation; summarize existing movie reports
  (``GRID_REPORT_JSON_PATHS`` / ``TIME_REPORT_JSON_PATHS``).

All live modes need an ESSOS checkout (``DRBX_ESSOS_ROOT``, or set
``ESSOS_ROOT``); the coil JSON / VMEC wout are resolved from it unless given.
Outputs land under ``docs/data/essos_imported_drb_movie_*_artifacts``
(cwd-relative, gitignored) and every path is printed. The defaults are the
compact documented movie case, not a converged/publication run.

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/imported_drb_movie.py

Presets used for the published artifacts (copy into PARAMETERS to reproduce):
stationarity = hybrid, grid (16, 96, 48), rho 0.20-0.60, MAXTIME 24,
TIMES_TO_TRACE 80, 12 frames x 3 substeps, DT 2e-3, 3072 jacobi iterations;
refinement = hybrid, GRID_SHAPES ((3, 4, 8), (4, 6, 12)) (publication candidate
((4, 6, 12), (8, 12, 24)) with TIME_SHAPE (8, 12, 24) and 3072 iterations),
4 frames x 2 substeps, DT 2e-3, 768 iterations, no preconditioner.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from drbx.runtime import configure_jax_runtime
from drbx.validation import (
    create_essos_imported_drb_movie_package,
    create_essos_imported_drb_movie_refinement_campaign_package,
    create_essos_imported_drb_movie_refinement_summary_package,
    create_essos_imported_drb_movie_stationarity_package,
)

# --- PARAMETERS ------------------------------------------------------------------
MODE = "movie"  # "movie", "stationarity", "refinement", or "refinement_summary"
MAP_SOURCE = "coil"  # "coil", "vmec", or "hybrid"
COIL_JSON_PATH: Path | None = None  # None -> resolved from the ESSOS checkout
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
CASE_LABEL, OUTPUT_DIR_NAME = {  # labels/roots of the published artifacts (cwd-relative)
    "movie": ("essos_imported_drb_movie_campaign", "essos_imported_drb_movie_artifacts"),
    "stationarity": ("essos_imported_drb_movie_stationarity_jacobi", "essos_imported_drb_movie_stationarity_jacobi_artifacts"),
    "refinement": ("essos_imported_drb_movie_refinement_campaign", "essos_imported_drb_movie_refinement_campaign_artifacts"),
    "refinement_summary": ("essos_imported_drb_movie_refinement_summary", "essos_imported_drb_movie_refinement_artifacts"),
}[MODE]
OUTPUT_ROOT = Path("docs/data") / OUTPUT_DIR_NAME

NX, NY, NZ = 8, 28, 80
RHO_MIN, RHO_MAX = 0.20, 0.92
MAXTIME = 135.0
TIMES_TO_TRACE = 720
FRAMES = 32
SUBSTEPS_PER_FRAME = 6
DT = 1.2e-3
POTENTIAL_ITERATIONS = 768
POTENTIAL_REGULARIZATION = 5.0
POTENTIAL_PRECONDITIONER: str | None = None  # or "jacobi"

TAIL_FRACTION = 0.50  # stationarity mode
RELATIVE_TOLERANCE = 0.35  # stationarity / refinement modes
MIN_FRAMES = 12
GRID_SHAPES = ((3, 4, 8), (4, 6, 12))  # refinement mode
TIME_SHAPE: tuple[int, int, int] | None = None
TIME_DT_VALUES = (2.0e-3, 1.0e-3)
REUSE_EXISTING_REPORTS = True
GRID_REPORT_JSON_PATHS = (  # refinement_summary mode
    Path("docs/data/essos_imported_drb_movie_hybrid_artifacts/data/essos_imported_drb_movie_hybrid_campaign.json"),
)
TIME_REPORT_JSON_PATHS = GRID_REPORT_JSON_PATHS
REQUIRE_PUBLICATION_READY = False

# --- run --------------------------------------------------------------------------
configure_jax_runtime(precision="float64")
source = dict(coil_json_path=COIL_JSON_PATH, vmec_wout_path=VMEC_WOUT_PATH, essos_root=ESSOS_ROOT, map_source=MAP_SOURCE)
trace = dict(rho_min=RHO_MIN, rho_max=RHO_MAX, maxtime=MAXTIME, times_to_trace=TIMES_TO_TRACE)
potential = dict(
    potential_iterations=POTENTIAL_ITERATIONS,
    potential_regularization=POTENTIAL_REGULARIZATION,
    potential_preconditioner=POTENTIAL_PRECONDITIONER,
)
print(f"imported DRB movie: mode={MODE}, map_source={MAP_SOURCE}, grid=({NX}, {NY}, {NZ}), "
      f"rho=[{RHO_MIN}, {RHO_MAX}], frames={FRAMES}x{SUBSTEPS_PER_FRAME}, dt={DT:g}, "
      f"potential={POTENTIAL_ITERATIONS} iterations ({POTENTIAL_PRECONDITIONER})")
print(f"output root: {OUTPUT_ROOT}")
start = time.perf_counter()
common = dict(output_root=OUTPUT_ROOT, case_label=CASE_LABEL)
if MODE == "movie":
    artifacts = create_essos_imported_drb_movie_package(
        **common, **source, **trace, **potential, nx=NX, ny=NY, nz=NZ,
        frames=FRAMES, substeps_per_frame=SUBSTEPS_PER_FRAME, dt=DT,
    )
elif MODE == "stationarity":
    artifacts = create_essos_imported_drb_movie_stationarity_package(
        **common, **source, **trace, **potential, nx=NX, ny=NY, nz=NZ,
        frames=FRAMES, substeps_per_frame=SUBSTEPS_PER_FRAME, dt=DT,
        tail_fraction=TAIL_FRACTION, relative_tolerance=RELATIVE_TOLERANCE, min_frames=MIN_FRAMES,
    )
elif MODE == "refinement":
    print(f"grid shapes={GRID_SHAPES}, time shape={TIME_SHAPE}, dt values={TIME_DT_VALUES}")
    artifacts = create_essos_imported_drb_movie_refinement_campaign_package(
        **common, **source, **trace, **potential, grid_shapes=GRID_SHAPES, time_shape=TIME_SHAPE,
        time_dt_values=TIME_DT_VALUES, frames=FRAMES, substeps_per_frame=SUBSTEPS_PER_FRAME,
        grid_dt=DT, relative_tolerance=RELATIVE_TOLERANCE, reuse_existing_reports=REUSE_EXISTING_REPORTS,
    )
elif MODE == "refinement_summary":
    artifacts = create_essos_imported_drb_movie_refinement_summary_package(
        **common, grid_report_json_paths=GRID_REPORT_JSON_PATHS,
        time_report_json_paths=TIME_REPORT_JSON_PATHS, relative_tolerance=RELATIVE_TOLERANCE,
    )
else:
    raise ValueError(f"unknown MODE={MODE!r}")
print(f"finished in {time.perf_counter() - start:.1f} s")

# --- report -----------------------------------------------------------------------
for name in ("report_json_path", "arrays_npz_path", "snapshot_png_path", "diagnostics_png_path",
             "poster_png_path", "movie_gif_path", "grid_report_json_paths", "time_report_json_paths"):
    if getattr(artifacts, name, None) is not None:
        print(f"wrote {name.removesuffix('_path').removesuffix('_paths')}: {getattr(artifacts, name)}")
report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
keys = ("passed", "publication_ready", "stationarity_passed", "grid_refinement_passed",
        "time_refinement_passed", "movie_promotion_rejection_reasons")
print("evidence: " + ", ".join(f"{key}={report[key]}" for key in keys if key in report))
if report.get("next_campaign_suggestion"):
    print(f"suggested next campaign: {report['next_campaign_suggestion']}")
if REQUIRE_PUBLICATION_READY and not report.get("publication_ready", False):
    raise RuntimeError(f"movie gate failed: {report.get('movie_promotion_rejection_reasons')}")
