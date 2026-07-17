from __future__ import annotations

from pathlib import Path

from dkx.runtime import configure_jax_runtime
from dkx.validation import create_essos_imported_drb_movie_package

COIL_JSON_PATH: Path | None = None
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
MAP_SOURCE = "coil"
OUTPUT_ROOT = Path("docs/data/essos_imported_drb_movie_artifacts")
CASE_LABEL = "essos_imported_drb_movie_campaign"

NX = 8
NY = 28
NZ = 80
RHO_MIN = 0.20
RHO_MAX = 0.92
TIMES_TO_TRACE = 720
MAXTIME = 135.0
FRAMES = 32
SUBSTEPS_PER_FRAME = 6
DT = 1.2e-3


configure_jax_runtime(precision="float64")
artifacts = create_essos_imported_drb_movie_package(
    output_root=OUTPUT_ROOT,
    case_label=CASE_LABEL,
    coil_json_path=COIL_JSON_PATH,
    vmec_wout_path=VMEC_WOUT_PATH,
    essos_root=ESSOS_ROOT,
    map_source=MAP_SOURCE,
    nx=NX,
    ny=NY,
    nz=NZ,
    rho_min=RHO_MIN,
    rho_max=RHO_MAX,
    maxtime=MAXTIME,
    times_to_trace=TIMES_TO_TRACE,
    frames=FRAMES,
    substeps_per_frame=SUBSTEPS_PER_FRAME,
    dt=DT,
)

print(f"wrote report: {artifacts.report_json_path}")
print(f"wrote arrays: {artifacts.arrays_npz_path}")
print(f"wrote snapshots: {artifacts.snapshot_png_path}")
print(f"wrote diagnostics: {artifacts.diagnostics_png_path}")
print(f"wrote poster: {artifacts.poster_png_path}")
print(f"wrote movie: {artifacts.movie_gif_path}")
