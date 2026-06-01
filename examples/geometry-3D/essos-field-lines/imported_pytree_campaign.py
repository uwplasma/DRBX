from __future__ import annotations

from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import create_essos_imported_pytree_campaign_package

COIL_JSON_PATH: Path | None = None
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
MAP_SOURCE = "coil"
OUTPUT_ROOT = Path("docs/data/essos_imported_pytree_artifacts")
CASE_LABEL = "essos_imported_pytree_campaign"

NX = 4
NY = 6
NZ = 12
RHO_MIN = 0.12
RHO_MAX = 0.34
TIMES_TO_TRACE = 280
MAXTIME = 60.0
STEPS = 5


configure_jax_runtime(precision="float64")
artifacts = create_essos_imported_pytree_campaign_package(
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
    steps=STEPS,
)

print(f"wrote report: {artifacts.report_json_path}")
print(f"wrote arrays: {artifacts.arrays_npz_path}")
print(f"wrote plot: {artifacts.plot_png_path}")
