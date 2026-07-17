from __future__ import annotations

from pathlib import Path

from drbx.runtime import configure_jax_runtime
from drbx.validation import create_essos_vmec_fieldline_surface_package

COIL_JSON_PATH: Path | None = None
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
OUTPUT_ROOT = Path("docs/data/essos_vmec_fieldline_surface_artifacts")
CASE_LABEL = "essos_vmec_fieldline_surface_campaign"
FIELD_SOURCE = "coil"

RHO_MIN = 0.20
RHO_MAX = 0.92
N_SURFACES = 7
NTHETA_SURFACE = 320
TIMES_TO_TRACE = 4200
MAXTIME = 900.0


configure_jax_runtime(precision="float64")
artifacts = create_essos_vmec_fieldline_surface_package(
    output_root=OUTPUT_ROOT,
    case_label=CASE_LABEL,
    coil_json_path=COIL_JSON_PATH,
    vmec_wout_path=VMEC_WOUT_PATH,
    essos_root=ESSOS_ROOT,
    rho_min=RHO_MIN,
    rho_max=RHO_MAX,
    n_surfaces=N_SURFACES,
    ntheta_surface=NTHETA_SURFACE,
    times_to_trace=TIMES_TO_TRACE,
    maxtime=MAXTIME,
    field_source=FIELD_SOURCE,
)

print(f"wrote report: {artifacts.report_json_path}")
print(f"wrote arrays: {artifacts.arrays_npz_path}")
print(f"wrote plot: {artifacts.plot_png_path}")
