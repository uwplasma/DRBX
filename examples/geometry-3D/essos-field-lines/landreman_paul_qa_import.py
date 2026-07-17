from __future__ import annotations

from pathlib import Path

from drbx.runtime import configure_jax_runtime
from drbx.validation import create_essos_fieldline_import_package

COIL_JSON_PATH: Path | None = None
OUTPUT_ROOT = Path("docs/data/essos_fieldline_import_artifacts")
N_FIELD_LINES = 8
TIMES_TO_TRACE = 6000
MAXTIME = 1000.0
R_MIN = 1.21
R_MAX = 1.40


configure_jax_runtime(precision="float64")
artifacts = create_essos_fieldline_import_package(
    output_root=OUTPUT_ROOT,
    coil_json_path=COIL_JSON_PATH,
    r_min=R_MIN,
    r_max=R_MAX,
    n_field_lines=N_FIELD_LINES,
    maxtime=MAXTIME,
    times_to_trace=TIMES_TO_TRACE,
)

print(f"wrote report: {artifacts.report_json_path}")
print(f"wrote arrays: {artifacts.arrays_npz_path}")
print(f"wrote plot: {artifacts.plot_png_path}")
