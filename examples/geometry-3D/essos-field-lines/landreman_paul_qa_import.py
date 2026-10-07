"""Trace Landreman-Paul QA coil field lines with ESSOS and import them.

Traces ``N_FIELD_LINES`` field lines seeded on the outboard midplane between
``R_MIN`` and ``R_MAX`` through the ESSOS Biot-Savart field of the
Landreman-Paul QA coil set, imports the bundle into drbx, and writes the
report JSON, the field-line NPZ, and a Poincare/3D plot under
``docs/data/essos_fieldline_import_artifacts`` (cwd-relative, gitignored).
Needs an ESSOS checkout (``DRBX_ESSOS_ROOT``, or set ``COIL_JSON_PATH``).

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/landreman_paul_qa_import.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from drbx.runtime import configure_jax_runtime
from drbx.validation import create_essos_fieldline_import_package

# --- PARAMETERS ------------------------------------------------------------------
COIL_JSON_PATH: Path | None = None  # None -> resolved from the ESSOS checkout
OUTPUT_ROOT = Path("docs/data/essos_fieldline_import_artifacts")
N_FIELD_LINES = 8
TIMES_TO_TRACE = 6000
MAXTIME = 1000.0
R_MIN, R_MAX = 1.21, 1.40

# --- run --------------------------------------------------------------------------
configure_jax_runtime(precision="float64")
print(f"tracing {N_FIELD_LINES} coil field lines, R = {R_MIN}..{R_MAX} m, "
      f"maxtime={MAXTIME}, {TIMES_TO_TRACE} samples each")
start = time.perf_counter()
artifacts = create_essos_fieldline_import_package(
    output_root=OUTPUT_ROOT, coil_json_path=COIL_JSON_PATH, r_min=R_MIN, r_max=R_MAX,
    n_field_lines=N_FIELD_LINES, maxtime=MAXTIME, times_to_trace=TIMES_TO_TRACE,
)
print(f"finished in {time.perf_counter() - start:.1f} s")
report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
print(f"passed: {report.get('passed')}")
print(f"wrote report: {artifacts.report_json_path}")
print(f"wrote arrays: {artifacts.arrays_npz_path}")
print(f"wrote plot:   {artifacts.plot_png_path}")
