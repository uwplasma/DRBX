"""Coil-traced field lines versus VMEC flux surfaces (Landreman-Paul QA).

Traces field lines with ESSOS (``FIELD_SOURCE`` = ``"coil"`` Biot-Savart or
``"vmec"``) from ``N_SURFACES`` seeds between ``RHO_MIN`` and ``RHO_MAX``,
overlays their Poincare crossings on the scaled VMEC flux surfaces at four
toroidal sections, and writes the report JSON, arrays NPZ and plot under
``docs/data/essos_vmec_fieldline_surface_artifacts`` (cwd-relative,
gitignored). Needs an ESSOS checkout (``DRBX_ESSOS_ROOT`` or ``ESSOS_ROOT``).

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/vmec_fieldline_surface_campaign.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from drbx.runtime import configure_jax_runtime
from drbx.validation import create_essos_vmec_fieldline_surface_package

# --- PARAMETERS ------------------------------------------------------------------
FIELD_SOURCE = "coil"
COIL_JSON_PATH: Path | None = None  # None -> resolved from the ESSOS checkout
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
OUTPUT_ROOT = Path("docs/data/essos_vmec_fieldline_surface_artifacts")
CASE_LABEL = "essos_vmec_fieldline_surface_campaign"
RHO_MIN, RHO_MAX = 0.20, 0.92
N_SURFACES = 7
NTHETA_SURFACE = 320
TIMES_TO_TRACE = 4200
MAXTIME = 900.0

# --- run --------------------------------------------------------------------------
configure_jax_runtime(precision="float64")
print(f"field-line/surface campaign: field_source={FIELD_SOURCE}, {N_SURFACES} seeds in "
      f"rho=[{RHO_MIN}, {RHO_MAX}], maxtime={MAXTIME}, {TIMES_TO_TRACE} samples each")
start = time.perf_counter()
artifacts = create_essos_vmec_fieldline_surface_package(
    output_root=OUTPUT_ROOT, case_label=CASE_LABEL, coil_json_path=COIL_JSON_PATH,
    vmec_wout_path=VMEC_WOUT_PATH, essos_root=ESSOS_ROOT, rho_min=RHO_MIN, rho_max=RHO_MAX,
    n_surfaces=N_SURFACES, ntheta_surface=NTHETA_SURFACE, times_to_trace=TIMES_TO_TRACE,
    maxtime=MAXTIME, field_source=FIELD_SOURCE,
)
print(f"finished in {time.perf_counter() - start:.1f} s")
report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
print(f"passed: {report.get('passed')}")
print(f"wrote report: {artifacts.report_json_path}")
print(f"wrote arrays: {artifacts.arrays_npz_path}")
print(f"wrote plot:   {artifacts.plot_png_path}")
