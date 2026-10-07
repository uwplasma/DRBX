"""PyTree/JVP check of the FCI DRB right-hand side on an imported ESSOS map.

Builds the imported Landreman-Paul QA FCI geometry from ESSOS traces
(``MAP_SOURCE`` = ``"coil"``, ``"vmec"`` or ``"hybrid"``), evaluates the
native FCI DRB right-hand side as a JAX PyTree for ``STEPS`` explicit steps,
and checks forward-mode (JVP) derivatives through it. Writes the report JSON,
arrays NPZ and plot under ``docs/data/essos_imported_pytree_artifacts``
(cwd-relative, gitignored). Needs an ESSOS checkout (``DRBX_ESSOS_ROOT`` or
``ESSOS_ROOT``).

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/imported_pytree_campaign.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from drbx.runtime import configure_jax_runtime
from drbx.validation import create_essos_imported_pytree_campaign_package

# --- PARAMETERS ------------------------------------------------------------------
MAP_SOURCE = "coil"
COIL_JSON_PATH: Path | None = None  # None -> resolved from the ESSOS checkout
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
OUTPUT_ROOT = Path("docs/data/essos_imported_pytree_artifacts")
CASE_LABEL = "essos_imported_pytree_campaign"
NX, NY, NZ = 4, 6, 12
RHO_MIN, RHO_MAX = 0.12, 0.34
TIMES_TO_TRACE = 280
MAXTIME = 60.0
STEPS = 5

# --- run --------------------------------------------------------------------------
configure_jax_runtime(precision="float64")
print(f"imported PyTree campaign: map_source={MAP_SOURCE}, grid=({NX}, {NY}, {NZ}), "
      f"rho=[{RHO_MIN}, {RHO_MAX}], maxtime={MAXTIME}, times_to_trace={TIMES_TO_TRACE}, steps={STEPS}")
start = time.perf_counter()
artifacts = create_essos_imported_pytree_campaign_package(
    output_root=OUTPUT_ROOT, case_label=CASE_LABEL, coil_json_path=COIL_JSON_PATH,
    vmec_wout_path=VMEC_WOUT_PATH, essos_root=ESSOS_ROOT, map_source=MAP_SOURCE,
    nx=NX, ny=NY, nz=NZ, rho_min=RHO_MIN, rho_max=RHO_MAX, maxtime=MAXTIME,
    times_to_trace=TIMES_TO_TRACE, steps=STEPS,
)
print(f"finished in {time.perf_counter() - start:.1f} s")
report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
print(f"passed: {report.get('passed')}")
print(f"wrote report: {artifacts.report_json_path}")
print(f"wrote arrays: {artifacts.arrays_npz_path}")
print(f"wrote plot:   {artifacts.plot_png_path}")
