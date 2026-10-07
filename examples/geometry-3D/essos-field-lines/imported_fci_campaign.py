"""Imported-field FCI validation campaign (coil, VMEC, and hybrid maps).

For each map source in ``MAP_SOURCES`` the script builds the FCI maps of the
Landreman-Paul QA configuration from ESSOS field-line traces (``"coil"``:
Biot-Savart coils, ``"vmec"``: VMEC coordinates, ``"hybrid"``: VMEC coordinates
with coil-derived endpoint masks) and writes the validation report JSON
(connection length, endpoint/target labels, map quality), arrays NPZ and plot.

``DRY_RUN = True`` (default) needs no ESSOS: it prints the resolved settings
and, with ``WRITE_DRY_RUN_ARTIFACTS``, writes the self-contained dry-run
contract JSON listing every report field and array a live run must produce.
``DRY_RUN = False`` needs an ESSOS checkout (``DRBX_ESSOS_ROOT`` or
``ESSOS_ROOT``). Outputs land under ``docs/data/essos_imported_fci[_vmec|_hybrid]_artifacts``
(cwd-relative, gitignored); every path is printed.

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/imported_fci_campaign.py
"""

from __future__ import annotations

import time
from pathlib import Path

from drbx.runtime import configure_jax_runtime
from drbx.validation.essos_imported_fci_campaign import (
    create_essos_imported_fci_campaign_package,
    create_essos_imported_fci_dry_run_artifact_package,
)

# --- PARAMETERS ------------------------------------------------------------------
DRY_RUN = True
WRITE_DRY_RUN_ARTIFACTS = True
MAP_SOURCES = ("coil",)  # any of "coil", "vmec", "hybrid"
COIL_JSON_PATH: Path | None = None  # None -> resolved from the ESSOS checkout
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
NX, NY, NZ = 5, 8, 20
RHO_MIN, RHO_MAX = 0.12, 0.34
TIMES_TO_TRACE = 360
MAXTIME = 80.0
TRACE_TOLERANCE = 1.0e-8
PRECISION = "float64"
REQUIRE_CONNECTION_RESOLUTION = False

# --- run --------------------------------------------------------------------------
configure_jax_runtime(precision=PRECISION)
for source in MAP_SOURCES:
    suffix = "" if source == "coil" else f"_{source}"
    settings = dict(
        output_root=Path(f"docs/data/essos_imported_fci{suffix}_artifacts"),
        case_label=f"essos_imported_fci{suffix}_campaign",
        coil_json_path=COIL_JSON_PATH, vmec_wout_path=VMEC_WOUT_PATH, essos_root=ESSOS_ROOT,
        map_source=source, nx=NX, ny=NY, nz=NZ, rho_min=RHO_MIN, rho_max=RHO_MAX, maxtime=MAXTIME,
        times_to_trace=TIMES_TO_TRACE, trace_tolerance=TRACE_TOLERANCE,
        require_connection_resolution=REQUIRE_CONNECTION_RESOLUTION,
    )
    print(f"imported FCI campaign ({'dry run' if DRY_RUN else 'live'}): map_source={source}, "
          f"grid=({NX}, {NY}, {NZ}), rho=[{RHO_MIN:g}, {RHO_MAX:g}], maxtime={MAXTIME:g}, "
          f"times_to_trace={TIMES_TO_TRACE}, output_root={settings['output_root']}")
    start = time.perf_counter()
    if DRY_RUN:
        if WRITE_DRY_RUN_ARTIFACTS:
            artifacts = create_essos_imported_fci_dry_run_artifact_package(**settings, precision=PRECISION)
            print(f"  wrote dry-run contract: {artifacts.contract_json_path}")
        continue
    artifacts = create_essos_imported_fci_campaign_package(**settings)
    print(f"  traced and validated in {time.perf_counter() - start:.1f} s")
    print(f"  wrote report: {artifacts.report_json_path}")
    print(f"  wrote arrays: {artifacts.arrays_npz_path}")
    print(f"  wrote plot:   {artifacts.plot_png_path}")
