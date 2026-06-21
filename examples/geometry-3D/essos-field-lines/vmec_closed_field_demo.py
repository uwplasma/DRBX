from __future__ import annotations

from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import (
    create_essos_vmec_closed_field_dry_run_package,
    create_essos_vmec_closed_field_package,
)


# SIMSOPT-style user parameters: edit these values, then run this file.
RUN_EXAMPLE = True

# The default writes a self-contained live-run contract. Set RUN_LIVE_VMEC=True
# when an ESSOS checkout and the Landreman-Paul QA VMEC wout are available.
RUN_LIVE_VMEC = False

OUTPUT_ROOT = Path("artifacts/essos_vmec_closed_field")
CASE_LABEL = "essos_vmec_closed_field_campaign"
COIL_JSON_PATH: Path | None = None
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
PRECISION = "float64"

NX = 5
NY = 8
NZ = 20
RHO_MIN = 0.20
RHO_MAX = 0.82


if RUN_EXAMPLE:
    configure_jax_runtime(precision=PRECISION)
    if RUN_LIVE_VMEC:
        artifacts = create_essos_vmec_closed_field_package(
            output_root=OUTPUT_ROOT,
            case_label=CASE_LABEL,
            coil_json_path=COIL_JSON_PATH,
            vmec_wout_path=VMEC_WOUT_PATH,
            essos_root=ESSOS_ROOT,
            nx=NX,
            ny=NY,
            nz=NZ,
            rho_min=RHO_MIN,
            rho_max=RHO_MAX,
        )
        print(f"wrote report: {artifacts.report_json_path}")
        print(f"wrote arrays: {artifacts.arrays_npz_path}")
        print(f"wrote plot: {artifacts.plot_png_path}")
    else:
        artifacts = create_essos_vmec_closed_field_dry_run_package(
            output_root=OUTPUT_ROOT,
            case_label=CASE_LABEL,
            nx=NX,
            ny=NY,
            nz=NZ,
            rho_min=RHO_MIN,
            rho_max=RHO_MAX,
        )
        print(f"wrote dry-run contract: {artifacts.contract_json_path}")
