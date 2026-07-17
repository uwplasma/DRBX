"""VMEC closed-field control campaign (dry-run contracts by default).

The script drives the closed-field-line control lane on VMEC flux coordinates:
the steady closed-field campaign and its reduced transient companion. By
default (``RUN_LIVE_VMEC = False``) it writes self-contained dry-run contract
JSONs describing exactly what a live run would produce; with an ESSOS checkout
and a Landreman-Paul QA VMEC wout (set ``COIL_JSON_PATH``/``VMEC_WOUT_PATH``/
``ESSOS_ROOT`` and the ``RUN_LIVE_*`` flags) it produces the live report,
arrays, plot, and optional transient GIF instead.

Artifacts land under ``artifacts/essos_vmec_closed_field`` (relative to the
current working directory) and every written path is printed.

Edit the PARAMETERS constants below and run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/vmec_closed_field.py
"""

from __future__ import annotations

from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import (
    create_essos_vmec_closed_field_dry_run_package,
    create_essos_vmec_closed_field_package,
    create_essos_vmec_closed_field_transient_dry_run_package,
    create_essos_vmec_closed_field_transient_package,
)


# --- PARAMETERS: edit these values, then run this file -----------------------------
RUN_EXAMPLE = True

# The default writes a self-contained live-run contract. Set RUN_LIVE_VMEC=True
# when an ESSOS checkout and the Landreman-Paul QA VMEC wout are available.
RUN_LIVE_VMEC = False
RUN_LIVE_VMEC_TRANSIENT = False

OUTPUT_ROOT = Path("artifacts/essos_vmec_closed_field")
CASE_LABEL = "essos_vmec_closed_field_campaign"
TRANSIENT_CASE_LABEL = "essos_vmec_closed_field_transient"
COIL_JSON_PATH: Path | None = None
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
PRECISION = "float64"

NX = 5
NY = 8
NZ = 20
RHO_MIN = 0.20
RHO_MAX = 0.82

TRANSIENT_FRAMES = 14
TRANSIENT_SUBSTEPS_PER_FRAME = 3
TRANSIENT_DT = 2.0e-3
TRANSIENT_WRITE_MOVIE = True


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
    if RUN_LIVE_VMEC_TRANSIENT:
        transient_artifacts = create_essos_vmec_closed_field_transient_package(
            output_root=OUTPUT_ROOT,
            case_label=TRANSIENT_CASE_LABEL,
            coil_json_path=COIL_JSON_PATH,
            vmec_wout_path=VMEC_WOUT_PATH,
            essos_root=ESSOS_ROOT,
            nx=NX,
            ny=NY,
            nz=NZ,
            rho_min=RHO_MIN,
            rho_max=RHO_MAX,
            frames=TRANSIENT_FRAMES,
            substeps_per_frame=TRANSIENT_SUBSTEPS_PER_FRAME,
            dt=TRANSIENT_DT,
            write_movie=TRANSIENT_WRITE_MOVIE,
        )
        print(f"wrote transient report: {transient_artifacts.report_json_path}")
        print(f"wrote transient arrays: {transient_artifacts.arrays_npz_path}")
        print(f"wrote transient plot: {transient_artifacts.plot_png_path}")
        if transient_artifacts.movie_gif_path is not None:
            print(f"wrote transient movie: {transient_artifacts.movie_gif_path}")
    else:
        transient_artifacts = create_essos_vmec_closed_field_transient_dry_run_package(
            output_root=OUTPUT_ROOT,
            case_label=TRANSIENT_CASE_LABEL,
            nx=NX,
            ny=NY,
            nz=NZ,
            rho_min=RHO_MIN,
            rho_max=RHO_MAX,
            frames=TRANSIENT_FRAMES,
            substeps_per_frame=TRANSIENT_SUBSTEPS_PER_FRAME,
            dt=TRANSIENT_DT,
            write_movie=TRANSIENT_WRITE_MOVIE,
        )
        print(f"wrote transient dry-run contract: {transient_artifacts.contract_json_path}")
