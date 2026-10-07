"""VMEC closed-field control campaign: steady case and reduced transient.

The closed-field-line control lane on VMEC flux coordinates of the
Landreman-Paul QA equilibrium: a steady closed-field FCI campaign and its
reduced transient companion (optionally with a GIF). ``RUN_LIVE_VMEC`` /
``RUN_LIVE_VMEC_TRANSIENT`` off (default) writes self-contained dry-run
contract JSONs describing exactly what a live run produces; on, they need an
ESSOS checkout (``DRBX_ESSOS_ROOT`` or ``ESSOS_ROOT``) and write the live
report, arrays, plot and movie. Outputs land under
``output/vmec_closed_field`` (cwd-relative); every path is printed.

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/vmec_closed_field.py
"""

from __future__ import annotations

import time
from pathlib import Path

from drbx.runtime import configure_jax_runtime
from drbx.validation import (
    create_essos_vmec_closed_field_dry_run_package,
    create_essos_vmec_closed_field_package,
    create_essos_vmec_closed_field_transient_dry_run_package,
    create_essos_vmec_closed_field_transient_package,
)

# --- PARAMETERS ------------------------------------------------------------------
RUN_LIVE_VMEC = False
RUN_LIVE_VMEC_TRANSIENT = False
OUTPUT_ROOT = Path("output/vmec_closed_field")
CASE_LABEL = "essos_vmec_closed_field_campaign"
TRANSIENT_CASE_LABEL = "essos_vmec_closed_field_transient"
SOURCE = dict(coil_json_path=None, vmec_wout_path=None, essos_root=None)  # None -> ESSOS checkout
PRECISION = "float64"
NX, NY, NZ = 5, 8, 20
RHO_MIN, RHO_MAX = 0.20, 0.82
TRANSIENT_FRAMES = 14
TRANSIENT_SUBSTEPS_PER_FRAME = 3
TRANSIENT_DT = 2.0e-3
TRANSIENT_WRITE_MOVIE = True

# --- run --------------------------------------------------------------------------
configure_jax_runtime(precision=PRECISION)
grid = dict(output_root=OUTPUT_ROOT, nx=NX, ny=NY, nz=NZ, rho_min=RHO_MIN, rho_max=RHO_MAX)
transient = dict(frames=TRANSIENT_FRAMES, substeps_per_frame=TRANSIENT_SUBSTEPS_PER_FRAME, dt=TRANSIENT_DT,
                 write_movie=TRANSIENT_WRITE_MOVIE)
print(f"VMEC closed-field control: grid=({NX}, {NY}, {NZ}), rho=[{RHO_MIN}, {RHO_MAX}], "
      f"steady={'live' if RUN_LIVE_VMEC else 'dry run'}, transient={'live' if RUN_LIVE_VMEC_TRANSIENT else 'dry run'} "
      f"({TRANSIENT_FRAMES} frames x {TRANSIENT_SUBSTEPS_PER_FRAME}, dt={TRANSIENT_DT:g})")
start = time.perf_counter()
if RUN_LIVE_VMEC:
    steady = create_essos_vmec_closed_field_package(case_label=CASE_LABEL, **grid, **SOURCE)
    print(f"steady case done in {time.perf_counter() - start:.1f} s")
    print(f"wrote report: {steady.report_json_path}\nwrote arrays: {steady.arrays_npz_path}\nwrote plot:   {steady.plot_png_path}")
else:
    steady = create_essos_vmec_closed_field_dry_run_package(case_label=CASE_LABEL, **grid)
    print(f"wrote steady dry-run contract: {steady.contract_json_path}")
start = time.perf_counter()
if RUN_LIVE_VMEC_TRANSIENT:
    result = create_essos_vmec_closed_field_transient_package(case_label=TRANSIENT_CASE_LABEL, **grid, **SOURCE, **transient)
    print(f"transient done in {time.perf_counter() - start:.1f} s")
    print(f"wrote transient report: {result.report_json_path}\nwrote transient arrays: {result.arrays_npz_path}")
    print(f"wrote transient plot:   {result.plot_png_path}\nwrote transient movie:  {result.movie_gif_path}")
else:
    result = create_essos_vmec_closed_field_transient_dry_run_package(case_label=TRANSIENT_CASE_LABEL, **grid, **transient)
    print(f"wrote transient dry-run contract: {result.contract_json_path}")
