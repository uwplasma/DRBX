"""Landreman-Paul QA coils -> ESSOS field lines -> DRBX FCI maps.

Uses the installed ``essos`` package and the QA coil set bundled with DRBX.
Set ``DRBX_ESSOS_ROOT`` to an ESSOS checkout to also build the FCI maps
(they need the QA VMEC wout from that checkout).

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/landreman_paul_qa_import.py
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drbx.geometry import (
    EssosCoilField,
    build_essos_imported_fci_geometry,
    resolve_essos_landreman_qa_json,
    resolve_essos_landreman_qa_wout,
    trace_essos_coil_field_lines,
)
from drbx.runtime import configure_jax_runtime

OUTPUT = Path("essos_landreman_paul_qa_import.png")
N_FIELD_LINES = 6
TIMES_TO_TRACE = 1500
MAXTIME = 300.0
R_MIN, R_MAX = 1.21, 1.32
GRID = (4, 6, 8)

configure_jax_runtime(precision="float64")

coil_json = resolve_essos_landreman_qa_json()
field = EssosCoilField.from_json(coil_json)
b_axis = np.asarray(field.B(jnp.array([[R_MIN, 0.0, 0.0]])))[0]
print(f"coils: {coil_json.name}, nfp={field.nfp}, |B|(R={R_MIN} m) = {np.linalg.norm(b_axis):.4f} T")

bundle = trace_essos_coil_field_lines(
    r_min=R_MIN,
    r_max=R_MAX,
    n_field_lines=N_FIELD_LINES,
    maxtime=MAXTIME,
    times_to_trace=TIMES_TO_TRACE,
)
print(f"traced {bundle.n_field_lines} lines x {bundle.n_times} samples, {bundle.poincare_point_count} Poincare points")

# The FCI grid is laid on scaled VMEC surfaces, so this step needs the QA
# wout (not bundled; set DRBX_ESSOS_ROOT to an ESSOS checkout).
try:
    resolve_essos_landreman_qa_wout()
except FileNotFoundError as error:
    print(f"skipping the FCI map build: {error}")
else:
    geometry = build_essos_imported_fci_geometry(
        map_source="coil",
        nx=GRID[0],
        ny=GRID[1],
        nz=GRID[2],
        rho_min=0.12,
        rho_max=0.34,
        maxtime=40.0,
        times_to_trace=160,
    )
    forward_open = float(np.mean(np.asarray(geometry.maps.forward_boundary, dtype=bool)))
    print(
        f"FCI geometry {geometry.shape}: |B| in [{float(jnp.min(geometry.magnetic_field_magnitude)):.3f}, "
        f"{float(jnp.max(geometry.magnetic_field_magnitude)):.3f}] T, open forward endpoints {forward_open:.2f}"
    )

fig, axis = plt.subplots(figsize=(4.5, 4.5))
axis.scatter(bundle.poincare_r, bundle.poincare_z, s=1.0, c=bundle.poincare_line_index, cmap="viridis")
axis.set_xlabel("R [m]")
axis.set_ylabel("Z [m]")
axis.set_aspect("equal")
axis.set_title("Landreman-Paul QA coil field lines (ESSOS)")
fig.tight_layout()
fig.savefig(OUTPUT, dpi=150)
print(f"wrote {OUTPUT}")
