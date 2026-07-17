"""Build and plot a synthetic non-axisymmetric stellarator FCI geometry.

The script assembles the analytic synthetic stellarator geometry (islands,
mirror modulation, sheared iota profile) through the public
``build_synthetic_stellarator_geometry`` API, runs the geometry/metric QA
report, and saves both a compressed NPZ of the coordinate/metric arrays and a
multi-panel geometry figure. It prints the geometry family, the metric inverse
residual, the mirror ratio, and every artifact path.

Artifacts land under ``docs/data/stellarator_fci_example_artifacts/geometry``
(relative to the current working directory):
``stellarator_geometry_plotting.npz`` and ``stellarator_geometry_plotting.png``.

Edit the PARAMETERS constants below (grid shape, island/mirror amplitudes,
iota profile) and run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/stellarator-fci/geometry_plotting.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dkx.geometry import build_synthetic_stellarator_geometry
from dkx.validation import build_stellarator_fci_geometry_report, save_stellarator_fci_geometry_plot

# --- PARAMETERS ------------------------------------------------------------------
OUTPUT_ROOT = Path("docs/data/stellarator_fci_example_artifacts/geometry")  # artifact root (cwd-relative)
CASE_LABEL = "stellarator_geometry_plotting"

NX = 32
NY = 28
NZ = 56
MAJOR_RADIUS = 3.8
MINOR_RADIUS = 0.7
ELONGATION = 1.45
FIELD_PERIODS = 5
ISLAND_MODE = 2
ISLAND_AMPLITUDE = 0.030
MIRROR_AMPLITUDE = 0.16
IOTA_AXIS = 0.38
IOTA_EDGE = 0.58

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

geometry = build_synthetic_stellarator_geometry(
    nx=NX,
    ny=NY,
    nz=NZ,
    major_radius=MAJOR_RADIUS,
    minor_radius=MINOR_RADIUS,
    elongation=ELONGATION,
    field_periods=FIELD_PERIODS,
    island_mode=ISLAND_MODE,
    island_amplitude=ISLAND_AMPLITUDE,
    mirror_amplitude=MIRROR_AMPLITUDE,
    iota_axis=IOTA_AXIS,
    iota_edge=IOTA_EDGE,
)

report = build_stellarator_fci_geometry_report(geometry)
plot_path = save_stellarator_fci_geometry_plot(geometry, report, OUTPUT_ROOT / f"{CASE_LABEL}.png")
arrays_path = OUTPUT_ROOT / f"{CASE_LABEL}.npz"
np.savez_compressed(
    arrays_path,
    x=np.asarray(geometry.coordinates_x, dtype=np.float32),
    y=np.asarray(geometry.coordinates_y, dtype=np.float32),
    z=np.asarray(geometry.coordinates_z, dtype=np.float32),
    radial=np.asarray(geometry.radial, dtype=np.float32),
    toroidal_angle=np.asarray(geometry.toroidal_angle, dtype=np.float32),
    poloidal_angle=np.asarray(geometry.poloidal_angle, dtype=np.float32),
    Bxy=np.asarray(geometry.metric.Bxy, dtype=np.float32),
    curvature=np.asarray(geometry.curvature, dtype=np.float32),
    connection_length=np.asarray(geometry.connection_length, dtype=np.float32),
)

print(f"geometry family: {report['geometry']['geometry_family']}")
print(f"metric residual: {report['metric']['inverse_residual_linf']:.3e}")
print(f"mirror ratio: {report['magnetic_field']['mirror_ratio']:.3f}")
print(f"wrote arrays: {arrays_path}")
print(f"wrote plot:   {plot_path}")
