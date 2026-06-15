from __future__ import annotations

from pathlib import Path

from jax_drb.validation import (
    create_essos_imported_connection_length_refinement_package,
    create_live_essos_imported_connection_length_refinement_package,
)

# SIMSOPT-style user parameters: edit these first, then run this file.
# The default is self-contained and does not require an external coil or VMEC
# checkout. Live imported connection-length arrays can be passed to the same
# API from a larger campaign when ESSOS/VMEC artifacts are available.
OUTPUT_ROOT = Path("docs/data/essos_imported_connection_length_refinement_artifacts")
CASE_LABEL = "essos_imported_connection_length_refinement"
LIVE_IMPORT = False
MAP_SOURCE = "hybrid"  # "coil", "vmec", or "hybrid" when LIVE_IMPORT = True
CONNECTION_QUANTITY = "raw_connection_length"
# Use "parallel_step_per_toroidal_radian" for closed VMEC step-length
# refinement, where raw adjacent-plane length changes with toroidal spacing.

LEVEL_SHAPES = (
    (4, 6, 8),
    (8, 12, 16),
    (16, 24, 32),
)
LIVE_LEVEL_SHAPES = (
    (3, 4, 6),
    (6, 8, 12),
)
MAXTIME = 40.0
TIMES_TO_TRACE = 160
TRACE_TOLERANCE = 1.0e-8
CONVERGENCE_THRESHOLD = 0.02
LINF_THRESHOLD = 0.05
MINIMUM_OBSERVED_ORDER = 1.5


if LIVE_IMPORT:
    artifacts = create_live_essos_imported_connection_length_refinement_package(
        output_root=OUTPUT_ROOT,
        case_label=f"{CASE_LABEL}_{MAP_SOURCE}_live",
        map_source=MAP_SOURCE,
        connection_quantity=CONNECTION_QUANTITY,
        level_shapes=LIVE_LEVEL_SHAPES,
        maxtime=MAXTIME,
        times_to_trace=TIMES_TO_TRACE,
        trace_tolerance=TRACE_TOLERANCE,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        linf_threshold=LINF_THRESHOLD,
        minimum_observed_order=0.0,
    )
else:
    artifacts = create_essos_imported_connection_length_refinement_package(
        output_root=OUTPUT_ROOT,
        case_label=CASE_LABEL,
        level_shapes=LEVEL_SHAPES,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        linf_threshold=LINF_THRESHOLD,
        minimum_observed_order=MINIMUM_OBSERVED_ORDER,
    )

print(f"wrote report: {artifacts.report_json_path}")
print(f"wrote arrays: {artifacts.arrays_npz_path}")
print(f"wrote plot:   {artifacts.plot_png_path}")
