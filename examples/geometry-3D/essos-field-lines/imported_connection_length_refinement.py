"""Nested-grid connection-length refinement gate for imported FCI maps.

The promotion-blocking refinement diagnostic for connection-length-type
quantities: the same quantity is computed on nested grids (``LEVEL_SHAPES``,
each level an integer refinement of the previous), restricted to the coarse
cells, and checked for RMS/Linf convergence and observed order against the
thresholds below.

``LIVE_IMPORT = False`` (default) is self-contained: a manufactured nested-grid
field with a known answer; the gate must pass (``REQUIRE_PASS``). With
``LIVE_IMPORT = True`` (ESSOS checkout via ``DRBX_ESSOS_ROOT``/``ESSOS_ROOT``)
each map source in ``MAP_SOURCES`` is traced live; the grid-invariant quantity
is ``adjacent_step_length`` for ``"coil"`` and
``parallel_step_per_toroidal_radian`` for ``"vmec"``/``"hybrid"`` (pure coil
maps are an expected negative control). Report JSON, arrays NPZ, plot PNG and
a sweep summary land under
``docs/data/essos_imported_connection_length_refinement_artifacts``
(cwd-relative, gitignored).

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/imported_connection_length_refinement.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from drbx.validation import (
    create_essos_imported_connection_length_refinement_package,
    create_live_essos_imported_connection_length_refinement_package,
)

# --- PARAMETERS ------------------------------------------------------------------
LIVE_IMPORT = False
MAP_SOURCES = ("hybrid",)  # live mode: any of "coil", "vmec", "hybrid"
OUTPUT_ROOT = Path("docs/data/essos_imported_connection_length_refinement_artifacts")
CASE_LABEL = "essos_imported_connection_length_refinement"
COIL_JSON_PATH: Path | None = None  # None -> resolved from the ESSOS checkout
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
# manufactured gate (default) / live gate
LEVEL_SHAPES = ((4, 6, 8), (8, 12, 16), (16, 24, 32)) if not LIVE_IMPORT else ((3, 4, 6), (6, 8, 12), (12, 16, 24))
CONVERGENCE_THRESHOLD = 0.02 if not LIVE_IMPORT else 0.10  # finest normalized RMS error
LINF_THRESHOLD = 0.05 if not LIVE_IMPORT else 0.20
MINIMUM_OBSERVED_ORDER = 1.5 if not LIVE_IMPORT else 0.5
REQUIRE_OBSERVED_ORDER = True
RHO_MIN, RHO_MAX = 0.12, 0.34  # live tracing only
MAXTIME = 40.0
TIMES_TO_TRACE = 160
TRACE_TOLERANCE = 1.0e-8
REQUIRE_PASS = True
QUANTITY = {"coil": "adjacent_step_length", "vmec": "parallel_step_per_toroidal_radian",
            "hybrid": "parallel_step_per_toroidal_radian"}

# --- run --------------------------------------------------------------------------
gate = dict(level_shapes=LEVEL_SHAPES, convergence_threshold=CONVERGENCE_THRESHOLD, linf_threshold=LINF_THRESHOLD,
            minimum_observed_order=MINIMUM_OBSERVED_ORDER, require_observed_order=REQUIRE_OBSERVED_ORDER)
entries = []
for source in MAP_SOURCES if LIVE_IMPORT else ("manufactured",):
    label = f"{CASE_LABEL}_{source}_live" if LIVE_IMPORT else CASE_LABEL
    print(f"connection-length refinement: source={source}, quantity={QUANTITY.get(source, 'manufactured')}, "
          f"levels={LEVEL_SHAPES}, thresholds rms<{CONVERGENCE_THRESHOLD}, linf<{LINF_THRESHOLD}, "
          f"order>={MINIMUM_OBSERVED_ORDER}")
    start = time.perf_counter()
    if LIVE_IMPORT:
        artifacts = create_live_essos_imported_connection_length_refinement_package(
            output_root=OUTPUT_ROOT, case_label=label, coil_json_path=COIL_JSON_PATH, vmec_wout_path=VMEC_WOUT_PATH,
            essos_root=ESSOS_ROOT, map_source=source, connection_quantity=QUANTITY[source], rho_min=RHO_MIN,
            rho_max=RHO_MAX, maxtime=MAXTIME, times_to_trace=TIMES_TO_TRACE, trace_tolerance=TRACE_TOLERANCE, **gate,
        )
    else:
        artifacts = create_essos_imported_connection_length_refinement_package(
            output_root=OUTPUT_ROOT, case_label=label, **gate
        )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    entry = {key: report.get(key) for key in (
        "passed", "promotion_ready", "advisory_only", "evidence_role", "promotion_rejection_reasons",
        "finest_normalized_rms_error", "finest_normalized_linf_error", "minimum_observed_order_actual",
        "monotonic_rms_error_reduction", "monotonic_linf_error_reduction")}
    entry.update(case_label=label, map_source=source, report_json_path=str(artifacts.report_json_path))
    entries.append(entry)
    print(f"  done in {time.perf_counter() - start:.1f} s: promotion_ready={entry['promotion_ready']}, "
          f"role={entry['evidence_role']}, rms={entry['finest_normalized_rms_error']}, "
          f"linf={entry['finest_normalized_linf_error']}, order={entry['minimum_observed_order_actual']}")
    print(f"  wrote report: {artifacts.report_json_path}")
    print(f"  wrote arrays: {artifacts.arrays_npz_path}")
    print(f"  wrote plot:   {artifacts.plot_png_path}")
    if REQUIRE_PASS and not report.get("promotion_ready", report.get("passed")):
        raise RuntimeError(f"connection-length refinement gate failed: {entry}")

summary_path = OUTPUT_ROOT / "data" / f"{CASE_LABEL}_summary.json"
summary_path.write_text(json.dumps({
    "diagnostic": "essos_imported_connection_length_refinement_sweep",
    "report_count": len(entries),
    "promotion_ready_count": sum(bool(entry["promotion_ready"]) for entry in entries),
    "negative_control_count": sum(str(entry["evidence_role"]).startswith("negative_") for entry in entries),
    "all_promotion_ready": all(bool(entry["promotion_ready"]) for entry in entries),
    "entries": entries,
}, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote sweep summary: {summary_path}")
if REQUIRE_PASS:
    print("connection-length refinement gate passed")
