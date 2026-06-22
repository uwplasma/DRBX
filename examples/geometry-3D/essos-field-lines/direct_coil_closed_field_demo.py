from __future__ import annotations

from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import (
    create_essos_direct_coil_closed_control_package,
    create_essos_direct_coil_closed_control_refinement_package,
    create_essos_direct_coil_closed_control_transient_package,
)


# SIMSOPT-style user parameters: edit these values, then run this file.
RUN_EXAMPLE = True

# The default is a self-contained manufactured contract. Set this to True only
# when an ESSOS checkout and the Landreman-Paul QA coil/VMEC inputs are
# available. The live mode still remains a diagnostic closed/near-closed
# control, not an open-SOL target/sheath simulation.
RUN_LIVE_ESSOS = False
RUN_REFINEMENT_GATE = True
RUN_TRANSIENT_GATE = True
WRITE_TRANSIENT_MOVIE = True

OUTPUT_ROOT = Path("artifacts/essos_direct_coil_closed_control")
CASE_LABEL = "essos_direct_coil_closed_control"
REFINEMENT_CASE_LABEL = "essos_direct_coil_closed_control_refinement"
TRANSIENT_CASE_LABEL = "essos_direct_coil_closed_control_transient"
COIL_JSON_PATH: Path | None = None
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
PRECISION = "float64"

# Seed shell and trace controls for the optional live ESSOS run.
RHO_MIN = 0.20
RHO_MAX = 0.82
N_RADIAL_SEEDS = 5
N_POLOIDAL_SEEDS = 4
MAXTIME = 900.0
TIMES_TO_TRACE = 4200
TRACE_TOLERANCE = 1.0e-8

# Return-map classification. These tolerances are normalized by the reference
# minor extent inferred from the seed shell and traced radial/vertical spans.
CLOSED_RETURN_TOLERANCE = 3.0e-2
NEAR_CLOSED_RETURN_TOLERANCE = 1.5e-1
MINIMUM_CLOSED_OR_NEAR_FRACTION = 0.20
POINCARE_SECTIONS = (0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469)

# Closed-control refinement. Each tuple is
# (n_radial_seeds, n_poloidal_seeds, trace_samples). The self-contained
# default checks whether closed/near-closed classification and same-section
# return distance remain stable as seed/time samples increase.
REFINEMENT_LEVEL_SETTINGS = (
    (3, 3, 256),
    (5, 4, 512),
    (7, 6, 768),
)
MAXIMUM_CLOSED_OR_NEAR_FRACTION_SPREAD = 5.0e-2
MAXIMUM_CLASS_FRACTION_SPREAD = 2.5e-1
MINIMUM_POINCARE_POINTS_PER_LINE = 4.0

# Reduced closed-trace scalar transient. This is a closed/near-closed control
# movie, not an open-SOL sheath/recycling/neutral simulation.
TRANSIENT_FRAMES = 12
TRANSIENT_SUBSTEPS_PER_FRAME = 4
TRANSIENT_DT = 2.0e-2
TRANSIENT_SAMPLES_PER_LINE = 192
TRANSIENT_PARALLEL_DIFFUSIVITY = 2.5e-2
TRANSIENT_ADVECTION_STRENGTH = 6.0e-2
TRANSIENT_DRIVE_STRENGTH = 5.0e-2


if RUN_EXAMPLE:
    configure_jax_runtime(precision=PRECISION)
    artifacts = create_essos_direct_coil_closed_control_package(
        output_root=OUTPUT_ROOT,
        case_label=CASE_LABEL,
        use_live_essos=RUN_LIVE_ESSOS,
        coil_json_path=COIL_JSON_PATH,
        vmec_wout_path=VMEC_WOUT_PATH,
        essos_root=ESSOS_ROOT,
        rho_min=RHO_MIN,
        rho_max=RHO_MAX,
        n_radial_seeds=N_RADIAL_SEEDS,
        n_poloidal_seeds=N_POLOIDAL_SEEDS,
        maxtime=MAXTIME,
        times_to_trace=TIMES_TO_TRACE,
        trace_tolerance=TRACE_TOLERANCE,
        poincare_sections=POINCARE_SECTIONS,
        closed_return_tolerance=CLOSED_RETURN_TOLERANCE,
        near_closed_return_tolerance=NEAR_CLOSED_RETURN_TOLERANCE,
        minimum_closed_or_near_fraction=MINIMUM_CLOSED_OR_NEAR_FRACTION,
    )

    print(f"wrote report: {artifacts.report_json_path}")
    print(f"wrote arrays: {artifacts.arrays_npz_path}")
    print(f"wrote plot: {artifacts.plot_png_path}")

    if RUN_REFINEMENT_GATE:
        refinement = create_essos_direct_coil_closed_control_refinement_package(
            output_root=OUTPUT_ROOT / "refinement",
            case_label=REFINEMENT_CASE_LABEL,
            use_live_essos=RUN_LIVE_ESSOS,
            coil_json_path=COIL_JSON_PATH,
            vmec_wout_path=VMEC_WOUT_PATH,
            essos_root=ESSOS_ROOT,
            level_settings=REFINEMENT_LEVEL_SETTINGS,
            rho_min=RHO_MIN,
            rho_max=RHO_MAX,
            maxtime=MAXTIME,
            trace_tolerance=TRACE_TOLERANCE,
            poincare_sections=POINCARE_SECTIONS,
            closed_return_tolerance=CLOSED_RETURN_TOLERANCE,
            near_closed_return_tolerance=NEAR_CLOSED_RETURN_TOLERANCE,
            minimum_closed_or_near_fraction=MINIMUM_CLOSED_OR_NEAR_FRACTION,
            maximum_closed_or_near_fraction_spread=MAXIMUM_CLOSED_OR_NEAR_FRACTION_SPREAD,
            maximum_class_fraction_spread=MAXIMUM_CLASS_FRACTION_SPREAD,
            minimum_poincare_points_per_line=MINIMUM_POINCARE_POINTS_PER_LINE,
        )

        print(f"wrote refinement report: {refinement.report_json_path}")
        print(f"wrote refinement arrays: {refinement.arrays_npz_path}")
        print(f"wrote refinement plot: {refinement.plot_png_path}")

    if RUN_TRANSIENT_GATE:
        transient = create_essos_direct_coil_closed_control_transient_package(
            output_root=OUTPUT_ROOT / "transient",
            case_label=TRANSIENT_CASE_LABEL,
            use_live_essos=RUN_LIVE_ESSOS,
            coil_json_path=COIL_JSON_PATH,
            vmec_wout_path=VMEC_WOUT_PATH,
            essos_root=ESSOS_ROOT,
            rho_min=RHO_MIN,
            rho_max=RHO_MAX,
            n_radial_seeds=N_RADIAL_SEEDS,
            n_poloidal_seeds=N_POLOIDAL_SEEDS,
            maxtime=MAXTIME,
            times_to_trace=TIMES_TO_TRACE,
            trace_tolerance=TRACE_TOLERANCE,
            poincare_sections=POINCARE_SECTIONS,
            closed_return_tolerance=CLOSED_RETURN_TOLERANCE,
            near_closed_return_tolerance=NEAR_CLOSED_RETURN_TOLERANCE,
            minimum_closed_or_near_fraction=MINIMUM_CLOSED_OR_NEAR_FRACTION,
            frames=TRANSIENT_FRAMES,
            substeps_per_frame=TRANSIENT_SUBSTEPS_PER_FRAME,
            dt=TRANSIENT_DT,
            samples_per_line=TRANSIENT_SAMPLES_PER_LINE,
            parallel_diffusivity=TRANSIENT_PARALLEL_DIFFUSIVITY,
            advection_strength=TRANSIENT_ADVECTION_STRENGTH,
            drive_strength=TRANSIENT_DRIVE_STRENGTH,
            write_movie=WRITE_TRANSIENT_MOVIE,
        )

        print(f"wrote transient report: {transient.report_json_path}")
        print(f"wrote transient arrays: {transient.arrays_npz_path}")
        print(f"wrote transient plot: {transient.plot_png_path}")
        if transient.movie_gif_path is not None:
            print(f"wrote transient movie: {transient.movie_gif_path}")
