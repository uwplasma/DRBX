from __future__ import annotations

import argparse
import os
from pathlib import Path

from jax_drb.reference.paths import require_reference_root
from jax_drb.validation import (
    create_neutral_mixed_term_balance_campaign_package,
    run_neutral_mixed_hermes_diagnostic_rerun,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the neutral-mixed NVh term-balance campaign package.")
    parser.add_argument(
        "--rerun-hermes-diagnostics",
        action="store_true",
        help="Run a one-step Hermès neutral_mixed case with output_ddt=true and diagnose=true before packaging.",
    )
    parser.add_argument(
        "--diagnostic-workdir",
        type=Path,
        default=Path("tmp") / "neutral_mixed_hermes_diagnostics",
        help="Work directory used when --rerun-hermes-diagnostics is set.",
    )
    parser.add_argument(
        "--hermes-binary",
        type=Path,
        default=None,
        help="Optional explicit Hermès executable path.",
    )
    args = parser.parse_args()

    reference_root = require_reference_root()
    diagnostic_nc = os.environ.get("JAX_DRB_NEUTRAL_MIXED_HERMES_DIAGNOSTIC_NC")
    if args.rerun_hermes_diagnostics:
        diagnostic_nc = str(
            run_neutral_mixed_hermes_diagnostic_rerun(
                reference_root=reference_root,
                workdir=args.diagnostic_workdir,
                hermes_binary=args.hermes_binary,
            )
        )

    output_root = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "data"
        / "neutral_mixed_term_balance_campaign_artifacts"
    )
    artifacts = create_neutral_mixed_term_balance_campaign_package(
        output_root=output_root,
        reference_root=reference_root,
        hermes_diagnostic_nc=diagnostic_nc,
    )
    print(f"summary: {artifacts.report_json_path}")
    print(f"arrays: {artifacts.report_npz_path}")
    print(f"plot: {artifacts.report_plot_png_path}")


if __name__ == "__main__":
    main()
