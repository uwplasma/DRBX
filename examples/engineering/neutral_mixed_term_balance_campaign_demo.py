from __future__ import annotations

import os
from pathlib import Path

from jax_drb.reference.paths import require_reference_root
from jax_drb.validation import create_neutral_mixed_term_balance_campaign_package


def main() -> None:
    output_root = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "data"
        / "neutral_mixed_term_balance_campaign_artifacts"
    )
    artifacts = create_neutral_mixed_term_balance_campaign_package(
        output_root=output_root,
        reference_root=require_reference_root(),
        hermes_diagnostic_nc=os.environ.get("JAX_DRB_NEUTRAL_MIXED_HERMES_DIAGNOSTIC_NC"),
    )
    print(f"summary: {artifacts.report_json_path}")
    print(f"arrays: {artifacts.report_npz_path}")
    print(f"plot: {artifacts.report_plot_png_path}")


if __name__ == "__main__":
    main()
