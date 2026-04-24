from __future__ import annotations

from pathlib import Path

from jax_drb.reference.paths import default_reference_root, repo_root
from jax_drb.validation import create_tokamak_recycling_observable_campaign_package


def main() -> None:
    root = repo_root()
    reference_root = default_reference_root()
    artifacts = create_tokamak_recycling_observable_campaign_package(
        output_root=root / "docs" / "data" / "tokamak_recycling_observable_campaign_artifacts",
        reference_root=reference_root,
    )
    print("Tokamak recycling observable campaign artifacts:")
    print(f"  - report_json: {Path(artifacts.report_json_path)}")
    print(f"  - report_npz: {Path(artifacts.report_npz_path)}")
    print(f"  - report_png: {Path(artifacts.report_plot_png_path)}")


if __name__ == "__main__":
    main()
