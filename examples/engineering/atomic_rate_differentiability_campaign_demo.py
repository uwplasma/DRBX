from __future__ import annotations

from pathlib import Path

from jax_drb.validation import create_atomic_rate_differentiability_campaign_package


def main() -> None:
    output_root = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "data"
        / "atomic_rate_differentiability_campaign_artifacts"
    )
    artifacts = create_atomic_rate_differentiability_campaign_package(output_root=output_root)
    print(f"summary: {artifacts.summary_json_path}")
    print(f"arrays: {artifacts.arrays_npz_path}")
    print(f"plot: {artifacts.plot_png_path}")


if __name__ == "__main__":
    main()
