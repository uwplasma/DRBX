from __future__ import annotations

from pathlib import Path

from jax_drb.validation import create_local_cpu_scaling_campaign_package


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    artifacts = create_local_cpu_scaling_campaign_package(
        output_root=repo_root / "docs" / "data" / "local_cpu_scaling_campaign_artifacts",
    )
    print(f"summary: {artifacts.summary_json_path}")
    print(f"plot: {artifacts.summary_plot_png_path}")


if __name__ == "__main__":
    main()
