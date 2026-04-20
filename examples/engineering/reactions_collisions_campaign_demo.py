from __future__ import annotations

from pathlib import Path

from jax_drb.validation import create_reactions_collisions_campaign_package


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    artifacts = create_reactions_collisions_campaign_package(
        output_root=repo_root / "docs" / "data" / "reactions_collisions_campaign_artifacts",
    )
    print(f"summary: {artifacts.summary_json_path}")
    print(f"arrays: {artifacts.arrays_npz_path}")
    print(f"plot: {artifacts.plot_png_path}")


if __name__ == "__main__":
    main()
