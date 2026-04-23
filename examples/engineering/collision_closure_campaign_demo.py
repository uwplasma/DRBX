from __future__ import annotations

from pathlib import Path

from jax_drb.validation import create_collision_closure_campaign_package


if __name__ == "__main__":
    root = Path("docs/data/collision_closure_campaign_artifacts")
    artifacts = create_collision_closure_campaign_package(output_root=root)
    print(artifacts.summary_json_path)
    print(artifacts.arrays_npz_path)
    print(artifacts.plot_png_path)
