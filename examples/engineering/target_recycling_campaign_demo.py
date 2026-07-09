from __future__ import annotations

from pathlib import Path

from jax_drb.validation import create_target_recycling_campaign_package


if __name__ == "__main__":
    root = Path("docs/data/target_recycling_campaign_artifacts")
    artifacts = create_target_recycling_campaign_package(output_root='/pscratch/sd/y/yiqunx/tmp/target_recycling_demo')
    print(artifacts.summary_json_path)
    print(artifacts.arrays_npz_path)
    print(artifacts.plot_png_path)
