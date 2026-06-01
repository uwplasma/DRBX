from __future__ import annotations

from pathlib import Path

from jax_drb.validation import create_autodiff_diffusion_uncertainty_package

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "docs" / "data" / "autodiff_diffusion_uncertainty_artifacts"
SAMPLE_COUNT = 96
RANDOM_SEED = 7


artifacts = create_autodiff_diffusion_uncertainty_package(
    output_root=OUTPUT_ROOT,
    sample_count=SAMPLE_COUNT,
    random_seed=RANDOM_SEED,
)

print("== Autodiff Diffusion Uncertainty ==")
print(f"  - analysis_json: {artifacts.analysis_json_path}")
print(f"  - arrays_npz: {artifacts.arrays_npz_path}")
print(f"  - plot_png: {artifacts.plot_png_path}")
