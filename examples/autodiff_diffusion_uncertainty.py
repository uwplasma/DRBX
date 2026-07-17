"""Uncertainty quantification on the differentiable diffusion lane.

The script propagates a Gaussian uncertainty on the physical diffusion
parameters through the evolved final state in two complementary ways:

1. a linearized covariance pushforward built from the autodiff Jacobian, and
2. a vectorized (``vmap``) Monte Carlo ensemble of full nonlinear rollouts.

Both are produced by the public ``create_autodiff_diffusion_uncertainty_package``
API, which writes the analysis JSON, the sampled arrays NPZ, and the summary
figure under ``OUTPUT_ROOT`` (``data/autodiff_diffusion_uncertainty_analysis.json``,
``data/autodiff_diffusion_uncertainty_arrays.npz``, and
``images/autodiff_diffusion_uncertainty.png``, relative to the current working
directory) and prints the artifact paths.

Run from the repository root:

    PYTHONPATH=src python examples/autodiff_diffusion_uncertainty.py
"""

from __future__ import annotations

from pathlib import Path

from dkx.validation import create_autodiff_diffusion_uncertainty_package

# --- PARAMETERS ------------------------------------------------------------------
OUTPUT_ROOT = Path("docs/data/autodiff_diffusion_uncertainty_artifacts")  # artifact root (cwd-relative)
SAMPLE_COUNT = 96   # Monte Carlo ensemble size; raise for smoother statistics
RANDOM_SEED = 7     # PRNG seed for the parameter samples

# --- run the uncertainty package --------------------------------------------------
print("running the autodiff diffusion uncertainty package "
      f"({SAMPLE_COUNT} Monte Carlo samples, seed {RANDOM_SEED})...")
artifacts = create_autodiff_diffusion_uncertainty_package(
    output_root=OUTPUT_ROOT,
    sample_count=SAMPLE_COUNT,
    random_seed=RANDOM_SEED,
)

print("== Autodiff Diffusion Uncertainty ==")
print(f"  - analysis_json: {artifacts.analysis_json_path}")
print(f"  - arrays_npz: {artifacts.arrays_npz_path}")
print(f"  - plot_png: {artifacts.plot_png_path}")
