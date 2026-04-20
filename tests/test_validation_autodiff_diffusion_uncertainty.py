from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import create_autodiff_diffusion_uncertainty_package


def test_create_autodiff_diffusion_uncertainty_package_writes_artifacts(tmp_path: Path) -> None:
    artifacts = create_autodiff_diffusion_uncertainty_package(
        output_root=tmp_path / "output",
        sample_count=32,
        random_seed=3,
    )

    assert artifacts.analysis_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()

    payload = json.loads(artifacts.analysis_json_path.read_text(encoding="utf-8"))
    assert payload["case"] == "autodiff_diffusion_uncertainty"
    assert payload["sample_count"] == 32
    assert payload["scalar_qoi_name"] == "final_density_variance"
    assert payload["linearized_qoi_sigma"] > 0.0
    assert payload["monte_carlo_qoi_sigma"] > 0.0
    assert payload["qoi_sigma_relative_error"] < 0.5
    assert payload["profile_sigma_max_abs_gap"] < 0.05
