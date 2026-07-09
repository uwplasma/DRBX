from __future__ import annotations

from pathlib import Path

from jax_drb.validation import create_fluid_1d_mms_convergence_package


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    
    artifacts = create_fluid_1d_mms_convergence_package(
        output_root=Path("/pscratch/sd/y/yiqunx/tmp") / "fluid_1d_mms_convergence_artifacts",
    )
    print(f"summary: {artifacts.summary_json_path}")
    print(f"arrays: {artifacts.arrays_npz_path}")
    print(f"plot: {artifacts.summary_plot_png_path}")


if __name__ == "__main__":
    main()
