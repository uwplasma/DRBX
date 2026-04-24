from __future__ import annotations

from pathlib import Path

from jax_drb.validation import create_implicit_solver_profile_audit_package


def main() -> None:
    output_root = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "data"
        / "implicit_solver_profile_audit_artifacts"
    )
    artifacts = create_implicit_solver_profile_audit_package(output_root=output_root)
    print(f"summary: {artifacts.report_json_path}")
    print(f"plot: {artifacts.report_plot_png_path}")


if __name__ == "__main__":
    main()
