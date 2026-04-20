from __future__ import annotations

from pathlib import Path

from jax_drb.validation import create_jax_native_profile_audit_package


def main() -> None:
    output_root = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "data"
        / "jax_native_profile_audit_artifacts"
    )
    artifacts = create_jax_native_profile_audit_package(output_root=output_root)
    print(f"summary: {artifacts.summary_json_path}")
    print(f"plot: {artifacts.summary_plot_png_path}")
    print(f"traces: {output_root / 'traces'}")


if __name__ == "__main__":
    main()
