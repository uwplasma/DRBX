from __future__ import annotations

from pathlib import Path

from jax_drb.validation import create_native_stellarator_vmec_selected_field_package

DEFAULT_REFERENCE_EQUILIBRIUM_PATH = Path("/tmp/jax_drb_wout_reference.nc")
DEFAULT_CANDIDATE_EQUILIBRIUM_PATH = Path("/tmp/jax_drb_wout_candidate.nc")
REFERENCE_EQUILIBRIUM_PATH: Path | None = None
CANDIDATE_EQUILIBRIUM_PATH: Path | None = None
OUTPUT_ROOT = Path("docs/data/stellarator_vmec_native_selected_field_artifacts")


def resolve_equilibrium_paths() -> tuple[Path | None, Path | None]:
    reference_equilibrium_path = REFERENCE_EQUILIBRIUM_PATH
    candidate_equilibrium_path = CANDIDATE_EQUILIBRIUM_PATH
    if reference_equilibrium_path is None and DEFAULT_REFERENCE_EQUILIBRIUM_PATH.exists():
        reference_equilibrium_path = DEFAULT_REFERENCE_EQUILIBRIUM_PATH
    if candidate_equilibrium_path is None and DEFAULT_CANDIDATE_EQUILIBRIUM_PATH.exists():
        candidate_equilibrium_path = DEFAULT_CANDIDATE_EQUILIBRIUM_PATH
    return reference_equilibrium_path, candidate_equilibrium_path


reference_equilibrium_path, candidate_equilibrium_path = resolve_equilibrium_paths()
artifacts = create_native_stellarator_vmec_selected_field_package(
    reference_equilibrium_path=reference_equilibrium_path,
    candidate_equilibrium_path=candidate_equilibrium_path,
    output_root=OUTPUT_ROOT,
)

print("== Native Stellarator VMEC Reduced Selected-Field ==")
print(
    f"  - reference_equilibrium_path: "
    f"{reference_equilibrium_path if reference_equilibrium_path is not None else '<synthetic preview>'}"
)
print(
    f"  - candidate_equilibrium_path: "
    f"{candidate_equilibrium_path if candidate_equilibrium_path is not None else ('<materialized candidate>' if reference_equilibrium_path is not None else '<synthetic preview>')}"
)
print("")
print("== Artifacts ==")
print(f"  - parity_json: {artifacts.parity_json_path}")
print(f"  - comparison_json: {artifacts.comparison_json_path}")
print(f"  - observable_report_json: {artifacts.observable_report_json_path}")
print(f"  - runtime_report_json: {artifacts.runtime_report_json_path}")
print(f"  - parity_plot_png: {artifacts.parity_plot_png_path}")
print(f"  - comparison_plot_png: {artifacts.comparison_plot_png_path}")
