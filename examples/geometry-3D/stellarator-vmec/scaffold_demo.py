from __future__ import annotations

from pathlib import Path

from jax_drb.validation import create_stellarator_vmec_scaffold_package

OUTPUT_ROOT = Path("docs/data/stellarator_vmec_scaffold_artifacts")
EQUILIBRIUM_PATH: Path | None = None


artifacts = create_stellarator_vmec_scaffold_package(
    output_root=OUTPUT_ROOT,
    equilibrium_path=EQUILIBRIUM_PATH,
)

print(f"manifest: {artifacts.manifest_json_path}")
print(f"input report: {artifacts.input_report_json_path}")
print(f"validation contract: {artifacts.validation_contract_json_path}")
print(f"profile report: {artifacts.profile_report_json_path}")
print(f"profile arrays: {artifacts.profile_arrays_npz_path}")
print(f"profile plot: {artifacts.profile_plot_png_path}")
print(f"surface report: {artifacts.surface_report_json_path}")
print(f"surface arrays: {artifacts.surface_arrays_npz_path}")
print(f"surface plot: {artifacts.surface_plot_png_path}")
print(f"surface movie: {artifacts.surface_gif_path}")
print(f"observable report: {artifacts.observable_report_json_path}")
