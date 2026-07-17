"""Regenerate the full synthetic stellarator FCI validation bundle.

The script runs every promoted synthetic-stellarator validation campaign
package through the public ``drbx.validation`` creators: geometry, FCI
suite, operators, metric MMS, sheath/recycling, neutral physics, vorticity,
DRB pytree, and the SOL showcase. Each package writes its own JSON/NPZ/PNG
artifacts and prints progress; everything lands under
``docs/data/stellarator_fci_validation_artifacts/<campaign>`` (relative to the
current working directory).

This regenerates the documentation-gallery artifacts and takes several minutes.

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/stellarator-fci/validation_campaign.py
"""

from __future__ import annotations

from pathlib import Path

from drbx.validation import (
    create_stellarator_fci_geometry_campaign_package,
    create_stellarator_fci_operator_campaign_package,
    create_stellarator_fci_suite_campaign_package,
    create_stellarator_drb_pytree_campaign_package,
    create_stellarator_metric_mms_campaign_package,
    create_stellarator_neutral_physics_campaign_package,
    create_stellarator_sheath_recycling_campaign_package,
    create_stellarator_sol_showcase_package,
    create_stellarator_vorticity_campaign_package,
)

# --- PARAMETERS ------------------------------------------------------------------
OUTPUT_ROOT = Path("docs/data/stellarator_fci_validation_artifacts")  # artifact root (cwd-relative)


print("running geometry campaign package...")
geometry_artifacts = create_stellarator_fci_geometry_campaign_package(output_root=OUTPUT_ROOT / "geometry")
print("running suite campaign package...")
suite_artifacts = create_stellarator_fci_suite_campaign_package(output_root=OUTPUT_ROOT / "suite")
print("running operators campaign package...")
operator_artifacts = create_stellarator_fci_operator_campaign_package(output_root=OUTPUT_ROOT / "operators")
print("running metric_mms campaign package...")
metric_artifacts = create_stellarator_metric_mms_campaign_package(output_root=OUTPUT_ROOT / "metric_mms")
print("running sheath_recycling campaign package...")
sheath_artifacts = create_stellarator_sheath_recycling_campaign_package(output_root=OUTPUT_ROOT / "sheath_recycling")
print("running neutral_physics campaign package...")
neutral_artifacts = create_stellarator_neutral_physics_campaign_package(output_root=OUTPUT_ROOT / "neutral_physics")
print("running vorticity campaign package...")
vorticity_artifacts = create_stellarator_vorticity_campaign_package(output_root=OUTPUT_ROOT / "vorticity")
print("running pytree_drb campaign package...")
pytree_artifacts = create_stellarator_drb_pytree_campaign_package(output_root=OUTPUT_ROOT / "pytree_drb")
print("running showcase campaign package...")
showcase_artifacts = create_stellarator_sol_showcase_package(output_root=OUTPUT_ROOT / "showcase")

print(f"wrote stellarator FCI validation artifacts under {OUTPUT_ROOT}")
