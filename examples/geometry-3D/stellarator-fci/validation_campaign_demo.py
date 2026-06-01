from __future__ import annotations

from pathlib import Path

from jax_drb.validation import (
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

OUTPUT_ROOT = Path("docs/data/stellarator_fci_validation_artifacts")


geometry_artifacts = create_stellarator_fci_geometry_campaign_package(output_root=OUTPUT_ROOT / "geometry")
suite_artifacts = create_stellarator_fci_suite_campaign_package(output_root=OUTPUT_ROOT / "suite")
operator_artifacts = create_stellarator_fci_operator_campaign_package(output_root=OUTPUT_ROOT / "operators")
metric_artifacts = create_stellarator_metric_mms_campaign_package(output_root=OUTPUT_ROOT / "metric_mms")
sheath_artifacts = create_stellarator_sheath_recycling_campaign_package(output_root=OUTPUT_ROOT / "sheath_recycling")
neutral_artifacts = create_stellarator_neutral_physics_campaign_package(output_root=OUTPUT_ROOT / "neutral_physics")
vorticity_artifacts = create_stellarator_vorticity_campaign_package(output_root=OUTPUT_ROOT / "vorticity")
pytree_artifacts = create_stellarator_drb_pytree_campaign_package(output_root=OUTPUT_ROOT / "pytree_drb")
showcase_artifacts = create_stellarator_sol_showcase_package(output_root=OUTPUT_ROOT / "showcase")

print(f"wrote stellarator FCI validation artifacts under {OUTPUT_ROOT}")
