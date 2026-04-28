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


def main() -> None:
    output_root = Path("docs/data/stellarator_fci_validation_artifacts")
    create_stellarator_fci_geometry_campaign_package(output_root=output_root / "geometry")
    create_stellarator_fci_suite_campaign_package(output_root=output_root / "suite")
    create_stellarator_fci_operator_campaign_package(output_root=output_root / "operators")
    create_stellarator_metric_mms_campaign_package(output_root=output_root / "metric_mms")
    create_stellarator_sheath_recycling_campaign_package(output_root=output_root / "sheath_recycling")
    create_stellarator_neutral_physics_campaign_package(output_root=output_root / "neutral_physics")
    create_stellarator_vorticity_campaign_package(output_root=output_root / "vorticity")
    create_stellarator_drb_pytree_campaign_package(output_root=output_root / "pytree_drb")
    create_stellarator_sol_showcase_package(output_root=output_root / "showcase")
    print(f"wrote stellarator FCI validation artifacts under {output_root}")


if __name__ == "__main__":
    main()
