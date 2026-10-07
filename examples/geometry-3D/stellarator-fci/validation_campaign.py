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

import time
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


total = time.perf_counter()
for name, create in (
    ("geometry", create_stellarator_fci_geometry_campaign_package),
    ("suite", create_stellarator_fci_suite_campaign_package),
    ("operators", create_stellarator_fci_operator_campaign_package),
    ("metric_mms", create_stellarator_metric_mms_campaign_package),
    ("sheath_recycling", create_stellarator_sheath_recycling_campaign_package),
    ("neutral_physics", create_stellarator_neutral_physics_campaign_package),
    ("vorticity", create_stellarator_vorticity_campaign_package),
    ("pytree_drb", create_stellarator_drb_pytree_campaign_package),
    ("showcase", create_stellarator_sol_showcase_package),
):
    print(f"running {name} campaign package...")
    start = time.perf_counter()
    create(output_root=OUTPUT_ROOT / name)
    print(f"  {name} done in {time.perf_counter() - start:.1f} s -> {OUTPUT_ROOT / name}")
print(f"wrote stellarator FCI validation artifacts under {OUTPUT_ROOT} in {time.perf_counter() - total:.1f} s")
