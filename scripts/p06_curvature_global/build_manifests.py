#!/usr/bin/env python3
"""Regenerate portable P06 source and immutable-input manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / "scripts/hsx_remote_qualification"


def record(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": relative, "bytes": path.stat().st_size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO.parent,
        help="workspace root containing the immutable paths in input_manifest.json",
    )
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    old_inputs = json.loads((BASE / "input_manifest.json").read_text())
    input_paths = sorted(
        item["path"]
        for item in old_inputs["files"]
        if not item["path"].startswith("work/perpendicular_bracket_reference_requalification_")
    )
    inputs = {
        "schema": "drbx.p06-curvature-input-manifest-v1",
        "files": [record(input_root, path) | {"hash_verification": "computed-full"} for path in input_paths],
        "notes": "Actual HSX geometry, original midpoint-owner baseline, continuous-reference sidecar, metric cache and MAKEGRID only; no archived candidate answers.",
    }
    (HERE / "input_manifest.json").write_text(json.dumps(inputs, indent=2, sort_keys=True)+"\n")

    old_sources = json.loads((BASE / "source_manifest.json").read_text())
    source_paths = {
        item["path"] for item in old_sources["files"]
        if not item["path"].startswith("scripts/hsx_remote_qualification/")
    }
    source_paths.update({
        "scripts/hsx_remote_qualification/numerics.py",
        "scripts/hsx_remote_qualification/selection.py",
        "scripts/p06_curvature_global/campaign.py",
        "scripts/p06_curvature_global/numerics.py",
        "scripts/p06_curvature_global/parallel_runner.py",
        "scripts/p06_curvature_global/replay_bounded.py",
        "scripts/p06_curvature_global/selection.py",
        "scripts/p06_curvature_global/build_manifests.py",
        "src/drbx/geometry/fci_boundary_functional_reconstruction.py",
        "src/drbx/geometry/fci_perpendicular_bracket.py",
        "src/drbx/native/fci_curvature_production_flux.py",
        "src/drbx/native/fci_operators.py",
    })
    missing = [path for path in sorted(source_paths) if not (REPO/path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing source dependencies: {missing}")
    sources = {
        "schema": "drbx.p06-curvature-source-manifest-v1",
        "scope": "Transitive local Python dependencies plus all P06 campaign entry points; regenerate after any source edit.",
        "files": [record(REPO, path) for path in sorted(source_paths)],
    }
    (HERE / "source_manifest.json").write_text(json.dumps(sources, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
