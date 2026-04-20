from __future__ import annotations

from typing import Any


def build_geometry_adapter_manifest(
    *,
    case_label: str,
    geometry_family: str,
    benchmark_adapter: str,
    preview_mode: bool,
    artifacts: dict[str, str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "case_label": case_label,
        "geometry_family": geometry_family,
        "benchmark_adapter": benchmark_adapter,
        "preview_mode": preview_mode,
        "artifacts": artifacts,
    }
    if metadata:
        manifest.update(metadata)
    return manifest


def build_geometry_adapter_contract(
    *,
    geometry_family: str,
    benchmark_adapter: str,
    diagnostic_layer: str,
    references: list[dict[str, str]],
    promotion_gates: list[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "geometry_family": geometry_family,
        "benchmark_adapter": benchmark_adapter,
        "diagnostic_layer": diagnostic_layer,
        "references": references,
        "promotion_gates": promotion_gates,
    }
    if metadata:
        contract.update(metadata)
    return contract
