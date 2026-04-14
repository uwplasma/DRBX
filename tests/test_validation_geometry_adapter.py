from __future__ import annotations

from jax_drb.validation import build_geometry_adapter_contract, build_geometry_adapter_manifest


def test_build_geometry_adapter_manifest_keeps_shared_fields() -> None:
    manifest = build_geometry_adapter_manifest(
        case_label="demo",
        geometry_family="traced_field_line_3d",
        benchmark_adapter="demo_adapter",
        preview_mode=True,
        artifacts={"plot_png": "images/demo.png"},
        metadata={"extra": 3},
    )
    assert manifest["case_label"] == "demo"
    assert manifest["geometry_family"] == "traced_field_line_3d"
    assert manifest["benchmark_adapter"] == "demo_adapter"
    assert manifest["preview_mode"] is True
    assert manifest["artifacts"]["plot_png"] == "images/demo.png"
    assert manifest["extra"] == 3


def test_build_geometry_adapter_contract_keeps_shared_fields() -> None:
    contract = build_geometry_adapter_contract(
        geometry_family="diverted_tokamak_3d",
        benchmark_adapter="tcv_x21",
        diagnostic_layer="benchmark_adapter_on_general_3d_geometry",
        references=[{"label": "ref", "url": "https://example.com"}],
        promotion_gates=["scaffold", "native"],
        metadata={"metric_checks": ["finite"]},
    )
    assert contract["geometry_family"] == "diverted_tokamak_3d"
    assert contract["benchmark_adapter"] == "tcv_x21"
    assert contract["diagnostic_layer"] == "benchmark_adapter_on_general_3d_geometry"
    assert contract["references"][0]["label"] == "ref"
    assert contract["promotion_gates"] == ["scaffold", "native"]
    assert contract["metric_checks"] == ["finite"]
