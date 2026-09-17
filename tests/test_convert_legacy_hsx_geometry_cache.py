from __future__ import annotations

from pathlib import Path

import pytest

import convert_legacy_hsx_geometry_cache as converter


def test_parser_requires_explicit_cache_inputs() -> None:
    with pytest.raises(SystemExit):
        converter._parser().parse_args([])


def test_conversion_stages_only_named_files_and_delegates(monkeypatch, tmp_path: Path) -> None:
    metric = tmp_path / "old-metric.npz"
    maps = tmp_path / "old-maps.npz"
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "vessel.txt"
    for path in (metric, maps, makegrid, vessel):
        path.write_bytes(path.name.encode())
    output = tmp_path / "artifact"
    observed = {}

    def fake_build(config, *, status_path=None, log_path=None):
        observed["config"] = config
        observed["status"] = status_path
        observed["log"] = log_path
        observed["metric_files"] = sorted(path.name for path in config.metric_cache_dir.iterdir())
        observed["map"] = config.map_cache_path.read_bytes()
        return object()

    monkeypatch.setattr(converter, "build_hsx_simulation_geometry", fake_build)
    result = converter.convert_legacy_cache(
        legacy_metric_cache=metric,
        legacy_map_cache=maps,
        makegrid=makegrid,
        vessel=vessel,
        resolution=(16, 16, 32),
        metric_mesh_shape=(16, 16, 8),
        output=output,
        status=tmp_path / "status.json",
        log=tmp_path / "run.log",
    )
    assert result is not None
    config = observed["config"]
    assert config.resolution == (16, 16, 32)
    assert config.makegrid_path == makegrid.resolve()
    assert config.vessel_path == vessel.resolve()
    assert observed["metric_files"] == ["hsx_metric_legacy_input.npz"]
    assert observed["map"] == maps.read_bytes()
    assert config.map_cache_path.parent != maps.parent
    assert observed["status"] == tmp_path / "status.json"
    assert observed["log"] == tmp_path / "run.log"
    assert metric.read_bytes() == b"old-metric.npz"
    assert maps.read_bytes() == b"old-maps.npz"


def test_conversion_rejects_missing_explicit_file(tmp_path: Path) -> None:
    (tmp_path / "metric.npz").write_bytes(b"metric")
    with pytest.raises(FileNotFoundError, match="legacy map cache"):
        converter.convert_legacy_cache(
            legacy_metric_cache=tmp_path / "metric.npz",
            legacy_map_cache=tmp_path / "maps.npz",
            makegrid=tmp_path / "mgrid.nc",
            vessel=tmp_path / "vessel.txt",
            resolution=(16, 16, 32),
            output=tmp_path / "artifact",
        )
