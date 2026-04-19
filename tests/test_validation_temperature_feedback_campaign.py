from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np
import pytest

from jax_drb.validation import (
    create_temperature_feedback_campaign_package,
)
from jax_drb.validation.temperature_feedback_campaign import (
    _TemperatureFeedbackSeries,
    _build_patched_temperature_feedback_reference_binary,
    _build_temperature_feedback_series,
    _git_stdout,
    _patched_temperature_feedback_header_text,
    _prepare_temperature_feedback_reference_binary,
    _temperature_feedback_header_has_known_permission_bug,
    _extract_scalar_series,
    _extract_target_temperature,
    _extract_time_points,
    _extract_spatial_series,
    _reconstruct_temperature_controller,
    _replace_bout_setting,
    _run_temperature_feedback_example,
    _stage_temperature_feedback_example,
    _strip_solver_option_lines,
    build_temperature_feedback_campaign,
)


def test_reconstruct_temperature_controller_matches_trapezoid_pi_update() -> None:
    time_points = np.asarray([0.0, 1.0, 3.0], dtype=np.float64)
    error = np.asarray([2.0, 1.0, -1.0], dtype=np.float64)

    integral_state, proportional_term, integral_term, multiplier = _reconstruct_temperature_controller(
        time_points=time_points,
        error=error,
        proportional_gain=10.0,
        integral_gain=0.5,
        integral_positive=False,
        source_positive=True,
    )

    np.testing.assert_allclose(integral_state, np.asarray([0.0, 1.5, 1.5], dtype=np.float64))
    np.testing.assert_allclose(proportional_term, np.asarray([20.0, 10.0, -10.0], dtype=np.float64))
    np.testing.assert_allclose(integral_term, np.asarray([0.0, 0.75, 0.75], dtype=np.float64))
    np.testing.assert_allclose(multiplier, np.asarray([20.0, 10.75, 0.0], dtype=np.float64))


def test_create_temperature_feedback_campaign_package_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign.build_temperature_feedback_campaign",
        lambda **kwargs: {
            "summary": {
                "family": "temperature_feedback",
                "metric_count": 1,
                "passed_metric_count": 1,
                "metrics": [
                    {
                        "name": "temperature_feedback_src_mult_e_exact",
                        "kind": "max_abs_error",
                        "value": 1.0e-12,
                        "target": 1.0e-12,
                        "passed": True,
                        "notes": "demo",
                    }
                ],
            },
            "series": type(
                "Series",
                (),
                {
                    "time_points": np.asarray([0.0, 1.0], dtype=np.float64),
                    "target_temperature": np.asarray([0.1, 0.2], dtype=np.float64),
                    "setpoint": 0.15,
                    "reference_multiplier": np.asarray([1.0, 2.0], dtype=np.float64),
                    "reconstructed_multiplier": np.asarray([1.0, 2.0], dtype=np.float64),
                    "reference_proportional": np.asarray([1.0, 2.0], dtype=np.float64),
                    "reconstructed_proportional": np.asarray([1.0, 2.0], dtype=np.float64),
                    "reference_integral": np.asarray([0.0, 0.5], dtype=np.float64),
                    "reconstructed_integral": np.asarray([0.0, 0.5], dtype=np.float64),
                    "reference_integral_state": np.asarray([0.0, 50.0], dtype=np.float64),
                    "reconstructed_integral_state": np.asarray([0.0, 50.0], dtype=np.float64),
                    "reference_energy_source": np.asarray([[[[1.0]]], [[[2.0]]]], dtype=np.float64),
                    "reconstructed_energy_source": np.asarray([[[[1.0]]], [[[2.0]]]], dtype=np.float64),
                },
            )(),
        },
    )

    artifacts = create_temperature_feedback_campaign_package(
        output_root=tmp_path / "output",
        reference_root=tmp_path,
    )

    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["family"] == "temperature_feedback"
    assert payload["passed_metric_count"] == 1


def test_replace_bout_setting_handles_numeric_values_without_backreference_bug() -> None:
    text = "nout = 400\nny = 80\n"

    updated = _replace_bout_setting(text, "nout", "4")
    updated = _replace_bout_setting(updated, "ny", "20")

    assert "nout = 4\n" in updated
    assert "ny = 20\n" in updated


def test_run_temperature_feedback_example_streams_to_file_without_capture_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    invoked: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        invoked.update(kwargs)
        stdout = kwargs["stdout"]
        stdout.write("controller run\n")
        (Path(kwargs["cwd"]) / "BOUT.dmp.0.nc").write_bytes(b"stub")
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    monkeypatch.setattr("jax_drb.validation.temperature_feedback_campaign.subprocess.run", fake_run)

    _run_temperature_feedback_example(
        binary=tmp_path / "hermes",
        workdir=tmp_path,
        timeout_seconds=30,
    )

    assert invoked["stderr"] == subprocess.STDOUT
    assert invoked["timeout"] == 30
    assert "capture_output" not in invoked
    assert (tmp_path / "run.stdout").read_text(encoding="utf-8") == "controller run\n"


def test_build_temperature_feedback_campaign_maps_series_to_summary(monkeypatch) -> None:
    series = _TemperatureFeedbackSeries(
        time_points=np.asarray([0.0, 1.0], dtype=np.float64),
        target_temperature=np.asarray([0.5, 0.75], dtype=np.float64),
        setpoint=1.0,
        reference_multiplier=np.asarray([1.0, 1.2], dtype=np.float64),
        reconstructed_multiplier=np.asarray([1.0, 1.2], dtype=np.float64),
        reference_proportional=np.asarray([0.2, 0.1], dtype=np.float64),
        reconstructed_proportional=np.asarray([0.2, 0.1], dtype=np.float64),
        reference_integral=np.asarray([0.0, 0.05], dtype=np.float64),
        reconstructed_integral=np.asarray([0.0, 0.05], dtype=np.float64),
        reference_integral_state=np.asarray([0.0, 1.0], dtype=np.float64),
        reconstructed_integral_state=np.asarray([0.0, 1.0], dtype=np.float64),
        reference_energy_source=np.asarray([[[[1.0]]], [[[1.2]]]], dtype=np.float64),
        reconstructed_energy_source=np.asarray([[[[1.0]]], [[[1.2]]]], dtype=np.float64),
    )
    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign._build_temperature_feedback_series",
        lambda **kwargs: (series, {"total": 1.0}, {"mode": "explicit_reference_binary"}),
    )

    report = build_temperature_feedback_campaign(reference_root="/tmp")

    assert report["summary"]["family"] == "temperature_feedback"
    assert report["summary"]["passed_metric_count"] == report["summary"]["metric_count"] == 5
    assert report["summary"]["reference_provenance"] == {"mode": "explicit_reference_binary"}


def test_build_temperature_feedback_campaign_adds_integral_metric_when_diagnostic_is_exported(monkeypatch) -> None:
    series = _TemperatureFeedbackSeries(
        time_points=np.asarray([0.0, 1.0], dtype=np.float64),
        target_temperature=np.asarray([0.5, 0.5], dtype=np.float64),
        setpoint=0.5,
        reference_multiplier=np.asarray([1.0, 1.0], dtype=np.float64),
        reconstructed_multiplier=np.asarray([1.0, 1.0], dtype=np.float64),
        reference_proportional=np.asarray([0.2, 0.2], dtype=np.float64),
        reconstructed_proportional=np.asarray([0.2, 0.2], dtype=np.float64),
        reference_integral=np.asarray([0.0, 0.0], dtype=np.float64),
        reconstructed_integral=np.asarray([0.0, 0.0], dtype=np.float64),
        reference_integral_state=np.asarray([0.0, 1.0], dtype=np.float64),
        reconstructed_integral_state=np.asarray([0.0, 1.0], dtype=np.float64),
        reference_energy_source=np.asarray([[[[1.0]]], [[[1.0]]]], dtype=np.float64),
        reconstructed_energy_source=np.asarray([[[[1.0]]], [[[1.0]]]], dtype=np.float64),
    )
    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign._build_temperature_feedback_series",
        lambda **kwargs: (
            series,
            {"total": 1.0},
            {"mode": "explicit_reference_binary", "integral_state_reference_mode": "diagnostic_export"},
        ),
    )

    report = build_temperature_feedback_campaign(reference_root="/tmp")

    assert report["summary"]["metric_count"] == 6
    assert any(metric["name"] == "e_temperature_error_integral_exact" for metric in report["summary"]["metrics"])


def test_temperature_feedback_header_bug_detection_and_patch() -> None:
    original = (
        "    std::vector<std::string> species_stripped;\n"
        "    std::transform(species_list.begin(), species_list.end(), species_stripped.begin(),\n"
        "                   [](const std::string& val) { return trim(val); });\n"
    )

    assert _temperature_feedback_header_has_known_permission_bug(original) is True
    patched = _patched_temperature_feedback_header_text(original)
    assert "std::back_inserter(species_stripped)" in patched
    assert "reserve(species_list.size())" in patched
    assert _temperature_feedback_header_has_known_permission_bug(patched) is False
    assert _patched_temperature_feedback_header_text("clean header\n") == "clean header\n"


def test_prepare_temperature_feedback_reference_binary_uses_explicit_binary(tmp_path: Path) -> None:
    binary, provenance = _prepare_temperature_feedback_reference_binary(
        reference_root=tmp_path,
        reference_binary=tmp_path / "hermes-3",
    )

    assert binary == tmp_path / "hermes-3"
    assert provenance["mode"] == "explicit_reference_binary"
    assert provenance["temperature_feedback_permission_fix"] == "explicit_binary"


def test_prepare_temperature_feedback_reference_binary_handles_missing_and_clean_headers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    binary_path = tmp_path / "hermes-3"
    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign.discover_reference_binary",
        lambda **kwargs: binary_path,
    )

    binary, provenance = _prepare_temperature_feedback_reference_binary(
        reference_root=tmp_path,
        reference_binary=None,
    )
    assert binary == binary_path
    assert provenance["temperature_feedback_permission_fix"] == "not_checked"

    include_dir = tmp_path / "include"
    include_dir.mkdir()
    (include_dir / "temperature_feedback.hxx").write_text("clean header\n", encoding="utf-8")
    binary, provenance = _prepare_temperature_feedback_reference_binary(
        reference_root=tmp_path,
        reference_binary=None,
    )
    assert binary == binary_path
    assert provenance["temperature_feedback_permission_fix"] == "not_needed"


def test_prepare_temperature_feedback_reference_binary_uses_patched_builder_for_known_bug(
    tmp_path: Path,
    monkeypatch,
) -> None:
    binary_path = tmp_path / "hermes-3"
    cache_root = tmp_path / "cache"
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    (include_dir / "temperature_feedback.hxx").write_text(
        "    std::vector<std::string> species_stripped;\n"
        "    std::transform(species_list.begin(), species_list.end(), species_stripped.begin(),\n"
        "                   [](const std::string& val) { return trim(val); });\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign.discover_reference_binary",
        lambda **kwargs: tmp_path / "local-hermes",
    )
    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign._build_patched_temperature_feedback_reference_binary",
        lambda reference_root: (binary_path, cache_root),
    )

    binary, provenance = _prepare_temperature_feedback_reference_binary(
        reference_root=tmp_path,
        reference_binary=None,
    )

    assert binary == binary_path
    assert provenance["mode"] == "auto_patched_clean_reference_worktree"
    assert provenance["patch_cache_root"] == str(cache_root)


def test_build_patched_temperature_feedback_reference_binary_uses_existing_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "jax_drb_temperature_feedback_reference" / "deadbeef"
    source_root = cache_root / "src"
    build_root = cache_root / "build"
    source_root.mkdir(parents=True)
    build_root.mkdir(parents=True)
    binary_path = build_root / "hermes-3"
    binary_path.write_text("binary", encoding="utf-8")

    reference_root = tmp_path / "reference"
    reference_root.mkdir()

    def _fake_git_stdout(root: Path, *args: str) -> str:
        return "deadbeef"

    def _fake_tempdir() -> str:
        return str(tmp_path)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("jax_drb.validation.temperature_feedback_campaign._git_stdout", _fake_git_stdout)
    monkeypatch.setattr("jax_drb.validation.temperature_feedback_campaign.tempfile.gettempdir", _fake_tempdir)
    try:
        returned_binary, returned_cache = _build_patched_temperature_feedback_reference_binary(reference_root)
    finally:
        monkeypatch.undo()

    assert returned_binary == binary_path
    assert returned_cache == cache_root


def test_build_patched_temperature_feedback_reference_binary_builds_clean_worktree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reference_root = tmp_path / "reference"
    include_dir = reference_root / "include"
    include_dir.mkdir(parents=True)
    (include_dir / "temperature_feedback.hxx").write_text(
        "    std::vector<std::string> species_stripped;\n"
        "    std::transform(species_list.begin(), species_list.end(), species_stripped.begin(),\n"
        "                   [](const std::string& val) { return trim(val); });\n",
        encoding="utf-8",
    )

    cache_root = tmp_path / "jax_drb_temperature_feedback_reference" / "deadbeef"
    source_root = cache_root / "src"
    build_root = cache_root / "build"
    calls: list[list[str]] = []

    def _fake_git_stdout(root: Path, *args: str) -> str:
        return "deadbeef"

    def _fake_tempdir() -> str:
        return str(tmp_path)

    def _fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:5] == ["git", "-C", str(reference_root), "worktree", "add"]:
            (source_root / "include").mkdir(parents=True, exist_ok=True)
            (source_root / "include" / "temperature_feedback.hxx").write_text(
                (reference_root / "include" / "temperature_feedback.hxx").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        elif args[:2] == ["cmake", "--build"]:
            build_root.mkdir(parents=True, exist_ok=True)
            (build_root / "hermes-3").write_text("binary", encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr("jax_drb.validation.temperature_feedback_campaign._git_stdout", _fake_git_stdout)
    monkeypatch.setattr("jax_drb.validation.temperature_feedback_campaign.tempfile.gettempdir", _fake_tempdir)
    monkeypatch.setattr("jax_drb.validation.temperature_feedback_campaign.subprocess.run", _fake_run)

    binary, returned_cache = _build_patched_temperature_feedback_reference_binary(reference_root)

    assert binary == build_root / "hermes-3"
    assert returned_cache == cache_root
    patched_text = (source_root / "include" / "temperature_feedback.hxx").read_text(encoding="utf-8")
    assert "std::back_inserter(species_stripped)" in patched_text
    assert any(command[:2] == ["cmake", "-S"] for command in calls)
    assert any(command[:2] == ["cmake", "--build"] for command in calls)


def test_git_stdout_returns_stripped_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=0, stdout="abc123\n"),
    )

    assert _git_stdout(tmp_path, "rev-parse", "HEAD") == "abc123"


def test_stage_temperature_feedback_example_rewrites_input(tmp_path: Path) -> None:
    example_dir = tmp_path / "example"
    example_dir.mkdir()
    (example_dir / "BOUT.inp").write_text(
        "nout = 40\ntimestep = 5\nny = 80\ntype = beuler\nsnes_type = newtonls\nksp_type = gmres\n",
        encoding="utf-8",
    )
    (example_dir / "extra.dat").write_text("payload", encoding="utf-8")
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    _stage_temperature_feedback_example(
        example_dir,
        workdir=workdir,
        nout=4,
        timestep=100.0,
        ny=16,
        solver_type="cvode",
    )

    updated = (workdir / "BOUT.inp").read_text(encoding="utf-8")
    assert "nout = 4" in updated
    assert "timestep = 100" in updated
    assert "ny = 16" in updated
    assert "type = cvode" in updated
    assert "snes_type" not in updated
    assert "ksp_type" not in updated
    assert (workdir / "extra.dat").read_text(encoding="utf-8") == "payload"


def test_stage_temperature_feedback_example_keeps_beuler_specific_options(tmp_path: Path) -> None:
    example_dir = tmp_path / "example"
    example_dir.mkdir()
    (example_dir / "BOUT.inp").write_text(
        "nout = 40\ntimestep = 5\nny = 80\ntype = beuler\nsnes_type = newtonls\n",
        encoding="utf-8",
    )
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    _stage_temperature_feedback_example(
        example_dir,
        workdir=workdir,
        nout=2,
        timestep=50.0,
        ny=8,
        solver_type="beuler",
    )

    updated = (workdir / "BOUT.inp").read_text(encoding="utf-8")
    assert "type = beuler" in updated
    assert "snes_type = newtonls" in updated


def test_strip_solver_option_lines_removes_beuler_only_options() -> None:
    text = "type = cvode\nsnes_type = newtonls\nksp_type = gmres\nlag_jacobian = 500\n"

    updated = _strip_solver_option_lines(text, ("snes_type", "ksp_type", "lag_jacobian"))

    assert "snes_type" not in updated
    assert "ksp_type" not in updated
    assert "lag_jacobian" not in updated
    assert "type = cvode" in updated


def test_extract_spatial_series_broadcasts_static_scalar() -> None:
    class _Variable:
        dimensions = ()

        def __getitem__(self, key):
            return np.asarray(3.5, dtype=np.float64)

    class _Dataset:
        variables = {"sample": _Variable()}

    extracted = _extract_spatial_series(_Dataset(), "sample", time_count=3)

    assert extracted.shape == (3, 1, 1, 1)
    np.testing.assert_allclose(extracted[:, 0, 0, 0], np.asarray([3.5, 3.5, 3.5], dtype=np.float64))


def test_extract_target_temperature_uses_first_active_boundary_cell_not_guard() -> None:
    class _Variable:
        def __init__(self, values, dimensions):
            self._values = np.asarray(values, dtype=np.float64)
            self.dimensions = dimensions

        def __getitem__(self, key):
            return self._values

    class _Dataset:
        def __init__(self):
            te = np.asarray(
                [
                    [[[0.0], [0.9], [0.8], [0.0]]],
                    [[[0.0], [0.85], [0.75], [0.0]]],
                ],
                dtype=np.float64,
            )
            self.variables = {
                "t_array": _Variable(np.asarray([0.0, 1.0], dtype=np.float64), ("t",)),
                "Te": _Variable(te, ("t", "x", "y", "z")),
            }

    dataset = _Dataset()

    upstream = _extract_target_temperature(dataset, control_target=False)
    target = _extract_target_temperature(dataset, control_target=True)

    np.testing.assert_allclose(upstream, np.asarray([0.9, 0.85], dtype=np.float64))
    np.testing.assert_allclose(target, np.asarray([0.8, 0.75], dtype=np.float64))


def test_replace_bout_setting_raises_when_key_is_missing() -> None:
    with pytest.raises(ValueError, match="Could not replace"):
        _replace_bout_setting("nout = 10\n", "ny", "8")


def test_run_temperature_feedback_example_raises_on_timeout_nonzero_and_missing_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="hermes", timeout=30)

    monkeypatch.setattr("jax_drb.validation.temperature_feedback_campaign.subprocess.run", _raise_timeout)
    with pytest.raises(RuntimeError, match="did not finish within 30s"):
        _run_temperature_feedback_example(binary=tmp_path / "hermes", workdir=tmp_path, timeout_seconds=30)

    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=1),
    )
    with pytest.raises(RuntimeError, match="failed with exit code 1"):
        _run_temperature_feedback_example(binary=tmp_path / "hermes", workdir=tmp_path, timeout_seconds=30)

    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=0),
    )
    with pytest.raises(FileNotFoundError, match="did not produce BOUT.dmp.0.nc"):
        _run_temperature_feedback_example(binary=tmp_path / "hermes", workdir=tmp_path, timeout_seconds=30)


def test_extract_temperature_helpers_cover_fallback_shapes() -> None:
    class _Variable:
        def __init__(self, values, dimensions):
            self._values = np.asarray(values, dtype=np.float64)
            self.dimensions = dimensions

        def __getitem__(self, key):
            return self._values

    dataset = SimpleNamespace(
        variables={
            "t": _Variable([0.0, 1.0], ("t",)),
            "Te": _Variable([[[[1.0], [2.0]]], [[[3.0], [4.0]]]], ("t", "x", "y", "z")),
            "scalar2d": _Variable([[1.0], [2.0]], ("t", "x")),
        }
    )

    np.testing.assert_allclose(_extract_time_points(dataset), np.asarray([0.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(_extract_target_temperature(dataset, control_target=True), np.asarray([2.0, 4.0], dtype=np.float64))
    np.testing.assert_allclose(_extract_scalar_series(dataset, "scalar2d"), np.asarray([1.0, 2.0], dtype=np.float64))

    spatial_dataset = SimpleNamespace(variables={"sample": _Variable([[1.0, 2.0], [3.0, 4.0]], ("t", "y"))})
    extracted = _extract_spatial_series(spatial_dataset, "sample", time_count=2)
    assert extracted.shape == (2, 1, 2, 1)

    missing_time = SimpleNamespace(variables={})
    with pytest.raises(KeyError, match="missing time coordinate"):
        _extract_time_points(missing_time)


def test_extract_spatial_series_covers_remaining_shapes_and_errors() -> None:
    class _Variable:
        def __init__(self, values, dimensions):
            self._values = np.asarray(values, dtype=np.float64)
            self.dimensions = dimensions

        def __getitem__(self, key):
            return self._values

    timed_3d = SimpleNamespace(variables={"sample": _Variable(np.ones((2, 3, 4), dtype=np.float64), ("t", "y", "z"))})
    assert _extract_spatial_series(timed_3d, "sample", time_count=2).shape == (2, 1, 3, 4)

    timed_1d = SimpleNamespace(variables={"sample": _Variable(np.asarray([1.0, 2.0]), ("t",))})
    assert _extract_spatial_series(timed_1d, "sample", time_count=2).shape == (2, 1, 1, 1)

    timed_2d = SimpleNamespace(variables={"sample": _Variable(np.ones((2, 3), dtype=np.float64), ("t", "y"))})
    assert _extract_spatial_series(timed_2d, "sample", time_count=2).shape == (2, 1, 3, 1)

    static_3d = SimpleNamespace(variables={"sample": _Variable(np.ones((2, 3, 4), dtype=np.float64), ("x", "y", "z"))})
    assert _extract_spatial_series(static_3d, "sample", time_count=5).shape == (5, 2, 3, 4)

    static_2d = SimpleNamespace(variables={"sample": _Variable(np.ones((2, 3), dtype=np.float64), ("x", "y"))})
    assert _extract_spatial_series(static_2d, "sample", time_count=4).shape == (4, 1, 2, 3)

    static_1d = SimpleNamespace(variables={"sample": _Variable(np.asarray([1.0, 2.0]), ("y",))})
    assert _extract_spatial_series(static_1d, "sample", time_count=3).shape == (3, 1, 2, 1)

    bad = SimpleNamespace(variables={"sample": _Variable(np.ones((2, 2, 2, 2, 2), dtype=np.float64), ("t", "a", "b", "c", "d"))})
    with pytest.raises(ValueError, match="Unsupported variable shape"):
        _extract_spatial_series(bad, "sample", time_count=2)


def test_reconstruct_temperature_controller_validates_shape_and_positive_clamps() -> None:
    with pytest.raises(ValueError, match="matching shape"):
        _reconstruct_temperature_controller(
            time_points=np.asarray([0.0, 1.0], dtype=np.float64),
            error=np.asarray([1.0], dtype=np.float64),
            proportional_gain=1.0,
            integral_gain=1.0,
            integral_positive=False,
            source_positive=False,
        )

    integral_state, proportional_term, integral_term, multiplier = _reconstruct_temperature_controller(
        time_points=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        error=np.asarray([-1.0, -1.0, 1.0], dtype=np.float64),
        proportional_gain=1.0,
        integral_gain=1.0,
        integral_positive=True,
        source_positive=True,
    )

    assert integral_state[1] == 0.0
    assert multiplier[0] == 0.0
    assert proportional_term[-1] == 1.0
    assert integral_term[-1] >= 0.0


def test_build_temperature_feedback_series_loads_staged_reference_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    example_dir = tmp_path / "examples" / "tokamak-1D" / "extra" / "1D-recycling-with-Tt-control"
    example_dir.mkdir(parents=True)
    (example_dir / "BOUT.inp").write_text(
        "[hermes]\n"
        "Tnorm = 10\n"
        "\n"
        "[e]\n"
        "temperature_setpoint = 5\n"
        "temperature_controller_p = 2.0\n"
        "temperature_controller_i = 0.5\n"
        "control_target_temperature = true\n"
        "temperature_integral_positive = false\n"
        "temperature_source_positive = true\n"
        "\n"
        "nout = 40\n"
        "timestep = 5\n"
        "ny = 80\n"
        "type = beuler\n",
        encoding="utf-8",
    )

    time_points = np.asarray([0.0, 1.0], dtype=np.float64)
    target_temperature = np.asarray([0.0, 0.25], dtype=np.float64)
    error = 0.5 - target_temperature
    integral_state, proportional, integral, multiplier = _reconstruct_temperature_controller(
        time_points=time_points,
        error=error,
        proportional_gain=2.0,
        integral_gain=0.5,
        integral_positive=False,
        source_positive=True,
    )
    monkeypatch.setattr(
        "jax_drb.validation.temperature_feedback_campaign.discover_reference_binary",
        lambda **kwargs: tmp_path / "hermes",
    )

    def _fake_run(*, binary, workdir, timeout_seconds):
        with Dataset(workdir / "BOUT.dmp.0.nc", "w") as dataset:
            dataset.createDimension("t", 2)
            dataset.createDimension("x", 1)
            dataset.createDimension("y", 1)
            dataset.createDimension("z", 1)
            dataset.createVariable("t_array", "f8", ("t",))[:] = time_points
            dataset.createVariable("Te", "f8", ("t", "x", "y", "z"))[:] = target_temperature[:, None, None, None]
            dataset.createVariable("temperature_feedback_src_mult_e", "f8", ("t",))[:] = multiplier
            dataset.createVariable("temperature_feedback_src_p_e", "f8", ("t",))[:] = proportional
            dataset.createVariable("temperature_feedback_src_i_e", "f8", ("t",))[:] = integral
            dataset.createVariable("e_temperature_error_integral", "f8", ("t",))[:] = integral_state
            dataset.createVariable("temperature_feedback_src_shape_e", "f8", ("t", "x", "y", "z"))[:] = 1.0
            dataset.createVariable("SPe_feedback", "f8", ("t", "x", "y", "z"))[:] = multiplier[:, None, None, None]

    monkeypatch.setattr("jax_drb.validation.temperature_feedback_campaign._run_temperature_feedback_example", _fake_run)

    series, timing, provenance = _build_temperature_feedback_series(
        reference_root=tmp_path,
        reference_binary=None,
        nout=4,
        timestep=100.0,
        ny=16,
        solver_type="cvode",
        timeout_seconds=30,
    )

    np.testing.assert_allclose(series.reference_multiplier, multiplier)
    np.testing.assert_allclose(series.reconstructed_multiplier, multiplier)
    np.testing.assert_allclose(series.reference_energy_source.reshape(2), multiplier)
    assert timing["total"] >= 0.0
    assert provenance["mode"] == "discovered_reference_binary"


def test_build_temperature_feedback_series_raises_when_example_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Temperature-feedback example not found"):
        _build_temperature_feedback_series(
            reference_root=tmp_path,
            reference_binary=tmp_path / "hermes",
            nout=4,
            timestep=100.0,
            ny=16,
            solver_type="cvode",
            timeout_seconds=30,
        )
