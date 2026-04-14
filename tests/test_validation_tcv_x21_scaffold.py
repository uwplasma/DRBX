from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import create_tcv_x21_scaffold_package


def _write_reference_tree(root: Path) -> None:
    target = root / "examples" / "tokamak-3D" / "tcv-x21" / "data"
    target.mkdir(parents=True, exist_ok=True)
    (target / "BOUT.inp").write_text("[dummy]\nvalue = 1\n", encoding="utf-8")


def test_tcv_x21_scaffold_preview_generates_artifacts(tmp_path: Path) -> None:
    artifacts = create_tcv_x21_scaffold_package(
        reference_root=tmp_path,
        output_root=tmp_path / "output",
    )

    for path in (
        artifacts.manifest_json_path,
        artifacts.arrays_npz_path,
        artifacts.analysis_json_path,
        artifacts.snapshots_png_path,
        artifacts.poster_png_path,
        artifacts.movie_gif_path,
    ):
        assert path.exists()

    manifest = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
    assert manifest["case_name"] == "tokamak_tcv_x21_escalation"
    assert manifest["capability_tier"] == "scaffolded_reference_backed"
    assert manifest["preview_mode"] is True
    assert manifest["workdir_mode"] == "synthetic_preview"
    assert manifest["reference_exists"] is False
    assert manifest["artifacts"]["movie_gif"].endswith("movies/tokamak_tcv_x21_scaffold.gif")


def test_tcv_x21_scaffold_marks_reference_tree_when_present(tmp_path: Path) -> None:
    _write_reference_tree(tmp_path)
    artifacts = create_tcv_x21_scaffold_package(
        reference_root=tmp_path,
        output_root=tmp_path / "output",
    )

    manifest = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
    assert manifest["reference_exists"] is True
    assert manifest["reference_input_path"].endswith("examples/tokamak-3D/tcv-x21/data/BOUT.inp")
