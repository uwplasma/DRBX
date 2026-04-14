from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
from netCDF4 import Dataset

from jax_drb.validation.tokamak_tcv_x21_scaffold import _write_synthetic_preview_workdir
from jax_drb.validation.tokamak_tcv_x21_selected_field import (
    compare_tcv_x21_selected_field_workdirs,
    create_tcv_x21_selected_field_parity_package,
)


def test_compare_tcv_x21_selected_fields_is_zero_for_identical_synthetic_workdirs() -> None:
    with tempfile.TemporaryDirectory(prefix="jax_drb_tcv_x21_selected_field_test_") as temp_dir:
        root = Path(temp_dir)
        reference = _write_synthetic_preview_workdir(root / "reference", field_name="phi")
        candidate = _write_synthetic_preview_workdir(root / "candidate", field_name="phi")

        result = compare_tcv_x21_selected_field_workdirs(
            reference_workdir=reference.workdir,
            candidate_workdir=candidate.workdir,
        )

        assert tuple(result.field_names) == ("Ne", "Pe", "phi")
        for error in result.variable_errors.values():
            assert error.max_abs_error == 0.0
            assert error.rms_error == 0.0
            assert error.relative_l2_error == 0.0


def test_compare_tcv_x21_selected_fields_detects_candidate_field_offset() -> None:
    with tempfile.TemporaryDirectory(prefix="jax_drb_tcv_x21_selected_field_test_") as temp_dir:
        root = Path(temp_dir)
        reference = _write_synthetic_preview_workdir(root / "reference", field_name="phi")
        candidate = _write_synthetic_preview_workdir(root / "candidate", field_name="phi")
        for dump_path in sorted(candidate.workdir.glob("BOUT.dmp.*.nc")):
            with Dataset(dump_path, "r+") as dataset:
                values = np.asarray(dataset.variables["phi"][:], dtype=np.float64)
                dataset.variables["phi"][:] = values + 0.01

        result = compare_tcv_x21_selected_field_workdirs(
            reference_workdir=reference.workdir,
            candidate_workdir=candidate.workdir,
        )

        assert result.variable_errors["phi"].max_abs_error > 0.0
        assert result.variable_errors["phi"].relative_l2_error > 0.0
        assert result.variable_errors["Ne"].max_abs_error == 0.0


def test_create_tcv_x21_selected_field_parity_package_writes_artifacts(tmp_path: Path) -> None:
    artifacts = create_tcv_x21_selected_field_parity_package(
        reference_workdir=None,
        candidate_workdir=None,
        output_root=tmp_path / "output",
    )

    assert artifacts.parity_json_path.exists()
    assert artifacts.parity_arrays_npz_path.exists()
    assert artifacts.parity_plot_png_path.exists()

    payload = json.loads(artifacts.parity_json_path.read_text(encoding="utf-8"))
    assert payload["field_names"] == ["Ne", "Pe", "phi"]
    assert sorted(payload["variable_errors"]) == ["Ne", "Pe", "phi"]
