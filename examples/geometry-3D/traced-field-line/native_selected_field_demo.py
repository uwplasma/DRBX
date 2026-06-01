from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_drb.validation import create_native_traced_field_line_selected_field_package

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "data" / "traced_field_line_native_selected_field_artifacts"


def _write_demo_grid(path: Path, *, offset: float) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 4)
        dataset.createDimension("y", 3)
        dataset.createDimension("z", 2)
        for name, scale in (("g11", 2.0), ("g33", 3.0)):
            variable = dataset.createVariable(name, "f8", ("x", "y", "z"))
            values = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
            variable[:] = scale + values + offset


def _resolve_external_pair() -> tuple[Path, Path] | None:
    reference = Path("/tmp/zoidberg_better_metric/test/mms/poloidal_const_4_2_4_1.fci.nc")
    candidate = Path("/tmp/zoidberg_better_metric/test/mms/radial_const_4_2_4_1.fci.nc")
    if reference.exists() and candidate.exists():
        return reference, candidate
    return None


external_pair = _resolve_external_pair()
if external_pair is None:
    with tempfile.TemporaryDirectory(prefix="jax_drb_native_traced_demo_") as temp_dir:
        temp_root = Path(temp_dir)
        reference = temp_root / "reference.fci.nc"
        candidate = temp_root / "candidate.fci.nc"
        _write_demo_grid(reference, offset=0.0)
        _write_demo_grid(candidate, offset=0.25)
        artifacts = create_native_traced_field_line_selected_field_package(
            reference_mesh_spec=reference,
            candidate_mesh_spec=candidate,
            output_root=OUTPUT_ROOT,
        )
else:
    artifacts = create_native_traced_field_line_selected_field_package(
        reference_mesh_spec=external_pair[0],
        candidate_mesh_spec=external_pair[1],
        output_root=OUTPUT_ROOT,
    )

print(f"parity: {artifacts.parity_json_path}")
print(f"comparison: {artifacts.comparison_json_path}")
print(f"observable: {artifacts.observable_report_json_path}")
print(f"runtime: {artifacts.runtime_report_json_path}")
