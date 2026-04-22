from __future__ import annotations

from pathlib import Path

from jax_drb.native.runner_reference import (
    load_curated_case_config,
    reference_case_by_name_or_none,
    reference_root_from_input_path,
)
from jax_drb.reference.cases import ReferenceCase


_INPUT = """
nout = 1
timestep = 1

[mesh]
nx = 1
ny = 1
nz = 1

[solver]
mxstep = 1

[model]
components = h

[h]
type = evolve_density
AA = 1
charge = 1

[Nh]
function = 1.0
"""


def test_reference_root_from_input_path_resolves_relative_reference_tree(tmp_path: Path) -> None:
    input_path = tmp_path / "tests" / "integrated" / "1D-recycling" / "data" / "BOUT.inp"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(_INPUT, encoding="utf-8")
    case = ReferenceCase(
        name="recycling_1d_rhs",
        stage="stageX",
        reference_path="tests/integrated/1D-recycling/data/BOUT.inp",
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Nd",),
    )

    assert reference_root_from_input_path(case, input_path) == tmp_path


def test_load_curated_case_config_applies_reference_root_overrides(tmp_path: Path) -> None:
    input_path = tmp_path / "tests" / "integrated" / "1D-recycling" / "data" / "BOUT.inp"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(_INPUT, encoding="utf-8")
    case = ReferenceCase(
        name="recycling_1d_rhs",
        stage="stageX",
        reference_path="tests/integrated/1D-recycling/data/BOUT.inp",
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Nd",),
        extra_overrides=("mesh:nx=2", "mesh:cache_root=\"{reference_root}\""),
    )

    config = load_curated_case_config(case, input_path)

    assert config.parsed("mesh", "nx") == 2
    assert config.get("mesh", "cache_root").parsed == str(tmp_path)


def test_reference_case_by_name_or_none_finds_known_case() -> None:
    case = reference_case_by_name_or_none("neutral_mixed_short_window")
    assert case is not None
    assert case.name == "neutral_mixed_short_window"


def test_reference_case_by_name_or_none_returns_none_for_unknown_case() -> None:
    assert reference_case_by_name_or_none("definitely_unknown_case_name") is None
