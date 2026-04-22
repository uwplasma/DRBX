from __future__ import annotations

from pathlib import Path

from ..config.boutinp import BoutConfig, apply_bout_overrides, load_bout_input
from ..reference.cases import ReferenceCase, load_reference_cases


def reference_root_from_input_path(case: ReferenceCase, input_path: Path) -> Path:
    reference_path = Path(case.reference_path)
    offset = len(reference_path.parts) - 1
    if reference_path.is_absolute():
        offset -= 1
    return input_path.parents[offset]


def load_curated_case_config(case: ReferenceCase, input_path: Path) -> BoutConfig:
    config = load_bout_input(input_path)
    if not case.extra_overrides:
        return config
    reference_root = reference_root_from_input_path(case, input_path)
    resolved_overrides = tuple(
        override.format(reference_root=str(reference_root))
        for override in case.extra_overrides
    )
    return apply_bout_overrides(config, resolved_overrides)


def reference_case_by_name_or_none(case_name: str) -> ReferenceCase | None:
    return next((case for case in load_reference_cases() if case.name == case_name), None)
