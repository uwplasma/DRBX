from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..config.boutinp import load_bout_input
from ..runtime.run_config import RunConfiguration

DEFAULT_CASE_MANIFEST = Path(__file__).resolve().parents[3] / "references" / "reference_case_ladder.toml"


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    stage: str
    reference_path: str
    parity_mode: str
    rationale: str
    compare_variables: tuple[str, ...] = ()
    extra_overrides: tuple[str, ...] = ()
    trim_x_guards: bool = False
    trim_y_guards: bool = False
    process_count: int = 1
    artifact_bundle_url: str | None = None
    artifact_bundle_sha256: str | None = None
    artifact_bundle_files: tuple[str, ...] = ()

    def input_path(self, reference_root: str | Path) -> Path:
        return Path(reference_root) / self.reference_path


@dataclass(frozen=True)
class ResolvedReferenceCase:
    case: ReferenceCase
    input_path: Path
    exists: bool
    run_config: RunConfiguration | None


def load_reference_cases(manifest_path: str | Path | None = None) -> tuple[ReferenceCase, ...]:
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_CASE_MANIFEST
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    return tuple(
        ReferenceCase(
            name=entry["name"],
            stage=entry["stage"],
            reference_path=entry["reference_path"],
            parity_mode=entry["parity_mode"],
            rationale=entry["rationale"],
            compare_variables=tuple(entry.get("compare_variables", [])),
            extra_overrides=tuple(entry.get("extra_overrides", [])),
            trim_x_guards=bool(entry.get("trim_x_guards", False)),
            trim_y_guards=bool(entry.get("trim_y_guards", False)),
            process_count=int(entry.get("process_count", 1)),
            artifact_bundle_url=entry.get("artifact_bundle_url"),
            artifact_bundle_sha256=entry.get("artifact_bundle_sha256"),
            artifact_bundle_files=tuple(entry.get("artifact_bundle_files", [])),
        )
        for entry in payload.get("case", [])
    )


def resolve_reference_cases(
    reference_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[ResolvedReferenceCase, ...]:
    resolved_cases: list[ResolvedReferenceCase] = []
    for case in load_reference_cases(manifest_path):
        input_path = case.input_path(reference_root)
        if not input_path.exists():
            resolved_cases.append(
                ResolvedReferenceCase(
                    case=case,
                    input_path=input_path,
                    exists=False,
                    run_config=None,
                )
            )
            continue

        config = load_bout_input(input_path)
        resolved_cases.append(
            ResolvedReferenceCase(
                case=case,
                input_path=input_path,
                exists=True,
                run_config=RunConfiguration.from_config(config),
            )
        )

    return tuple(resolved_cases)
