from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib

from ..config.boutinp import load_bout_input
from ..runtime.run_config import RunConfiguration

DEFAULT_CASE_MANIFEST = Path(__file__).resolve().parents[3] / "references" / "reference_case_ladder.toml"
_VALID_CAPABILITY_TIERS = {
    "native_exact",
    "native_operational",
    "scaffolded_reference_backed",
}


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    stage: str
    reference_path: str
    parity_mode: str
    rationale: str
    capability_tier: str = "native_exact"
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
            capability_tier=_resolve_capability_tier(
                entry["name"],
                entry.get("capability_tier"),
            ),
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


def _resolve_capability_tier(case_name: str, configured_tier: str | None) -> str:
    if configured_tier is not None:
        tier = str(configured_tier).strip()
        if tier not in _VALID_CAPABILITY_TIERS:
            raise ValueError(
                f"unknown capability_tier {tier!r} for reference case {case_name!r}; "
                f"expected one of {sorted(_VALID_CAPABILITY_TIERS)}"
            )
        return tier

    if case_name.startswith(("tokamak_", "integrated_2d_", "alfven_wave_", "annulus_he_emag_")):
        return "scaffolded_reference_backed"
    if case_name.startswith(("recycling_1d_", "recycling_dthe_")):
        return "native_operational"
    return "native_exact"


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
