from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jax_drb.validation import audit_essos_imported_artifact_reports


# SIMSOPT-style user parameters: edit these first, then run this file.
RUN_EXAMPLE = True
REQUIRE_ALL_CURRENT = True

REPORT_JSON_PATHS = (
    Path("docs/data/essos_imported_fci_artifacts/data/essos_imported_fci_campaign.json"),
    Path("docs/data/essos_imported_fci_vmec_artifacts/data/essos_imported_fci_vmec_campaign.json"),
    Path("docs/data/essos_imported_fci_hybrid_artifacts/data/essos_imported_fci_hybrid_campaign.json"),
    Path("docs/data/essos_imported_drb_movie_artifacts/data/essos_imported_drb_movie_campaign.json"),
    Path("docs/data/essos_imported_drb_movie_hybrid_artifacts/data/essos_imported_drb_movie_hybrid_campaign.json"),
)


@dataclass(frozen=True)
class ImportedArtifactSchemaAuditSettings:
    """Resolved settings for a clean-clone imported-field artifact audit."""

    report_json_paths: tuple[Path, ...]
    require_all_current: bool


def build_audit_settings(
    *,
    report_json_paths: tuple[Path, ...] = REPORT_JSON_PATHS,
    require_all_current: bool = REQUIRE_ALL_CURRENT,
) -> ImportedArtifactSchemaAuditSettings:
    """Resolve top-level parameters into one audit setting object."""

    paths = tuple(Path(path) for path in report_json_paths)
    if not paths:
        raise ValueError("report_json_paths must contain at least one report.")
    return ImportedArtifactSchemaAuditSettings(
        report_json_paths=paths,
        require_all_current=bool(require_all_current),
    )


def run_artifact_schema_audit(
    settings: ImportedArtifactSchemaAuditSettings,
) -> dict[str, object]:
    """Audit committed imported-field JSON reports against current schemas."""

    summary = audit_essos_imported_artifact_reports(settings.report_json_paths)
    for report in summary["reports"]:
        status = "current" if not report["stale"] else "stale"
        missing = len(report["missing_report_fields"]) + sum(
            len(value) for value in report["missing_diagnostic_fields"].values()
        )
        print(
            "imported artifact schema audit: "
            f"status={status}, "
            f"kind={report['artifact_kind']}, "
            f"missing_items={missing}, "
            f"path={report['report_json_path']}"
        )
    print(
        "imported artifact schema audit summary: "
        f"reports={summary['report_count']}, "
        f"stale={summary['stale_report_count']}, "
        f"schema_passed={summary['schema_passed']}"
    )
    if settings.require_all_current and not summary["schema_passed"]:
        raise RuntimeError(
            "Imported-field artifact schema audit failed; regenerate stale reports "
            "before promoting these figures or movies."
        )
    return summary


if RUN_EXAMPLE:
    AUDIT_SETTINGS = build_audit_settings()
    AUDIT_SUMMARY = run_artifact_schema_audit(AUDIT_SETTINGS)
