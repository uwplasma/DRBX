"""Audit committed imported-field report JSONs against the current schemas.

Reads the imported-field FCI (coil, VMEC, hybrid) and DRB movie report JSONs
listed in ``REPORT_JSON_PATHS`` -- written by ``imported_fci_campaign.py`` and
``imported_drb_movie.py`` -- and checks each against the report/diagnostic
schema the current code would write. Prints one status line per report and a
summary; with ``REQUIRE_ALL_CURRENT`` it fails if any report is stale or
missing, so stale figures/movies are never promoted. Writes nothing; no ESSOS
needed.

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/imported_artifact_schema_audit.py
"""

from __future__ import annotations

from pathlib import Path

from drbx.validation import audit_essos_imported_artifact_reports

# --- PARAMETERS ------------------------------------------------------------------
REQUIRE_ALL_CURRENT = False
REPORT_JSON_PATHS = (
    Path("docs/data/essos_imported_fci_artifacts/data/essos_imported_fci_campaign.json"),
    Path("docs/data/essos_imported_fci_vmec_artifacts/data/essos_imported_fci_vmec_campaign.json"),
    Path("docs/data/essos_imported_fci_hybrid_artifacts/data/essos_imported_fci_hybrid_campaign.json"),
    Path("docs/data/essos_imported_drb_movie_artifacts/data/essos_imported_drb_movie_campaign.json"),
    Path("docs/data/essos_imported_drb_movie_hybrid_artifacts/data/essos_imported_drb_movie_hybrid_campaign.json"),
)

# --- run --------------------------------------------------------------------------
present = tuple(path for path in REPORT_JSON_PATHS if path.exists())
for path in REPORT_JSON_PATHS:
    if not path.exists():
        print(f"missing report (regenerate it first): {path}")
if not present:
    raise SystemExit("no imported-field reports found; run imported_fci_campaign.py / imported_drb_movie.py first")
summary = audit_essos_imported_artifact_reports(present)
for report in summary["reports"]:
    missing = len(report["missing_report_fields"]) + sum(map(len, report["missing_diagnostic_fields"].values()))
    print(f"status={'stale' if report['stale'] else 'current'}, kind={report['artifact_kind']}, "
          f"missing_items={missing}, path={report['report_json_path']}")
print(f"summary: reports={summary['report_count']}, stale={summary['stale_report_count']}, "
      f"missing={len(REPORT_JSON_PATHS) - len(present)}, schema_passed={summary['schema_passed']}")
if REQUIRE_ALL_CURRENT and (not summary["schema_passed"] or len(present) < len(REPORT_JSON_PATHS)):
    raise RuntimeError("imported-field artifact audit failed; regenerate stale/missing reports before promotion")
