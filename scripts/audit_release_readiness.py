#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DOCS_MEDIA = (
    "docs/data/diverted_tokamak_turbulence_artifacts/movies/diverted_tokamak_turbulence.gif",
    "docs/data/tokamak_tcv_x21_toroidal_movie_artifacts/movies/tokamak_tcv_x21_toroidal.gif",
    "docs/data/stellarator_fci_validation_artifacts/showcase/movies/stellarator_sol_showcase.gif",
    "docs/data/essos_imported_drb_movie_stationarity_jacobi_media/movies/movie_compact.gif",
)
VERSION_OPERATORS = re.compile(r"(===|==|~=|!=|<=|>=|<|>)")


def _read_text(repo_root: Path, relative_path: str) -> str:
    return (repo_root / relative_path).read_text(encoding="utf-8")


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo_root), *args),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


def _project_metadata(repo_root: Path) -> dict[str, Any]:
    payload = tomllib.loads(_read_text(repo_root, "pyproject.toml"))
    return dict(payload["project"])


def _init_version(repo_root: Path) -> str:
    match = re.search(r"^__version__ = [\"']([^\"']+)[\"']", _read_text(repo_root, "src/jax_drb/__init__.py"), re.M)
    if match is None:
        raise AssertionError("src/jax_drb/__init__.py does not define __version__")
    return match.group(1)


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _require_contains(text: str, needle: str, *, path: str, failures: list[str]) -> None:
    _require(needle in text, f"{path} does not contain {needle!r}", failures)


def _require_file(repo_root: Path, relative_path: str, failures: list[str]) -> None:
    _require((repo_root / relative_path).exists(), f"{relative_path} is missing", failures)


def _is_unpinned_requirement(requirement: str) -> bool:
    requirement_without_marker = requirement.split(";", 1)[0].strip()
    return VERSION_OPERATORS.search(requirement_without_marker) is None


def _check_version_and_release_notes(
    repo_root: Path,
    *,
    allow_existing_tag: bool,
    failures: list[str],
) -> str:
    project = _project_metadata(repo_root)
    version = str(project["version"])
    release_notes = f"docs/release_notes_{version.replace('.', '_')}.md"
    tag_name = f"v{version}"

    _require(_init_version(repo_root) == version, "pyproject.toml and jax_drb.__version__ disagree", failures)
    _require_file(repo_root, release_notes, failures)
    _require_file(repo_root, "CITATION.cff", failures)

    citation = _read_text(repo_root, "CITATION.cff")
    manifest = _read_text(repo_root, "MANIFEST.in")
    _require_contains(citation, f'version: "{version}"', path="CITATION.cff", failures=failures)
    _require_contains(citation, "repository-code: \"https://github.com/uwplasma/jax_drb\"", path="CITATION.cff", failures=failures)
    _require_contains(manifest, "include CITATION.cff", path="MANIFEST.in", failures=failures)

    notes = _read_text(repo_root, release_notes)
    _require_contains(notes, f"# Release Notes: {version}", path=release_notes, failures=failures)
    _require_contains(notes, "## Validation", path=release_notes, failures=failures)
    _require_contains(notes, "## Current Boundary", path=release_notes, failures=failures)
    _require_contains(notes, "full output-window recycling BDF default remains", path=release_notes, failures=failures)

    readme = _read_text(repo_root, "README.md")
    mkdocs = _read_text(repo_root, "mkdocs.yml")
    packaging = _read_text(repo_root, "docs/release_packaging.md")
    _require_contains(readme, f"[docs/{release_notes.removeprefix('docs/')}]", path="README.md", failures=failures)
    _require_contains(mkdocs, f"Release Notes {version}: {release_notes.removeprefix('docs/')}", path="mkdocs.yml", failures=failures)
    _require_contains(packaging, f"release target is `{version}`", path="docs/release_packaging.md", failures=failures)

    tags = set(_run_git(repo_root, "tag", "--list", "v*").split())
    if not allow_existing_tag:
        _require(tag_name not in tags, f"{tag_name} already exists; do not reuse or move published version tags", failures)
    return version


def _check_dependencies(repo_root: Path, failures: list[str]) -> None:
    project = _project_metadata(repo_root)
    _require(str(project["requires-python"]) == ">=3.10", "requires-python must remain >=3.10", failures)
    for dependency in project.get("dependencies", ()):
        _require(
            _is_unpinned_requirement(str(dependency)),
            f"runtime dependency must remain unpinned: {dependency}",
            failures,
        )


def _check_artifact_manifest(repo_root: Path, failures: list[str]) -> None:
    manifest_path = repo_root / "docs/release_artifacts_manifest.json"
    _require(manifest_path.exists(), "docs/release_artifacts_manifest.json is missing", failures)
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    media = payload.get("media", {})
    docs_media = payload.get("bundles", {}).get("docs_media", {})
    _require(int(docs_media.get("contents_count", -1)) == len(media), "docs media contents_count does not match media map", failures)
    _require(int(docs_media.get("size_bytes", 0)) > 0, "docs media bundle size is not positive", failures)
    _require(str(docs_media.get("asset")) == "jax_drb_docs_media.zip", "docs media asset name changed unexpectedly", failures)
    for relative_path in REQUIRED_DOCS_MEDIA:
        _require(relative_path in media, f"release artifact manifest is missing {relative_path}", failures)


def _check_workflows(repo_root: Path, failures: list[str]) -> None:
    publish = _read_text(repo_root, ".github/workflows/publish-pypi.yml")
    docs = _read_text(repo_root, ".github/workflows/docs.yml")
    coverage = _read_text(repo_root, ".github/workflows/coverage.yml")
    rtd = _read_text(repo_root, ".readthedocs.yaml")
    mkdocs = _read_text(repo_root, "mkdocs.yml")

    _require_contains(publish, "types: [published]", path=".github/workflows/publish-pypi.yml", failures=failures)
    _require_contains(publish, "startsWith(github.event.release.tag_name, 'v')", path=".github/workflows/publish-pypi.yml", failures=failures)
    _require_contains(publish, "id-token: write", path=".github/workflows/publish-pypi.yml", failures=failures)
    _require_contains(docs, "mkdocs build --strict --clean", path=".github/workflows/docs.yml", failures=failures)
    _require_contains(coverage, "--cov=jax_drb", path=".github/workflows/coverage.yml", failures=failures)
    _require_contains(rtd, "configuration: mkdocs.yml", path=".readthedocs.yaml", failures=failures)
    _require_contains(mkdocs, "site_url: https://jax-drb.readthedocs.io/", path="mkdocs.yml", failures=failures)


def _load_footprint_module(repo_root: Path) -> Any:
    module_path = repo_root / "scripts/audit_repository_footprint.py"
    spec = importlib.util.spec_from_file_location("audit_repository_footprint", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_footprint(repo_root: Path, failures: list[str]) -> None:
    footprint = _load_footprint_module(repo_root)
    audit = footprint.build_audit(repo_root, top=20, min_size_bytes=footprint._bytes_from_mib(1.0))
    for key in (
        "tracked_large_files",
        "current_tree_large_blobs",
        "reachable_history_large_blobs",
        "untracked_large_files",
    ):
        _require(not audit[key], f"footprint audit reports large entries in {key}", failures)


def run_audit(
    repo_root: Path = REPO_ROOT,
    *,
    allow_existing_tag: bool = False,
    check_footprint: bool = True,
) -> dict[str, Any]:
    failures: list[str] = []
    resolved_root = repo_root.resolve()
    version = _check_version_and_release_notes(
        resolved_root,
        allow_existing_tag=allow_existing_tag,
        failures=failures,
    )
    _check_dependencies(resolved_root, failures)
    _check_artifact_manifest(resolved_root, failures)
    _check_workflows(resolved_root, failures)
    if check_footprint:
        _check_footprint(resolved_root, failures)
    return {
        "passed": not failures,
        "version": version,
        "checked_footprint": check_footprint,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast pre-tag release-readiness audit for metadata, artifacts, workflows, and footprint.",
    )
    parser.add_argument(
        "--allow-existing-tag",
        action="store_true",
        help="Allow v<version> to exist. Use only after a release is already published.",
    )
    parser.add_argument(
        "--skip-footprint",
        action="store_true",
        help="Skip the git history/footprint portion of the audit.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    args = parser.parse_args()

    result = run_audit(
        REPO_ROOT,
        allow_existing_tag=args.allow_existing_tag,
        check_footprint=not args.skip_footprint,
    )
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"release readiness audit for jax_drb {result['version']}")
        print(f"footprint checked: {result['checked_footprint']}")
        if result["passed"]:
            print("release readiness audit passed")
        else:
            print("release readiness audit failed", file=sys.stderr)
            for failure in result["failures"]:
                print(f"  - {failure}", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
