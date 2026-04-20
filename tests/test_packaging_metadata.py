from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _has_version_specifier(requirement: str) -> bool:
    package_requirement = requirement.split(";", 1)[0]
    return any(token in package_requirement for token in ("<", ">", "=", "~", "!"))


def test_pyproject_dependencies_are_unpinned() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    build_requires = payload["build-system"]["requires"]
    project_dependencies = payload["project"]["dependencies"]
    optional_dependencies = payload["project"].get("optional-dependencies", {})

    assert all(not _has_version_specifier(item) for item in build_requires)
    assert all(not _has_version_specifier(item) for item in project_dependencies)
    for items in optional_dependencies.values():
        assert all(not _has_version_specifier(item) for item in items)


def test_core_runtime_dependencies_are_installed_by_default() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_dependencies = payload["project"]["dependencies"]

    for requirement in ("jax", "diffrax", "scipy", "equinox", "matplotlib", "netCDF4", "pillow", "rich"):
        assert any(item == requirement or item.startswith(f"{requirement};") for item in project_dependencies)


def test_publish_pypi_workflow_uses_trusted_publishing() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text(encoding="utf-8")

    assert "release:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "environment: pypi" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
