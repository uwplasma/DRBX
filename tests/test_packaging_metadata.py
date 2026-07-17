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


def _package_name(requirement: str) -> str:
    package_requirement = requirement.split(";", 1)[0]
    for token in ("<", ">", "=", "~", "!"):
        package_requirement = package_requirement.split(token, 1)[0]
    return package_requirement.strip()


# solvax carries the extracted structured-solver machinery; the Fourier--Helmholtz
# elliptic solve the vorticity model uses landed in solvax 0.8.1, so it is the one
# runtime dependency allowed a lower-bound version floor.
_VERSION_FLOOR_EXCEPTIONS = {"solvax"}


def test_pyproject_dependencies_are_unpinned() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    build_requires = payload["build-system"]["requires"]
    project_dependencies = payload["project"]["dependencies"]
    optional_dependencies = payload["project"].get("optional-dependencies", {})

    assert all(not _has_version_specifier(item) for item in build_requires)
    assert all(
        not _has_version_specifier(item)
        for item in project_dependencies
        if _package_name(item) not in _VERSION_FLOOR_EXCEPTIONS
    )
    for items in optional_dependencies.values():
        assert all(not _has_version_specifier(item) for item in items)


def test_core_runtime_dependencies_are_installed_by_default() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_dependencies = payload["project"]["dependencies"]

    for requirement in ("jax", "scipy", "matplotlib", "netCDF4", "pillow", "rich", "solvax"):
        assert any(_package_name(item) == requirement for item in project_dependencies)

    # Declared-but-unimported packages were removed in the v2 plan's Phase 0;
    # they must not silently return.
    for requirement in ("diffrax", "equinox"):
        assert not any(
            item == requirement or item.startswith(f"{requirement};") for item in project_dependencies
        )


def test_import_version_matches_pyproject() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    init_text = (REPO_ROOT / "src" / "drbx" / "__init__.py").read_text(encoding="utf-8")

    assert f'__version__ = "{payload["project"]["version"]}"' in init_text


def test_publish_pypi_workflow_uses_trusted_publishing() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text(encoding="utf-8")

    assert "release:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "environment: pypi" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
