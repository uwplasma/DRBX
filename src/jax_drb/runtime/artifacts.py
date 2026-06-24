from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile

from ..reference.paths import repo_root


ARTIFACT_RELEASE_TAG = "validation-artifacts-2026-04-28"
ARTIFACT_BASE_URL = f"https://github.com/uwplasma/jax_drb/releases/download/{ARTIFACT_RELEASE_TAG}"
ARTIFACT_REPO = "uwplasma/jax_drb"
DOCS_MEDIA_ASSET = "jax_drb_docs_media.zip"
REFERENCE_BASELINES_ASSET = "jax_drb_reference_baselines.zip"

DOCS_MEDIA_SENTINELS = (
    Path(
        "docs/data/diverted_tokamak_turbulence_artifacts/movies/"
        "diverted_tokamak_turbulence.gif"
    ),
    Path(
        "docs/data/tokamak_tcv_x21_toroidal_movie_artifacts/movies/"
        "tokamak_tcv_x21_toroidal.gif"
    ),
    Path(
        "docs/data/stellarator_fci_validation_artifacts/showcase/movies/"
        "stellarator_sol_showcase.gif"
    ),
)

OPTIONAL_DOCS_MEDIA_SENTINELS = (
    Path(
        "docs/data/essos_imported_drb_movie_artifacts/movies/"
        "essos_imported_drb_movie_campaign.gif"
    ),
)


def ensure_reference_baselines(
    *,
    root: str | Path | None = None,
    base_url: str | None = None,
    force: bool = False,
) -> Path:
    """Restore heavy validation baselines for research tests when absent."""

    resolved_root = Path(root) if root is not None else repo_root()
    reference_arrays = resolved_root / "references" / "baselines" / "reference_arrays"
    reference_snapshots = resolved_root / "references" / "baselines" / "reference_snapshots"
    sentinel = reference_arrays / "alfven_wave_short_window.npz"
    snapshot_sentinel = reference_snapshots / "tokamak_turbulence_rhs_field_history.npz"
    if not force and sentinel.exists() and snapshot_sentinel.exists():
        return resolved_root / "references" / "baselines"
    if os.environ.get("JAX_DRB_OFFLINE_ARTIFACTS", "").lower() in {"1", "true", "yes"}:
        raise FileNotFoundError(
            "Heavy reference baselines are not present and JAX_DRB_OFFLINE_ARTIFACTS is enabled."
        )

    cache_dir = Path(
        os.environ.get("JAX_DRB_ARTIFACT_CACHE", resolved_root / ".jax_drb_artifact_cache")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / REFERENCE_BASELINES_ASSET
    if force or not archive_path.exists():
        resolved_base_url = (
            base_url or os.environ.get("JAX_DRB_ARTIFACT_BASE_URL") or ARTIFACT_BASE_URL
        ).rstrip("/")
        _download_release_asset(
            f"{resolved_base_url}/{REFERENCE_BASELINES_ASSET}",
            archive_path,
            asset_name=REFERENCE_BASELINES_ASSET,
        )
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(resolved_root)
    if not sentinel.exists() or not snapshot_sentinel.exists():
        raise FileNotFoundError(
            f"Reference artifact archive did not restore expected baselines under {resolved_root}"
        )
    return resolved_root / "references" / "baselines"


def ensure_docs_media(
    *,
    root: str | Path | None = None,
    base_url: str | None = None,
    force: bool = False,
) -> Path:
    """Restore release-backed documentation figures, movies, and NPZ arrays."""

    resolved_root = Path(root) if root is not None else repo_root()
    missing_required = [
        relative_path
        for relative_path in DOCS_MEDIA_SENTINELS
        if not (resolved_root / relative_path).exists()
    ]
    if not force and not missing_required:
        return resolved_root / "docs" / "data"
    if os.environ.get("JAX_DRB_OFFLINE_ARTIFACTS", "").lower() in {"1", "true", "yes"}:
        raise FileNotFoundError(
            "Docs media are not present and JAX_DRB_OFFLINE_ARTIFACTS is enabled."
        )

    cache_dir = Path(
        os.environ.get("JAX_DRB_ARTIFACT_CACHE", resolved_root / ".jax_drb_artifact_cache")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / DOCS_MEDIA_ASSET
    if force or not archive_path.exists():
        resolved_base_url = (
            base_url or os.environ.get("JAX_DRB_ARTIFACT_BASE_URL") or ARTIFACT_BASE_URL
        ).rstrip("/")
        _download_release_asset(
            f"{resolved_base_url}/{DOCS_MEDIA_ASSET}",
            archive_path,
            asset_name=DOCS_MEDIA_ASSET,
        )
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(resolved_root)
    remaining = [
        str(relative_path)
        for relative_path in DOCS_MEDIA_SENTINELS
        if not (resolved_root / relative_path).exists()
    ]
    if remaining:
        raise FileNotFoundError(
            "Docs media artifact archive did not restore expected files under "
            f"{resolved_root}: {', '.join(remaining)}"
        )
    optional_missing = [
        str(relative_path)
        for relative_path in OPTIONAL_DOCS_MEDIA_SENTINELS
        if not (resolved_root / relative_path).exists()
    ]
    if optional_missing:
        print(
            "Warning: docs media artifact archive is missing optional files under "
            f"{resolved_root}: {', '.join(optional_missing)}"
        )
    return resolved_root / "docs" / "data"


def _download_release_asset(url: str, destination: Path, *, asset_name: str) -> None:
    if _download_with_gh(asset_name, destination):
        return
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        _download_with_github_api(asset_name, destination, token=token)
        return
    _download_with_urllib(url, destination)


def _download_with_gh(asset_name: str, destination: Path) -> bool:
    gh = shutil.which("gh")
    if gh is None:
        return False
    completed = subprocess.run(
        [
            gh,
            "release",
            "download",
            ARTIFACT_RELEASE_TAG,
            "--repo",
            ARTIFACT_REPO,
            "--pattern",
            asset_name,
            "--dir",
            str(destination.parent),
            "--clobber",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    downloaded = destination.parent / asset_name
    if completed.returncode == 0 and downloaded.exists():
        if downloaded != destination:
            downloaded.replace(destination)
        return True
    return False


def _download_with_urllib(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _download_with_github_api(asset_name: str, destination: Path, *, token: str) -> None:
    asset_url = _resolve_release_asset_api_url(asset_name, token=token)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(asset_url)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/octet-stream")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_release_asset_api_url(asset_name: str, *, token: str) -> str:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{ARTIFACT_REPO}/releases/tags/{ARTIFACT_RELEASE_TAG}"
    )
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise FileNotFoundError(
                "Release tag or repository not visible to the configured GitHub token."
            ) from error
        raise
    for asset in payload.get("assets", ()):
        if asset.get("name") == asset_name:
            asset_url = asset.get("url")
            if asset_url:
                return asset_url
            break
    raise FileNotFoundError(
        f"Could not find release asset {asset_name!r} in {ARTIFACT_REPO}@{ARTIFACT_RELEASE_TAG}."
    )
