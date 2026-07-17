from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
import zipfile

from .paths import repo_root


ARTIFACT_RELEASE_TAG = "validation-artifacts-2026-04-28"
ARTIFACT_BASE_URL = f"https://github.com/uwplasma/drbx/releases/download/{ARTIFACT_RELEASE_TAG}"
DOCS_MEDIA_ASSET = "drbx_docs_media.zip"

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
    Path(
        "docs/data/essos_imported_drb_movie_stationarity_jacobi_media/movies/"
        "movie_compact.gif"
    ),
)


def ensure_docs_media(
    *,
    root: str | Path | None = None,
    base_url: str | None = None,
    force: bool = False,
) -> Path:
    """Restore release-backed documentation figures, movies, and NPZ arrays."""

    resolved_root = Path(root) if root is not None else repo_root()
    missing = [
        relative_path
        for relative_path in DOCS_MEDIA_SENTINELS
        if not (resolved_root / relative_path).exists()
    ]
    if not force and not missing:
        return resolved_root / "docs" / "data"
    if os.environ.get("DRBX_OFFLINE_ARTIFACTS", "").lower() in {"1", "true", "yes"}:
        raise FileNotFoundError(
            "Docs media are not present and DRBX_OFFLINE_ARTIFACTS is enabled."
        )

    cache_dir = _artifact_cache_dir(resolved_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / DOCS_MEDIA_ASSET
    if force or not archive_path.exists():
        resolved_base_url = (
            base_url or os.environ.get("DRBX_ARTIFACT_BASE_URL") or ARTIFACT_BASE_URL
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
    return resolved_root / "docs" / "data"


def _download_release_asset(url: str, destination: Path, *, asset_name: str) -> None:
    if _download_with_gh(asset_name, destination):
        return
    _download_with_urllib(url, destination)


def _artifact_cache_dir(root: Path) -> Path:
    configured = (
        os.environ.get("DRBX_ARTIFACT_CACHE_DIR")
        or os.environ.get("DRBX_ARTIFACT_CACHE")
    )
    if configured:
        return Path(configured).expanduser()
    return root / ".drbx_artifact_cache"


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
            "uwplasma/drbx",
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
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    timeout = float(os.environ.get("DRBX_ARTIFACT_DOWNLOAD_TIMEOUT", "120"))
    attempts = max(1, int(os.environ.get("DRBX_ARTIFACT_DOWNLOAD_ATTEMPTS", "3")))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open(
                "wb"
            ) as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            temporary.replace(destination)
            return
        except Exception as error:  # pragma: no cover - type depends on urllib backend
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < attempts:
                time.sleep(min(2.0, 0.25 * attempt))
    assert last_error is not None
    raise last_error
