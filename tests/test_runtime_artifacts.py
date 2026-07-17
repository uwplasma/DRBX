from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

import pytest

from dkx.runtime import artifacts


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_ensure_docs_media_restores_release_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_archive = tmp_path / "source" / artifacts.DOCS_MEDIA_ASSET
    _write_zip(
        source_archive,
        {
            str(relative_path): b"example-artifact"
            for relative_path in artifacts.DOCS_MEDIA_SENTINELS
        },
    )
    root = tmp_path / "repo"
    root.mkdir()
    calls: list[tuple[str, Path, str]] = []

    def fake_download(url: str, destination: Path, *, asset_name: str) -> None:
        calls.append((url, destination, asset_name))
        shutil.copyfile(source_archive, destination)

    monkeypatch.delenv("DKX_OFFLINE_ARTIFACTS", raising=False)
    monkeypatch.setattr(artifacts, "_download_release_asset", fake_download)

    restored = artifacts.ensure_docs_media(root=root)

    assert restored == root / "docs" / "data"
    assert calls == [
        (
            f"{artifacts.ARTIFACT_BASE_URL}/{artifacts.DOCS_MEDIA_ASSET}",
            root / ".dkx_artifact_cache" / artifacts.DOCS_MEDIA_ASSET,
            artifacts.DOCS_MEDIA_ASSET,
        )
    ]
    for relative_path in artifacts.DOCS_MEDIA_SENTINELS:
        assert (root / relative_path).read_bytes() == b"example-artifact"


def test_ensure_docs_media_uses_configured_cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_archive = tmp_path / "source" / artifacts.DOCS_MEDIA_ASSET
    _write_zip(
        source_archive,
        {
            str(relative_path): b"cached-artifact"
            for relative_path in artifacts.DOCS_MEDIA_SENTINELS
        },
    )
    root = tmp_path / "repo"
    cache_dir = tmp_path / "artifact-cache"
    root.mkdir()
    calls: list[Path] = []

    def fake_download(url: str, destination: Path, *, asset_name: str) -> None:
        calls.append(destination)
        shutil.copyfile(source_archive, destination)

    monkeypatch.setenv("DKX_ARTIFACT_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("DKX_ARTIFACT_CACHE", raising=False)
    monkeypatch.delenv("DKX_OFFLINE_ARTIFACTS", raising=False)
    monkeypatch.setattr(artifacts, "_download_release_asset", fake_download)

    restored = artifacts.ensure_docs_media(root=root)

    assert restored == root / "docs" / "data"
    assert calls == [cache_dir / artifacts.DOCS_MEDIA_ASSET]
    assert (cache_dir / artifacts.DOCS_MEDIA_ASSET).exists()


def test_ensure_docs_media_is_noop_when_sentinels_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "repo"
    for relative_path in artifacts.DOCS_MEDIA_SENTINELS:
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"existing")

    def unexpected_download(url: str, destination: Path, *, asset_name: str) -> None:
        raise AssertionError("download should not be called when docs media exist")

    monkeypatch.setattr(artifacts, "_download_release_asset", unexpected_download)

    assert artifacts.ensure_docs_media(root=root) == root / "docs" / "data"


def test_ensure_docs_media_honors_offline_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DKX_OFFLINE_ARTIFACTS", "1")

    with pytest.raises(FileNotFoundError, match="Docs media are not present"):
        artifacts.ensure_docs_media(root=tmp_path / "repo")


def test_urllib_download_retries_with_configured_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
            self._read_count = 0

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            del size
            if self._read_count == 0:
                self._read_count += 1
                return self._payload
            return b""

    calls: list[float] = []

    def flaky_urlopen(request: object, *, timeout: float) -> FakeResponse:
        del request
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError("transient release-asset timeout")
        return FakeResponse(b"downloaded")

    monkeypatch.setenv("DKX_ARTIFACT_DOWNLOAD_TIMEOUT", "0.5")
    monkeypatch.setenv("DKX_ARTIFACT_DOWNLOAD_ATTEMPTS", "2")
    monkeypatch.setattr(artifacts.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(artifacts.time, "sleep", lambda seconds: None)

    destination = tmp_path / "artifact.zip"
    artifacts._download_with_urllib("https://example.invalid/artifact.zip", destination)

    assert calls == [0.5, 0.5]
    assert destination.read_bytes() == b"downloaded"
