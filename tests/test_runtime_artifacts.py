from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

import pytest

from jax_drb.runtime import artifacts


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

    monkeypatch.delenv("JAX_DRB_OFFLINE_ARTIFACTS", raising=False)
    monkeypatch.setattr(artifacts, "_download_release_asset", fake_download)

    restored = artifacts.ensure_docs_media(root=root)

    assert restored == root / "docs" / "data"
    assert calls == [
        (
            f"{artifacts.ARTIFACT_BASE_URL}/{artifacts.DOCS_MEDIA_ASSET}",
            root / ".jax_drb_artifact_cache" / artifacts.DOCS_MEDIA_ASSET,
            artifacts.DOCS_MEDIA_ASSET,
        )
    ]
    for relative_path in artifacts.DOCS_MEDIA_SENTINELS:
        assert (root / relative_path).read_bytes() == b"example-artifact"


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
    monkeypatch.setenv("JAX_DRB_OFFLINE_ARTIFACTS", "1")

    with pytest.raises(FileNotFoundError, match="Docs media are not present"):
        artifacts.ensure_docs_media(root=tmp_path / "repo")
