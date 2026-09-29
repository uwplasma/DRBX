"""Oracle manifest: exactly the frozen-campaign files ``p_shared.replay_units``
reads, per campaign and grid -- P08 step-1 deliverable 2 ("Oracle inputs").

Only the files the replay *actually reads* are listed (never a whole chunk
folder "just in case"): see :func:`campaign_files`, which enumerates the
same paths :func:`p_shared.replay_units.init_worker`'s
``_load_oracle_owner_values`` and :func:`p_shared.replay_units.reduce_grid`
open, nothing more. A campaign's saved ``owner_values``/raw/faces/global
arrays are always listed individually; the one exception that is
necessarily a whole (but bounded, per-grid) set of small files is P05's
"upwind" oracle (:data:`P05_UPWIND_GLOB`), since the accepted (pre-fix) P05
campaign itself saved that array chunked, one file per face-chunk, with no
single merged array anywhere on disk.

Every path in the manifest is stored **workspace-relative** (the workspace
root, ``.../HSX drbx``, not ``DRBX/``) so a manifest built here matches one
rebuilt against a different local copy or a remote extraction of a
delivered tarball -- see ``--oracle-root`` below.

Delivery: rather than point the remote run at each frozen campaign's own
``work/`` folder over the network, the oracle files this manifest lists are
tarred locally (:func:`pack_oracle_tar` / the campaign CLI's
``pack-oracles`` command) and the tarball is uploaded once; the remote run
then points ``--oracle-root`` at wherever it extracted that tarball. This
module's ``verify`` step is identical either way: hash every manifest file
under whatever root it is given.

Run from ``DRBX/scripts``.
"""
from __future__ import annotations

import glob
import json
import tarfile
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
_REPO = _SCRIPTS.parent
_WORKSPACE = _REPO.parent

MANIFEST_SCHEMA = "drbx.p08-step1-oracle-manifest.v1"

#: The accepted (pre-fix) P05 campaign's own per-face-chunk "upwind" array
#: (design section 5's saved oracle table) -- never merged into one file on
#: disk by that campaign, so every matching chunk file for the requested
#: grid is listed individually (see the module docstring).
P05_UPWIND_GLOB = "N{n}_faces_*.npz"


def _rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(_WORKSPACE))


def campaign_files(campaign: str, n: int, paths: dict) -> list[Path]:
    """Every file :mod:`p_shared.replay_units` reads for ``campaign`` at
    grid ``n`` -- the same paths ``init_worker``/``reduce_grid`` open,
    duplicated here rather than imported so this module (and its ``pack
    -oracles``/``verify-inputs`` CLI use) can run without first loading a
    row artifact or JAX."""
    files: list[Path] = []
    if campaign == "p05":
        files += [paths["p05"] / f"N{n}.owner_results.npz",
                  paths["p05"] / "reuse_inputs" / f"N{n}.reuse.npz"]
        files += sorted(Path(p) for p in glob.glob(str(paths["p05_upwind_chunks"] / P05_UPWIND_GLOB.format(n=n))))
    elif campaign == "p05n_frozen":
        root = paths["p05n_frozen"]
        files += [root / f"N{n}.owner_values.npz", root / f"N{n}.raw.npz", root / f"N{n}.faces.npz"]
    elif campaign == "p05n_upwind":
        root = paths["p05n_p06n_upwind"] / "p05n_upwind"
        files += [root / f"N{n}.owner_values.npz", root / f"N{n}.raw.npz", root / f"N{n}.faces.npz"]
    elif campaign == "p06n":
        root = paths["p05n_p06n_upwind"] / "p06n"
        files += [root / f"N{n}.owner_values.npz", root / f"N{n}.raw.npz", root / f"N{n}.faces.npz"]
    elif campaign == "p06_legacy":
        root = paths["p06_legacy"]
        files += [root / f"N{n}.prepare.npz", root / f"N{n}.npz"]
    elif campaign == "p07":
        root = paths["p07"]
        files += [root / "owner_values.npz", root / f"N{n}.global.npz"]
    elif campaign == "p07n":
        root = paths["p07n"]
        files += [root / f"N{n}.owner_values.npz", root / f"N{n}.global.npz"]
    else:
        raise ValueError(f"unknown campaign {campaign!r}")
    return files


def build_manifest(*, campaigns, grids, paths: dict, hash_fn=None) -> dict:
    """``{"schema", "workspace_root", "campaigns": {name: {str(n): {"files": [...], "missing": [...],
    "total_bytes": ...}}}}``. ``hash_fn`` defaults to sha256 (overridable for tests, never for real use)."""
    from p_shared import runner as _runner

    hash_fn = hash_fn or _runner.sha256_file
    out = {"schema": MANIFEST_SCHEMA, "workspace_root": str(_WORKSPACE), "campaigns": {}}
    for campaign in campaigns:
        out["campaigns"][campaign] = {}
        for n in grids:
            entries = []
            missing = []
            total = 0
            seen = set()
            for path in campaign_files(campaign, n, paths):
                key = _rel(path) if path.exists() else None
                if key is not None and key in seen:
                    continue
                if key is not None:
                    seen.add(key)
                if not path.is_file():
                    missing.append(str(path))
                    continue
                size = path.stat().st_size
                entries.append({"path": key, "bytes": size, "sha256": hash_fn(path)})
                total += size
            out["campaigns"][campaign][str(n)] = {"files": entries, "missing": missing, "total_bytes": total}
    return out


def verify_manifest(manifest: dict, oracle_root: Path, *, campaigns=None, grids=None) -> list[str]:
    """Verify every manifest file's hash under ``oracle_root``. Returns a
    list of human-readable error strings (empty means every file matched);
    never raises for a missing/mismatched file itself so a caller can report
    every failure at once, not just the first."""
    from p_shared import runner as _runner

    errors: list[str] = []
    oracle_root = Path(oracle_root)
    for campaign, by_grid in manifest["campaigns"].items():
        if campaigns is not None and campaign not in campaigns:
            continue
        for n_str, record in by_grid.items():
            if grids is not None and int(n_str) not in grids:
                continue
            for entry in record["files"]:
                full = oracle_root / entry["path"]
                if not full.is_file():
                    errors.append(f"{campaign} N{n_str}: missing {entry['path']}")
                    continue
                actual = _runner.sha256_file(full)
                if actual != entry["sha256"]:
                    errors.append(f"{campaign} N{n_str}: hash mismatch {entry['path']} "
                                  f"(expected {entry['sha256']}, got {actual})")
            for missing_path in record.get("missing", []):
                errors.append(f"{campaign} N{n_str}: not present anywhere locally: {missing_path}")
    return errors


def pack_oracle_tar(manifest: dict, oracle_root: Path, tar_path: Path, *, campaigns=None, grids=None) -> dict:
    """Write exactly the manifest's files (plus the manifest itself, as
    ``oracle_manifest.json``) into one tar at ``tar_path``, preserving each
    file's workspace-relative path. Returns ``{"tar_sha256", "tar_bytes",
    "file_count", "total_bytes"}``. Every listed file must exist under
    ``oracle_root`` (verified first, via :func:`verify_manifest`) -- refuses
    to pack a tar silently missing a file the manifest promises."""
    from p_shared import runner as _runner

    errors = verify_manifest(manifest, oracle_root, campaigns=campaigns, grids=grids)
    if errors:
        raise ValueError("refusing to pack: " + "; ".join(errors[:20]) +
                         (f" (+{len(errors) - 20} more)" if len(errors) > 20 else ""))
    oracle_root = Path(oracle_root)
    tar_path = Path(tar_path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    file_count = 0
    tmp = tar_path.with_suffix(tar_path.suffix + ".tmp")
    with tarfile.open(tmp, "w") as tar:
        for campaign, by_grid in manifest["campaigns"].items():
            if campaigns is not None and campaign not in campaigns:
                continue
            for n_str, record in by_grid.items():
                if grids is not None and int(n_str) not in grids:
                    continue
                for entry in record["files"]:
                    tar.add(oracle_root / entry["path"], arcname=entry["path"])
                    total_bytes += entry["bytes"]
                    file_count += 1
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo(name="oracle_manifest.json")
        info.size = len(manifest_bytes)
        import io
        tar.addfile(info, io.BytesIO(manifest_bytes))
    tmp.replace(tar_path)
    return {"tar_sha256": _runner.sha256_file(tar_path), "tar_bytes": tar_path.stat().st_size,
           "file_count": file_count, "total_bytes": total_bytes}
