"""Input packing of the P10 evolved-MMS campaign (chunk C7): the metric files the bundles are built from, copied into one
relocatable directory with a hash manifest, for the remote run.

``bundle.build_bundle(arm, n, nodal_root=..., laplacian_root=...)`` opens exactly two files per ``(arm, n)`` (``bundle.metric_paths``;
``load_nodal_metric`` and ``p09_sbp_laplacian.common.MetricData`` read the arrays and the embedded ``meta`` / ``identity`` strings
from the ``.npz`` itself, the sidecar of the metrics is only *recorded* as ``sidecar_sha256``, never opened; the layouts are
analytic):

======================================================  ===================================================================
source (``configuration.json["roots"]``, ``WORKSPACE`` =  packed as
the parent of the repository)
======================================================  ===================================================================
``<roots.nodal_metric[arm]>/N<n>/nodal_metric.npz``      ``DEST/nodal/<arm>/N<n>/nodal_metric.npz``
``<roots.laplacian_metric>/<arm>/N<n>/laplacian_metric.npz``  ``DEST/laplacian/<arm>/N<n>/laplacian_metric.npz``
======================================================  ===================================================================

for ``arm`` in ``raw``, ``filtered`` and ``n`` in ``32, 48, 64`` (12 files), so that a bundle is built with
``nodal_root = DEST/nodal/<arm>`` and ``laplacian_root = DEST/laplacian`` (:func:`nodal_root`, :func:`laplacian_root`).
``DEST/inputs_manifest.json`` (deterministic: no timestamps) lists every file's relative path, size and sha256.

    python -m p10_evolved_mms.inputs pack DEST [--arms raw filtered] [--resolutions 32 48 64]
                                               [--nodal-root ARM=DIR ...] [--laplacian-root DIR]
    python -m p10_evolved_mms.inputs verify DIR

(run from ``scripts/``). ``verify`` re-hashes every listed file and exits 1 on any missing file, size or hash mismatch (files
present but not listed are reported as extras, not as errors). No JAX import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent
WORKSPACE = REPO.parent
CONFIG = json.loads((HERE / "configuration.json").read_text())

SCHEMA = "drbx.p10-inputs-v1"
MANIFEST = "inputs_manifest.json"
NODAL_FILE = "nodal_metric.npz"
LAPLACIAN_FILE = "laplacian_metric.npz"


def nodal_root(dest, arm: str) -> Path:
    """``nodal_root`` argument of ``bundle.build_bundle`` for a packed directory."""
    return Path(dest) / "nodal" / arm


def laplacian_root(dest) -> Path:
    """``laplacian_root`` argument of ``bundle.build_bundle`` for a packed directory."""
    return Path(dest) / "laplacian"


def default_roots(config: dict | None = None) -> tuple[dict, Path]:
    """``({arm: nodal root}, laplacian root)`` of ``configuration.json["roots"]`` (relative to the workspace)."""
    roots = (config or CONFIG)["roots"]
    return {arm: WORKSPACE / p for arm, p in roots["nodal_metric"].items()}, WORKSPACE / roots["laplacian_metric"]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_sha256(directory) -> str:
    """sha256 of the bytes of ``DIR/inputs_manifest.json`` (the identity of a packed input set)."""
    return sha256_file(Path(directory) / MANIFEST)


def rel_paths(arm: str, n: int) -> tuple[str, str]:
    """``(nodal, laplacian)`` relative paths of ``(arm, n)`` inside a packed directory."""
    return f"nodal/{arm}/N{n}/{NODAL_FILE}", f"laplacian/{arm}/N{n}/{LAPLACIAN_FILE}"


def source_files(arms=None, resolutions=None, nodal_roots: dict | None = None, laplacian_root_=None, config=None) -> list[dict]:
    """The files ``bundle.build_bundle`` reads for ``arms`` x ``resolutions`` (same path rule as ``bundle.metric_paths``):
    ``[{"arm", "n", "kind", "source", "rel"}]`` sorted by ``rel``."""
    cfg = config or CONFIG
    arms = tuple(cfg["arms"]) if arms is None else tuple(arms)
    resolutions = tuple(cfg["resolutions"]) if resolutions is None else tuple(int(n) for n in resolutions)
    d_nodal, d_lap = default_roots(cfg)
    nodal_roots = {**d_nodal, **{k: Path(v) for k, v in (nodal_roots or {}).items()}}
    lap_root = d_lap if laplacian_root_ is None else Path(laplacian_root_)
    out = []
    for arm in arms:
        for n in resolutions:
            rel_nodal, rel_lap = rel_paths(arm, n)
            out.append({"arm": arm, "n": n, "kind": "nodal", "source": nodal_roots[arm] / f"N{n}" / NODAL_FILE, "rel": rel_nodal})
            out.append({"arm": arm, "n": n, "kind": "laplacian", "source": lap_root / arm / f"N{n}" / LAPLACIAN_FILE,
                        "rel": rel_lap})
    return sorted(out, key=lambda e: e["rel"])


def _copy_hashed(src: Path, dst: Path) -> tuple[int, str]:
    """Copy ``src`` to ``dst`` (temporary file, fsync, atomic rename) and return ``(size, sha256)`` of what was written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    h, size = hashlib.sha256(), 0
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for block in iter(lambda: fi.read(1 << 20), b""):
            h.update(block)
            size += len(block)
            fo.write(block)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)
    return size, h.hexdigest()


def pack_inputs(dest, *, arms=None, resolutions=None, nodal_roots: dict | None = None, laplacian_root_=None, config=None) -> dict:
    """Copy the files of :func:`source_files` into ``dest`` and write ``dest/inputs_manifest.json``; returns the manifest.

    All sources must exist (otherwise nothing is copied and ``FileNotFoundError`` lists the missing ones). A file already at its
    destination with the source's size and hash is kept. The manifest records the hash of the bytes written; run
    :func:`verify_inputs` on the destination to re-check it."""
    dest = Path(dest)
    files = source_files(arms, resolutions, nodal_roots, laplacian_root_, config)
    missing = [str(f["source"]) for f in files if not Path(f["source"]).is_file()]
    if missing:
        raise FileNotFoundError("missing input files:\n  " + "\n  ".join(missing))
    entries = []
    for f in files:
        src, dst = Path(f["source"]), dest / f["rel"]
        if dst.is_file() and dst.stat().st_size == src.stat().st_size and sha256_file(dst) == sha256_file(src):
            size, digest = dst.stat().st_size, sha256_file(dst)
        else:
            size, digest = _copy_hashed(src, dst)
        entries.append({"path": f["rel"], "arm": f["arm"], "n": f["n"], "kind": f["kind"], "bytes": size, "sha256": digest,
                        "source": str(src)})
    manifest = {"schema": SCHEMA, "arms": sorted({f["arm"] for f in files}), "resolutions": sorted({f["n"] for f in files}),
                "files": entries}
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(tmp, dest / MANIFEST)
    return manifest


def load_manifest(directory) -> dict:
    path = Path(directory) / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    m = json.loads(path.read_text())
    if m.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported schema {m.get('schema')!r}")
    return m


def file_shas(directory) -> dict:
    """``{(arm, n): {"nodal": sha256, "laplacian": sha256}}`` from the manifest of a packed directory."""
    out: dict = {}
    for e in load_manifest(directory)["files"]:
        out.setdefault((e["arm"], int(e["n"])), {})[e["kind"]] = e["sha256"]
    return out


def verify_inputs(directory, require=None) -> dict:
    """Re-hash every file of ``DIR/inputs_manifest.json``.

    ``require``: optional iterable of ``(arm, n)`` pairs that must be listed (both files). Returns ``{"ok", "errors", "extras",
    "n_files", "total_bytes", "manifest_sha256"}``; ``ok`` is False on a missing / unreadable manifest, a missing, resized or
    re-hashed file, an unsafe path or a missing required pair."""
    directory = Path(directory)
    rep = {"ok": False, "errors": [], "extras": [], "n_files": 0, "total_bytes": 0, "manifest_sha256": None}
    try:
        manifest = load_manifest(directory)
    except (OSError, ValueError) as exc:
        rep["errors"].append(f"manifest: {exc}")
        return rep
    rep["manifest_sha256"] = manifest_sha256(directory)
    listed = set()
    for e in manifest["files"]:
        rel = e["path"]
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            rep["errors"].append(f"{rel}: unsafe path")
            continue
        listed.add(rel)
        path = directory / rel
        if not path.is_file():
            rep["errors"].append(f"{rel}: missing")
            continue
        size = path.stat().st_size
        if size != e["bytes"]:
            rep["errors"].append(f"{rel}: size {size} != manifest {e['bytes']}")
            continue
        digest = sha256_file(path)
        if digest != e["sha256"]:
            rep["errors"].append(f"{rel}: sha256 {digest} != manifest {e['sha256']}")
            continue
        rep["n_files"] += 1
        rep["total_bytes"] += size
    have = {(e["arm"], int(e["n"]), e["kind"]) for e in manifest["files"]}
    for arm, n in require or ():
        for kind in ("nodal", "laplacian"):
            if (arm, int(n), kind) not in have:
                rep["errors"].append(f"required input not in the manifest: {kind} metric of ({arm}, N{n})")
    for p in sorted(directory.rglob("*.npz")):
        rel = p.relative_to(directory).as_posix()
        if rel not in listed:
            rep["extras"].append(rel)
    rep["ok"] = not rep["errors"]
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="copy the metric files into DEST and write inputs_manifest.json")
    p.add_argument("dest", type=Path)
    p.add_argument("--arms", nargs="+")
    p.add_argument("--resolutions", nargs="+", type=int)
    p.add_argument("--nodal-root", action="append", default=[], metavar="ARM=DIR", help="override of the nodal-metric root of an arm")
    p.add_argument("--laplacian-root", type=Path)
    v = sub.add_parser("verify", help="re-hash the files of DIR against its manifest")
    v.add_argument("dir", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "pack":
        roots = dict(item.split("=", 1) for item in args.nodal_root)
        m = pack_inputs(args.dest, arms=args.arms, resolutions=args.resolutions, nodal_roots=roots, laplacian_root_=args.laplacian_root)
        for e in m["files"]:
            print(f"{e['bytes']:>12d}  {e['sha256']}  {e['path']}")
        print(f"packed {len(m['files'])} files, {sum(e['bytes'] for e in m['files'])} bytes into {args.dest}; "
              f"manifest sha256 {manifest_sha256(args.dest)}")
        return 0
    rep = verify_inputs(args.dir)
    for line in rep["errors"]:
        print("ERROR", line)
    for line in rep["extras"]:
        print("extra (unlisted)", line)
    print(f"{'OK' if rep['ok'] else 'FAILED'}: {rep['n_files']} files verified, {rep['total_bytes']} bytes, "
          f"manifest sha256 {rep['manifest_sha256']}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
