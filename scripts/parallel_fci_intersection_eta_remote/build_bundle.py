#!/usr/bin/env python3
"""Build the frozen portable source and immutable external-input manifests."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile
from typing import Iterable


SCHEMA = "drbx.parallel-fci-intersection-source-bundle-v1"
EXTERNAL_SCHEMA = "drbx.parallel-fci-intersection-external-inputs-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_files(root: Path) -> list[tuple[Path, Path]]:
    selected: dict[str, Path] = {}

    def add(path: Path, archive: Path | None = None) -> None:
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and not any(part.startswith(".") or part.endswith(".egg-info") for part in path.parts)
            and path.suffix != ".pyc"
        ):
            name = archive or path.relative_to(root)
            selected[name.as_posix()] = path

    package = root / "DRBX"
    for path in (package / "src").rglob("*"):
        add(path)
    add(package / "pyproject.toml")
    add(package / "LICENSE")

    for path in (root / "work").rglob("*.py"):
        add(path)

    data_roots = [
        root / "work" / "parallel_phase_a_q00_q04_20260918" / "q01_references",
    ]
    for directory in data_roots:
        for path in directory.rglob("*"):
            add(path)

    explicit = [
        "work/parallel_fci_intersection_eta_followup_20260922/ASSIGNMENT.md",
        "work/parallel_fci_intersection_eta_followup_20260922/FORMULATION.md",
        "work/parallel_fci_intersection_eta_followup_20260922/source_snapshot_optimized.json",
        "work/parallel_fci_intersection_eta_followup_20260922/intersection/run.py",
        "work/parallel_fci_intersection_eta_followup_20260922/intersection/run_optimized.py",
        "work/parallel_fci_intersection_eta_followup_20260922/intersection/preflight/arrays.npz",
        "work/parallel_fci_intersection_eta_followup_20260922/intersection/preflight/summary.json",
        "work/parallel_fci_intersection_eta_followup_20260922/intersection/preflight_optimized/equivalence.json",
        "work/parallel_fci_intersection_eta_followup_20260922/eta_gradient/run.py",
        "work/parallel_fci_intersection_eta_followup_20260922/eta_gradient/results/arrays.npz",
        "work/parallel_fci_intersection_eta_followup_20260922/eta_gradient/results/summary.json",
        "work/parallel_fci_shared_patch_20260922/report.md",
        "work/parallel_fci_shared_patch_20260922/final/summary.json",
        "work/parallel_fci_shared_patch_20260922/sensitivity/geometry/manifest.json",
        "work/parallel_fci_shared_patch_20260922/sensitivity/exact/arrays.npz",
        "work/parallel_fci_matched_tube_20260922/report.md",
        "work/parallel_fci_matched_tube_20260922/final/summary.json",
        "work/parallel_fci_cached_gradient_20260922/report.md",
        "work/parallel_fci_cached_gradient_20260922/summary.json",
        "work/parallel_fci_cached_gradient_20260922/comparison.npz",
        "work/parallel_fci_reopening_20260922/N32/selection.json",
        "work/parallel_q03_reference_closure_20260922/reference_bundle.json",
        "work/parallel_q03_reference_closure_20260922/identity.json",
        "work/parallel_q03_consistency_correction_20260919/checkpoints/targeted_operator.pkl",
    ]
    for relative in explicit:
        add(root / relative)
    return [(selected[name], Path(name)) for name in sorted(selected)]


def build_archive(files: Iterable[tuple[Path, Path]], output: Path) -> list[dict]:
    records = []
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for source, relative in files:
                    data = source.read_bytes()
                    info = tarfile.TarInfo(relative.as_posix())
                    info.size = len(data)
                    info.mtime = 0
                    info.mode = source.stat().st_mode & 0o777
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    from io import BytesIO

                    archive.addfile(info, BytesIO(data))
                    records.append({
                        "path": relative.as_posix(),
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    })
    temporary.replace(output)
    return records


def external_files(root: Path) -> list[dict]:
    sources = [
        (Path("hsx_metric_d58d392545fd3917efeb83b6.npz"), root / "hsx_metric_d58d392545fd3917efeb83b6.npz"),
        (Path("mgrid_res2p5cm_180pln.nc"), root / "mgrid_res2p5cm_180pln.nc"),
    ]
    for resolution in (32, 64):
        geometry = Path(
            f"geometry_artifacts/rlp_convergence_32_48_64_20260917/{resolution}x{resolution}x{resolution}"
        )
        geometry_source = (root / geometry).resolve()
        sources.extend(
            (geometry / path.relative_to(geometry_source), path)
            for path in sorted(geometry_source.rglob("*"))
            if path.is_file()
        )
    records = []
    for relative, path in sources:
        records.append({
            "path": relative.as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "frozen_source_bundle.tar.gz"
    records = build_archive(source_files(root), archive)
    atomic_json(output / "source_manifest.json", {
        "schema": SCHEMA,
        "archive": {
            "path": archive.name,
            "size": archive.stat().st_size,
            "sha256": sha256(archive),
        },
        "files": records,
    })
    atomic_json(output / "external_input_manifest.json", {
        "schema": EXTERNAL_SCHEMA,
        "files": external_files(root),
    })
    print(json.dumps({
        "archive": str(archive),
        "archive_size": archive.stat().st_size,
        "source_files": len(records),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
