#!/usr/bin/env python3
"""Verify and package the immutable P05 data needed by the direct campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
OLD = WORKSPACE / "work/p05_failed_58880191/p05"
ANALYSIS = WORKSPACE / "work/p05_failed_58880191/local_analysis"
PINNED_COMMIT = "2458dbf6a62b1ac790701fe0001f85e060bc8778"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def producer_source_path(relative):
    return REPO / relative


def build(output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OLD / "manifest.json").read_text())
    if manifest["content"]["commit"] != PINNED_COMMIT:
        raise ValueError("unexpected P05 producer commit")
    source_records = {}
    for name, expected in manifest["content"]["sources"].items():
        path = producer_source_path(name)
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"producer source dependency changed: {name}")
        source_records[name] = {"path": str(path.relative_to(REPO)), "sha256": actual}
    local_manifest = json.loads((ANALYSIS / "manifest.json").read_text())
    required_analysis_names = [f"N{n}.{suffix}.npz" for n in (32,48,64)
                               for suffix in ("diagnostic_reduction", "exact_input_diagnostic")]
    analysis_records = {}
    for name in required_analysis_names:
        expected = local_manifest["files"][name]
        path = ANALYSIS / name
        if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"saved P05 analysis artifact failed its manifest: {name}")
        analysis_records[name] = expected

    input_records = []
    input_manifest = manifest["content"]["inputs"]
    for entry in input_manifest["files"]:
        rel = Path(entry["path"])
        if rel.parts[0] == "geometry_artifacts":
            n = int(rel.parts[-2][:2])
            path = WORKSPACE / "prototype_runs/geometry" / f"hsx_fci_{n}x{n}x{n}" / rel.name
        elif rel.parts[0] == "DRBX":
            path = REPO.joinpath(*rel.parts[1:])
        else:
            path = WORKSPACE / rel.name
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise FileNotFoundError(f"required producer input unavailable: {path}")
        actual = file_hash(path) if rel.parts[0] in ("geometry_artifacts", "DRBX") else entry["sha256"]
        if actual != entry["sha256"]:
            raise ValueError(f"geometry producer hash mismatch: {rel}")
        input_records.append({"path": entry["path"],
                              "bytes": entry["bytes"], "sha256": actual,
                              "hash_source": "computed" if rel.parts[0] in ("geometry_artifacts", "DRBX") else "pinned campaign input manifest"})

    if file_hash(OLD / "reference_sidecar.json") != manifest["sidecar_sha256"]:
        raise ValueError("saved campaign runtime sidecar hash mismatch")

    sys.path.insert(0, str(REPO / "scripts"))
    from p05_structured_global import numerics as k
    cases = list(k.CASES)
    pairs = [list(pair) for pair in k.PAIRS]
    fields = list(k.FIELDS)
    zip_members = {}
    npz_records = {}
    for n in (32, 48, 64):
        with np.load(OLD / f"N{n}.observations.npz", allow_pickle=False) as source:
            observations = source["values"]
        observation_file = OLD / f"N{n}.observations.npz"
        observation_identity = {"path": str(observation_file.relative_to(WORKSPACE)),
                                "bytes": observation_file.stat().st_size, "sha256": file_hash(observation_file)}
        with np.load(ANALYSIS / f"N{n}.diagnostic_reduction.npz", allow_pickle=False) as source:
            forms = source["forms"]
            reference = source["reference"]
            volume = source["volume"]
            control_owner_ids = source["control_owner_ids"]
            control_nominal = source["control_nominal"]
            control_halfstep = source["control_halfstep"]
        with np.load(ANALYSIS / f"N{n}.exact_input_diagnostic.npz", allow_pickle=False) as source:
            exact_input_ABC = source["oracle_ABC"]
            exact_input_valid_slots = source["valid_slots"]
        owners = observations.shape[0]
        if observations.shape != (owners, len(fields)) or forms.shape != (owners, len(pairs), 4):
            raise ValueError(f"field/case shape mismatch at N{n}")
        if reference.shape != (owners, len(pairs)) or volume.shape != (owners,):
            raise ValueError(f"reference/volume shape mismatch at N{n}")
        if exact_input_ABC.shape != (owners, len(pairs), 3) or exact_input_valid_slots.shape != (len(pairs), 3):
            raise ValueError(f"exact-input shape mismatch at N{n}")
        if not all(np.isfinite(a).all() for a in (observations, forms, reference, volume, exact_input_ABC)):
            raise ValueError(f"nonfinite saved P05 source arrays at N{n}")
        if np.any(volume <= 0.0):
            raise ValueError(f"nonpositive P05 owner volume at N{n}")
        raw_name = f"N{n}.reuse.npz"
        import io
        buffer = io.BytesIO()
        old_o_slots = np.flatnonzero(exact_input_valid_slots[:, 2])
        old_exact_input_O = exact_input_ABC[:, old_o_slots, 2]
        np.savez_compressed(buffer, observations=observations, reference=reference, volume=volume,
                            old_C=forms[:, :, 2], old_U_minus_A=forms[:, :, 3] - forms[:, :, 0],
                            old_exact_input_O=old_exact_input_O,
                            old_exact_input_O_pair_slots=old_o_slots,
                            exact_input_valid_slots=exact_input_valid_slots,
                            control_owner_ids=control_owner_ids, control_nominal=control_nominal,
                            control_halfstep=control_halfstep)
        payload = buffer.getvalue()
        zip_members[raw_name] = payload
        npz_records[raw_name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                                 "resolution": n, "owners": owners,
                                 "observation_source": observation_identity,
                                 "arrays": ["observations", "reference", "volume", "old_C", "old_U_minus_A",
                                            "old_exact_input_O", "old_exact_input_O_pair_slots", "exact_input_valid_slots",
                                            "control_owner_ids", "control_nominal", "control_halfstep"]}

    bundle_manifest = {
        "schema": "drbx.p05-direct-midpoint-reuse-bundle-v1",
        "producer": {"commit": PINNED_COMMIT, "campaign_identity": manifest["identity"],
                     "campaign_manifest_sha256": file_hash(OLD / "manifest.json"),
                     "local_analysis_manifest_sha256": file_hash(ANALYSIS / "manifest.json"),
                     "producer_sources": source_records, "saved_analysis_files": analysis_records},
        "cases": cases, "pairs": pairs, "fields": fields,
        "old_exact_input_O_policy": "available only where exact_input_valid_slots[:,2] is true; actual-vorticity B/C remain unavailable",
        "jump_policy": "old_forms[:,:,3] - old_forms[:,:,0] is the saved owner-normalized U-A contribution",
        "runtime_geometry_inputs": input_records,
        "resolution_artifacts": npz_records,
        "source_data": {"P05 campaign_artifact": "work/p05_failed_58880191/p05",
                        "P05 local_reduction_artifact": "work/p05_failed_58880191/local_analysis"},
    }
    embedded = json.dumps(bundle_manifest, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    zip_members["reuse_manifest.json"] = embedded
    archive = output_dir / "reuse_inputs_v1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, data in sorted(zip_members.items()):
            zf.writestr(name, data)
    bundle_manifest["archive"] = {"file": archive.name, "bytes": archive.stat().st_size,
                                   "sha256": file_hash(archive)}
    json_write(output_dir / "reuse_manifest.json", bundle_manifest)
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": bundle_manifest["archive"]["sha256"],
                      "members": list(npz_records), "sources_verified": len(source_records),
                      "inputs_verified": len(input_records)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "reuse_bundle")
    build(parser.parse_args().output_dir)
