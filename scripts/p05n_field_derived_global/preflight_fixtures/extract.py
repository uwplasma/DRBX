#!/usr/bin/env python3
"""Build the small, tracked P05N preflight fixtures from local ``work/`` sources.

Run this locally (never on the remote; never imported by campaign.py or
preflight.py -- it is the one sanctioned place in this package allowed to
reference ``work/`` paths):

    python extract.py --workspace-root "/Users/yxie/Desktop/HSX drbx" --resolutions 32 48 64

Writes, into this directory, for each resolution:
  N{n}.selection.json            -- byte-identical copy of the bounded
                                     verification run's 54-owner selection
                                     (wall-adjacent/seam/control owners).
  N{n}.accepted_p05_replay.npz   -- only what preflight.py's gate (ii) needs:
                                     the accepted P05 campaign's expected
                                     `centered`/`jump` rows at those 54
                                     owners, and its `observations` values
                                     restricted to the union of donor owners
                                     the replay's row construction touches
                                     (computed by actually running the same
                                     batched_cell_values/face_common_gradient/
                                     side_values calls preflight.py uses, with
                                     a donor_accumulator, over all 54 owners).
and fixtures_manifest.json: sha256 of every source file read, every fixture
written, and this script itself.

The donor closure is what makes fixtures small (a few hundred owners' worth
of 8-field doubles, not the full owner_results.npz/reuse.npz, which are
14-114 MB each): preflight.py fills every *other* owner with NaN at runtime,
so any row construction that (after a future code change) needs a donor
outside this closure fails loudly instead of silently reading stale zeros.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SCRIPTS = PACKAGE.parent
sys.path.insert(0, str(SCRIPTS))  # DRBX/scripts; never PACKAGE itself (shadows stdlib operator)

from p05n_field_derived_global import core  # noqa: E402
from p05n_field_derived_global import operator as p05n_operator  # noqa: E402
from p05_structured_global import numerics as oldnum  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def old_dirichlet_trace_fn(ref):
    def trace(q):
        return oldnum.boundary_trace(ref, q)
    return trace


def zero_normal_data_fn(n_fields):
    def fn(q):
        return np.zeros((len(q), n_fields))
    return fn


def donor_closure(t, S, ctx, ref, owner_values, normal_coefficients, owners):
    """Union of owner ids touched building every raw-cell and boundary-face
    row (both variants) for ``owners`` -- exactly preflight.py's gate (ii)
    row construction, instrumented with ``donor_accumulator``."""
    patch_cache = {}
    closure = set()
    trace_fn = old_dirichlet_trace_fn(ref)
    nd_fn = zero_normal_data_fn(owner_values.shape[1])
    for owner in owners:
        raw_ids, points, keys = p05n_operator.owner_raw(t, owner)
        core.batched_cell_values(t, S, owner_values, keys, points, normal_coefficients=normal_coefficients,
                                  ctx=ctx, patch_cache=patch_cache, dirichlet_trace_fn=trace_fn,
                                  normal_data_fn=nd_fn, donor_accumulator=closure)
        for key, lo, hi in p05n_operator.owner_boundary_faces(t, owner):
            points, _ = core.pk.num.quadrature(t.faces, np.array([key]), 3, face=True)
            points = points[0]
            core.batched_face_common_gradient(t, S, owner_values, key, points, normal_coefficients=normal_coefficients,
                                               ctx=ctx, patch_cache=patch_cache, dirichlet_trace_fn=trace_fn,
                                               normal_data_fn=nd_fn, donor_accumulator=closure)
            core.batched_side_values(t, S, owner_values, key, points, normal_coefficients=normal_coefficients,
                                      ctx=ctx, patch_cache=patch_cache, dirichlet_trace_fn=trace_fn,
                                      normal_data_fn=nd_fn, donor_accumulator=closure)
    return np.array(sorted(closure), dtype=np.int64)


def extract_one(workspace_root, n):
    workspace_root = Path(workspace_root)
    accepted_root = workspace_root / "work/p05_direct_midpoint_global_565e1d1a_HsoyFbJ3"
    bounded_root = workspace_root / "work/p05n_bounded_20260927"
    sidecar = workspace_root / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
    runtime_inputs = accepted_root / "runtime_inputs"

    selection_src = bounded_root / f"N{n}.selection.json"
    reuse_src = accepted_root / "reuse_inputs" / f"N{n}.reuse.npz"
    accepted_src = accepted_root / f"N{n}.owner_results.npz"
    sources = {"selection": selection_src, "reuse": reuse_src, "accepted": accepted_src}
    source_hashes = {name: sha(path) for name, path in sources.items()}

    t, ref = core.load(runtime_inputs, sidecar, n)
    S = core.StructuredReconstruction(t)
    ctx = core.context(t)
    normal_coefficients = lambda q: core.p05n_fields.normal(ref, q)  # noqa: E731

    selection = json.loads(selection_src.read_text())
    selected_owners = np.array([int(rec["owner"]) for rec in selection["all_owners"]], dtype=np.int64)

    with np.load(reuse_src, allow_pickle=False) as data:
        reuse = {name: data[name] for name in data.files}
    if not np.array_equal(t.vol, reuse["volume"]):
        raise ValueError(f"N{n}: stored owner-volume mismatch against the accepted P05 reuse bundle")
    with np.load(accepted_src, allow_pickle=False) as data:
        accepted = {name: data[name] for name in data.files}

    donor_owner_ids = donor_closure(t, S, ctx, ref, reuse["observations"], normal_coefficients, selected_owners)
    donor_values = reuse["observations"][donor_owner_ids]
    expected_centered = accepted["centered"][selected_owners]
    expected_jump = accepted["jump"][selected_owners]

    # Copy the selection byte-identically (do not re-serialize: JSON key
    # ordering/formatting must match the source exactly).
    selection_dst = HERE / f"N{n}.selection.json"
    selection_dst.write_bytes(selection_src.read_bytes())

    replay_dst = HERE / f"N{n}.accepted_p05_replay.npz"
    tmp = replay_dst.with_name(replay_dst.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, selected_owners=selected_owners, expected_centered=expected_centered,
                            expected_jump=expected_jump, donor_owner_ids=donor_owner_ids, donor_values=donor_values,
                            owner_count=np.int64(len(t.vol)))
    tmp.replace(replay_dst)

    return {
        "n": n, "selected_owners": len(selected_owners), "donor_owners": len(donor_owner_ids),
        "owner_count": len(t.vol), "source_hashes": source_hashes,
        "fixture_hashes": {"selection": sha(selection_dst), "accepted_p05_replay": sha(replay_dst)},
        "fixture_bytes": {"selection": selection_dst.stat().st_size, "accepted_p05_replay": replay_dst.stat().st_size},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace-root", type=Path, required=True)
    p.add_argument("--resolutions", type=int, nargs="+", choices=(32, 48, 64), default=[32, 48, 64])
    args = p.parse_args()

    results = {}
    for n in args.resolutions:
        print(f"extracting N{n} fixtures...", flush=True)
        results[str(n)] = extract_one(args.workspace_root, n)
        print(f"  N{n}: {results[str(n)]['donor_owners']} donor owners, "
              f"fixture bytes {results[str(n)]['fixture_bytes']}", flush=True)

    manifest = {
        "schema": "drbx.p05n-preflight-fixtures-v1",
        "extractor_sha256": sha(HERE / "extract.py"),
        "resolutions": results,
    }
    (HERE / "fixtures_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("wrote fixtures_manifest.json", flush=True)


if __name__ == "__main__":
    main()
