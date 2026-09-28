#!/usr/bin/env python3
"""Build the small, tracked P06N preflight fixtures from local ``work/`` sources.

Run this locally (never on a remote allocation; never imported by campaign.py
or preflight.py -- it is the one sanctioned place in this package allowed to
reference ``work/`` paths), from DRBX/scripts:

    python p06n_field_derived_global/preflight_fixtures/extract.py \\
        --workspace-root "/Users/yxie/Desktop/HSX drbx" --resolutions 32 48 64

Writes, into this directory, for each resolution:
  N{n}.selection.json  -- byte-identical copy of the P06N bounded
                           verification run's owner selection
                           (work/p06n_bounded_20260927/N{n}.selection.json:
                           wall-adjacent/seam/control/hotspot owners), used
                           by preflight.py's structural gate (b) and the
                           seam gate (c).
  N{n}.replay.npz       -- only what preflight.py's gate (a) needs: the
                           accepted P06 campaign's own saved
                           ``candidate_centered``/``candidate_U``/
                           ``evolution_volume`` at the owners
                           ``work/p06_completed_58880303/p06/N{n}.preflight.npz``
                           selected, and its ``prepare.npz`` ``owner_values``
                           restricted to the donor-owner closure the replay's
                           row construction touches (computed by actually
                           running ``operator.owner_q1``/``owner_q3`` -- the
                           SAME calls preflight.py's gate (a) uses -- with a
                           ``donor_accumulator``, over all four accepted
                           states at every selected owner).
and fixtures_manifest.json: sha256 of every source file read, every fixture
written, and this script itself.

The donor closure is what makes the fixture small: preflight.py fills every
*other* owner with NaN at runtime, so a future code change that needs a donor
outside this closure fails loudly (NaN propagates into the action) instead of
silently comparing garbage.
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

import p06_structured_global.numerics as p06numerics  # noqa: E402
from p06n_field_derived_global import core  # noqa: E402
from p06n_field_derived_global import operator as p06n_operator  # noqa: E402
from p06n_field_derived_global.operator import Case, Role, owner_q1, owner_q3  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def accepted_role(name, field_index, ref, time_value):
    def evaluate(q, _name=name, _idx=field_index):
        v, g = p06numerics._evaluate_fields(_name, ref, q, time_value)
        return v[_idx], g[_idx]
    return Role(name=f"accepted:{name}:{field_index}", bc="dirichlet", evaluate=evaluate)


def accepted_case(name, ref, time_value):
    roles = [accepted_role(name, i, ref, time_value) for i in range(5)]
    return Case(n=roles[0], Te=roles[1], Ti=roles[2], omega=roles[3], phi=roles[4])


def donor_closure(t, S, ctx, ref, owners, normal_coefficients, patch_cache, time_value):
    closure = set()
    for state_idx, name in enumerate(p06numerics.FIELD_NAMES):
        case = accepted_case(name, ref, time_value)
        dummy_owner_values = np.ones((len(t.vol), 5))  # only donor_ids are used here, not values (n>0 avoids 1/n NaNs)
        for owner in owners:
            owner_q1(t, S, ref, owner, case, dummy_owner_values, normal_coefficients=normal_coefficients,
                     context=ctx, patch_cache=patch_cache, donor_accumulator=closure)
            owner_q3(t, S, ref, owner, case, dummy_owner_values, normal_coefficients=normal_coefficients,
                      context=ctx, patch_cache=patch_cache, dedupe=False, donor_accumulator=closure)
    return np.array(sorted(closure), dtype=np.int64)


def extract_one(workspace_root, n):
    workspace_root = Path(workspace_root)
    runtime_inputs = workspace_root / "work/p05_direct_midpoint_global_565e1d1a_HsoyFbJ3/runtime_inputs"
    sidecar = workspace_root / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
    bounded_root = workspace_root / "work/p06n_bounded_20260927"
    accepted_root = workspace_root / "work/p06_completed_58880303/p06"
    p06_config = json.loads((SCRIPTS / "p06_structured_global/configuration.json").read_text())
    time_value = float(p06_config["time"])

    selection_src = bounded_root / f"N{n}.selection.json"
    prepare_src = accepted_root / f"N{n}.prepare.npz"
    preflight_src = accepted_root / f"N{n}.preflight.npz"
    sources = {"selection": selection_src, "prepare": prepare_src, "preflight": preflight_src}
    source_hashes = {name: sha(path) for name, path in sources.items()}

    t, ref = core.load(runtime_inputs, sidecar, n)
    S = core.StructuredReconstruction(t)
    ctx = core.context(t)
    from p06n_field_derived_global import fields as p06n_fields
    normal_coefficients = lambda q: p06n_fields.normal(ref, q)  # noqa: E731

    with np.load(prepare_src, allow_pickle=False) as data:
        prepare = {name: data[name] for name in data.files}
    with np.load(preflight_src, allow_pickle=False) as data:
        preflight = {name: data[name] for name in data.files}
    if not np.array_equal(np.asarray(t.vol), prepare["owner_volume"]):
        raise ValueError(f"N{n}: stored owner-volume mismatch against the accepted P06 prepare bundle")

    owner_ids = np.asarray(preflight["owner_ids"], dtype=np.int64)
    patch_cache = {}
    donor_owner_ids = donor_closure(t, S, ctx, ref, [int(o) for o in owner_ids], normal_coefficients, patch_cache,
                                     time_value)
    # owner_values: (state, 5, owners_total) -> restrict to donor closure, keep state axis.
    donor_values = prepare["owner_values"][:, :, donor_owner_ids]

    selection_dst = HERE / f"N{n}.selection.json"
    selection_dst.write_bytes(selection_src.read_bytes())

    replay_dst = HERE / f"N{n}.replay.npz"
    tmp = replay_dst.with_name(replay_dst.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, owner_ids=owner_ids,
                             candidate_centered=preflight["candidate_centered"],
                             candidate_U=preflight["candidate_U"],
                             evolution_volume=preflight["evolution_volume"],
                             donor_owner_ids=donor_owner_ids, donor_values=donor_values,
                             owner_count=np.int64(len(t.vol)), time_value=np.float64(time_value))
    tmp.replace(replay_dst)

    return {
        "n": n, "selected_owners": len(owner_ids), "donor_owners": len(donor_owner_ids),
        "owner_count": len(t.vol), "source_hashes": source_hashes,
        "fixture_hashes": {"selection": sha(selection_dst), "replay": sha(replay_dst)},
        "fixture_bytes": {"selection": selection_dst.stat().st_size, "replay": replay_dst.stat().st_size},
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
        "schema": "drbx.p06n-preflight-fixtures-v1",
        "extractor_sha256": sha(HERE / "extract.py"),
        "resolutions": results,
    }
    (HERE / "fixtures_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("wrote fixtures_manifest.json", flush=True)


if __name__ == "__main__":
    main()
