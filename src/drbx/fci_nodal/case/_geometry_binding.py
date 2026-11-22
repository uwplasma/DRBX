"""Verify the geometry consumed by gathered or indexed Q construction."""

from contextlib import closing
from pathlib import Path

from ._receipts import read_json


def verify_parallel_geometry_binding(q_runtime, indexed):
    q_runtime = Path(q_runtime).resolve()
    owner_manifest = q_runtime / "owners/manifest.json"
    owner = read_json(owner_manifest)

    from ..operator_build.parallel.indexed_input import IndexedInput
    from ..operator_build.parallel.construction_provenance import owner_plane_paths, verify_parallel_construction

    with closing(IndexedInput(indexed)) as reader:
        verify_parallel_construction(owner, reader, owner_plane_paths(q_runtime, owner))
    return owner["index_binding"]
