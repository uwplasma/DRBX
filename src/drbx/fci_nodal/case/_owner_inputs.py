"""Receipt-bound inputs to the owner stages."""

from contextlib import closing
from pathlib import Path

from .._hashing import sha256_file as digest
from ._receipts import read_json


ROLES = ("owner_perpendicular_bundle", "owner_parallel", "owner_parallel_construction",
         "indexed_geometry", "metric_cache", "field_table", "manufactured_ion_flow_fit")


def validate_inputs(inputs, *, parts=None, layout=None, indexed_identity=None):
    from ..operator_build.perp.owner_bundle import OwnerPerpendicularBundle
    from ..operator_build.parallel.indexed_input import IndexedInput

    p = OwnerPerpendicularBundle(inputs["owner_perpendicular_bundle"])
    q = read_json(inputs["owner_parallel"] / "metadata.json")
    construction_path = inputs["owner_parallel_construction"] / "manifest.json"
    construction = read_json(construction_path)
    expected = tuple(p.manifest[key] for key in ("N", "n_eta", "symmetry", "field_periods"))
    if expected != (q["N"], q["eta_planes"], q["symmetry"], q["field_periods"]):
        raise ValueError("Owner perpendicular and parallel layouts differ")
    if layout is not None and expected != layout:
        raise ValueError("Owner inputs differ from the case layout")
    if (parts is not None and p.parts != parts) or p.parts != q["owner_count"]:
        raise ValueError("Owner count differs from the case")
    if not p.manifest.get("sbp_leaves") or q["version"] != 3 or not q["complete"]:
        raise ValueError("Incomplete owner P SBP or Q application products")
    if indexed_identity is None:
        with closing(IndexedInput(inputs["indexed_geometry"])) as indexed:
            identity = indexed.identity()
    else:
        identity = indexed_identity
    edges = [e*p.manifest["P"] for e in p.edges]
    cut = [(a, b) for a, b in zip(p.edges[:-1], p.edges[1:])]
    if (q["indexed_geometry"] != identity or q["structure"]["owner_node_edges"] != edges
            or construction["index_binding"] != identity
            or [(row["begin"], row["end"]) for row in construction["owners"]] != cut):
        raise ValueError("Owner construction, application and indexed geometry bindings differ")
    if q["source_manifest_sha256"] != digest(construction_path):
        raise ValueError("Owner application construction manifest differs")
    moment_reduced_face_quadrature = construction["moment_reduced_face_quadrature"]
    if q["face_quadrature_degree"] != moment_reduced_face_quadrature["degree"] or q["recipe_sha256"] != moment_reduced_face_quadrature["sha256"]:
        raise ValueError("Owner application canonical moment-reduced face quadrature binding differs")
    return p, q, construction, identity


class OwnerParallelInputs:
    """Read construction plane leaves and indexed physical hit rows one owner at a time."""

    def __init__(self, root, metadata, construction, p, *, checksums=None):
        self.root = Path(root).resolve()
        self.checksums = checksums
        self.meta = metadata
        self.construction = construction
        self.edges = p.edges
        self.width = p.manifest["P"]
        self.n = self.edges[-1] * self.width

    def _read(self, owner, plane, name):
        import numpy as np

        # The established construction descriptor is relative to the runtime,
        # and this reader is bound to only its owners directory.
        directory = self.root.parent / self.construction["owners"][owner]["directory"]
        path = (directory / f"plane_{plane:05d}" / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Q construction leaf escapes its owner store")
        if self.checksums is not None and self.checksums.get(str(path))!=digest(path):
            raise ValueError(f"Q construction leaf checksum differs: {path}")
        return np.load(path, allow_pickle=False, mmap_mode="r")

    def planes(self, owner):
        return range(self.edges[owner], self.edges[owner+1])

    def _nodes(self, owner, node_ids):
        import numpy as np

        if not np.array_equal(node_ids, np.arange(
                self.edges[owner]*self.width, self.edges[owner+1]*self.width)):
            raise ValueError("Q construction requested with noncanonical node IDs")

    def mass(self, owner, *, node_ids):
        import numpy as np

        self._nodes(owner, node_ids)
        return tuple(np.concatenate([self._read(owner, plane, key+".npy")[:self.width]
                                     for plane in self.planes(owner)]) for key in ("wall_credited_parallel_mass", "MB_V"))

    def K(self, owner, *, node_ids):
        from scipy import sparse

        self._nodes(owner, node_ids)
        rows = []
        for plane in self.planes(owner):
            values, cols, ptr = (self._read(owner, plane, "K/"+key+".npy")
                                 for key in ("data", "indices", "indptr"))
            mat = sparse.csr_matrix((values, cols, ptr),
                                    shape=(len(ptr)-1, self.construction["plane_edges"][-1]))
            rows.append(mat)
        return sparse.vstack(rows, format="csr")

    def hits(self, owner, q_products, indexed):
        import numpy as np

        from ..operator_build.parallel.indexed_input import ROWS

        q_products = Path(q_products).resolve()
        entry = self.meta["layouts"]["walls"]["files"][str(owner)]
        id_path = (q_products / entry["path"]).resolve()
        if not id_path.is_relative_to(q_products) or digest(id_path) != entry["sha256"]:
            raise ValueError("Invalid Q hit layout")
        ids = np.load(id_path, allow_pickle=False, mmap_mode="r")
        rows, lengths, rr = [], [], []
        for plane in self.planes(owner):
            source = self._read(owner, plane, "wall_ids.npy")
            raw_rr = self._read(owner, plane, "wall_foot_to_hit_area_ratio.npy")
            for a in range(0, len(source), ROWS):
                batch = source[a:a+ROWS]
                rows.append(indexed.take("wall", batch))
                lengths.append(batch)
                rr.append(raw_rr[a:a+ROWS])
        order = np.argsort(np.concatenate(lengths), kind="stable")
        wall = np.concatenate(rows)[order]
        ids = ids[ids >= 0]
        if len(wall) != len(ids) or np.any(ids[1:] <= ids[:-1]):
            raise ValueError("Construction physical wall rows differ from Q hit layout")
        return ids, wall, np.concatenate(rr)[order]


def field_manifest(inputs, p):
    return dict(N=p.manifest["N"], eta_planes=p.manifest["n_eta"],
                symmetry=p.manifest["symmetry"], field_periods=p.manifest["field_periods"],
                files={key: inputs[key] for key in ("metric_cache", "field_table")})
