"""Disk-backed whole-eta-plane P bundle, with cyclic neighbour reads."""

from ...operators.perp.sbp_owner import OwnerSBPSlice
import json
from pathlib import Path

import numpy as np

from ..._hashing import sha256_file as _sha
from ...operators.nodal.sbp_laplacian_plan import LaplacianMetric
from ...operators.perp.sbp_laplacian_apply import PLAN_HALO
from ...operators.nodal.nodal_plan import NodalMetric
from .sbp_setup import _write_sbp_setup


_SCHEMA = "drbx.perpendicular.owner_bundle.v1"


def _plane_edges(n_eta, parts):
    if type(parts) is not int or not 1 <= parts <= n_eta // 4:
        raise ValueError("P owner bundles require at least four eta planes per owner")
    return tuple((owner * n_eta + parts - 1) // parts for owner in range(parts + 1))


def _leaf_entry(root, path, shape, dtype):
    return dict(path=path.relative_to(root).as_posix(), shape=shape, dtype=dtype.str,
                bytes=path.stat().st_size, sha256=_sha(path))


def _record_leaf(root, owners, paths, name, shape, dtype, halo=0):
    for owner, path in zip(owners, paths, strict=True):
        local_shape = (owner["end"] - owner["begin"] + 2 * halo, *shape[1:]) if shape else ()
        owner["files"][name] = _leaf_entry(root, path, local_shape, dtype)
    return dict(shape=shape, dtype=dtype.str)


def _publish_owner_bundle(out, identity_path, identity, edges, directories, owners, leaves):
    inputs = identity["created"]["inputs"]
    sbp_leaves = {}
    for name, (shape, dtype, halo) in _write_sbp_setup(
            identity, directories, edges).items():
        paths = [directory / f"{name}.npy" for directory in directories]
        entry = _record_leaf(out, owners, paths, name, shape, dtype, halo)
        sbp_leaves[name] = dict(entry, halo=halo)
    manifest = dict(schema=_SCHEMA, N=identity["n"], n_eta=identity["n_eta"], P=identity["P"],
                    owner_count=len(owners), symmetry=inputs["symmetry"], field_periods=inputs["field_periods"],
                    edges=edges, leaves=leaves, sbp_leaves=sbp_leaves, owners=owners,
                    source_identity_sha256=_sha(identity_path))
    if identity_path.parent == out:
        manifest["source_identity"] = identity_path.name
    # Publish only after all owners and setup leaves have been checksummed.
    pending = out / "manifest.pending"
    pending.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    pending.rename(out / "manifest.json")
    return out


class OwnerPerpendicularBundle:
    """Open the catalog only. Each read maps just its requested owner files."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema"] != _SCHEMA:
            raise ValueError("Unknown P owner bundle schema")
        if name := self.manifest.get("source_identity"):
            if name != "identity.json" or _sha(self.root/name) != self.manifest["source_identity_sha256"]:
                raise ValueError("Owner P source identity differs from its manifest")
        self.parts = self.manifest["owner_count"]
        self.edges = _plane_edges(self.manifest["n_eta"], self.parts)
        if self.edges != tuple(self.manifest["edges"]):
            raise ValueError("Noncanonical P owner plane edges")
        self._owners = self.manifest["owners"]
        bounds = [(item["begin"], item["end"]) for item in self._owners]
        if bounds != list(zip(self.edges[:-1], self.edges[1:])):
            raise ValueError("P owner records differ from the plane edges")

    def _entries(self, owner, leaves):
        if type(owner) is not int or not 0 <= owner < self.parts:
            raise ValueError("Invalid P owner")
        files = self._owners[owner]["files"]
        keys = files if leaves is None else leaves
        for name in keys:
            entry = files[name]
            rel = f"owner{owner:04d}/{name}.npy"
            path = self.root / rel
            if not name.isidentifier() or entry["path"] != rel or path.resolve() != path:
                raise ValueError("Invalid P owner leaf path")
            yield name, path, entry

    def verify(self, owner: int, *, leaves=None) -> None:
        """Verify checksums and byte counts with bounded reads for one owner."""
        for _, path, entry in self._entries(owner, leaves):
            if path.stat().st_size != entry["bytes"] or _sha(path) != entry["sha256"]:
                raise ValueError(f"P owner file checksum failed: {path}")

    def _load(self, owner, leaves, specs):
        arrays = {}
        for name, path, entry in self._entries(owner, leaves):
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            spec = specs[name]
            shape = tuple(spec["shape"])
            length = self.edges[owner + 1] - self.edges[owner] + 2 * spec.get("halo", 0)
            expected = (length, *shape[1:]) if shape else ()
            if (array.shape != expected or list(expected) != entry["shape"]
                    or array.dtype.str != spec["dtype"] or spec["dtype"] != entry["dtype"]):
                raise ValueError(f"P owner leaf shape or dtype differs: {path}")
            arrays[name] = array
        return arrays

    def _with_halo(self, owner, arrays, halo, specs):
        names = [name for name, values in arrays.items() if values.ndim]
        left, right = (owner - 1) % self.parts, (owner + 1) % self.parts
        lower = arrays if left == owner else self._load(left, names, specs)
        upper = lower if right == left else self._load(right, names, specs)
        for name in names:
            arrays[name] = np.concatenate((lower[name][-halo:], arrays[name], upper[name][:halo]))
        return arrays

    def read(self, owner: int, *, halo: int = 0, leaves=None) -> dict[str, np.ndarray]:
        """Return plane-major metric leaves as lower halo, owned slab, upper halo."""
        if type(halo) is not int or not 0 <= halo <= 4:
            raise ValueError("P halo must be between zero and four planes")
        specs = self.manifest["leaves"]
        arrays = self._load(owner, specs if leaves is None else leaves, specs)
        return self._with_halo(owner, arrays, halo, specs) if halo else arrays

    def read_owner_sbp(self, owner: int, *, edges: tuple[int, ...]) -> OwnerSBPSlice:
        """Read C's owner SBP payload using the bundle's canonical periodic windows."""
        if edges != self.edges:
            raise ValueError("The requested SBP edges differ from the owner bundle")
        specs = self.manifest.get("sbp_leaves")
        if specs is None:
            raise ValueError("The owner bundle lacks precomputed SBP products")
        names = ("aee_h", "ge_diag", "tau", "tau_w", "rings_per_block")
        self.verify(owner, leaves=names)
        setup = self._load(owner, names, specs)
        aee_halo = specs["aee_h"]["halo"]
        if aee_halo not in (0, PLAN_HALO) or specs["ge_diag"]["halo"] != 0:
            raise ValueError("Unexpected SBP setup halo")
        if aee_halo == 0:
            for neighbor in {(owner-1) % self.parts, (owner+1) % self.parts} - {owner}:
                self.verify(neighbor, leaves=("aee_h",))
            setup["aee_h"] = self._with_halo(
                owner, {"aee_h": setup["aee_h"]}, PLAN_HALO, specs)["aee_h"]
        nodal = self.read(owner, leaves=("h", "B", "K"))
        lap = self.read(owner, halo=PLAN_HALO, leaves=("A", "jac", "ginv_u"))
        return OwnerSBPSlice(
            NodalMetric(nodal["h"], lap["jac"][PLAN_HALO:-PLAN_HALO], nodal["B"], nodal["K"]),
            LaplacianMetric(lap["A"], lap["jac"], lap["ginv_u"]),
            setup["aee_h"], float(setup["tau"]), float(setup["tau_w"]),
            setup["ge_diag"], int(setup["rings_per_block"]),
        )
