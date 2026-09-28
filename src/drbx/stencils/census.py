"""Field-independent face census over the P06 slot space (P08 step 1, task 1
-- see ``work/p08_step1_consolidation_design_20260928/design.md`` sections 1,
3 and 6).

The P06 campaigns (``p06_structured_global``, ``p06n_field_derived_global``)
enumerate every face of the structured raw grid in the "slot space"
``3*(n+1)*n**2``: ``(n+1)`` radial slots times ``n*n`` transverse positions,
for each of the three axes (radial, theta, eta). Because theta and eta are
periodic, slot ``n`` of the transverse axes is the *same physical face* as
slot ``0`` -- the accepted P06 campaign counted it twice (``dedupe=False``);
the fix (``p06_structured_global.numerics._periodic_duplicate_face``) drops
it. The P07 campaigns (``p07_combined_global.topology``) instead enumerate
only physical faces (no periodic duplicate, and no face at all where both
sides already share an owner -- an "internal" aggregation seam).

This module builds one table that carries both views at once: every slot of
the P06 space, tagged with its P07 topology-census id (``-1`` where the P07
census omits it), its P07 family code, its "zero-jump kind", boundary/seam
flags, and a ``legacy_alias_slots`` mask a caller can use to reproduce either
census (``dedupe=True`` drops the alias slots; ``dedupe=False`` keeps them,
matching the accepted P06 campaign's saved artifact).

The whole computation is a pure function of ``(n, raw_to_owner)``: nothing
here needs geometry, field values, or the P07 census's own machinery beyond
what it needs of ``raw_to_owner`` itself. Everything is re-derived (never
imported) from the frozen originals it must equal bitwise (verified in
``tests/test_stencils_census.py`` against real N32 data):

* ``scripts/p07_combined_global/topology.py``'s ``census``/``decode`` (the
  P07 topology-census id space, family classification, and the
  ``support_rings`` donor-capacity rule);
* ``scripts/p06_structured_global/numerics.py``'s ``_face_incidence`` (raw
  endpoints of a P06-slot-space key) and ``_periodic_duplicate_face`` (the
  theta/eta slot-``n`` alias predicate);
* ``scripts/p07_diffusion_global/numerics.py``'s ``face_keys``/``face_indices``
  (the P06 slot-space numbering itself -- this module's row order is exactly
  that numbering, so a caller already holding P06/P07-diffusion face ids can
  index this census directly by row);
* ``scripts/p05n_field_derived_global/operator.py`` and
  ``scripts/p06n_field_derived_global/operator.py``'s ``_face_kind`` (the
  three zero-jump-checked families, identical in both encodings).

Nothing here imports from ``scripts/``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

SCHEMA = "drbx.p-face-census.v1"

#: P07 topology-census family names, indexed by the stored ``family`` code
#: (``-1`` -- :data:`NO_FAMILY` -- means the P07 census omits this face).
FAMILY_NAMES: tuple[str, ...] = (
    "collapsed_r0",
    "physical_wall_quartic_BC",
    "adjacent_radial_quartic_BC",
    "centered_radial_cubic",
    "boundary_transverse_cubic_BC",
    "direct_ordinary_singleton",
    "direct_ringwise",
    "direct_coupled_quartic",
)

#: The three zero-jump-checked "kinds" ``_face_kind`` recognizes, indexed by
#: the stored ``kind`` code (``-1`` -- :data:`NO_KIND` -- means none apply).
KIND_NAMES: tuple[str, ...] = (
    "physical_wall",
    "radial_n_minus_1",
    "transverse_last_two_layers",
)

NO_FAMILY = -1
NO_KIND = -1
NO_ID = -1


# ---------------------------------------------------------------------------
# The P06 slot space: keys, raw incidence, and the periodic-alias predicate.
# ---------------------------------------------------------------------------
def _p06_slot_keys(n: int) -> np.ndarray:
    """Every ``(axis, i, j, k)`` key of the P06 slot space, in the canonical
    row order ``scripts/p07_diffusion_global/numerics.py``'s ``face_keys``/
    ``face_indices`` use: radial block first (``(n+1, n, n)``, C order), then
    theta (``(n, n+1, n)``), then eta (``(n, n, n+1)``). A row's index in the
    returned array is therefore exactly that module's face id for the row's
    key, and vice versa -- re-derived here, not imported.
    """
    blocks = []
    for axis, shape in enumerate(((n + 1, n, n), (n, n + 1, n), (n, n, n + 1))):
        grids = np.meshgrid(*(np.arange(s) for s in shape), indexing="ij")
        count = shape[0] * shape[1] * shape[2]
        block = np.empty((count, 4), dtype=np.int64)
        block[:, 0] = axis
        block[:, 1] = grids[0].ravel()
        block[:, 2] = grids[1].ravel()
        block[:, 3] = grids[2].ravel()
        blocks.append(block)
    return np.concatenate(blocks, axis=0)


def _raw_face_incidence(n: int, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw-cell face incidence with periodic theta/eta and radial exterior
    ends, re-derived from ``p06_structured_global.numerics._face_incidence``
    (same formula: face ``i`` on the radial axis separates raw cells ``i-1``
    and ``i``; ``i=0``/``i=n`` have one exterior end each). Returns
    ``(lower, lower_valid, upper, upper_valid)`` raw flat cell indices."""
    lower = np.full(len(keys), -1, dtype=np.int64)
    upper = np.full(len(keys), -1, dtype=np.int64)
    for axis in range(3):
        rows = np.flatnonzero(keys[:, 0] == axis)
        if not len(rows):
            continue
        xyz = np.asarray(keys[rows, 1:], dtype=np.int64)
        m = xyz[:, axis]
        lower_valid = (m > 0) if axis == 0 else np.ones(len(rows), dtype=bool)
        upper_valid = (m < n) if axis == 0 else np.ones(len(rows), dtype=bool)
        lo = xyz.copy(); hi = xyz.copy()
        lo[:, axis] = (m - 1) % n
        hi[:, axis] = m % n
        lower[rows[lower_valid]] = np.ravel_multi_index(lo[lower_valid].T, (n, n, n))
        upper[rows[upper_valid]] = np.ravel_multi_index(hi[upper_valid].T, (n, n, n))
    return lower, lower >= 0, upper, upper >= 0


def _periodic_duplicate(n: int, keys: np.ndarray) -> np.ndarray:
    """True for theta slot ``n`` and eta slot ``n``: the same physical face
    as slot ``0``, re-derived from
    ``p06_structured_global.numerics._periodic_duplicate_face``."""
    return ((keys[:, 0] == 1) & (keys[:, 2] == n)) | ((keys[:, 0] == 2) & (keys[:, 3] == n))


def _face_kind_code(n: int, axis: np.ndarray, i: np.ndarray) -> np.ndarray:
    """The three zero-jump-checked families, re-derived from
    ``p05n_field_derived_global.operator._face_kind`` /
    ``p06n_field_derived_global.operator._face_kind`` (identical in both)."""
    kind = np.full(len(axis), NO_KIND, dtype=np.int8)
    radial = axis == 0
    kind[radial & (i == n)] = 0
    kind[radial & (i == n - 1)] = 1
    kind[(~radial) & (i >= n - 2)] = 2
    return kind


# ---------------------------------------------------------------------------
# The P07 topology-census id space and family classification.
# ---------------------------------------------------------------------------
def _profile(n: int, ro3: np.ndarray) -> np.ndarray:
    """Owner multiplicity per radial layer, re-derived from
    ``scripts/p07_combined_global/topology.py``'s ``census`` (the same
    per-layer theta sample at ``k=0`` and uniformity check)."""
    profile = np.empty(n, dtype=np.int64)
    for i in range(n):
        owners = ro3[i, :, 0]
        counts = np.bincount(owners - owners.min())
        nonzero = np.unique(counts[counts > 0])
        if nonzero.size != 1:
            raise ValueError(f"owner multiplicity is not uniform at radial layer {i}")
        profile[i] = int(counts.max())
    return profile


def _support_layers(i_values: np.ndarray, lo_offset: int, hi_offset: int) -> np.ndarray:
    offsets = np.arange(lo_offset, hi_offset)
    layers = i_values[:, None] + offsets[None, :]
    return np.where(layers < 0, -layers - 1, layers)


def _classify_direct(idxs: np.ndarray, profile: np.ndarray, angular: np.ndarray,
                      lo_offset: int, hi_offset: int, n: int) -> np.ndarray:
    """The three "direct" families (singleton/ringwise/coupled), re-derived
    from ``topology.py``'s ``support_rings`` donor-capacity rule."""
    layers = _support_layers(idxs, lo_offset, hi_offset)
    if np.any((layers < 0) | (layers >= n)):
        raise ValueError("a support-ring layer leaves the grid; n is too small for this family scheme")
    plain = np.all(profile[layers] == 1, axis=1)
    ring = np.all(angular[layers] >= 7, axis=1)
    return np.where(plain, 5, np.where(ring, 6, 7)).astype(np.int8)


def _family_by_layer(n: int, profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Family code per radial layer, re-derived from ``topology.py``'s
    ``census``. Returns ``(fam_radial (n+1,), fam_trans (n,))``: ``fam_trans``
    serves both the theta and eta axes (the classification depends only on
    the radial layer, never on which transverse axis)."""
    if n < 7:
        raise ValueError("the P07 family scheme's near-wall thresholds require n >= 7")
    if np.any(n % profile != 0):
        raise ValueError("owner multiplicity must evenly divide the grid resolution")
    angular = n // profile

    fam_radial = np.full(n + 1, NO_FAMILY, dtype=np.int8)
    ridx = np.arange(n + 1)
    fam_radial[ridx == 0] = 0
    fam_radial[ridx == n] = 1
    fam_radial[np.isin(ridx, (n - 1, n - 2))] = 2
    fam_radial[(ridx >= n - 6) & (ridx <= n - 3)] = 3
    todo = np.flatnonzero(fam_radial == NO_FAMILY)
    if todo.size:
        fam_radial[todo] = _classify_direct(todo, profile, angular, -2, 2, n)

    fam_trans = np.full(n, NO_FAMILY, dtype=np.int8)
    tidx = np.arange(n)
    fam_trans[tidx >= n - 2] = 4
    todo = np.flatnonzero(fam_trans == NO_FAMILY)
    if todo.size:
        fam_trans[todo] = _classify_direct(todo, profile, angular, -1, 3, n)

    return fam_radial, fam_trans


def _p07_topology_ids_and_family(n: int, ro3: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Dense per-axis P07 topology-census id and family arrays, ``-1``
    wherever the face is internal to one owner (both sides share an owner --
    the aggregation seam the P07 census omits). Radial has shape
    ``(n+1, n, n)``; theta/eta have shape ``(n, n, n)`` (no alias slot --
    this is the P07 census's own, non-duplicated numbering)."""
    radial_lo = np.full((n + 1, n, n), -1, dtype=np.int64)
    radial_hi = radial_lo.copy()
    radial_lo[1:n] = ro3[:-1]; radial_lo[n] = ro3[-1]
    radial_hi[1:n] = ro3[1:]; radial_hi[0] = ro3[0]; radial_hi[n] = -1
    theta_lo = np.roll(ro3, 1, axis=1); theta_hi = ro3
    eta_lo = np.roll(ro3, 1, axis=2); eta_hi = ro3

    radial_valid = radial_lo != radial_hi
    theta_valid = theta_lo != theta_hi
    eta_valid = eta_lo != eta_hi

    off = (n + 1) * n * n
    off2 = off + n ** 3
    ri, rj, rk = np.meshgrid(np.arange(n + 1), np.arange(n), np.arange(n), indexing="ij")
    radial_id = (ri * n * n + rj * n + rk).astype(np.int64)
    ti, tj, tk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    local_id = (ti * n * n + tj * n + tk).astype(np.int64)
    theta_id = off + local_id
    eta_id = off2 + local_id

    radial_id = np.where(radial_valid, radial_id, NO_ID)
    theta_id = np.where(theta_valid, theta_id, NO_ID)
    eta_id = np.where(eta_valid, eta_id, NO_ID)

    profile = _profile(n, ro3)
    fam_radial, fam_trans = _family_by_layer(n, profile)

    radial_family = np.where(radial_valid, fam_radial[:, None, None], NO_FAMILY).astype(np.int8)
    trans_family = np.broadcast_to(fam_trans[:, None, None], (n, n, n))
    theta_family = np.where(theta_valid, trans_family, NO_FAMILY).astype(np.int8)
    eta_family = np.where(eta_valid, trans_family, NO_FAMILY).astype(np.int8)

    ids = {0: radial_id, 1: theta_id, 2: eta_id}
    families = {0: radial_family, 1: theta_family, 2: eta_family}
    return ids, families


def _gather_dense(axis: np.ndarray, i: np.ndarray, j: np.ndarray, k: np.ndarray,
                   dense_by_axis: dict[int, np.ndarray]) -> np.ndarray:
    sample = next(iter(dense_by_axis.values()))
    out = np.full(axis.shape, -1, dtype=sample.dtype)
    for code, dense in dense_by_axis.items():
        mask = axis == code
        out[mask] = dense[i[mask], j[mask], k[mask]]
    return out


def _array(value) -> np.ndarray:
    return np.asarray(value)


def _hash_arrays(names_and_arrays) -> "hashlib._Hash":
    digest = hashlib.sha256()
    for name, array in names_and_arrays:
        digest.update(name.encode("utf-8"))
        digest.update(repr(array.dtype).encode("utf-8"))
        digest.update(repr(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest


# Field order fixes the identity hash and the npz layout; do not reorder
# without also bumping SCHEMA (identity would silently change meaning).
_ARRAY_FIELDS = (
    "axis", "i", "j", "k",
    "raw_lo", "raw_hi", "owner_lo", "owner_hi",
    "p07_id", "family", "kind",
    "collapsed_r0", "physical_wall", "theta_seam", "eta_seam",
    "internal_to_owner", "owner_boundary", "legacy_alias_slots",
)


@dataclass(frozen=True)
class FaceCensus:
    """The full P06 slot-space face census: ``3*(n+1)*n**2`` rows, one per
    ``(axis, i, j, k)`` slot, in the canonical P06/P07-diffusion face-id row
    order (see :func:`_p06_slot_keys`).

    Every array below shares that row count. ``raw_lo``/``raw_hi`` are raw
    flat cell indices (``-1`` where that side is exterior); ``owner_lo``/
    ``owner_hi`` are the corresponding owner ids (``-1`` likewise).
    ``p07_id``/``family``/``kind`` are ``-1`` where not applicable (an
    internal-to-owner face never gets a P07 id or family; a face outside the
    three zero-jump-checked kinds never gets a kind).  ``legacy_alias_slots``
    marks the theta/eta slot-``n`` rows the accepted P06 campaign double
    counted (``dedupe=False``); the deduplicated census
    (``dedupe=True``) is ``~legacy_alias_slots``.

    ``schema``/``identity`` are recomputed on save and checked on load, never
    trusted from a caller.
    """

    schema: str
    identity: str
    n: int
    axis: np.ndarray
    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    raw_lo: np.ndarray
    raw_hi: np.ndarray
    owner_lo: np.ndarray
    owner_hi: np.ndarray
    p07_id: np.ndarray
    family: np.ndarray
    kind: np.ndarray
    collapsed_r0: np.ndarray
    physical_wall: np.ndarray
    theta_seam: np.ndarray
    eta_seam: np.ndarray
    internal_to_owner: np.ndarray
    owner_boundary: np.ndarray
    legacy_alias_slots: np.ndarray

    @classmethod
    def build(cls, n: int, raw_to_owner) -> "FaceCensus":
        """Build the census purely from ``(n, raw_to_owner)``.

        ``raw_to_owner`` is the per-raw-cell owner id, shape ``(n, n, n)`` or
        flat ``(n**3,)`` (the same array ``PointRowContext.from_arrays``
        calls ``raw_to_owner``/stores as ``ro``).
        """
        n = int(n)
        ro3 = np.asarray(raw_to_owner)
        if ro3.shape == (n ** 3,):
            ro3 = ro3.reshape(n, n, n)
        elif ro3.shape != (n, n, n):
            raise ValueError("raw_to_owner must have shape (n, n, n) or (n**3,)")
        ro3 = ro3.astype(np.int64)
        if np.any(ro3 < 0):
            raise ValueError("raw_to_owner must contain only non-negative owner ids")
        ro_flat = ro3.reshape(-1)

        keys = _p06_slot_keys(n)
        axis, i, j, k = (keys[:, c] for c in range(4))

        raw_lo, raw_lo_valid, raw_hi, raw_hi_valid = _raw_face_incidence(n, keys)
        owner_lo = np.where(raw_lo_valid, ro_flat[np.clip(raw_lo, 0, n ** 3 - 1)], -1)
        owner_hi = np.where(raw_hi_valid, ro_flat[np.clip(raw_hi, 0, n ** 3 - 1)], -1)

        legacy_alias_slots = _periodic_duplicate(n, keys)
        rep_j = np.where((axis == 1) & (j == n), 0, j)
        rep_k = np.where((axis == 2) & (k == n), 0, k)

        internal_to_owner = (owner_lo >= 0) & (owner_hi >= 0) & (owner_lo == owner_hi)
        owner_boundary = (owner_lo >= 0) & (owner_hi >= 0) & (owner_lo != owner_hi)
        collapsed_r0 = (axis == 0) & (i == 0)
        physical_wall = (axis == 0) & (i == n)
        theta_seam = (axis == 1) & ((j == 0) | (j == n))
        eta_seam = (axis == 2) & ((k == 0) | (k == n))

        dense_ids, dense_families = _p07_topology_ids_and_family(n, ro3)
        p07_id = _gather_dense(axis, i, rep_j, rep_k, dense_ids)
        family = _gather_dense(axis, i, rep_j, rep_k, dense_families)
        kind = _face_kind_code(n, axis, i)

        arrays = dict(
            axis=axis.astype(np.int8), i=i.astype(np.int32), j=j.astype(np.int32), k=k.astype(np.int32),
            raw_lo=raw_lo.astype(np.int64), raw_hi=raw_hi.astype(np.int64),
            owner_lo=owner_lo.astype(np.int64), owner_hi=owner_hi.astype(np.int64),
            p07_id=p07_id.astype(np.int64), family=family.astype(np.int8), kind=kind.astype(np.int8),
            collapsed_r0=collapsed_r0, physical_wall=physical_wall,
            theta_seam=theta_seam, eta_seam=eta_seam,
            internal_to_owner=internal_to_owner, owner_boundary=owner_boundary,
            legacy_alias_slots=legacy_alias_slots,
        )
        identity = cls._compute_identity(n, arrays)
        return cls(schema=SCHEMA, identity=identity, n=n, **arrays)

    @staticmethod
    def _compute_identity(n: int, arrays: dict) -> str:
        ordered = [(name, _array(arrays[name])) for name in _ARRAY_FIELDS]
        digest = _hash_arrays(ordered)
        digest.update(SCHEMA.encode("utf-8"))
        digest.update(repr(int(n)).encode("utf-8"))
        return digest.hexdigest()

    def _recompute_identity(self) -> str:
        return self._compute_identity(self.n, {name: getattr(self, name) for name in _ARRAY_FIELDS})

    def verify(self) -> None:
        """Raise ``ValueError`` if the stored identity no longer matches the
        arrays (hand-edited, truncated, or otherwise corrupted content)."""
        if self.schema != SCHEMA:
            raise ValueError(f"unsupported face-census schema {self.schema!r}")
        recomputed = self._recompute_identity()
        if recomputed != self.identity:
            raise ValueError(
                "face-census identity does not match its arrays "
                f"(stored {self.identity!r}, recomputed {recomputed!r})"
            )

    def save(self, path: str | Path) -> None:
        """Write every array plus ``schema``/``identity``/``n`` to one npz file."""
        self.verify()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {name: getattr(self, name) for name in _ARRAY_FIELDS}
        payload["schema"] = np.array(self.schema)
        payload["identity"] = np.array(self.identity)
        payload["n"] = np.array(self.n, dtype=np.int64)
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, *, expected_identity: str | None = None) -> "FaceCensus":
        """Read a file written by :meth:`save`, verifying its identity.

        Raises ``ValueError`` if the file's own arrays no longer match its
        stored identity, or if ``expected_identity`` is given and differs
        from the file's identity (a caller asking for a specific, now-stale
        census)."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            schema = str(data["schema"])
            identity = str(data["identity"])
            n = int(data["n"])
            arrays = {name: np.asarray(data[name]) for name in _ARRAY_FIELDS}
        instance = cls(schema=schema, identity=identity, n=n, **arrays)
        instance.verify()
        if expected_identity is not None and instance.identity != expected_identity:
            raise ValueError(
                f"face-census identity {instance.identity!r} at {path} does not match "
                f"the expected identity {expected_identity!r}"
            )
        return instance

    @property
    def slot_count(self) -> int:
        return 3 * (self.n + 1) * self.n * self.n

    def keys(self) -> np.ndarray:
        """The ``(S, 4)`` ``(axis, i, j, k)`` key of every row, as ``int64``."""
        return np.column_stack((self.axis, self.i, self.j, self.k)).astype(np.int64)

    def dedupe_mask(self, *, dedupe: bool) -> np.ndarray:
        """The row mask selecting the accepted P06 census: every slot
        (``dedupe=False``, reproducing the accepted campaign's double count
        of the periodic seam) or only the deduplicated physical faces
        (``dedupe=True``, dropping :attr:`legacy_alias_slots`)."""
        if dedupe:
            return ~self.legacy_alias_slots
        return np.ones(len(self.axis), dtype=bool)


assert tuple(f.name for f in fields(FaceCensus) if f.name in _ARRAY_FIELDS) == _ARRAY_FIELDS
