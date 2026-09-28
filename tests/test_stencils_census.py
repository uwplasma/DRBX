"""Tests for ``drbx.stencils.census`` (P08 step 1, task 1 -- see
``work/p08_step1_consolidation_design_20260928/design.md`` sections 1, 3, 6).

Two independent groups:

* Synthetic, always-run tests: API validation, the invariants any
  ``FaceCensus`` must satisfy regardless of the specific ``raw_to_owner``
  (family/p07_id are ``-1`` exactly on internal-to-owner faces, the seam/alias
  flags are pure functions of the key, ``kind`` matches an independent
  reimplementation of ``_face_kind``), and npz save/load round-trip +
  tamper/identity rejection -- against a small in-process, partly-aggregated
  structured grid (the same block-aggregation style
  ``tests/test_p05n_field_derived_campaign.py``'s ``geometry`` fixture uses,
  rebuilt here so this file needs no HSX workspace inputs for that half).
* Real-N32 equality against the frozen implementations this module
  re-derives (never imports) from, skipped cleanly if the local HSX
  workspace's ``geometry_artifacts/rlp_convergence_32_48_64_20260917`` is not
  present:

  - ``scripts/p07_combined_global/topology.py``'s ``census``/``decode``
    (P07 topology-census id, family, endpoints);
  - ``scripts/p06_structured_global/numerics.py``'s ``_face_incidence`` and
    ``_periodic_duplicate_face`` (raw endpoints and the alias predicate,
    over the *full* P06 slot space);
  - ``scripts/p06n_field_derived_global/core.py``'s ``all_face_ids`` /
    ``face_keys_for_ids``, and ``operator.owner_all_faces`` for both
    ``dedupe`` values.

``owner_all_faces`` needs one documented adjustment (see
``_owner_all_faces_from_census`` below): at the theta/eta periodic seam
(``j``/``k`` in ``{0, n}``) its per-owner traversal is asymmetric -- a raw
member's own (unwrapped) index only ever produces *one* of the two owners'
keys for that shared physical face, never both (this is the general form of
the "distinct id" quirk its own docstring documents for the alias side; the
canonical (``j``/``k`` == 0) side turns out to have the mirror-image
asymmetry). The census itself (``owner_lo``/``owner_hi``) is symmetric, as a
physical face table must be; reproducing ``owner_all_faces`` bitwise from it
needs that one extra rule, applied only at the seam.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
WORKSPACE = REPO.parent  # .../HSX drbx
sys.path.insert(0, str(REPO / "scripts"))

from drbx.stencils.census import FAMILY_NAMES, KIND_NAMES, NO_FAMILY, NO_ID, NO_KIND, SCHEMA, FaceCensus

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
N32 = 32


def _n32_geometry_available() -> bool:
    return (GEOMETRY / f"{N32}x{N32}x{N32}" / "rlp_topology.npz").is_file()


needs_n32_geometry = pytest.mark.skipif(
    not _n32_geometry_available(),
    reason="HSX workspace geometry_artifacts/rlp_convergence_32_48_64_20260917 N32 data is unavailable",
)


# ---------------------------------------------------------------------------
# Synthetic fixtures (no HSX workspace inputs).
# ---------------------------------------------------------------------------
def _identity_owner_grid(n: int) -> np.ndarray:
    """No aggregation at all: every raw cell is its own owner."""
    return np.arange(n ** 3, dtype=np.int64)


def _block_aggregated_grid(n: int) -> np.ndarray:
    """A partly-aggregated structured grid: theta cells group into blocks of
    decreasing size near the axis, one owner per (radial layer, block, eta
    index) -- the same construction
    ``tests/test_p05n_field_derived_campaign.py``'s ``geometry`` fixture
    uses (rebuilt here, independently, so this file is self-contained)."""
    ro = np.empty((n, n, n), dtype=np.int64)
    count = 0
    blocks = [32, 8, 4, 2]
    for i in range(n):
        block = blocks[i] if i < len(blocks) else 1
        for j in range(0, n, block):
            for k in range(n):
                ro[i, j:j + block, k] = count
                count += 1
    return ro.ravel()


# ---------------------------------------------------------------------------
# API / shape / validation.
# ---------------------------------------------------------------------------
def test_slot_count_matches_the_p06_slot_space_formula():
    n = 16
    census = FaceCensus.build(n, _identity_owner_grid(n))
    assert census.slot_count == 3 * (n + 1) * n * n
    assert len(census.axis) == census.slot_count
    for name in ("i", "j", "k", "raw_lo", "raw_hi", "owner_lo", "owner_hi", "p07_id", "family", "kind",
                 "collapsed_r0", "physical_wall", "theta_seam", "eta_seam",
                 "internal_to_owner", "owner_boundary", "legacy_alias_slots"):
        assert len(getattr(census, name)) == census.slot_count, name
    assert census.axis.dtype == np.int8
    assert census.family.dtype == np.int8 and census.kind.dtype == np.int8
    assert census.p07_id.dtype == np.int64
    assert census.collapsed_r0.dtype == np.bool_
    assert census.schema == SCHEMA


def test_build_rejects_malformed_raw_to_owner():
    n = 8
    with pytest.raises(ValueError):
        FaceCensus.build(n, np.arange(n ** 3 - 1))  # wrong size
    with pytest.raises(ValueError):
        FaceCensus.build(n, -np.ones((n, n, n), dtype=np.int64))  # negative owner ids


def test_build_accepts_flat_or_cubic_raw_to_owner():
    n = 8
    flat = _identity_owner_grid(n)
    cubic = flat.reshape(n, n, n)
    a = FaceCensus.build(n, flat)
    b = FaceCensus.build(n, cubic)
    assert a.identity == b.identity


def test_keys_round_trip_axis_i_j_k():
    n = 8
    census = FaceCensus.build(n, _identity_owner_grid(n))
    keys = census.keys()
    assert keys.shape == (census.slot_count, 4)
    assert np.array_equal(keys[:, 0], census.axis.astype(np.int64))
    assert np.array_equal(keys[:, 1], census.i.astype(np.int64))
    assert np.array_equal(keys[:, 2], census.j.astype(np.int64))
    assert np.array_equal(keys[:, 3], census.k.astype(np.int64))


# ---------------------------------------------------------------------------
# Invariants that must hold for any raw_to_owner, checked on the
# block-aggregated synthetic grid (exercises every family, including the two
# "direct" ring/coupled branches, and both internal-to-owner axes).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def block_census() -> FaceCensus:
    n = 32
    return FaceCensus.build(n, _block_aggregated_grid(n))


def test_family_and_p07_id_are_omitted_exactly_on_internal_faces(block_census):
    census = block_census
    assert np.array_equal(census.family == NO_FAMILY, census.internal_to_owner)
    assert np.array_equal(census.p07_id == NO_ID, census.internal_to_owner)
    # Every family code actually appears (this synthetic grid was chosen to
    # exercise all eight, including the ring/coupled "direct" branches).
    present = set(np.unique(census.family[census.family != NO_FAMILY]).tolist())
    assert present == set(range(8))
    assert census.internal_to_owner.sum() > 0  # both theta and eta, by construction


def test_flags_are_pure_functions_of_the_key(block_census):
    census = block_census
    n = census.n
    axis, i, j, k = (census.axis.astype(np.int64), census.i.astype(np.int64),
                     census.j.astype(np.int64), census.k.astype(np.int64))
    assert np.array_equal(census.collapsed_r0, (axis == 0) & (i == 0))
    assert np.array_equal(census.physical_wall, (axis == 0) & (i == n))
    assert np.array_equal(census.theta_seam, (axis == 1) & ((j == 0) | (j == n)))
    assert np.array_equal(census.eta_seam, (axis == 2) & ((k == 0) | (k == n)))
    assert np.array_equal(census.legacy_alias_slots, ((axis == 1) & (j == n)) | ((axis == 2) & (k == n)))


def test_kind_matches_an_independent_face_kind_reimplementation(block_census):
    """Reimplements ``_face_kind`` (p05n/p06n ``operator.py``) directly here,
    independent of ``census.py``'s own ``_face_kind_code``."""
    census = block_census
    n = census.n
    axis, i = census.axis.astype(np.int64), census.i.astype(np.int64)
    expected = np.full(len(axis), NO_KIND, dtype=np.int8)
    for row in range(len(axis)):
        a, ii = int(axis[row]), int(i[row])
        if a == 0:
            if ii == n:
                expected[row] = 0
            elif ii == n - 1:
                expected[row] = 1
        elif ii >= n - 2:
            expected[row] = 2
    assert np.array_equal(census.kind, expected)


def test_collapsed_physical_wall_internal_owner_boundary_partition_every_row(block_census):
    census = block_census
    exterior = census.collapsed_r0 | census.physical_wall
    categories = np.stack([census.collapsed_r0, census.physical_wall,
                            census.internal_to_owner, census.owner_boundary], axis=1)
    # Exactly one of the four categories holds for every row.
    assert np.array_equal(categories.sum(axis=1), np.ones(census.slot_count, dtype=np.int64))
    # Exterior rows (collapsed/physical-wall) have exactly one invalid
    # owner endpoint; interior rows (internal-to-owner/owner-boundary) have both valid.
    assert np.all(np.where(exterior, (census.owner_lo == -1) | (census.owner_hi == -1),
                            (census.owner_lo >= 0) & (census.owner_hi >= 0)))


def test_raw_and_owner_endpoints_agree_with_a_direct_recomputation(block_census):
    """Independently recomputes raw/owner endpoints from ``ro`` via
    ``np.roll``/modulo arithmetic (not by calling into ``census.py``'s
    private helpers) and compares against the stored arrays."""
    n = 32
    ro = _block_aggregated_grid(n).reshape(n, n, n)
    census = block_census
    axis, i, j, k = (census.axis.astype(np.int64), census.i.astype(np.int64),
                     census.j.astype(np.int64), census.k.astype(np.int64))

    for code, size in ((0, n + 1), (1, n), (2, n)):
        rows = np.flatnonzero(axis == code)
        if not len(rows):
            continue
        ii, jj, kk = i[rows], j[rows], k[rows]
        if code == 0:
            lo_valid = ii > 0
            hi_valid = ii < n
            lo_idx = np.ravel_multi_index(((ii - 1) % n, jj, kk), (n, n, n))
            hi_idx = np.ravel_multi_index((ii % n, jj, kk), (n, n, n))
        elif code == 1:
            lo_valid = np.ones(len(rows), dtype=bool)
            hi_valid = lo_valid
            lo_idx = np.ravel_multi_index((ii, (jj - 1) % n, kk), (n, n, n))
            hi_idx = np.ravel_multi_index((ii, jj % n, kk), (n, n, n))
        else:
            lo_valid = np.ones(len(rows), dtype=bool)
            hi_valid = lo_valid
            lo_idx = np.ravel_multi_index((ii, jj, (kk - 1) % n), (n, n, n))
            hi_idx = np.ravel_multi_index((ii, jj, kk % n), (n, n, n))
        expected_raw_lo = np.where(lo_valid, lo_idx, -1)
        expected_raw_hi = np.where(hi_valid, hi_idx, -1)
        assert np.array_equal(census.raw_lo[rows], expected_raw_lo)
        assert np.array_equal(census.raw_hi[rows], expected_raw_hi)
        ro_flat = ro.ravel()
        expected_owner_lo = np.where(lo_valid, ro_flat[np.clip(lo_idx, 0, n ** 3 - 1)], -1)
        expected_owner_hi = np.where(hi_valid, ro_flat[np.clip(hi_idx, 0, n ** 3 - 1)], -1)
        assert np.array_equal(census.owner_lo[rows], expected_owner_lo)
        assert np.array_equal(census.owner_hi[rows], expected_owner_hi)


# ---------------------------------------------------------------------------
# npz save/load, identity, and tamper rejection.
# ---------------------------------------------------------------------------
def test_save_load_round_trip_is_bitwise(tmp_path, block_census):
    census = block_census
    path = tmp_path / "N32.census.npz"
    census.save(path)
    reloaded = FaceCensus.load(path)
    assert reloaded.identity == census.identity
    assert reloaded.n == census.n
    for name in ("axis", "i", "j", "k", "raw_lo", "raw_hi", "owner_lo", "owner_hi",
                 "p07_id", "family", "kind", "collapsed_r0", "physical_wall",
                 "theta_seam", "eta_seam", "internal_to_owner", "owner_boundary",
                 "legacy_alias_slots"):
        assert np.array_equal(getattr(reloaded, name), getattr(census, name)), name


def test_load_rejects_a_tampered_array(tmp_path, block_census):
    census = block_census
    path = tmp_path / "N32.census.npz"
    census.save(path)
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    payload["family"] = payload["family"].copy()
    payload["family"][0] = payload["family"][0] + 1 if payload["family"][0] < 100 else 0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError):
        FaceCensus.load(path)


def test_load_rejects_an_unexpected_identity(tmp_path, block_census):
    census = block_census
    path = tmp_path / "N32.census.npz"
    census.save(path)
    with pytest.raises(ValueError):
        FaceCensus.load(path, expected_identity="not-the-real-identity")


def test_dedupe_mask_matches_legacy_alias_slots(block_census):
    census = block_census
    assert np.array_equal(census.dedupe_mask(dedupe=True), ~census.legacy_alias_slots)
    assert np.all(census.dedupe_mask(dedupe=False))


# ---------------------------------------------------------------------------
# Real N32 equality against the frozen implementations.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_n32():
    if not _n32_geometry_available():
        pytest.skip("HSX workspace geometry_artifacts/rlp_convergence_32_48_64_20260917 N32 data is unavailable")
    from p07_combined_global import topology as pt
    ro3, rv = pt.raw_owner(N32, str(WORKSPACE))
    census = FaceCensus.build(N32, ro3)
    return ro3, census


@needs_n32_geometry
def test_matches_topology_census_bitwise(real_n32, tmp_path):
    from p07_combined_global import topology as pt
    ro3, census = real_n32
    result = pt.census(N32, str(WORKSPACE), str(tmp_path))
    data = np.load(tmp_path / f"N{N32}.topology.npz")
    face_ids, endpoints, family = data["face_ids"], data["endpoints"], data["family"]
    keys_topo = pt.decode(N32, face_ids)

    census_keys = census.keys()
    key_to_row = {tuple(int(x) for x in kk): row for row, kk in enumerate(census_keys.tolist())}

    rows = np.array([key_to_row[tuple(int(x) for x in k)] for k in keys_topo])
    assert np.array_equal(census.p07_id[rows], face_ids)
    assert np.array_equal(census.family[rows], family)
    assert np.array_equal(census.owner_lo[rows], endpoints[:, 0])
    assert np.array_equal(census.owner_hi[rows], endpoints[:, 1])

    # Every non-alias row NOT in the P07 census must be internal-to-owner
    # (the P07 census's own omission rule) and carry no id/family.
    non_alias_rows = np.flatnonzero(~census.legacy_alias_slots)
    found = np.zeros(census.slot_count, dtype=bool)
    found[rows] = True
    omitted = non_alias_rows[~found[non_alias_rows]]
    assert np.all(census.internal_to_owner[omitted])
    assert np.all(census.p07_id[omitted] == NO_ID)
    assert np.all(census.family[omitted] == NO_FAMILY)

    # Design doc section 1: only theta internal faces at N32 (7424); no
    # radial or eta internal faces.
    internal_theta = int(np.sum(census.internal_to_owner & (census.axis == 1) & (census.j < N32)))
    internal_radial = int(np.sum(census.internal_to_owner & (census.axis == 0)))
    internal_eta = int(np.sum(census.internal_to_owner & (census.axis == 2) & (census.k < N32)))
    assert internal_theta == 7424
    assert internal_radial == 0
    assert internal_eta == 0

    print("\n[census N32] families:", {FAMILY_NAMES[c]: int((census.family == c).sum()) for c in range(8)})
    print("[census N32] kinds:", {KIND_NAMES[c]: int((census.kind == c).sum()) for c in range(3)})
    print("[census N32] alias slots:", int(census.legacy_alias_slots.sum()))
    print("[census N32] internal theta/radial/eta:", internal_theta, internal_radial, internal_eta)
    print("[census N32] distinct P07 faces:", len(face_ids), "== census non-internal rows:",
          int((~census.internal_to_owner).sum()))


@needs_n32_geometry
def test_matches_p06_face_incidence_and_periodic_duplicate_bitwise(real_n32):
    import p06_structured_global.numerics as p06numerics
    ro3, census = real_n32
    keys = census.keys()

    assert p06numerics._face_count(N32) == census.slot_count
    keys_ref = p06numerics.base._face_keys(N32, np.arange(p06numerics._face_count(N32)))
    assert np.array_equal(keys, keys_ref)

    lower, lower_valid, upper, upper_valid = p06numerics._face_incidence(N32, keys)
    assert np.array_equal(census.raw_lo, np.where(lower_valid, lower, -1))
    assert np.array_equal(census.raw_hi, np.where(upper_valid, upper, -1))

    dup = p06numerics._periodic_duplicate_face(N32, keys)
    assert np.array_equal(census.legacy_alias_slots, dup)


@needs_n32_geometry
def test_matches_p06n_all_face_ids_and_face_keys(real_n32):
    from p06n_field_derived_global import core as p06n_core
    ro3, census = real_n32
    ids = p06n_core.all_face_ids(N32)
    expected_rows = np.flatnonzero(~census.collapsed_r0)
    assert np.array_equal(ids, expected_rows)
    keys_for_ids = p06n_core.face_keys_for_ids(N32, ids)
    assert np.array_equal(keys_for_ids, census.keys()[expected_rows])


def _owner_all_faces_from_census(census: FaceCensus, owner: int, *, dedupe: bool) -> dict:
    """Reproduce ``p06n_field_derived_global.operator.owner_all_faces``'s
    ``{key: (lo, hi)}`` view for one owner, from the (symmetric) census.

    Away from the theta/eta periodic seam this is exactly "any row incident
    to ``owner``" (``owner_lo == owner or owner_hi == owner``). At the seam
    (``j``/``k`` in ``{0, n}``) ``owner_all_faces``'s traversal is
    asymmetric: a raw member's own (unwrapped) index produces the canonical
    key (``j``/``k`` == 0) only for the owner on the *high* side, and the
    alias key (``j``/``k`` == n) only for the owner on the *low* side --
    never both from the same owner, even though both keys' ``(lo, hi)``
    resolve (via ``% n``) to the same owner pair. See that function's own
    docstring point 2 for the alias half of this; the canonical half is its
    mirror image and is not separately documented there.
    """
    axis, j, k = census.axis.astype(np.int64), census.j.astype(np.int64), census.k.astype(np.int64)
    n = census.n
    theta_canonical = (axis == 1) & (j == 0)
    theta_alias = (axis == 1) & (j == n)
    eta_canonical = (axis == 2) & (k == 0)
    eta_alias = (axis == 2) & (k == n)
    seam = theta_canonical | theta_alias | eta_canonical | eta_alias
    lo_match = census.owner_lo == owner
    hi_match = census.owner_hi == owner
    incident = np.where(seam, np.where(theta_canonical | eta_canonical, hi_match, lo_match), lo_match | hi_match)
    mask = incident & (~census.collapsed_r0)
    if dedupe:
        mask &= ~census.legacy_alias_slots
    rows = np.flatnonzero(mask)
    keys = census.keys()
    out = {}
    for row in rows:
        key = tuple(int(x) for x in keys[row])
        lo = int(census.owner_lo[row]) if census.owner_lo[row] >= 0 else None
        hi = int(census.owner_hi[row]) if census.owner_hi[row] >= 0 else None
        out[key] = (lo, hi)
    return out


@needs_n32_geometry
def test_matches_owner_all_faces_both_dedupe_values(real_n32):
    from types import SimpleNamespace
    from p06n_field_derived_global import operator as p06n_operator
    ro3, census = real_n32
    ro_flat = ro3.reshape(-1)
    t = SimpleNamespace(n=N32, ro=ro_flat)

    num_owners = int(ro_flat.max()) + 1
    rng = np.random.default_rng(20260928)
    owners = {int(ro3[0, 0, 0]), int(ro3[-1, 0, 0]), int(ro3[N32 // 2, N32 // 2, N32 // 2])}
    owners.update(int(x) for x in rng.integers(0, num_owners, size=40))

    checked = 0
    for dedupe in (False, True):
        for owner in owners:
            expected = p06n_operator.owner_all_faces(t, owner, dedupe=dedupe)
            expected_map = {tuple(int(x) for x in key): (lo, hi) for key, lo, hi in expected}
            got_map = _owner_all_faces_from_census(census, owner, dedupe=dedupe)
            assert got_map == expected_map, (dedupe, owner)
            checked += 1
    assert checked == 2 * len(owners)
