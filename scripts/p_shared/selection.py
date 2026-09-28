"""Frozen, deterministic replay-owner selection for the P08 step-1 N48/N64 gate.

Implements design task 6 ("Selection freeze") of
``work/p08_step1_consolidation_design_20260928/design.md`` (see its section 5,
"Replay gate" -> "Coverage", and section 6, task 6: "Selection freeze (after
1), before any saved array is read.").

This module is deliberately read-only over **geometry/topology** inputs only:
the frozen HSX RLP owner map (raw-to-owner assignment, raw volumes, grid
centers/faces), loaded exactly the way the accepted P05N/P06N/P07(N) research
packages load it (``perpendicular_structured.reconstruction.load_context``,
which itself resolves to ``drbx``-adjacent research kernels reading only
``base_geometry.npz`` and ``rlp_topology.npz``); plus a handful of small,
non-result JSON files:

- ``scripts/p05_direct_midpoint_global/configuration.json`` for the frozen
  P05 "eight deterministic angular locations" convention
  (``preflight_angular_fractions_eighths``), reused verbatim (see that
  package's ``select_preflight_owners``);
- ``scripts/p05n_field_derived_global/preflight_fixtures/N{n}.selection.json``
  and ``scripts/p06n_field_derived_global/preflight_fixtures/N{n}.selection.json``
  -- the **selection** files only, never the paired
  ``N{n}.accepted_p05_replay.npz`` / ``N{n}.replay.npz`` result fixtures that
  live in the same directories;
- ``scripts/p06_structured_global/configuration.json`` for the frozen
  ``preflight_hotspot_owners`` (the archived reference-error "hotspot" owner
  ids, cross-checked against the P06N fixture's own ``hotspots`` entries).

It never opens any saved campaign action/result array: no ``N*.raw.npz``,
``N*.faces.npz``, ``owner_results``, global ``action``/``N``/``D``/``R``/
``correction`` arrays, ``summary.json``, or any ``*.replay.npz`` /
``*.accepted_p05_replay.npz``. The owner selection this module computes is
frozen (written to disk, hashed) before any such array is ever read by the
later replay-gate step (design task 7).

Coverage rule (design section 5), reproduced here as executable, documented
categories -- see ``RULE_TEXT`` below for the exact prose baked into every
frozen ``selection.json``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # scripts/p_shared
SCRIPTS = HERE.parent                              # scripts/
REPO = SCRIPTS.parent                              # DRBX/
WORKSPACE = REPO.parent                            # the "HSX drbx" workspace

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Load topology the way the frozen packages do (design task 6's own hint):
# ``perpendicular_structured.reconstruction.load_context`` resolves to
# ``p07_combined_global.kernels.load``, which reads only
# ``base_geometry.npz`` and ``rlp_topology.npz`` under the geometry root --
# no saved campaign result is touched by this import or this call.
from perpendicular_structured.reconstruction import load_context  # noqa: E402

GEOMETRY_ROOT = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
P05_DIRECT_CONFIG = SCRIPTS / "p05_direct_midpoint_global/configuration.json"
P06_STRUCTURED_CONFIG = SCRIPTS / "p06_structured_global/configuration.json"
P05N_FIXTURES = SCRIPTS / "p05n_field_derived_global/preflight_fixtures"
P06N_FIXTURES = SCRIPTS / "p06n_field_derived_global/preflight_fixtures"

SCHEMA = "drbx.p08-step1-replay-selection.v1"
GRIDS = (48, 64)
DEFAULT_OUTPUT_ROOT = WORKSPACE / "work/p08_step1_replay_selection_20260928"

RULE_TEXT = """\
Deterministic replay-owner selection (P08 step 1, task 6; design section 5
"Coverage"). All of N32 is replayed completely and needs no selection. At
N48/N64, owners are preselected by this rule, hashed before any saved
campaign action/result array is read:

1. radial_axis_0_1: radial indices 0 and 1.
2. profile_transitions_pm1: for each radial index i where the owner-count
   profile[i] (number of distinct owners at radial layer i, matching
   ``drbx.geometry.fci_perpendicular_reconstruction.StructuredReconstruction
   .profile`` and ``scripts/p07_combined_global/topology.py``'s own
   ``profile_transitions``) differs from profile[i-1], the set {i-1, i, i+1}
   (clipped to the grid).
3. singleton_ring_and_before: the first radial index i with profile[i] == n
   (the first fully-resolved singleton ring, i.e. one owner per raw cell),
   and the index immediately before it.
4. aggregate_layer: one radial index representative of the near-axis
   "aggregate" owner region -- owners with more than one raw member, that are
   not axis-core (touch radius 0), not wall (radius n-1), and not adjacent to
   a profile transition. This is the same partition
   ``scripts/p05n_field_derived_global/campaign.py:regional_masks`` (mirrored
   verbatim in ``scripts/p06n_field_derived_global/core.py``) computes;
   reproduced here read-only from the raw owner map, not imported, to avoid
   pulling in those packages' heavier runtime dependencies. The chosen index
   is the smallest radial index carrying such an aggregate owner.
5. midplane_n_over_2: radial index n // 2.
6. wall_band: radial indices n-6, n-3, n-2, n-1.

At every index selected by 1-6, the owner is sampled at the frozen P05
"eight deterministic angular locations" convention: the (theta, eta) eighth
fractions in ``scripts/p05_direct_midpoint_global/configuration.json``'s
``preflight_angular_fractions_eighths``, converted to grid indices via
j = round(theta_numerator * n / 8) % n, k = round(eta_numerator * n / 8) % n
(the exact arithmetic of that package's ``select_preflight_owners``).

7. theta_eta_seams: at three radii (1, n // 2, n - 1), the owners at the
   theta seam (theta index 0 and n-1, eta fixed at n // 2) and the eta seam
   (eta index 0 and n-1, theta fixed at n // 2).
8. existing_preflight_owners: the union of "all_owners" from the frozen
   P05N and P06N preflight_fixtures/N{n}.selection.json files (selection
   files only; their paired *.replay.npz / *.accepted_p05_replay.npz result
   fixtures in the same directories are never opened).
9. p05_hotspots: the archived reference-error "hotspot" owner ids frozen in
   scripts/p06_structured_global/configuration.json's
   preflight_hotspot_owners[str(n)], cross-checked equal to the P06N
   preflight fixture's own "hotspots" strata when present.

Closure: given the union of every category's owners above, the full raw-
member closure (every raw cell belonging to a selected owner) and the full
incident-face closure under both face censuses:
  - the P07 census (scripts/p07_combined_global/topology.py: periodic
    theta/eta via np.roll, lo != hi only, no duplicate seam faces);
  - the P06 slot census (scripts/p06_structured_global/numerics.py's
    _face_incidence: 3*(n+1)*n**2 slots, theta/eta slot n kept as an
    explicit duplicate ("legacy alias slot") of slot 0, per
    _periodic_duplicate_face).
The legacy alias slots (theta or eta slot index n) among the incident P06
slot ids are additionally called out on their own.

This rule and its inputs are pinned by the source_sha256 map recorded
alongside this selection: this module's own source, the config/fixture JSON
files it reads, and the two geometry files (base_geometry.npz,
rlp_topology.npz) for this grid.
"""


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path):
    return json.loads(Path(path).read_text())


def _as_owner_array(owners: Iterable[int]) -> np.ndarray:
    owners = sorted(int(o) for o in owners)
    return np.asarray(owners, dtype=np.int64)


# ---------------------------------------------------------------------------
# Topology loading (geometry-only; no saved campaign array).
# ---------------------------------------------------------------------------
def load_topology(n: int, workspace_root: Path = WORKSPACE):
    """Load the frozen HSX owner map/geometry the way the accepted packages do."""
    return load_context(n, str(workspace_root))


def radial_profile(t) -> np.ndarray:
    """Owners per radial layer -- identical formula to
    ``drbx.geometry.fci_perpendicular_reconstruction.StructuredReconstruction
    .profile`` and to ``scripts/p07_combined_global/topology.py``'s
    ``census``: the number of distinct owners along theta at eta index 0,
    for each radial index."""
    n = t.n
    ro3 = t.ro.reshape((n, n, n))
    return np.array([len(np.unique(ro3[i, :, 0])) for i in range(n)])


def profile_transition_indices(profile: np.ndarray) -> list[int]:
    """The first radial index of every new profile regime -- the exact
    formula ``scripts/p07_combined_global/topology.py`` records as
    ``profile_transitions`` in its own census output."""
    return (np.flatnonzero(profile[1:] != profile[:-1]) + 1).tolist()


def regional_owner_masks(t) -> dict[str, np.ndarray]:
    """Owner-level axis-core/wall/transition/aggregate partition.

    Reproduces (does not import) the identical body of
    ``scripts/p05n_field_derived_global/campaign.py:regional_masks`` (mirrored
    in ``scripts/p06n_field_derived_global/core.py``), from the raw owner map
    alone -- no saved result array is involved.
    """
    n = t.n
    no = len(t.vol)
    count = np.bincount(t.ro, minlength=no)
    rawcount = count[t.ro].reshape((n, n, n))
    radius = np.broadcast_to(np.arange(n)[:, None, None], (n, n, n))

    def owner_has(mask3d: np.ndarray) -> np.ndarray:
        return np.bincount(t.ro, weights=mask3d.reshape(-1).astype(np.float64), minlength=no) > 0

    axis_core = owner_has(radius == 0)
    wall = owner_has(radius == n - 1)
    diff = rawcount[1:] != rawcount[:-1]
    trans3 = np.zeros((n, n, n), dtype=bool)
    trans3[1:] |= diff
    trans3[:-1] |= diff
    transition_owner = owner_has(trans3)
    transition = transition_owner & ~axis_core & ~wall
    aggregate = (count > 1) & ~axis_core & ~wall & ~transition
    return {"axis_core": axis_core, "wall": wall, "transition": transition, "aggregate": aggregate}


def aggregate_layer_index(t) -> int:
    """The smallest radial index carrying a raw member owned by an
    "aggregate" owner (see ``regional_owner_masks``)."""
    n = t.n
    masks = regional_owner_masks(t)
    raw_mask = masks["aggregate"][t.ro].reshape((n, n, n))
    radius = np.broadcast_to(np.arange(n)[:, None, None], (n, n, n))
    radii = radius[raw_mask]
    if not radii.size:
        raise ValueError("no aggregate-owner raw members found; the aggregate-layer coverage category would be empty")
    return int(radii.min())


def eighth_fraction_pairs() -> list[tuple[int, int]]:
    """The frozen P05 "eight deterministic angular locations" convention,
    read from ``scripts/p05_direct_midpoint_global/configuration.json``
    (see that package's README and ``select_preflight_owners``)."""
    config = _load_json(P05_DIRECT_CONFIG)
    return [tuple(pair) for pair in config["preflight_angular_fractions_eighths"]]


def owners_at_index_eighths(t, radial_index: int, fractions: list[tuple[int, int]]) -> set[int]:
    n = t.n
    owners: set[int] = set()
    for num_theta, num_eta in fractions:
        j = int(round(num_theta * n / 8)) % n
        k = int(round(num_eta * n / 8)) % n
        raw = int(np.ravel_multi_index((int(radial_index) % n, j, k), (n, n, n)))
        owners.add(int(t.ro[raw]))
    return owners


def radial_index_categories(t) -> dict[str, set[int]]:
    n = t.n
    profile = radial_profile(t)
    transitions = profile_transition_indices(profile)
    singleton = np.flatnonzero(profile == n)
    if not singleton.size:
        raise ValueError("no fully-resolved singleton radial ring found in the owner profile")
    first_singleton = int(singleton[0])

    transition_indices: set[int] = set()
    for ti in transitions:
        for cand in (ti - 1, ti, ti + 1):
            if 0 <= cand < n:
                transition_indices.add(cand)

    return {
        "radial_axis_0_1": {i for i in (0, 1) if i < n},
        "profile_transitions_pm1": transition_indices,
        "singleton_ring_and_before": {i for i in (first_singleton - 1, first_singleton) if 0 <= i < n},
        "aggregate_layer": {aggregate_layer_index(t)},
        "midplane_n_over_2": {n // 2},
        "wall_band": {i for i in (n - 6, n - 3, n - 2, n - 1) if 0 <= i < n},
    }


def seam_owners(t) -> tuple[set[int], list[int]]:
    n = t.n
    radii = sorted({1, n // 2, n - 1})
    mid = n // 2
    owners: set[int] = set()
    for r in radii:
        for j in (0, n - 1):
            owners.add(int(t.ro[int(np.ravel_multi_index((r, j, mid), (n, n, n)))]))
        for k in (0, n - 1):
            owners.add(int(t.ro[int(np.ravel_multi_index((r, mid, k), (n, n, n)))]))
    return owners, radii


def existing_preflight_owners(n: int) -> tuple[set[int], list[Path]]:
    """The union of "all_owners" from the frozen P05N/P06N preflight
    fixtures' *selection* files -- never their paired result fixtures."""
    owners: set[int] = set()
    sources: list[Path] = []
    for fixtures in (P05N_FIXTURES, P06N_FIXTURES):
        path = fixtures / f"N{n}.selection.json"
        data = _load_json(path)
        owners.update(int(e["owner"]) for e in data.get("all_owners", []))
        sources.append(path)
    return owners, sources


def p05_hotspot_owners(n: int) -> tuple[set[int], list[Path]]:
    config = _load_json(P06_STRUCTURED_CONFIG)
    owners = {int(o) for o in config["preflight_hotspot_owners"].get(str(n), [])}
    sources = [P06_STRUCTURED_CONFIG]
    fixture_path = P06N_FIXTURES / f"N{n}.selection.json"
    if fixture_path.exists():
        fixture = _load_json(fixture_path)
        fixture_hotspots = {int(e["owner"]) for e in fixture.get("hotspots", [])}
        sources.append(fixture_path)
        if fixture_hotspots and fixture_hotspots != owners:
            raise ValueError(f"p05/p06 hotspot owner sets disagree for N{n}: {owners} vs {fixture_hotspots}")
    return owners, sources


# ---------------------------------------------------------------------------
# Closure: raw members and incident faces under both censuses.
# ---------------------------------------------------------------------------
def p07_census_face_ids(t, owners: Iterable[int]) -> np.ndarray:
    """Incident face ids under the P07 census numbering
    (scripts/p07_combined_global/topology.py:23-34): periodic theta/eta via
    np.roll (lo != hi only, no duplicate seam faces)."""
    n = t.n
    owners_arr = _as_owner_array(owners)
    ro3 = t.ro.reshape((n, n, n))
    radial_lo = np.full((n + 1, n, n), -1, np.int64)
    radial_hi = np.full((n + 1, n, n), -1, np.int64)
    radial_lo[1:n] = ro3[:-1]
    radial_lo[n] = ro3[-1]
    radial_hi[1:n] = ro3[1:]
    radial_hi[0] = ro3[0]
    theta_lo = np.roll(ro3, 1, axis=1)
    theta_hi = ro3
    eta_lo = np.roll(ro3, 1, axis=2)
    eta_hi = ro3
    off = (n + 1) * n * n
    off2 = off + n ** 3
    ids = []
    for lo, hi, start in ((radial_lo, radial_hi, 0), (theta_lo, theta_hi, off), (eta_lo, eta_hi, off2)):
        valid = (lo != hi).ravel()
        incident = valid & (np.isin(lo.ravel(), owners_arr) | np.isin(hi.ravel(), owners_arr))
        ids.append(start + np.flatnonzero(incident))
    return np.sort(np.concatenate(ids)).astype(np.int64)


def p06_slot_census_face_ids(t, owners: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Incident face ids under the P06 slot census numbering
    (scripts/p06_structured_global/numerics.py's _face_incidence /
    _periodic_duplicate_face, and the same 3*(n+1)*n**2 id scheme as
    scripts/p07_diffusion_global/numerics.py's face_indices/face_keys):
    3*(n+1)*n**2 slots, theta/eta slot n kept as an explicit duplicate of
    slot 0. Returns (all incident ids, the subset that are legacy alias
    slots -- theta or eta slot index n)."""
    n = t.n
    owners_arr = _as_owner_array(owners)
    ro3 = t.ro.reshape((n, n, n))
    size0 = (n + 1) * n * n
    size1 = n * (n + 1) * n

    # Axis 0 (radial): slot m in 0..n over (j, k); exterior at m=0 (no lower)
    # and m=n (no upper).
    lo0 = np.full((n + 1, n, n), -1, np.int64)
    hi0 = np.full((n + 1, n, n), -1, np.int64)
    lo0[1:] = ro3
    hi0[:-1] = ro3
    ids0 = (np.arange(n + 1)[:, None, None] * n * n
            + np.arange(n)[None, :, None] * n
            + np.arange(n)[None, None, :])

    # Axis 1 (theta): slot m in 0..n over (i, k); periodic, m=n aliases m=0.
    m = np.arange(n + 1)
    lo1 = ro3[:, (m - 1) % n, :]
    hi1 = ro3[:, m % n, :]
    ids1 = size0 + (np.arange(n)[:, None, None] * (n + 1) * n
                     + m[None, :, None] * n
                     + np.arange(n)[None, None, :])
    alias1 = np.zeros((n, n + 1, n), dtype=bool)
    alias1[:, n, :] = True

    # Axis 2 (eta): slot m in 0..n over (i, j); periodic, m=n aliases m=0.
    lo2 = ro3[:, :, (m - 1) % n]
    hi2 = ro3[:, :, m % n]
    ids2 = size0 + size1 + (np.arange(n)[:, None, None] * n * (n + 1)
                              + np.arange(n)[None, :, None] * (n + 1)
                              + m[None, None, :])
    alias2 = np.zeros((n, n, n + 1), dtype=bool)
    alias2[:, :, n] = True

    incident_ids = []
    alias_ids = []
    for lo, hi, ids, alias in ((lo0, hi0, ids0, np.zeros((n + 1, n, n), dtype=bool)),
                               (lo1, hi1, ids1, alias1),
                               (lo2, hi2, ids2, alias2)):
        incident = np.isin(lo, owners_arr) | np.isin(hi, owners_arr)
        incident_ids.append(np.broadcast_to(ids, incident.shape)[incident])
        alias_incident = incident & alias
        alias_ids.append(np.broadcast_to(ids, alias_incident.shape)[alias_incident])
    all_ids = np.sort(np.concatenate(incident_ids)).astype(np.int64)
    alias_only = np.sort(np.unique(np.concatenate(alias_ids))).astype(np.int64)
    return all_ids, alias_only


# ---------------------------------------------------------------------------
# Top-level build / freeze / verify.
# ---------------------------------------------------------------------------
def build_selection(n: int, *, workspace_root: Path = WORKSPACE, timestamp: str) -> dict:
    """Build the full, deterministic selection payload for grid ``n``.

    ``timestamp`` must be supplied explicitly (an ISO-8601 string) so that
    this function is a pure, deterministic function of its inputs -- the
    "frozen with a timestamp" requirement lives in ``freeze``, which chooses
    the timestamp once and passes it through here.
    """
    if n not in GRIDS:
        raise ValueError(f"selection freeze is only defined for grids {GRIDS}, got {n}")

    t = load_topology(n, workspace_root)

    categories: dict[str, dict] = {}
    source_paths: set[Path] = {P05_DIRECT_CONFIG}

    idx_categories = radial_index_categories(t)
    fractions = eighth_fraction_pairs()
    for name, indices in idx_categories.items():
        owners: set[int] = set()
        for i in sorted(indices):
            owners |= owners_at_index_eighths(t, i, fractions)
        categories[name] = {"radial_indices": sorted(int(i) for i in indices), "owners": sorted(owners)}

    seam_owner_set, seam_radii = seam_owners(t)
    categories["theta_eta_seams"] = {"radii": seam_radii, "owners": sorted(seam_owner_set)}

    preflight_owners, preflight_sources = existing_preflight_owners(n)
    source_paths.update(preflight_sources)
    categories["existing_preflight_owners"] = {"owners": sorted(preflight_owners)}

    hotspot_owners, hotspot_sources = p05_hotspot_owners(n)
    source_paths.update(hotspot_sources)
    categories["p05_hotspots"] = {"owners": sorted(hotspot_owners)}

    all_owners: set[int] = set()
    for cat in categories.values():
        all_owners.update(cat["owners"])
    if not all_owners:
        raise ValueError(f"empty replay-owner selection for N{n}")

    owners_arr = _as_owner_array(all_owners)
    raw_ids = np.flatnonzero(np.isin(t.ro, owners_arr))
    p07_ids = p07_census_face_ids(t, all_owners)
    p06_ids, alias_ids = p06_slot_census_face_ids(t, all_owners)

    closure = {
        "owners": sorted(all_owners),
        "raw_member_ids": raw_ids.astype(np.int64).tolist(),
        "p07_census_face_ids": p07_ids.tolist(),
        "p06_slot_census_face_ids": p06_ids.tolist(),
        "legacy_alias_slot_ids": alias_ids.tolist(),
    }

    geometry_dir = GEOMETRY_ROOT / f"{n}x{n}x{n}"
    geometry_sources = [geometry_dir / "base_geometry.npz", geometry_dir / "rlp_topology.npz"]
    all_sources = sorted(set(source_paths) | set(geometry_sources) | {Path(__file__).resolve()})
    source_sha256 = {str(p.relative_to(WORKSPACE)): _sha256_file(p) for p in all_sources}

    counts = {
        "owners_total": len(all_owners),
        "raw_member_total": int(raw_ids.size),
        "p07_census_face_total": int(p07_ids.size),
        "p06_slot_census_face_total": int(p06_ids.size),
        "legacy_alias_slot_total": int(alias_ids.size),
    }
    counts.update({f"category:{name}": len(cat["owners"]) for name, cat in categories.items()})

    return {
        "schema": SCHEMA,
        "n": n,
        "generated_at": timestamp,
        "rule": RULE_TEXT,
        "categories": categories,
        "closure": closure,
        "counts": counts,
        "source_sha256": source_sha256,
    }


def _serialize(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def freeze(output_root: Path | None = None, *, workspace_root: Path = WORKSPACE,
           grids: tuple[int, ...] = GRIDS, timestamp: str | None = None) -> dict[int, dict]:
    """Compute and write the frozen selection for every grid in ``grids``.

    Writes, for each ``n``, ``<output_root>/N{n}/selection.json`` and its
    sha256 in a sibling ``SELECTION_SHA256`` file.
    """
    output_root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    written: dict[int, dict] = {}
    for n in grids:
        payload = build_selection(n, workspace_root=workspace_root, timestamp=ts)
        grid_dir = output_root / f"N{n}"
        grid_dir.mkdir(parents=True, exist_ok=True)
        text = _serialize(payload)
        sel_path = grid_dir / "selection.json"
        sel_path.write_text(text)
        digest = _sha256_bytes(text.encode())
        (grid_dir / "SELECTION_SHA256").write_text(digest + "\n")
        written[n] = {"selection_path": sel_path, "sha256_path": grid_dir / "SELECTION_SHA256",
                      "sha256": digest, "payload": payload}
    return written


def verify_frozen(grid_dir: Path, *, workspace_root: Path = WORKSPACE) -> tuple[bool, dict, dict]:
    """Recompute a frozen selection and check it against its own recorded hash.

    Returns ``(matches, recomputed_payload, recorded_payload)``.
    """
    grid_dir = Path(grid_dir)
    sel_path = grid_dir / "selection.json"
    sha_path = grid_dir / "SELECTION_SHA256"
    recorded_text = sel_path.read_text()
    recorded = json.loads(recorded_text)
    recorded_hash = sha_path.read_text().strip()
    if _sha256_bytes(recorded_text.encode()) != recorded_hash:
        raise ValueError("frozen selection.json does not match its own SELECTION_SHA256 file")
    recomputed = build_selection(int(recorded["n"]), workspace_root=workspace_root,
                                 timestamp=recorded["generated_at"])
    recomputed_hash = _sha256_bytes(_serialize(recomputed).encode())
    return recomputed_hash == recorded_hash, recomputed, recorded


if __name__ == "__main__":
    for grid, info in freeze().items():
        print(grid, info["sha256"], info["selection_path"])
